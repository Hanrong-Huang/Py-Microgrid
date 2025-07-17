"""
System optimization utilities for Py-Microgrid - Refined and Unified Version
This version uses a single, clear cost calculation pipeline driven by YAML configuration files,
resolves the negative replacement cost bug, and removes all redundant methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from scipy.optimize import minimize
import os  # Import the os module to handle file paths

# Core py_microgrid and component imports
from py_microgrid.utilities import ConfigManager
from py_microgrid.tools.analysis.bos import EconomicCalculator
from .load_analyzer import LoadAnalyzer
from py_microgrid.simulation.technologies.sites import SiteInfo
from py_microgrid.simulation.technologies.pv.pv_plant import PVConfig, PVPlant
from py_microgrid.simulation.technologies.wind.wind_plant import WindConfig, WindPlant

class SystemOptimizer:
    """
    Refined SystemOptimizer that uses a unified cost model driven by YAML configuration files.
    Supports both 4-component (no grid) and 5-component (with grid) optimization.
    """
    
    def __init__(self,
                 yaml_file_path: str,
                 economic_calculator: EconomicCalculator,
                 enable_flexible_load: bool = True,
                 max_load_reduction_percentage: float = 0.2):
        """Initialize the SystemOptimizer."""
        self.yaml_file_path = yaml_file_path
        self.economic_calculator = economic_calculator
        self.config_manager = ConfigManager()
        self.load_analyzer = LoadAnalyzer(
            enable_flexible_load=enable_flexible_load,
            max_load_reduction_percentage=max_load_reduction_percentage
        )
        self.grid_enabled = False
        
        # Load cost configurations from config files, which are the single source of truth.
        self._load_cost_configs()

    def _load_cost_configs(self):
        """Load all cost parameters from the project's config files."""
        try:
            # Navigate to the config directory from the project's main YAML file
            yaml_dir = os.path.dirname(os.path.abspath(self.yaml_file_path))
            py_microgrid_root = yaml_dir
            while not os.path.basename(py_microgrid_root) == 'py_microgrid':
                py_microgrid_root = os.path.dirname(py_microgrid_root)
                if py_microgrid_root == os.path.dirname(py_microgrid_root):
                    raise FileNotFoundError("Could not find py_microgrid root directory")
            
            config_dir = os.path.join(py_microgrid_root, "simulation", "config")
            
            self.pv_costs = self.config_manager.load_yaml_safely(os.path.join(config_dir, "pv_config.yaml"))
            self.wind_costs = self.config_manager.load_yaml_safely(os.path.join(config_dir, "wind_config.yaml"))
            self.battery_costs = self.config_manager.load_yaml_safely(os.path.join(config_dir, "battery_config.yaml"))
            self.genset_costs = self.config_manager.load_yaml_safely(os.path.join(config_dir, "genset_config.yaml"))
            self.grid_costs = self.config_manager.load_yaml_safely(os.path.join(config_dir, "grid_config.yaml"))
            
            print(f"✓ Successfully loaded cost configs from: {config_dir}")
            print(f"  PV: ${self.pv_costs['pv']['costs']['installed_cost_per_kw']}/kW + ${self.pv_costs['pv']['costs']['om_cost_per_kw_per_year']}/kW/year O&M")
            print(f"  Wind: ${self.wind_costs['wind']['costs']['installed_cost_per_kw']}/kW + ${self.wind_costs['wind']['costs']['om_cost_per_kw_per_year']}/kW/year O&M")
            print(f"  Battery: ${self.battery_costs['battery']['costs']['installed_cost_per_kwh']}/kWh + ${self.battery_costs['battery']['costs']['om_cost_per_kwh_per_year']}/kWh/year O&M")
            print(f"  Genset: ${self.genset_costs['genset']['costs']['install_cost_per_kw']}/kW + ${self.genset_costs['genset']['costs']['fuel_cost_per_liter']}/L fuel")

        except Exception as e:
            print(f"CRITICAL ERROR: Could not load cost configs. Please check paths and file integrity: {e}")
            self._set_default_costs()

    def _set_default_costs(self):
        """Sets fallback default costs if YAML files cannot be loaded."""
        print("Warning: Setting default cost values. Results may not be accurate.")
        self.pv_costs = {'pv': {'costs': {'installed_cost_per_kw': 2000.0, 'om_cost_per_kw_per_year': 10.0}}}
        self.wind_costs = {'wind': {'turbine_rating_kw': 1000.0, 'costs': {'installed_cost_per_kw': 2500.0, 'om_cost_per_kw_per_year': 40.0}}}
        self.battery_costs = {'battery': {'costs': {'installed_cost_per_kw': 0.0, 'installed_cost_per_kwh': 700.0, 'replacement_cost_per_kwh': 700.0, 'om_cost_per_kwh_per_year': 10.0}, 'operation': {'replacement_year': 15}}}
        self.genset_costs = {'genset': {'costs': {'install_cost_per_kw': 500.0, 'replacement_cost_per_kw': 500.0, 'om_cost_per_kw_per_op_hour': 0.03, 'fuel_cost_per_liter': 1.20}, 'performance': {'specific_fuel_consumption_l_per_kwh': 0.25, 'operational_life_hours': 15000}, 'environment': {'specific_co2_per_l_fuel': 2.618, 'carbon_cost_per_tonne': 50.0}}}
        self.grid_costs = {'grid': {'costs': {'base_import_price': 0.12, 'base_export_price': 0.08, 'connection_cost_per_kw': 100.0}}}

    def optimize_system(self, bounds: List[Tuple[float, float]],
                       initial_conditions: List[List[float]]) -> Optional[Dict[str, Any]]:
        """Optimize system configuration using a unified cost model."""
        config = self.config_manager.load_yaml_safely(self.yaml_file_path)
        self.grid_enabled = config.get('technologies', {}).get('grid', {}).get('enabled', False)
        
        expected_components = 6 if self.grid_enabled else 5
        if len(bounds) != expected_components:
            raise ValueError(f"Expected {expected_components} bounds for {'5-component (grid)' if self.grid_enabled else '4-component'}-system, got {len(bounds)}")

        best_result, best_lcoe = None, float('inf')
        
        for x0 in initial_conditions:
            try:
                result = minimize(
                    self.penalized_objective, x0, method='Nelder-Mead',
                    bounds=bounds, options={'maxiter': 300, 'xatol': 1, 'fatol': 1e-3}
                )
                if result.success:
                    optimal_config = [int(round(v)) for v in result.x]
                    lcoe, optimal_results = self.objective_function(optimal_config)
                    if lcoe < best_lcoe:
                        best_lcoe, best_result = lcoe, optimal_results
            except Exception as e:
                print(f"Optimization failed for initial point {x0}: {e}")
        return best_result

    def penalized_objective(self, x: List[float]) -> float:
        """Calculate a penalized objective function value to ensure demand satisfaction."""
        x_rounded = [int(round(val)) for val in x]
        lcoe, results = self.objective_function(x_rounded)
        demand_met_percentage = results.get('Demand Met Percentage', 0)
        if demand_met_percentage < 100.0:
            penalty_factor = (100 / max(demand_met_percentage, 0.01)) ** 2
            return lcoe * penalty_factor
        return lcoe

    def objective_function(self, x: List[float]) -> Tuple[float, Dict[str, Any]]:
        """Calculate the LCOE and system performance for a given configuration 'x'."""
        if self.grid_enabled:
            pv_kw, num_turbines, battery_kwh, battery_kw, genset_kw, grid_kw = map(lambda val: max(0, val), x)
        else:
            pv_kw, num_turbines, battery_kwh, battery_kw, genset_kw = map(lambda val: max(0, val), x)
            grid_kw = 0
        
        turbine_rating = self.wind_costs['wind'].get('turbine_rating_kw', 1000.0)
        wind_kw = num_turbines * turbine_rating

        try:
            config = self.config_manager.load_yaml_safely(self.yaml_file_path)
            
            project_dir = os.path.dirname(os.path.abspath(self.yaml_file_path))
            relative_load_path = os.path.normpath(config['site']['desired_schedule'])
            absolute_load_path = os.path.join(project_dir, relative_load_path)
            demand_df = pd.read_csv(absolute_load_path, header=0, dtype=np.float64)
            demand_profile_mw = demand_df.values.flatten()
            
            if len(demand_profile_mw) < 8760:
                raise ValueError(f"Load data file '{absolute_load_path}' only contains {len(demand_profile_mw)} points, but 8760 are required.")
            
            site = SiteInfo(data=config['site']['data'], 
                            solar_resource_file=config['site']['solar_resource_file'],
                            wind_resource_file=config['site']['wind_resource_file'],
                            desired_schedule=demand_profile_mw.tolist())
            
            pv_plant = PVPlant(site, PVConfig(system_capacity_kw=pv_kw))
            wind_plant = WindPlant(site, WindConfig(num_turbines=num_turbines, turbine_rating_kw=turbine_rating))
            
            pv_plant.simulate_power(self.economic_calculator.project_lifetime)
            wind_plant.simulate_power(self.economic_calculator.project_lifetime)

            pv_gen = pv_plant.generation_profile[:8760]
            wind_gen = wind_plant.generation_profile[:8760]

            renewable_gen = np.array(pv_gen) + np.array(wind_gen)
            demand = demand_profile_mw[:8760] * 1000

            battery_discharge, genset_generation, grid_generation = self._dispatch_simulation(
                renewable_gen, demand, battery_kwh, battery_kw, genset_kw, grid_kw)
        except Exception as e:
            print(f"Simulation failed for configuration {x}: {e}")
            return 1e9, {}

        costs = self._calculate_component_costs(
            pv_kw, wind_kw, battery_kwh, battery_kw, genset_kw, grid_kw, genset_generation)
        total_system_cost = sum(c['total'] for c in costs.values())

        df = pd.DataFrame({'PV Generation (kW)': pv_gen, 'Wind Generation (kW)': wind_gen,
                           'Genset Generation (kW)': genset_generation,
                           'Original Battery Generation (kW)': battery_discharge,
                           'Original Load (kW)': demand})
        metrics = self.load_analyzer.calculate_performance_metrics(df, self.economic_calculator.project_lifetime)
        lcoe = self.economic_calculator.calculate_lcoe(total_system_cost, metrics['Total Load Served (kWh)'])

        result_data = {
            'x': x, 'costs': costs, 'metrics': metrics, 'lcoe': lcoe, 'total_system_cost': total_system_cost,
            'pv_total_gen': np.sum(pv_gen), 'wind_total_gen': np.sum(wind_gen),
            'battery_total_gen': np.sum(battery_discharge), 'genset_total_gen': np.sum(genset_generation),
            'grid_total_gen': np.sum(grid_generation)
        }
        return lcoe, self._compile_results(result_data)

    def _dispatch_simulation(self, renewable_gen, demand, battery_kwh, battery_kw, genset_kw, grid_kw):
        """Simulates the hourly dispatch of battery, genset, and grid."""
        battery_discharge, genset_generation, grid_generation = (np.zeros(8760) for _ in range(3))
        current_soc = battery_kwh * 0.5

        for hour in range(8760):
            energy_balance = renewable_gen[hour] - demand[hour]
            if energy_balance > 0:
                charge_amount = min(energy_balance, battery_kw, battery_kwh - current_soc)
                current_soc += charge_amount
            else:
                deficit = -energy_balance
                discharge_amount = min(deficit, battery_kw, current_soc)
                battery_discharge[hour] = discharge_amount
                current_soc -= discharge_amount
                remaining_deficit = deficit - discharge_amount
                
                if remaining_deficit > 0 and self.grid_enabled:
                    grid_supply = min(remaining_deficit, grid_kw)
                    grid_generation[hour] = grid_supply
                    remaining_deficit -= grid_supply
                
                if remaining_deficit > 0:
                    genset_supply = min(remaining_deficit, genset_kw)
                    genset_generation[hour] = genset_supply
        return battery_discharge, genset_generation, grid_generation

    def _calculate_component_costs(self, pv_kw, wind_kw, battery_kwh, battery_kw, genset_kw, grid_kw, genset_gen_profile):
        """Master function to calculate lifetime costs for all components based on YAML configs."""
        print(f"\nCalculating costs for: PV:{pv_kw}kW, Wind:{wind_kw}kW, Bat:{battery_kwh}kWh, Gen:{genset_kw}kW")
        costs = {
            'pv': self._calculate_pv_costs(pv_kw),
            'wind': self._calculate_wind_costs(wind_kw),
            'battery': self._calculate_battery_costs(battery_kwh, battery_kw),
            'genset': self._calculate_genset_costs(genset_kw, genset_gen_profile),
            'grid': self._calculate_grid_costs(grid_kw) if self.grid_enabled else {'total': 0, 'co2_emissions': 0}
        }
        print(f"  PV Cost: ${costs['pv']['total']:,.0f}")
        print(f"  Wind Cost: ${costs['wind']['total']:,.0f}")
        print(f"  Battery Cost: ${costs['battery']['total']:,.0f}")
        print(f"  Genset Cost: ${costs['genset']['total']:,.0f} (Fuel: ${costs['genset']['fuel_cost']:,.0f}, Replace: ${costs['genset']['replace_cost']:,.0f}, Carbon: ${costs['genset'].get('carbon_cost', 0):,.0f})")
        if self.grid_enabled: print(f"  Grid Cost: ${costs['grid']['total']:,.0f}")
        return costs

    def _calculate_pv_costs(self, pv_kw: float) -> Dict[str, float]:
        costs = self.pv_costs['pv']['costs']
        install_cost = pv_kw * costs['installed_cost_per_kw']
        om_cost = pv_kw * costs['om_cost_per_kw_per_year'] * self.economic_calculator.project_lifetime
        return {'total': install_cost + om_cost}

    def _calculate_wind_costs(self, wind_kw: float) -> Dict[str, float]:
        costs = self.wind_costs['wind']['costs']
        install_cost = wind_kw * costs['installed_cost_per_kw']
        om_cost = wind_kw * costs['om_cost_per_kw_per_year'] * self.economic_calculator.project_lifetime
        return {'total': install_cost + om_cost}

    def _calculate_battery_costs(self, battery_kwh: float, battery_kw: float) -> Dict[str, float]:
        costs = self.battery_costs['battery']['costs']
        op = self.battery_costs['battery']['operation']
        install_cost = (battery_kwh * costs['installed_cost_per_kwh'] + battery_kw * costs.get('installed_cost_per_kw', 0))
        num_replacements = max(0, np.floor((self.economic_calculator.project_lifetime / op['replacement_year']) - 1))
        replacement_cost = num_replacements * battery_kwh * costs['replacement_cost_per_kwh']
        om_cost = battery_kwh * costs['om_cost_per_kwh_per_year'] * self.economic_calculator.project_lifetime
        return {'total': install_cost + replacement_cost + om_cost}

    def _calculate_genset_costs(self, genset_kw: float, genset_gen_profile: np.ndarray) -> Dict[str, float]:
        """Calculate Genset costs including a new carbon cost penalty."""
        c = self.genset_costs['genset']['costs']
        p = self.genset_costs['genset']['performance']
        e = self.genset_costs['genset']['environment']
        
        total_op_hours = np.sum(np.array(genset_gen_profile) > 0)
        op_hours_yr = total_op_hours / self.economic_calculator.project_lifetime if self.economic_calculator.project_lifetime > 1 else total_op_hours
        
        life_yrs = p['operational_life_hours'] / op_hours_yr if op_hours_yr > 0 else float('inf')
        num_replacements = max(0, np.floor((self.economic_calculator.project_lifetime / life_yrs) - 1))

        install_cost = genset_kw * c['install_cost_per_kw']
        replace_cost = num_replacements * genset_kw * c['replacement_cost_per_kw']
        om_cost = c['om_cost_per_kw_per_op_hour'] * genset_kw * total_op_hours
        
        total_gen_kwh = np.sum(genset_gen_profile)
        fuel_consumption = total_gen_kwh * p['specific_fuel_consumption_l_per_kwh']
        fuel_cost = fuel_consumption * c['fuel_cost_per_liter']
        co2_emissions = fuel_consumption * e['specific_co2_per_l_fuel']
        
        carbon_cost_per_tonne = e.get('carbon_cost_per_tonne', 0.0)
        carbon_cost = (co2_emissions / 1000) * carbon_cost_per_tonne
        
        return {
            'total': install_cost + replace_cost + om_cost + fuel_cost + carbon_cost,
            'replace_cost': replace_cost,
            'fuel_cost': fuel_cost,
            'carbon_cost': carbon_cost,
            'co2_emissions': co2_emissions
        }

    def _calculate_grid_costs(self, grid_kw: float) -> Dict[str, float]:
        costs = self.grid_costs['grid']['costs']
        connection_cost = grid_kw * costs.get('connection_cost_per_kw', 100)
        return {'total': connection_cost, 'co2_emissions': 0}

    def _compile_results(self, data: dict) -> Dict[str, Any]:
        """Helper function to neatly package all results into a dictionary."""
        if self.grid_enabled:
            pv_kw, num_turbines, battery_kwh, battery_kw, genset_kw, grid_kw = data['x']
        else:
            pv_kw, num_turbines, battery_kwh, battery_kw, genset_kw = data['x']
            grid_kw = 0
            
        turbine_rating = self.wind_costs['wind'].get('turbine_rating_kw', 1000.0)
        wind_kw = num_turbines * turbine_rating
        total_gen = data['pv_total_gen'] + data['wind_total_gen'] + data['genset_total_gen'] + data['grid_total_gen']

        result = {
            "PV Capacity (kW)": pv_kw, "Wind Turbine Count": int(num_turbines),
            # === THIS IS THE FIX FOR THE KEYERROR ===
            "Wind Turbine Capacity (kW)": wind_kw,
            "Battery Energy Capacity (kWh)": battery_kwh,
            "Battery Power Capacity (kW)": battery_kw, "Genset Capacity (kW)": genset_kw,
            "Total System Generation (kWh)": total_gen, "Total PV Generation (kWh)": data['pv_total_gen'],
            "Total Wind Generation (kWh)": data['wind_total_gen'], "Total Genset Generation (kWh)": data['genset_total_gen'],
            "Total Battery Generation (kWh)": data['battery_total_gen'],
            "Renewable Fraction (%)": (data['pv_total_gen'] + data['wind_total_gen']) / max(total_gen, 1) * 100,
            "System NPC ($)": self.economic_calculator.calculate_present_value(data['total_system_cost']),
            "System LCOE ($/kWh)": data['lcoe'], "PV Cost ($)": data['costs']['pv']['total'],
            "Wind Cost ($)": data['costs']['wind']['total'], "Battery Cost ($)": data['costs']['battery']['total'],
            "Genset Cost ($)": data['costs']['genset']['total'], "Total System Cost ($)": data['total_system_cost'],
            "Total CO2 emissions (tonne)": data['costs']['genset']['co2_emissions'] / 1000,
            **data['metrics']
        }
        if self.grid_enabled:
            result.update({"Grid Capacity (kW)": grid_kw, "Total Grid Generation (kWh)": data['grid_total_gen'],
                           "Grid Cost ($)": data['costs']['grid']['total']})
        return result