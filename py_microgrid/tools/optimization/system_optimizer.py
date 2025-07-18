"""
System optimization utilities for py_microgrid - Refined and Unified Version
This version uses a single, clear cost calculation pipeline driven by YAML configuration files,
resolves the negative replacement cost bug, and removes all redundant methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from scipy.optimize import minimize
import os

# Core py_microgrid and component imports
from py_microgrid.utilities import ConfigManager
from .economic_calculator import EconomicCalculator
from .load_analyzer import LoadAnalyzer
from py_microgrid.simulation.technologies.sites import SiteInfo
from py_microgrid.simulation.technologies.pv.pv_plant import PVConfig, PVPlant
from py_microgrid.simulation.technologies.wind.wind_plant import WindConfig, WindPlant
from py_microgrid.simulation.technologies.genset import Genset, GensetConfig

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
        self._load_cost_configs()

    def _load_cost_configs(self):
        """Load all cost parameters from the project's config files."""
        try:
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

        except Exception as e:
            print(f"CRITICAL ERROR: Could not load cost configs. Please check paths and file integrity: {e}")
            self._set_default_costs()

    def _set_default_costs(self):
        """Sets fallback default costs if YAML files cannot be loaded."""
        print("Warning: Setting default cost values. Results may not be accurate.")
        self.pv_costs = {'pv': {'costs': {'installed_cost_per_kw': 1350.0, 'om_cost_per_kw_per_year': 15.0}}}
        self.wind_costs = {'wind': {'turbine_rating_kw': 2500.0, 'costs': {'installed_cost_per_kw': 1750.0, 'om_cost_per_kw_per_year': 40.0}}}
        self.battery_costs = {'battery': {'costs': {'installed_cost_per_kw': 250.0, 'installed_cost_per_kwh': 350.0, 'replacement_cost_per_kwh': 250.0, 'om_cost_per_kwh_per_year': 12.0}, 'operation': {'replacement_year': 15}}}
        self.genset_costs = {'genset': {'costs': {'install_cost_per_kw': 650.0, 'replacement_cost_per_kw': 650.0, 'om_cost_per_kw_per_op_hour': 0.03, 'fuel_cost_per_liter': 0.95}, 'performance': {'specific_fuel_consumption_l_per_kwh': 0.26, 'operational_life_hours': 20000}, 'environment': {'specific_co2_per_l_fuel': 2.68, 'carbon_cost_per_tonne': 51.0}}}
        self.grid_costs = {'grid': {'costs': {'connection_cost_per_kw': 100.0}}}

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
        """Realistic industrial microgrid penalty based on value of lost load (VoLL)."""
        x_rounded = [int(round(val)) for val in x]
        lcoe, results = self.objective_function(x_rounded)
        demand_met_percentage = results.get('Demand Met Percentage', 0)

        # Industrial reliability targets: 95% minimum, 99% excellent
        if demand_met_percentage >= 99.0:
            # Excellent reliability - no penalty
            penalty_multiplier = 1.0
        elif demand_met_percentage >= 97.0:
            # Good reliability (97-99%) - small penalty
            unmet_percentage = 100.0 - demand_met_percentage
            penalty_multiplier = 1.0 + 0.3 * unmet_percentage  # Gentle penalty
        elif demand_met_percentage >= 95.0:
            # Acceptable reliability (95-97%) - moderate penalty
            unmet_percentage = 100.0 - demand_met_percentage
            penalty_multiplier = 1.0 + 0.6 + 0.8 * (unmet_percentage - 3.0)  # Moderate penalty
        elif demand_met_percentage >= 90.0:
            # Poor reliability (90-95%) - significant penalty
            unmet_percentage = 100.0 - demand_met_percentage
            penalty_multiplier = 1.0 + 2.2 + 1.5 * (unmet_percentage - 5.0)  # Significant penalty
        else:
            # Unacceptable reliability (<90%) - very high penalty
            unmet_percentage = 100.0 - demand_met_percentage
            penalty_multiplier = 1.0 + 9.7 + 3.0 * (unmet_percentage - 10.0)  # Very high penalty

        # Cap penalty at industrial maximum (15x base cost for totally unreliable systems)
        penalty_multiplier = min(penalty_multiplier, 15.0)

        return lcoe * penalty_multiplier

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
            
            # 1. Load demand data robustly first to get a clean data array
            project_dir = os.path.dirname(os.path.abspath(self.yaml_file_path))
            relative_load_path = os.path.normpath(config['site']['desired_schedule'])
            absolute_load_path = os.path.join(project_dir, relative_load_path)
            demand_df = pd.read_csv(absolute_load_path, header=0, dtype=np.float64)
            demand_profile_mw = demand_df.values.flatten()
            
            if len(demand_profile_mw) < 8760:
                raise ValueError(f"Load data file '{absolute_load_path}' only contains {len(demand_profile_mw)} points, but 8760 are required.")
            
            # 2. Prepare the complete site configuration dictionary
            site_config = config['site']
            # 3. Inject our pre-loaded, clean demand data into this dictionary, overriding the file path
            site_config['desired_schedule'] = demand_profile_mw.tolist()
            
            # 4. Initialize SiteInfo by unpacking the entire site_config dictionary.
            # This ensures it receives all necessary data, including 'site_boundaries'.
            site = SiteInfo(**site_config)
            
            # Now the rest of the simulation can proceed with a complete SiteInfo object
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
            
            # Create proper Genset object for dispatch logic and cost calculation
            genset_config = self._create_genset_config(genset_kw)
            genset = Genset(site=site, config=genset_config)
            
            # Recalculate genset generation using proper dispatch logic
            genset_generation = self._calculate_proper_genset_dispatch(
                renewable_gen, demand, battery_discharge, genset)
            
            # Simulate genset with the proper dispatch profile
            genset.simulate_power(genset_generation, self.economic_calculator.project_lifetime)
        except Exception as e:
            print(f"Simulation failed for configuration {x}: {e}")
            return 1e9, {}

        try:
            costs_dict, total_system_cost = self._calculate_component_costs(
                pv_kw, wind_kw, battery_kwh, battery_kw, genset_kw, grid_kw, genset)
        except Exception as e:
            print(f"Cost calculation failed: {e}")
            return 1e9, {}
        
        df = pd.DataFrame({'PV Generation (kW)': pv_gen, 'Wind Generation (kW)': wind_gen,
                           'Genset Generation (kW)': genset_generation,
                           'Original Battery Generation (kW)': battery_discharge,
                           'Original Load (kW)': demand})
        metrics = self.load_analyzer.calculate_performance_metrics(df, self.economic_calculator.project_lifetime)
        
        # Use the correct total load served from the metrics for the LCOE calculation
        total_load_served_kwh = metrics['Total Load Served (kWh)']
        lcoe = self.economic_calculator.calculate_lcoe(total_system_cost, total_load_served_kwh)

        result_data = {
            'x': x, 'costs': costs_dict, 'metrics': metrics, 'lcoe': lcoe, 'total_system_cost': total_system_cost,
            'pv_total_gen': np.sum(pv_gen), 'wind_total_gen': np.sum(wind_gen),
            'battery_total_gen': np.sum(battery_discharge), 'genset_total_gen': np.sum(genset.generation_profile[:8760]),
            'grid_total_gen': np.sum(grid_generation)
        }
        return lcoe, self._compile_results(result_data)

    def _dispatch_simulation(self, renewable_gen, demand, battery_kwh, battery_kw, genset_kw, grid_kw):
        """Simulates the hourly dispatch of battery, genset, and grid with improved battery management."""
        battery_discharge, genset_generation, grid_generation = (np.zeros(8760) for _ in range(3))
        current_soc = battery_kwh * 0.5
        
        # Battery management parameters (more conservative for reliability)
        min_soc_reserve = battery_kwh * 0.10  # 10% min SOC
        genset_threshold_soc = battery_kwh * 0.25  # Turn on genset when battery drops to 25%
        
        # Genset minimum turn-on power (reduced to 20% for better reliability)
        min_genset_turn_on = genset_kw * 0.20

        for hour in range(8760):
            energy_balance = renewable_gen[hour] - demand[hour]
            if energy_balance > 0:
                charge_amount = min(energy_balance, battery_kw, battery_kwh - current_soc)
                current_soc += charge_amount
            else:
                deficit = -energy_balance
                remaining_deficit = deficit
                
                # Priority 1: Always try battery first (as long as SOC > minimum reserve)
                if current_soc > min_soc_reserve:
                    available_battery_capacity = current_soc - min_soc_reserve
                    discharge_amount = min(remaining_deficit, battery_kw, available_battery_capacity)
                    if discharge_amount > 0:
                        battery_discharge[hour] = discharge_amount
                        current_soc -= discharge_amount
                        remaining_deficit -= discharge_amount
                
                # Priority 2: Economic dispatch between genset and grid based on YAML config costs
                if remaining_deficit > 0:
                    # Calculate real-time costs for economic dispatch
                    hour_of_day = hour % 24
                    
                    # Grid import cost from YAML config with time-of-use pricing
                    if self.grid_enabled:
                        grid_pricing = self.grid_costs['grid']['pricing']
                        dispatch_factors = self.grid_costs['grid']['dispatch_factors']
                        tou = self.grid_costs['grid']['time_of_use']
                        
                        base_price = grid_pricing['base_import_price']
                        
                        if tou['off_peak_start'] <= hour_of_day < tou['off_peak_end']:  # Off-peak
                            grid_cost_per_kwh = base_price * dispatch_factors['off_peak_factor']
                        elif tou['peak_start'] <= hour_of_day < tou['peak_end']:  # Peak
                            grid_cost_per_kwh = base_price * dispatch_factors['peak_factor']
                        else:  # Standard
                            grid_cost_per_kwh = base_price * dispatch_factors['standard_factor']
                    else:
                        grid_cost_per_kwh = float('inf')  # Grid not available
                    
                    # Genset marginal cost from YAML config (fuel + O&M)
                    genset_costs = self.genset_costs['genset']['costs']
                    genset_perf = self.genset_costs['genset']['performance']
                    fuel_cost = genset_perf['specific_fuel_consumption_l_per_kwh'] * genset_costs['fuel_cost_per_liter']
                    om_cost = genset_costs['om_cost_per_kw_per_op_hour'] / genset_kw if genset_kw > 0 else 0
                    genset_cost_per_kwh = fuel_cost + om_cost
                    
                    # Economic dispatch decision
                    if self.grid_enabled and grid_cost_per_kwh < genset_cost_per_kwh:
                        # Grid is cheaper - use grid first
                        grid_supply = min(remaining_deficit, grid_kw)
                        grid_generation[hour] = grid_supply
                        remaining_deficit -= grid_supply
                        
                        # Use genset for any remaining deficit
                        if remaining_deficit > 0 and remaining_deficit >= min_genset_turn_on:
                            genset_supply = min(remaining_deficit, genset_kw)
                            genset_generation[hour] = genset_supply
                            remaining_deficit -= genset_supply
                    else:
                        # Genset is cheaper or grid not available - use genset first
                        if remaining_deficit >= min_genset_turn_on:
                            genset_supply = min(remaining_deficit, genset_kw)
                            genset_generation[hour] = genset_supply
                            remaining_deficit -= genset_supply
                        elif current_soc <= genset_threshold_soc:
                            # If battery is critically low, run genset at minimum load for reliability
                            genset_supply = min_genset_turn_on
                            genset_generation[hour] = genset_supply
                            remaining_deficit = max(0, remaining_deficit - genset_supply)
                        
                        # Use grid for any remaining deficit
                        if remaining_deficit > 0 and self.grid_enabled:
                            grid_supply = min(remaining_deficit, grid_kw)
                            grid_generation[hour] = grid_supply
                            remaining_deficit -= grid_supply
                
                # If still unmet demand, use genset as emergency backup (full capacity available)
                if remaining_deficit > 0:
                    # Emergency genset dispatch (use full genset capacity for reliability)
                    available_genset_capacity = genset_kw - genset_generation[hour]
                    emergency_genset = min(remaining_deficit, available_genset_capacity)
                    genset_generation[hour] += emergency_genset
                    remaining_deficit -= emergency_genset
                
        return battery_discharge, genset_generation, grid_generation
    
    def _calculate_proper_genset_dispatch(self, renewable_gen, demand, battery_discharge, genset):
        """
        Calculate proper genset dispatch considering minimum turn-on power and capacity constraints.
        
        Args:
            renewable_gen: Renewable generation profile [kW]
            demand: Demand profile [kW]
            battery_discharge: Battery discharge profile [kW]
            genset: Genset object with dispatch_power method
            
        Returns:
            np.ndarray: Proper genset generation profile [kW]
        """
        genset_generation = np.zeros(8760)
        
        for hour in range(8760):
            # Calculate net deficit after renewables and battery
            net_deficit = demand[hour] - renewable_gen[hour] - battery_discharge[hour]
            
            if net_deficit > 0:
                # Use genset's dispatch method to properly handle minimum turn-on power
                actual_output, power_served = genset.dispatch_power(net_deficit)
                genset_generation[hour] = actual_output
                
        return genset_generation

    def _calculate_component_costs(self, pv_kw, wind_kw, battery_kwh, battery_kw, genset_kw, grid_kw, genset_obj):
        """Master function to calculate lifetime costs for all components based on YAML configs."""
        if self.grid_enabled:
            print(f"\nCalculating costs for: PV:{pv_kw}kW, Wind:{wind_kw}kW, Bat:{battery_kwh}kWh, Gen:{genset_kw}kW, Grid:{grid_kw}kW")
        else:
            print(f"\nCalculating costs for: PV:{pv_kw}kW, Wind:{wind_kw}kW, Bat:{battery_kwh}kWh, Gen:{genset_kw}kW")
        costs = {
            'pv': self._calculate_pv_costs(pv_kw),
            'wind': self._calculate_wind_costs(wind_kw),
            'battery': self._calculate_battery_costs(battery_kwh, battery_kw),
            'genset': self._calculate_genset_costs_from_object(genset_obj),
            'grid': self._calculate_grid_costs(grid_kw) if self.grid_enabled else {'total': 0, 'co2_emissions': 0}
        }
        total_cost = sum(c['total'] for c in costs.values())
        print(f"  PV Cost: ${costs['pv']['total']:,.0f}")
        print(f"  Wind Cost: ${costs['wind']['total']:,.0f}")
        print(f"  Battery Cost: ${costs['battery']['total']:,.0f}")
        print(f"  Genset Cost: ${costs['genset']['total']:,.0f} (Fuel: ${costs['genset']['fuel_cost']:,.0f}, Replace: ${costs['genset']['replace_cost']:,.0f}, Carbon: ${costs['genset'].get('carbon_cost', 0):,.0f})")
        if self.grid_enabled: print(f"  Grid Cost: ${costs['grid']['total']:,.0f}")
        return costs, total_cost

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

    def _create_genset_config(self, genset_kw: float) -> GensetConfig:
        """Create GensetConfig from YAML configuration."""
        c = self.genset_costs['genset']['costs']
        p = self.genset_costs['genset']['performance']
        e = self.genset_costs['genset']['environment']
        
        return GensetConfig(
            interconnect_kw=genset_kw,
            efficiency=p.get('efficiency', 0.35),
            minimum_load_factor=p.get('minimum_load_factor', 0.30),
            specific_fuel_consumption=p.get('specific_fuel_consumption_l_per_kwh', 0.25),
            fuel_cost_per_liter=c.get('fuel_cost_per_liter', 1.20),
            start_cost=c.get('start_cost', 50.0),
            maintenance_cost_per_hour=c.get('om_cost_per_kw_per_op_hour', 0.03) * genset_kw,
            co2_emissions_per_liter=e.get('specific_co2_per_l_fuel', 2.618),
            operational_life_hours=p.get('operational_life_hours', 15000.0),
            install_cost_per_kw=c.get('install_cost_per_kw', 650.0),
            replacement_cost_per_kw=c.get('replacement_cost_per_kw', 650.0)
        )
    
    def _calculate_genset_costs_from_object(self, genset_obj: Genset) -> Dict[str, float]:
        """Calculate Genset costs using the proper genset object."""
        cost_breakdown = genset_obj.calculate_total_costs(self.economic_calculator.project_lifetime)
        
        # Add carbon cost from YAML config
        e = self.genset_costs['genset']['environment']
        carbon_cost_per_tonne = e.get('carbon_cost_per_tonne', 0.0)
        carbon_cost = cost_breakdown['co2_emissions_tonnes'] * carbon_cost_per_tonne
        
        return {
            'total': cost_breakdown['total_cost'] + carbon_cost,
            'replace_cost': cost_breakdown['replacement_cost'],
            'fuel_cost': cost_breakdown['fuel_cost'],
            'carbon_cost': carbon_cost,
            'co2_emissions': cost_breakdown['co2_emissions_kg']
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
            "PV Capacity (kW)": pv_kw,
            "Wind Turbine Count": int(num_turbines),
            "Wind Turbine Capacity (kW)": wind_kw,
            "Battery Energy Capacity (kWh)": battery_kwh,
            "Battery Power Capacity (kW)": battery_kw,
            "Genset Capacity (kW)": genset_kw,
            "Total System Generation (kWh)": total_gen,
            "Total PV Generation (kWh)": data['pv_total_gen'],
            "Total Wind Generation (kWh)": data['wind_total_gen'],
            "Total Genset Generation (kWh)": data['genset_total_gen'],
            "Total Battery Generation (kWh)": data['battery_total_gen'],
            "Renewable Fraction (%)": (data['pv_total_gen'] + data['wind_total_gen']) / max(total_gen, 1) * 100,
            "System LCOE ($/kWh)": data['lcoe'],
            "PV Cost ($)": data['costs']['pv']['total'],
            "Wind Cost ($)": data['costs']['wind']['total'],
            "Battery Cost ($)": data['costs']['battery']['total'],
            "Genset Cost ($)": data['costs']['genset']['total'],
            "Total System Cost ($)": data['total_system_cost'],
            "Net Present Cost ($)": self.economic_calculator.calculate_npv(data['total_system_cost']),
            "Total CO2 emissions (tonne)": data['costs']['genset']['co2_emissions'] / 1000,
            **data['metrics']
        }
        if self.grid_enabled:
            result.update({"Grid Capacity (kW)": grid_kw, "Total Grid Generation (kWh)": data['grid_total_gen'],
                           "Grid Cost ($)": data['costs']['grid']['total']})
        return result