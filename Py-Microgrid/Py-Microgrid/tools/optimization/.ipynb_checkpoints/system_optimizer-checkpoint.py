"""
System optimization utilities for HOPP.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from scipy.optimize import minimize

from hopp.simulation import HoppInterface
from hopp.utilities import ConfigManager
from hopp.tools.analysis.bos import EconomicCalculator
from .load_analyzer import LoadAnalyzer

class SystemOptimizer:
    """Handles system optimization and configuration."""
    
    def __init__(self, 
                 yaml_file_path: str, 
                 economic_calculator: EconomicCalculator,
                 enable_flexible_load: bool = True,
                 max_load_reduction_percentage: float = 0.2):
        """Initialize SystemOptimizer."""
        self.yaml_file_path = yaml_file_path
        self.economic_calculator = economic_calculator
        self.config_manager = ConfigManager()
        self.load_analyzer = LoadAnalyzer(
            enable_flexible_load=enable_flexible_load,
            max_load_reduction_percentage=max_load_reduction_percentage
        )

    def optimize_system(self, bounds: List[Tuple[float, float]], 
                       initial_conditions: List[List[float]]) -> Optional[Dict[str, Any]]:
        """
        Optimize system configuration.
        
        Args:
            bounds: List of (min, max) tuples for each parameter.
            initial_conditions: List of initial parameter sets to try.
            
        Returns:
            Dictionary containing optimal system configuration and metrics.
        """
        best_result = None
        best_lcoe = float('inf')
        
        for x0 in initial_conditions:
            try:
                result = minimize(
                    self.penalized_objective,
                    x0,
                    method='Nelder-Mead',
                    bounds=bounds,
                    options={'maxiter': 300, 'xatol': 1, 'fatol': 1e-3}
                )
                
                if result.success:
                    optimal_config = [
                        int(round(result.x[0])),
                        int(round(result.x[1])),
                        self.round_battery_capacity(result.x[2]),
                        int(round(result.x[3])),
                        int(round(result.x[4]))
                    ]
                    lcoe, optimal_results = self.objective_function(optimal_config)
                    
                    if lcoe < best_lcoe:
                        best_lcoe = lcoe
                        best_result = optimal_results
            
            except Exception as e:
                print(f"Optimization failed for initial point {x0}: {str(e)}")
                continue
        
        return best_result

    @staticmethod
    def round_battery_capacity(capacity: float) -> float:
        """Round battery capacity to nearest MWh."""
        return round(capacity / 1000) * 1000
    
    def penalized_objective(self, x: List[float]) -> float:
        """Calculate penalized objective function value."""
        x_rounded = [
            int(round(x[0])),  # PV
            int(round(x[1])),  # Wind turbines
            self.round_battery_capacity(x[2]),  # Battery capacity kWh
            int(round(x[3])),  # Battery capacity kW
            int(round(x[4]))   # Grid interconnect
        ]
        penalized_lcoe, _ = self.objective_function(x_rounded)
        return penalized_lcoe
    
    def objective_function(self, x: List[float]) -> Tuple[float, Dict[str, Any]]:
        """Calculate objective function value and system metrics."""
        pv_size, num_turbines, battery_capacity_kwh, battery_capacity_kw, grid_interconnect_kw = x
        battery_capacity_kwh = self.round_battery_capacity(battery_capacity_kwh)
        
        # Update configuration
        config = self.config_manager.load_yaml_safely(self.yaml_file_path)
        config['technologies']['pv']['system_capacity_kw'] = float(pv_size)
        config['technologies']['wind']['num_turbines'] = int(num_turbines)
        config['technologies']['battery']['system_capacity_kwh'] = float(battery_capacity_kwh)
        config['technologies']['battery']['system_capacity_kw'] = float(battery_capacity_kw)
        config['technologies']['grid']['interconnect_kw'] = float(grid_interconnect_kw)
        self.config_manager.save_yaml_safely(config, self.yaml_file_path)

        # Run simulation
        try:
            hopp = HoppInterface(self.yaml_file_path)
            hopp.simulate(project_life=self.economic_calculator.project_lifetime)
        except Exception as e:
            print(f"Simulation failed: {e}")
            return 1e6, {}  # Return high LCOE for failed simulations

        hybrid_plant = hopp.system

        # Calculate generation
        pv_total_generation = np.sum(hybrid_plant.generation_profile.pv)
        wind_total_generation = np.sum(hybrid_plant.generation_profile.wind)
        battery_total_generation = np.sum(hybrid_plant.generation_profile.battery)
        genset_total_generation = np.sum(hybrid_plant.generation_profile.grid)
        total_system_generation = pv_total_generation + wind_total_generation + genset_total_generation

        # Calculate costs
        costs = self._calculate_costs(hybrid_plant, config, genset_total_generation)
        total_system_cost = sum(cost['total'] for cost in costs.values())

        # Create DataFrame for load analysis
        df = pd.DataFrame({
            'PV Generation (kW)': np.array(hybrid_plant.generation_profile.pv[:8760]),
            'Wind Generation (kW)': np.array(hybrid_plant.generation_profile.wind[:8760]),
            'Genset Generation (kW)': np.array(hybrid_plant.generation_profile.grid[:8760]),
            'Original Battery Generation (kW)': np.array(hybrid_plant.generation_profile.battery[:8760]),
            'Original Load (kW)': np.array(hybrid_plant.site.desired_schedule[:8760]) * 1000
        })

        # Calculate metrics
        metrics = self.load_analyzer.calculate_performance_metrics(df, self.economic_calculator.project_lifetime)
        
        # Calculate LCOE
        lcoe = self.economic_calculator.calculate_lcoe(total_system_cost, metrics['Total Load Served (kWh)'])

        # Prepare results
        result = {
            "PV Capacity (kW)": pv_size,
            "Wind Turbine Capacity (kW)": num_turbines * 1000,
            "Genset Capacity (kW)": grid_interconnect_kw,
            "Battery Energy Capacity (kWh)": battery_capacity_kwh,
            "Battery Power Capacity (kW)": battery_capacity_kw,
            "Total System Generation (kWh)": total_system_generation,
            "Total PV Generation (kWh)": pv_total_generation,
            "Total Wind Generation (kWh)": wind_total_generation,
            "Total Genset Generation (kWh)": genset_total_generation,
            "Total Battery Generation (kWh)": battery_total_generation,
            "Total CO2 emissions (tonne)": costs['genset']['co2_emissions']/1000,
            "System NPC ($)": self.economic_calculator.calculate_present_value(total_system_cost),
            "System LCOE ($/kWh)": lcoe,
            **metrics
        }

        return lcoe, result

    def _calculate_costs(self, hybrid_plant, config, genset_total_generation) -> Dict[str, Dict[str, float]]:
        """Calculate costs for all system components."""
        genset_capacity_kw = hybrid_plant.grid.interconnect_kw
        genset_op_hours_per_year = np.sum(np.array(hybrid_plant.grid.generation_profile) > 0) / self.economic_calculator.project_lifetime
        generator_life_hours = 15000
        generator_life_years = generator_life_hours / genset_op_hours_per_year
        num_genset_replacements = float(self.economic_calculator.project_lifetime / generator_life_years) - 1

        # Genset costs
        genset_install_cost = genset_capacity_kw * 500
        genset_replace_cost = num_genset_replacements * genset_capacity_kw * 500
        genset_om_cost = 0.03 * genset_capacity_kw * genset_op_hours_per_year * self.economic_calculator.project_lifetime
        fuel_consumption = genset_total_generation * 0.250
        fuel_cost = fuel_consumption * 1.20
        co2_emissions = fuel_consumption * 2.618

        # Battery costs
        battery_capacity_kwh = config['technologies']['battery']['system_capacity_kwh']
        battery_installed_cost = hybrid_plant.battery.total_installed_cost
        battery_replace_cost = ((self.economic_calculator.project_lifetime/15) - 1) * battery_capacity_kwh * (battery_installed_cost/battery_capacity_kwh)
        battery_om_cost = 10 * battery_capacity_kwh * self.economic_calculator.project_lifetime

        return {
            'genset': {
                'total': genset_install_cost + genset_replace_cost + genset_om_cost + fuel_cost,
                'co2_emissions': co2_emissions
            },
            'battery': {
                'total': battery_installed_cost + battery_replace_cost + battery_om_cost
            },
            'pv': {
                'total': hybrid_plant.pv.total_installed_cost + 
                        10 * hybrid_plant.system_capacity_kw.pv * self.economic_calculator.project_lifetime
            },
            'wind': {
                'total': hybrid_plant.wind.total_installed_cost + 
                        40 * hybrid_plant.system_capacity_kw.wind * self.economic_calculator.project_lifetime
            }
        }