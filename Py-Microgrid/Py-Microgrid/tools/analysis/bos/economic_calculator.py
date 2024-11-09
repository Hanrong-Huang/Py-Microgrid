"""
Economic analysis utilities for HOPP BOS calculations.
Handles financial calculations for microgrid systems including LCOE, penalties, and cost calculations.
"""

from typing import Dict, Any

class EconomicCalculator:
    """Handles economic calculations and BOS costs for the microgrid system."""
    
    def __init__(self, discount_rate: float, project_lifetime: int):
        """
        Initialize EconomicCalculator.
        
        Args:
            discount_rate: Annual discount rate for present value calculations.
            project_lifetime: Project lifetime in years.
        """
        self.discount_rate = discount_rate
        self.project_lifetime = project_lifetime
        
        # Default BOS component costs
        self.default_costs = {
            'genset': {
                'install_cost_per_kw': 500,
                'replacement_cost_per_kw': 500,
                'om_cost_per_kw_per_op_hour': 0.03,
                'fuel_cost_per_l': 1.20,
                'specific_fuel_consumption_l_per_kwh': 0.250,
                'operational_life_hours': 15000,
                'specific_co2_per_l_fuel': 2.618  # kg/L
            },
            'battery': {
                'om_cost_per_kwh_per_year': 10,
                'operational_life_years': 15
            },
            'pv': {
                'om_cost_per_kw_per_year': 10
            },
            'wind': {
                'om_cost_per_kw_per_year': 40
            }
        }
    
    def calculate_present_value(self, future_value: float) -> float:
        """
        Calculate present value of future costs.
        
        Args:
            future_value: Future value to be discounted.
            
        Returns:
            float: Present value.
        """
        return future_value / ((1 + self.discount_rate) ** self.project_lifetime)

    def calculate_lcoe(self, total_system_cost: float, total_load_served: float) -> float:
            """
            Calculate Levelized Cost of Energy (LCOE).
            
            Args:
                total_system_cost: Total system cost over lifetime.
                total_load_served: Total energy delivered to load over lifetime.
                
            Returns:
                float: LCOE in $/kWh.
            """
            present_value_costs = self.calculate_present_value(total_system_cost)
            present_value_load = self.calculate_present_value(total_load_served)
            return present_value_costs / present_value_load
    
    def calculate_penalty(self, demand_met_percentage: float) -> float:
        """
        Calculate penalty for unmet demand in BOS cost calculations.
        
        This function applies a graduated penalty based on the percentage of unmet demand:
        - For small deviations (up to 1% unmet demand), a gentler penalty is applied.
        - For larger deviations (more than 1% unmet demand), a steeper penalty is used.
        
        Args:
            demand_met_percentage: Percentage of demand met by the system.
            
        Returns:
            float: Penalty value to be added to LCOE.
        """
        unmet_percentage = 100 - demand_met_percentage
        if unmet_percentage <= 1:
            return 0.1 * unmet_percentage  # More gradual for small deviations
        else:
            return 0.2 * unmet_percentage  # Steeper for larger deviations
    
    def get_component_costs(self, component: str) -> Dict[str, Any]:
        """
        Get default BOS costs for a specific component.
        
        Args:
            component: Component name ('genset', 'battery', 'pv', or 'wind').
            
        Returns:
            dict: Default costs for the specified component.
            
        Raises:
            ValueError: If component is not recognized.
        """
        if component not in self.default_costs:
            raise ValueError(f"Unknown component: {component}. Valid options are: {list(self.default_costs.keys())}")
        return self.default_costs[component]