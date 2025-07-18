"""
Diesel Generator (Genset) implementation for py_microgrid
This module provides a proper diesel generator model with fuel consumption,
emissions, operational constraints, and cost calculations.
"""

from typing import List, Optional, Union, Any
import numpy as np
from attrs import define, field
from py_microgrid.simulation.technologies.sites import SiteInfo
from py_microgrid.simulation.technologies.power_source import PowerSource
from py_microgrid.simulation.base import BaseClass
from py_microgrid.utilities.validators import gt_zero


@define
class GensetConfig(BaseClass):
    """
    Configuration data class for Genset (diesel generator).
    
    Args:
        interconnect_kw: Maximum genset capacity [kW]
        fuel_type: Type of fuel used (diesel, natural_gas, etc.)
        efficiency: Electrical efficiency of the genset [0-1]
        minimum_load_factor: Minimum load factor as fraction of rated capacity [0-1]
        specific_fuel_consumption: Fuel consumption rate [L/kWh]
        fuel_cost_per_liter: Cost of fuel per liter [$/L]
        start_cost: Cost to start the genset [$]
        maintenance_cost_per_hour: Maintenance cost per operating hour [$/h]
        co2_emissions_per_liter: CO2 emissions per liter of fuel [kg CO2/L]
        operational_life_hours: Expected operational life [hours]
    """
    interconnect_kw: float = field(validator=gt_zero)
    fuel_type: str = field(default="diesel")
    efficiency: float = field(default=0.35)
    minimum_load_factor: float = field(default=0.30)
    specific_fuel_consumption: float = field(default=0.25)  # L/kWh
    fuel_cost_per_liter: float = field(default=1.20)  # $/L
    start_cost: float = field(default=50.0)  # $ per start
    maintenance_cost_per_hour: float = field(default=2.0)  # $/hour
    co2_emissions_per_liter: float = field(default=2.618)  # kg CO2/L
    operational_life_hours: float = field(default=15000.0)  # hours
    install_cost_per_kw: float = field(default=650.0)  # $/kW
    replacement_cost_per_kw: float = field(default=650.0)  # $/kW


@define
class Genset(BaseClass):
    """
    Diesel Generator (Genset) class that models a backup power generator.
    
    This class properly implements diesel generator characteristics including:
    - Minimum load constraints
    - Fuel consumption calculations
    - Emissions tracking
    - Operational hours tracking
    - Start/stop costs
    - Efficiency curves
    """
    site: SiteInfo
    config: GensetConfig
    
    # Calculated properties (set after simulation)
    generation_profile: np.ndarray = field(init=False, default=None)
    fuel_consumption_profile: np.ndarray = field(init=False, default=None) 
    operational_hours_profile: np.ndarray = field(init=False, default=None)
    starts_profile: np.ndarray = field(init=False, default=None)
    
    # Aggregate metrics
    total_generation_kwh: float = field(init=False, default=0.0)
    total_fuel_consumption_liters: float = field(init=False, default=0.0)
    total_operational_hours: float = field(init=False, default=0.0)
    total_starts: int = field(init=False, default=0)
    total_co2_emissions_kg: float = field(init=False, default=0.0)
    
    def __attrs_post_init__(self):
        """Initialize the genset with default values."""
        self.generation_profile = np.zeros(8760)
        self.fuel_consumption_profile = np.zeros(8760)
        self.operational_hours_profile = np.zeros(8760)
        self.starts_profile = np.zeros(8760)
        
    def simulate_power(self, demand_profile: np.ndarray, project_lifetime: int = 1) -> None:
        """
        Simulate genset power generation based on demand profile with proper diesel generator constraints.
        
        Args:
            demand_profile: Hourly power demand for genset [kW]
            project_lifetime: Project lifetime in years
        """
        n_hours = len(demand_profile)
        self.generation_profile = np.zeros(n_hours)
        self.fuel_consumption_profile = np.zeros(n_hours)
        self.operational_hours_profile = np.zeros(n_hours)
        self.starts_profile = np.zeros(n_hours)
        
        # Minimum load constraint
        min_load_kw = self.config.interconnect_kw * self.config.minimum_load_factor
        previous_state = False  # Track if genset was running in previous hour
        
        for hour in range(n_hours):
            demand_kw = demand_profile[hour]
            
            if demand_kw > 0:
                # Apply minimum load constraint
                if demand_kw < min_load_kw:
                    # If demand is below minimum, run at minimum load
                    actual_output = min_load_kw
                else:
                    # Run at demanded output, capped at rated capacity
                    actual_output = min(demand_kw, self.config.interconnect_kw)
                
                # Calculate fuel consumption based on efficiency
                fuel_consumption = self._calculate_fuel_consumption(actual_output)
                
                # Track start if genset was off in previous hour
                if not previous_state:
                    self.starts_profile[hour] = 1
                    
                # Update profiles
                self.generation_profile[hour] = actual_output
                self.fuel_consumption_profile[hour] = fuel_consumption
                self.operational_hours_profile[hour] = 1.0
                previous_state = True
            else:
                # Genset is off
                previous_state = False
                
        # Calculate aggregate metrics over project lifetime
        self.total_generation_kwh = np.sum(self.generation_profile) * project_lifetime
        self.total_fuel_consumption_liters = np.sum(self.fuel_consumption_profile) * project_lifetime
        self.total_operational_hours = np.sum(self.operational_hours_profile) * project_lifetime
        self.total_starts = int(np.sum(self.starts_profile) * project_lifetime)
        self.total_co2_emissions_kg = self.total_fuel_consumption_liters * self.config.co2_emissions_per_liter

    def dispatch_power(self, deficit_kw: float) -> tuple:
        """
        Dispatch genset power to meet a specific deficit, respecting minimum turn-on power and max capacity.
        
        Args:
            deficit_kw: Power deficit that needs to be met [kW]
            
        Returns:
            tuple: (actual_power_output, power_served_to_deficit)
                - actual_power_output: Actual genset output considering minimum load constraint
                - power_served_to_deficit: Power that actually serves the deficit (may be less than actual output)
        """
        if deficit_kw <= 0:
            return 0.0, 0.0
        
        # Minimum turn-on power constraint
        min_turn_on_power = self.config.interconnect_kw * self.config.minimum_load_factor
        
        # If deficit is too small to justify turning on genset, don't turn on
        if deficit_kw < min_turn_on_power:
            return 0.0, 0.0
        
        # Calculate actual genset output (must be at least minimum load)
        actual_output = max(min_turn_on_power, min(deficit_kw, self.config.interconnect_kw))
        
        # Power that serves the deficit is the minimum of actual output and deficit
        power_served = min(actual_output, deficit_kw)
        
        return actual_output, power_served
        
    @property
    def minimum_turn_on_power_kw(self) -> float:
        """Minimum power genset must produce when turned on [kW]"""
        return self.config.interconnect_kw * self.config.minimum_load_factor
        
    @property
    def maximum_capacity_kw(self) -> float:
        """Maximum genset capacity [kW]"""
        return self.config.interconnect_kw
        
    def _calculate_fuel_consumption(self, output_kw: float) -> float:
        """
        Calculate fuel consumption based on genset output and efficiency curve.
        
        Args:
            output_kw: Genset output power [kW]
            
        Returns:
            Fuel consumption [L/hour]
        """
        if output_kw <= 0:
            return 0.0
            
        # Fuel consumption based on specific consumption rate
        # The specific_fuel_consumption already accounts for average efficiency
        fuel_consumption = output_kw * self.config.specific_fuel_consumption
        return fuel_consumption
        
    def calculate_total_costs(self, project_lifetime: int) -> dict:
        """
        Calculate total costs for the genset over project lifetime.
        
        Args:
            project_lifetime: Project lifetime in years
            
        Returns:
            Dictionary with cost breakdown
        """
        # Installation cost
        install_cost = self.config.interconnect_kw * self.config.install_cost_per_kw
        
        # Replacement cost based on operational hours vs operational life
        if self.total_operational_hours > 0:
            # Calculate how many times the genset exceeds its operational life
            life_cycles = self.total_operational_hours / self.config.operational_life_hours
            num_replacements = max(0, int(life_cycles))
        else:
            num_replacements = 0
            
        replacement_cost = num_replacements * self.config.interconnect_kw * self.config.replacement_cost_per_kw
        
        # Fuel cost
        fuel_cost = self.total_fuel_consumption_liters * self.config.fuel_cost_per_liter
        
        # Maintenance cost
        maintenance_cost = self.total_operational_hours * self.config.maintenance_cost_per_hour
        
        # Start costs
        start_cost = self.total_starts * self.config.start_cost
        
        # CO2 emissions cost (if carbon pricing is applied)
        carbon_cost = 0.0  # This can be set via configuration
        
        total_cost = install_cost + replacement_cost + fuel_cost + maintenance_cost + start_cost + carbon_cost
        
        return {
            'install_cost': install_cost,
            'replacement_cost': replacement_cost,
            'fuel_cost': fuel_cost,
            'maintenance_cost': maintenance_cost,
            'start_cost': start_cost,
            'carbon_cost': carbon_cost,
            'total_cost': total_cost,
            'co2_emissions_kg': self.total_co2_emissions_kg,
            'co2_emissions_tonnes': self.total_co2_emissions_kg / 1000.0
        }
        
    @property
    def capacity_kw(self) -> float:
        """Genset rated capacity [kW]"""
        return self.config.interconnect_kw
        
    @property
    def capacity_factor(self) -> float:
        """Capacity factor based on generation profile [%]"""
        if self.config.interconnect_kw == 0:
            return 0.0
        return (np.sum(self.generation_profile) / (self.config.interconnect_kw * len(self.generation_profile))) * 100
        
    @property
    def average_load_factor(self) -> float:
        """Average load factor when operating [%]"""
        operating_hours = self.operational_hours_profile > 0
        if np.sum(operating_hours) == 0:
            return 0.0
        avg_output = np.mean(self.generation_profile[operating_hours])
        return (avg_output / self.config.interconnect_kw) * 100
        
    def get_performance_summary(self) -> dict:
        """Get a summary of genset performance metrics."""
        return {
            'Total Generation (kWh)': self.total_generation_kwh,
            'Total Fuel Consumption (L)': self.total_fuel_consumption_liters,
            'Total Operational Hours': self.total_operational_hours,
            'Total Starts': self.total_starts,
            'Total CO2 Emissions (kg)': self.total_co2_emissions_kg,
            'Capacity Factor (%)': self.capacity_factor,
            'Average Load Factor (%)': self.average_load_factor,
            'Fuel Efficiency (kWh/L)': self.total_generation_kwh / max(self.total_fuel_consumption_liters, 1)
        }