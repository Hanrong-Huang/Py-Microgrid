"""
Load analysis and flexible load management utilities for HOPP.
"""

import pandas as pd
import numpy as np
from typing import Dict

class LoadAnalyzer:
    """Analyzes and adjusts load profiles with flexible demand."""

    def __init__(self, max_load_reduction_percentage: float = 0.2, enable_flexible_load: bool = True):
        """
        Initialize LoadAnalyzer.
        
        Args:
            max_load_reduction_percentage: Maximum percentage of load that can be reduced.
                                         Default is 0.2 (20%).
            enable_flexible_load: Whether to enable flexible load management.
                                If False, no load reduction will be applied regardless of percentage.
        """
        self.max_load_reduction_percentage = max_load_reduction_percentage
        self.enable_flexible_load = enable_flexible_load
    
    def apply_flexible_load_and_adjust_battery(self, row: pd.Series) -> pd.Series:
        """
        Adjust battery usage and load to optimize energy balance.
        
        Args:
            row: DataFrame row containing generation and load data
        """
        pv = row['PV Generation (kW)']
        wind = row['Wind Generation (kW)']
        genset = row['Genset Generation (kW)']
        battery = row['Original Battery Generation (kW)']
        original_load = row['Original Load (kW)']
        
        total_generation_without_battery = pv + wind + genset
        
        # Prevent battery charging during deficits
        if battery < 0 and total_generation_without_battery + battery < original_load:
            battery = 0
        
        adjusted_generation = total_generation_without_battery + battery
        
        # Apply flexible load reduction only if enabled and needed
        if self.enable_flexible_load and adjusted_generation < original_load:
            max_reduction = self.max_load_reduction_percentage * original_load
            load_reduction = min(original_load - adjusted_generation, max_reduction)
            adjusted_load = original_load - load_reduction
        else:
            adjusted_load = original_load
        
        new_deficit = adjusted_generation - adjusted_load
        
        return pd.Series({
            'Adjusted Battery Generation (kW)': battery,
            'Adjusted Load (kW)': adjusted_load,
            'Adjusted Deficit (kW)': new_deficit
        })
    
    def calculate_performance_metrics(self, df: pd.DataFrame, 
                                   project_lifetime: int) -> Dict[str, float]:
        """
        Calculate system performance metrics.
        Uses current flexible load settings.
        
        Args:
            df: DataFrame containing generation and load data
            project_lifetime: Project lifetime in years
        """
        # Apply load adjustments with current settings
        adjusted_results = df.apply(self.apply_flexible_load_and_adjust_battery, axis=1)
        
        df['Adjusted Deficit (kW)'] = adjusted_results['Adjusted Deficit (kW)']
        df['Adjusted Load (kW)'] = adjusted_results['Adjusted Load (kW)']
        df['Adjusted Battery Generation (kW)'] = adjusted_results['Adjusted Battery Generation (kW)']

        # Calculate metrics
        original_deficits = df[
            df['Original Load (kW)'] > 
            df['PV Generation (kW)'] + df['Wind Generation (kW)'] + 
            df['Genset Generation (kW)'] + df['Original Battery Generation (kW)']
        ]
        remaining_deficits = df[df['Adjusted Deficit (kW)'] < 0]
        
        deficit_hours_fixed_one_year = len(original_deficits) - len(remaining_deficits)
        total_load_served_one_year = np.sum(df['Adjusted Load (kW)'])
        total_load_reduction_one_year = np.sum(df['Original Load (kW)'] - df['Adjusted Load (kW)'])
        total_charging_prevented_one_year = np.sum(
            np.maximum(0, -df['Original Battery Generation (kW)'] + 
                         df['Adjusted Battery Generation (kW)'])
        )
        total_load_not_served_one_year = np.sum(np.maximum(0, -df['Adjusted Deficit (kW)']))

        return {
            "Deficit Hours Fixed": deficit_hours_fixed_one_year * project_lifetime,
            "Total Load Reduction (kWh)": total_load_reduction_one_year * project_lifetime,
            "Total Load Served (kWh)": total_load_served_one_year * project_lifetime,
            "Total Charging Prevented (kWh)": total_charging_prevented_one_year * project_lifetime,
            "Demand Not Served (kWh)": total_load_not_served_one_year * project_lifetime,
            "Load Reduction Percentage": (total_load_reduction_one_year / 
                                       np.sum(df['Original Load (kW)'])) * 100,
            "Demand Met Percentage": ((total_load_served_one_year - total_load_not_served_one_year) / 
                                    total_load_served_one_year) * 100,
            "Project Lifetime (years)": project_lifetime
        }