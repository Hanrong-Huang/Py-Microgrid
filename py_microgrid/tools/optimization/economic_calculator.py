"""
Economic analysis utilities for financial calculations.
This module provides a focused calculator for standard energy finance metrics like LCOE.
"""

import numpy as np

class EconomicCalculator:
    """
    Handles core financial calculations for energy projects.
    This class is technology-agnostic and focuses on standardized metrics.
    """
    
    def __init__(self, discount_rate: float, project_lifetime: int):
        """
        Initialize the EconomicCalculator.
        
        Args:
            discount_rate: The annual discount rate for present value calculations (e.g., 0.06 for 6%).
            project_lifetime: The analysis period for the project in years.
        """
        if not 0 < discount_rate < 1:
            raise ValueError("Discount rate must be a float between 0 and 1.")
        if not project_lifetime > 0:
            raise ValueError("Project lifetime must be greater than zero.")
            
        self.discount_rate = discount_rate
        self.project_lifetime = project_lifetime
        
    def calculate_lcoe(self, total_lifetime_cost: float, total_lifetime_load_served_kwh: float) -> float:
        """
        Calculates the Levelized Cost of Energy (LCOE) based on total lifetime values.

        This simplified LCOE formula is suitable when annual cost and generation streams
        are treated as levelized (averaged) over the project life.

        Args:
            total_lifetime_cost: The total, undiscounted cost of the system over its entire lifetime.
            total_lifetime_load_served_kwh: The total, undiscounted energy delivered to the load (kWh)
                                            over the entire project lifetime.

        Returns:
            The LCOE in [$/kWh].
        """
        if total_lifetime_load_served_kwh == 0:
            return float('inf')

        # Capital Recovery Factor (CRF) calculates the present value of an annuity.
        # It converts a present value into a stream of equal annual payments.
        if self.discount_rate > 0:
            crf = (self.discount_rate * (1 + self.discount_rate) ** self.project_lifetime) / \
                  ((1 + self.discount_rate) ** self.project_lifetime - 1)
        else:
            crf = 1 / self.project_lifetime

        # Annualized cost is the total lifetime cost spread over the project life, considering the time value of money.
        annualized_cost = total_lifetime_cost * crf

        # Annual load served is the average energy delivered each year.
        annual_load_served = total_lifetime_load_served_kwh / self.project_lifetime
        
        if annual_load_served == 0:
            return float('inf')

        # LCOE is the annualized cost divided by the annual energy delivered.
        return annualized_cost / annual_load_served
    
    def calculate_npv(self, total_lifetime_cost: float) -> float:
        """
        Calculate the Net Present Value (NPV) of the total system cost.
        
        This method converts the total lifetime cost to its present value equivalent
        using the discount rate, assuming costs are spread evenly over the project lifetime.
        
        Args:
            total_lifetime_cost: The total, undiscounted cost over the entire lifetime.
            
        Returns:
            The Net Present Value of the total system cost in [$].
        """
        if total_lifetime_cost == 0:
            return 0.0
            
        # For simplicity, assume costs are spread evenly over the project lifetime
        annual_cost = total_lifetime_cost / self.project_lifetime
        
        # Calculate present value of annual costs over project lifetime
        if self.discount_rate > 0:
            # Present Value of Annuity formula: PV = PMT * [(1 - (1 + r)^-n) / r]
            pv_factor = (1 - (1 + self.discount_rate) ** -self.project_lifetime) / self.discount_rate
            npv = annual_cost * pv_factor
        else:
            # If discount rate is 0, NPV equals total lifetime cost
            npv = total_lifetime_cost
            
        return npv