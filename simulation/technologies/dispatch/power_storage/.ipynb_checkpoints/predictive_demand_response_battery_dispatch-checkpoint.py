import numpy as np
import pyomo.environ as pyomo
from pyomo.environ import units as u
import PySAM.BatteryStateful as BatteryModel
import PySAM.Singleowner as Singleowner
from typing import Optional, List, Dict
from hopp.simulation.technologies.dispatch.power_storage.simple_battery_dispatch_heuristic import (
    SimpleBatteryDispatchHeuristic,
)


class PredictiveDemandResponseBatteryDispatch(SimpleBatteryDispatchHeuristic):
    """
    Advanced battery dispatch class incorporating predictive elements and demand response strategies.

    Key Features:
    1. Strict adherence to load requirements: Never causes deficits by charging.
    2. Excess energy utilization: Charges battery only when there's surplus generation.
    3. Sophisticated prediction: Uses exponential smoothing for forecasting.
    4. Demand Response: Shifts load from peak to off-peak periods.
    5. Adaptive prediction horizon: Adjusts based on battery state of charge.
    6. Enforces battery power fraction limits.
    """

    def __init__(
        self,
        pyomo_model: pyomo.ConcreteModel,
        index_set: pyomo.Set,
        system_model: BatteryModel.BatteryStateful,
        financial_model: Singleowner.Singleowner,
        block_set_name: str = "predictive_demand_response_battery",
        dispatch_options: Optional[Dict] = None,
    ):
        """
        Initialize the PredictiveDemandResponseBatteryDispatch class.

        Args:
            pyomo_model (pyomo.ConcreteModel): The Pyomo optimization model.
            index_set (pyomo.Set): The set of indices for time periods.
            system_model (BatteryModel.BatteryStateful): The battery system model.
            financial_model (Singleowner.Singleowner): The financial model.
            block_set_name (str, optional): Name for the block set. Defaults to "predictive_demand_response_battery".
            dispatch_options (Optional[dict], optional): Additional dispatch options. Defaults to None.
        """
        super().__init__(
            pyomo_model,
            index_set,
            system_model,
            financial_model,
            fixed_dispatch=None,
            block_set_name=block_set_name,
            dispatch_options=dispatch_options,
        )
        # Prediction and Demand Response Parameters
        self.base_prediction_horizon = 24  # Base number of hours to predict
        self.demand_response_factor = 0.1  # Fraction of peak demand to shift (10%)
        self.peak_threshold_percentile = 75  # Percentile to identify peak demand
        self.demand_response_window = 6  # Hours before and after peak to shift demand
        self.charge_threshold = 0.7  # Charge when SOC is below this level
        self.discharge_threshold = 0.3  # Avoid discharging when SOC is below this level
        self.alpha = 0.3  # Smoothing factor for exponential smoothing

    def _adjust_prediction_horizon(self, current_soc: float) -> int:
        """
        Dynamically adjust prediction horizon based on battery SOC.

        Args:
            current_soc (float): Current State of Charge (SOC) of the battery.

        Returns:
            int: Adjusted prediction horizon in hours.
        """
        if current_soc < 0.3:
            return max(6, self.base_prediction_horizon // 2)
        elif current_soc > 0.7:
            return min(48, self.base_prediction_horizon * 2)
        else:
            return self.base_prediction_horizon

    def _predict_future_demand_and_generation(
        self,
        gen: np.ndarray,
        demand: np.ndarray,
        current_soc: float,
    ) -> (np.ndarray, np.ndarray):
        """
        Predict future demand and generation using exponential smoothing.

        Args:
            gen (np.ndarray): Array of generation values.
            demand (np.ndarray): Array of demand values.
            current_soc (float): Current SOC of the battery.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Predicted future generation and demand.
        """
        prediction_horizon = len(gen)  # Adjusted horizon length
        alpha = self.alpha

        # Apply exponential smoothing
        future_gen = np.zeros(prediction_horizon)
        future_demand = np.zeros(prediction_horizon)

        if prediction_horizon > 0:
            future_gen[0] = gen[0]
            future_demand[0] = demand[0]
            for t in range(1, prediction_horizon):
                future_gen[t] = alpha * gen[t] + (1 - alpha) * future_gen[t - 1]
                future_demand[t] = alpha * demand[t] + (1 - alpha) * future_demand[t - 1]

        return future_gen, future_demand

    def _identify_peak_demand_periods(self, demand: np.ndarray) -> np.ndarray:
        """
        Identify periods of peak demand for demand response.

        Args:
            demand (np.ndarray): Array of demand values.

        Returns:
            np.ndarray: Boolean array indicating peak periods.
        """
        threshold = np.percentile(demand, self.peak_threshold_percentile)
        return demand > threshold

    def _apply_demand_response(self, demand: np.ndarray) -> np.ndarray:
        """
        Apply demand response by shifting load from peak to off-peak periods.

        Args:
            demand (np.ndarray): Array of demand values.

        Returns:
            np.ndarray: Adjusted demand profile after applying demand response.
        """
        peak_periods = self._identify_peak_demand_periods(demand)
        adjusted_demand = demand.copy()
        shift_amounts = adjusted_demand[peak_periods] * self.demand_response_factor
        adjusted_demand[peak_periods] -= shift_amounts

        # Distribute the shifted load to off-peak periods
        off_peak_periods = np.where(~peak_periods)[0]
        total_off_peak_hours = len(off_peak_periods)

        if total_off_peak_hours > 0:
            total_shift_amount = np.sum(shift_amounts)
            shift_per_hour = total_shift_amount / total_off_peak_hours
            adjusted_demand[off_peak_periods] += shift_per_hour

        return adjusted_demand

    def _calculate_net_load(
        self,
        demand: float,
        generation: float,
        grid_limit: float,
    ) -> float:
        """
        Calculate net load considering generation and grid limits.

        Args:
            demand (float): Demand value.
            generation (float): Generation value.
            grid_limit (float): Grid power limit.

        Returns:
            float: Net load value.
        """
        # Net load is demand minus total available generation
        return demand - generation - grid_limit

    def _predictive_method(
        self,
        gen: List[float],
        grid_limit: List[float],
        demand_profile: List[float],
    ):
        """
        Main predictive dispatch method.

        Args:
            gen (List[float]): Generation profiles (e.g., PV, wind).
            grid_limit (List[float]): Grid power limits.
            demand_profile (List[float]): Desired load profile.
        """
        # Convert lists to NumPy arrays for efficient computation
        gen = np.array(gen)
        grid_limit = np.array(grid_limit)
        demand_profile = np.array(demand_profile)

        num_periods = len(gen)

        # Apply Demand Response to adjust the demand profile
        adjusted_demand = self._apply_demand_response(demand_profile)

        # Initialize SOC values
        soc_values = np.zeros(num_periods + 1)
        soc_values[0] = self.initial_soc

        # Prepare the fixed dispatch list
        self._fixed_dispatch = np.zeros(num_periods)

        # Iterate over each time period to determine dispatch actions
        for t in range(num_periods):
            current_soc = soc_values[t]
            horizon = self._adjust_prediction_horizon(current_soc)
            end_idx = min(t + horizon, num_periods)

            # Extract future data for prediction
            gen_future = gen[t:end_idx]
            demand_future = adjusted_demand[t:end_idx]
            grid_limit_future = grid_limit[t:end_idx]

            # Predict future generation and demand
            future_gen, future_demand = self._predict_future_demand_and_generation(
                gen_future, demand_future, current_soc
            )

            # Current generation includes grid limits
            current_gen = gen[t] + grid_limit[t]

            # Calculate net load for current period
            net_load = self._calculate_net_load(
                adjusted_demand[t], current_gen, grid_limit[t]
            )
            excess_energy = -net_load  # Positive if surplus, negative if deficit

            if net_load > 0:
                # Deficit: discharge the battery
                discharge_amount = min(
                    net_load,
                    self.maximum_power,
                    (current_soc - self.discharge_threshold) * self.capacity,
                )
                fd = (
                    discharge_amount / self.maximum_power
                    if self.maximum_power > 0
                    else 0
                )
            else:
                # Surplus: consider charging
                future_net_load = future_demand - future_gen - grid_limit_future
                future_deficit = np.any(future_net_load > 0)

                if future_deficit and current_soc < self.charge_threshold:
                    # Future deficit predicted and battery isn't full
                    charge_amount = min(
                        -excess_energy,
                        self.maximum_power,
                        (self.charge_threshold - current_soc) * self.capacity,
                    )
                    fd = (
                        -charge_amount / self.maximum_power
                        if self.maximum_power > 0
                        else 0
                    )
                else:
                    fd = 0

            # Enforce dispatch limits
            self._fixed_dispatch[t] = fd

            # Update SOC for the next period
            soc_values[t + 1] = self.update_soc(fd, current_soc)

    def set_fixed_dispatch(
        self,
        gen: List[float],
        grid_limit: List[float],
        goal_power: List[float],
    ):
        """
        Set fixed dispatch based on predictive and heuristic methods.

        Args:
            gen (List[float]): Generation profiles (e.g., PV, wind).
            grid_limit (List[float]): Grid power limits.
            goal_power (List[float]): Desired load profile.
        """
        # Load profile from the site desired schedule
        demand_profile = self.site.desired_schedule

        # Ensure the demand_profile length matches gen length
        if len(demand_profile) < len(gen):
            # Pad demand_profile with the last value to match the length
            demand_profile = np.pad(
                demand_profile, (0, len(gen) - len(demand_profile)), "edge"
            )
        elif len(demand_profile) > len(gen):
            demand_profile = demand_profile[: len(gen)]

        # Apply the predictive dispatch method
        self._predictive_method(gen, grid_limit, demand_profile)

        # Enforce power fraction limits
        self.check_gen_grid_limit(gen, grid_limit)
        self._set_power_fraction_limits(gen, grid_limit)
        self._enforce_power_fraction_limits()

        # Fix the dispatch variables in the Pyomo model
        self._fix_dispatch_model_variables()

    def _enforce_power_fraction_limits(self):
        """
        Enforces battery power fraction limits and adjusts _fixed_dispatch accordingly.
        """
        for t in self.blocks.index_set():
            fd = self._fixed_dispatch[t]
            # Enforce SOC limits
            if fd > 0.0:  # Discharging
                if fd > self.max_discharge_fraction[t]:
                    fd = self.max_discharge_fraction[t]
            elif fd < 0.0:  # Charging
                if -fd > self.max_charge_fraction[t]:
                    fd = -self.max_charge_fraction[t]
            self._fixed_dispatch[t] = fd

    def _fix_dispatch_model_variables(self):
        """
        Fix dispatch variables in the Pyomo model based on calculated dispatch factors.
        """
        soc0 = self.model.initial_soc.value
        for t in self.blocks.index_set():
            dispatch_factor = self._fixed_dispatch[t]

            # Update SOC based on dispatch factor
            soc_new = self.update_soc(dispatch_factor, soc0)
            self.blocks[t].soc.fix(soc_new)
            soc0 = soc_new

            if dispatch_factor == 0.0:
                # No charging or discharging
                self.blocks[t].charge_power.fix(0.0)
                self.blocks[t].discharge_power.fix(0.0)
            elif dispatch_factor > 0.0:
                # Discharging
                self.blocks[t].charge_power.fix(0.0)
                self.blocks[t].discharge_power.fix(
                    dispatch_factor * self.maximum_power
                )
            elif dispatch_factor < 0.0:
                # Charging
                self.blocks[t].discharge_power.fix(0.0)
                self.blocks[t].charge_power.fix(
                    -dispatch_factor * self.maximum_power
                )

    def update_prediction_parameters(
        self,
        base_prediction_horizon: int = 24,
        demand_response_factor: float = 0.1,
        peak_threshold_percentile: float = 75,
        demand_response_window: int = 6,
        alpha: float = 0.3,
    ):
        """
        Update the prediction and demand response parameters.

        Args:
            base_prediction_horizon (int, optional): Base number of hours to predict. Defaults to 24.
            demand_response_factor (float, optional): Fraction of peak demand to shift. Defaults to 0.1.
            peak_threshold_percentile (float, optional): Percentile to identify peak demand. Defaults to 75.
            demand_response_window (int, optional): Hours before and after peak to shift demand. Defaults to 6.
            alpha (float, optional): Smoothing factor for exponential smoothing. Defaults to 0.3.
        """
        self.base_prediction_horizon = base_prediction_horizon
        self.demand_response_factor = demand_response_factor
        self.peak_threshold_percentile = peak_threshold_percentile
        self.demand_response_window = demand_response_window
        self.alpha = alpha

    def update_soc(self, power_fraction: float, soc0: float) -> float:
        """
        Update the State of Charge (SOC) based on the dispatch factor.

        Args:
            power_fraction (float): Dispatch factor for the current period.
            soc0 (float): Previous SOC value.

        Returns:
            float: Updated SOC value.
        """
        if power_fraction > 0.0:
            discharge_power = power_fraction * self.maximum_power
            soc = (
                soc0
                - self.time_duration[0]
                * (1 / (self.discharge_efficiency / 100.0) * discharge_power)
                / self.capacity
            )
        elif power_fraction < 0.0:
            charge_power = -power_fraction * self.maximum_power
            soc = (
                soc0
                + self.time_duration[0]
                * (self.charge_efficiency / 100.0 * charge_power)
                / self.capacity
            )
        else:
            soc = soc0

        min_soc = self._system_model.value("minimum_SOC") / 100
        max_soc = self._system_model.value("maximum_SOC") / 100

        soc = max(min_soc, min(max_soc, soc))

        return soc
