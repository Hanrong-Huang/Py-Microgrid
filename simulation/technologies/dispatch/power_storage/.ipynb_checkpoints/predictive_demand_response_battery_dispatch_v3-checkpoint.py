import numpy as np
import pyomo.environ as pyomo
from pyomo.environ import units as u
import PySAM.BatteryStateful as BatteryModel
import PySAM.Singleowner as Singleowner
from typing import Optional, List
from hopp.simulation.technologies.dispatch.power_storage.simple_battery_dispatch_heuristic import SimpleBatteryDispatchHeuristic


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
        dispatch_options: Optional[dict] = None
    ):
        super().__init__(
            pyomo_model,
            index_set,
            system_model,
            financial_model,
            fixed_dispatch=None,
            block_set_name=block_set_name,
            dispatch_options=dispatch_options
        )
        self.base_prediction_horizon = 24
        self.demand_response_factor = 0.1  # Amount of peak demand to shift (10%)
        self.peak_threshold_percentile = 75  # 75th percentile for peak demand identification
        self.demand_response_window = 6  # Hours before and after peak to shift demand
        self.charge_threshold = 0.7  # Charge when SOC is below this level
        self.discharge_threshold = 0.3  # Avoid discharging when SOC is below this level

    def _adjust_prediction_horizon(self, current_soc):
        """Dynamically adjust prediction horizon based on battery SOC."""
        if current_soc < 0.3:
            return max(6, self.base_prediction_horizon // 2)
        elif current_soc > 0.7:
            return min(48, self.base_prediction_horizon * 2)
        else:
            return self.base_prediction_horizon

    def _predict_future_demand_and_generation(self, gen, goal_power, current_soc):
        """Predict future demand and generation using exponential smoothing."""
        prediction_horizon = self._adjust_prediction_horizon(current_soc)
        alpha = 0.3  # Smoothing factor
        future_gen = np.zeros(prediction_horizon)
        future_demand = np.zeros(prediction_horizon)

        future_gen[0] = gen[0]
        future_demand[0] = goal_power[0]

        for t in range(1, prediction_horizon):
            future_gen[t] = alpha * gen[t] + (1 - alpha) * future_gen[t - 1]
            future_demand[t] = alpha * goal_power[t] + (1 - alpha) * future_demand[t - 1]

        return future_gen, future_demand

    def _identify_peak_demand_periods(self, goal_power):
        """Identify periods of peak demand for demand response."""
        threshold = np.percentile(goal_power, self.peak_threshold_percentile)
        return goal_power > threshold

    def _apply_demand_response(self, goal_power):
        """Apply demand response by shifting load from peak to off-peak periods."""
        peak_periods = self._identify_peak_demand_periods(goal_power)
        adjusted_demand = goal_power.copy()

        for t in range(len(goal_power)):
            if peak_periods[t]:
                shift_amount = goal_power[t] * self.demand_response_factor
                adjusted_demand[t] -= shift_amount

                # Distribute the shifted load to surrounding off-peak hours
                start = max(0, t - self.demand_response_window)
                end = min(len(goal_power), t + self.demand_response_window + 1)
                off_peak_hours = [i for i in range(start, end) if not peak_periods[i]]

                if off_peak_hours:
                    shift_per_hour = shift_amount / len(off_peak_hours)
                    for i in off_peak_hours:
                        adjusted_demand[i] += shift_per_hour

        return adjusted_demand

    def _calculate_net_load(self, demand, generation, grid_limit):
        """Calculate net load considering generation and grid limits."""
        return demand - (generation + grid_limit)

    def _predictive_method(self, gen, goal_power, grid_limit):
        adjusted_demand = self._apply_demand_response(goal_power)

        for t in self.blocks.index_set():
            current_soc = self.blocks[t].soc.value if t > 0 else self.initial_soc
            future_gen, future_demand = self._predict_future_demand_and_generation(gen[t:], adjusted_demand[t:], current_soc)

            current_demand = adjusted_demand[t]
            current_gen = gen[t] + grid_limit[t]  # Total available generation including grid

            net_load = self._calculate_net_load(current_demand, current_gen, grid_limit[t])
            excess_energy = current_gen - current_demand

            if excess_energy < 0:
                # We have a deficit, discharge the battery
                discharge_amount = min(-excess_energy, self.maximum_power,
                                       (current_soc - self.minimum_soc) * self.capacity)
                fd = discharge_amount / self.maximum_power
            else:
                # We have excess energy, consider charging
                future_net_load = self._calculate_net_load(future_demand, future_gen, grid_limit[t:t + len(future_gen)])
                future_deficit = np.sum(future_net_load > 0)

                if future_deficit > 0 and current_soc < self.maximum_soc:
                    # There's a future deficit predicted and the battery isn't full
                    charge_amount = min(excess_energy, self.maximum_power,
                                        (self.maximum_soc - current_soc) * self.capacity)
                    fd = -charge_amount / self.maximum_power
                else:
                    # No future deficit or battery is full, don't charge
                    fd = 0

            # Ensure we're within the discharge/charge limits
            fd = max(min(fd, self.max_discharge_fraction[t]), -self.max_charge_fraction[t])

            self._fixed_dispatch[t] = fd

    def set_fixed_dispatch(self, gen: list, grid_limit: list, goal_power: list):
        self._predictive_method(gen, goal_power, grid_limit)
        self._heuristic_method(gen, goal_power)  # Apply the heuristic method to enforce limits
        self._fix_dispatch_model_variables()

    def _fix_dispatch_model_variables(self):
        soc0 = self.model.initial_soc.value
        for t in self.blocks.index_set():
            dispatch_factor = self._fixed_dispatch[t]
            self.blocks[t].soc.fix(self.update_soc(dispatch_factor, soc0))
            soc0 = self.blocks[t].soc.value

            if dispatch_factor == 0.0:
                # Do nothing
                self.blocks[t].charge_power.fix(0.0)
                self.blocks[t].discharge_power.fix(0.0)
            elif dispatch_factor > 0.0:
                # Discharging
                self.blocks[t].charge_power.fix(0.0)
                self.blocks[t].discharge_power.fix(dispatch_factor * self.maximum_power)
            elif dispatch_factor < 0.0:
                # Charging
                self.blocks[t].discharge_power.fix(0.0)
                self.blocks[t].charge_power.fix(-dispatch_factor * self.maximum_power)

    def update_prediction_parameters(self,
                                     base_prediction_horizon: int = 24,
                                     demand_response_factor: float = 0.1,
                                     peak_threshold_percentile: float = 75,
                                     demand_response_window: int = 6):
        """
        Update the prediction and demand response parameters.
        """
        self.base_prediction_horizon = base_prediction_horizon
        self.demand_response_factor = demand_response_factor
        self.peak_threshold_percentile = peak_threshold_percentile
        self.demand_response_window = demand_response_window
