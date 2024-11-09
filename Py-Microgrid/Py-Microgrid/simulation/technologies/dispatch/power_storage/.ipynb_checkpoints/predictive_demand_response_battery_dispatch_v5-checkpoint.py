import numpy as np
import pyomo.environ as pyomo
from pyomo.environ import units as u
import PySAM.BatteryStateful as BatteryModel
import PySAM.Singleowner as Singleowner
from typing import Optional, List, Dict
from hopp.simulation.technologies.dispatch.power_storage.simple_battery_dispatch_heuristic import (
    SimpleBatteryDispatchHeuristic,
)
from statsmodels.tsa.holtwinters import ExponentialSmoothing

class PredictiveDemandResponseBatteryDispatch(SimpleBatteryDispatchHeuristic):
    """
    Advanced battery dispatch class incorporating predictive elements and demand response strategies.

    Key Features:
    1. Uses Holt-Winters exponential smoothing for accurate forecasting.
    2. Formulates an optimization problem over the prediction horizon for optimal dispatch.
    3. Implements demand response by shifting loads during peak periods.
    4. Enforces battery power fraction limits and SOC constraints.
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
        super().__init__(
            pyomo_model,
            index_set,
            system_model,
            financial_model,
            fixed_dispatch=None,
            block_set_name=block_set_name,
            dispatch_options=dispatch_options,
        )
        # Prediction and demand response parameters
        self.prediction_horizon = 24  # hours
        self.demand_response_factor = 0.1  # Fraction of peak demand to shift
        self.peak_threshold_percentile = 75  # Percentile to identify peak demand
        self.demand_response_window = 6  # Hours to consider for demand response

    def _predict_future_demand_and_generation(
        self,
        gen: np.ndarray,
        demand: np.ndarray,
    ) -> (np.ndarray, np.ndarray):
        """
        Predict future demand and generation using Holt-Winters exponential smoothing.

        Args:
            gen (np.ndarray): Array of generation values.
            demand (np.ndarray): Array of demand values.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Predicted future generation and demand.
        """
        # Apply Holt-Winters exponential smoothing
        future_gen_model = ExponentialSmoothing(
            gen, trend='add', seasonal='add', seasonal_periods=24
        ).fit()
        future_demand_model = ExponentialSmoothing(
            demand, trend='add', seasonal='add', seasonal_periods=24
        ).fit()

        future_gen = future_gen_model.forecast(self.prediction_horizon)
        future_demand = future_demand_model.forecast(self.prediction_horizon)

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

        # Redistribute shifted load to off-peak periods
        off_peak_periods = np.where(~peak_periods)[0]
        total_off_peak_hours = len(off_peak_periods)

        if total_off_peak_hours > 0:
            total_shift_amount = np.sum(shift_amounts)
            shift_per_hour = total_shift_amount / total_off_peak_hours
            adjusted_demand[off_peak_periods] += shift_per_hour

        return adjusted_demand

    def _optimize_dispatch(
        self,
        gen: np.ndarray,
        demand: np.ndarray,
        soc_initial: float,
    ) -> np.ndarray:
        """
        Optimize the battery dispatch over the prediction horizon.

        Args:
            gen (np.ndarray): Predicted generation values.
            demand (np.ndarray): Adjusted demand values.
            soc_initial (float): Initial State of Charge.

        Returns:
            np.ndarray: Optimal dispatch schedule.
        """
        model = pyomo.ConcreteModel()

        T = len(gen)
        model.T = pyomo.RangeSet(0, T - 1)

        # Variables
        model.charge_power = pyomo.Var(model.T, bounds=(0, self.maximum_power))
        model.discharge_power = pyomo.Var(model.T, bounds=(0, self.maximum_power))
        model.soc = pyomo.Var(model.T, bounds=(self.min_soc, self.max_soc))

        # Objective: Minimize the total net load (demand - generation + battery)
        model.obj = pyomo.Objective(
            expr=sum(
                (demand[t] - gen[t] + model.discharge_power[t] - model.charge_power[t])
                for t in model.T
            ),
            sense=pyomo.minimize,
        )

        # Constraints
        def soc_constraint(model, t):
            if t == 0:
                return model.soc[t] == soc_initial + (
                    self.charge_efficiency / 100.0 * model.charge_power[t]
                    - (1 / (self.discharge_efficiency / 100.0)) * model.discharge_power[t]
                ) * self.time_duration[0] / self.capacity
            else:
                return model.soc[t] == model.soc[t - 1] + (
                    self.charge_efficiency / 100.0 * model.charge_power[t]
                    - (1 / (self.discharge_efficiency / 100.0)) * model.discharge_power[t]
                ) * self.time_duration[0] / self.capacity

        model.soc_constraint = pyomo.Constraint(model.T, rule=soc_constraint)

        # Solve the optimization problem
        solver = pyomo.SolverFactory('glpk')
        result = solver.solve(model, tee=False)

        # Extract dispatch schedule
        dispatch_schedule = np.zeros(T)
        for t in model.T:
            dispatch_schedule[t] = (
                model.discharge_power[t].value - model.charge_power[t].value
            ) / self.maximum_power

        return dispatch_schedule

    def set_fixed_dispatch(
        self,
        gen: List[float],
        grid_limit: List[float],
        goal_power: List[float],
    ):
        """
        Set fixed dispatch based on predictive optimization method.

        Args:
            gen (List[float]): Generation profiles (e.g., PV, wind).
            grid_limit (List[float]): Grid power limits.
            goal_power (List[float]): Desired load profile.
        """
        gen = np.array(gen)
        grid_limit = np.array(grid_limit)
        demand_profile = np.array(goal_power)

        num_periods = len(gen)
        self._fixed_dispatch = np.zeros(num_periods)

        # Adjust demand profile with demand response
        adjusted_demand = self._apply_demand_response(demand_profile)

        soc_current = self.initial_soc

        for t in range(0, num_periods, self.prediction_horizon):
            end_t = min(t + self.prediction_horizon, num_periods)
            gen_future = gen[t:end_t]
            demand_future = adjusted_demand[t:end_t]

            # Predict future generation and demand
            future_gen, future_demand = self._predict_future_demand_and_generation(
                gen_future, demand_future
            )

            # Optimize dispatch over the prediction horizon
            dispatch_schedule = self._optimize_dispatch(
                future_gen, future_demand, soc_current
            )

            # Update fixed dispatch and SOC
            self._fixed_dispatch[t:end_t] = dispatch_schedule
            soc_current = self.update_soc(dispatch_schedule[-1], soc_current)

        # Enforce power fraction limits
        self._enforce_power_fraction_limits()

        # Fix dispatch variables in the model
        self._fix_dispatch_model_variables()

    def _enforce_power_fraction_limits(self):
        """
        Enforces battery power fraction limits and adjusts _fixed_dispatch accordingly.
        """
        for t in self.blocks.index_set():
            fd = self._fixed_dispatch[t]
            if fd > 0.0:  # Discharging
                max_fd = self.max_discharge_fraction[t]
                if fd > max_fd:
                    fd = max_fd
            elif fd < 0.0:  # Charging
                max_fc = -self.max_charge_fraction[t]
                if fd < max_fc:
                    fd = max_fc
            self._fixed_dispatch[t] = fd

    def _fix_dispatch_model_variables(self):
        """
        Fix dispatch variables in the Pyomo model based on calculated dispatch factors.
        """
        soc0 = self.model.initial_soc.value
        for t in self.blocks.index_set():
            dispatch_factor = self._fixed_dispatch[t]
            soc_new = self.update_soc(dispatch_factor, soc0)
            self.blocks[t].soc.fix(soc_new)
            soc0 = soc_new

            if dispatch_factor == 0.0:
                self.blocks[t].charge_power.fix(0.0)
                self.blocks[t].discharge_power.fix(0.0)
            elif dispatch_factor > 0.0:
                self.blocks[t].charge_power.fix(0.0)
                self.blocks[t].discharge_power.fix(
                    dispatch_factor * self.maximum_power
                )
            elif dispatch_factor < 0.0:
                self.blocks[t].discharge_power.fix(0.0)
                self.blocks[t].charge_power.fix(
                    -dispatch_factor * self.maximum_power
                )

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

        soc = max(self.min_soc, min(self.max_soc, soc))
        return soc

    def update_prediction_parameters(
        self,
        prediction_horizon: int = 24,
        demand_response_factor: float = 0.1,
        peak_threshold_percentile: float = 75,
        demand_response_window: int = 6,
    ):
        """
        Update the prediction and demand response parameters.

        Args:
            prediction_horizon (int, optional): Number of hours to predict. Defaults to 24.
            demand_response_factor (float, optional): Fraction of peak demand to shift. Defaults to 0.1.
            peak_threshold_percentile (float, optional): Percentile to identify peak demand. Defaults to 75.
            demand_response_window (int, optional): Hours to consider for demand response. Defaults to 6.
        """
        self.prediction_horizon = prediction_horizon
        self.demand_response_factor = demand_response_factor
        self.peak_threshold_percentile = peak_threshold_percentile
        self.demand_response_window = demand_response_window
