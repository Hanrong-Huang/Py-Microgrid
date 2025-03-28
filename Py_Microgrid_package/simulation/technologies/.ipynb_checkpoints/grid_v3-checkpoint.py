from typing import Iterable, List, Sequence, Optional, Union, Any

import numpy as np
from attrs import define, field
import PySAM.Grid as GridModel
import PySAM.Singleowner as Singleowner

from hopp.simulation.technologies.sites import SiteInfo
from hopp.simulation.technologies.power_source import PowerSource
from hopp.simulation.base import BaseClass
from hopp.simulation.technologies.financial import FinancialModelType, CustomFinancialModel
from hopp.type_dec import NDArrayFloat
from hopp.utilities.validators import gt_zero


@define
class GridConfig(BaseClass):
    """
    Configuration data class for Grid. 

    Args:
        interconnect_kw: grid interconnection limit (kW)
        fin_model: Financial model. Can be any of the following:

            - a string representing an argument to `Singleowner.default`

            - a dict representing a `CustomFinancialModel`

            - an object representing a `CustomFinancialModel` or `Singleowner.Singleowner` instance

        ppa_price: PPA price [$/kWh] used in the financial model
    """
    interconnect_kw: float = field(validator=gt_zero)
    fin_model: Optional[Union[str, dict, FinancialModelType]] = None
    ppa_price: Optional[Union[Iterable, float]] = None


@define
class Grid(PowerSource):
    site: SiteInfo
    config: GridConfig
    hopp: Optional[Any] = None

    missed_load: np.ndarray = field(init=False)
    missed_load_percentage: float = field(init=False, default=0.)
    schedule_curtailed: np.ndarray = field(init=False)
    schedule_curtailed_percentage: float = field(init=False, default=0.)
    total_gen_max_feasible_year1: np.ndarray = field(init=False)

    def __attrs_post_init__(self):
        """
        Class that houses the hybrid system performance and financials. Enforces interconnection and curtailment
        limits based on PySAM's Grid module.

        Args:
            site: Power source site information
            config: dict, used to instantiate a `GridConfig` instance
            hopp: HOPP system object
        """
        system_model = GridModel.default("GenericSystemSingleOwner")

        # parse user input for financial model
        if isinstance(self.config.fin_model, str):
            financial_model = Singleowner.default(self.config.fin_model)
        elif isinstance(self.config.fin_model, dict):
            financial_model = CustomFinancialModel(self.config.fin_model)
        else:
            financial_model = self.config.fin_model

        # default
        if financial_model is None:
            financial_model = Singleowner.from_existing(system_model, "GenericSystemSingleOwner")
            financial_model.value("add_om_num_types", 1)

        super().__init__("Grid", self.site, system_model, financial_model)

        if self.config.ppa_price is not None:
            self.ppa_price = self.config.ppa_price

        self._system_model.GridLimits.enable_interconnection_limit = 1
        self._system_model.GridLimits.grid_interconnection_limit_kwac = self.config.interconnect_kw
        self._dispatch = None

        self.missed_load = np.array([0.])
        self.schedule_curtailed = np.array([0.])
        self.total_gen_max_feasible_year1 = np.array([0.])

    def simulate_grid_connection(
        self,
        hybrid_size_kw: float,
        total_gen: Union[List[float], np.ndarray],
        project_life: int,
        lifetime_sim: bool,
        total_gen_max_feasible_year1: Union[List[float], np.ndarray],
        dispatch_options: Optional[dict] = None
    ):
        # Ensure total_gen is an array
        total_gen = np.array(total_gen)
        # Extract pv and wind generation profile and calculate total renewable_gen
        pv_gen = np.array(self.hopp.system.generation_profile.pv)
        wind_gen = np.array(self.hopp.system.generation_profile.wind)
        renewable_gen = pv_gen + wind_gen

        # Calculate the maximum grid output
        max_grid_output = np.max(total_gen)
        
        # Calculate the minimum grid generation (10% of max)
        min_grid_generation = 0.1 * max_grid_output

        if self.site.follow_desired_schedule:
            lifetime_schedule = np.tile(
                [x * 1e3 for x in self.site.desired_schedule],
                int(project_life / (len(self.site.desired_schedule) // self.site.n_timesteps))
            )

            # Ensure all arrays have the same length
            min_length = min(len(lifetime_schedule), len(renewable_gen), len(total_gen))
            lifetime_schedule = lifetime_schedule[:min_length]
            renewable_gen = renewable_gen[:min_length]
            total_gen = total_gen[:min_length]

            # Create a new numpy array for the generation profile
            # This array will store the grid's output for each time step
            generation_profile = np.zeros(min_length)
            
            # Iterate through each time step
            for i in range(min_length):
                # Get the energy demand for this time step
                demand = lifetime_schedule[i]
                
                # Get the available renewable energy (PV + wind) for this time step
                renewable_available = renewable_gen[i]
                
                # Calculate the energy deficit
                # This is how much additional energy is needed from grid after renewable generation
                energy_deficit = max(0, demand - renewable_available)
                
                if energy_deficit > 0:
                    # There's a deficit, so the grid needs to be turned on
                    if energy_deficit >= min_grid_generation:
                        # If the deficit is larger than or equal to the minimum generation:
                        # The grid will try to match the deficit exactly,
                        # but won't exceed its maximum output capacity
                        generation_profile[i] = min(energy_deficit, max_grid_output)
                    else:
                        # If the deficit is smaller than the minimum generation:
                        # The grid will produce its minimum output to maintain efficiency
                        # This may result in some overgeneration
                        generation_profile[i] = min_grid_generation
                else:
                    # There's no deficit, so the grid stays off
                    generation_profile[i] = 0

            # Assign the newly created array to self.generation_profile
            self.generation_profile = generation_profile

            # Calculate missed load and curtailment
            total_generation = renewable_gen + self.generation_profile
            self.missed_load = np.maximum(lifetime_schedule - total_generation, 0)
            self.missed_load_percentage = np.sum(self.missed_load) / np.sum(lifetime_schedule)
            self.schedule_curtailed = np.maximum(total_generation - lifetime_schedule, 0)
            self.schedule_curtailed_percentage = np.sum(self.schedule_curtailed) / np.sum(lifetime_schedule)

        else:
            # If not following desired schedule, use the original total_gen
            self.generation_profile = total_gen

            
        # Ensure these attributes are set as numpy arrays
        self.generation_profile = np.array(self.generation_profile)
        self.total_gen_max_feasible_year1 = np.array(total_gen_max_feasible_year1)
        self.system_capacity_kw = hybrid_size_kw
        self.gen_max_feasible = np.minimum(
            total_gen_max_feasible_year1,
            self.config.interconnect_kw * self.site.interval / 60
        )
        self.simulate_power(project_life, lifetime_sim)

        # FIXME: updating capacity credit for reporting only.
        self.capacity_credit_percent = [i * (self.system_capacity_kw / self.config.interconnect_kw) for i in self.capacity_credit_percent]

    def calc_gen_max_feasible_kwh(self, interconnect_kw: float) -> list:
        """
        Calculates the maximum feasible generation profile that could have occurred (year 1)

        Args:
        :param interconnect_kw: Interconnection limit [kW]

        :return: maximum feasible generation [kWh]
        """
        W_ac_nom = self.calc_nominal_capacity(interconnect_kw)
        t_step = self.site.interval / 60  # hr
        E_net_max_feasible = [min(x, W_ac_nom) * t_step for x in self.total_gen_max_feasible_year1[0:self.site.n_timesteps]]  # [kWh]
        return E_net_max_feasible

    @property
    def system_capacity_kw(self) -> float:
        return self._financial_model.value('system_capacity')

    @system_capacity_kw.setter
    def system_capacity_kw(self, size_kw: float):
        self._financial_model.value('system_capacity', size_kw)

    @property
    def interconnect_kw(self) -> float:
        """Interconnection limit [kW]"""
        return self._system_model.GridLimits.grid_interconnection_limit_kwac

    @interconnect_kw.setter
    def interconnect_kw(self, interconnect_limit_kw: float):
        self._system_model.GridLimits.grid_interconnection_limit_kwac = interconnect_limit_kw

    @property
    def curtailment_ts_kw(self) -> list:
        """Grid curtailment as energy delivery limit (first year) [MW]"""
        return [i for i in self._system_model.GridLimits.grid_curtailment]

    @curtailment_ts_kw.setter
    def curtailment_ts_kw(self, curtailment_limit_timeseries_kw: Sequence):
        self._system_model.GridLimits.grid_curtailment = curtailment_limit_timeseries_kw

    @property
    def generation_profile(self) -> Sequence:
        """System power generated [kW]"""
        return self._system_model.SystemOutput.gen

    @generation_profile.setter
    def generation_profile(self, system_generation_kw: Sequence):
        self._system_model.SystemOutput.gen = system_generation_kw

    @property
    def generation_profile_wo_battery(self) -> Sequence:
        """System power generated without battery [kW]"""
        return self._financial_model.value('gen_without_battery')

    @generation_profile_wo_battery.setter
    def generation_profile_wo_battery(self, system_generation_wo_battery_kw: Sequence):
        self._system_model.SystemOutput.gen = system_generation_wo_battery_kw

    @property
    def generation_profile_pre_curtailment(self) -> Sequence:
        """System power before grid interconnect [kW]"""
        return self._system_model.Outputs.system_pre_interconnect_kwac

    @property
    def generation_curtailed(self) -> Sequence:
        """Generation curtailed due to interconnect limit [kW]"""
        curtailed = self.generation_profile
        pre_curtailed = self.generation_profile_pre_curtailment
        return [pre_curtailed[i] - curtailed[i] for i in range(len(curtailed))]

    @property
    def curtailment_percent(self) -> float:
        """Annual energy loss from curtailment and interconnection limit [%]"""
        return self._system_model.Outputs.annual_ac_curtailment_loss_percent \
               + self._system_model.Outputs.annual_ac_interconnect_loss_percent

    @property
    def capacity_factor_after_curtailment(self) -> float:
        """Capacity factor of the curtailment (year 1) [%]"""
        return self._system_model.Outputs.capacity_factor_curtailment_ac

    @property
    def capacity_factor_at_interconnect(self) -> float:
        """Capacity factor of the curtailment (year 1) [%]"""
        return self._system_model.Outputs.capacity_factor_interconnect_ac