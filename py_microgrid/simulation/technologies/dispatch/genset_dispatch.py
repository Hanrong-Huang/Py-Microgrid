from typing import Union

import pyomo.environ as pyomo
from pyomo.network import Port, Arc
from pyomo.environ import units as u

from py_microgrid.simulation.technologies.dispatch.dispatch import Dispatch


class GensetDispatch(Dispatch):
    genset_obj: Union[pyomo.Expression, float]
    _model: pyomo.ConcreteModel
    _blocks: pyomo.Block
    """

    """

    def __init__(
        self,
        pyomo_model: pyomo.ConcreteModel,
        index_set: pyomo.Set,
        system_model,
        financial_model,
        block_set_name: str = "genset",
    ):

        super().__init__(
            pyomo_model,
            index_set,
            system_model,
            financial_model,
            block_set_name=block_set_name,
        )

    def dispatch_block_rule(self, genset):
        # Parameters
        self._create_genset_parameters(genset)
        # Variables
        self._create_genset_variables(genset)
        # Constraints
        self._create_genset_constraints(genset)
        # Ports
        self._create_genset_ports(genset)

    def max_gross_profit_objective(self, hybrid_blocks):
        self.obj = pyomo.Expression(
            expr=sum(
                hybrid_blocks[t].time_weighting_factor
                * self.blocks[t].time_duration
                * self.blocks[t].electricity_sell_price
                * hybrid_blocks[t].electricity_sold
                - (1 / hybrid_blocks[t].time_weighting_factor)
                * self.blocks[t].time_duration
                * self.blocks[t].electricity_purchase_price
                * hybrid_blocks[t].electricity_purchased
                - self.blocks[t].epsilon * self.blocks[t].is_generating
                for t in hybrid_blocks.index_set()
            )
        )

    def min_operating_cost_objective(self, hybrid_blocks):
        self.obj = sum(
            hybrid_blocks[t].time_weighting_factor
            * self.blocks[t].time_duration
            * self.blocks[t].electricity_sell_price
            * (
                self.blocks[t].generation_transmission_limit
                - hybrid_blocks[t].electricity_sold
            )
            + (
                hybrid_blocks[t].time_weighting_factor
                * self.blocks[t].time_duration
                * self.blocks[t].electricity_purchase_price
                * hybrid_blocks[t].electricity_purchased
            )
            + (self.blocks[t].epsilon * self.blocks[t].is_generating)
            for t in hybrid_blocks.index_set()
        )

    def _create_variables(self, hybrid):
        hybrid.system_generation = pyomo.Var(
            doc="System generation [MW]",
            domain=pyomo.NonNegativeReals,
            units=u.MW,
        )
        hybrid.system_load = pyomo.Var(
            doc="System load [MW]",
            domain=pyomo.NonNegativeReals,
            units=u.MW,
        )
        hybrid.electricity_sold = pyomo.Var(
            doc="Electricity sold [MW]",
            domain=pyomo.NonNegativeReals,
            units=u.MW,
        )
        hybrid.electricity_purchased = pyomo.Var(
            doc="Electricity purchased [MW]",
            domain=pyomo.NonNegativeReals,
            units=u.MW,
        )

        return 0, 0

    def _create_port(self, hybrid):
        hybrid.grid_port = Port(
            initialize={
                "system_generation": hybrid.system_generation,
                "system_load": hybrid.system_load,
                "electricity_sold": hybrid.electricity_sold,
                "electricity_purchased": hybrid.electricity_purchased,
            }
        )
        return hybrid.grid_port

    def _create_constraints(self, hybrid, t):
        hybrid.generation_total = pyomo.Constraint(
            doc="hybrid system generation total",
            rule=hybrid.system_generation == sum(self.power_source_gen_vars[t]),
        )

    @staticmethod
    def _create_genset_parameters(genset):
        ##################################
        # Parameters                     #
        ##################################
        genset.epsilon = pyomo.Param(
            doc="A small value used in objective for binary logic",
            default=1e-3,
            within=pyomo.NonNegativeReals,
            mutable=True,
            units=u.USD,
        )
        genset.time_duration = pyomo.Param(
            doc="Time step [hour]",
            default=1.0,
            within=pyomo.NonNegativeReals,
            mutable=True,
            units=u.hr,
        )
        genset.electricity_sell_price = pyomo.Param(
            doc="Electricity sell price [$/MWh]",
            default=0.0,
            within=pyomo.Reals,
            mutable=True,
            units=u.USD / u.MWh,
        )
        genset.electricity_purchase_price = pyomo.Param(
            doc="Electricity purchase price [$/MWh]",
            default=0.0,
            within=pyomo.Reals,
            mutable=True,
            units=u.USD / u.MWh,
        )
        genset.generation_transmission_limit = pyomo.Param(
            doc="Grid transmission limit for generation [MW]",
            default=1000.0,
            within=pyomo.NonNegativeReals,
            mutable=True,
            units=u.MW,
        )
        genset.load_transmission_limit = pyomo.Param(
            doc="Grid transmission limit for load [MW]",
            default=1000.0,
            within=pyomo.NonNegativeReals,
            mutable=True,
            units=u.MW,
        )

    @staticmethod
    def _create_genset_variables(genset):
        ##################################
        # Variables                      #
        ##################################
        genset.system_generation = pyomo.Var(
            doc="System generation [MW]", domain=pyomo.NonNegativeReals, units=u.MW
        )
        genset.system_load = pyomo.Var(
            doc="System load [MW]", domain=pyomo.NonNegativeReals, units=u.MW
        )
        genset.electricity_sold = pyomo.Var(
            doc="Electricity sold to the grid [MW]",
            domain=pyomo.NonNegativeReals,
            bounds=(0, genset.generation_transmission_limit),
            units=u.MW,
        )
        genset.electricity_purchased = pyomo.Var(
            doc="Electricity purchased from the grid [MW]",
            domain=pyomo.NonNegativeReals,
            bounds=(0, genset.load_transmission_limit),
            units=u.MW,
        )
        genset.is_generating = pyomo.Var(
            doc="System is generating power", domain=pyomo.Binary, units=u.dimensionless
        )

    @staticmethod
    def _create_genset_constraints(genset):
        ##################################
        # Constraints                    #
        ##################################
        genset.balance = pyomo.Constraint(
            doc="Transmission energy balance",
            expr=(
                genset.electricity_sold - genset.electricity_purchased
                == genset.system_generation - genset.system_load
            ),
        )
        genset.sales_transmission_limit = pyomo.Constraint(
            doc="Transmission limit on electricity sales",
            expr=genset.electricity_sold
            <= genset.generation_transmission_limit * genset.is_generating,
        )
        genset.purchases_transmission_limit = pyomo.Constraint(
            doc="Transmission limit on electricity purchases",
            expr=(
                genset.electricity_purchased
                <= genset.load_transmission_limit * (1 - genset.is_generating)
            ),
        )

    @staticmethod
    def _create_genset_ports(genset):
        ##################################
        # Ports                          #
        ##################################
        genset.port = Port()
        genset.port.add(genset.system_generation)
        genset.port.add(genset.system_load)
        genset.port.add(genset.electricity_sold)
        genset.port.add(genset.electricity_purchased)

    def initialize_parameters(self):
        grid_limit_kw = self._system_model.value("grid_interconnection_limit_kwac")
        self.generation_transmission_limit = [grid_limit_kw / 1e3] * len(
            self.blocks.index_set()
        )
        self.load_transmission_limit = [grid_limit_kw / 1e3] * len(
            self.blocks.index_set()
        )

    def update_time_series_parameters(self, start_time: int):
        n_horizon = len(self.blocks.index_set())
        dispatch_factors = self._financial_model.value("dispatch_factors_ts")
        ppa_price = self._financial_model.value("ppa_price_input")[0]
        if start_time + n_horizon > len(dispatch_factors):
            prices = list(dispatch_factors[start_time:])
            prices.extend(list(dispatch_factors[0 : n_horizon - len(prices)]))
        else:
            prices = dispatch_factors[start_time : start_time + n_horizon]
        # NOTE: Assuming the same prices
        self.electricity_sell_price = [
            norm_price * ppa_price * 1e3 for norm_price in prices
        ]
        self.electricity_purchase_price = [
            norm_price * ppa_price * 1e3 for norm_price in prices
        ]

    @property
    def electricity_sell_price(self) -> list:
        return [
            self.blocks[t].electricity_sell_price.value for t in self.blocks.index_set()
        ]

    @electricity_sell_price.setter
    def electricity_sell_price(self, price_per_mwh: list):
        if len(price_per_mwh) == len(self.blocks):
            for t, price in zip(self.blocks, price_per_mwh):
                self.blocks[t].electricity_sell_price.set_value(
                    round(price, self.round_digits)
                )
        else:
            raise ValueError(
                "'price_per_mwh' list must be the same length as time horizon"
            )

    @property
    def electricity_purchase_price(self) -> list:
        return [
            self.blocks[t].electricity_purchase_price.value
            for t in self.blocks.index_set()
        ]

    @electricity_purchase_price.setter
    def electricity_purchase_price(self, price_per_mwh: list):
        if len(price_per_mwh) == len(self.blocks):
            for t, price in zip(self.blocks, price_per_mwh):
                self.blocks[t].electricity_purchase_price.set_value(
                    round(price, self.round_digits)
                )
        else:
            raise ValueError(
                "'price_per_mwh' list must be the same length as time horizon"
            )

    @property
    def generation_transmission_limit(self) -> list:
        return [
            self.blocks[t].generation_transmission_limit.value
            for t in self.blocks.index_set()
        ]

    @generation_transmission_limit.setter
    def generation_transmission_limit(self, limit_mw: list):
        if len(limit_mw) == len(self.blocks):
            for t, limit in zip(self.blocks, limit_mw):
                self.blocks[t].generation_transmission_limit.set_value(
                    round(limit, self.round_digits)
                )
        else:
            raise ValueError("'limit_mw' list must be the same length as time horizon")

    @property
    def load_transmission_limit(self) -> list:
        return [
            self.blocks[t].load_transmission_limit.value
            for t in self.blocks.index_set()
        ]

    @load_transmission_limit.setter
    def load_transmission_limit(self, limit_mw: list):
        if len(limit_mw) == len(self.blocks):
            for t, limit in zip(self.blocks, limit_mw):
                self.blocks[t].load_transmission_limit.set_value(
                    round(limit, self.round_digits)
                )
        else:
            raise ValueError("'limit_mw' list must be the same length as time horizon")

    @property
    def system_generation(self) -> list:
        return [self.blocks[t].system_generation.value for t in self.blocks.index_set()]

    # @system_generation.setter
    # def system_generation(self, system_gen_mw: list):
    #     if len(system_gen_mw) == len(self.blocks):
    #         for t, gen in zip(self.blocks, system_gen_mw):
    #             self.blocks[t].system_generation.set_value(round(gen, self.round_digits))
    #     else:
    #         raise ValueError("'system_gen_mw' list must be the same length as time horizon")

    @property
    def system_load(self) -> list:
        return [self.blocks[t].system_load.value for t in self.blocks.index_set()]

    # @system_load.setter
    # def system_load(self, system_load_mw: list):
    #     if len(system_load_mw) == len(self.blocks):
    #         for t, load in zip(self.blocks, system_load_mw):
    #             self.blocks[t].system_load.set_value(round(load, self.round_digits))
    #     else:
    #         raise ValueError("'system_load_mw' list must be the same length as time horizon")

    @property
    def electricity_sold(self) -> list:
        return [self.blocks[t].electricity_sold.value for t in self.blocks.index_set()]

    @property
    def electricity_purchased(self) -> list:
        return [
            self.blocks[t].electricity_purchased.value for t in self.blocks.index_set()
        ]

    @property
    def is_generating(self) -> list:
        return [self.blocks[t].is_generating.value for t in self.blocks.index_set()]

    @property
    def not_generating(self) -> list:
        return [self.blocks[t].not_generating.value for t in self.blocks.index_set()]
