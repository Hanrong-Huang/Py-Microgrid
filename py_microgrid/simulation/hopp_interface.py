from __future__ import annotations
from pathlib import Path
from typing import Union, TYPE_CHECKING

from .hopp import Hopp

# avoid potential circular dep
if TYPE_CHECKING:
    from py_microgrid.simulation.hybrid_simulation import HybridSimulation


class HoppInterface:
    """
    Main interface for HOPP simulations.

    Args:
        configuration: Top level configuration for a HOPP simulation. Can be either
            a string/Path to a YAML configuration file, or a dict with the same 
            structure. The structure is:

                - **name**: Optional name for the simulation

                - **site**: Site information. See :class:`py_microgrid.simulation.technologies.sites.SiteInfo`

                - **technologies**: Technology information. See :class:`py_microgrid.simulation.hybrid_simulation.TechnologiesConfig`

                - **config**: Additional config options

                    - **dispatch_options**: Dispatch optimization options. See :class:`py_microgrid.simulation.technologies.dispatch.hybrid_dispatch_options.HybridDispatchOptions`

                    - **cost_info**: Cost info. See :class:`py_microgrid.tools.analysis.bos.cost_calculator.CostCalculator`

                    - **simulation_options**: Nested ``dict``, i.e., ``{'pv': {'skip_financial': bool}}`` (optional) nested dictionary of simulation options. First level key is technology consistent with ``technologies``

    """
    def __init__(self, configuration: Union[dict, str, Path]):
        self.configuration = configuration

        if isinstance(self.configuration, (str, Path)):
            self.py_microgrid = Hopp.from_file(self.configuration)

        elif isinstance(self.configuration, dict):
            self.py_microgrid = Hopp.from_dict(self.configuration)

    def reinitialize(self):
        pass

    def simulate(self, project_life: int = 25, lifetime_sim: bool = False):
        self.py_microgrid.simulate(project_life, lifetime_sim)

    @property
    def system(self) -> "HybridSimulation":
        """Returns the configured simulation instance."""
        return self.py_microgrid.system

    def parse_input(self):
        pass

    def parse_output(self):
        self.annual_energies = self.py_microgrid.system.annual_energies
        self.wind_plus_solar_npv = self.py_microgrid.system.net_present_values.wind + self.py_microgrid.system.net_present_values.pv
        self.npvs = self.py_microgrid.system.net_present_values
        self.wind_installed_cost = self.py_microgrid.system.wind.total_installed_cost
        self.solar_installed_cost = self.py_microgrid.system.pv.total_installed_cost
        self.hybrid_installed_cost = self.py_microgrid.system.grid.total_installed_cost

    def print_output(self):
        print("Wind Installed Cost: {}".format(self.wind_installed_cost))
        print("Solar Installed Cost: {}".format(self.solar_installed_cost))
        print("Hybrid Installed Cost: {}".format(self.hybrid_installed_cost))
        print("Wind NPV: {}".format(self.py_microgrid.system.net_present_values.wind))
        print("Solar NPV: {}".format(self.py_microgrid.system.net_present_values.pv))
        print("Hybrid NPV: {}".format(self.py_microgrid.system.net_present_values.hybrid))
        print("Wind + Solar Expected NPV: {}".format(self.wind_plus_solar_npv))
