from typing import Union

import pyomo.environ as pyomo
from pyomo.network import Port, Arc
from pyomo.environ import units as u

from py_microgrid.simulation.technologies.dispatch.dispatch import Dispatch


class GridDispatch(Dispatch):
    """
    Grid dispatch class for true electrical grid connection.
    
    This class handles the optimization of grid import/export decisions
    based on time-varying pricing and grid constraints.
    """
    grid_obj: Union[pyomo.Expression, float]
    _model: pyomo.ConcreteModel
    _blocks: pyomo.Block

    def __init__(
        self,
        pyomo_model: pyomo.ConcreteModel,
        index_set: pyomo.Set,
        system_model,
        financial_model,
        block_set_name: str = "grid",
    ):

        super().__init__(
            pyomo_model,
            index_set,
            system_model,
            financial_model,
            block_set_name=block_set_name,
        )

    def dispatch_block_rule(self, grid):
        """
        Define the dispatch block rule for grid connection.
        
        This includes parameters, variables, constraints, and ports
        for grid import/export optimization.
        """
        # Parameters
        self._create_grid_parameters(grid)
        # Variables
        self._create_grid_variables(grid)
        # Constraints
        self._create_grid_constraints(grid)
        # Ports
        self._create_grid_ports(grid)

    def max_gross_profit_objective(self, hybrid_blocks):
        """
        Objective function for maximizing gross profit from grid interaction.
        
        This considers both the revenue from selling to grid and costs of purchasing from grid.
        """
        self.obj = pyomo.Expression(
            expr=sum(
                hybrid_blocks[t].time_weighting_factor
                * self.blocks[t].time_duration
                * (self.blocks[t].electricity_sell_price * self.blocks[t].electricity_sold
                   - self.blocks[t].electricity_purchase_price * self.blocks[t].electricity_purchased)
                for t in hybrid_blocks.index_set()
            )
        )

    def min_operating_cost_objective(self, hybrid_blocks):
        """
        Objective function for minimizing operating cost from grid interaction.
        
        This focuses on minimizing the net cost of grid electricity.
        """
        self.obj = pyomo.Expression(
            expr=sum(
                hybrid_blocks[t].time_weighting_factor
                * self.blocks[t].time_duration
                * (self.blocks[t].electricity_purchase_price * self.blocks[t].electricity_purchased
                   - self.blocks[t].electricity_sell_price * self.blocks[t].electricity_sold)
                for t in hybrid_blocks.index_set()
            )
        )

    def _create_grid_parameters(self, grid):
        """
        Create parameters for grid dispatch optimization.
        """
        # Time-related parameters
        grid.epsilon = pyomo.Param(
            initialize=1e-3, 
            doc="Small epsilon value for numerical stability"
        )
        grid.time_duration = pyomo.Param(
            initialize=1.0,
            doc="Duration of time step (hours)"
        )
        
        # Pricing parameters
        grid.electricity_sell_price = pyomo.Param(
            initialize=0.0,
            doc="Price for selling electricity to grid ($/kWh)"
        )
        grid.electricity_purchase_price = pyomo.Param(
            initialize=0.0,
            doc="Price for purchasing electricity from grid ($/kWh)"
        )
        
        # Grid connection limits
        grid.import_limit = pyomo.Param(
            initialize=0.0,
            doc="Maximum power import from grid (kW)"
        )
        grid.export_limit = pyomo.Param(
            initialize=0.0,
            doc="Maximum power export to grid (kW)"
        )
        
        # Grid availability
        grid.grid_available = pyomo.Param(
            initialize=1.0,
            doc="Grid availability factor (0-1)"
        )

    def _create_grid_variables(self, grid):
        """
        Create variables for grid dispatch optimization.
        """
        # Power flows
        grid.power_imported = pyomo.Var(
            bounds=(0, None),
            doc="Power imported from grid (kW)"
        )
        grid.power_exported = pyomo.Var(
            bounds=(0, None),
            doc="Power exported to grid (kW)"
        )
        
        # Net grid power (positive = import, negative = export)
        grid.net_grid_power = pyomo.Var(
            doc="Net grid power - positive for import, negative for export (kW)"
        )
        
        # Binary variable to prevent simultaneous import/export
        grid.importing = pyomo.Var(
            domain=pyomo.Binary,
            doc="Binary variable: 1 if importing, 0 if exporting"
        )
        
        # Grid connection status
        grid.grid_connected = pyomo.Var(
            domain=pyomo.Binary,
            initialize=1,
            doc="Binary variable: 1 if grid connected, 0 if off-grid"
        )

    def _create_grid_constraints(self, grid):
        """
        Create constraints for grid dispatch optimization.
        """
        # Net power balance constraint
        grid.net_power_balance = pyomo.Constraint(
            expr=grid.net_grid_power == grid.power_imported - grid.power_exported,
            doc="Net grid power equals import minus export"
        )
        
        # Import limit constraint
        grid.import_limit_constraint = pyomo.Constraint(
            expr=grid.power_imported <= grid.import_limit * grid.importing * grid.grid_connected * grid.grid_available,
            doc="Power import cannot exceed import limit"
        )
        
        # Export limit constraint
        grid.export_limit_constraint = pyomo.Constraint(
            expr=grid.power_exported <= grid.export_limit * (1 - grid.importing) * grid.grid_connected * grid.grid_available,
            doc="Power export cannot exceed export limit"
        )
        
        # Prevent simultaneous import and export
        grid.no_simultaneous_import_export = pyomo.Constraint(
            expr=grid.power_imported * grid.power_exported <= grid.epsilon,
            doc="Cannot import and export simultaneously"
        )
        
        # Grid connection constraints
        grid.grid_connection_import = pyomo.Constraint(
            expr=grid.power_imported <= grid.import_limit * grid.grid_connected,
            doc="No import when grid disconnected"
        )
        
        grid.grid_connection_export = pyomo.Constraint(
            expr=grid.power_exported <= grid.export_limit * grid.grid_connected,
            doc="No export when grid disconnected"
        )

    def _create_grid_ports(self, grid):
        """
        Create ports for grid dispatch optimization.
        
        Ports allow the grid to interface with the hybrid system.
        """
        grid.port = Port()
        grid.port.add(grid.net_grid_power, "net_grid_power")
        grid.port.add(grid.power_imported, "power_imported")
        grid.port.add(grid.power_exported, "power_exported")
        grid.port.add(grid.grid_connected, "grid_connected")

    def update_dispatch_parameters(self, grid_block, grid_config, time_step):
        """
        Update dispatch parameters for a specific time step.
        
        Args:
            grid_block: Pyomo block for this time step
            grid_config: Grid configuration object
            time_step: Current time step index
        """
        # Update pricing based on dispatch factors
        if hasattr(grid_config, 'import_prices') and len(grid_config.import_prices) > time_step:
            grid_block.electricity_purchase_price.set_value(grid_config.import_prices[time_step])
        
        if hasattr(grid_config, 'export_prices') and len(grid_config.export_prices) > time_step:
            grid_block.electricity_sell_price.set_value(grid_config.export_prices[time_step])
        
        # Update limits
        if hasattr(grid_config, 'import_limit_kw'):
            grid_block.import_limit.set_value(grid_config.import_limit_kw)
        
        if hasattr(grid_config, 'export_limit_kw'):
            grid_block.export_limit.set_value(grid_config.export_limit_kw)
        
        # Update availability
        if hasattr(grid_config, 'enabled'):
            grid_block.grid_available.set_value(1.0 if grid_config.enabled else 0.0)
            grid_block.grid_connected.set_value(1 if grid_config.enabled else 0)

    def extract_dispatch_results(self, grid_block):
        """
        Extract dispatch results from the optimization solution.
        
        Args:
            grid_block: Pyomo block containing the solution
            
        Returns:
            dict: Dictionary containing dispatch results
        """
        try:
            results = {
                'power_imported': pyomo.value(grid_block.power_imported),
                'power_exported': pyomo.value(grid_block.power_exported),
                'net_grid_power': pyomo.value(grid_block.net_grid_power),
                'importing': pyomo.value(grid_block.importing),
                'grid_connected': pyomo.value(grid_block.grid_connected),
                'import_cost': (pyomo.value(grid_block.power_imported) * 
                               pyomo.value(grid_block.electricity_purchase_price)),
                'export_revenue': (pyomo.value(grid_block.power_exported) * 
                                  pyomo.value(grid_block.electricity_sell_price))
            }
            
            return results
            
        except Exception as e:
            # Return zeros if extraction fails
            return {
                'power_imported': 0.0,
                'power_exported': 0.0,
                'net_grid_power': 0.0,
                'importing': 0.0,
                'grid_connected': 1.0,
                'import_cost': 0.0,
                'export_revenue': 0.0
            }