from typing import Iterable, List, Sequence, Optional, Union, Any
import os
import pandas as pd
import numpy as np
from attrs import define, field
import PySAM.Grid as GridModel
import PySAM.Singleowner as Singleowner

from py_microgrid.simulation.technologies.sites import SiteInfo
from py_microgrid.simulation.technologies.power_source import PowerSource
from py_microgrid.simulation.base import BaseClass
from py_microgrid.simulation.technologies.financial import FinancialModelType, CustomFinancialModel
from py_microgrid.type_dec import NDArrayFloat
from py_microgrid.utilities.validators import gt_zero


@define
class GridConfig(BaseClass):
    """
    Configuration data class for Grid (actual electrical grid connection).
    
    This represents a true grid connection that can import/export power based on dispatch factors.

    Args:
        enabled: Whether grid connection is enabled (True for grid-connected, False for off-grid)
        interconnect_kw: Maximum grid interconnection capacity (kW)
        import_limit_kw: Maximum power import from grid (kW). If None, uses interconnect_kw
        export_limit_kw: Maximum power export to grid (kW). If None, uses interconnect_kw
        dispatch_factors_file: Path to CSV file containing hourly dispatch factors (8760 values)
        import_price_multiplier: Multiplier for import price (dispatch factor * base_price * multiplier)
        export_price_multiplier: Multiplier for export price (dispatch factor * base_price * multiplier)
        base_import_price: Base import price ($/kWh) before applying dispatch factors
        base_export_price: Base export price ($/kWh) before applying dispatch factors
        allow_export: Whether power export to grid is allowed
        fin_model: Financial model configuration
    """
    interconnect_kw: float = field(validator=gt_zero)
    enabled: bool = field(default=True)
    import_limit_kw: Optional[float] = field(default=None)
    export_limit_kw: Optional[float] = field(default=None)
    dispatch_factors_file: Optional[str] = field(default=None)
    import_price_multiplier: float = field(default=1.0)
    export_price_multiplier: float = field(default=None)
    base_import_price: float = field(default=None)  # $/kWh
    base_export_price: float = field(default=None)  # $/kWh
    allow_export: bool = field(default=True)
    fin_model: Optional[Union[str, dict, FinancialModelType]] = None


@define
class Grid(PowerSource):
    """
    True electrical grid connection for microgrid systems.
    
    This class represents an actual connection to the electrical grid that can:
    - Import power when local generation is insufficient
    - Export excess power when enabled
    - Use dispatch factors for time-varying pricing
    - Operate in grid-connected or off-grid modes
    """
    site: SiteInfo
    config: GridConfig
    py_microgrid: Optional[Any] = None

    # Grid-specific attributes
    dispatch_factors: np.ndarray = field(init=False)
    import_prices: np.ndarray = field(init=False)
    export_prices: np.ndarray = field(init=False)
    power_imported: np.ndarray = field(init=False)
    power_exported: np.ndarray = field(init=False)
    grid_revenue: float = field(init=False, default=0.0)
    grid_cost: float = field(init=False, default=0.0)
    
    def __attrs_post_init__(self):
        """
        Initialize the grid connection with dispatch factors and pricing.
        """
        # Initialize financial model
        system_model = GridModel.default("PVWattsSingleOwner")
        
        if isinstance(self.config.fin_model, str):
            financial_model = Singleowner.default(self.config.fin_model)
        elif isinstance(self.config.fin_model, dict):
            financial_model = CustomFinancialModel(self.config.fin_model)
        else:
            financial_model = self.config.fin_model

        if financial_model is None:
            financial_model = Singleowner.from_existing(system_model, "PVWattsSingleOwner")
            financial_model.value("add_om_num_types", 1)

        super().__init__("Grid", self.site, system_model, financial_model)
        
        # Load default pricing parameters from configuration if not specified
        from py_microgrid.simulation.config import get_parameter_with_default
        if self.config.base_import_price is None:
            self.config.base_import_price = get_parameter_with_default('grid', 0.12, 'pricing', 'base_import_price')
        if self.config.base_export_price is None:
            self.config.base_export_price = get_parameter_with_default('grid', 0.08, 'pricing', 'base_export_price')
        if self.config.export_price_multiplier is None:
            self.config.export_price_multiplier = get_parameter_with_default('grid', 0.8, 'pricing', 'export_price_multiplier')
        
        # Set interconnection limits
        self._system_model.GridLimits.enable_interconnection_limit = 1
        self._system_model.GridLimits.grid_interconnection_limit_kwac = self.config.interconnect_kw
        
        # Set import/export limits
        if self.config.import_limit_kw is None:
            self.config.import_limit_kw = self.config.interconnect_kw
        if self.config.export_limit_kw is None:
            self.config.export_limit_kw = self.config.interconnect_kw
            
        # Load dispatch factors
        self._load_dispatch_factors()
        
        # Initialize arrays
        self.power_imported = np.zeros(8760)
        self.power_exported = np.zeros(8760)
        self.generation_profile = np.zeros(8760)
        
    def _load_dispatch_factors(self):
        """
        Load dispatch factors from CSV file or use default values.
        """
        if self.config.dispatch_factors_file and os.path.exists(self.config.dispatch_factors_file):
            # Load custom dispatch factors
            try:
                df = pd.read_csv(self.config.dispatch_factors_file, header=None)
                if len(df) != 8760:
                    raise ValueError(f"Dispatch factors file must contain exactly 8760 values, got {len(df)}")
                self.dispatch_factors = df.iloc[:, 0].values
            except Exception as e:
                raise ValueError(f"Error loading dispatch factors file: {e}")
        else:
            # Use default dispatch factors file
            default_file = os.path.join(
                os.path.dirname(__file__), 
                "..", "resource_files", "grid", "dispatch_factors_ts.csv"
            )
            if os.path.exists(default_file):
                try:
                    df = pd.read_csv(default_file, header=None)
                    self.dispatch_factors = df.iloc[:, 0].values
                except Exception as e:
                    # Fallback to simple time-of-use pattern
                    self.dispatch_factors = self._generate_default_dispatch_factors()
            else:
                # Fallback to simple time-of-use pattern
                self.dispatch_factors = self._generate_default_dispatch_factors()
        
        # Calculate time-varying prices
        self.import_prices = (self.dispatch_factors * 
                             self.config.base_import_price * 
                             self.config.import_price_multiplier)
        self.export_prices = (self.dispatch_factors * 
                             self.config.base_export_price * 
                             self.config.export_price_multiplier)
    
    def _generate_default_dispatch_factors(self):
        """
        Generate default dispatch factors representing typical grid pricing patterns.
        Loads time-of-use patterns from configuration.
        """
        # Load time-of-use parameters from configuration
        from py_microgrid.simulation.config import get_parameter_with_default
        off_peak_factor = get_parameter_with_default('grid', 0.7, 'dispatch_factors', 'off_peak_factor')
        peak_factor = get_parameter_with_default('grid', 1.3, 'dispatch_factors', 'peak_factor')
        standard_factor = get_parameter_with_default('grid', 1.0, 'dispatch_factors', 'standard_factor')
        
        off_peak_start = get_parameter_with_default('grid', 0, 'time_of_use', 'off_peak_start')
        off_peak_end = get_parameter_with_default('grid', 6, 'time_of_use', 'off_peak_end')
        peak_start = get_parameter_with_default('grid', 18, 'time_of_use', 'peak_start')
        peak_end = get_parameter_with_default('grid', 22, 'time_of_use', 'peak_end')
        
        factors = np.ones(8760) * standard_factor
        
        for day in range(365):
            day_start = day * 24
            # Off-peak hours
            factors[day_start + off_peak_start:day_start + off_peak_end] = off_peak_factor
            # Peak hours
            factors[day_start + peak_start:day_start + peak_end] = peak_factor
            # Standard hours (6 AM to 6 PM, 10 PM to midnight): 1.0
            # (already set to 1.0 by default)
            
        return factors
    
    def simulate_grid_connection(
        self,
        hybrid_size_kw: float,
        local_generation: Union[List[float], np.ndarray],
        load_demand: Union[List[float], np.ndarray],
        project_life: int,
        lifetime_sim: bool
    ):
        """
        Simulate grid connection behavior for import/export based on local generation and demand.
        
        Args:
            hybrid_size_kw: Total hybrid system capacity (kW)
            local_generation: Local generation profile (kW) from PV, wind, etc.
            load_demand: Load demand profile (kW) 
            project_life: Project lifetime (years)
            lifetime_sim: Whether to simulate full lifetime
        """
        if not self.config.enabled:
            # Off-grid mode: no grid interaction
            self.power_imported = np.zeros(len(local_generation))
            self.power_exported = np.zeros(len(local_generation))
            self.generation_profile = np.zeros(len(local_generation))
            return
            
        # Convert to numpy arrays
        local_generation = np.array(local_generation)
        load_demand = np.array(load_demand)
        
        # Ensure arrays are same length
        min_length = min(len(local_generation), len(load_demand), len(self.dispatch_factors))
        local_generation = local_generation[:min_length]
        load_demand = load_demand[:min_length]
        
        # Initialize arrays
        self.power_imported = np.zeros(min_length)
        self.power_exported = np.zeros(min_length)
        self.generation_profile = np.zeros(min_length)
        
        # Calculate grid interaction for each time step
        for i in range(min_length):
            local_gen = local_generation[i]
            demand = load_demand[i]
            
            # Calculate energy balance
            energy_balance = local_gen - demand
            
            if energy_balance < 0:
                # Local generation insufficient - import from grid
                import_needed = abs(energy_balance)
                self.power_imported[i] = min(import_needed, self.config.import_limit_kw)
                self.power_exported[i] = 0
                
                # Grid acts as a source to meet deficit
                self.generation_profile[i] = self.power_imported[i]
                
            elif energy_balance > 0 and self.config.allow_export:
                # Excess generation - export to grid
                export_available = energy_balance
                self.power_exported[i] = min(export_available, self.config.export_limit_kw)
                self.power_imported[i] = 0
                
                # Grid acts as a sink for excess (negative generation)
                self.generation_profile[i] = -self.power_exported[i]
                
            else:
                # Balanced or export not allowed
                self.power_imported[i] = 0
                self.power_exported[i] = 0
                self.generation_profile[i] = 0
        
        # Calculate financial metrics
        self._calculate_grid_financials()
        
        # Set system capacity for PySAM
        self.system_capacity_kw = hybrid_size_kw
        
        # Simulate power using PySAM
        self.simulate_power(project_life, lifetime_sim)
    
    def _calculate_grid_financials(self):
        """
        Calculate grid-related costs and revenues.
        """
        if not self.config.enabled:
            self.grid_cost = 0.0
            self.grid_revenue = 0.0
            return
            
        # Calculate hourly costs and revenues
        import_costs = self.power_imported * self.import_prices[:len(self.power_imported)]
        export_revenues = self.power_exported * self.export_prices[:len(self.power_exported)]
        
        # Annual totals
        self.grid_cost = np.sum(import_costs)
        self.grid_revenue = np.sum(export_revenues)
    
    @property
    def net_grid_cost(self):
        """Net grid cost (cost - revenue)"""
        return self.grid_cost - self.grid_revenue
    
    @property
    def total_import_energy(self):
        """Total energy imported from grid (kWh)"""
        return np.sum(self.power_imported)
    
    @property
    def total_export_energy(self):
        """Total energy exported to grid (kWh)"""
        return np.sum(self.power_exported)
    
    @property
    def grid_dependency_ratio(self):
        """Ratio of imported energy to total energy demand"""
        if hasattr(self, 'py_microgrid') and self.py_microgrid:
            total_demand = np.sum(self.py_microgrid.site.desired_schedule) * 1000  # Convert MW to kW
            if total_demand > 0:
                return self.total_import_energy / total_demand
        return 0.0
    
    @property
    def grid_export_ratio(self):
        """Ratio of exported energy to total local generation"""
        if hasattr(self, 'py_microgrid') and self.py_microgrid:
            # Calculate total local generation
            total_local_gen = 0
            if hasattr(self.py_microgrid.system.generation_profile, 'pv'):
                total_local_gen += np.sum(self.py_microgrid.system.generation_profile.pv)
            if hasattr(self.py_microgrid.system.generation_profile, 'wind'):
                total_local_gen += np.sum(self.py_microgrid.system.generation_profile.wind)
            
            if total_local_gen > 0:
                return self.total_export_energy / total_local_gen
        return 0.0