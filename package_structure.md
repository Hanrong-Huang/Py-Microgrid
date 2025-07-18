# Py-Microgrid Package Structure

```
py_microgrid/                          # Main Python package
├── README.md                          # Project documentation
├── LICENSE                            # Apache 2.0 license
├── setup.py                           # Package installation configuration
├── quick_start_config.yaml            # Main configuration file
├── quick_start_example.ipynb          # Quick start tutorial notebook
│
├── py_microgrid/                      # Core package modules
│   ├── __init__.py                    # Package initialization
│   ├── version.py                     # Version information
│   ├── type_dec.py                    # Type declarations
│   ├── logging_manager.py             # System logging
│   │
│   ├── simulation/                    # Core simulation engine
│   │   ├── base.py                    # Base simulation classes
│   │   ├── hopp.py                    # HOPP framework integration
│   │   ├── hopp_interface.py          # HOPP interface wrapper
│   │   ├── hybrid_simulation.py       # Main simulation controller
│   │   │
│   │   ├── config/                    # v0.2.0 YAML Configuration System
│   │   │   ├── config_manager.py      # Configuration manager
│   │   │   ├── pv_config.yaml         # PV costs & parameters
│   │   │   ├── wind_config.yaml       # Wind costs & parameters
│   │   │   ├── battery_config.yaml    # Battery costs & parameters
│   │   │   ├── genset_config.yaml     # Genset costs & parameters
│   │   │   ├── grid_config.yaml       # Grid pricing & dispatch
│   │   │   └── dispatch_config.yaml   # Economic dispatch settings
│   │   │
│   │   ├── resource_files/            # Resource data management
│   │   │   ├── resource_data_manager.py
│   │   │   ├── site_details.csv       # Site location database
│   │   │   ├── solar/                 # NREL solar resource data
│   │   │   ├── wind/                  # NREL wind resource data
│   │   │   ├── grid/                  # Grid pricing data
│   │   │   └── wave/                  # Wave resource data
│   │   │
│   │   └── technologies/              # Component implementations
│   │       ├── __init__.py
│   │       ├── power_source.py        # Base power source class
│   │       ├── genset.py              # Backup generator
│   │       ├── grid.py                # Grid connection
│   │       │
│   │       ├── pv/                    # Solar PV systems
│   │       │   ├── pv_plant.py        # PV plant implementation
│   │       │   └── detailed_pv_plant.py
│   │       │
│   │       ├── wind/                  # Wind turbine systems
│   │       │   ├── wind_plant.py      # Wind plant implementation
│   │       │   └── floris.py          # Wind farm modeling
│   │       │
│   │       ├── battery/               # Energy storage systems
│   │       │   ├── battery.py         # Battery implementation
│   │       │   └── battery_stateless.py
│   │       │
│   │       ├── sites/                 # Site definitions
│   │       │   ├── site_info.py       # Site information
│   │       │   └── locations.py       # Location utilities
│   │       │
│   │       ├── dispatch/              # v0.2.0 Enhanced Dispatch System
│   │       │   ├── dispatch.py        # Main dispatch logic
│   │       │   ├── hybrid_dispatch.py # Hybrid system dispatch
│   │       │   ├── genset_dispatch.py # Genset economic dispatch
│   │       │   └── power_sources/     # Component-specific dispatch
│   │       │       ├── pv_dispatch.py
│   │       │       ├── wind_dispatch.py
│   │       │       └── ...
│   │       │
│   │       ├── financial/             # Financial modeling
│   │       │   └── custom_financial_model.py
│   │       │
│   │       ├── layout/                # System layout optimization
│   │       │   ├── hybrid_layout.py
│   │       │   ├── pv_layout.py
│   │       │   └── wind_layout.py
│   │       │
│   │       ├── resource/              # Resource management
│   │       │   ├── solar_resource.py
│   │       │   ├── wind_resource.py
│   │       │   └── wave_resource.py
│   │       │
│   │       ├── csp/                   # Concentrated Solar Power
│   │       ├── wave/                  # Wave energy systems
│   │       └── hydrogen/              # Hydrogen production & storage
│   │
│   ├── tools/                         # Analysis and optimization tools
│   │   ├── optimization/              # v0.2.0 Enhanced Optimization
│   │   │   ├── system_optimizer.py    # Main system optimizer with economic dispatch
│   │   │   ├── economic_calculator.py # LCOE and economic analysis
│   │   │   ├── load_analyzer.py       # Load demand analysis
│   │   │   ├── optimization_driver.py # Optimization coordination
│   │   │   ├── optimization_problem.py
│   │   │   │
│   │   │   ├── optimizer/             # Optimization algorithms
│   │   │   │   ├── ask_tell_optimizer.py
│   │   │   │   ├── CMA_ES_optimizer.py
│   │   │   │   ├── GA_optimizer.py
│   │   │   │   └── ...
│   │   │   │
│   │   │   ├── candidate_converter/   # Solution format converters
│   │   │   ├── driver/                # Optimization drivers
│   │   │   └── data_logging/          # Results logging
│   │   │
│   │   ├── analysis/                  # System analysis tools
│   │   │   ├── determine_curtailment.py
│   │   │   └── bos_legacy/            # Legacy BOS cost models
│   │   │
│   │   ├── resource/                  # Resource data tools
│   │   │   ├── download_resource.py   # NREL API integration
│   │   │   └── resource_loader/       # Resource file management
│   │   │
│   │   ├── dispatch/                  # Dispatch visualization
│   │   └── layout/                    # Layout visualization
│   │
│   ├── utilities/                     # Utility modules
│   │   ├── config_manager.py          # YAML configuration utilities
│   │   ├── keys.py                    # API key management
│   │   ├── log.py                     # Logging configuration
│   │   ├── utilities.py               # General utilities
│   │   └── validators.py              # Input validation
│   │
│   └── examples/                      # Usage examples
│       └── parallel_simulations/      # Multi-location examples
│           ├── input_yaml/            # Configuration examples
│           │   ├── config_pv_battery_genset.yaml
│           │   ├── config_pv_grid.yaml
│           │   ├── config_renewable_only.yaml
│           │   └── multiple_locations_config.yaml
│           │
│           ├── py_microgrid_example/
│           │   └── multiple_locations_example.ipynb
│           │
│           ├── load_data/             # Load profile data
│           └── deposit_data/          # Mining site data
│
├── log/                               # Application logs
```

## Key Components - v0.2.0

### 🎯 Core Architecture (5-Component System)
- **PV**: Solar photovoltaic panels with YAML-configurable costs
- **Wind**: Wind turbines (1MW each) with performance parameters
- **Battery**: Energy storage with SOC management and replacement schedules
- **Genset**: Backup generators with economic dispatch logic
- **Grid**: Electrical grid connection with time-of-use pricing

### 🆕 v0.2.0 Enhancements
- **Economic Dispatch Engine**: Real-time cost comparison between genset and grid
- **YAML Configuration System**: All parameters externalized to `simulation/config/`
- **Industrial Reliability Standards**: Realistic penalty functions (95-99% targets)
- **Enhanced Dispatch Logic**: Proper priority order with economic optimization

### 🔧 Main Entry Points
- **`system_optimizer.py`**: Main optimization engine with economic dispatch
- **`quick_start_example.ipynb`**: Tutorial notebook for new users
- **`multiple_locations_example.ipynb`**: Multi-site analysis example
- **`config/`**: YAML configuration files for all cost parameters
