# Py-Microgrid

A Python-based microgrid simulation and optimization framework for hybrid renewable energy systems, built on the HOPP (Hybrid Optimization and Performance Platform) framework.

## Overview

Py-Microgrid enables users to design, simulate, and optimize hybrid renewable energy systems with a flexible 5-component architecture:

1. **PV** - Solar photovoltaic panels
2. **Wind** - Wind turbines
3. **Battery** - Energy storage systems
4. **Genset** - Backup generators (diesel/natural gas)
5. **Grid** - Electrical grid connection

## Key Features

- **5-Component Architecture**: Clear separation between backup generators (genset) and grid connection
- **Nelder-Mead Optimization**: Advanced optimization algorithms using scipy.optimize for cost-effective system sizing
- **Hardcoded Economic Parameters**: Configurable discount rates (5.88%), project lifetime (25 years), and cost assumptions
- **Resource Integration**: Automatic download of solar and wind resource data via NREL and NASA APIs
- **Fixed Grid Initialization**: Resolved critical grid initialization bugs for stable simulation runs
- **Economic Analysis**: Comprehensive economic evaluation including LCOE, NPC, CO2 emissions, and lifecycle costs
- **Flexible Load Management**: Up to 20% demand reduction capabilities
- **Multi-Location Processing**: Batch analysis for multiple sites with parallel simulation support

## Installation

```bash
# Clone the repository
git clone https://github.com/Hanrong-Huang/Py-Microgrid.git
cd Py-Microgrid

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Required Dependencies

1. **Python Version**: Python 3.10 or 3.11 required

2. **HOPP (Hybrid Optimization and Performance Platform)**
   ```bash
   git clone https://github.com/NREL/HOPP.git
   cd HOPP
   pip install -e .
   ```
   Or install directly via pip:
   ```bash
   pip install HOPP
   ```

3. **Required Conda-forge Packages**
   ```bash
   conda install -c conda-forge glpk -y
   conda install -c conda-forge coin-or-cbc -y
   ```

## Quick Start

### 1. Set Up API Key

Get your free NREL API key from [developer.nrel.gov](https://developer.nrel.gov/) and set it up:

```python
from py_microgrid.utilities.keys import set_developer_nrel_gov_key
set_developer_nrel_gov_key('YOUR-API-KEY')
```

### 2. Basic Usage

```python
# Import fixed optimizer that preserves original structure but fixes grid bug
from py_microgrid.tools.optimization.fixed_system_optimizer import SystemOptimizer
from py_microgrid.tools.analysis.bos import EconomicCalculator

# Initialize components with hardcoded economic parameters
economic_calculator = EconomicCalculator(
    discount_rate=0.0588,    # 5.88% discount rate (hardcoded)
    project_lifetime=25      # 25 year project lifetime (hardcoded)
)

optimizer = SystemOptimizer(
    yaml_file_path='py_microgrid/examples/quick_start_config.yaml',
    economic_calculator=economic_calculator,
    enable_flexible_load=True,      # Enable 20% load reduction
    max_load_reduction_percentage=0.2
)

# Define optimization bounds for 5 components (hardcoded)
bounds = [
    (5000, 50000),    # PV capacity (kW)
    (1, 50),          # Wind turbines (1MW each)
    (5000, 30000),    # Battery energy capacity (kWh)
    (1000, 10000),    # Battery power capacity (kW)
    (17000, 30000)    # Genset capacity (kW) - backup generator
]

# Define initial conditions (10% of range)
initial_conditions = [
    [bound[0] + (bound[1] - bound[0]) * 0.1 for bound in bounds]
]

# Run Nelder-Mead optimization
result = optimizer.optimize_system(bounds, initial_conditions)

# Print results
if result:
    print(f"PV Capacity: {result['PV Capacity (kW)']:.2f} kW")
    print(f"LCOE: ${result['System LCOE ($/kWh)']:.4f}/kWh")
    print(f"System Cost: ${result['System NPC ($)']:,.2f}")
```

### 3. Configuration Structure

**IMPORTANT**: Only the `grid` component supports the `enabled` parameter. Other components (PV, Wind, Battery, Genset) are controlled through the optimization bounds.

```yaml
technologies:
  pv:
    system_capacity_kw: 9500.0
    
  wind:
    num_turbines: 6
    turbine_rating_kw: 1000
    
  battery:
    chemistry: LFPGraphite
    system_capacity_kw: 1900.0
    system_capacity_kwh: 8000.0
    
  genset:
    interconnect_kw: 18300.0
    
  grid:
    enabled: true  # Only grid supports enabled/disabled
    interconnect_kw: 10000
```

### 4. Working Notebooks

Both example notebooks have been fixed and are ready to use:

- **Quick Start**: `py_microgrid/quick_start_example.ipynb`
- **Parallel Simulation**: `py_microgrid/examples/parallel_simulations/py_microgrid_example/simulation_chunk_0.ipynb`

## Examples

### Quick Start Example

The fastest way to get started is with the simplified quick start example:

```python
# Run the quick start notebook
jupyter notebook py_microgrid/quick_start_example.ipynb
```

This example uses:
- **Configuration**: `examples/quick_start_config.yaml` - Simplified 5-component configuration
- **Location**: Atlanta, GA (33.7490, -84.3880)
- **System**: Off-grid demonstration with all components except grid connection

### Complete Example

See `py_microgrid/examples/parallel_simulations/py_microgrid_example/simulation_chunk_0.ipynb` for a complete working example including:
- Resource data download
- System configuration
- Optimization execution
- Results analysis

### Configuration Examples

- **Quick Start**: `examples/quick_start_config.yaml` - Simplified configuration for learning
- **Basic Configuration**: `examples/parallel_simulations/input_yaml/input_file_chunk_0.yaml`
- **Minimal Configuration**: `examples/parallel_simulations/input_yaml/input_file_minimal_example.yaml`

## Bug Fixes and Improvements

### Critical Fixes Applied

1. **Grid Initialization Bug**: Fixed 'NoneType' object errors by creating `FixedSystemOptimizer`
2. **YAML Configuration**: Corrected 'enabled' parameter usage (only for Grid component)
3. **API Key Timing**: Resolved NREL API key initialization issues
4. **Unicode Handling**: Fixed character encoding issues in console output
5. **Resource File Paths**: Normalized file path separators for cross-platform compatibility

### Enhanced Features

- **Preserved Original Structure**: All original algorithms and notebook structures maintained
- **Nelder-Mead Optimization**: Original scipy.optimize implementation preserved
- **Hardcoded Economics**: Original economic parameters and calculations maintained
- **Working Notebooks**: Both example notebooks now run without errors

## Configuration Parameters

The system uses externalized configuration files located in `py_microgrid/simulation/config/`:

- `battery_config.yaml` - Battery efficiency, costs, and operational parameters
- `pv_config.yaml` - PV efficiency, costs, and performance parameters
- `wind_config.yaml` - Wind performance, costs, and operational parameters
- `genset_config.yaml` - Generator costs, performance, and environmental parameters
- `grid_config.yaml` - Grid pricing, dispatch factors, and connection parameters
- `dispatch_config.yaml` - Optimization and dispatch control parameters

## Architecture

### Component Overview

| Component | Description | Purpose |
|-----------|-------------|---------|
| **PV** | Solar photovoltaic panels | Clean energy generation |
| **Wind** | Wind turbines | Clean energy generation |
| **Battery** | Energy storage system | Energy storage and dispatch |
| **Genset** | Backup generator | Backup power generation |
| **Grid** | Electrical grid connection | Import/export power |

### Key Improvements

- **Clear Separation**: Genset (backup generator) vs Grid (electrical connection)
- **Flexible Control**: Individual component enable/disable flags
- **Configurable Parameters**: Easy customization without code changes
- **Simplified Input**: Load schedules can be loaded from CSV files
- **Backwards Compatible**: Existing configurations continue to work

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the Apache License 2.0. See LICENSE file for details.

## Support

For questions, issues, or contributions, please visit the [GitHub repository](https://github.com/Hanrong-Huang/Py-Microgrid).

## Acknowledgments

- NREL for providing renewable energy resource data APIs
- The open-source community for foundational tools and libraries
- Contributors to the hybrid energy system modeling domain

