# Py-Microgrid

A Python-based microgrid simulation and optimization framework for hybrid renewable energy systems.

## Overview

Py-Microgrid enables users to design, simulate, and optimize hybrid renewable energy systems with a flexible 5-component architecture:

1. **PV** - Solar photovoltaic panels
2. **Wind** - Wind turbines
3. **Battery** - Energy storage systems
4. **Genset** - Backup generators
5. **Grid** - Electrical grid connection

## Key Features

- **5-Component Architecture**: Clear separation between backup generators (genset) and grid connection
- **Flexible Configuration**: Enable/disable individual components as needed
- **Optimization**: Built-in optimization algorithms for cost-effective system sizing
- **Resource Integration**: Automatic download of solar and wind resource data via NREL APIs
- **Economic Analysis**: Comprehensive economic evaluation including LCOE, NPV, and lifecycle costs
- **Configurable Parameters**: Externalized efficiency and cost parameters for easy customization
- **Multi-Location Processing**: Batch analysis for multiple sites simultaneously

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
from py_microgrid.tools.optimization import SystemOptimizer
from py_microgrid.tools.analysis.bos import EconomicCalculator

# Initialize components
economic_calculator = EconomicCalculator(discount_rate=0.0588, project_lifetime=25)
optimizer = SystemOptimizer(
    yaml_file_path='examples/parallel_simulations/input_yaml/input_file_chunk_0.yaml',
    economic_calculator=economic_calculator
)

# Define optimization bounds for 5 components
bounds = [
    (5000, 50000),    # PV capacity (kW)
    (1, 50),          # Wind turbines (1MW each)
    (5000, 30000),    # Battery energy capacity (kWh)
    (1000, 10000),    # Battery power capacity (kW)
    (17000, 30000),   # Genset capacity (kW)
    (5000, 20000),    # Grid interconnect capacity (kW)
]

# Run optimization
result = optimizer.optimize_system(bounds, initial_conditions)
```

### 3. Configuration Structure

```yaml
technologies:
  pv:
    enabled: true
    system_capacity_kw: 25000
    
  wind:
    enabled: true
    num_turbines: 8
    turbine_rating_kw: 1000
    
  battery:
    enabled: true
    system_capacity_kw: 3000
    system_capacity_kwh: 12000
    
  genset:
    enabled: true
    interconnect_kw: 20000
    
  grid:
    enabled: true  # Set to false for off-grid systems
    interconnect_kw: 15000
```

## Examples

### Complete Example

See `py_microgrid/examples/parallel_simulations/py_microgrid_example/simulation_chunk_0.ipynb` for a complete working example including:
- Resource data download
- System configuration
- Optimization execution
- Results analysis

### Configuration Examples

- **Basic Configuration**: `examples/parallel_simulations/input_yaml/input_file_chunk_0.yaml`
- **Minimal Configuration**: `examples/parallel_simulations/input_yaml/input_file_minimal_example.yaml`
- **Test Configuration**: `examples/parallel_simulations/input_yaml/input_file_chunk_0_test.yaml`

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

