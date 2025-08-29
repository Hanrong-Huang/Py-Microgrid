# Py-Microgrid v0.2.0

A Python-based microgrid simulation and optimization framework for hybrid renewable energy systems, built on the HOPP (Hybrid Optimization and Performance Platform) framework.

## Overview

Py-Microgrid enables users to design, simulate, and optimize hybrid renewable energy systems with a flexible 5-component architecture and advanced economic dispatch capabilities:

1. **PV** - Solar photovoltaic panels
2. **Wind** - Wind turbines  
3. **Battery** - Energy storage systems
4. **Genset** - Backup generators (diesel/natural gas)
5. **Grid** - Electrical grid connection with time-of-use pricing

## Key Features - v0.2.0

### 🆕 New in v0.2.0
- **Economic Dispatch**: Real-time cost-based dispatch between genset and grid
- **YAML-Based Configuration**: All costs and parameters externalized to config files
- **Industrial Penalty Functions**: Realistic reliability targets (95-99%) with value-of-lost-load economics
- **Time-of-Use Grid Pricing**: Dynamic pricing with off-peak, standard, and peak rates
- **Enhanced Genset Logic**: Improved minimum load constraints and reliability-first dispatch
- **Proper Priority Dispatch**: Renewables → Battery → Economic dispatch (Genset/Grid) → Emergency backup

### 🔧 Core Features
- **5-Component Architecture**: Clear separation between backup generators (genset) and grid connection
- **Nelder-Mead Optimization**: Advanced optimization algorithms using scipy.optimize
- **YAML Configuration System**: Externalized parameters in `py_microgrid/simulation/config/`
- **Resource Integration**: Automatic download of solar and wind resource data via NREL APIs
- **Economic Analysis**: LCOE, NPC, CO2 emissions, and lifecycle cost calculations
- **Flexible Load Management**: Configurable demand response capabilities
- **Multi-Location Processing**: Batch analysis for multiple sites

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
from py_microgrid.tools.optimization.system_optimizer import SystemOptimizer
from py_microgrid.tools.optimization.economic_calculator import EconomicCalculator

# Initialize with configurable economic parameters
economic_calculator = EconomicCalculator(
    discount_rate=0.0588,    # 5.88% discount rate
    project_lifetime=25      # 25 year project lifetime
)

optimizer = SystemOptimizer(
    yaml_file_path='py_microgrid/quick_start_config.yaml',
    economic_calculator=economic_calculator,
    enable_flexible_load=True,
    max_load_reduction_percentage=0.2
)

# Define optimization bounds for 5 components
bounds = [
    (5000, 50000),    # PV capacity (kW)
    (5, 50),          # Wind turbines (1MW each)
    (5000, 30000),    # Battery capacity (kWh)
    (1000, 10000),    # Battery power (kW)
    (5000, 20000)     # Genset capacity (kW)
]

# Run optimization
result = optimizer.optimize_system(bounds, initial_conditions)

# Results include economic dispatch analysis
print(f"PV Capacity: {result['PV Capacity (kW)']:.2f} kW")
print(f"LCOE: ${result['System LCOE ($/kWh)']:.4f}/kWh")
print(f"Demand Met: {result['Demand Met Percentage']:.1f}%")
```

### 3. Configuration Structure

**Grid Component** (only component with enable/disable):
```yaml
technologies:
  grid:
    enabled: true  # Enable grid for economic dispatch
    interconnect_kw: 10000
```

**Other Components** (controlled through optimization bounds):
```yaml
technologies:
  pv:
    system_capacity_kw: 25000.0
    
  wind:
    num_turbines: 8
    turbine_rating_kw: 1000
    
  battery:
    system_capacity_kw: 5000.0
    system_capacity_kwh: 15000.0
    
  genset:
    interconnect_kw: 10000.0
```

## Configuration System - v0.2.0

All costs and operational parameters are now externalized to YAML files in `py_microgrid/simulation/config/`:

### Cost Configuration Files
- **`pv_config.yaml`** - PV installation costs, O&M costs, performance parameters
- **`wind_config.yaml`** - Wind turbine costs, O&M costs, turbine specifications  
- **`battery_config.yaml`** - Battery costs, replacement schedules, efficiency parameters
- **`genset_config.yaml`** - Generator costs, fuel prices, emissions factors, O&M costs
- **`grid_config.yaml`** - Grid pricing, time-of-use factors, connection costs
- **`dispatch_config.yaml`** - Optimization parameters, dispatch priorities

### Economic Dispatch Configuration

**Grid Pricing** (`grid_config.yaml`):
```yaml
grid:
  pricing:
    base_import_price: 0.12  # $/kWh base rate
    
  dispatch_factors:
    off_peak_factor: 0.7     # 0-6 AM: $0.084/kWh
    standard_factor: 1.0     # Standard: $0.12/kWh  
    peak_factor: 1.3         # 6-10 PM: $0.156/kWh
```

**Genset Economics** (`genset_config.yaml`):
```yaml
genset:
  costs:
    fuel_cost_per_liter: 1.20           # $/L
  performance:
    specific_fuel_consumption_l_per_kwh: 0.25  # L/kWh
  # Results in ~$0.30/kWh marginal cost
```

## Examples

### Quick Start Example
```python
# Run the quick start notebook
jupyter notebook py_microgrid/quick_start_example.ipynb
```

**Features:**
- Download solar and wind data based on location input (lat, lon)  
- multiple components optimization with predictive dispatch
- Real-time genset vs grid cost comparison
- Industrial reliability targets (95-99%)

### Multi-Location Example
```python
# Run parallel simulations
jupyter notebook py_microgrid/examples/parallel_simulations/py_microgrid_example/multiple_locations_example.ipynb
```

### Configuration Examples
- **`py_microgrid/quick_start_config.yaml`** - Simple off-grid system
- **`py_microgrid/examples/parallel_simulations/input_yaml/`** - Various system architectures

## Architecture - v0.2.0

### Enhanced Dispatch Logic

**Priority Order:**
1. **Renewables First** - PV and Wind generation used first
2. **Battery Dispatch** - Based on SOC thresholds and available capacity  
3. **Economic Dispatch** - Real-time cost comparison between genset and grid:
   - Off-peak hours: Grid preferred ($0.084/kWh vs $0.30/kWh genset)
   - Peak hours: Cost-dependent dispatch ($0.156/kWh grid vs $0.30/kWh genset)
4. **Emergency Backup** - Full genset capacity for unmet demand

### Component Specifications

| Component | v0.2.0 Enhancements | Configuration |
|-----------|-------------------|---------------|
| **PV** | YAML-based costs, performance parameters | `pv_config.yaml` |
| **Wind** | Turbine specifications, cost parameters | `wind_config.yaml` |
| **Battery** | SOC management, replacement schedules | `battery_config.yaml` |
| **Genset** | Economic dispatch, minimum load logic | `genset_config.yaml` |
| **Grid** | Time-of-use pricing, economic dispatch | `grid_config.yaml` |

### Reliability & Economics

**Industrial Reliability Targets:**
- **≥99%**: Excellent reliability - No penalty
- **95-99%**: Acceptable reliability - Moderate penalties
- **<95%**: Poor reliability - Significant penalties based on value of lost load

**Economic Dispatch:**
- Real-time cost comparison every hour
- Grid typically preferred due to lower cost
- Genset used during peak hours or grid outages
- Configurable pricing through YAML files

## Working Examples

All example notebooks have been updated for v0.2.0:

- **Quick Start**: `py_microgrid/quick_start_example.ipynb` ✅
- **Multi-Location**: `py_microgrid/examples/parallel_simulations/py_microgrid_example/multiple_locations_example.ipynb` ✅

## What's New in v0.2.0

### Major Enhancements
1. **Economic Dispatch Engine**: Real-time cost-based decisions between genset and grid
2. **YAML Configuration System**: All parameters externalized for easy customization
3. **Industrial Reliability Standards**: Realistic penalty functions based on industrial microgrids
4. **Enhanced Genset Logic**: Proper minimum load constraints and reliability-first operation
5. **Time-of-Use Grid Pricing**: Dynamic pricing with off-peak, standard, and peak rates

### Bug Fixes
1. **Genset Dispatch Priority**: Fixed to follow industrial best practices
2. **Battery SOC Management**: Improved state-of-charge thresholds and logic
3. **Grid Integration**: Proper economic comparison with genset costs
4. **Penalty Functions**: Realistic industrial reliability targets (not academic 99.9%+)

### Configuration Improvements
1. **Cost Parameters**: All costs moved to YAML files for easy modification
2. **Dispatch Logic**: Configurable dispatch priorities and thresholds
3. **Grid Pricing**: Comprehensive time-of-use pricing structure
4. **Component Parameters**: Externalized performance and operational parameters

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Update configuration files in `py_microgrid/simulation/config/` as needed
5. Test with example notebooks
6. Submit a pull request

## License

This project is licensed under the Apache License 2.0. See LICENSE file for details.

## Support

For questions, issues, or contributions, please visit the [GitHub repository](https://github.com/Hanrong-Huang/Py-Microgrid).

## Acknowledgments

- NREL for providing renewable energy resource data APIs and HOPP framework
- The industrial microgrid community for real-world operational insights  
  

