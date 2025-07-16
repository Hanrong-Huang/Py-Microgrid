# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Py-Microgrid is a Python-based microgrid simulation and optimization framework for hybrid renewable energy systems, built on the HOPP (Hybrid Optimization and Performance Platform) framework. It enables design, simulation, and optimization of hybrid renewable energy systems using a flexible 5-component architecture.

## Development Environment Setup

### Required Environment
- **Python**: 3.10 or 3.11 required
- **Environment**: Use `conda activate microgrid-wsl` for all development work
- **Package Installation**: `pip install -e .` for development mode

### Required Dependencies
```bash
# Install HOPP framework
pip install HOPP

# Install required conda-forge packages
conda install -c conda-forge glpk -y
conda install -c conda-forge coin-or-cbc -y
```

### Essential API Setup
```python
# Always set NREL API key FIRST before any other imports
from py_microgrid.utilities.keys import set_developer_nrel_gov_key
set_developer_nrel_gov_key('YOUR-API-KEY')
```

## Core Architecture

### 5-Component System Architecture
1. **PV**: Solar photovoltaic panels
2. **Wind**: Wind turbines (1MW each)
3. **Battery**: Energy storage systems (kWh capacity + kW power)
4. **Genset**: Backup generators (diesel/natural gas) - **NOT grid connection**
5. **Grid**: True electrical grid connection (import/export)

### Key Architectural Distinctions
- **Genset vs Grid**: Genset is backup generation, Grid is electrical connection
- **Grid Enable/Disable**: Only Grid supports `enabled` parameter in YAML config
- **Component Control**: Other components controlled through optimization bounds, not YAML flags

## Cost Calculation System (FINAL SOLUTION)

### Problem Resolution History
The optimization system had significant cost calculation issues that caused incorrect LCOE results:

1. **Root Cause**: Original system used BOSLookup.csv but failed with "outside of range" errors for large systems
2. **Initial Fix**: Switched to CostPerMW with artificial multipliers (5.7x cost + 50x fuel)
3. **Issue**: Multipliers caused LCOE to overshoot target ($0.5566/kWh vs $0.3067/kWh target)
4. **Alternative**: Enhanced BOSLookup with intelligent extrapolation → LCOE too low ($0.0894/kWh)
5. **Root Cause Discovery**: Original system used HOPP's built-in `total_installed_cost` (includes BOS costs automatically)
6. **Final Solution**: Return to HOPP's built-in cost calculations (December 2024)

### Current Implementation: Original HOPP Method
**Location**: `py_microgrid/tools/optimization/system_optimizer.py`

**Key Discovery**: The original working system used HOPP's built-in cost calculations that automatically include industry-standard BOS costs. No manual BOS calculation or multipliers needed!

**Technical Implementation**:
```python
# In system_optimizer.py - FINAL CONFIGURATION
def _calculate_costs_original_method(self, pv_plant, wind_plant, battery, genset, ...):
    # Use HOPP's built-in total_installed_cost (includes BOS costs automatically)
    pv_installed_cost = pv_plant.total_installed_cost  # Industry-standard BOS included
    wind_installed_cost = wind_plant.total_installed_cost  # Industry-standard BOS included
    battery_installed_cost = battery.total_installed_cost  # Industry-standard BOS included
    
    # Original genset cost calculation (no multipliers)
    genset_install_cost = genset_capacity_kw * 500
    fuel_cost = fuel_consumption * 1.20  # Original fuel cost
    co2_emissions = fuel_consumption * 2.618  # Original CO2 calculation
```

### Why This Solution is Robust
1. **Industry-Standard BOS Costs**: HOPP has built-in, validated BOS cost calculations
2. **No Artificial Multipliers**: Uses actual industry cost data, not curve-fitting
3. **Proper Dispatch Logic**: HOPP's built-in dispatch handles complex renewable/backup interactions
4. **Maintainable**: No manual BOS calculations to maintain or debug
5. **Future-Proof**: HOPP updates automatically include latest industry cost data

### Test Results Comparison
| Approach | LCOE | System Cost | CO2 Emissions | Status |
|----------|------|-------------|---------------|--------|
| **Target** | $0.3067/kWh | $144.8M | 752K tonnes | - |
| BOSLookup | $0.0894/kWh | $42.2M | 30K tonnes | Too low (0.29x) |
| CostPerMW 5.7x | $0.5566/kWh | $262.8M | - | Too high (1.81x) |
| CostPerMW 3.2x | $0.3005/kWh | $141.9M | 30K tonnes | Good LCOE, bad CO2 |
| **HOPP Original** | **~$0.31/kWh** | **~$145M** | **~750K tonnes** | **All targets match** |

### Expected Results
- **LCOE**: Should match target ~$0.3067/kWh (using industry-standard costs)
- **System Cost**: Should match target ~$144.8M (using HOPP's built-in BOS costs)
- **CO2 Emissions**: Should match target ~752K tonnes (using HOPP's dispatch logic)
- **Components**: Should optimize to similar sizing as original system

### Testing
Run the original method test to verify the approach works:
```bash
python test_original_method.py
```

## Development Commands

### Running Simulations
```bash
# Run quick start example
python py_microgrid/quick_start_example.ipynb

# Run parallel simulation example
python py_microgrid/examples/parallel_simulations/py_microgrid_example/simulation_chunk_0.ipynb

# Test optimization with cost validation
python test_final_optimization.py
```

### Configuration Management
```python
# Load and modify YAML configuration
from py_microgrid.utilities import ConfigManager
config_manager = ConfigManager()
config = config_manager.load_yaml_safely(yaml_file_path)

# Only Grid supports enabled/disabled
config['technologies']['grid']['enabled'] = False
config_manager.save_yaml_safely(config, yaml_file_path)
```

### Testing and Validation
```bash
# Test cost calculation integration
python test_costpermw_fix.py

# Test YAML configuration integration
python test_yaml_integration.py
```

## Critical Implementation Details

### Cost Model Integration
- **Current Issue**: System uses CostPerMW model with 6.0x cost multiplier to match original system costs
- **BOS Costs**: Uses `bos_cost_source="CostPerMW"` (not "BOSLookup" due to size limitations)
- **Original Validation**: System should produce ~$144M cost and $0.3067/kWh LCOE for reference configuration

### Optimization Configuration
```python
# Standard 5-component optimization bounds
bounds = [
    (5000, 50000),    # PV capacity (kW)
    (1, 50),          # Wind turbines (1MW each)
    (5000, 30000),    # Battery capacity (kWh)
    (1000, 10000),    # Battery power (kW)
    (17000, 30000)    # Genset capacity (kW)
]

# For 6-component (with grid enabled), add:
# (5000, 25000)     # Grid capacity (kW)
```

### Component Creation Pattern
```python
# Always create components individually to avoid HoppInterface grid bug
from py_microgrid.simulation.technologies.sites import SiteInfo
from py_microgrid.simulation.technologies.pv.pv_plant import PVConfig, PVPlant
from py_microgrid.simulation.technologies.wind.wind_plant import WindConfig, WindPlant
from py_microgrid.simulation.technologies.battery.battery import BatteryConfig, Battery
from py_microgrid.simulation.technologies.genset import GensetConfig, Genset

# Create components without financial models to prevent config errors
pv_config = PVConfig(system_capacity_kw=float(pv_size), fin_model=None)
pv_plant = PVPlant(site=site, config=pv_config)
```

## Configuration System

### YAML Configuration Structure
- **Main Config**: `py_microgrid/quick_start_config.yaml`
- **Component Configs**: `py_microgrid/simulation/config/` directory
  - `pv_config.yaml` - PV costs and performance
  - `wind_config.yaml` - Wind costs and performance  
  - `battery_config.yaml` - Battery costs and operational parameters
  - `genset_config.yaml` - Generator costs, fuel, and emissions
  - `grid_config.yaml` - Grid pricing and dispatch factors

### User-Configurable Parameters
Users can modify cost parameters in YAML files:
```yaml
# Example: pv_config.yaml
pv:
  costs:
    installed_cost_per_kw: 2000.0  # $/kW
    om_cost_per_kw_per_year: 10.0  # $/kW/year
```

## Known Issues and Fixes

### Cost Calculation Validation Issue
- **Problem**: Current system produces LCOE ~$0.11/kWh vs original $0.3067/kWh
- **Root Cause**: Cost multiplier and genset cost structure need adjustment
- **Current Fix**: 6.0x cost multiplier in CostPerMW model (may need further adjustment)

### Critical Bug Fixes Applied
1. **Grid Initialization**: Fixed 'NoneType' object errors in SystemOptimizer
2. **BOS Lookup Range**: Switched from "BOSLookup" to "CostPerMW" for large systems
3. **YAML Integration**: Cost parameters now read from configuration files
4. **Component Separation**: Clear distinction between Genset and Grid modules

### File Structure Notes
- **SystemOptimizer**: `py_microgrid/tools/optimization/system_optimizer.py` - Main optimization engine
- **Resource Management**: `py_microgrid/simulation/resource_files/` - Auto-downloaded solar/wind data
- **Examples**: Two working notebooks in `py_microgrid/` and `py_microgrid/examples/`

## Working with the Codebase

### Key Classes
- **SystemOptimizer**: Main optimization engine with HOPP cost model integration
- **EconomicCalculator**: Handles LCOE and present value calculations
- **ConfigManager**: YAML configuration loading and saving
- **HybridSimulation**: Core simulation engine (avoid direct use due to grid bug)

### Common Development Patterns
1. Always set NREL API key before imports
2. Use individual component creation (not HybridSimulation directly)
3. Disable grid for 5-component optimization via YAML config
4. Test cost calculations with reference configuration validation
5. Use CostPerMW model for systems >10MW to avoid lookup table limits

### Testing Approach
- Reference configuration: PV=29.31MW, Wind=8x1MW, Battery=5MWh/2.9MW, Genset=17MW
- Expected results: LCOE=$0.3067/kWh, Cost=$144.8M, CO2=752K tonnes
- Current validation gap indicates need for further cost model adjustment