# Py-Microgrid v0.2.0 Configuration Files Guide

This directory contains configuration files for the Py-Microgrid hybrid energy system optimization tool. These files demonstrate different system architectures and have been updated for v0.2.0 with YAML-based cost configuration and economic dispatch.

## 📁 File Overview

### **Core Configuration Files**
- **`multiple_locations_config.yaml`** - Clean, minimal configuration template for multiple locations optimization
- **`config_parameters_dictionary.yaml`** - Comprehensive parameter reference

### **System Architecture Examples**
- **`config_pv_battery_genset.yaml`** - Off-grid solar with backup generator
- **`config_wind_grid.yaml`** - Grid-tied wind farm with economic dispatch
- **`config_renewable_only.yaml`** - 100% renewable off-grid (PV+Wind+Battery)
- **`config_pv_grid.yaml`** - Simple grid-tied solar with time-of-use pricing
- **`config_storage_backup.yaml`** - Battery + generator with enhanced dispatch logic

## 🆕 What's New in v0.2.0

### **YAML-Based Cost Configuration**
All cost parameters are now externalized to configuration files in `py_microgrid/simulation/config/`:
- **`pv_config.yaml`** - PV costs, installation, O&M parameters
- **`wind_config.yaml`** - Wind turbine costs and specifications
- **`battery_config.yaml`** - Battery costs, replacement schedules, efficiency
- **`genset_config.yaml`** - Generator costs, fuel prices, emissions factors
- **`grid_config.yaml`** - Grid pricing, time-of-use factors, connection costs

### **Economic Dispatch Engine**
- Real-time cost comparison between genset and grid
- Time-of-use grid pricing (off-peak, standard, peak rates)
- Grid typically preferred due to lower cost ($0.084-0.156/kWh vs $0.30/kWh genset)

### **Enhanced Dispatch Logic**
- **Priority Order**: Renewables → Battery → Economic dispatch (Genset/Grid) → Emergency backup
- **Industrial Reliability**: Realistic targets (95-99%) instead of academic 99.9%+
- **Genset Improvements**: Proper minimum load constraints and reliability-first operation

## 🚀 Quick Start Guide

### **For New Users:**
1. **Start with**: `multiple_locations_config.yaml` (clean template)
2. **Modify**: Location (lat/lon), load file path, component capacities
3. **Customize costs**: Edit files in `py_microgrid/simulation/config/` as needed
4. **Reference**: `config_parameters_dictionary.yaml` for additional options

### **For Specific Applications:**
1. **Choose** the architecture file that matches your use case
2. **Copy** it to a new file (e.g., `my_system_config.yaml`)
3. **Customize** parameters and review cost configuration files
4. **Set optimization bounds** to match your enabled components

## 🔧 System Architectures

### **Off-Grid Systems**
- **`config_pv_battery_genset.yaml`** - Solar + storage + backup (enhanced genset logic)
- **`config_renewable_only.yaml`** - 100% renewable (improved battery management)
- **`config_storage_backup.yaml`** - Storage + generator with dispatch optimization

### **Grid-Connected Systems**  
- **`config_pv_grid.yaml`** - Solar + grid (time-of-use pricing)
- **`config_wind_grid.yaml`** - Wind + grid (economic dispatch)

## 📋 Cost Configuration Guide - v0.2.0

### **Primary Cost Parameters** (externalized to YAML files):

**PV Costs** (`py_microgrid/simulation/config/pv_config.yaml`):
```yaml
pv:
  costs:
    installed_cost_per_kw: 1350.0    # $/kW - Installation cost
    om_cost_per_kw_per_year: 15.0    # $/kW/year - Annual O&M
```

**Grid Pricing** (`py_microgrid/simulation/config/grid_config.yaml`):
```yaml
grid:
  pricing:
    base_import_price: 0.12          # $/kWh - Base rate
  dispatch_factors:
    off_peak_factor: 0.7             # 0-6 AM: $0.084/kWh
    peak_factor: 1.3                 # 6-10 PM: $0.156/kWh
```

**Genset Economics** (`py_microgrid/simulation/config/genset_config.yaml`):
```yaml
genset:
  costs:
    fuel_cost_per_liter: 1.20        # $/L
  performance:
    specific_fuel_consumption_l_per_kwh: 0.25  # L/kWh
  # Results in ~$0.30/kWh marginal cost
```

### **Component Enable/Disable Rules**

**Grid Control** (only component with enable/disable):
```yaml
grid:
  enabled: true   # Grid-connected with economic dispatch
  # vs
  enabled: false  # Off-grid system
```

**Other Components** (controlled by capacity and optimization bounds):
```yaml
# To disable a component, set capacity to 0 and optimization bounds to (0, 0):
pv:
  system_capacity_kw: 0     # Disables PV

wind:
  num_turbines: 0           # Disables wind

battery:
  system_capacity_kwh: 0    # Disables battery
  system_capacity_kw: 0
  
genset:
  interconnect_kw: 0        # Disables genset
```

## 🔄 Optimization Bounds - v0.2.0

When using `SystemOptimizer`, set bounds to match your enabled components:

```python
# Example: PV + Battery + Genset with Grid (5 components)
bounds = [
    (5000, 50000),    # PV capacity (kW) - ACTIVE
    (5, 25),          # Wind turbines - ACTIVE (1MW each)
    (5000, 30000),    # Battery energy (kWh) - ACTIVE 
    (1000, 10000),    # Battery power (kW) - ACTIVE
    (5000, 20000)     # Genset capacity (kW) - ACTIVE
]

# For grid-enabled systems, add:
# (5000, 25000)     # Grid capacity (kW) - ACTIVE
```

## 📖 v0.2.0 Economic Dispatch Features

### **Real-Time Cost Comparison**
The system now compares costs every hour:
- **Off-peak (0-6 AM)**: Grid $0.084/kWh vs Genset $0.30/kWh → Grid preferred
- **Standard hours**: Grid $0.12/kWh vs Genset $0.30/kWh → Grid preferred  
- **Peak (6-10 PM)**: Grid $0.156/kWh vs Genset $0.30/kWh → Grid still preferred
- **Grid outage**: Genset automatically takes over

### **Industrial Reliability Targets**
- **≥99%**: Excellent reliability - No penalty
- **95-99%**: Acceptable reliability - Moderate penalties
- **<95%**: Poor reliability - Significant penalties based on value of lost load

### **Enhanced Genset Logic**
- Minimum turn-on power reduced to 20% (from 30%) for better flexibility
- Reliability-first dispatch when battery SOC is critically low
- Emergency dispatch uses full genset capacity for unmet demand

## 🏗️ System Sizing Guidelines

### **Residential Scale**
- PV: 5-50 kW
- Wind: 1-5 turbines × 1MW
- Battery: 10-100 kWh  
- Genset: 5-50 kW

### **Commercial Scale**
- PV: 50-1000 kW
- Wind: 1-10 turbines × 1-2MW
- Battery: 100-5000 kWh
- Genset: 50-1000 kW

### **Utility Scale**
- PV: 1-100 MW
- Wind: 10-100 turbines × 2-3MW
- Battery: 5-100 MWh
- Genset: 1-50 MW

## ⚠️ Important Notes - v0.2.0

1. **Cost Configuration**: All costs now in `py_microgrid/simulation/config/` YAML files
2. **Economic Dispatch**: Grid typically preferred due to lower costs
3. **Industrial Reliability**: Targets are 95-99%, not academic 99.9%+
4. **Grid Enable/Disable**: Only Grid supports `enabled: true/false` parameter
5. **Optimization Bounds**: Must match enabled components for proper sizing
6. **Battery Management**: Enhanced SOC thresholds and dispatch logic

## 🔍 Troubleshooting - v0.2.0

### **Common Issues:**
- **High LCOE results** → Check cost parameters in config YAML files
- **Poor reliability** → Verify genset and battery sizing bounds
- **Grid not used** → Check grid pricing in `grid_config.yaml`
- **Genset over-running** → Review economic dispatch parameters

### **Configuration Checks:**
- Verify cost parameters in `py_microgrid/simulation/config/` files
- Ensure optimization bounds match enabled components  
- Check grid pricing for economic dispatch
- Review reliability penalty function settings

### **Getting Help:**
- Check cost configuration files for parameter options
- Review architecture examples for similar use cases
- Verify economic dispatch settings for grid vs genset
- Ensure load data and resource files are accessible

---

*For more information about v0.2.0 features, see the main Py-Microgrid documentation and updated example notebooks.*