# Py-Microgrid Configuration Files Guide

This directory contains configuration files for the Py-Microgrid hybrid energy system optimization tool. These files demonstrate how to set up different system architectures and customize parameters.

## 📁 File Overview

### **Core Configuration Files**
- **`input_file_chunk_0.yaml`** - Clean, minimal configuration template
- **`input_file_parameters_dictionary.yaml`** - Comprehensive parameter reference
- **`input_file_minimal_example.yaml`** - Basic example for learning

### **System Architecture Examples**
- **`config_pv_battery_genset.yaml`** - Off-grid solar with backup generator
- **`config_wind_grid.yaml`** - Grid-tied wind farm
- **`config_renewable_only.yaml`** - 100% renewable off-grid (PV+Wind+Battery)
- **`config_pv_grid.yaml`** - Simple grid-tied solar
- **`config_storage_backup.yaml`** - Battery + generator only (no renewables)

## 🚀 Quick Start Guide

### **For New Users:**
1. **Start with**: `input_file_chunk_0.yaml` (clean template)
2. **Modify**: Location (lat/lon), load file path, component capacities
3. **Reference**: `input_file_parameters_dictionary.yaml` for additional options

### **For Specific Applications:**
1. **Choose** the architecture file that matches your use case
2. **Copy** it to a new file (e.g., `my_system_config.yaml`)
3. **Customize** the parameters for your specific requirements

## 🔧 System Architectures

### **Off-Grid Systems**
- **`config_pv_battery_genset.yaml`** - Solar + storage + backup
- **`config_renewable_only.yaml`** - 100% renewable (no fossil backup)
- **`config_storage_backup.yaml`** - Storage + generator only

### **Grid-Connected Systems**
- **`config_pv_grid.yaml`** - Simple solar + grid
- **`config_wind_grid.yaml`** - Wind + grid

## 📋 Parameter Control Guide

### **Primary Control Parameters** (what users typically modify):
1. **Component Sizing** (main optimization variables):
   - `technologies.pv.system_capacity_kw` - PV array size
   - `technologies.wind.num_turbines` - Number of wind turbines  
   - `technologies.battery.system_capacity_kwh` - Battery energy
   - `technologies.battery.system_capacity_kw` - Battery power
   - `technologies.genset.interconnect_kw` - Generator capacity

2. **System Architecture**:
   - `technologies.grid.enabled` - Grid connection on/off (**ONLY GRID supports enabled/disabled**)
   - Set other components to 0 to disable them

3. **Site Configuration**:
   - `site.data.lat/lon` - Location coordinates
   - `site.desired_schedule` - Load profile (CSV file path)

### **Secondary Parameters** (advanced customization):
- Performance parameters (efficiency, cut-in speeds, etc.)
- Cost parameters (installation costs, O&M costs, fuel costs)
- Operational parameters (minimum loads, operational limits)
- Dispatch parameters (how system operates hour-by-hour)

## 🎯 Component Enable/Disable Rules

### **Grid Control** (only component with enable/disable):
```yaml
grid:
  enabled: true   # Grid-connected system
  # vs
  enabled: false  # Off-grid system
```

### **Other Components** (controlled by capacity):
```yaml
# To disable a component, set capacity to 0:
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

## 🔄 Optimization Bounds

When using `SystemOptimizer`, set bounds to match your enabled components:

```python
# Example: PV + Battery + Genset only
bounds = [
    (5000, 30000),    # PV capacity (active)
    (0, 0),           # Wind turbines (DISABLED)  
    (5000, 20000),    # Battery energy (active)
    (1000, 5000),     # Battery power (active)
    (15000, 25000)    # Genset capacity (active)
]
```

## 📖 Parameter Dictionary Usage

The `input_file_parameters_dictionary.yaml` shows ALL available parameters. To customize:

1. **Find** the parameter you want to change
2. **Copy** the parameter line to your configuration file
3. **Uncomment** it (remove the `#`) and modify the value

Example:
```yaml
# From parameters_dictionary.yaml:
# module_type: 0    # 0=standard silicon (19%)

# In your config file:
pv:
  system_capacity_kw: 15000
  module_type: 1    # Use premium modules (21% efficiency)
```

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

## ⚠️ Important Notes

1. **Always ensure at least one power source** is enabled (PV, Wind, Genset, or Grid)
2. **Battery is optional** - system can work without storage if grid/genset provides backup
3. **Resource files** are only needed for enabled renewable components
4. **Only Grid supports** `enabled: true/false` parameter
5. **Optimization will only size** enabled components (non-zero capacity, non-zero bounds)

## 🔍 Troubleshooting

### **Common Issues:**
- **"PVConfig got extraneous inputs 'enabled'"** → Remove `enabled:` from PV config (only Grid supports this)
- **Optimization finds minimum sizes** → Check bounds are appropriate for your load
- **"No module named numpy"** → Install in correct Python environment
- **Resource download fails** → Check NREL API key and internet connection

### **Getting Help:**
- Check parameter dictionary for available options
- Review architecture examples for similar use cases
- Ensure component bounds match enabled components
- Verify file paths for load data and resource files

---

*For more information, see the main Py-Microgrid documentation and examples.*