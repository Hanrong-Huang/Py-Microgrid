# Py-Microgrid: Hybrid Microgrid Simulation & Optimisation Model

**Project Webpage:** https://hanrong-huang.github.io/py_microgrid_webpage/

Extension package based on HOPP (Hybrid Optimisation and Performance Platform) enabling hybrid microgrid system simulation & optimisation with flexible load management and predictive battery dispatch.

## Features
- **System Optimisation**: Optimises component sizes for PV, wind, battery storage, and genset.
- **Flexible Load Management**: Implements up to 20% load reduction capability.
- **Predictive Battery Dispatch**: Enhanced battery management with demand response.
- **Economic Analysis**: LCOE, NPC, CO2 emissions, and lifetime costs.
- **Multi-Location Processing**: Batch analysis for multiple sites simultaneously.

## Installation

```bash
# Clone Py-Microgrid repository
git clone https://github.com/Hanrong-Huang/Py-Microgrid.git
```
Or download the file directly from the [repository page](https://github.com/Hanrong-Huang/Py-Microgrid).

### Required Dependencies

1. **Python Version**
   - Python 3.10 or 3.11 required.

2. **HOPP (Hybrid Optimization and Performance Platform)**
   ```bash
   git clone https://github.com/NREL/HOPP.git
   cd HOPP
   pip install -e .
   ```
   Or, download directly via pip:
   ```bash
   pip install HOPP
   ```

3. **Required Conda-forge Packages**
   ```bash
   conda install -c conda-forge glpk -y
   conda install -c conda-forge coin-or-cbc -y
   ```

## Documentation

For comprehensive documentation about HOPP (Hybrid Optimization and Performance Platform), please visit:
- **Official Documentation**: https://hopp.readthedocs.io/en/latest/
- **Original HOPP Repository**: https://github.com/NREL/HOPP

## Quick Start

The quick start example is at (API key prefilled)
```
py_microgrid/quick_start_example.ipynb.
```

Below is a quick start example with your own API key
```python
# Required imports
from py_microgrid.utilities import ConfigManager
from py_microgrid.utilities.keys import set_developer_nrel_gov_key
from py_microgrid.tools.optimization import SystemOptimizer, LoadAnalyzer
from py_microgrid.tools.analysis.bos import EconomicCalculator
from py_microgrid.simulation.resource_files import ResourceDataManager

# Set your NREL API key (required for downloading solar data)
set_developer_nrel_gov_key('YOUR-NREL-API-KEY')

# Initialize resource manager for downloading data
resource_manager = ResourceDataManager(
    api_key='YOUR-NREL-API-KEY',
    email='your.email@example.com'
)

# Location information
latitude = -33.5265
longitude = 149.1588

# Download resource data
solar_path = resource_manager.download_solar_data(
    latitude=latitude,
    longitude=longitude,
    year="2020"
)
wind_path = resource_manager.download_wind_data(
    latitude=latitude,
    longitude=longitude,
    start_date="20200101",
    end_date="20201231"
)

# Load and update YAML configuration
yaml_file_path = "path/to/your/config.yaml"  # Base configuration file
config_manager = ConfigManager()
config = config_manager.load_yaml_safely(yaml_file_path)

# Update configuration with location and resource files
config['site']['data']['lat'] = latitude
config['site']['data']['lon'] = longitude
config['site']['solar_resource_file'] = solar_path
config['site']['wind_resource_file'] = wind_path

# Save updated configuration
config_manager.save_yaml_safely(config, yaml_file_path)

# Initialize components for optimization
economic_calculator = EconomicCalculator(
    discount_rate=0.0588,
    project_lifetime=25
)

optimizer = SystemOptimizer(
    yaml_file_path=yaml_file_path,
    economic_calculator=economic_calculator,
    enable_flexible_load=True,  # Set to False to disable flexible load
    max_load_reduction_percentage=0.2  # 20% maximum load reduction
)

# Define optimization bounds
bounds = [
    (5000, 50000),    # PV capacity (kW)
    (1, 50),          # Wind turbines (1MW each)
    (5000, 30000),    # Battery capacity (kWh)
    (1000, 10000),    # Battery power (kW)
    (17000, 30000)    # Genset capacity (kW)
]

# Define initial conditions (10% of range)
initial_conditions = [
    [bound[0] + (bound[1] - bound[0]) * 0.1 for bound in bounds]
]

# Run optimization
result = optimizer.optimize_system(bounds, initial_conditions)

# Print results
if result:
    print("\nOptimization Results:")
    print(f"PV Capacity: {result['PV Capacity (kW)']:.2f} kW")
    print(f"Wind Turbines: {result['Wind Turbine Capacity (kW)'] / 1000:.0f} x 1MW")
    print(f"Battery Capacity: {result['Battery Energy Capacity (kWh)']:.2f} kWh")
    print(f"Battery Power: {result['Battery Power Capacity (kW)']:.2f} kW")
    print(f"Genset Capacity: {result['Genset Capacity (kW)']:.2f} kW")
    print(f"\nLCOE: ${result['System LCOE ($/kWh)']:.4f}/kWh")
    print(f"System Net Present Cost: ${result['System NPC ($)']:,.2f}")
    print(f"CO2 Emissions: {result['Total CO2 emissions (tonne)']:.2f} tonnes")
    print(f"Demand Met: {result['Demand Met Percentage']:.2f}%")
```

### Required Files

1. **Example YAML system configuration file**
```yaml
site:
  data:
    lat: 0
    lon: 0
  solar_resource_file: ""
  wind_resource_file: ""
technologies:
  pv:
    system_capacity_kw: 10000
  wind:
    num_turbines: 5
  battery:
    system_capacity_kwh: 10000
    system_capacity_kw: 2000
  grid:
    interconnect_kw: 20000
```

> **Important:** Update the resource files in your YAML to point to the correct local paths on your system.

## Multi-Location Batch Processing

Py-Microgrid supports the optimization of multiple locations in a single run, demonstrated in the example notebook:
```
py_microgrid/examples/parallel_simulations/py_microgrid_example/simulation_chunk_0.ipynb
```

### How It Works

1. **Location Data**: Provide a CSV file with location information:
   ```python
   # Load sites from CSV file with coordinates
   import pandas as pd
   locations = pd.read_csv("locations.csv")  # Contains latitude, longitude, and site IDs
   ```

2. **Batch Processing**: Process each location sequentially:
   ```python
   results = []
   for _, row in locations.iterrows():
       result = optimizer.process_location(
           latitude=row['DEPOSIT_LATITUDE'],
           longitude=row['DEPOSIT_LONGITUDE'],
           location_id=row['DEPOSIT_UID']
       )
       if result:
           results.append(result)
   ```

3. **Results Analysis**: Generate analytics across all sites:
   ```python
   # Save consolidated results
   results_df = pd.DataFrame(results)
   results_df.to_csv("optimization_results.csv", index=False)

   # Generate summary statistics
   print(f"Total locations processed: {len(results)}")
   print(f"Average LCOE: ${results_df['System LCOE ($/kWh)'].mean():.4f}/kWh")
   print(f"Best LCOE: ${results_df['System LCOE ($/kWh)'].min():.4f}/kWh")
   ```

### Applications
- Regional microgrid planning
- Site comparison and selection
- Portfolio optimization
- Geospatial energy analysis

## API Reference

### NREL Developer API
For accessing NREL Himawari solar resource data:
1. Register at https://developer.nrel.gov/signup/
2. Get your API key from the account dashboard.
3. Set your API key:
```python
from py_microgrid.utilities.keys import set_developer_nrel_gov_key
set_developer_nrel_gov_key('YOUR-NREL-API-KEY')
```

### NASA POWER API
For accessing NASA POWER MERRA-2 wind resource data:
- No API key required
- User registration through email is sufficient

## Resource Data Management

```python
from py_microgrid.utilities.keys import set_developer_nrel_gov_key
set_developer_nrel_gov_key('YOUR-API-KEY')

# Example usage of ResourceDataManager
from py_microgrid.simulation.resource_files import ResourceDataManager

manager = ResourceDataManager(api_key='YOUR-API-KEY', email='your.email@example.com')

# Download solar data (NREL Himawari)
solar_file = manager.download_solar_data(latitude=-33.9, longitude=151.2, year='2023')

# Download wind data (NASA POWER MERRA-2)
wind_file = manager.download_wind_data(
    latitude=-33.9,
    longitude=151.2,
    start_date='20230101',
    end_date='20231231'
)
```

## Key Features Detail

### Flexible Load Management
- Up to 20% load reduction capability.
- Automatic adjustment during peak demand.
- Optimization of demand response.

### Enhanced Genset Modeling
- Improved operational characteristics
- Detailed cost modeling
- CO2 emissions tracking

## Results Output

The optimization provides comprehensive results including:
- Optimal component sizes
- System LCOE and NPC
- CO2 emissions
- Performance metrics

## License

This module is licensed under the BSD 3-Clause License.

```
BSD 3-Clause License

Copyright (c) 2024, HOPP Contributors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Contact

For questions and support, contact:
hanrong.huang@unsw.edu.au
