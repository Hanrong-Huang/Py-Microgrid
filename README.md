# Py_Microgrid

This package provides tools for simulating and optimizing hybrid microgrids with various technologies.

## Installation

Clone this repository and ensure you have all the required dependencies installed:

```bash
# Clone the repository
git clone https://github.com/Hanrong-Huang/Py-Microgrid.git

# Navigate to the directory
cd Py-Microgrid

# Install dependencies
pip install -r requirements.txt
```

## YAML Configuration Setup

When using Py_Microgrid, you'll need to properly configure the YAML files with the correct paths to your resource files.

### Important Configuration Steps:

1. In your YAML configuration file, update the following paths to point to your local file locations:
   - `solar_resource_file`: Path to your solar resource data file
   - `wind_resource_file`: Path to your wind resource data file
   - `grid_resource_file`: Path to your grid resource data file

2. Base your configuration on the example files provided in the repository:
   ```
   Py_Microgrid\examples\parallel_simulations\input_yaml\input_file_chunk_0.yaml
   ```

### Example YAML Configuration:

```yaml
# Sample configuration - update file paths with your local paths
simulation:
  solar_resource_file: "C:/path/to/your/solar_data.csv"
  wind_resource_file: "C:/path/to/your/wind_data.csv"
  grid_resource_file: "C:/path/to/your/grid_data.csv"
  # Other configuration parameters...
```

## Quick Start

A quick start Jupyter notebook is available in the repository:
```
Py_Microgrid\quick_start_example.ipynb
```

This notebook demonstrates the basic usage of the package and provides examples of how to set up and run simulations.

## License

[License information]

## Contact

[Contact information]
