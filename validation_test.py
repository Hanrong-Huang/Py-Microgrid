#!/usr/bin/env python3
"""
Validation test script for py_microgrid backwards compatibility.

This script runs a simple optimization test case and saves the results to a JSON file.
Run this script on both the old version and new version to compare results.

Usage:
    python validation_test.py

The results will be saved to 'validation_results.json'
"""

import json
import os
import sys
from pathlib import Path

# Add the py_microgrid directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'py_microgrid'))

def create_test_yaml():
    """Create a simple test configuration for validation."""
    test_config = {
        'config': {
            'dispatch_options': {
                'battery_dispatch': 'heuristic_load_following',
                'grid_charging': True,
                'include_lifecycle_count': False,
                'n_look_ahead_periods': 48,
                'pv_charging_only': False,
                'solver': 'glpk'
            }
        },
        'name': 'validation_test',
        'site': {
            'data': {
                'elev': 1000,
                'lat': -33.5,
                'lon': 149.2,
                'tz': -6,
                'year': 2020
            },
            'desired_schedule': [10.0] * 8760,  # Simple constant 10 MW load
            'follow_desired_schedule': True,
            'solar_resource_file': 'py_microgrid/simulation/resource_files/solar/-33.5265_149.1588_psmv3_60_2020.csv',
            'wind_resource_file': 'py_microgrid/simulation/resource_files/wind/-33.5265_149.1588_NASA_2020_60min_50m.srw'
        },
        'technologies': {
            'battery': {
                'enabled': True,
                'chemistry': 'LFPGraphite',
                'initial_SOC': 90.0,
                'maximum_SOC': 90.0,
                'minimum_SOC': 20.0,
                'system_capacity_kw': 2000.0,
                'system_capacity_kwh': 5000.0
            },
            'financial': {
                'Singleowner': {
                    'analysis_period': 25,
                    'inflation_rate': 2,
                    'real_discount_rate': 5.88
                }
            },
            'genset': {
                'enabled': True,
                'interconnect_kw': 15000.0,
                'ppa_price': 0.4
            },
            'grid': {
                'enabled': False,  # CRITICAL: Must be disabled for backwards compatibility
                'interconnect_kw': 10000.0,
                'import_limit_kw': 10000.0,
                'export_limit_kw': 5000.0,
                'base_import_price': 0.12,
                'base_export_price': 0.08,
                'allow_export': True,
                'dispatch_factors_file': None
            },
            'pv': {
                'enabled': True,
                'system_capacity_kw': 10000.0,
                'dc_degradation': [0.5] * 25
            },
            'wind': {
                'enabled': True,
                'num_turbines': 5,
                'turbine_rating_kw': 1000
            }
        }
    }
    
    return test_config

def run_validation_test():
    """Run the validation test and return results."""
    try:
        # Import py_microgrid modules
        from py_microgrid.utilities import ConfigManager
        from py_microgrid.tools.analysis.bos import EconomicCalculator
        from py_microgrid.tools.optimization import SystemOptimizer
        
        print("Starting validation test...")
        
        # Create test configuration
        test_config = create_test_yaml()
        
        # Save test configuration to temporary file
        config_manager = ConfigManager()
        temp_yaml_path = "validation_test_config.yaml"
        config_manager.save_yaml_safely(test_config, temp_yaml_path)
        
        print(f"Created test configuration: {temp_yaml_path}")
        
        # Initialize economic calculator
        economic_calculator = EconomicCalculator(
            discount_rate=0.0588,
            project_lifetime=25
        )
        
        # Initialize system optimizer
        optimizer = SystemOptimizer(
            yaml_file_path=temp_yaml_path,
            economic_calculator=economic_calculator,
            enable_flexible_load=True,
            max_load_reduction_percentage=0.2
        )
        
        print("Initialized optimizer, running optimization...")
        
        # Define simple optimization bounds (small ranges for quick testing)
        bounds = [
            (8000, 12000),   # PV capacity (kW)
            (3, 7),          # Wind turbines (1MW each)
            (4000, 6000),    # Battery capacity (kWh)
            (1500, 2500),    # Battery power (kW)
            (12000, 18000)   # Genset capacity (kW)
        ]
        
        # Define initial conditions
        initial_conditions = [
            [10000, 5, 5000, 2000, 15000]  # Single starting point for consistency
        ]
        
        print("Running optimization with bounds:", bounds)
        print("Initial conditions:", initial_conditions)
        
        # Run optimization
        result = optimizer.optimize_system(bounds, initial_conditions)
        
        if result is None:
            raise Exception("Optimization failed - no result returned")
        
        print("Optimization completed successfully!")
        
        # Extract key metrics for comparison
        validation_results = {
            'success': True,
            'pv_capacity_kw': result.get('PV Capacity (kW)', 0),
            'wind_capacity_kw': result.get('Wind Turbine Capacity (kW)', 0),
            'battery_energy_kwh': result.get('Battery Energy Capacity (kWh)', 0),
            'battery_power_kw': result.get('Battery Power Capacity (kW)', 0),
            'genset_capacity_kw': result.get('Genset Capacity (kW)', 0),
            'total_system_generation_kwh': result.get('Total System Generation (kWh)', 0),
            'total_pv_generation_kwh': result.get('Total PV Generation (kWh)', 0),
            'total_wind_generation_kwh': result.get('Total Wind Generation (kWh)', 0),
            'total_genset_generation_kwh': result.get('Total Genset Generation (kWh)', 0),
            'total_battery_generation_kwh': result.get('Total Battery Generation (kWh)', 0),
            'system_lcoe_dollars_per_kwh': result.get('System LCOE ($/kWh)', 0),
            'system_npc_dollars': result.get('System NPC ($)', 0),
            'total_co2_emissions_tonnes': result.get('Total CO2 emissions (tonne)', 0),
            'demand_met_percentage': result.get('Demand Met Percentage', 0),
            'total_load_served_kwh': result.get('Total Load Served (kWh)', 0)
        }
        
        # Add version info
        try:
            import py_microgrid
            validation_results['py_microgrid_version'] = getattr(py_microgrid, '__version__', 'unknown')
        except:
            validation_results['py_microgrid_version'] = 'unknown'
        
        # Add git commit info if available
        try:
            import subprocess
            commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
            validation_results['git_commit'] = commit_hash
        except:
            validation_results['git_commit'] = 'unknown'
        
        # Cleanup temporary file
        try:
            os.remove(temp_yaml_path)
        except:
            pass
        
        print("Validation test completed successfully!")
        return validation_results
        
    except Exception as e:
        print(f"Validation test failed: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }

def main():
    """Main function to run validation test and save results."""
    print("=" * 60)
    print("PY_MICROGRID VALIDATION TEST")
    print("=" * 60)
    
    # Run the validation test
    results = run_validation_test()
    
    # Save results to JSON file
    output_file = "validation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION TEST SUMMARY")
    print("=" * 60)
    
    if results['success']:
        print("✓ Test PASSED")
        print(f"PV Capacity: {results['pv_capacity_kw']:.0f} kW")
        print(f"Wind Capacity: {results['wind_capacity_kw']:.0f} kW")
        print(f"Battery Energy: {results['battery_energy_kwh']:.0f} kWh")
        print(f"Battery Power: {results['battery_power_kw']:.0f} kW")
        print(f"Genset Capacity: {results['genset_capacity_kw']:.0f} kW")
        print(f"System LCOE: ${results['system_lcoe_dollars_per_kwh']:.4f}/kWh")
        print(f"System NPC: ${results['system_npc_dollars']:,.0f}")
        print(f"CO2 Emissions: {results['total_co2_emissions_tonnes']:.1f} tonnes")
        print(f"Demand Met: {results['demand_met_percentage']:.1f}%")
    else:
        print("✗ Test FAILED")
        print(f"Error: {results.get('error', 'Unknown error')}")
        print(f"Error Type: {results.get('error_type', 'Unknown')}")
    
    print("=" * 60)
    
    return 0 if results['success'] else 1

if __name__ == "__main__":
    exit(main())