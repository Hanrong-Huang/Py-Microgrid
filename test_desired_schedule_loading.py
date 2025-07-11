#!/usr/bin/env python3
"""
Test script for desired_schedule file path loading functionality.
Tests both backwards compatibility (list input) and new file path feature.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'py_microgrid'))

import yaml
import numpy as np
from py_microgrid.simulation.technologies.sites.site_info import SiteInfo
from py_microgrid.utilities.keys import set_developer_nrel_gov_key

# Set up NREL API key for testing
try:
    set_developer_nrel_gov_key("ZaurwKOnwDUp8rMyNBIxI4XiBo3b7L5oruTi0VX3")  # Use actual key
    os.environ['NREL_API_EMAIL'] = 'test@example.com'  # Set email for API calls
except:
    pass


def test_desired_schedule_loading():
    """Test the new desired_schedule file path loading functionality."""
    
    print("Testing desired_schedule file path loading...")
    
    # Test 1: Backwards compatibility - list input
    print("\n1. Testing backwards compatibility (list input):")
    try:
        config_list = {
            'data': {
                'lat': 39.7555,
                'lon': -105.2211,
                'year': 2020
            },
            'desired_schedule': [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            'solar_resource_file': '',
            'wind_resource_file': '',
            'wave_resource_file': '',
            'grid_resource_file': ''
        }
        
        site_list = SiteInfo(**config_list)
        print(f"   [OK] List input: {len(site_list.desired_schedule)} values loaded")
        print(f"   [OK] First few values: {site_list.desired_schedule[:5]}")
        print(f"   [OK] follow_desired_schedule: {site_list.follow_desired_schedule}")
        
    except Exception as e:
        print(f"   [FAIL] List input failed: {e}")
        return False
    
    # Test 2: File path input
    print("\n2. Testing file path input:")
    try:
        config_file = {
            'data': {
                'lat': 39.7555,
                'lon': -105.2211,
                'year': 2020
            },
            'desired_schedule': "py_microgrid/examples/parallel_simulations/load_data/desired_schedule_sample.csv",
            'solar_resource_file': '',
            'wind_resource_file': '',
            'wave_resource_file': '',
            'grid_resource_file': ''
        }
        
        site_file = SiteInfo(**config_file)
        print(f"   [OK] File path input: {len(site_file.desired_schedule)} values loaded")
        print(f"   [OK] First few values: {site_file.desired_schedule[:5]}")
        print(f"   [OK] follow_desired_schedule: {site_file.follow_desired_schedule}")
        
    except Exception as e:
        print(f"   [FAIL] File path input failed: {e}")
        return False
    
    # Test 3: Empty input
    print("\n3. Testing empty input:")
    try:
        config_empty = {
            'data': {
                'lat': 39.7555,
                'lon': -105.2211,
                'year': 2020
            },
            'desired_schedule': [],
            'solar_resource_file': '',
            'wind_resource_file': '',
            'wave_resource_file': '',
            'grid_resource_file': ''
        }
        
        site_empty = SiteInfo(**config_empty)
        print(f"   [OK] Empty input: {len(site_empty.desired_schedule)} values loaded")
        print(f"   [OK] follow_desired_schedule: {site_empty.follow_desired_schedule}")
        
    except Exception as e:
        print(f"   [FAIL] Empty input failed: {e}")
        return False
    
    # Test 4: Compare values between list and file
    print("\n4. Testing data consistency:")
    try:
        if np.allclose(site_list.desired_schedule, site_file.desired_schedule):
            print("   [OK] List and file inputs produce identical results")
        else:
            print("   [FAIL] List and file inputs produce different results")
            return False
            
    except Exception as e:
        print(f"   [FAIL] Data consistency test failed: {e}")
        return False
    
    # Test 5: Test with YAML loading
    print("\n5. Testing YAML file loading:")
    try:
        with open('py_microgrid/examples/parallel_simulations/input_yaml/input_file_with_schedule_path.yaml', 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        site_yaml = SiteInfo(**yaml_config['site'])
        print(f"   [OK] YAML file loaded: {len(site_yaml.desired_schedule)} values")
        print(f"   [OK] First few values: {site_yaml.desired_schedule[:5]}")
        print(f"   [OK] follow_desired_schedule: {site_yaml.follow_desired_schedule}")
        
    except Exception as e:
        print(f"   [FAIL] YAML file loading failed: {e}")
        return False
    
    print("\n[OK] All tests passed! desired_schedule file path loading is working correctly.")
    return True


if __name__ == "__main__":
    success = test_desired_schedule_loading()
    if success:
        print("\n[SUCCESS] Step 7 implementation complete!")
        sys.exit(0)
    else:
        print("\n[FAIL] Step 7 implementation failed!")
        sys.exit(1)