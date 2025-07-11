#!/usr/bin/env python3
"""
Test script for the new configuration system.
Tests loading of externalized parameters from YAML files.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'py_microgrid'))


def test_config_system():
    """Test the new configuration system."""
    
    print("Testing externalized configuration system...")
    
    # Test 1: Configuration Manager Loading
    print("\n1. Testing configuration manager loading:")
    try:
        from py_microgrid.simulation.config import ConfigManager
        config_manager = ConfigManager()
        print("   [OK] ConfigManager loaded successfully")
        
        # Test loading component configs
        battery_config = config_manager.get_config('battery')
        pv_config = config_manager.get_config('pv')
        wind_config = config_manager.get_config('wind')
        grid_config = config_manager.get_config('grid')
        
        print(f"   [OK] Battery config loaded: {len(battery_config)} sections")
        print(f"   [OK] PV config loaded: {len(pv_config)} sections")
        print(f"   [OK] Wind config loaded: {len(wind_config)} sections")
        print(f"   [OK] Grid config loaded: {len(grid_config)} sections")
        
    except Exception as e:
        print(f"   [FAIL] Configuration manager loading failed: {e}")
        return False
    
    # Test 2: Parameter Access
    print("\n2. Testing parameter access:")
    try:
        from py_microgrid.simulation.config import get_parameter, get_parameter_with_default
        
        # Test battery parameters
        battery_efficiency = get_parameter('battery', 'efficiency', 'round_trip_efficiency')
        battery_cost = get_parameter('battery', 'costs', 'installed_cost_per_kw')
        battery_soc = get_parameter('battery', 'operation', 'soc_min_pct')
        
        print(f"   [OK] Battery round trip efficiency: {battery_efficiency}%")
        print(f"   [OK] Battery installed cost: ${battery_cost}/kW")
        print(f"   [OK] Battery minimum SOC: {battery_soc}%")
        
        # Test PV parameters
        pv_standard_eff = get_parameter('pv', 'efficiency', 'standard_silicon')
        pv_premium_eff = get_parameter('pv', 'efficiency', 'premium_silicon')
        pv_cost = get_parameter('pv', 'costs', 'installed_cost_per_kw')
        
        print(f"   [OK] PV standard efficiency: {pv_standard_eff}")
        print(f"   [OK] PV premium efficiency: {pv_premium_eff}")
        print(f"   [OK] PV installed cost: ${pv_cost}/kW")
        
        # Test wind parameters
        wind_cp = get_parameter('wind', 'performance', 'default_max_cp')
        wind_cost = get_parameter('wind', 'costs', 'installed_cost_per_kw')
        
        print(f"   [OK] Wind max Cp: {wind_cp}")
        print(f"   [OK] Wind installed cost: ${wind_cost}/kW")
        
        # Test grid parameters
        grid_import_price = get_parameter('grid', 'pricing', 'base_import_price')
        grid_export_price = get_parameter('grid', 'pricing', 'base_export_price')
        
        print(f"   [OK] Grid import price: ${grid_import_price}/kWh")
        print(f"   [OK] Grid export price: ${grid_export_price}/kWh")
        
    except Exception as e:
        print(f"   [FAIL] Parameter access failed: {e}")
        return False
    
    # Test 3: Default Values
    print("\n3. Testing default values:")
    try:
        # Test parameter with default
        nonexistent_param = get_parameter_with_default('battery', 99.99, 'nonexistent', 'parameter')
        print(f"   [OK] Default value returned: {nonexistent_param}")
        
        # Test existing parameter
        existing_param = get_parameter_with_default('battery', 99.99, 'efficiency', 'round_trip_efficiency')
        print(f"   [OK] Existing parameter returned: {existing_param}")
        
        if existing_param != 99.99:
            print("   [OK] Existing parameter correctly overrode default")
        else:
            print("   [FAIL] Default value incorrectly returned for existing parameter")
            return False
            
    except Exception as e:
        print(f"   [FAIL] Default value testing failed: {e}")
        return False
    
    # Test 4: Configuration File Structure
    print("\n4. Testing configuration file structure:")
    try:
        expected_sections = {
            'battery': ['efficiency', 'costs', 'operation', 'dispatch'],
            'pv': ['efficiency', 'costs', 'performance', 'system'],
            'wind': ['performance', 'costs', 'operation', 'layout'],
            'grid': ['pricing', 'dispatch_factors', 'time_of_use', 'connection'],
        }
        
        for component, sections in expected_sections.items():
            config = config_manager.get_config(component)
            for section in sections:
                if section in config:
                    print(f"   [OK] {component}.{section} section found")
                else:
                    print(f"   [FAIL] {component}.{section} section missing")
                    return False
    
    except Exception as e:
        print(f"   [FAIL] Configuration structure testing failed: {e}")
        return False
    
    print("\n[OK] All configuration system tests passed!")
    return True


if __name__ == "__main__":
    success = test_config_system()
    if success:
        print("\n[SUCCESS] Step 8 implementation working correctly!")
        sys.exit(0)
    else:
        print("\n[FAIL] Step 8 implementation failed!")
        sys.exit(1)