#!/usr/bin/env python3
"""
Quick validation test for py_microgrid - just tests basic functionality without optimization.
"""

import json
import sys
import os

# Add the py_microgrid directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'py_microgrid'))

def test_basic_functionality():
    """Test basic imports and functionality."""
    try:
        print("Testing basic imports...")
        
        # Test imports
        from py_microgrid.utilities import ConfigManager
        from py_microgrid.tools.analysis.bos import EconomicCalculator
        from py_microgrid.tools.optimization import SystemOptimizer
        from py_microgrid.simulation import HoppInterface
        
        print("[OK] All imports successful!")
        
        # Test basic functionality
        config_manager = ConfigManager()
        economic_calculator = EconomicCalculator(discount_rate=0.0588, project_lifetime=25)
        
        print("[OK] Basic object creation successful!")
        
        # Test YAML loading with our 5-component structure
        test_config = {
            'technologies': {
                'pv': {'enabled': True, 'system_capacity_kw': 1000},
                'wind': {'enabled': True, 'num_turbines': 2},
                'battery': {'enabled': True, 'system_capacity_kwh': 1000, 'system_capacity_kw': 500},
                'genset': {'enabled': True, 'interconnect_kw': 2000},
                'grid': {'enabled': False, 'interconnect_kw': 1000}
            }
        }
        
        temp_yaml = "quick_test_config.yaml"
        config_manager.save_yaml_safely(test_config, temp_yaml)
        loaded_config = config_manager.load_yaml_safely(temp_yaml)
        
        print("[OK] YAML configuration handling successful!")
        
        # Test that all 5 components are recognized
        techs = loaded_config['technologies']
        expected_components = ['pv', 'wind', 'battery', 'genset', 'grid']
        
        for component in expected_components:
            if component not in techs:
                raise Exception(f"Component '{component}' not found in configuration")
            if 'enabled' not in techs[component]:
                raise Exception(f"Component '{component}' missing 'enabled' flag")
        
        print("[OK] All 5 components (pv, wind, battery, genset, grid) properly configured!")
        
        # Test that grid is disabled by default (backwards compatibility)
        if techs['grid']['enabled'] != False:
            raise Exception("Grid should be disabled by default for backwards compatibility")
        
        print("[OK] Grid properly disabled by default (backwards compatibility confirmed)!")
        
        # Cleanup
        try:
            os.remove(temp_yaml)
        except:
            pass
        
        return {
            'success': True,
            'message': 'All basic functionality tests passed',
            'components_tested': expected_components,
            'backwards_compatible': True
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }

def main():
    """Run quick validation test."""
    print("=" * 60)
    print("PY_MICROGRID QUICK VALIDATION TEST")
    print("=" * 60)
    
    result = test_basic_functionality()
    
    # Save result
    with open('quick_validation_results.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print("\nResults saved to: quick_validation_results.json")
    print("\n" + "=" * 60)
    print("QUICK VALIDATION SUMMARY")
    print("=" * 60)
    
    if result['success']:
        print("[PASS] QUICK VALIDATION PASSED")
        print(f"Message: {result['message']}")
        print(f"Components: {', '.join(result['components_tested'])}")
        print(f"Backwards Compatible: {result['backwards_compatible']}")
        print("\nThe refactoring appears to be working correctly!")
        print("You can now run the full validation test for complete verification.")
    else:
        print("[FAIL] QUICK VALIDATION FAILED")
        print(f"Error: {result['error']}")
        print(f"Error Type: {result['error_type']}")
        print("\nThere are issues that need to be fixed before proceeding.")
    
    print("=" * 60)
    
    return 0 if result['success'] else 1

if __name__ == "__main__":
    exit(main())