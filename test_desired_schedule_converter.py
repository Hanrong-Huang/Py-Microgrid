#!/usr/bin/env python3
"""
Test script for desired_schedule converter functionality only.
Tests both backwards compatibility (list input) and new file path feature.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'py_microgrid'))

import numpy as np
from py_microgrid.type_dec import desired_schedule_converter


def test_desired_schedule_converter():
    """Test the desired_schedule_converter function directly."""
    
    print("Testing desired_schedule_converter...")
    
    # Test 1: List input (backwards compatibility)
    print("\n1. Testing list input (backwards compatibility):")
    try:
        test_list = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
        result_list = desired_schedule_converter(test_list)
        print(f"   [OK] List input: {len(result_list)} values loaded")
        print(f"   [OK] Type: {type(result_list)}")
        print(f"   [OK] First few values: {result_list[:5]}")
        
    except Exception as e:
        print(f"   [FAIL] List input failed: {e}")
        return False
    
    # Test 2: File path input
    print("\n2. Testing file path input:")
    try:
        test_file = "py_microgrid/examples/parallel_simulations/load_data/desired_schedule_sample.csv"
        result_file = desired_schedule_converter(test_file)
        print(f"   [OK] File path input: {len(result_file)} values loaded")
        print(f"   [OK] Type: {type(result_file)}")
        print(f"   [OK] First few values: {result_file[:5]}")
        
    except Exception as e:
        print(f"   [FAIL] File path input failed: {e}")
        return False
    
    # Test 3: Empty list
    print("\n3. Testing empty list:")
    try:
        result_empty = desired_schedule_converter([])
        print(f"   [OK] Empty list: {len(result_empty)} values loaded")
        print(f"   [OK] Type: {type(result_empty)}")
        
    except Exception as e:
        print(f"   [FAIL] Empty list failed: {e}")
        return False
    
    # Test 4: Empty string
    print("\n4. Testing empty string:")
    try:
        result_empty_str = desired_schedule_converter("")
        print(f"   [OK] Empty string: {len(result_empty_str)} values loaded")
        print(f"   [OK] Type: {type(result_empty_str)}")
        
    except Exception as e:
        print(f"   [FAIL] Empty string failed: {e}")
        return False
    
    # Test 5: Compare values between list and file
    print("\n5. Testing data consistency:")
    try:
        if np.allclose(result_list, result_file):
            print("   [OK] List and file inputs produce identical results")
        else:
            print("   [FAIL] List and file inputs produce different results")
            print(f"   List: {result_list}")
            print(f"   File: {result_file}")
            return False
            
    except Exception as e:
        print(f"   [FAIL] Data consistency test failed: {e}")
        return False
    
    # Test 6: Invalid input type
    print("\n6. Testing invalid input type:")
    try:
        result_invalid = desired_schedule_converter(123)
        print(f"   [FAIL] Invalid input should have failed: {result_invalid}")
        return False
        
    except TypeError as e:
        print(f"   [OK] Invalid input correctly rejected: {e}")
        
    except Exception as e:
        print(f"   [FAIL] Unexpected error: {e}")
        return False
    
    # Test 7: Non-existent file
    print("\n7. Testing non-existent file:")
    try:
        result_nonexistent = desired_schedule_converter("nonexistent_file.csv")
        print(f"   [FAIL] Non-existent file should have failed: {result_nonexistent}")
        return False
        
    except FileNotFoundError as e:
        print(f"   [OK] Non-existent file correctly rejected: {e}")
        
    except Exception as e:
        print(f"   [FAIL] Unexpected error: {e}")
        return False
    
    print("\n[OK] All converter tests passed! desired_schedule file path loading is working correctly.")
    return True


if __name__ == "__main__":
    success = test_desired_schedule_converter()
    if success:
        print("\n[SUCCESS] Step 7 implementation complete!")
        sys.exit(0)
    else:
        print("\n[FAIL] Step 7 implementation failed!")
        sys.exit(1)