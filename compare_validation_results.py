#!/usr/bin/env python3
"""
Compare validation results between old and new versions of py_microgrid.

This script compares the results from validation_test.py run on different versions.

Usage:
    python compare_validation_results.py old_results.json new_results.json

Or if you have results named with version suffixes:
    python compare_validation_results.py validation_results_old.json validation_results_new.json
"""

import json
import sys
import os
from typing import Dict, Any

def load_results(filepath: str) -> Dict[str, Any]:
    """Load validation results from JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file '{filepath}'.")
        return {}

def compare_numeric_values(old_val: float, new_val: float, tolerance: float = 1e-6) -> Dict[str, Any]:
    """Compare two numeric values with tolerance."""
    if old_val == 0 and new_val == 0:
        return {'match': True, 'difference': 0, 'percent_diff': 0}
    
    diff = abs(new_val - old_val)
    percent_diff = (diff / abs(old_val)) * 100 if old_val != 0 else float('inf')
    
    return {
        'match': diff <= tolerance,
        'difference': diff,
        'percent_diff': percent_diff
    }

def compare_results(old_results: Dict[str, Any], new_results: Dict[str, Any]) -> Dict[str, Any]:
    """Compare validation results between old and new versions."""
    
    if not old_results.get('success', False):
        return {'error': 'Old version test failed', 'old_error': old_results.get('error', 'Unknown')}
    
    if not new_results.get('success', False):
        return {'error': 'New version test failed', 'new_error': new_results.get('error', 'Unknown')}
    
    # Key metrics to compare
    metrics = [
        ('pv_capacity_kw', 'PV Capacity (kW)'),
        ('wind_capacity_kw', 'Wind Capacity (kW)'),
        ('battery_energy_kwh', 'Battery Energy (kWh)'),
        ('battery_power_kw', 'Battery Power (kW)'),
        ('genset_capacity_kw', 'Genset Capacity (kW)'),
        ('system_lcoe_dollars_per_kwh', 'System LCOE ($/kWh)'),
        ('system_npc_dollars', 'System NPC ($)'),
        ('total_co2_emissions_tonnes', 'CO2 Emissions (tonnes)'),
        ('demand_met_percentage', 'Demand Met (%)'),
        ('total_system_generation_kwh', 'Total Generation (kWh)'),
        ('total_pv_generation_kwh', 'PV Generation (kWh)'),
        ('total_wind_generation_kwh', 'Wind Generation (kWh)'),
        ('total_genset_generation_kwh', 'Genset Generation (kWh)'),
        ('total_battery_generation_kwh', 'Battery Generation (kWh)'),
        ('total_load_served_kwh', 'Load Served (kWh)')
    ]
    
    comparison = {
        'overall_match': True,
        'metrics': {},
        'summary': {
            'total_metrics': len(metrics),
            'matching_metrics': 0,
            'failing_metrics': 0
        }
    }
    
    # Compare each metric
    for metric_key, metric_name in metrics:
        old_val = old_results.get(metric_key, 0)
        new_val = new_results.get(metric_key, 0)
        
        # Use different tolerances for different types of values
        if 'percentage' in metric_key or 'lcoe' in metric_key:
            tolerance = 1e-3  # 0.1% tolerance for percentages and LCOE
        elif 'dollars' in metric_key:
            tolerance = 1.0   # $1 tolerance for dollar amounts
        else:
            tolerance = 1e-6  # Very small tolerance for other values
        
        comp_result = compare_numeric_values(old_val, new_val, tolerance)
        
        comparison['metrics'][metric_key] = {
            'name': metric_name,
            'old_value': old_val,
            'new_value': new_val,
            'match': comp_result['match'],
            'difference': comp_result['difference'],
            'percent_difference': comp_result['percent_diff']
        }
        
        if comp_result['match']:
            comparison['summary']['matching_metrics'] += 1
        else:
            comparison['summary']['failing_metrics'] += 1
            comparison['overall_match'] = False
    
    return comparison

def print_comparison_report(comparison: Dict[str, Any], old_results: Dict[str, Any], new_results: Dict[str, Any]):
    """Print a detailed comparison report."""
    
    print("=" * 80)
    print("PY_MICROGRID BACKWARDS COMPATIBILITY VALIDATION REPORT")
    print("=" * 80)
    
    # Print version info
    print(f"Old Version Git Commit: {old_results.get('git_commit', 'unknown')}")
    print(f"New Version Git Commit: {new_results.get('git_commit', 'unknown')}")
    print()
    
    # Check for errors
    if 'error' in comparison:
        print("❌ VALIDATION FAILED")
        print(f"Error: {comparison['error']}")
        if 'old_error' in comparison:
            print(f"Old version error: {comparison['old_error']}")
        if 'new_error' in comparison:
            print(f"New version error: {comparison['new_error']}")
        return
    
    # Print overall result
    if comparison['overall_match']:
        print("✅ BACKWARDS COMPATIBILITY VALIDATED")
        print("All metrics match within acceptable tolerances!")
    else:
        print("❌ BACKWARDS COMPATIBILITY ISSUES DETECTED")
        print(f"Failing metrics: {comparison['summary']['failing_metrics']}")
    
    print()
    print(f"Summary: {comparison['summary']['matching_metrics']}/{comparison['summary']['total_metrics']} metrics match")
    print()
    
    # Print detailed comparison
    print("DETAILED COMPARISON:")
    print("-" * 80)
    
    for metric_key, metric_data in comparison['metrics'].items():
        status = "✅" if metric_data['match'] else "❌"
        print(f"{status} {metric_data['name']}")
        print(f"    Old: {metric_data['old_value']:,.6f}")
        print(f"    New: {metric_data['new_value']:,.6f}")
        
        if not metric_data['match']:
            print(f"    Difference: {metric_data['difference']:,.6f}")
            if metric_data['percent_difference'] != float('inf'):
                print(f"    Percent Diff: {metric_data['percent_difference']:.3f}%")
        print()
    
    print("=" * 80)
    
    # Print recommendations
    if comparison['overall_match']:
        print("✅ RECOMMENDATION: Safe to proceed with the refactoring.")
        print("The new version produces identical results to the old version.")
    else:
        print("⚠️  RECOMMENDATION: Review the failing metrics before proceeding.")
        print("There may be issues with the refactoring that need to be addressed.")
    
    print("=" * 80)

def main():
    """Main function to compare validation results."""
    
    # Check command line arguments
    if len(sys.argv) == 3:
        old_file = sys.argv[1]
        new_file = sys.argv[2]
    else:
        # Look for default files
        possible_files = [
            ('validation_results_old.json', 'validation_results_new.json'),
            ('validation_results_baseline.json', 'validation_results.json'),
            ('validation_results.json', 'validation_results_new.json')
        ]
        
        old_file = new_file = None
        for old_candidate, new_candidate in possible_files:
            if os.path.exists(old_candidate) and os.path.exists(new_candidate):
                old_file = old_candidate
                new_file = new_candidate
                break
        
        if not old_file or not new_file:
            print("Usage: python compare_validation_results.py <old_results.json> <new_results.json>")
            print("\nOr ensure you have one of these file pairs:")
            for old_candidate, new_candidate in possible_files:
                print(f"  - {old_candidate} and {new_candidate}")
            return 1
    
    print(f"Comparing: {old_file} vs {new_file}")
    
    # Load results
    old_results = load_results(old_file)
    new_results = load_results(new_file)
    
    if not old_results or not new_results:
        return 1
    
    # Compare results
    comparison = compare_results(old_results, new_results)
    
    # Print report
    print_comparison_report(comparison, old_results, new_results)
    
    # Return appropriate exit code
    return 0 if comparison.get('overall_match', False) else 1

if __name__ == "__main__":
    exit(main())