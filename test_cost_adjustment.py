#!/usr/bin/env python3
"""
Test optimization with fine-tuned CostPerMW multipliers
"""

from py_microgrid.utilities.keys import set_developer_nrel_gov_key
set_developer_nrel_gov_key('ZaurwKOnwDUp8rMyNBIxI4XiBo3b7L5oruTi0VX3')

from py_microgrid.utilities import ConfigManager
from py_microgrid.tools.optimization.system_optimizer import SystemOptimizer
from py_microgrid.tools.analysis.bos import EconomicCalculator

def test_cost_adjustment():
    """Test single objective function call with fine-tuned costs."""
    
    yaml_file_path = "./py_microgrid/quick_start_config.yaml"
    config_manager = ConfigManager()
    config = config_manager.load_yaml_safely(yaml_file_path)
    
    # Make sure grid is disabled
    config['technologies']['grid']['enabled'] = False
    config_manager.save_yaml_safely(config, yaml_file_path)
    
    # Initialize optimizer
    economic_calculator = EconomicCalculator(discount_rate=0.0588, project_lifetime=25)
    optimizer = SystemOptimizer(
        yaml_file_path=yaml_file_path,
        economic_calculator=economic_calculator,
        enable_flexible_load=True,
        max_load_reduction_percentage=0.2
    )
    
    # Test with original optimal configuration
    test_config = [29310.0, 8.0, 5000.0, 2931.0, 17000.0]
    
    print("Testing original HOPP cost calculation method...")
    print("Config: PV=29.31MW, Wind=8x1MW, Battery=5MWh/2.9MW, Genset=17MW")
    print("Applied: HOPP's built-in total_installed_cost (includes industry-standard BOS)")
    
    try:
        lcoe, results = optimizer.objective_function(test_config)
        
        print(f"\nResults with original HOPP method:")
        print(f"LCOE: ${lcoe:.4f}/kWh")
        print(f"System Cost: ${results.get('System NPC ($)', 0):,.2f}")
        print(f"CO2 Emissions: {results.get('Total CO2 emissions (tonne)', 0):,.2f} tonnes")
        
        print(f"\nOriginal target results:")
        print(f"LCOE: $0.3067/kWh")
        print(f"System Cost: $144,779,752")
        print(f"CO2 Emissions: 751,960 tonnes")
        
        print(f"\nPrevious results comparison:")
        print(f"BOSLookup: LCOE $0.0894/kWh (0.29x target)")
        print(f"CostPerMW 5.7x: LCOE $0.5566/kWh (1.81x target)")
        print(f"CostPerMW 3.2x: LCOE $0.3005/kWh (0.98x target)")
        print(f"HOPP Original: LCOE ${lcoe:.4f}/kWh ({lcoe/0.3067:.2f}x target)")
        
        # Check progress
        lcoe_ratio = lcoe / 0.3067
        cost_ratio = results.get('System NPC ($)', 0) / 144779752
        co2_ratio = results.get('Total CO2 emissions (tonne)', 0) / 751960
        
        print(f"\nProgress ratios (target = 1.0):")
        print(f"LCOE ratio: {lcoe_ratio:.2f}")
        print(f"Cost ratio: {cost_ratio:.2f}")
        print(f"CO2 ratio: {co2_ratio:.2f}")
        
        if 0.85 <= lcoe_ratio <= 1.15 and 0.85 <= cost_ratio <= 1.15 and 0.85 <= co2_ratio <= 1.15:
            print("✓ Original HOPP method is working excellently!")
        elif 0.7 <= lcoe_ratio <= 1.3 and 0.7 <= cost_ratio <= 1.3 and 0.7 <= co2_ratio <= 1.3:
            print("✓ Original HOPP method is working well!")
        elif 0.5 <= lcoe_ratio <= 1.5 and 0.5 <= cost_ratio <= 1.5:
            print("⚠ Original HOPP method is working but CO2 emissions need attention")
        else:
            print("⚠ Original HOPP method needs further investigation")
            
    except Exception as e:
        print(f"✗ Error testing original HOPP method: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_cost_adjustment()