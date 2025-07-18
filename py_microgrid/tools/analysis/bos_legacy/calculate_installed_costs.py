"""
Calculate installed costs for wind, solar, and hybrid systems.
"""

def calculate_installed_costs(wind_size_mw=0, solar_size_mw=0, hybrid_size_mw=0):
    """
    Calculate installation costs for renewable energy systems.
    
    Args:
        wind_size_mw: Wind system size in MW
        solar_size_mw: Solar system size in MW  
        hybrid_size_mw: Hybrid system size in MW
        
    Returns:
        Dict with wind, solar, and hybrid installed costs
    """
    # Fixed cost rates per MW
    solar_cost_per_mw = 960000  # $960k/MW
    wind_cost_per_mw = 1450000  # $1.45M/MW
    
    # Calculate costs
    wind_cost = wind_size_mw * wind_cost_per_mw
    solar_cost = solar_size_mw * solar_cost_per_mw
    
    # Hybrid systems split cost 50/50 between wind and solar
    hybrid_wind_cost = (hybrid_size_mw / 2) * wind_cost_per_mw
    hybrid_solar_cost = (hybrid_size_mw / 2) * solar_cost_per_mw
    hybrid_cost = hybrid_wind_cost + hybrid_solar_cost
    
    return {
        'wind_cost': wind_cost,
        'solar_cost': solar_cost,
        'hybrid_cost': hybrid_cost,
        'hybrid_wind_cost': hybrid_wind_cost,
        'hybrid_solar_cost': hybrid_solar_cost
    }