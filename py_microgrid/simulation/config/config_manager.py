"""
Configuration manager for py_microgrid.

This module provides centralized configuration management, loading parameters
from YAML files instead of hardcoded values throughout the codebase.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from py_microgrid.utilities.log import hybrid_logger as logger


class ConfigManager:
    """
    Centralized configuration manager for py_microgrid.
    
    Loads configuration parameters from YAML files and provides
    easy access to component-specific parameters.
    """
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Directory containing configuration files.
                       If None, uses default directory.
        """
        if config_dir is None:
            config_dir = Path(__file__).parent
        else:
            config_dir = Path(config_dir)
            
        self.config_dir = config_dir
        self._configs = {}
        
        # Load all configuration files
        self._load_configs()
    
    def _load_configs(self):
        """Load all configuration files from the config directory."""
        config_files = {
            'battery': 'battery_config.yaml',
            'pv': 'pv_config.yaml',
            'wind': 'wind_config.yaml',
            'genset': 'genset_config.yaml',
            'grid': 'grid_config.yaml',
            'dispatch': 'dispatch_config.yaml'
        }
        
        for component, filename in config_files.items():
            file_path = self.config_dir / filename
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        config = yaml.safe_load(f)
                        self._configs[component] = config[component]
                        logger.info(f"Loaded {component} configuration from {filename}")
                except Exception as e:
                    logger.error(f"Error loading {component} configuration: {e}")
                    self._configs[component] = {}
            else:
                logger.warning(f"Configuration file not found: {filename}")
                self._configs[component] = {}
    
    def get_config(self, component: str) -> Dict[str, Any]:
        """
        Get configuration for a specific component.
        
        Args:
            component: Component name (battery, pv, wind, genset, grid, dispatch)
            
        Returns:
            Dictionary containing component configuration
        """
        return self._configs.get(component, {})
    
    def get_parameter(self, component: str, *keys) -> Any:
        """
        Get a specific parameter from component configuration.
        
        Args:
            component: Component name
            *keys: Nested keys to access the parameter
            
        Returns:
            Parameter value, or None if not found
            
        Example:
            get_parameter('battery', 'efficiency', 'round_trip_efficiency')
        """
        config = self.get_config(component)
        
        for key in keys:
            if isinstance(config, dict) and key in config:
                config = config[key]
            else:
                return None
        
        return config
    
    def get_parameter_with_default(self, component: str, default_value: Any, *keys) -> Any:
        """
        Get a parameter with a default value if not found.
        
        Args:
            component: Component name
            default_value: Default value to return if parameter not found
            *keys: Nested keys to access the parameter
            
        Returns:
            Parameter value or default value
        """
        value = self.get_parameter(component, *keys)
        return value if value is not None else default_value
    
    def update_parameter(self, component: str, value: Any, *keys):
        """
        Update a parameter in the configuration.
        
        Args:
            component: Component name
            value: New value for the parameter
            *keys: Nested keys to access the parameter
        """
        if component not in self._configs:
            self._configs[component] = {}
        
        config = self._configs[component]
        
        # Navigate to the nested dictionary
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Set the final value
        if keys:
            config[keys[-1]] = value
    
    def reload_config(self, component: Optional[str] = None):
        """
        Reload configuration from files.
        
        Args:
            component: Specific component to reload, or None for all
        """
        if component is None:
            self._load_configs()
            logger.info("Reloaded all configurations")
        else:
            # Reload specific component
            config_files = {
                'battery': 'battery_config.yaml',
                'pv': 'pv_config.yaml',
                'wind': 'wind_config.yaml',
                'genset': 'genset_config.yaml',
                'grid': 'grid_config.yaml',
                'dispatch': 'dispatch_config.yaml'
            }
            
            if component in config_files:
                filename = config_files[component]
                file_path = self.config_dir / filename
                if file_path.exists():
                    try:
                        with open(file_path, 'r') as f:
                            config = yaml.safe_load(f)
                            self._configs[component] = config[component]
                            logger.info(f"Reloaded {component} configuration")
                    except Exception as e:
                        logger.error(f"Error reloading {component} configuration: {e}")
    
    def save_config(self, component: str):
        """
        Save configuration for a component back to file.
        
        Args:
            component: Component name to save
        """
        config_files = {
            'battery': 'battery_config.yaml',
            'pv': 'pv_config.yaml',
            'wind': 'wind_config.yaml',
            'genset': 'genset_config.yaml',
            'grid': 'grid_config.yaml',
            'dispatch': 'dispatch_config.yaml'
        }
        
        if component in config_files:
            filename = config_files[component]
            file_path = self.config_dir / filename
            
            try:
                config_data = {component: self._configs[component]}
                with open(file_path, 'w') as f:
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
                logger.info(f"Saved {component} configuration to {filename}")
            except Exception as e:
                logger.error(f"Error saving {component} configuration: {e}")


# Global configuration manager instance
_config_manager = None


def load_config(config_dir: Optional[Union[str, Path]] = None) -> ConfigManager:
    """
    Load or get the global configuration manager.
    
    Args:
        config_dir: Directory containing configuration files.
                   If None, uses default directory.
        
    Returns:
        ConfigManager instance
    """
    global _config_manager
    
    if _config_manager is None or config_dir is not None:
        _config_manager = ConfigManager(config_dir)
    
    return _config_manager


def get_config(component: str) -> Dict[str, Any]:
    """
    Convenience function to get component configuration.
    
    Args:
        component: Component name
        
    Returns:
        Component configuration dictionary
    """
    return load_config().get_config(component)


def get_parameter(component: str, *keys) -> Any:
    """
    Convenience function to get a specific parameter.
    
    Args:
        component: Component name
        *keys: Nested keys to access the parameter
        
    Returns:
        Parameter value or None if not found
    """
    return load_config().get_parameter(component, *keys)


def get_parameter_with_default(component: str, default_value: Any, *keys) -> Any:
    """
    Convenience function to get a parameter with default value.
    
    Args:
        component: Component name
        default_value: Default value if parameter not found
        *keys: Nested keys to access the parameter
        
    Returns:
        Parameter value or default value
    """
    return load_config().get_parameter_with_default(component, default_value, *keys)