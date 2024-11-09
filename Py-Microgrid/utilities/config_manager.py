"""
Configuration management utilities for HOPP.
Provides safe YAML configuration file handling with backup and error checking.
"""

import os
import yaml
from typing import Dict

class ConfigManager:
    """Handles YAML configuration file operations with safety measures."""
    
    @staticmethod
    def load_yaml_safely(file_path: str) -> dict:
        """
        Load YAML configuration file with error handling.
        
        Args:
            file_path: Path to the YAML configuration file.
            
        Returns:
            dict: Loaded configuration dictionary.
            
        Raises:
            ValueError: If the YAML file is empty or invalid.
            FileNotFoundError: If the file doesn't exist.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found at: {file_path}")
            
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
        
        if config is None:
            raise ValueError(f"The YAML configuration file at {file_path} is empty or invalid.")
        
        return config
    
    @staticmethod
    def save_yaml_safely(config: Dict, file_path: str) -> None:
        """
        Save YAML configuration file with backup and error handling.
        
        Args:
            config: Configuration dictionary to save.
            file_path: Path where to save the YAML file.
            
        Raises:
            IOError: If writing to the file fails.
        """
        backup_path = file_path + '.bak'
        if os.path.exists(file_path):
            os.rename(file_path, backup_path)
        
        try:
            with open(file_path, 'w') as file:
                yaml.safe_dump(config, file, default_flow_style=False, sort_keys=False)
            
            if os.path.getsize(file_path) == 0:
                raise IOError("Failed to write data to YAML file")
            
            if os.path.exists(backup_path):
                os.remove(backup_path)
                
        except Exception as e:
            if os.path.exists(backup_path):
                os.rename(backup_path, file_path)
            raise IOError(f"Error saving YAML file: {str(e)}")