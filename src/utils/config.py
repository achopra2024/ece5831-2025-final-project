"""
Configuration management utilities.
"""

import yaml
import os


def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to config file
    
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, save_path):
    """
    Save configuration to YAML file.
    
    Args:
        config (dict): Configuration dictionary
        save_path (str): Path to save config
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def update_config(config, updates):
    """
    Update configuration with new values.
    
    Args:
        config (dict): Original configuration
        updates (dict): Updates to apply
    
    Returns:
        dict: Updated configuration
    """
    config.update(updates)
    return config
