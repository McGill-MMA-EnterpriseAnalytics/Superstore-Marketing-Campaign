import os
from pathlib import Path

import yaml


def load_config(config_path=None):
    """
    Load the configuration from a YAML file.
    
    Parameters:
        config_path (str, optional): Path to the configuration file. If None, uses CONFIG_PATH env var or defaults to config.yaml.
        
    Returns:
        dict: Configuration dictionary.
    """
    # First check environment variable, then use provided path or default
    if config_path is None:
        config_path = os.environ.get("CONFIG_PATH", "config.yaml")
    
    # Get the absolute path to ensure file is found regardless of where code is executed from
    config_file = Path(config_path).resolve()

    # Check if configuration file exists
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_file}")

    # Load the YAML file
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    return config


def get_data_paths(config=None):
    """
    Get data paths from the configuration.
    
    Parameters:
        config (dict, optional): Configuration dictionary. If None, loads from default path.
        
    Returns:
        dict: Data paths.
    """
    if config is None:
        config = load_config()

    return config.get("data", {})


def get_model_config(model_name, tuned=False, config=None):
    """
    Get configuration for a specific model.
    
    Parameters:
        model_name (str): Name of the model (e.g., 'xgboost', 'lightgbm', 'catboost').
        tuned (bool): Whether to get the tuned configuration (True) or baseline (False).
        config (dict, optional): Configuration dictionary. If None, loads from default path.
        
    Returns:
        dict: Model configuration.
    """
    if config is None:
        config = load_config()

    model_type = "tuned" if tuned else "baseline"

    try:
        return config["models"][model_name][model_type]
    except KeyError:
        available_models = list(config.get("models", {}).keys())
        raise ValueError(f"Model '{model_name}' not found. Available models: {available_models}")


def get_preprocessing_config(config=None):
    """
    Get preprocessing configuration.
    
    Parameters:
        config (dict, optional): Configuration dictionary. If None, loads from default path.
        
    Returns:
        dict: Preprocessing configuration.
    """
    if config is None:
        config = load_config()

    return config.get("preprocessing", {})


def get_training_config(config=None):
    """
    Get training configuration.
    
    Parameters:
        config (dict, optional): Configuration dictionary. If None, loads from default path.
        
    Returns:
        dict: Training configuration.
    """
    if config is None:
        config = load_config()

    return config.get("training", {})


def get_paths(config=None):
    """
    Get paths for saving artifacts.
    
    Parameters:
        config (dict, optional): Configuration dictionary. If None, loads from default path.
        
    Returns:
        dict: Paths for saving artifacts.
    """
    if config is None:
        config = load_config()

    return config.get("paths", {})


def create_directories():
    """
    Create directories for saving artifacts based on the configuration.
    """
    paths = get_paths()

    for path_name, path_value in paths.items():
        os.makedirs(path_value, exist_ok=True)
