"""Configuration utility for loading config.yaml."""
import os
import yaml
import re

_CONFIG_CACHE = None

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'config.yaml')

def _resolve_env_vars(value):
    """Resolve environment variables in a string value."""
    if isinstance(value, str):
        # Replace ${VAR_NAME} with actual environment variable values
        pattern = r'\$\{([^}]+)\}'
        def replace_env_var(match):
            env_var = match.group(1)
            return os.getenv(env_var, '')
        return re.sub(pattern, replace_env_var, value)
    return value

def _resolve_env_vars_recursive(obj):
    """Recursively resolve environment variables in nested objects."""
    if isinstance(obj, dict):
        return {key: _resolve_env_vars_recursive(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_env_vars_recursive(item) for item in obj]
    else:
        return _resolve_env_vars(obj)

def get_config():
    """Load and cache configuration from config.yaml."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        # Resolve environment variables
        _CONFIG_CACHE = _resolve_env_vars_recursive(config)
        return _CONFIG_CACHE
    except Exception as e:
        raise RuntimeError(f"Failed to load config.yaml: {e}") 