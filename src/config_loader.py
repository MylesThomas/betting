"""
Configuration loader for betting project.

Loads paths from config/config.yaml to ensure consistency across scripts.

Usage:
    from config_loader import get_config, get_data_path, get_file_path
    
    # Get full config
    config = get_config()
    
    # Get specific paths
    arbs_dir = get_data_path('output_arbs')  # data/03_output/arbs
    roster_file = get_file_path('nba_full_roster_cache')  # data/02_intermediate/nba_full_roster_cache.csv
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def get_project_root() -> Path:
    """Find project root by locating config directory."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / 'config' / 'config.yaml').exists():
            return parent
    raise FileNotFoundError("Could not find project root (no config/config.yaml found)")


def get_config() -> Dict[str, Any]:
    """
    Load configuration from config.yaml.
    
    Returns:
        Dictionary with config data
    """
    config_path = get_project_root() / 'config' / 'config.yaml'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_data_path(key: str, absolute: bool = False) -> Path:
    """
    Get a data directory path from config.
    
    Args:
        key: Key from data_directories section (e.g., 'output_arbs', 'cache')
        absolute: If True, return absolute path. If False, return relative to project root.
        
    Returns:
        Path object
        
    Example:
        >>> get_data_path('output_arbs')
        PosixPath('data/03_output/arbs')
        
        >>> get_data_path('output_arbs', absolute=True)
        PosixPath('/Users/thomasmyles/dev/betting/data/03_output/arbs')
    """
    config = get_config()
    relative_path = config['data_directories'].get(key)
    
    if not relative_path:
        raise KeyError(f"'{key}' not found in data_directories config")
    
    if absolute:
        return get_project_root() / relative_path
    else:
        return Path(relative_path)


def get_file_path(key: str, absolute: bool = False) -> Path:
    """
    Get a file path from config.
    
    Args:
        key: Key from files section (e.g., 'nba_full_roster_cache')
        absolute: If True, return absolute path. If False, return relative to project root.
        
    Returns:
        Path object
        
    Example:
        >>> get_file_path('nba_full_roster_cache')
        PosixPath('data/02_intermediate/nba_full_roster_cache.csv')
        
        >>> get_file_path('nba_full_roster_cache', absolute=True)
        PosixPath('/Users/thomasmyles/dev/betting/data/02_intermediate/nba_full_roster_cache.csv')
    """
    config = get_config()
    relative_path = config['files'].get(key)
    
    if not relative_path:
        raise KeyError(f"'{key}' not found in files config")
    
    if absolute:
        return get_project_root() / relative_path
    else:
        return Path(relative_path)


def ensure_dir_exists(key: str) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        key: Key from data_directories section
        
    Returns:
        Absolute path to the directory
    """
    dir_path = get_data_path(key, absolute=True)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


if __name__ == '__main__':
    # Test the config loader
    print("Testing config loader...")
    print()
    
    config = get_config()
    print(f"✓ Loaded config")
    print(f"  Project root: {get_project_root()}")
    print()
    
    print("Data directories:")
    for key in ['input', 'intermediate', 'output', 'temp', 'output_arbs']:
        try:
            path = get_data_path(key)
            print(f"  {key}: {path}")
        except KeyError:
            pass
    print()
    
    print("File paths:")
    for key in ['nba_full_roster_cache', 'player_team_cache', 'nba_calendar_all_games']:
        try:
            path = get_file_path(key)
            print(f"  {key}: {path}")
        except KeyError:
            pass

