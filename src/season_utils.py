"""
Season utility functions for NBA and other sports.

This module provides:
- Season determination logic (current season based on date)
- Season validation functions
- Season format conversions
- Season dates (start, playoff start, playoff end) from centralized config

Used by:
- scripts/fetch_historical_nba_season_lines.py
- scripts/fetch_all_nba_shot_charts.py
- scripts/fetch_nba_player_props.py
- tmp/analyze_ats_records.py
- Any script that needs to determine the current season or season dates
"""

from datetime import datetime
from pathlib import Path
import yaml


def _load_season_dates_config():
    """Load season dates from config/season_dates.yaml"""
    config_path = Path(__file__).parent.parent / 'config' / 'season_dates.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_season_dates(sport, season):
    """
    Get season dates from centralized config.
    
    Args:
        sport: Sport name ('nba', 'nfl', 'ncaab', 'ncaaf')
        season: Season string (e.g., '2024-25' for NBA/NCAAB, '2024' for NFL/NCAAF)
    
    Returns:
        dict with keys: season_start, regular_season_end, playoff_start, playoff_end
        (or tournament_start/tournament_end for NCAAB)
    
    Examples:
        >>> get_season_dates('nba', '2024-25')
        {'season_start': '2024-10-22', 'regular_season_end': '2025-04-13', 
         'playoff_start': '2025-04-15', 'playoff_end': '2025-06-22'}
        
        >>> get_season_dates('nfl', '2024')
        {'season_start': '2024-09-05', 'regular_season_end': '2025-01-05',
         'playoff_start': '2025-01-11', 'playoff_end': '2025-02-09'}
    """
    config = _load_season_dates_config()
    
    sport = sport.lower()
    if sport not in config:
        raise ValueError(f"Sport '{sport}' not found in season_dates.yaml. Available: {list(config.keys())}")
    
    if season not in config[sport]:
        raise ValueError(f"Season '{season}' not found for {sport}. Available: {list(config[sport].keys())}")
    
    return config[sport][season]


def get_playoff_start_date(sport, season):
    """
    Get playoff start date for a given sport and season.
    
    Args:
        sport: Sport name ('nba', 'nfl', 'ncaab', 'ncaaf')
        season: Season string
    
    Returns:
        str: Playoff start date in 'YYYY-MM-DD' format
        (or tournament_start for NCAAB)
    
    Example:
        >>> get_playoff_start_date('nba', '2024-25')
        '2025-04-15'
    """
    dates = get_season_dates(sport, season)
    
    # NCAAB uses 'tournament_start' instead of 'playoff_start'
    if sport == 'ncaab':
        return dates['tournament_start']
    
    return dates['playoff_start']


def get_current_nba_season():
    """
    Get the current NBA season string based on today's date.
    
    NBA seasons run from October to June of the following year.
    - Oct-Dec: Current year is the start year (e.g., Oct 2025 → '2025-26')
    - Jan-Sep: Previous year is the start year (e.g., Jan 2026 → '2025-26')
    
    Returns:
        str: Season string in format 'YYYY-YY' (e.g., '2025-26')
    
    Examples:
        >>> # If today is Oct 15, 2025
        >>> get_current_nba_season()
        '2025-26'
        
        >>> # If today is Jan 10, 2026
        >>> get_current_nba_season()
        '2025-26'
        
        >>> # If today is June 20, 2026
        >>> get_current_nba_season()
        '2025-26'
    """
    today = datetime.now()
    if today.month >= 10:  # Oct-Dec
        return f"{today.year}-{str(today.year + 1)[-2:]}"
    else:  # Jan-Sep
        return f"{today.year - 1}-{str(today.year)[-2:]}"


def parse_season_to_years(season):
    """
    Parse season string to start and end years.
    
    Args:
        season: Season string (e.g., '2025-26')
    
    Returns:
        tuple: (start_year: int, end_year: int)
    
    Example:
        >>> parse_season_to_years('2025-26')
        (2025, 2026)
    """
    if '-' not in season:
        raise ValueError(f"Invalid season format: {season}. Expected 'YYYY-YY'")
    
    start_year_str, end_year_suffix = season.split('-')
    start_year = int(start_year_str)
    end_year = int(f"{start_year_str[:2]}{end_year_suffix}")
    
    return start_year, end_year


def season_to_underscore(season):
    """
    Convert season from dash format to underscore format.
    
    Args:
        season: Season string (e.g., '2025-26')
    
    Returns:
        str: Season with underscore (e.g., '2025_26')
    
    Example:
        >>> season_to_underscore('2025-26')
        '2025_26'
    """
    return season.replace('-', '_')


def season_to_dash(season):
    """
    Convert season from underscore format to dash format.
    
    Args:
        season: Season string (e.g., '2025_26')
    
    Returns:
        str: Season with dash (e.g., '2025-26')
    
    Example:
        >>> season_to_dash('2025_26')
        '2025-26'
    """
    return season.replace('_', '-')

