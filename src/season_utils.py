"""
Season utility functions for NBA and other sports.

This module provides:
- Season determination logic (current season based on date)
- Season validation functions
- Season format conversions

Used by:
- scripts/fetch_historical_nba_season_lines.py
- scripts/fetch_all_nba_shot_charts.py
- scripts/fetch_nba_player_props.py
- Any script that needs to determine the current season
"""

from datetime import datetime


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

