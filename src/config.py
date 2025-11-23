"""
Global configuration for NBA betting analysis.

This file contains shared constants and settings used across the project.
"""

from datetime import datetime


def get_current_nba_season() -> str:
    """
    Get the current NBA season string based on today's date.
    
    NBA seasons run from October to June, so:
    - Oct-Dec: Use current year (e.g., Oct 2025 = "2025-26")
    - Jan-Sep: Use previous year (e.g., Jan 2026 = "2025-26")
    
    Returns:
        Season string in format "YYYY-YY" (e.g., "2025-26")
    """
    today = datetime.now()
    year = today.year
    month = today.month
    
    # If we're in Jan-Sep, the season started last year
    if month < 10:  # Jan-Sep
        start_year = year - 1
    else:  # Oct-Dec
        start_year = year
    
    end_year = start_year + 1
    
    return f"{start_year}-{str(end_year)[-2:]}"


# Current NBA season (auto-calculated)
CURRENT_NBA_SEASON = get_current_nba_season()

# NBA team abbreviations (30 teams)
NBA_TEAMS = {
    'ATL': 'Atlanta Hawks',
    'BOS': 'Boston Celtics',
    'BKN': 'Brooklyn Nets',
    'CHA': 'Charlotte Hornets',
    'CHI': 'Chicago Bulls',
    'CLE': 'Cleveland Cavaliers',
    'DAL': 'Dallas Mavericks',
    'DEN': 'Denver Nuggets',
    'DET': 'Detroit Pistons',
    'GSW': 'Golden State Warriors',
    'HOU': 'Houston Rockets',
    'IND': 'Indiana Pacers',
    'LAC': 'Los Angeles Clippers',
    'LAL': 'Los Angeles Lakers',
    'MEM': 'Memphis Grizzlies',
    'MIA': 'Miami Heat',
    'MIL': 'Milwaukee Bucks',
    'MIN': 'Minnesota Timberwolves',
    'NOP': 'New Orleans Pelicans',
    'NYK': 'New York Knicks',
    'OKC': 'Oklahoma City Thunder',
    'ORL': 'Orlando Magic',
    'PHI': 'Philadelphia 76ers',
    'PHX': 'Phoenix Suns',
    'POR': 'Portland Trail Blazers',
    'SAC': 'Sacramento Kings',
    'SAS': 'San Antonio Spurs',
    'TOR': 'Toronto Raptors',
    'UTA': 'Utah Jazz',
    'WAS': 'Washington Wizards',
}

# Cache settings
# Player-team cache: 7 days (trades don't happen daily)
# Validation logic in Streamlit app will trigger API refresh if team doesn't match game
PLAYER_TEAM_CACHE_MAX_AGE_HOURS = 168  # 7 days
FULL_ROSTER_CACHE_MAX_AGE_DAYS = 7


if __name__ == '__main__':
    print("NBA Betting Configuration")
    print("=" * 50)
    print(f"Current NBA Season: {CURRENT_NBA_SEASON}")
    print(f"Total NBA Teams: {len(NBA_TEAMS)}")
    print(f"Cache Max Age: {PLAYER_TEAM_CACHE_MAX_AGE_HOURS} hours")
    print(f"Roster Cache Max Age: {FULL_ROSTER_CACHE_MAX_AGE_DAYS} days")

