"""
Global configuration for betting analysis.

This file contains shared constants and settings used across the project.
"""

from datetime import datetime
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================

DATA_ROOT = Path("/Users/thomasmyles/dev/betting/data")

# =============================================================================
# NBA CONFIG
# =============================================================================

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
    
    if month < 10:  # Jan-Sep
        start_year = year - 1
    else:  # Oct-Dec
        start_year = year
    
    end_year = start_year + 1
    return f"{start_year}-{str(end_year)[-2:]}"


CURRENT_NBA_SEASON = get_current_nba_season()

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
PLAYER_TEAM_CACHE_MAX_AGE_HOURS = 168  # 7 days
FULL_ROSTER_CACHE_MAX_AGE_DAYS = 7


# =============================================================================
# NFL CONFIG
# =============================================================================

NFL_CURRENT_SEASON = 2025

# Luck threshold for categorizing teams as Lucky/Neutral/Unlucky
# Lucky: luck >= +threshold, Unlucky: luck <= -threshold
NFL_LUCK_THRESHOLD_DEFAULT = 3

# 2025 bye weeks (verified from nfl_luck_utils.py)
NFL_2025_BYE_WEEKS = {
    'ATL': 5, 'CHI': 5, 'GB': 5, 'PIT': 5,
    'HOU': 6, 'MIN': 6,
    'BAL': 7, 'BUF': 7,
    'ARI': 8, 'DET': 8, 'JAX': 8, 'LV': 8, 'LAR': 8, 'SEA': 8,
    'CLE': 9, 'NYJ': 9, 'PHI': 9, 'TB': 9,
    'CIN': 10, 'DAL': 10, 'KC': 10, 'TEN': 10,
    'IND': 11, 'NO': 11,
    'DEN': 12, 'LAC': 12, 'MIA': 12, 'WAS': 12,
    'CAR': 14, 'NE': 14, 'NYG': 14, 'SF': 14,
}


# =============================================================================
# EMOJI MAP
# =============================================================================

EMOJI = {
    # Status
    'success': '✅',
    'error': '❌',
    'warning': '⚠️',
    'info': 'ℹ️',
    'refresh': '🔄',
    'save': '💾',
    
    # Analysis
    'chart': '📊',
    'target': '🎯',
    'calendar': '📅',
    'star': '⭐',
    'money': '💰',
    'up': '📈',
    'down': '📉',
    
    # Luck categories
    'lucky': '🍀',
    'unlucky': '💔',
    'neutral': '😐',
    
    # Sports
    'nba': '🏀',
    'nfl': '🏈',
    
    # Test/Debug
    'test': '🧪',
}


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    print("=" * 50)
    print("BETTING CONFIG")
    print("=" * 50)
    
    print(f"\n{EMOJI['nba']} NBA:")
    print(f"   Season: {CURRENT_NBA_SEASON}")
    print(f"   Teams: {len(NBA_TEAMS)}")
    print(f"   Cache: {PLAYER_TEAM_CACHE_MAX_AGE_HOURS}h")
    
    print(f"\n{EMOJI['nfl']} NFL:")
    print(f"   Season: {NFL_CURRENT_SEASON}")
    print(f"   Luck Threshold: ±{NFL_LUCK_THRESHOLD_DEFAULT}")
    print(f"   Bye Weeks: {len(NFL_2025_BYE_WEEKS)} teams configured")
