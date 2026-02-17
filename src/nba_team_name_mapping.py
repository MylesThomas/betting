"""
NBA Team Name Normalization Mapping

Maps The Odds API team names → ESPN team names.
ESPN is the source of truth (used for live game scores/status).

Key differences:
- Odds API: "Los Angeles Clippers"
- ESPN API: "LA Clippers"

All other 29 NBA teams use identical names.

Last verified: 2026-02-16
"""


# Complete mapping: Odds API → ESPN (all 30 NBA teams)
# 1 teams have different names, 29 are identical
ODDS_API_TO_ESPN_NBA = {
    # ============================================================================
    # TEAMS REQUIRING NORMALIZATION (1 teams)
    # ============================================================================
    "Los Angeles Clippers": "LA Clippers",

    # ============================================================================
    # TEAMS WITH IDENTICAL NAMES (29 teams)
    # ============================================================================
    "Atlanta Hawks": "Atlanta Hawks",
    "Boston Celtics": "Boston Celtics",
    "Brooklyn Nets": "Brooklyn Nets",
    "Charlotte Hornets": "Charlotte Hornets",
    "Chicago Bulls": "Chicago Bulls",
    "Cleveland Cavaliers": "Cleveland Cavaliers",
    "Dallas Mavericks": "Dallas Mavericks",
    "Denver Nuggets": "Denver Nuggets",
    "Detroit Pistons": "Detroit Pistons",
    "Golden State Warriors": "Golden State Warriors",
    "Houston Rockets": "Houston Rockets",
    "Indiana Pacers": "Indiana Pacers",
    "Los Angeles Lakers": "Los Angeles Lakers",
    "Memphis Grizzlies": "Memphis Grizzlies",
    "Miami Heat": "Miami Heat",
    "Milwaukee Bucks": "Milwaukee Bucks",
    "Minnesota Timberwolves": "Minnesota Timberwolves",
    "New Orleans Pelicans": "New Orleans Pelicans",
    "New York Knicks": "New York Knicks",
    "Oklahoma City Thunder": "Oklahoma City Thunder",
    "Orlando Magic": "Orlando Magic",
    "Philadelphia 76ers": "Philadelphia 76ers",
    "Phoenix Suns": "Phoenix Suns",
    "Portland Trail Blazers": "Portland Trail Blazers",
    "Sacramento Kings": "Sacramento Kings",
    "San Antonio Spurs": "San Antonio Spurs",
    "Toronto Raptors": "Toronto Raptors",
    "Utah Jazz": "Utah Jazz",
    "Washington Wizards": "Washington Wizards",
}


# Validation assertions (run at import time)
assert len(ODDS_API_TO_ESPN_NBA) == 30, \
    f"Expected 30 total NBA teams, got {len(ODDS_API_TO_ESPN_NBA)}"

# Count teams with different names (key != value)
differences_count = sum(1 for k, v in ODDS_API_TO_ESPN_NBA.items() if k != v)
assert differences_count == 1, \
    f"Expected 1 teams with different names, got {differences_count}"


def normalize_nba_team_name(odds_api_name: str) -> str:
    """
    Normalize NBA team name from The Odds API format to ESPN format.
    
    Args:
        odds_api_name: Team name from The Odds API
        
    Returns:
        Normalized team name matching ESPN format
        
    Examples:
        >>> normalize_nba_team_name("Los Angeles Clippers")
        'LA Clippers'
        >>> normalize_nba_team_name("Boston Celtics")
        'Boston Celtics'
    """
    # Check exact mapping first
    if odds_api_name in ODDS_API_TO_ESPN_NBA:
        return ODDS_API_TO_ESPN_NBA[odds_api_name]
    
    # If not in mapping, return as-is (most teams are identical)
    return odds_api_name
