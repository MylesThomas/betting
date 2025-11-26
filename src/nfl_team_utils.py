"""
NFL Team Name Utilities

Handles mapping between different NFL team name formats:
- Full names (from Odds API): "Dallas Cowboys", "Philadelphia Eagles"
- Abbreviations (from Unexpected Points): "DAL", "PHI"
- City names: "Dallas", "Philadelphia"

Usage:
    from nfl_team_utils import full_name_to_abbr, abbr_to_full_name
    
    abbr = full_name_to_abbr("Dallas Cowboys")  # Returns "DAL"
    full = abbr_to_full_name("DAL")  # Returns "Dallas Cowboys"
"""

from typing import Dict, Optional
import pandas as pd


# ============================================================================
# NFL TEAM MAPPINGS
# ============================================================================

# Full name -> Abbreviation mapping
NFL_TEAM_MAPPING = {
    # AFC East
    "Buffalo Bills": "BUF",
    "Miami Dolphins": "MIA",
    "New England Patriots": "NE",
    "New York Jets": "NYJ",
    
    # AFC North
    "Baltimore Ravens": "BAL",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Pittsburgh Steelers": "PIT",
    
    # AFC South
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Tennessee Titans": "TEN",
    
    # AFC West
    "Denver Broncos": "DEN",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    
    # NFC East
    "Dallas Cowboys": "DAL",
    "New York Giants": "NYG",
    "Philadelphia Eagles": "PHI",
    "Washington Commanders": "WAS",
    
    # NFC North
    "Chicago Bears": "CHI",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Minnesota Vikings": "MIN",
    
    # NFC South
    "Atlanta Falcons": "ATL",
    "Carolina Panthers": "CAR",
    "New Orleans Saints": "NO",
    "Tampa Bay Buccaneers": "TB",
    
    # NFC West
    "Arizona Cardinals": "ARI",
    "Los Angeles Rams": "LAR",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
}

# Reverse mapping: Abbreviation -> Full name
NFL_ABBR_TO_FULL = {v: k for k, v in NFL_TEAM_MAPPING.items()}

# City/Nickname -> Full name (for flexible matching)
NFL_CITY_MAPPING = {
    # AFC East
    "Buffalo": "Buffalo Bills",
    "Miami": "Miami Dolphins",
    "New England": "New England Patriots",
    "New York Jets": "New York Jets",  # Need full for disambiguation
    
    # AFC North
    "Baltimore": "Baltimore Ravens",
    "Cincinnati": "Cincinnati Bengals",
    "Cleveland": "Cleveland Browns",
    "Pittsburgh": "Pittsburgh Steelers",
    
    # AFC South
    "Houston": "Houston Texans",
    "Indianapolis": "Indianapolis Colts",
    "Jacksonville": "Jacksonville Jaguars",
    "Tennessee": "Tennessee Titans",
    
    # AFC West
    "Denver": "Denver Broncos",
    "Kansas City": "Kansas City Chiefs",
    "Las Vegas": "Las Vegas Raiders",
    "Los Angeles Chargers": "Los Angeles Chargers",  # Need full for disambiguation
    
    # NFC East
    "Dallas": "Dallas Cowboys",
    "New York Giants": "New York Giants",  # Need full for disambiguation
    "Philadelphia": "Philadelphia Eagles",
    "Washington": "Washington Commanders",
    
    # NFC North
    "Chicago": "Chicago Bears",
    "Detroit": "Detroit Lions",
    "Green Bay": "Green Bay Packers",
    "Minnesota": "Minnesota Vikings",
    
    # NFC South
    "Atlanta": "Atlanta Falcons",
    "Carolina": "Carolina Panthers",
    "New Orleans": "New Orleans Saints",
    "Tampa Bay": "Tampa Bay Buccaneers",
    
    # NFC West
    "Arizona": "Arizona Cardinals",
    "Los Angeles Rams": "Los Angeles Rams",  # Need full for disambiguation
    "San Francisco": "San Francisco 49ers",
    "Seattle": "Seattle Seahawks",
}


# ============================================================================
# CONVERSION FUNCTIONS
# ============================================================================

def full_name_to_abbr(full_name: str) -> Optional[str]:
    """
    Convert full team name to abbreviation.
    
    Args:
        full_name: Full team name (e.g., "Dallas Cowboys")
        
    Returns:
        Team abbreviation (e.g., "DAL") or None if not found
        
    Example:
        >>> full_name_to_abbr("Dallas Cowboys")
        'DAL'
    """
    return NFL_TEAM_MAPPING.get(full_name)


def abbr_to_full_name(abbr: str) -> Optional[str]:
    """
    Convert team abbreviation to full name.
    
    Args:
        abbr: Team abbreviation (e.g., "DAL")
        
    Returns:
        Full team name (e.g., "Dallas Cowboys") or None if not found
        
    Example:
        >>> abbr_to_full_name("DAL")
        'Dallas Cowboys'
    """
    return NFL_ABBR_TO_FULL.get(abbr)


def add_team_abbr_columns(df: pd.DataFrame, 
                          away_col: str = 'away_team', 
                          home_col: str = 'home_team') -> pd.DataFrame:
    """
    Add team abbreviation columns to odds dataframe.
    
    Args:
        df: DataFrame with full team names
        away_col: Column name for away team (default: 'away_team')
        home_col: Column name for home team (default: 'home_team')
        
    Returns:
        DataFrame with added 'away_abbr' and 'home_abbr' columns
        
    Example:
        >>> odds_df = pd.read_csv('nfl_game_lines_2025-09-04.csv')
        >>> odds_df = add_team_abbr_columns(odds_df)
        >>> odds_df[['away_team', 'away_abbr', 'home_team', 'home_abbr']]
    """
    df = df.copy()
    
    # Add abbreviation columns
    df['away_abbr'] = df[away_col].map(full_name_to_abbr)
    df['home_abbr'] = df[home_col].map(full_name_to_abbr)
    
    # Check for unmapped teams
    unmapped_away = df[df['away_abbr'].isna()][away_col].unique()
    unmapped_home = df[df['home_abbr'].isna()][home_col].unique()
    
    if len(unmapped_away) > 0:
        print(f"⚠️  Unmapped away teams: {unmapped_away.tolist()}")
    if len(unmapped_home) > 0:
        print(f"⚠️  Unmapped home teams: {unmapped_home.tolist()}")
    
    return df


def get_all_abbrs() -> list:
    """Get list of all team abbreviations."""
    return sorted(NFL_ABBR_TO_FULL.keys())


def get_all_full_names() -> list:
    """Get list of all full team names."""
    return sorted(NFL_TEAM_MAPPING.keys())


# ============================================================================
# DATA SOURCE SPECIFIC MAPPINGS
# ============================================================================

def normalize_unexpected_points_abbr(abbr: str) -> str:
    """
    Convert Unexpected Points abbreviation to canonical abbreviation.
    
    Unexpected Points uses "LA" for Rams, we use "LAR".
    
    Args:
        abbr: Team abbreviation from Unexpected Points data
        
    Returns:
        Canonical team abbreviation
        
    Example:
        >>> normalize_unexpected_points_abbr("LA")
        'LAR'
        >>> normalize_unexpected_points_abbr("KC")
        'KC'
    """
    # Mapping for Unexpected Points quirks
    UP_TO_CANONICAL = {
        'LA': 'LAR',  # Unexpected Points uses "LA", we use "LAR", so we need to normalize it to "LAR"
    }
    
    return UP_TO_CANONICAL.get(abbr, abbr)


def validate_mappings():
    """Validate that all mappings are consistent and complete."""
    print("=" * 80)
    print("NFL TEAM MAPPING VALIDATION")
    print("=" * 80)
    
    print(f"\nTotal teams: {len(NFL_TEAM_MAPPING)}")
    print(f"Expected: 32 NFL teams")
    
    if len(NFL_TEAM_MAPPING) != 32:
        print("❌ WARNING: Should have exactly 32 teams!")
    else:
        print("✅ Correct number of teams")
    
    # Check reverse mapping
    print(f"\nReverse mapping count: {len(NFL_ABBR_TO_FULL)}")
    if len(NFL_ABBR_TO_FULL) != len(NFL_TEAM_MAPPING):
        print("❌ WARNING: Reverse mapping mismatch!")
    else:
        print("✅ Reverse mapping complete")
    
    # Check for duplicate abbreviations
    abbrs = list(NFL_TEAM_MAPPING.values())
    if len(abbrs) != len(set(abbrs)):
        print("❌ WARNING: Duplicate abbreviations found!")
        from collections import Counter
        duplicates = [abbr for abbr, count in Counter(abbrs).items() if count > 1]
        print(f"   Duplicates: {duplicates}")
    else:
        print("✅ No duplicate abbreviations")
    
    # List all teams
    print("\n" + "=" * 80)
    print("ALL TEAMS")
    print("=" * 80)
    print(f"\n{'Full Name':<35s} {'Abbr':<6s}")
    print("-" * 41)
    for full, abbr in sorted(NFL_TEAM_MAPPING.items()):
        print(f"{full:<35s} {abbr:<6s}")
    
    print("\n" + "=" * 80)


# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    # Validate mappings
    validate_mappings()
    
    # Test conversions
    print("\n" + "=" * 80)
    print("CONVERSION TESTS")
    print("=" * 80)
    
    test_cases = [
        ("Dallas Cowboys", "DAL"),
        ("Philadelphia Eagles", "PHI"),
        ("Kansas City Chiefs", "KC"),
        ("San Francisco 49ers", "SF"),
    ]
    
    print("\nFull name -> Abbreviation:")
    for full, expected_abbr in test_cases:
        result = full_name_to_abbr(full)
        status = "✅" if result == expected_abbr else "❌"
        print(f"  {status} {full:<30s} -> {result}")
    
    print("\nAbbreviation -> Full name:")
    for expected_full, abbr in test_cases:
        result = abbr_to_full_name(abbr)
        status = "✅" if result == expected_full else "❌"
        print(f"  {status} {abbr:<6s} -> {result}")
    
    print("\n" + "=" * 80)
    print("✅ NFL team utilities ready!")
    print("=" * 80)

