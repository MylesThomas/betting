"""
Team Code Normalization for NBA Historical Data.

PROBLEM:
--------
The NBA API returns historical team abbreviations for games played before team
relocations, rebranding, or abbreviation changes. These old codes need to be
normalized to current team abbreviations for consistency.

EXAMPLES:
---------
- Tim Hardaway played for "GOS" (Golden State Warriors old code) in 1989-1996
- Kevin Porter played for "BLT" (Baltimore Bullets) which became Washington
- Games from Seattle SuperSonics show as "SEA" but team moved to OKC

WHERE THESE CODES COME FROM:
----------------------------
The NBA API (`nba_api.stats.endpoints.playergamelog`) returns the MATCHUP field
with the team code that was used AT THE TIME the game was played.

For example:
- Tim Hardaway (Sr.)'s 1989 game: MATCHUP = "GOS vs. LAL" (not "GSW vs. LAL")
- Kevin Durant's 2007 game: MATCHUP = "SEA @ LAL" (not "OKC @ LAL")
- Brook Lopez's 2008 game: MATCHUP = "NJN vs. BOS" (not "BKN vs. BOS")

This is historically accurate but inconsistent for modern analysis where we want
all Golden State games to be "GSW", all OKC games to be "OKC", etc.

SOLUTION:
---------
This module provides `normalize_team_code()` to map historical abbreviations
to current team codes.

Author: Myles Thomas
Date: 2025-02-05
"""

import pandas as pd
from typing import Optional


def normalize_team_code(team_code: str) -> Optional[str]:
    """
    Normalize historical NBA team codes to current abbreviations.
    
    The NBA API returns team abbreviations that were valid at the time the game
    was played. This function maps those historical codes to the current team.
    
    Args:
        team_code: Team abbreviation from NBA API (may be historical)
        
    Returns:
        Current team abbreviation, or original code if no mapping exists
        
    Examples:
        >>> normalize_team_code('GOS')
        'GSW'
        
        >>> normalize_team_code('BLT')
        'WAS'
        
        >>> normalize_team_code('LAL')
        'LAL'  # No change for current teams
        
    Team History Reference:
        - Golden State: GOS (1962-1971) → GSW (1971-present)
        - Philadelphia: PHL (old) → PHI (current)
        - Washington: BLT (1963-1973) → CAP (1973-1974) → WAS (1974-present)
        - Charlotte: CHH (1988-2002) → CHA (2014-present)
        - Memphis: VAN (1995-2001) → MEM (2001-present)
        - Oklahoma City: SEA (1967-2008) → OKC (2008-present)
        - Brooklyn: NJN (1977-2012) → BKN (2012-present)
        - New Orleans: NOH/NOK (various) → NOP (2013-present)
    """
    if pd.isna(team_code):
        return None
    
    # Historical team code mappings: old abbreviation → current abbreviation
    historical_mappings = {
        # Golden State Warriors
        'GOS': 'GSW',  # Golden State Warriors (1962-1971 used "GOS")
        
        # Philadelphia 76ers
        'PHL': 'PHI',  # Philadelphia 76ers (old abbreviation)
        
        # Washington Wizards (complex history)
        'BLT': 'WAS',  # Baltimore Bullets (1963-1973)
        'CAP': 'WAS',  # Capital Bullets (1973-1974)
        'WSB': 'WAS',  # Washington Bullets (1974-1997)
        
        # Charlotte Hornets
        'CHH': 'CHA',  # Charlotte Hornets original (1988-2002, now Charlotte again)
        
        # Memphis Grizzlies
        'VAN': 'MEM',  # Vancouver Grizzlies (1995-2001)
        
        # Oklahoma City Thunder
        'SEA': 'OKC',  # Seattle SuperSonics (1967-2008)
        
        # Brooklyn Nets
        'NJN': 'BKN',  # New Jersey Nets (1977-2012)
        
        # New Orleans Pelicans
        'NOH': 'NOP',  # New Orleans Hornets (various periods)
        'NOK': 'NOP',  # New Orleans/Oklahoma City Hornets (2005-2007 post-Katrina)
        
        # Los Angeles Clippers
        'SDC': 'LAC',  # San Diego Clippers (1978-1984)
        
        # Sacramento Kings (complex history)
        'CIN': 'SAC',  # Cincinnati Royals (1957-1972)
        'KCK': 'SAC',  # Kansas City Kings (1972-1985)
        
        # Atlanta Hawks
        'STL': 'ATL',  # St. Louis Hawks (1955-1968)
        
        # Utah Jazz
        'NOJ': 'UTA',  # New Orleans Jazz (1974-1979)
    }
    
    return historical_mappings.get(team_code, team_code)


def get_all_current_team_codes() -> set:
    """
    Get set of all current NBA team codes (30 teams).
    
    Returns:
        Set of current 3-letter team abbreviations
        
    Note:
        This is the definitive list of valid modern team codes.
        Any code not in this list is either:
        - Historical (needs normalization)
        - Invalid/garbage data
    """
    return {
        'ATL',  # Atlanta Hawks
        'BKN',  # Brooklyn Nets
        'BOS',  # Boston Celtics
        'CHA',  # Charlotte Hornets
        'CHI',  # Chicago Bulls
        'CLE',  # Cleveland Cavaliers
        'DAL',  # Dallas Mavericks
        'DEN',  # Denver Nuggets
        'DET',  # Detroit Pistons
        'GSW',  # Golden State Warriors
        'HOU',  # Houston Rockets
        'IND',  # Indiana Pacers
        'LAC',  # Los Angeles Clippers
        'LAL',  # Los Angeles Lakers
        'MEM',  # Memphis Grizzlies
        'MIA',  # Miami Heat
        'MIL',  # Milwaukee Bucks
        'MIN',  # Minnesota Timberwolves
        'NOP',  # New Orleans Pelicans
        'NYK',  # New York Knicks
        'OKC',  # Oklahoma City Thunder
        'ORL',  # Orlando Magic
        'PHI',  # Philadelphia 76ers
        'PHX',  # Phoenix Suns
        'POR',  # Portland Trail Blazers
        'SAC',  # Sacramento Kings
        'SAS',  # San Antonio Spurs
        'TOR',  # Toronto Raptors
        'UTA',  # Utah Jazz
        'WAS',  # Washington Wizards
    }


def is_valid_current_team_code(team_code: str) -> bool:
    """
    Check if a team code is a valid current NBA team.
    
    Args:
        team_code: 3-letter team abbreviation
        
    Returns:
        True if code is a current NBA team, False otherwise
        
    Example:
        >>> is_valid_current_team_code('LAL')
        True
        
        >>> is_valid_current_team_code('GOS')
        False  # Historical code
    """
    if pd.isna(team_code):
        return False
    return team_code in get_all_current_team_codes()


# =============================================================================
# TESTING
# =============================================================================

if __name__ == '__main__':
    """Test team normalization."""
    print("="*80)
    print("Testing Team Code Normalization")
    print("="*80)
    print()
    
    # Test historical codes
    test_cases = [
        ('GOS', 'GSW', 'Golden State Warriors (old)'),
        ('PHL', 'PHI', 'Philadelphia 76ers (old)'),
        ('BLT', 'WAS', 'Baltimore Bullets'),
        ('CAP', 'WAS', 'Capital Bullets'),
        ('SEA', 'OKC', 'Seattle SuperSonics'),
        ('NJN', 'BKN', 'New Jersey Nets'),
        ('VAN', 'MEM', 'Vancouver Grizzlies'),
        ('NOH', 'NOP', 'New Orleans Hornets'),
        ('LAL', 'LAL', 'Los Angeles Lakers (no change)'),
    ]
    
    print("Historical Code Mappings:")
    print()
    for old_code, expected, description in test_cases:
        result = normalize_team_code(old_code)
        status = '✅' if result == expected else '❌'
        print(f"  {old_code} → {result:3} {status}  ({description})")
    
    print()
    print(f"Total current NBA teams: {len(get_all_current_team_codes())}")
    print()
    
    # Test validation
    print("Validation Tests:")
    print(f"  is_valid_current_team_code('LAL'): {is_valid_current_team_code('LAL')}")
    print(f"  is_valid_current_team_code('GOS'): {is_valid_current_team_code('GOS')}")
    print(f"  is_valid_current_team_code('XXX'): {is_valid_current_team_code('XXX')}")
