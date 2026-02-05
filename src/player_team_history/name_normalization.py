"""
Player Name Normalization for Team History Module.

SOURCE OF TRUTH:
----------------
**NBA API (nba_api library) is the canonical source.**

All player names from other sources (Odds API, ESPN API) are normalized 
TO MATCH the names in NBA API.

WHY NBA API?
- Official NBA data source
- Used for fetching game logs and team history

PROBLEM:
--------
Player names come from 3 different APIs with inconsistent formatting:

1. **THE ODDS API** (S3 props data - LOW QUALITY):
   - All caps: "AARON NESMITH"
   - Full legal names: "Alfred Joel Horford Reynoso" 
   - Suffix variations: "Anthony Davis Jr." vs "Anthony Davis"
   - Non-players leak in: "Atl Hawks Alternate Total"
   - Reversed names: "Caldwell Pope Kentavious"
   - Typos/abbreviations: "Mil Bridges" (Miles Bridges)
   
2. **NBA API** (nba_api library - SOURCE OF TRUTH ✅):
   - Title case: "Aaron Nesmith"
   - Consistent nicknames: "Al Horford" (not Alfred)
   - Consistent suffix handling: removes Jr/Sr unless both played
   - Clean data, no garbage
   - **THIS IS OUR CANONICAL FORMAT**

3. **ESPN API** (future support):
   - Title case: "Aaron Nesmith"
   - Mixed nickname/full name usage
   - Accents sometimes included: "Luka Dončić"
   - Generally reliable but less consistent than NBA API

SOLUTION:
---------
This module provides a single normalize_player_name() function that:
1. Filters out garbage data (team names, betting lines, malformed names)
2. Fixes common data quality issues (reversed names, typos)
3. Converts all names to consistent format (Title Case, no periods, no accents)
4. Removes Jr/Sr suffixes (except for exceptions where both Sr and Jr played)
5. Applies known mappings to match NBA API canonical names
6. Returns a canonical name that matches NBA API format

ARCHITECTURE:
-------------
**2-Step Normalization Process:**

Step 1: BASIC NORMALIZATION (applies to ALL sources)
- Remove periods, accents, extra whitespace
- Convert to Title Case
- Remove Jr/Sr suffixes (with exceptions)
- Filter garbage data

Step 2: SOURCE-SPECIFIC MAPPINGS (handle edge cases)
- Odds API → NBA API mappings (full legal names, typos, etc.)
- ESPN API → NBA API mappings (nickname variations, etc.)
- NBA API → no mappings (it's the source of truth)

USAGE:
------
    from src.player_team_history.name_normalization import (
        normalize_from_odds_api,
        normalize_from_espn_api,
        normalize_from_nba_api
    )
    
    # ODDS API (S3 props data)
    normalize_from_odds_api("AARON NESMITH")                # → "Aaron Nesmith"
    normalize_from_odds_api("Alfred Joel Horford Reynoso")  # → "Al Horford"
    normalize_from_odds_api("Cameron Johnson")              # → "Cam Johnson"
    
    # ESPN API
    normalize_from_espn_api("Aaron Nesmith")     # → "Aaron Nesmith"
    normalize_from_espn_api("Luka Dončić")       # → "Luka Doncic"
    
    # NBA API (source of truth)
    normalize_from_nba_api("Aaron Nesmith")      # → "Aaron Nesmith"
    normalize_from_nba_api("Al Horford")         # → "Al Horford"
    
    # All normalize to same NBA API canonical name:
    assert normalize_from_odds_api("Alfred Joel Horford Reynoso") == "Al Horford"
    assert normalize_from_nba_api("Al Horford") == "Al Horford"
    # ✅ Both return "Al Horford"

WORKFLOW:
---------
1. Discovery: Extract from S3 → normalize_from_odds_api() → dedupe
2. Lookup: Use normalized name to find player_id in NBA API
3. Fetch: Get game logs from NBA API → normalize_from_nba_api()
4. Join: All names now match NBA API canonical format

Author: Myles Thomas
Date: 2025-02-04
"""

import pandas as pd
import unicodedata
import re
from typing import Optional


def remove_accents(text: str) -> str:
    """
    Remove accents/diacritics from text.
    
    Examples:
        'Luka Dončić' -> 'Luka Doncic'
        'Kristaps Porziņģis' -> 'Kristaps Porzingis'
        'Nikola Jokić' -> 'Nikola Jokic'
    """
    if pd.isna(text):
        return text
    
    # Normalize to NFD (decompose), filter out combining characters
    nfd = unicodedata.normalize('NFD', text)
    return ''.join(char for char in nfd if unicodedata.category(char) != 'Mn')


def is_valid_player_name(name: str) -> bool:
    """
    Check if a name is actually a player name (not garbage data).
    
    Filters out:
    - Team names ("Atl Hawks Alternate Total")
    - Betting lines ("Over 223.5")
    - Malformed names ("Jalen (2001) Johnson", "Irving, Kyrie")
    - Single initials ("D Jones Jr", "G Antetokounmpo")
    - Empty/whitespace
    - Single words (need at least first + last name)
    """
    if pd.isna(name) or not name or not name.strip():
        return False
    
    name = name.strip()
    
    # Filter out names with commas (malformed: "Irving, Kyrie")
    if ',' in name:
        return False
    
    # Filter out names with parentheses (malformed: "Jalen (2001) Johnson")
    if '(' in name or ')' in name:
        return False
    
    # Must have at least 2 words (first + last name)
    words = name.split()
    if len(words) < 2:
        return False
    
    # First word must be at least 2 characters (filter out "D Jones Jr", "G Antetokounmpo")
    first_word = words[0].replace('.', '')  # Remove periods first
    if len(first_word) < 2:
        return False
    
    # Filter out common garbage patterns
    garbage_patterns = [
        r'total',           # "Atl Hawks Alternate Total"
        r'over \d',         # "Over 223.5"
        r'under \d',        # "Under 223.5"
        r'\d+\.\d',         # Any decimal number
        r'hawks',           # Team names
        r'lakers',
        r'celtics',
        r'alternate',
    ]
    
    name_lower = name.lower()
    for pattern in garbage_patterns:
        if re.search(pattern, name_lower):
            return False
    
    return True


def fix_reversed_names(name: str) -> str:
    """
    Fix reversed names from Odds API.
    
    Some Odds API data has names reversed like "Caldwell Pope Kentavious"
    instead of "Kentavious Caldwell-Pope".
    
    Returns:
        Fixed name (or original if not reversed)
    """
    # Known reversed names (before normalization, so keep original case/format)
    reversed_patterns = {
        'Caldwell Pope Kentavious': 'Kentavious Caldwell-Pope',
        'Caldwell-Pope Kentavious': 'Kentavious Caldwell-Pope',
        'Grant Jerami': 'Jerami Grant',
        'Highsmith Haywood': 'Haywood Highsmith',
        'Huerter Kevin': 'Kevin Huerter',
        'Love Kevin': 'Kevin Love',
        'Murray Dejounte': 'Dejounte Murray',
        'Portis Bobby': 'Bobby Portis',
        'Wembanyama Victor': 'Victor Wembanyama',
    }
    
    # Try exact match first
    if name in reversed_patterns:
        return reversed_patterns[name]
    
    # Try case-insensitive match
    for pattern, fixed in reversed_patterns.items():
        if name.lower() == pattern.lower():
            return fixed
    
    return name


def get_jr_sr_exceptions() -> set:
    """
    Players where BOTH Sr and Jr (or II) played in NBA.
    
    For these players, we need to KEEP the suffix to distinguish them.
    For all others, we remove Jr/Sr during normalization.
    
    Returns:
        Set of player names that need suffix preserved
    """
    return {
        'Tim Hardaway Jr',     # Father: Tim Hardaway
        'Gary Payton Ii',      # Father: Gary Payton (note: II not Jr)
        'Glen Rice Jr',        # Father: Glen Rice
        'Larry Nance Jr',      # Father: Larry Nance
        'Kenyon Martin Jr',    # Father: Kenyon Martin (goes by KJ Martin)
    }


def get_odds_api_to_nba_mappings() -> dict:
    """
    Odds API → NBA API name mappings (AFTER basic normalization).
    
    These mappings handle edge cases where the Odds API normalized name
    still doesn't match the NBA API normalized name.
    
    MAPPING DIRECTION:
        Odds API normalized → NBA API canonical
    
    Examples:
        'Alfred Joel Horford Reynoso' → 'Al Horford'  (Full legal name → nickname)
        'Cameron Johnson' → 'Cam Johnson'              (Full name → shortened)
        'Christian James Mccollum' → 'Cj Mccollum'     (Full legal → initials)
    
    Common Odds API issues:
    - Full legal names (middle names included)
    - Full first names instead of nicknames
    - Typos and abbreviations
    - Reversed name order
    
    Returns:
        Dict mapping {odds_api_normalized: nba_api_canonical}
    """
    return {
        # =================================================================
        # NICKNAMES: Map all variations → NBA API canonical name
        # Format: 'Other API name' → 'NBA API name'
        # =================================================================
        'Herbert Jones': 'Herb Jones',           # NBA API canonical: Herb
        'Moritz Wagner': 'Moe Wagner',           # NBA API canonical: Moe
        'Nic Claxton': 'Nicolas Claxton',        # NBA API canonical: Nicolas
        'Ronald Holland': 'Ron Holland',         # NBA API canonical: Ron
        'Vincent Williams Jr': 'Vince Williams Jr',  # NBA API canonical: Vince
        
        # =================================================================
        # FULL LEGAL NAMES: Map Odds API legal names → NBA API nicknames
        # Format: 'Odds API full legal name' → 'NBA API common name'
        # =================================================================
        'Alfred Joel Horford Reynoso': 'Al Horford',     # NBA API canonical: Al Horford
        'Wardell Stephen Curry': 'Stephen Curry',        # NBA API canonical: Stephen Curry
        'William Anthony Perry': 'Anthony Perry',        # NBA API canonical: Anthony Perry
        
        # =================================================================
        # JR SUFFIX EXCEPTIONS: Only for players where BOTH Sr and Jr played
        # (All other Jr/Sr suffixes are removed during normalization)
        # Format: 'Any variation' → 'NBA API name (WITH suffix)'
        # =================================================================
        'Kenyon Martin Jr': 'Kj Martin',  # NBA API canonical: KJ Martin (son of Kenyon Martin)
        
        # =================================================================
        # NAME CHANGES / ROOKIES: Pre-draft → NBA names
        # Format: 'Pre-draft/college name' → 'NBA API current name'
        # =================================================================
        'Carlton Carrington': 'Bub Carrington',   # NBA API canonical: Bub Carrington
        
        # =================================================================
        # SHORTENED NAMES: Odds API full → NBA API shortened
        # Format: 'Odds API full name' → 'NBA API shortened'
        # =================================================================
        'Cameron Johnson': 'Cam Johnson',         # NBA API canonical: Cam
        'Cameron Reddish': 'Cam Reddish',         # NBA API canonical: Cam
        'Cameron Thomas': 'Cam Thomas',           # NBA API canonical: Cam
        'Joshua Giddey': 'Josh Giddey',           # NBA API canonical: Josh
        'Obadiah Toppin': 'Obi Toppin',           # NBA API canonical: Obi
        
        # =================================================================
        # HYPHENATED NAMES: Standardize hyphenation
        # Format: 'Odds API no hyphen' → 'NBA API with hyphen'
        # =================================================================
        'Dorian Finney Smith': 'Dorian Finney-Smith',  # NBA API canonical: Finney-Smith
        
        # =================================================================
        # TYPOS/ABBREVIATIONS: Odds API errors → NBA API correct
        # Format: 'Odds API typo' → 'NBA API correct name'
        # =================================================================
        'Mil Bridges': 'Miles Bridges',           # NBA API canonical: Miles
        'Xavier Tillman Sr': 'Xavier Tillman',    # NBA API canonical: Xavier Tillman (no Sr)
        
        # =================================================================
        # FULL LEGAL NAMES: Odds API legal → NBA API common
        # Format: 'Odds API full legal name' → 'NBA API common name'
        # =================================================================
        'Brook Robert Lopez': 'Brook Lopez',           # NBA API canonical: Brook Lopez
        'Christian James Mccollum': 'Cj Mccollum',     # NBA API canonical: CJ McCollum
        'Eric Ambrose Gordon Jr': 'Eric Gordon',       # NBA API canonical: Eric Gordon
    }


def get_espn_api_to_nba_mappings() -> dict:
    """
    ESPN API → NBA API name mappings (AFTER basic normalization).
    
    These mappings handle edge cases where the ESPN API normalized name
    still doesn't match the NBA API normalized name.
    
    MAPPING DIRECTION:
        ESPN API normalized → NBA API canonical
    
    Examples:
        'TBD' → 'TBD'  (placeholder - add mappings as we discover ESPN differences)
    
    Common ESPN API issues:
    - Accented characters (handled in basic normalization)
    - Mixed nickname/full name usage
    - Different abbreviations
    
    Returns:
        Dict mapping {espn_api_normalized: nba_api_canonical}
    """
    return {
        # Add ESPN-specific mappings here as we discover them
        # Most ESPN data matches NBA API after basic normalization
    }


def normalize_player_name_base(name: str) -> Optional[str]:
    """
    BASIC normalization - applies to ALL sources (Odds API, NBA API, ESPN API).
    
    This performs universal cleaning without source-specific mappings:
    1. Validate it's a real player name
    2. Fix reversed names
    3. Remove periods (P.J. → Pj)
    4. Convert to Title Case
    5. Remove accents (Dončić → Doncic)
    6. Remove Jr/Sr suffixes (with exceptions)
    7. Clean whitespace
    
    Does NOT apply source-specific mappings - use normalize_from_*() functions for that.
    
    Args:
        name: Raw player name from any source
        
    Returns:
        Normalized player name, or None if invalid
        
    Examples:
        >>> normalize_player_name_base("AARON NESMITH")
        'Aaron Nesmith'
        
        >>> normalize_player_name_base("P.J. Washington")
        'Pj Washington'
        
        >>> normalize_player_name_base("Luka Dončić")
        'Luka Doncic'
        
        >>> normalize_player_name_base("Atl Hawks Alternate Total")
        None
    """
    if pd.isna(name):
        return None
    
    # Strip whitespace
    name = name.strip()
    
    # Fix reversed names FIRST (before validation)
    name = fix_reversed_names(name)
    
    # Validate it's a player name
    if not is_valid_player_name(name):
        return None
    
    # Remove ALL periods (handles P.J. vs PJ, Jr. vs Jr, etc.)
    name = name.replace('.', '')
    
    # Convert to Title Case
    name = name.title()
    
    # Remove accents
    name = remove_accents(name)
    
    # Remove ALL generational suffixes (Jr, Sr, II, III, IV, V)
    # UNLESS the player is in our exceptions list (where both Sr and Jr played)
    exceptions = get_jr_sr_exceptions()
    
    if name not in exceptions:
        suffixes_to_remove = [' Iii', ' Ii', ' Iv', ' V', ' Jr', ' Sr']
        for suffix in suffixes_to_remove:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
                break
    
    # Clean up multiple spaces
    name = ' '.join(name.split())
    
    return name


def normalize_from_odds_api(name: str) -> Optional[str]:
    """
    Normalize a player name from THE ODDS API.
    
    Process:
    1. Apply basic normalization (universal)
    2. Apply Odds API → NBA API mappings
    
    Use this when processing names from:
    - S3 props data (s3://the-odds-api-mt/)
    - The Odds API responses
    - Any betting/odds data
    
    Args:
        name: Raw player name from Odds API
        
    Returns:
        NBA API canonical name, or None if invalid
        
    Examples:
        >>> normalize_from_odds_api("AARON NESMITH")
        'Aaron Nesmith'
        
        >>> normalize_from_odds_api("Alfred Joel Horford Reynoso")
        'Al Horford'
        
        >>> normalize_from_odds_api("Cameron Johnson")
        'Cam Johnson'
    """
    # Step 1: Basic normalization
    name = normalize_player_name_base(name)
    
    if not name:
        return None
    
    # Step 2: Apply Odds API → NBA API mappings
    odds_mappings = get_odds_api_to_nba_mappings()
    if name in odds_mappings:
        name = odds_mappings[name]
    
    return name


def normalize_from_espn_api(name: str) -> Optional[str]:
    """
    Normalize a player name from ESPN API.
    
    Process:
    1. Apply basic normalization (universal)
    2. Apply ESPN API → NBA API mappings
    
    Use this when processing names from:
    - ESPN API responses
    - ESPN play-by-play data
    - ESPN box scores
    
    Args:
        name: Raw player name from ESPN API
        
    Returns:
        NBA API canonical name, or None if invalid
        
    Examples:
        >>> normalize_from_espn_api("Aaron Nesmith")
        'Aaron Nesmith'
        
        >>> normalize_from_espn_api("Luka Dončić")
        'Luka Doncic'
    """
    # Step 1: Basic normalization
    name = normalize_player_name_base(name)
    
    if not name:
        return None
    
    # Step 2: Apply ESPN API → NBA API mappings
    espn_mappings = get_espn_api_to_nba_mappings()
    if name in espn_mappings:
        name = espn_mappings[name]
    
    return name


def normalize_from_nba_api(name: str) -> Optional[str]:
    """
    Normalize a player name from NBA API (source of truth).
    
    Process:
    1. Apply basic normalization (universal)
    2. No mappings needed - NBA API is the canonical source
    
    Use this when processing names from:
    - nba_api library responses
    - NBA.com data
    - Official NBA sources
    
    Args:
        name: Raw player name from NBA API
        
    Returns:
        Normalized NBA API name, or None if invalid
        
    Examples:
        >>> normalize_from_nba_api("Aaron Nesmith")
        'Aaron Nesmith'
        
        >>> normalize_from_nba_api("Al Horford")
        'Al Horford'
    """
    # NBA API names only need basic normalization (no mappings)
    return normalize_player_name_base(name)


def normalize_player_name(name: str) -> Optional[str]:
    """
    DEPRECATED: Use normalize_from_odds_api(), normalize_from_espn_api(), or normalize_from_nba_api() instead.
    
    For backwards compatibility, this defaults to Odds API normalization
    (since most of our data comes from S3/Odds API).
    """
    return normalize_from_odds_api(name)


def normalize_player_names_series(series: pd.Series, source: str = 'odds_api') -> pd.Series:
    """
    Normalize all player names in a pandas Series.
    
    Args:
        series: Series containing player names
        source: Which API the names come from ('odds_api', 'espn_api', 'nba_api')
        
    Returns:
        Series with normalized names (invalid names become None)
    """
    if source == 'odds_api':
        return series.apply(normalize_from_odds_api)
    elif source == 'espn_api':
        return series.apply(normalize_from_espn_api)
    elif source == 'nba_api':
        return series.apply(normalize_from_nba_api)
    else:
        raise ValueError(f"Invalid source: {source}. Must be 'odds_api', 'espn_api', or 'nba_api'")


def normalize_player_names_df(df: pd.DataFrame, player_col: str = 'player', source: str = 'odds_api') -> pd.DataFrame:
    """
    Normalize player names in a DataFrame column.
    
    Args:
        df: DataFrame containing player names
        player_col: Name of column containing player names
        source: Which API the names come from ('odds_api', 'espn_api', 'nba_api')
        
    Returns:
        DataFrame with normalized player names
    """
    df = df.copy()
    df[player_col] = normalize_player_names_series(df[player_col], source=source)
    # Drop rows with invalid player names
    df = df[df[player_col].notna()]
    return df


# =============================================================================
# TESTING
# =============================================================================

if __name__ == '__main__':
    """
    Test name normalization with examples from all 3 APIs.
    
    Tests both basic normalization and source-specific mappings.
    """
    print("="*80)
    print("PLAYER NAME NORMALIZATION TEST")
    print("(Testing source-specific normalization)")
    print("="*80)
    print()
    
    test_cases = [
        # Format: (input, source, expected_output)
        
        # ===== ODDS API =====
        ("AARON NESMITH", "odds_api", "Aaron Nesmith"),
        ("Alfred Joel Horford Reynoso", "odds_api", "Al Horford"),
        ("Cameron Johnson", "odds_api", "Cam Johnson"),
        ("Cameron Thomas", "odds_api", "Cam Thomas"),
        ("Christian James Mccollum", "odds_api", "Cj Mccollum"),
        ("Brook Robert Lopez", "odds_api", "Brook Lopez"),
        ("Anthony Davis Jr.", "odds_api", "Anthony Davis"),
        ("Bobby Portis Jr", "odds_api", "Bobby Portis"),
        ("P.J. Washington", "odds_api", "Pj Washington"),
        ("Atl Hawks Alternate Total", "odds_api", None),
        ("Over 223.5", "odds_api", None),
        ("D Robinson", "odds_api", None),
        
        # ===== NBA API (source of truth) =====
        ("Aaron Nesmith", "nba_api", "Aaron Nesmith"),
        ("Al Horford", "nba_api", "Al Horford"),
        ("Cam Johnson", "nba_api", "Cam Johnson"),
        ("Anthony Davis", "nba_api", "Anthony Davis"),
        ("Bobby Portis", "nba_api", "Bobby Portis"),
        ("Pj Washington", "nba_api", "Pj Washington"),
        ("Tim Hardaway Jr", "nba_api", "Tim Hardaway Jr"),  # Exception
        
        # ===== ESPN API =====
        ("Aaron Nesmith", "espn_api", "Aaron Nesmith"),
        ("Luka Dončić", "espn_api", "Luka Doncic"),
        ("Nikola Jokić", "espn_api", "Nikola Jokic"),
        ("aaron nesmith", "espn_api", "Aaron Nesmith"),
    ]
    
    print("TEST CASES:")
    print("-" * 80)
    success = 0
    fail = 0
    
    for input_name, source, expected in test_cases:
        # Apply source-specific normalization
        if source == "odds_api":
            result = normalize_from_odds_api(input_name)
        elif source == "nba_api":
            result = normalize_from_nba_api(input_name)
        elif source == "espn_api":
            result = normalize_from_espn_api(input_name)
        
        status = "✅" if result == expected else "❌"
        
        if result == expected:
            success += 1
        else:
            fail += 1
        
        print(f"{status} {source:12} | {input_name:35} -> {result}")
        if result != expected:
            print(f"   Expected: {expected}")
    
    print("-" * 80)
    print(f"\nResults: {success} passed, {fail} failed")
    print()
    
    # Test cross-source matching
    print("="*80)
    print("CROSS-SOURCE MATCHING TEST")
    print("(Verify all sources normalize to same NBA API canonical name)")
    print("="*80)
    print()
    
    cross_tests = [
        ("Alfred Joel Horford Reynoso", "odds_api", "Al Horford", "nba_api"),
        ("Cameron Johnson", "odds_api", "Cam Johnson", "nba_api"),
        ("Luka Dončić", "espn_api", "Luka Doncic", "nba_api"),
    ]
    
    for name1, source1, name2, source2 in cross_tests:
        result1 = normalize_from_odds_api(name1) if source1 == "odds_api" else normalize_from_espn_api(name1)
        result2 = normalize_from_nba_api(name2)
        
        match = result1 == result2
        status = "✅" if match else "❌"
        
        print(f"{status} {name1} ({source1}) == {name2} ({source2})")
        print(f"   Result: '{result1}' == '{result2}' → {match}")
    
    print()
    print("="*80)
