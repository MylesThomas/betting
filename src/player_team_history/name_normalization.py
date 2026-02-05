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
        
    Note:
        These exceptions are critical for data integrity. Without them,
        father and son will be collapsed into a single player in the cache,
        causing incorrect team histories.
    """
    return {
        # Current/recent players with father who also played in NBA
        # NOTE: These names are as they appear in NBA API after basic normalization
        # (periods removed, Title Case, Jr suffix preserved)
        
        'Tim Hardaway Jr',       # NBA API: "Tim Hardaway Jr." | Father: Tim Hardaway (1989-2003)
        'Gary Payton Ii',        # NBA API: "Gary Payton II" | Father: Gary Payton (1990-2007)
        'Glen Rice Jr',          # NBA API: "Glen Rice Jr." | Father: Glen Rice (1989-2004)
        'Larry Nance Jr',        # NBA API: "Larry Nance Jr." | Father: Larry Nance (1981-1994)
        'Kenyon Martin Jr',      # NBA API: "Kenyon Martin Jr." (goes by KJ Martin) | Father: Kenyon Martin (2000-2015)
        'Jaren Jackson Jr',      # NBA API: "Jaren Jackson Jr." | Father: Jaren Jackson (1989-2002)
        'Kevin Porter Jr',       # NBA API: "Kevin Porter Jr." | Father: Kevin Porter (1972-1983)
        'Scotty Pippen Jr',      # NBA API: "Scotty Pippen Jr." (CORRECT spelling - his real name) | Father: Scottie Pippen (1987-2004)
        'Scottie Pippen Jr',     # Odds API: "Scottie Pippen Jr" (INCORRECT spelling, normalize to Scotty) | Same player, both spellings needed
    }


def get_college_players() -> set:
    """
    College players who appear in Odds API data but are NOT in NBA.
    
    These players should be filtered out during processing or explicitly
    marked as "college player - not in NBA API" in failure reports.
    
    Common reasons they appear in data:
    - G-League games included in betting data
    - College games included in betting data
    - Data quality issues from The Odds API
    - Players who were drafted but haven't played yet
    
    Returns:
        Set of normalized player names who are college players
        
    Notes:
        - Names should be in normalized form (Title Case, no periods, etc.)
        - Update this list as new college players appear in failures
        - Remove players from this list once they play in NBA
    """
    return {
        # Confirmed college players (2024-25 season)
        'Jj Starling',         # Syracuse - Found in 2023-11-28.csv (OKC vs MIN)
        'Jalen Reed',          # Texas - Found in 2023-11-28.csv (OKC vs MIN)
        'Chris Bell',          # Cal - Appears in betting data
        'Judah Mintz',         # Oak Hill Academy - Appears in betting data
        
        # Likely college/G-League players (need verification)
        'Jordan Wright',       # Appears in failures, not in NBA API
        'Tyrell Ward',         # Appears in failures, not in NBA API
        # 'Vincent Williams',    # Appears in failures, not in NBA API (possibly Vince Williams Jr typo?) [Just traded from Grizzlies -> Jazz]
        'Will Baker',          # Appears in failures, not in NBA API
        
        # Recent draft picks who haven't played yet
        # 'Ron Holland',         # 2024 draft pick - may not have played yet [Plays for the Detroit Pistons]
    }


def is_college_player(name: str) -> bool:
    """
    Check if a normalized player name is a known college player.
    
    Args:
        name: Normalized player name (Title Case, no periods, etc.)
        
    Returns:
        True if player is in college player list, False otherwise
        
    Example:
        >>> is_college_player("Jj Starling")
        True
        >>> is_college_player("Anthony Davis")
        False
    """
    return name in get_college_players()


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
        'Herb Jones': 'Herbert Jones',           # NBA API canonical: Herbert (not Herb!)
        'Herbert Jones': 'Herbert Jones',        # Keep as is
        'Nicholas Batum': 'Nicolas Batum',       # h vs no h
        'Ronald Holland': 'Ron Holland',         # NBA API canonical: Ron
        'Vincent Williams Jr': 'Vince Williams Jr',  # NBA API canonical: Vince
        
        # =================================================================
        # FULL LEGAL NAMES: Map Odds API legal names → NBA API nicknames
        # Format: 'Odds API full legal name' → 'NBA API common name'
        # =================================================================
        'Alfred Joel Horford Reynoso': 'Al Horford',     # NBA API canonical: Al Horford
        'Edrice Femi Adebayo': 'Bam Adebayo',            # NBA API canonical: Bam Adebayo
        'Eric Ambrose Gordon': 'Eric Gordon',            # NBA API canonical: Eric Gordon
        'Kevin Devon Knox': 'Kevin Knox',                # NBA API canonical: Kevin Knox
        'Wardell Stephen Curry': 'Stephen Curry',        # NBA API canonical: Stephen Curry
        'William Anthony Perry': 'Anthony Perry',        # NBA API canonical: Anthony Perry
        
        # =================================================================
        # JR SUFFIX EXCEPTIONS: Only for players where BOTH Sr and Jr played
        # (All other Jr/Sr suffixes are removed during normalization)
        # Format: 'Any variation' → 'NBA API name (WITH suffix)'
        # =================================================================
        'Bj Boston': 'Bj Boston Jr',              # NBA API: B.J. Boston Jr.
        'Kenyon Martin Jr': 'Kj Martin',          # NBA API canonical: KJ Martin (son of Kenyon Martin)
        
        # =================================================================
        # NAME CHANGES / ROOKIES: Pre-draft → NBA names
        # Format: 'Pre-draft/college name' → 'NBA API current name'
        # =================================================================
        'Carlton Carrington': 'Bub Carrington',   # NBA API canonical: Bub Carrington
        
        # =================================================================
        # SHORTENED NAMES: Odds API shortened → NBA API full OR vice versa
        # Format: 'Odds API name' → 'NBA API canonical'
        # =================================================================
        'Cam Johnson': 'Cameron Johnson',         # NBA API canonical: Cameron (full)
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
        'Bojan Bogdanovich': 'Bojan Bogdanovic',  # Typo: ch → c
        'Mil Bridges': 'Miles Bridges',           # NBA API canonical: Miles
        'Xavier Tillman Sr': 'Xavier Tillman',    # NBA API canonical: Xavier Tillman (no Sr)
        
        # =================================================================
        # NICKNAME VARIATIONS: Odds API full → NBA API nickname
        # Format: 'Odds API full name' → 'NBA API nickname'
        # =================================================================
        'Moe Wagner': 'Moritz Wagner',             # Odds API: "Moe Wagner" → NBA API: "Moritz Wagner" (full name)
        'Mohamed Bamba': 'Mo Bamba',               # Odds API: "Mohamed Bamba" → NBA API: "Mo Bamba" (nickname)
        'Nicolas Claxton': 'Nic Claxton',          # Odds API: "Nicolas Claxton" → NBA API: "Nic Claxton" (nickname)
        'Bj Boston': 'Brandon Boston',             # Odds API: "BJ Boston Jr" → NBA API: "Brandon Boston" (full first name)
        'Ron Holland': 'Ronald Holland',           # Odds API: "Ron Holland" → NBA API: "Ronald Holland" (full first name)
        'Vincent Williams': 'Vince Williams',      # Odds API: "Vincent Williams Jr" → NBA API: "Vince Williams Jr." (nickname)
        'Scottie Pippen Jr': 'Scotty Pippen Jr',   # Odds API uses "Scottie" → normalize to correct "Scotty" (his real name)
        
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
    1. Standardize apostrophes (curly → straight)
    2. Validate it's a real player name
    3. Fix reversed names
    4. Remove periods (P.J. → Pj)
    5. Convert to Title Case
    6. Remove accents (Dončić → Doncic)
    7. Remove Jr/Sr suffixes (with exceptions)
    8. Clean whitespace
    
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
    
    # Standardize ALL apostrophes to straight quote (')
    # Converts: ' (8217), ` (96), ′ (8242) → ' (39)
    apostrophe_variants = [
        '\u2019',  # ' (right single quotation mark, ord 8217)
        '\u2018',  # ' (left single quotation mark, ord 8216)
        '\u0060',  # ` (grave accent, ord 96)
        '\u2032',  # ′ (prime, ord 8242)
    ]
    for variant in apostrophe_variants:
        name = name.replace(variant, "'")
    
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


def get_nba_api_corrections() -> dict:
    """
    NBA API name corrections (AFTER basic normalization).
    
    Even though NBA API is the source of truth, sometimes it has:
    - Spelling variations (Scotty vs Scottie)
    - Inconsistent Jr suffix handling
    
    These corrections ensure consistency with our canonical names.
    
    Returns:
        Dict mapping {nba_api_normalized: corrected_canonical}
        
    Format:
        'Name As Normalized From NBA API' → 'Canonical Name'
        
    Examples:
        (Currently no NBA API corrections needed - NBA API returns correct spellings)
    """
    return {
        # Currently empty - NBA API is source of truth and returns correct names
        # If NBA API ever returns incorrect variations, add corrections here
    }


def normalize_from_nba_api(name: str) -> Optional[str]:
    """
    Normalize a player name from NBA API (source of truth).
    
    Process:
    1. Apply basic normalization (universal)
    2. Apply NBA API corrections (spelling fixes, Jr preservation)
    
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
        
        >>> normalize_from_nba_api("Scotty Pippen Jr.")
        'Scotty Pippen Jr'
    """
    # Step 1: Basic normalization
    name = normalize_player_name_base(name)
    
    if not name:
        return None
    
    # Step 2: Apply NBA API corrections
    corrections = get_nba_api_corrections()
    if name in corrections:
        name = corrections[name]
    
    return name


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
