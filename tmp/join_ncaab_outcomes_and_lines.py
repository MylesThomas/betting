"""
Join NCAAB Game Outcomes with Betting Lines

Reads game results from ESPN API and betting lines from The Odds API,
then joins them to create a modeling dataset. Identifies team name
mismatches and coverage gaps.

S3 Sources:
- Game outcomes: s3://ncaab-betting-mt/data/01_input/historical_game_results/*.csv
- Game lines: s3://ncaab-betting-mt/data/01_input/the-odds-api/ncaab/game_lines/*.csv

Output:
- Joined dataset saved to: s3://ncaab-betting-mt/data/03_intermediate/modeling/
- Team name mapping for unmatched teams

Join Strategy:
- LEFT JOIN on game outcomes (keep all games with results)
- Match on: game_date, home_team, away_team
- Identify games with results but no lines (coverage gaps)
- Identify games with lines but no results (rare, but possible)

Usage:
    # STEP 1: Find team name mismatches and get suggested mappings (creates cache)
    python tmp/join_ncaab_outcomes_and_lines.py --season 2024-25 --find-school-matches --test
    
    # STEP 2: After adding mappings to TEAM_NAME_MAP, test the join (FAST - uses cache!)
    python tmp/join_ncaab_outcomes_and_lines.py --season 2024-25 --test --use-cache
    
    # STEP 3: Iterate on mappings quickly with cache
    python tmp/join_ncaab_outcomes_and_lines.py --season 2024-25 --find-school-matches --test --use-cache
    
    # STEP 4: Build full dataset for S3
    python tmp/join_ncaab_outcomes_and_lines.py --season 2024-25 --s3
    
    # Build for multiple seasons at once
    python tmp/join_ncaab_outcomes_and_lines.py --seasons 2025-26,2024-25,2023-24,2022-23,2021-22,2020-21 --s3
    
Cache Location:
    Data cached to: ~/Downloads/tmp/ncaab_cache/
    Files: outcomes_{start}_{end}.parquet, lines_{start}_{end}.parquet

Context:
Building ML model to predict NCAAB scores and compare against market-implied scores.
Need to identify team name mismatches between ESPN (game outcomes) and 
The Odds API (betting lines) before full modeling pipeline.

Author: Thomas Myles
Date: 2026-01-15
"""

import sys
import os
import pandas as pd
import boto3
from pathlib import Path
from datetime import datetime
from io import StringIO
import argparse
from difflib import SequenceMatcher

# Find project root
def find_project_root():
    """Find project root by looking for .gitignore file."""
    current = Path.cwd()
    while current != current.parent:
        if (current / '.gitignore').exists():
            return current
        current = current.parent
    return Path.cwd()

PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from config_loader import get_config
from ncaab_team_name_mapping import ODDS_API_TO_ESPN_NCAAB

CONFIG = get_config()

# Case-insensitive Odds API → ESPN lookup (lines data may have varying case)
_ODDS_TO_ESPN_LOWER = {k.lower(): v for k, v in ODDS_API_TO_ESPN_NCAAB.items()}

# =============================================================================
# CONFIGURATION
# =============================================================================

# S3 Configuration
OUTCOMES_BUCKET = 'ncaab-betting-mt'
OUTCOMES_PATH = 'data/01_input/historical_game_results/'
LINES_BUCKET = 'ncaab-betting-mt'
LINES_PATH = 'data/01_input/the-odds-api/ncaab/game_lines/'
OUTPUT_BUCKET = 'ncaab-betting-mt'
OUTPUT_PATH = 'data/03_intermediate/modeling/'

# Local test directory
LOCAL_TEST_DIR = Path.home() / 'Downloads' / 'tmp'

# Cache directory for faster iteration
CACHE_DIR = LOCAL_TEST_DIR / 'ncaab_cache'

# Dynamic today's date
TODAY = datetime.now().strftime('%Y-%m-%d')
if TODAY > '2026-04-30':
    TODAY = '2026-04-30'

# Season date ranges (approximate)
SEASON_DATES = {
    '2025-26': ('2025-11-03', TODAY),
    '2024-25': ('2024-11-03', '2025-04-20'),
    '2023-24': ('2023-11-06', '2024-04-20'),
    '2022-23': ('2022-11-07', '2023-04-30'),
    '2021-22': ('2021-11-09', '2022-04-04'),
    '2020-21': ('2020-11-25', '2021-04-04'),
    # '2019-20': ('2019-11-05', '2020-03-14'), (no data in the-odds-api for this season)
}

# Team name mapping: Maps Odds API names to ESPN names
# This handles cases where the two sources use different naming conventions
TEAM_NAME_MAP = {
    # Odds API name (lowercase) -> ESPN name (as it appears in data)
    
    # NEW MAPPINGS FROM 2024-25 SEASON (13 mismatches found)
    'albany great danes': 'UAlbany Great Danes',
    'appalachian st mountaineers': 'App State Mountaineers',
    'csu bakersfield roadrunners': 'Cal State Bakersfield Roadrunners',
    'csu fullerton titans': 'Cal State Fullerton Titans',
    'csu northridge matadors': 'Cal State Northridge Matadors',
    'central connecticut st blue devils': 'Central Connecticut Blue Devils',
    'east tennessee st buccaneers': 'East Tennessee State Buccaneers',
    'illinois st redbirds': 'Illinois State Redbirds',
    'indiana st sycamores': 'Indiana State Sycamores',
    'liu sharks': 'Long Island University Sharks',
    'missouri st bears': 'Missouri State Bears',
    'portland st vikings': 'Portland State Vikings',
    'ut martin skyhawks': 'UT Martin Skyhawks',
    'umkc kangaroos': 'Kansas City Roos',  # UMKC vs Kansas City, Kangaroos vs Roos
    
    # State abbreviations (St → State) - high confidence matches
    'alabama st hornets': 'Alabama State Hornets',
    'alcorn st braves': 'Alcorn State Braves',
    'arizona st sun devils': 'Arizona State Sun Devils',
    'arizona st sun devils': 'Arizona State Sun Devils',
    'arkansas st red wolves': 'Arkansas State Red Wolves',
    'chicago st cougars': 'Chicago State Cougars',
    'cleveland st vikings': 'Cleveland State Vikings',
    'colorado st rams': 'Colorado State Rams',
    'coppin st eagles': 'Coppin State Eagles',
    'delaware st hornets': 'Delaware State Hornets',
    'florida st seminoles': 'Florida State Seminoles',
    'fresno st bulldogs': 'Fresno State Bulldogs',
    'georgia st panthers': 'Georgia State Panthers',
    'jackson st tigers': 'Jackson State Tigers',
    'jacksonville st gamecocks': 'Jacksonville State Gamecocks',
    'kansas st wildcats': 'Kansas State Wildcats',
    'kennesaw st owls': 'Kennesaw State Owls',
    'michigan st spartans': 'Michigan State Spartans',
    'mississippi st bulldogs': 'Mississippi State Bulldogs',
    'montana st bobcats': 'Montana State Bobcats',
    'morehead st eagles': 'Morehead State Eagles',
    'morgan st bears': 'Morgan State Bears',
    'murray st racers': 'Murray State Racers',
    'new mexico st aggies': 'New Mexico State Aggies',
    'norfolk st spartans': 'Norfolk State Spartans',
    'north dakota st bison': 'North Dakota State Bison',
    'northwestern st demons': 'Northwestern State Demons',
    'oklahoma st cowboys': 'Oklahoma State Cowboys',
    'oregon st beavers': 'Oregon State Beavers',
    'sacramento st hornets': 'Sacramento State Hornets',
    'san diego st aztecs': 'San Diego State Aztecs',
    'san josé st spartans': 'San José State Spartans',
    'south carolina st bulldogs': 'South Carolina State Bulldogs',
    'south dakota st jackrabbits': 'South Dakota State Jackrabbits',
    'tennessee st tigers': 'Tennessee State Tigers',
    'washington st cougars': 'Washington State Cougars',
    'wichita st shockers': 'Wichita State Shockers',
    'wright st raiders': 'Wright State Raiders',
    'youngstown st penguins': 'Youngstown State Penguins',
    
    # University/college name variations
    'american eagles': 'American University Eagles',
    'boston univ. terriers': 'Boston University Terriers',
    
    # Little Rock (Arkansas-Little Rock)
    'arkansas-little rock trojans': 'Little Rock Trojans',
    
    # Army (Army Black Knights)
    'army knights': 'Army Black Knights',
    
    # California Baptist
    'cal baptist lancers': 'California Baptist Lancers',
    
    # Florida International
    'florida int\'l golden panthers': 'Florida International Panthers',
    
    # Fort Wayne (Purdue Fort Wayne)
    'fort wayne mastodons': 'Purdue Fort Wayne Mastodons',
    
    # Gardner-Webb
    'gardner-webb bulldogs': 'Gardner-Webb Runnin\' Bulldogs',
    
    # GW (George Washington)
    'gw revolutionaries': 'George Washington Revolutionaries',
    
    # Grambling
    'grambling st tigers': 'Grambling Tigers',
    
    # Grand Canyon (Lopes vs Antelopes)
    'grand canyon antelopes': 'Grand Canyon Lopes',
    
    # IUPUI → IU Indianapolis
    'iupui jaguars': 'IU Indianapolis Jaguars',
    
    # Long Beach State
    'long beach st 49ers': 'Long Beach State Beach',
    
    # Loyola Chicago
    'loyola (chi) ramblers': 'Loyola Chicago Ramblers',
    
    # Loyola Maryland
    'loyola (md) greyhounds': 'Loyola Maryland Greyhounds',
    
    # Maryland Eastern Shore
    'maryland-eastern shore hawks': 'Maryland Eastern Shore Hawks',
    
    # Mississippi Valley State
    'miss valley st delta devils': 'Mississippi Valley State Delta Devils',
    
    # Mount St. Mary's
    'mt. st. mary\'s mountaineers': 'Mount St. Mary\'s Mountaineers',
    
    # Northern Colorado
    'n colorado bears': 'Northern Colorado Bears',
    
    # Nicholls
    'nicholls st colonels': 'Nicholls Colonels',
    
    # Prairie View A&M
    'prairie view panthers': 'Prairie View A&M Panthers',
    
    # Sam Houston
    'sam houston st bearkats': 'Sam Houston Bearkats',
    
    # Seattle U
    'seattle redhawks': 'Seattle U Redhawks',
    
    # Southeast Missouri State
    'se missouri st redhawks': 'Southeast Missouri State Redhawks',
    
    # SIU Edwardsville
    'siu-edwardsville cougars': 'SIU Edwardsville Cougars',
    
    # Saint Francis
    'st. francis (pa) red flash': 'Saint Francis Red Flash',
    
    # St. Thomas Minnesota
    'st. thomas (mn) tommies': 'St. Thomas-Minnesota Tommies',
    
    # UT Martin
    'tenn-martin skyhawks': 'UT Martin Skyhawks',
    
    # Texas A&M-Corpus Christi
    'texas a&m-cc islanders': 'Texas A&M-Corpus Christi Islanders',
    
    # East Texas A&M (formerly Texas A&M-Commerce)
    'texas a&m-commerce lions': 'East Texas A&M Lions',
    
    # UT Arlington
    'ut-arlington mavericks': 'UT Arlington Mavericks',
}


# =============================================================================
# HELPER FUNCTIONS - DATA LOADING
# =============================================================================

def list_s3_files(bucket, prefix):
    """List all files in S3 bucket with given prefix."""
    s3_client = boto3.client('s3')
    
    files = []
    paginator = s3_client.get_paginator('list_objects_v2')
    
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                files.append(obj['Key'])
    
    return files


def read_s3_csv(bucket, key):
    """Read CSV file from S3."""
    s3_client = boto3.client('s3')
    
    try:
        obj = s3_client.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
        return df
    except Exception as e:
        print(f"   ❌ Error reading {key}: {e}")
        return None


def get_cache_path(data_type, start_date, end_date):
    """Get cache file path for a given data type and date range."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{data_type}_{start_date}_{end_date}.parquet"
    return CACHE_DIR / filename


def save_to_cache(df, data_type, start_date, end_date):
    """Save DataFrame to cache."""
    cache_path = get_cache_path(data_type, start_date, end_date)
    df.to_parquet(cache_path, index=False)
    print(f"   💾 Cached to: {cache_path}")


def load_from_cache(data_type, start_date, end_date):
    """Load DataFrame from cache if it exists."""
    cache_path = get_cache_path(data_type, start_date, end_date)
    
    if cache_path.exists():
        print(f"   📦 Loading from cache: {cache_path}")
        df = pd.read_parquet(cache_path)
        return df
    
    return None


def load_game_outcomes(start_date, end_date, use_cache=False):
    """
    Load game outcomes from S3 for date range.
    
    Args:
        start_date: Start date string
        end_date: End date string
        use_cache: If True, load from cache if available
    
    Returns DataFrame with columns:
    - GAME_DATE, GAME_ID, HOME_TEAM, AWAY_TEAM, HOME_SCORE, AWAY_SCORE, etc.
    """
    print(f"\n📥 Loading game outcomes...")
    
    # Try cache first if requested
    if use_cache:
        cached_df = load_from_cache('outcomes', start_date, end_date)
        if cached_df is not None:
            print(f"   ✅ Loaded {len(cached_df)} games from cache")
            return cached_df
        print(f"   ℹ️  No cache found, loading from S3...")
    
    print(f"   Bucket: {OUTCOMES_BUCKET}")
    print(f"   Path: {OUTCOMES_PATH}")
    print(f"   Date range: {start_date} to {end_date}")
    
    # List all files
    all_files = list_s3_files(OUTCOMES_BUCKET, OUTCOMES_PATH)
    print(f"   Found {len(all_files)} total outcome files in S3")
    
    # Filter files to date range BEFORE reading them
    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()
    
    files_in_range = []
    for file_key in all_files:
        # Extract date from file key (e.g., "data/01_input/historical_game_results/2024-11-03.csv")
        try:
            file_date_str = file_key.split('/')[-1].replace('.csv', '')
            file_date = pd.to_datetime(file_date_str).date()
            
            if start <= file_date <= end:
                files_in_range.append(file_key)
        except:
            # Skip files that don't match date format
            continue
    
    print(f"   Filtered to {len(files_in_range)} files in date range ({start_date} to {end_date})")
    
    if not files_in_range:
        print("   ❌ No files found in date range")
        return pd.DataFrame()
    
    # Read files in range with progress logging
    dfs = []
    files_loaded = 0
    total_games = 0
    
    for i, file_key in enumerate(files_in_range, 1):
        df = read_s3_csv(OUTCOMES_BUCKET, file_key)
        if df is not None and not df.empty:
            # Extract date from file key
            file_date = file_key.split('/')[-1].replace('.csv', '')
            
            dfs.append(df)
            files_loaded += 1
            total_games += len(df)
            
            # Show progress for first 5, last 5, or every 10 files
            if i <= 5 or i >= len(files_in_range) - 5 or i % 10 == 0:
                print(f"      Loaded {file_date}: {len(df)} games (total so far: {total_games})")
        
        # Progress update every 25 files
        if 1==1:
            print(f"   📊 Progress: {i}/{len(files_in_range)} files processed, {files_loaded} loaded, {total_games} games")
    
    if not dfs:
        print("   ❌ No game outcome data found")
        return pd.DataFrame()
    
    # Combine all files
    df = pd.concat(dfs, ignore_index=True)
    print(f"   ✅ Combined {files_loaded} files with {len(df)} total games")
    
    # Ensure GAME_DATE is date type
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE']).dt.date
    
    # Remove duplicates
    games_before_dedup = len(df)
    df = df.drop_duplicates(subset=['GAME_DATE', 'HOME_TEAM', 'AWAY_TEAM'])
    duplicates_removed = games_before_dedup - len(df)
    
    if duplicates_removed > 0:
        print(f"   🔄 Removed {duplicates_removed} duplicate games")
    
    print(f"   ✅ Final: {len(df)} games with outcomes")
    
    # Save to cache
    save_to_cache(df, 'outcomes', start_date, end_date)
    
    return df


def load_game_lines(start_date, end_date, use_cache=False):
    """
    Load game lines from S3 for date range.
    
    Args:
        start_date: Start date string
        end_date: End date string
        use_cache: If True, load from cache if available
    
    Returns DataFrame with columns:
    - date, event_id, home_team, away_team, consensus_spread, consensus_total, etc.
    """
    print(f"\n📥 Loading game lines...")
    
    # Try cache first if requested
    if use_cache:
        cached_df = load_from_cache('lines', start_date, end_date)
        if cached_df is not None:
            print(f"   ✅ Loaded {len(cached_df)} games from cache")
            return cached_df
        print(f"   ℹ️  No cache found, loading from S3...")
    
    print(f"   Bucket: {LINES_BUCKET}")
    print(f"   Path: {LINES_PATH}")
    print(f"   Date range: {start_date} to {end_date}")
    
    # Generate list of expected files (one per date)
    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()
    
    date_range = pd.date_range(start=start, end=end, freq='D')
    
    dfs = []
    files_found = 0
    files_missing = 0
    
    for date in date_range:
        date_str = date.strftime('%Y-%m-%d')
        file_key = f"{LINES_PATH}{date_str}.csv"
        
        df = read_s3_csv(LINES_BUCKET, file_key)
        if df is not None and not df.empty:
            dfs.append(df)
            files_found += 1
        else:
            files_missing += 1
    
    print(f"   Files found: {files_found}")
    print(f"   Files missing: {files_missing}")
    
    if not dfs:
        print("   ❌ No game line data found")
        return pd.DataFrame()
    
    # Combine all files
    df = pd.concat(dfs, ignore_index=True)
    
    # Ensure date is date type
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['date', 'home_team', 'away_team'])
    
    print(f"   ✅ Loaded {len(df)} games with lines")
    
    # Save to cache
    save_to_cache(df, 'lines', start_date, end_date)
    
    return df


# =============================================================================
# HELPER FUNCTIONS - TEAM NAME NORMALIZATION
# =============================================================================


def odds_api_name_to_espn(odds_api_name: str) -> str:
    """Convert Odds API team name to ESPN name using ODDS_API_TO_ESPN_NCAAB (case-insensitive)."""
    if pd.isna(odds_api_name):
        return ""
    key = str(odds_api_name).lower().strip()
    return _ODDS_TO_ESPN_LOWER.get(key, odds_api_name)


def normalize_team_name(team_name, use_mapping=True):
    """
    Normalize team names for matching.
    
    Strategy:
    1. Check TEAM_NAME_MAP for explicit mapping
    2. Remove common mascot suffixes
    3. Return normalized school name
    
    Examples:
    - "Duke Blue Devils" → "duke"
    - "North Carolina Tar Heels" → "north carolina"
    - "UConn Huskies" → "connecticut" (if mapped)
    """
    if pd.isna(team_name):
        return ""
    
    team = str(team_name).lower().strip()
    
    # Check explicit mapping first
    if use_mapping and team in TEAM_NAME_MAP:
        team = TEAM_NAME_MAP[team].lower().strip()
    
    # Remove common suffixes
    suffixes = [
        'blue devils', 'tar heels', 'wildcats', 'tigers', 'bulldogs',
        'huskies', 'spartans', 'buckeyes', 'wolverines', 'jayhawks',
        'cardinals', 'orangemen', 'terrapins', 'hoosiers', 'badgers',
        'fighting irish', 'scarlet knights', 'golden gophers', 'nittany lions',
        'crimson tide', 'volunteers', 'aggies', 'razorbacks', 'longhorns',
        'sooners', 'cyclones', 'bears', 'red raiders', 'mountaineers',
        'orange', 'knights', 'rebels', 'pirates', 'panthers', 'eagles',
        'hawks', 'owls', 'rams', 'bruins', 'trojans', 'cougars', 'broncos',
        'demon deacons', 'yellow jackets', 'red storm', 'friars', 'musketeers',
        'explorers', 'peacocks', 'gaels', 'minutemen', 'bonnies', 'dukes',
        'scarlet knight', 'golden gopher', 'nittany lion', 'fighting illini',
        'hurricane', 'hurricanes'
    ]
    
    for suffix in suffixes:
        if team.endswith(suffix):
            team = team[:-len(suffix)].strip()
            break
    
    return team


def calculate_similarity(s1, s2):
    """Calculate similarity ratio between two strings (0-100)."""
    return SequenceMatcher(None, s1.lower(), s2.lower()).ratio() * 100


def find_fuzzy_matches(lines_only, outcomes_teams, threshold=60):
    """
    Find potential matches for unmatched teams using fuzzy string matching.
    
    Args:
        lines_only: Set of team names from lines that don't match outcomes
        outcomes_teams: Set of all team names from outcomes
        threshold: Minimum similarity score (0-100) to consider a match
    
    Returns:
        Dict mapping lines team name to list of (outcome team name, similarity score)
    """
    matches = {}
    
    for lines_team in lines_only:
        candidates = []
        
        for outcomes_team in outcomes_teams:
            similarity = calculate_similarity(lines_team, outcomes_team)
            if similarity >= threshold:
                candidates.append((outcomes_team, similarity))
        
        # Sort by similarity score (highest first)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        if candidates:
            matches[lines_team] = candidates[:5]  # Top 5 matches
    
    return matches


def find_team_name_mismatches(outcomes_df, lines_df):
    """
    Identify team names that appear in outcomes but not lines (and vice versa).
    
    Returns:
    - outcomes_only: Teams in outcomes but not lines
    - lines_only: Teams in lines but not outcomes
    - matched: Teams that appear in both
    """
    # Get unique team names
    outcomes_teams = set()
    for col in ['HOME_TEAM', 'AWAY_TEAM']:
        if col in outcomes_df.columns:
            outcomes_teams.update(outcomes_df[col].dropna().unique())
    
    lines_teams = set()
    for col in ['home_team', 'away_team']:
        if col in lines_df.columns:
            lines_teams.update(lines_df[col].dropna().unique())
    
    # Normalize for matching (lines: Odds API → ESPN first, then same norm as join)
    outcomes_normalized = {normalize_team_name(t): t for t in outcomes_teams}
    lines_normalized = {
        normalize_team_name(odds_api_name_to_espn(t), use_mapping=False): t
        for t in lines_teams
    }
    
    # Find mismatches
    outcomes_only_norm = set(outcomes_normalized.keys()) - set(lines_normalized.keys())
    lines_only_norm = set(lines_normalized.keys()) - set(outcomes_normalized.keys())
    matched_norm = set(outcomes_normalized.keys()) & set(lines_normalized.keys())
    
    # Map back to original names
    outcomes_only = {outcomes_normalized[n] for n in outcomes_only_norm}
    lines_only = {lines_normalized[n] for n in lines_only_norm}
    matched = {outcomes_normalized[n] for n in matched_norm}
    
    return outcomes_only, lines_only, matched


# =============================================================================
# HELPER FUNCTIONS - JOINING
# =============================================================================

def join_outcomes_and_lines(outcomes_df, lines_df, min_games=5):
    """
    Join game outcomes with betting lines.
    
    Strategy: LEFT JOIN on outcomes (keep all games with results)
    Match on: date, home_team, away_team (normalized)
    
    Args:
        outcomes_df: DataFrame with game outcomes
        lines_df: DataFrame with betting lines
        min_games: Minimum games for a team to be included (filters out exhibition opponents)
    
    Returns:
    - joined_df: Combined dataset
    - stats: Dictionary with join statistics
    """
    print(f"\n🔗 Joining outcomes and lines...")
    
    # Filter out teams with < min_games (likely D2/D3/NAIA exhibition opponents)
    print(f"\n📋 Filtering teams with <{min_games} games...")
    outcomes_df = outcomes_df.copy()
    
    # Count games per team
    home_counts = outcomes_df.groupby('HOME_TEAM').size()
    away_counts = outcomes_df.groupby('AWAY_TEAM').size()
    total_counts = home_counts.add(away_counts, fill_value=0)
    
    teams_to_keep = set(total_counts[total_counts >= min_games].index)
    
    # Filter to only games where BOTH teams meet minimum
    before_filter = len(outcomes_df)
    outcomes_df = outcomes_df[
        outcomes_df['HOME_TEAM'].isin(teams_to_keep) & 
        outcomes_df['AWAY_TEAM'].isin(teams_to_keep)
    ].copy()
    after_filter = len(outcomes_df)
    
    print(f"   Teams before filter: {len(total_counts)}")
    print(f"   Teams after filter (≥{min_games} games): {len(teams_to_keep)}")
    print(f"   Games before filter: {before_filter:,}")
    print(f"   Games after filter: {after_filter:,}")
    print(f"   Games removed: {before_filter - after_filter:,} (exhibition games)")
    
    # Filter out 0-0 games (incomplete/cancelled)
    before_zero_filter = len(outcomes_df)
    outcomes_df = outcomes_df[
        (outcomes_df['HOME_SCORE'] > 0) | (outcomes_df['AWAY_SCORE'] > 0)
    ].copy()
    after_zero_filter = len(outcomes_df)
    
    if before_zero_filter > after_zero_filter:
        print(f"   Removed {before_zero_filter - after_zero_filter} games with 0-0 score")
    
    # Add normalized team names for joining
    outcomes_df['home_norm'] = outcomes_df['HOME_TEAM'].apply(normalize_team_name)
    outcomes_df['away_norm'] = outcomes_df['AWAY_TEAM'].apply(normalize_team_name)
    outcomes_df['date_key'] = outcomes_df['GAME_DATE']
    
    lines_df = lines_df.copy()
    # Odds API → ESPN mapping: track usage before we apply it
    unique_lines_teams = set(lines_df['home_team'].dropna().unique()) | set(lines_df['away_team'].dropna().unique())
    in_map = {t for t in unique_lines_teams if str(t).lower().strip() in _ODDS_TO_ESPN_LOWER}
    not_in_map = unique_lines_teams - in_map
    lines_df['_home_in_map'] = lines_df['home_team'].apply(lambda t: str(t).lower().strip() in _ODDS_TO_ESPN_LOWER if pd.notna(t) else False)
    lines_df['_away_in_map'] = lines_df['away_team'].apply(lambda t: str(t).lower().strip() in _ODDS_TO_ESPN_LOWER if pd.notna(t) else False)
    n_both_mapped = (lines_df['_home_in_map'] & lines_df['_away_in_map']).sum()
    n_one_mapped = (lines_df['_home_in_map'] != lines_df['_away_in_map']).sum()
    n_neither_mapped = (~lines_df['_home_in_map'] & ~lines_df['_away_in_map']).sum()
    print(f"\n📋 Odds API → ESPN mapping (ncaab_team_name_mapping):")
    print(f"   Unique team names in lines: {len(unique_lines_teams)}")
    print(f"   In mapping (converted to ESPN): {len(in_map)}")
    print(f"   Not in mapping (used as-is): {len(not_in_map)}")
    if not_in_map:
        print(f"   Unmapped names (first 15): {sorted(not_in_map)[:15]}")
    print(f"   Line rows: both teams mapped={n_both_mapped:,}, one mapped={n_one_mapped:,}, neither={n_neither_mapped:,}")
    lines_df.drop(columns=['_home_in_map', '_away_in_map'], inplace=True)

    # Convert Odds API names → ESPN first (so join matches outcomes), then normalize for key
    lines_df['home_norm'] = lines_df['home_team'].apply(
        lambda t: normalize_team_name(odds_api_name_to_espn(t), use_mapping=False)
    )
    lines_df['away_norm'] = lines_df['away_team'].apply(
        lambda t: normalize_team_name(odds_api_name_to_espn(t), use_mapping=False)
    )
    lines_df['date_key'] = lines_df['date']

    # LEFT JOIN on outcomes
    joined_df = outcomes_df.merge(
        lines_df,
        left_on=['date_key', 'home_norm', 'away_norm'],
        right_on=['date_key', 'home_norm', 'away_norm'],
        how='left',
        suffixes=('', '_line')
    )
    
    # Calculate join statistics
    total_outcomes = len(outcomes_df)
    total_lines = len(lines_df)
    total_joined = len(joined_df)
    matched = joined_df['consensus_spread'].notna().sum()
    unmatched = total_joined - matched
    
    stats = {
        'total_outcomes': total_outcomes,
        'total_lines': total_lines,
        'total_joined': total_joined,
        'matched': matched,
        'unmatched': unmatched,
        'coverage_pct': (matched / total_outcomes * 100) if total_outcomes > 0 else 0
    }
    
    print(f"\n📊 Join Results:")
    print(f"   Total games with outcomes (D1 only): {total_outcomes:,}")
    print(f"   Total games with lines: {total_lines:,}")
    print(f"   Matched games: {matched:,}")
    print(f"   Unmatched games (no lines): {unmatched:,}")
    print(f"   Coverage: {stats['coverage_pct']:.1f}%")
    
    return joined_df, stats


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description='Join NCAAB game outcomes with betting lines')
    parser.add_argument('--season', type=str, default='2024-25',
                       help='Season to process (e.g., "2024-25")')
    parser.add_argument('--seasons', type=str, default=None,
                       help='Multiple seasons comma-separated (e.g., "2024-25,2023-24")')
    parser.add_argument('--find-school-matches', action='store_true',
                       help='Find fuzzy matches for unmatched teams and print suggested mappings')
    parser.add_argument('--use-cache', action='store_true',
                       help='Load data from cache instead of S3 (much faster for iteration)')
    parser.add_argument('--s3', action='store_true',
                       help='Upload results to S3')
    parser.add_argument('--test', action='store_true',
                       help='Save results locally to ~/Downloads/tmp')
    
    args = parser.parse_args()
    
    # Determine seasons to process
    if args.seasons:
        seasons = args.seasons.split(',')
    else:
        seasons = [args.season]
    
    print("=" * 80)
    print("NCAAB: JOIN GAME OUTCOMES + BETTING LINES")
    print("=" * 80)
    print(f"Seasons: {', '.join(seasons)}")
    print(f"Use Cache: {'✅ Enabled' if args.use_cache else '❌ Disabled'}")
    print(f"S3 Upload: {'✅ Enabled' if args.s3 else '❌ Disabled'}")
    print(f"Test Mode: {'✅ Enabled' if args.test else '❌ Disabled'}")
    
    all_joined_dfs = []
    all_stats = []
    
    for season in seasons:
        if season not in SEASON_DATES:
            print(f"\n❌ Unknown season: {season}")
            print(f"   Available: {', '.join(SEASON_DATES.keys())}")
            continue
        
        start_date, end_date = SEASON_DATES[season]
        
        print(f"\n{'='*80}")
        print(f"PROCESSING SEASON: {season}")
        print(f"{'='*80}")
        
        # Load data
        outcomes_df = load_game_outcomes(start_date, end_date, use_cache=args.use_cache)
        lines_df = load_game_lines(start_date, end_date, use_cache=args.use_cache)
        
        if outcomes_df.empty:
            print(f"\n⚠️  No game outcomes found for {season}")
            continue
        
        if lines_df.empty:
            print(f"\n⚠️  No betting lines found for {season}")
            continue
        
        # Check for team name mismatches
        print(f"\n🔍 Checking for team name mismatches...")
        outcomes_only, lines_only, matched = find_team_name_mismatches(outcomes_df, lines_df)
        
        print(f"   Teams matched: {len(matched)}")
        print(f"   Teams in outcomes only: {len(outcomes_only)}")
        print(f"   Teams in lines only: {len(lines_only)}")
        
        if outcomes_only:
            print(f"\n   ⚠️  Teams in outcomes but NOT in lines:")
            for team in sorted(outcomes_only)[:10]:  # Show first 10
                print(f"      - {team}")
            if len(outcomes_only) > 10:
                print(f"      ... and {len(outcomes_only) - 10} more")
        
        if lines_only:
            print(f"\n   ⚠️  Teams in lines but NOT in outcomes:")
            for team in sorted(lines_only)[:10]:
                print(f"      - {team}")
            if len(lines_only) > 10:
                print(f"      ... and {len(lines_only) - 10} more")
        
        # Find fuzzy matches if requested
        if args.find_school_matches and lines_only:
            print(f"\n🔍 Finding fuzzy matches for unmatched teams...")
            
            # Get all outcome team names
            outcomes_teams = set()
            for col in ['HOME_TEAM', 'AWAY_TEAM']:
                if col in outcomes_df.columns:
                    outcomes_teams.update(outcomes_df[col].dropna().unique())
            
            fuzzy_matches = find_fuzzy_matches(lines_only, outcomes_teams, threshold=60)
            
            if fuzzy_matches:
                print(f"\n   💡 Suggested mappings for TEAM_NAME_MAP:")
                print(f"   {'='*70}")
                
                for lines_team in sorted(fuzzy_matches.keys()):
                    matches = fuzzy_matches[lines_team]
                    if matches:
                        best_match, best_score = matches[0]
                        print(f"\n   '{lines_team.lower()}': '{best_match}',")
                        print(f"      Confidence: {best_score:.1f}%")
                        
                        if len(matches) > 1:
                            print(f"      Other matches:")
                            for match, score in matches[1:3]:  # Show next 2
                                print(f"        - {match} ({score:.1f}%)")
                
                print(f"\n   {'='*70}")
                print(f"\n   💡 Copy the mappings above into the TEAM_NAME_MAP dictionary")
            else:
                print(f"\n   No fuzzy matches found above threshold")
        
        # Join data
        joined_df, stats = join_outcomes_and_lines(outcomes_df, lines_df)
        
        # Add season column
        joined_df['season'] = season
        
        all_joined_dfs.append(joined_df)
        all_stats.append({'season': season, **stats})
        
        print(f"\n✅ {season} processing complete")
    
    # Combine all seasons
    if all_joined_dfs:
        print(f"\n{'='*80}")
        print("COMBINING ALL SEASONS")
        print(f"{'='*80}")
        
        final_df = pd.concat(all_joined_dfs, ignore_index=True)
        
        print(f"\n📊 Final Dataset:")
        print(f"   Total games: {len(final_df):,}")
        print(f"   Games with lines: {final_df['consensus_spread'].notna().sum():,}")
        print(f"   Coverage: {final_df['consensus_spread'].notna().sum() / len(final_df) * 100:.1f}%")
        print(f"   Date range: {final_df['GAME_DATE'].min()} to {final_df['GAME_DATE'].max()}")
        
        # Summary by season
        print(f"\n   By Season:")
        for stat in all_stats:
            print(f"      {stat['season']}: {stat['matched']:,} games ({stat['coverage_pct']:.1f}% coverage)")
        
        # Save results
        if args.test:
            LOCAL_TEST_DIR.mkdir(parents=True, exist_ok=True)
            output_path = LOCAL_TEST_DIR / 'ncaab_joined_outcomes_lines.csv'
            final_df.to_csv(output_path, index=False)
            print(f"\n💾 Saved locally: {output_path}")
        
        if args.s3:
            # Save to S3
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            s3_key = f"{OUTPUT_PATH}ncaab_outcomes_lines_{timestamp}.csv"
            
            csv_buffer = StringIO()
            final_df.to_csv(csv_buffer, index=False)
            
            s3_client = boto3.client('s3')
            s3_client.put_object(
                Bucket=OUTPUT_BUCKET,
                Key=s3_key,
                Body=csv_buffer.getvalue(),
                ContentType='text/csv'
            )
            print(f"\n✅ Uploaded to S3: s3://{OUTPUT_BUCKET}/{s3_key}")
    
    print("\n" + "=" * 80)
    print("✅ Join complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()

