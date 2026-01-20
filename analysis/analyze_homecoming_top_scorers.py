#!/usr/bin/env python3
"""
Top 20 Scorers Homecoming Analysis

Research Question:
    Do the league's top scorers cover their POINTS props at different rates
    in homecoming games vs regular games?

Approach:
    1. Focus on top 20 scorers by PPG (2025-26 season)
    2. Load ALL historical points O/U props (as far back as data exists)
    3. Identify homecoming games (birth state or college state)
    4. Compare over/under rates: homecoming vs non-homecoming

Context:
    Inspired by Jalen Johnson under miss in homecoming game.
    This focuses specifically on top scorers to see if stars show
    different behavior in homecoming situations.

Usage:
    cd betting
    python analysis/analyze_homecoming_top_scorers.py
    
    # Save detailed results
    python analysis/analyze_homecoming_top_scorers.py --save results/homecoming_scorers.csv
    
    # Show per-player breakdown
    python analysis/analyze_homecoming_top_scorers.py --by-player

Author: Thomas Myles
Date: 2025-01-19
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse
from datetime import datetime
import boto3
from io import StringIO
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from player_name_utils import normalize_player_name

# Load config for markets
CONFIG_PATH = Path(__file__).parent.parent / 'config' / 'config.yaml'
with open(CONFIG_PATH, 'r') as f:
    CONFIG = yaml.safe_load(f)

# =============================================================================
# MARKET CONFIGURATION
# =============================================================================

# Map market names to game log stat columns
MARKET_CONFIG = {
    'player_points': {
        'stat_col': 'PTS',
        'display_name': 'Points',
        'unit': 'pts'
    },
    'player_rebounds': {
        'stat_col': 'REB',
        'display_name': 'Rebounds',
        'unit': 'reb'
    },
    'player_assists': {
        'stat_col': 'AST',
        'display_name': 'Assists',
        'unit': 'ast'
    },
    'player_threes': {
        'stat_col': 'FG3M',
        'display_name': '3-Pointers Made',
        'unit': '3PM'
    },
    'player_steals': {
        'stat_col': 'STL',
        'display_name': 'Steals',
        'unit': 'stl'
    },
    'player_blocks': {
        'stat_col': 'BLK',
        'display_name': 'Blocks',
        'unit': 'blk'
    },
    'player_points_rebounds_assists': {
        'stat_col': 'PRA',  # Will be calculated
        'display_name': 'Points+Rebounds+Assists',
        'unit': 'PRA'
    },
    'player_double_double': {
        'stat_col': 'DOUBLE_DOUBLE',  # Will be calculated (1 if DD, 0 if not)
        'display_name': 'Double-Double',
        'unit': 'DD'
    },
    'player_triple_double': {
        'stat_col': 'TRIPLE_DOUBLE',  # Will be calculated (1 if TD, 0 if not)
        'display_name': 'Triple-Double',
        'unit': 'TD'
    },
}

# Get available markets from config (fallback to MARKET_CONFIG keys)
AVAILABLE_MARKETS = CONFIG.get('odds_api', {}).get('markets', {}).get('player_props', list(MARKET_CONFIG.keys()))

# =============================================================================
# CONFIGURATION - TOP 20 SCORERS WITH BIRTHPLACE INFO
# =============================================================================

# TODO: Fill in birth_state and college_state for each player
# birth_state: 2-letter state code where player was born (or 'INTL' for international)
# college_state: 2-letter state code where player went to college (or None if no college/international)

TOP_20_SCORERS = {
    'Luka Doncic': {
        'ppg': 35.2,
        'birth_state': 'INTL',  # Slovenia
        'college_state': None,
    },
    'Shai Gilgeous-Alexander': {
        'ppg': 32.8,
        'birth_state': 'INTL',  # Canada (Hamilton)
        'college_state': 'KY',  # Kentucky (no NBA team in KY)
    },
    'Tyrese Maxey': {
        'ppg': 32.6,
        'birth_state': 'TX',  # Garland, Texas
        'college_state': 'KY',  # Kentucky
    },
    'Donovan Mitchell': {
        'ppg': 30.7,
        'birth_state': 'NY',  # Elmsford, New York
        'college_state': 'KY',  # Louisville (Kentucky)
    },
    'Giannis Antetokounmpo': {
        'ppg': 30.6,
        'birth_state': 'INTL',  # Greece
        'college_state': None,
    },
    'Anthony Edwards': {
        'ppg': 30.2,
        'birth_state': 'GA',  # Atlanta, Georgia
        'college_state': 'GA',  # Georgia
    },
    'Jalen Green': {
        'ppg': 29.0,
        'birth_state': 'CA',  # Merced, California
        'college_state': None,  # G-League Ignite
    },
    'Jaylen Brown': {
        'ppg': 29.0,
        'birth_state': 'GA',  # Marietta, Georgia
        'college_state': 'CA',  # California
    },
    'Austin Reaves': {
        'ppg': 28.9,
        'birth_state': 'AR',  # Newark, Arkansas (no NBA team in AR)
        'college_state': 'OK',  # Oklahoma
    },
    'Nikola Jokic': {
        'ppg': 28.7,
        'birth_state': 'INTL',  # Serbia
        'college_state': None,
    },
    'Lauri Markkanen': {
        'ppg': 28.1,
        'birth_state': 'INTL',  # Finland
        'college_state': 'AZ',  # Arizona
    },
    'Stephen Curry': {
        'ppg': 27.9,
        'birth_state': 'OH',  # Akron, Ohio
        'college_state': 'NC',  # Davidson (North Carolina)
    },
    'Jalen Brunson': {
        'ppg': 27.6,
        'birth_state': 'IL',  # New Brunswick, New Jersey (no NBA team in NJ) [SWITCHED TO IL, BECAUSE OF STEVENSON HS]
        'college_state': 'PA',  # Villanova (Pennsylvania)
    },
    'Cade Cunningham': {
        'ppg': 27.6,
        'birth_state': 'TX',  # Arlington, Texas
        'college_state': 'OK',  # Oklahoma State
    },
    'James Harden': {
        'ppg': 26.9,
        'birth_state': 'CA',  # Los Angeles, California
        'birth_city': 'Los Angeles',  # City-specific: only LAL/LAC count as homecoming
        'college_state': 'AZ',  # Arizona State
    },
    'Victor Wembanyama': {
        'ppg': 26.2,
        'birth_state': 'INTL',  # France
        'college_state': None,
    },
    'Kawhi Leonard': {
        'ppg': 25.9,
        'birth_state': 'CA',  # Los Angeles, California
        'birth_city': 'Los Angeles',  # City-specific: only LAL/LAC count as homecoming
        'college_state': 'CA',  # San Diego State
    },
    'Deni Avdija': {
        'ppg': 25.8,
        'birth_state': 'INTL',  # Israel
        'college_state': None,
    },
    'Michael Porter Jr.': {
        'ppg': 25.3,
        'birth_state': 'MO',  # Columbia, Missouri (no NBA team in MO)
        'college_state': 'MO',  # Missouri
    },
    'Kevin Durant': {
        'ppg': 25.0,
        'birth_state': 'DC',  # Washington, D.C. (treated as state for homecoming)
        'college_state': 'TX',  # Texas
    },
}

# =============================================================================
# NBA TEAM TO STATE MAPPING
# =============================================================================

# State-centric mapping: each state lists its NBA teams
# Format: {state_code: [team_abbrevs]}
# Sorted A-Z by state, includes all US states (empty list if no teams)
STATE_NBA_TEAMS = {
    'AL': [],  # Alabama
    'AK': [],  # Alaska
    'AZ': ['PHX'],  # Arizona - Phoenix Suns
    'AR': [],  # Arkansas
    'CA': ['GSW', 'LAC', 'LAL', 'SAC'],  # California - Warriors, Clippers, Lakers, Kings
    'CO': ['DEN'],  # Colorado - Denver Nuggets
    'CT': [],  # Connecticut
    'DC': ['WAS'],  # Washington, D.C. - Wizards (treated as state for homecoming)
    'DE': [],  # Delaware
    'FL': ['MIA', 'ORL'],  # Florida - Heat, Magic
    'GA': ['ATL'],  # Georgia - Atlanta Hawks
    'HI': [],  # Hawaii
    'ID': [],  # Idaho
    'IL': ['CHI'],  # Illinois - Chicago Bulls
    'IN': ['IND'],  # Indiana - Indiana Pacers
    'IA': [],  # Iowa
    'KS': [],  # Kansas
    'KY': [],  # Kentucky
    'LA': ['NOP'],  # Louisiana - New Orleans Pelicans
    'ME': [],  # Maine
    'MD': [],  # Maryland
    'MA': ['BOS'],  # Massachusetts - Boston Celtics
    'MI': ['DET'],  # Michigan - Detroit Pistons
    'MN': ['MIN'],  # Minnesota - Minnesota Timberwolves
    'MS': [],  # Mississippi
    'MO': [],  # Missouri
    'MT': [],  # Montana
    'NE': [],  # Nebraska
    'NV': [],  # Nevada
    'NH': [],  # New Hampshire
    'NJ': [],  # New Jersey
    'NM': [],  # New Mexico
    'NY': ['BKN', 'NYK'],  # New York - Nets, Knicks
    'NC': ['CHA'],  # North Carolina - Charlotte Hornets
    'ND': [],  # North Dakota
    'OH': ['CLE'],  # Ohio - Cleveland Cavaliers
    'OK': ['OKC'],  # Oklahoma - Oklahoma City Thunder
    'ON': ['TOR'],  # Ontario, Canada - Toronto Raptors
    'OR': ['POR'],  # Oregon - Portland Trail Blazers
    'PA': ['PHI'],  # Pennsylvania - Philadelphia 76ers
    'RI': [],  # Rhode Island
    'SC': [],  # South Carolina
    'SD': [],  # South Dakota
    'TN': ['MEM'],  # Tennessee - Memphis Grizzlies
    'TX': ['DAL', 'HOU', 'SAS'],  # Texas - Mavericks, Rockets, Spurs
    'UT': ['UTA'],  # Utah - Utah Jazz
    'VT': [],  # Vermont
    'VA': [],  # Virginia
    'WA': [],  # Washington
    'WV': [],  # West Virginia
    'WI': ['MIL'],  # Wisconsin - Milwaukee Bucks
    'WY': [],  # Wyoming
}

# Reverse mapping: team -> state (for backward compatibility)
NBA_TEAM_STATES = {
    team: state 
    for state, teams in STATE_NBA_TEAMS.items() 
    for team in teams
}

# =============================================================================
# HOMECOMING CONFIGURATION
# =============================================================================

# Teams that DON'T trigger homecoming excitement (even if in same state)
# Reasoning: Less fan interest, smaller market, not culturally significant
HOMECOMING_BAN_LIST = {
    'SAC',  # Sacramento Kings - CA but not culturally significant for LA players
    'UTA',  # Utah Jazz - No major cities
}

# City-specific homecoming rules
# Format: {birth_city: [allowed_teams]}
# Players from these cities only get homecoming for specific teams, not entire state
CITY_SPECIFIC_HOMECOMING = {
    'Los Angeles': ['LAL', 'LAC'],  # LA players -> only Lakers/Clippers, NOT Warriors/Kings
    'San Francisco': ['GSW'],  # SF players -> only Warriors
    'Oakland': ['GSW'],  # Oakland players -> only Warriors
    'San Diego': ['LAL', 'LAC'],  # SD players -> Lakers/Clippers (closest)
}

# Note: DC is treated as a "state" for homecoming purposes
# Washington, D.C. has its own NBA team (Wizards) and distinct identity

# =============================================================================
# S3 CONFIGURATION
# =============================================================================

S3_BUCKET_PROPS = 'the-odds-api-mt'
S3_PREFIX_PROPS = 'nba/historical_player_props'

S3_BUCKET_GAME_LOGS = 'nba-api-mt'
S3_PREFIX_GAME_LOGS = 'player_game_logs'

# Seasons to load (will load all available data from these seasons)
SEASONS_TO_LOAD = ['2023-24', '2024-25', '2025-26']  # 3 seasons of historical data

# Path to season dates config
PROJECT_ROOT = Path(__file__).parent.parent
SEASON_DATES_CONFIG = PROJECT_ROOT / 'config' / 'season_dates.yaml'

# Cache configuration
CACHE_DIR = Path.home() / 'Downloads' / 'tmp'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Cache file names (include seasons in name for easy identification)
PROPS_CACHE_FILE = CACHE_DIR / 'homecoming_top_scorers_props_cache.parquet'
GAMELOGS_CACHE_FILE = CACHE_DIR / 'homecoming_top_scorers_gamelogs_cache.parquet'

# Cache expiration (hours) - refresh if cache is older than this
CACHE_MAX_AGE_HOURS = 24

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_player_homecoming_info(player_name):
    """
    Get homecoming info for a player from TOP_20_SCORERS config.
    
    Returns:
        Dict with birth_state and college_state, or None if not in top 20
    """
    if player_name in TOP_20_SCORERS:
        return TOP_20_SCORERS[player_name]
    
    # Also try normalized name matching
    normalized_search = normalize_player_name(player_name)
    for name, info in TOP_20_SCORERS.items():
        if normalize_player_name(name) == normalized_search:
            return info
    
    return None


def is_cache_valid(cache_file):
    """
    Check if cache file exists and is not expired.
    
    Args:
        cache_file: Path to cache file
        
    Returns:
        True if cache is valid, False otherwise
    """
    if not cache_file.exists():
        return False
    
    # Check age
    cache_age = datetime.now().timestamp() - cache_file.stat().st_mtime
    cache_age_hours = cache_age / 3600
    
    if cache_age_hours > CACHE_MAX_AGE_HOURS:
        print(f"   Cache expired ({cache_age_hours:.1f} hours old, max {CACHE_MAX_AGE_HOURS} hours)")
        return False
    
    print(f"   Cache valid ({cache_age_hours:.1f} hours old)")
    return True


def load_from_cache(cache_file, data_type='data'):
    """
    Load data from cache file.
    
    Args:
        cache_file: Path to cache file
        data_type: Description of data (for logging)
        
    Returns:
        DataFrame or None
    """
    try:
        print(f"📂 Loading {data_type} from cache: {cache_file.name}")
        df = pd.read_parquet(cache_file)
        print(f"   ✅ Loaded {len(df):,} rows from cache")
        return df
    except Exception as e:
        print(f"   ⚠️  Error loading cache: {e}")
        return None


def save_to_cache(df, cache_file, data_type='data'):
    """
    Save data to cache file.
    
    Args:
        df: DataFrame to save
        cache_file: Path to cache file
        data_type: Description of data (for logging)
    """
    try:
        print(f"\n💾 Saving {data_type} to cache: {cache_file.name}")
        df.to_parquet(cache_file, index=False)
        print(f"   ✅ Cached {len(df):,} rows")
    except Exception as e:
        print(f"   ⚠️  Error saving cache: {e}")


def load_season_dates_config():
    """
    Load season dates from config/season_dates.yaml.
    
    Returns:
        Dict with NBA season dates
    """
    try:
        with open(SEASON_DATES_CONFIG, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('nba', {})
    except Exception as e:
        print(f"⚠️  Error loading season_dates.yaml: {e}")
        return {}


def filter_to_regular_season(df, season):
    """
    Filter DataFrame to ONLY regular season games.
    Uses config/season_dates.yaml to determine date range.
    
    Args:
        df: DataFrame with 'date' column
        season: Season string (e.g., '2023-24')
        
    Returns:
        Filtered DataFrame (regular season only)
    """
    season_config = load_season_dates_config()
    
    if season not in season_config:
        print(f"   ⚠️  Season {season} not in config - including all dates")
        return df
    
    season_info = season_config[season]
    start_date = pd.to_datetime(season_info['season_start'])
    end_date = pd.to_datetime(season_info['regular_season_end'])
    
    # Filter to regular season date range
    original_count = len(df)
    df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
    filtered_count = len(df_filtered)
    
    print(f"   {season}: {original_count:,} total → {filtered_count:,} regular season ({original_count - filtered_count:,} playoff games excluded)")
    
    return df_filtered


def load_all_historical_props(use_cache=True, market='player_points'):
    """
    Load ALL historical player props from S3, cache all markets, then filter by market.
    
    S3 location: s3://the-odds-api-mt/nba/historical_player_props/{season}/{date}.csv
    
    Args:
        use_cache: If True, use cached data if available
        market: Market to filter (e.g., 'player_points', 'player_rebounds')
    
    Returns:
        DataFrame with all historical props for specified market
    """
    # Single cache file for ALL markets
    props_cache_file = CACHE_DIR / 'homecoming_top_scorers_props_all_markets_cache.parquet'
    
    # Try loading from cache first
    if use_cache:
        if is_cache_valid(props_cache_file):
            cached_data = load_from_cache(props_cache_file, 'all props data')
            if cached_data is not None:
                # Filter to requested market
                market_data = cached_data[cached_data['market'] == market].copy()
                print(f"\n✅ Loaded and filtered to {MARKET_CONFIG[market]['display_name']} ({len(market_data):,} props)")
                return market_data
        else:
            print("\n   Cache not found or expired - loading from S3...")
    
    print(f"\n📊 Loading historical player props from S3 (ALL markets)...")
    print(f"   Bucket: s3://{S3_BUCKET_PROPS}/{S3_PREFIX_PROPS}/")
    
    s3_client = boto3.client('s3')
    all_props = []
    
    # Load each season
    for season in SEASONS_TO_LOAD:
        print(f"\n   Loading season: {season}...")
        prefix = f"{S3_PREFIX_PROPS}/{season}/"
        
        try:
            # List all CSV files in this season
            response = s3_client.list_objects_v2(
                Bucket=S3_BUCKET_PROPS,
                Prefix=prefix
            )
            
            if 'Contents' not in response:
                print(f"      ⚠️  No data found for {season}")
                continue
            
            csv_files = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.csv')]
            print(f"      Found {len(csv_files)} files")
            
            season_props = []
            for i, key in enumerate(csv_files):
                try:
                    # Load CSV from S3
                    obj = s3_client.get_object(Bucket=S3_BUCKET_PROPS, Key=key)
                    df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
                    
                    # Keep ALL markets (filter after caching)
                    if not df.empty:
                        # Extract date from filename (already in ET timezone)
                        # Example: nba/historical_player_props/2023-24/2023-10-26.csv -> 2023-10-26
                        filename = key.split('/')[-1].replace('.csv', '')
                        df['date'] = pd.to_datetime(filename)
                        
                        # Rename prop_line to line for consistency
                        if 'prop_line' in df.columns:
                            df['line'] = df['prop_line']
                        
                        season_props.append(df)
                    
                    # Progress indicator
                    if (i + 1) % 20 == 0:
                        print(f"         Processed {i+1}/{len(csv_files)} files...")
                        
                except Exception as e:
                    print(f"      ⚠️  Error loading {key}: {e}")
            
            if season_props:
                season_df = pd.concat(season_props, ignore_index=True)
                all_props.append(season_df)
                print(f"      ✓ Loaded {len(season_df):,} props from {season}")
        
        except Exception as e:
            print(f"      ❌ Error accessing S3 for {season}: {e}")
    
    if not all_props:
        print("\n❌ No historical props data found!")
        print(f"\n📝 S3 location: s3://{S3_BUCKET_PROPS}/{S3_PREFIX_PROPS}/{{season}}/{{date}}.csv")
        return pd.DataFrame()
    
    # Combine all seasons
    df_all = pd.concat(all_props, ignore_index=True)
    
    # Add normalized player name
    if 'player_normalized' not in df_all.columns and 'player' in df_all.columns:
        df_all['player_normalized'] = df_all['player'].apply(normalize_player_name)
    
    print(f"\n✅ Total historical props loaded: {len(df_all):,}")
    if 'date' in df_all.columns and not df_all.empty:
        print(f"   Date range: {df_all['date'].min()} to {df_all['date'].max()}")
    if 'player' in df_all.columns:
        print(f"   Unique players: {df_all['player'].nunique():,}")
    
    # Filter to ONLY regular season games
    print(f"\n🏀 Filtering to regular season only (excluding playoffs)...")
    df_regular_season = pd.DataFrame()
    
    for season in SEASONS_TO_LOAD:
        season_props = df_all[df_all['date'].dt.year.isin([int(season[:4]), int(season[5:7]) + 2000])]
        filtered = filter_to_regular_season(season_props, season)
        df_regular_season = pd.concat([df_regular_season, filtered], ignore_index=True)
    
    print(f"\n✅ Regular season props: {len(df_regular_season):,}")
    print(f"   Excluded {len(df_all) - len(df_regular_season):,} playoff props")
    
    # Always save ALL markets to cache (regardless of use_cache flag)
    save_to_cache(df_regular_season, props_cache_file, 'all props data')
    
    # Filter to requested market before returning
    market_data = df_regular_season[df_regular_season['market'] == market].copy()
    print(f"\n🎯 Filtered to {MARKET_CONFIG[market]['display_name']}: {len(market_data):,} props")
    
    return market_data


def load_game_logs_multi_season(use_cache=True):
    """
    Load game logs for multiple seasons from S3.
    
    S3 location: s3://nba-api-mt/player_game_logs/{season}/{date}.csv
    
    Args:
        use_cache: If True, use cached data if available
    
    Returns:
        DataFrame with all game logs
    """
    # Try loading from cache first
    if use_cache:
        if is_cache_valid(GAMELOGS_CACHE_FILE):
            cached_data = load_from_cache(GAMELOGS_CACHE_FILE, 'game logs')
            if cached_data is not None:
                print("\n✅ Loaded game logs from cache - skipping S3 download")
                return cached_data
        else:
            print("\n   Cache not found or expired - loading from S3...")
    
    print("\n📊 Loading game logs from S3...")
    print(f"   Bucket: s3://{S3_BUCKET_GAME_LOGS}/{S3_PREFIX_GAME_LOGS}/")
    
    s3_client = boto3.client('s3')
    all_logs = []
    
    # Load each season
    for season in SEASONS_TO_LOAD:
        print(f"\n   Loading season: {season}...")
        prefix = f"{S3_PREFIX_GAME_LOGS}/{season}/"
        
        try:
            # List all CSV files in this season
            response = s3_client.list_objects_v2(
                Bucket=S3_BUCKET_GAME_LOGS,
                Prefix=prefix
            )
            
            if 'Contents' not in response:
                print(f"      ⚠️  No data found for {season}")
                continue
            
            csv_files = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.csv')]
            print(f"      Found {len(csv_files)} files")
            
            season_logs = []
            for i, key in enumerate(csv_files):
                try:
                    # Load CSV from S3
                    obj = s3_client.get_object(Bucket=S3_BUCKET_GAME_LOGS, Key=key)
                    df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
                    
                    # Parse date from GAME_DATE column (actual column name in S3)
                    df['date'] = pd.to_datetime(df['GAME_DATE'])
                    
                    # Rename PLAYER_NAME to player for consistency
                    if 'PLAYER_NAME' in df.columns:
                        df['player'] = df['PLAYER_NAME']
                    
                    # Add normalized player name
                    if 'player' in df.columns:
                        df['player_normalized'] = df['player'].apply(normalize_player_name)
                    
                    # Parse matchup to get opponent and home/away
                    if 'MATCHUP' in df.columns:
                        df['opponent'] = df['MATCHUP'].apply(lambda x: x.split(' @ ')[1] if ' @ ' in x else x.split(' vs. ')[1])
                        df['home_away'] = df['MATCHUP'].apply(lambda x: 'AWAY' if ' @ ' in x else 'HOME')
                    
                    # Rename PTS to pts for consistency
                    if 'PTS' in df.columns:
                        df['pts'] = df['PTS']
                    
                    season_logs.append(df)
                    
                    # Progress indicator
                    if (i + 1) % 20 == 0:
                        print(f"         Processed {i+1}/{len(csv_files)} files...")
                        
                except Exception as e:
                    print(f"      ⚠️  Error loading {key}: {e}")
            
            if season_logs:
                season_df = pd.concat(season_logs, ignore_index=True)
                
                # Ensure player_normalized exists after concat
                if 'player_normalized' not in season_df.columns and 'player' in season_df.columns:
                    season_df['player_normalized'] = season_df['player'].apply(normalize_player_name)
                
                all_logs.append(season_df)
                print(f"      ✓ Loaded {len(season_df):,} game logs from {season}")
        
        except Exception as e:
            print(f"      ❌ Error accessing S3 for {season}: {e}")
    
    if not all_logs:
        print("❌ No game logs found!")
        print(f"\n📝 S3 location: s3://{S3_BUCKET_GAME_LOGS}/{S3_PREFIX_GAME_LOGS}/{{season}}/{{date}}.csv")
        return pd.DataFrame()
    
    df_all = pd.concat(all_logs, ignore_index=True)
    
    print(f"\n✅ Total game logs loaded: {len(df_all):,}")
    print(f"   Date range: {df_all['date'].min()} to {df_all['date'].max()}")
    print(f"   Unique players: {df_all['player'].nunique():,}")
    
    # Filter to ONLY regular season games
    print(f"\n🏀 Filtering to regular season only (excluding playoffs)...")
    df_regular_season = pd.DataFrame()
    
    for season in SEASONS_TO_LOAD:
        # Get year range for season (e.g., 2023-24 -> 2023 and 2024)
        start_year = int(season[:4])
        end_year = start_year + 1
        season_logs = df_all[df_all['date'].dt.year.isin([start_year, end_year])]
        filtered = filter_to_regular_season(season_logs, season)
        df_regular_season = pd.concat([df_regular_season, filtered], ignore_index=True)
    
    print(f"\n✅ Regular season game logs: {len(df_regular_season):,}")
    print(f"   Excluded {len(df_all) - len(df_regular_season):,} playoff games")
    
    # Add calculated stat columns
    if 'PTS' in df_regular_season.columns and 'REB' in df_regular_season.columns and 'AST' in df_regular_season.columns:
        # PRA (Points + Rebounds + Assists)
        df_regular_season['PRA'] = df_regular_season['PTS'] + df_regular_season['REB'] + df_regular_season['AST']
        
        # Double-Double: 10+ in any 2 of PTS, REB, AST, STL, BLK
        stats = [df_regular_season['PTS'], df_regular_season['REB'], df_regular_season['AST']]
        if 'STL' in df_regular_season.columns:
            stats.append(df_regular_season['STL'])
        if 'BLK' in df_regular_season.columns:
            stats.append(df_regular_season['BLK'])
        
        double_doubles = sum((stat >= 10).astype(int) for stat in stats)
        df_regular_season['DOUBLE_DOUBLE'] = (double_doubles >= 2).astype(int)
        
        # Triple-Double: 10+ in any 3 of PTS, REB, AST, STL, BLK
        df_regular_season['TRIPLE_DOUBLE'] = (double_doubles >= 3).astype(int)
    
    # Always save to cache (regardless of use_cache flag)
    save_to_cache(df_regular_season, GAMELOGS_CACHE_FILE, 'game logs')
    
    return df_regular_season


def identify_homecoming_games_for_top_scorers(game_logs_df):
    """
    Identify homecoming games for the top 20 scorers.
    
    A homecoming game is when a player plays AT their birth state or college state
    while on a DIFFERENT team (i.e., away game in their hometown).
    
    SPECIAL RULES:
    1. Ban list: Some teams don't trigger homecoming (e.g., SAC for CA players)
    2. City-specific: LA players only count LAL/LAC, not all CA teams
    
    Example: LeBron playing @ CLE while on LAL = homecoming
             LeBron playing FOR CLE = NOT homecoming (regular home game)
             James Harden @ SAC = NOT homecoming (banned team)
             James Harden @ LAL = YES homecoming (LA native, allowed team)
    
    Returns:
        DataFrame with homecoming flags added
    """
    print("\n🏠 Identifying homecoming games for top 20 scorers...")
    print("   (Must be AWAY game in birth state or college state)")
    print(f"   Ban list: {', '.join(HOMECOMING_BAN_LIST)} (excluded from homecoming)")
    
    df = game_logs_df.copy()
    
    # Add opponent state
    df['opponent_state'] = df['opponent'].map(NBA_TEAM_STATES)
    
    # Initialize homecoming columns
    df['is_homecoming'] = False
    df['homecoming_type'] = None
    df['birth_state'] = None
    df['college_state'] = None
    
    # Process each top scorer
    homecoming_counts = {}
    
    for player_name, info in TOP_20_SCORERS.items():
        birth_state = info.get('birth_state')
        birth_city = info.get('birth_city')
        college_state = info.get('college_state')
        
        # Skip international players with no college
        if (birth_state == 'INTL' or birth_state is None) and (college_state is None):
            homecoming_counts[player_name] = 0
            continue
        
        # Find this player's games
        player_normalized = normalize_player_name(player_name)
        player_mask = df['player_normalized'] == player_normalized
        
        # Mark homecoming games (AWAY games only)
        for idx in df[player_mask].index:
            opponent_team = df.at[idx, 'opponent']
            opponent_state = df.at[idx, 'opponent_state']
            home_away = df.at[idx, 'home_away']
            
            # Store states
            df.at[idx, 'birth_state'] = birth_state
            df.at[idx, 'college_state'] = college_state
            
            # CRITICAL: Must be AWAY game (playing @ their home state while on different team)
            if home_away != 'AWAY':
                continue
            
            # Check ban list
            if opponent_team in HOMECOMING_BAN_LIST:
                continue
            
            # Check for homecoming
            is_birth_home = False
            is_college_home = False
            
            # Birth state homecoming
            if birth_state and birth_state != 'INTL' and birth_state == opponent_state:
                # Check city-specific rules
                if birth_city and birth_city in CITY_SPECIFIC_HOMECOMING:
                    # City-specific: only allowed teams count
                    if opponent_team in CITY_SPECIFIC_HOMECOMING[birth_city]:
                        is_birth_home = True
                else:
                    # No city rule: any team in state counts
                    is_birth_home = True
            
            # College state homecoming (no city-specific rules for college)
            if college_state and college_state == opponent_state:
                is_college_home = True
            
            if is_birth_home or is_college_home:
                df.at[idx, 'is_homecoming'] = True
                
                if is_birth_home and is_college_home:
                    df.at[idx, 'homecoming_type'] = 'both'
                elif is_birth_home:
                    df.at[idx, 'homecoming_type'] = 'birth'
                else:
                    df.at[idx, 'homecoming_type'] = 'college'
        
        # Count homecoming games for this player
        homecoming_count = df[player_mask & df['is_homecoming']].shape[0]
        homecoming_counts[player_name] = homecoming_count
    
    total_homecoming = df['is_homecoming'].sum()
    
    print(f"✅ Found {total_homecoming:,} homecoming games total")
    print("\nHomecoming games per player:")
    for player, count in sorted(homecoming_counts.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            info = TOP_20_SCORERS[player]
            states = []
            if info.get('birth_state') not in ['INTL', None]:
                birth_info = f"birth={info['birth_state']}"
                if info.get('birth_city'):
                    birth_info += f" ({info['birth_city']})"
                states.append(birth_info)
            if info.get('college_state'):
                states.append(f"college={info['college_state']}")
            state_str = ', '.join(states) if states else 'N/A'
            print(f"   {player:25s} {count:3d} games ({state_str})")
    
    return df


def calculate_cover_rates(props_df, by_player=False, log_each_game=False, college_v_hometown=False, market_config=None):
    """
    Calculate over/under cover rates for homecoming vs non-homecoming games.
    
    Args:
        props_df: DataFrame with props and homecoming flags
        by_player: If True, show per-player breakdown
        log_each_game: If True, show detailed game-by-game breakdown for homecoming games
        college_v_hometown: If True, show college vs hometown breakdown (requires by_player)
        market_config: Dict with market configuration (display_name, unit, etc.)
        
    Returns:
        DataFrame with summary statistics
    """
    if market_config is None:
        market_config = MARKET_CONFIG['player_points']  # Default
    
    unit = market_config['unit']
    if props_df.empty:
        return pd.DataFrame()
    
    print("\n📊 Calculating cover rates...")
    
    # Ensure we have necessary columns
    if 'actual' not in props_df.columns or 'line' not in props_df.columns:
        print("❌ Missing 'actual' or 'line' columns in props data")
        return pd.DataFrame()
    
    # Calculate cover/miss
    props_df['covered_over'] = props_df['actual'] > props_df['line']
    props_df['covered_under'] = props_df['actual'] < props_df['line']
    props_df['push'] = props_df['actual'] == props_df['line']
    
    # Overall stats
    total = len(props_df)
    homecoming = props_df['is_homecoming'].sum()
    non_homecoming = (~props_df['is_homecoming']).sum()
    
    print(f"\nTotal props analyzed: {total:,}")
    print(f"  Homecoming games: {homecoming:,} ({100*homecoming/total:.1f}%)")
    print(f"  Non-homecoming: {non_homecoming:,} ({100*non_homecoming/total:.1f}%)")
    
    # Calculate rates by homecoming status
    print("\n" + "=" * 80)
    print("OVERALL RESULTS: HOMECOMING vs NON-HOMECOMING")
    print("=" * 80)
    
    for is_home, label in [(True, 'HOMECOMING'), (False, 'Non-homecoming')]:
        subset = props_df[props_df['is_homecoming'] == is_home]
        n = len(subset)
        
        if n == 0:
            continue
        
        over_count = subset['covered_over'].sum()
        under_count = subset['covered_under'].sum()
        push_count = subset['push'].sum()
        over_rate = subset['covered_over'].mean()
        under_rate = subset['covered_under'].mean()
        push_rate = subset['push'].mean()
        
        print(f"\n{label} (n={n:,}):")
        print(f"  Over:  {int(over_count)}-{int(under_count)} ({over_rate:.2%})")
        print(f"  Under: {int(under_count)}-{int(over_count)} ({under_rate:.2%})")
        if push_count > 0:
            print(f"  Push:  {int(push_count)} ({push_rate:.2%})")
    
    # Hometown vs College breakdown if requested
    if college_v_hometown and 'homecoming_type' in props_df.columns:
        print("\n" + "=" * 80)
        print("HOMECOMING BREAKDOWN: HOMETOWN vs COLLEGE")
        print("=" * 80)
        
        homecoming_props = props_df[props_df['is_homecoming']]
        
        if not homecoming_props.empty:
            # Split by type
            birth_props = homecoming_props[homecoming_props['homecoming_type'].isin(['birth', 'both'])]
            college_props = homecoming_props[homecoming_props['homecoming_type'].isin(['college', 'both'])]
            
            for props_subset, label, emoji in [
                (birth_props, 'HOMETOWN (Birth State)', '🏠'),
                (college_props, 'COLLEGE (College State)', '🏫')
            ]:
                n = len(props_subset)
                
                if n == 0:
                    continue
                
                over_count = props_subset['covered_over'].sum()
                under_count = props_subset['covered_under'].sum()
                push_count = props_subset['push'].sum()
                over_rate = props_subset['covered_over'].mean()
                under_rate = props_subset['covered_under'].mean()
                push_rate = props_subset['push'].mean()
                avg_pts = props_subset['actual'].mean()
                avg_line = props_subset['line'].mean()
                margin = avg_pts - avg_line
                
                print(f"\n{emoji} {label} (n={n:,}):")
                print(f"  Over:  {int(over_count)}-{int(under_count)} ({over_rate:.2%})")
                print(f"  Under: {int(under_count)}-{int(over_count)} ({under_rate:.2%})")
                if push_count > 0:
                    print(f"  Push:  {int(push_count)} ({push_rate:.2%})")
                print(f"  Avg margin: {margin:+.1f} {unit} (actual {avg_pts:.1f} vs line {avg_line:.1f})")
    
    # Per-player breakdown if requested
    if by_player:
        print("\n" + "=" * 80)
        print("PER-PLAYER BREAKDOWN")
        print("=" * 80)
        
        for player_name in sorted(TOP_20_SCORERS.keys()):
            player_normalized = normalize_player_name(player_name)
            player_props = props_df[props_df['player_normalized'] == player_normalized]
            
            if player_props.empty:
                continue
            
            # Count unique games (deduplicated by date)
            total_games = player_props['date'].nunique()
            overall_over = player_props['covered_over'].mean()
            
            # Homecoming for this player
            home_props = player_props[player_props['is_homecoming']]
            non_home_props = player_props[~player_props['is_homecoming']]
            
            home_games = home_props['date'].nunique() if not home_props.empty else 0
            non_home_games = non_home_props['date'].nunique() if not non_home_props.empty else 0
            
            print(f"\n{player_name}:")
            print(f"  Total games: {total_games:,} (overall over rate: {overall_over:.2%})")
            
            if home_games > 0:
                home_over = home_props['covered_over'].mean()
                print(f"  Homecoming: {home_games:3d} games, over rate: {home_over:6.2%}")
            else:
                print(f"  Homecoming: 0 games")
            
            if non_home_games > 0:
                non_home_over = non_home_props['covered_over'].mean()
                print(f"  Regular:    {non_home_games:3d} games, over rate: {non_home_over:6.2%}")
            
            if home_games > 0 and non_home_games > 0:
                diff = home_props['covered_over'].mean() - non_home_props['covered_over'].mean()
                print(f"  Difference: {diff:+.2%} (homecoming vs regular)")
            
            # Show college vs hometown breakdown if requested
            if college_v_hometown and home_games > 0 and 'homecoming_type' in home_props.columns:
                print(f"\n  🏫 College vs 🏠 Hometown Breakdown:")
                
                # Split by homecoming type
                birth_props = home_props[home_props['homecoming_type'].isin(['birth', 'both'])]
                college_props = home_props[home_props['homecoming_type'].isin(['college', 'both'])]
                
                birth_games = birth_props['date'].nunique() if not birth_props.empty else 0
                college_games = college_props['date'].nunique() if not college_props.empty else 0
                
                if birth_games > 0:
                    birth_over = birth_props['covered_over'].mean()
                    birth_avg = birth_props['actual'].mean()
                    birth_avg_line = birth_props['line'].mean()
                    birth_margin = birth_avg - birth_avg_line
                    print(f"     🏠 Hometown ({TOP_20_SCORERS[player_name].get('birth_state', '?')}): {birth_games} games, over {birth_over:.0%}, avg {birth_avg:.1f}{unit} (line {birth_avg_line:.1f}, {birth_margin:+.1f})")
                
                if college_games > 0:
                    college_over = college_props['covered_over'].mean()
                    college_avg = college_props['actual'].mean()
                    college_avg_line = college_props['line'].mean()
                    college_margin = college_avg - college_avg_line
                    print(f"     🏫 College ({TOP_20_SCORERS[player_name].get('college_state', '?')}): {college_games} games, over {college_over:.0%}, avg {college_avg:.1f}{unit} (line {college_avg_line:.1f}, {college_margin:+.1f})")
                
                if birth_games > 0 and college_games > 0:
                    diff_type = birth_over - college_over
                    print(f"     📊 Difference: {diff_type:+.1%} (hometown vs college)")
            
            # Show detailed game-by-game breakdown if requested
            if log_each_game and home_games > 0:
                print(f"\n  📋 Homecoming Games Detail:")
                
                # Calculate summary stats for homecoming games
                home_avg = home_props['actual'].mean()
                home_avg_line = home_props['line'].mean()
                margin = home_avg - home_avg_line
                
                # Get date range
                min_date = home_props['date'].min().strftime('%Y-%m-%d')
                max_date = home_props['date'].max().strftime('%Y-%m-%d')
                
                print(f"     Summary: {min_date} to {max_date}")
                print(f"     Avg {market_config['display_name']}: {home_avg:.1f} | Avg Line: {home_avg_line:.1f} | Margin: {margin:+.1f}")
                print(f"     Over Rate: {home_over:.0%} ({int(home_over * home_games)}/{home_games})")
                print()
                
                # Show each game sorted by date
                home_sorted = home_props.sort_values('date')
                for idx, row in home_sorted.iterrows():
                    date_str = row['date'].strftime('%Y-%m-%d')
                    opponent = row.get('opponent', '???')
                    result = "✅ OVER" if row['covered_over'] else ("❌ UNDER" if row['covered_under'] else "➖ PUSH")
                    margin_game = row['actual'] - row['line']
                    
                    print(f"     {date_str} (@ {opponent}): {row['actual']:.0f} {unit} (line {row['line']:.1f}) {result} ({margin_game:+.1f})")
    
    # Create summary DataFrame
    summary_rows = []
    
    for is_home in [True, False]:
        subset = props_df[props_df['is_homecoming'] == is_home]
        n = len(subset)
        
        if n > 0:
            summary_rows.append({
                'segment': 'Homecoming' if is_home else 'Non-homecoming',
                'n_props': n,
                'over_rate': subset['covered_over'].mean(),
                'under_rate': subset['covered_under'].mean(),
                'push_rate': subset['push'].mean(),
            })
    
    return pd.DataFrame(summary_rows)


def main():
    """Main analysis pipeline"""
    parser = argparse.ArgumentParser(
        description='Analyze top 20 scorers homecoming props performance'
    )
    parser.add_argument(
        '--save',
        type=str,
        help='Path to save results CSV'
    )
    parser.add_argument(
        '--by-player',
        action='store_true',
        help='Show per-player breakdown'
    )
    parser.add_argument(
        '--use-cache',
        action='store_true',
        help='Use cached data from ~/Downloads/tmp/ (much faster, refreshes every 24 hours)'
    )
    parser.add_argument(
        '--log-each-homecoming-game',
        action='store_true',
        help='Show detailed game-by-game breakdown for each homecoming game (requires --by-player)'
    )
    parser.add_argument(
        '--college-v-hometown',
        action='store_true',
        help='Show breakdown of college homecoming vs hometown homecoming (requires --by-player)'
    )
    parser.add_argument(
        '--market',
        type=str,
        default='player_points',
        choices=AVAILABLE_MARKETS,
        help=f'Market to analyze (default: player_points). Options: {", ".join(AVAILABLE_MARKETS)}'
    )
    
    args = parser.parse_args()
    
    # Get market config
    market_config = MARKET_CONFIG[args.market]
    
    print("=" * 80)
    print("TOP 20 SCORERS HOMECOMING ANALYSIS")
    print("=" * 80)
    print(f"\nMarket: {market_config['display_name']} ({args.market})")
    print("Top 20 scorers (2025-26 PPG leaders)")
    print(f"Seasons: {', '.join(SEASONS_TO_LOAD)}")
    print(f"Cache location: {CACHE_DIR}")
    if args.use_cache:
        print("✅ Using cache (add --use-cache to speed up subsequent runs)")
    else:
        print("⚠️  Cache disabled - will load from S3 (first run or use --use-cache for speed)")
    print()
    
    # Check config completion
    incomplete = []
    for player, info in TOP_20_SCORERS.items():
        if info['birth_state'] is None and info['college_state'] is None:
            incomplete.append(player)
    
    if incomplete:
        print("⚠️  WARNING: The following players need birthplace/college info:")
        for player in incomplete:
            print(f"   - {player}")
        print("\nPlease fill in the TOP_20_SCORERS config in this script.")
        print("These players will be excluded from homecoming analysis.\n")
    
    # Step 1: Load all historical props for specified market
    market_config = MARKET_CONFIG[args.market]
    props_df = load_all_historical_props(use_cache=args.use_cache, market=args.market)
    
    if props_df.empty:
        print("\n❌ Cannot proceed without props data")
        return
    
    # Filter to top 20 scorers only
    print("\n🎯 Filtering to top 20 scorers...")
    top_20_normalized = [normalize_player_name(name) for name in TOP_20_SCORERS.keys()]
    props_df = props_df[props_df['player_normalized'].isin(top_20_normalized)]
    
    print(f"✅ Filtered to {len(props_df):,} props for top 20 scorers")
    print(f"   Players with props: {props_df['player'].nunique()}")
    
    # Step 2: Load game logs
    game_logs = load_game_logs_multi_season(use_cache=args.use_cache)
    
    if game_logs.empty:
        print("\n❌ Cannot proceed without game logs")
        return
    
    # Filter to top 20 scorers
    game_logs = game_logs[game_logs['player_normalized'].isin(top_20_normalized)]
    
    # Step 3: Identify homecoming games
    game_logs_with_homecoming = identify_homecoming_games_for_top_scorers(game_logs)
    
    # Step 4: Take consensus line per player per game
    # Filter to main market lines (odds close to -110) to avoid alternate lines
    print("\n📊 Taking consensus lines (1 line per player per game)...")
    print("   Filtering to main market lines (odds near -110, excluding alternates)")
    
    # Filter to lines with standard odds (between -140 and -100)
    # Alternate lines have very different odds (e.g., -200, +150, etc.)
    if 'over_odds' in props_df.columns and 'under_odds' in props_df.columns:
        main_market = props_df[
            ((props_df['over_odds'] >= -140) & (props_df['over_odds'] <= -100)) |
            ((props_df['under_odds'] >= -140) & (props_df['under_odds'] <= -100))
        ].copy()
        print(f"   Filtered {len(props_df):,} → {len(main_market):,} props (removed alternate lines)")
    else:
        print("   ⚠️  No odds columns found, using all lines")
        main_market = props_df.copy()
    
    # Group by player + date and take median of main market lines
    consensus_props = main_market.groupby(['player_normalized', 'date']).agg({
        'player': 'first',
        'line': 'median',  # Median of main market lines
        'market': 'first',
    }).reset_index()
    
    print(f"   Reduced to {len(consensus_props):,} consensus props (1 per player per game)")
    
    # Step 5: Join with game logs to get actual points scored
    print("\n🔗 Joining props with game logs...")
    
    # DEBUG: Check date ranges and types
    print(f"\n🔍 DEBUG - Props date range:")
    print(f"   Type: {consensus_props['date'].dtype}")
    print(f"   Min: {consensus_props['date'].min()}")
    print(f"   Max: {consensus_props['date'].max()}")
    print(f"   Sample dates: {consensus_props['date'].head(5).tolist()}")
    
    print(f"\n🔍 DEBUG - Game logs date range:")
    print(f"   Type: {game_logs['date'].dtype}")
    print(f"   Min: {game_logs['date'].min()}")
    print(f"   Max: {game_logs['date'].max()}")
    print(f"   Sample dates: {game_logs['date'].head(5).tolist()}")
    
    print(f"\n🔍 DEBUG - Sample player matches BEFORE join:")
    for player_norm in list(top_20_normalized)[:5]:
        props_count = len(consensus_props[consensus_props['player_normalized'] == player_norm])
        logs_count = len(game_logs[game_logs['player_normalized'] == player_norm])
        print(f"   {player_norm}: {props_count} props, {logs_count} game logs")
    
    # DEBUG: Check for date overlap issues
    print(f"\n🔍 DEBUG - Checking why join is losing data...")
    sample_player = list(top_20_normalized)[0]
    sample_props = consensus_props[consensus_props['player_normalized'] == sample_player].copy()
    sample_logs = game_logs[game_logs['player_normalized'] == sample_player].copy()
    
    print(f"\n   Sample player: {sample_player}")
    print(f"   Props dates (first 10): {sorted(sample_props['date'].unique())[:10]}")
    print(f"   Logs dates (first 10): {sorted(sample_logs['date'].unique())[:10]}")
    
    # Find dates in props but not in logs
    props_dates = set(sample_props['date'])
    logs_dates = set(sample_logs['date'])
    only_in_props = props_dates - logs_dates
    only_in_logs = logs_dates - props_dates
    
    print(f"\n   Dates only in props: {len(only_in_props)}")
    if len(only_in_props) > 0:
        print(f"      Sample: {sorted(list(only_in_props))[:5]}")
    print(f"   Dates only in logs: {len(only_in_logs)}")
    if len(only_in_logs) > 0:
        print(f"      Sample: {sorted(list(only_in_logs))[:5]}")
    
    # Get stat column for this market
    stat_col = market_config['stat_col']
    
    props_with_actuals = consensus_props.merge(
        game_logs[['player_normalized', 'date', stat_col]],
        on=['player_normalized', 'date'],
        how='inner',
        suffixes=('', '_actual')
    )
    
    # Rename stat column to 'actual' for consistency
    if stat_col in props_with_actuals.columns:
        props_with_actuals['actual'] = props_with_actuals[stat_col]
    
    print(f"✅ Matched {len(props_with_actuals):,} props with actual game results")
    
    print(f"\n🔍 DEBUG - Sample player matches AFTER join:")
    for player_norm in list(top_20_normalized)[:5]:
        matched_count = len(props_with_actuals[props_with_actuals['player_normalized'] == player_norm])
        props_count = len(consensus_props[consensus_props['player_normalized'] == player_norm])
        logs_count = len(game_logs[game_logs['player_normalized'] == player_norm])
        print(f"   {player_norm}: {matched_count} matched (from {props_count} props, {logs_count} logs)")
        
        # Check for mismatches
        if matched_count < min(props_count, logs_count) * 0.5:
            print(f"      ⚠️  LOW MATCH RATE! Expected ~{min(props_count, logs_count)}, got {matched_count}")
    
    # DEBUG: Show sample data
    print(f"\n🔍 DEBUG - Sample of props vs actuals ({market_config['display_name']}):")
    debug_sample = props_with_actuals[['player', 'date', 'line', 'actual']].head(20)
    print(debug_sample.to_string())
    print(f"\n   Line stats: min={props_with_actuals['line'].min():.1f}, max={props_with_actuals['line'].max():.1f}, mean={props_with_actuals['line'].mean():.1f}")
    print(f"   Actual stats: min={props_with_actuals['actual'].min():.1f}, max={props_with_actuals['actual'].max():.1f}, mean={props_with_actuals['actual'].mean():.1f}")
    
    # Now merge with homecoming flags (include opponent for game context)
    props_with_homecoming = props_with_actuals.merge(
        game_logs_with_homecoming[['player_normalized', 'date', 'is_homecoming', 'homecoming_type', 'birth_state', 'college_state', 'opponent']],
        on=['player_normalized', 'date'],
        how='inner'  # Inner join - only keep games where we have props, actuals, AND homecoming info
    )
    
    print(f"✅ Matched {len(props_with_homecoming):,} props with game logs")
    print(f"   Players: {props_with_homecoming['player'].nunique()}")
    print(f"   Homecoming props: {props_with_homecoming['is_homecoming'].sum():,}")
    
    # Step 5: Calculate cover rates
    summary_df = calculate_cover_rates(
        props_with_homecoming, 
        by_player=args.by_player,
        log_each_game=args.log_each_homecoming_game,
        college_v_hometown=args.college_v_hometown,
        market_config=market_config
    )
    
    # Save detailed results to Downloads/tmp
    print("\n💾 Saving detailed results to ~/Downloads/tmp/...")
    
    # 1. Summary CSV
    if not summary_df.empty:
        summary_path = CACHE_DIR / 'homecoming_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        print(f"   ✅ Summary: {summary_path}")
    
    # 2. Full detailed data (every prop with homecoming flag)
    detailed_path = CACHE_DIR / 'homecoming_detailed_props.csv'
    props_with_homecoming.to_csv(detailed_path, index=False)
    print(f"   ✅ Detailed props: {detailed_path}")
    print(f"      ({len(props_with_homecoming):,} rows - use Excel/pandas to filter by player)")
    
    # 3. Per-player summary
    player_summary_rows = []
    for player_name in sorted(TOP_20_SCORERS.keys()):
        player_normalized = normalize_player_name(player_name)
        player_props = props_with_homecoming[props_with_homecoming['player_normalized'] == player_normalized]
        
        if player_props.empty:
            continue
        
        # Count unique games (deduplicated by date)
        total_games = player_props['date'].nunique()
        
        # Overall stats
        overall_over = player_props['covered_over'].mean()
        
        # Homecoming stats
        home_props = player_props[player_props['is_homecoming']]
        non_home_props = player_props[~player_props['is_homecoming']]
        
        home_games = home_props['date'].nunique() if not home_props.empty else 0
        non_home_games = non_home_props['date'].nunique() if not non_home_props.empty else 0
        
        home_over = home_props['covered_over'].mean() if home_games > 0 else None
        non_home_over = non_home_props['covered_over'].mean() if non_home_games > 0 else None
        
        diff = (home_over - non_home_over) if (home_games > 0 and non_home_games > 0) else None
        
        player_summary_rows.append({
            'player': player_name,
            'total_games': total_games,
            'overall_over_rate': overall_over,
            'homecoming_games': home_games,
            'homecoming_over_rate': home_over,
            'regular_games': non_home_games,
            'regular_over_rate': non_home_over,
            'difference_pct': diff,
            'birth_state': TOP_20_SCORERS[player_name].get('birth_state'),
            'college_state': TOP_20_SCORERS[player_name].get('college_state'),
        })
    
    player_summary_df = pd.DataFrame(player_summary_rows)
    player_summary_path = CACHE_DIR / 'homecoming_by_player.csv'
    player_summary_df.to_csv(player_summary_path, index=False)
    print(f"   ✅ Per-player summary: {player_summary_path}")
    
    # Save custom output if requested
    if args.save and not summary_df.empty:
        output_path = Path(args.save)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        summary_df.to_csv(output_path, index=False)
        print(f"\n💾 Custom output saved to {output_path}")
    
    print("\n✅ Analysis complete!")
    print(f"\n📂 Output files in: {CACHE_DIR}")
    print("   1. homecoming_summary.csv - Overall results")
    print("   2. homecoming_detailed_props.csv - Every prop (filter by player in Excel)")
    print("   3. homecoming_by_player.csv - Per-player summary")
    print("\n💡 To view Jalen Johnson's homecoming games:")
    print("   Open homecoming_detailed_props.csv and filter:")
    print("   - player = 'Jalen Johnson'")
    print("   - is_homecoming = True")


if __name__ == '__main__':
    main()
