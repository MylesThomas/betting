"""
Shared Line Steam Detection Utilities - Multi-Sport Support

Provides sport-agnostic utilities for detecting line movement steam across
NBA, NFL, NCAAB, and NCAAF. All sport-specific configuration is externalized
to config/line_steam_config.yaml.

Core Functionality:
- Load line movement snapshots from S3
- Calculate consensus movements (opening → current)
- Detect steam threshold+ point movement
- Save plays to S3
- Load game results (NBA API or ESPN API)
- Calculate play outcomes (won/lost/push)

Used by:
- scripts/check_line_steam.py (steam detection)
- scripts/calculate_line_steam_results.py (results calculation)
- scripts/lambda_function_line_steam_alerts.py (Lambda handlers)

Author: Thomas Myles
Date: 2026-01-23
"""

import pandas as pd
import boto3
from io import BytesIO, StringIO
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
import yaml


# =============================================================================
# CONFIGURATION
# =============================================================================

def _load_line_steam_config():
    """Load line steam config from config/line_steam_config.yaml"""
    config_path = Path(__file__).parent.parent / 'config' / 'line_steam_config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class SportConfig:
    """
    Sport-specific configuration for line steam detection.
    
    Loads from config/line_steam_config.yaml and provides convenient access
    to S3 buckets, thresholds, API keys, etc.
    
    Example:
        >>> config = SportConfig('ncaab')
        >>> config.threshold
        1.0
        >>> config.s3_bucket_plays
        'ncaab-betting-mt'
    """
    
    def __init__(self, sport):
        """
        Initialize sport config.
        
        Args:
            sport: Sport key ('nba', 'nfl', 'ncaab', 'ncaaf')
        """
        config = _load_line_steam_config()
        
        if sport not in config['sports']:
            raise ValueError(f"Sport '{sport}' not found in line_steam_config.yaml. "
                           f"Available: {list(config['sports'].keys())}")
        
        sport_config = config['sports'][sport]
        
        self.sport = sport
        self.name = sport_config['name']
        self.icon = sport_config['icon']
        self.api_key = sport_config['api_key']
        self.threshold = sport_config['threshold']
        self.days_back = sport_config['days_back']
        self.s3_bucket_snapshots = sport_config['s3_bucket_snapshots']
        self.s3_bucket_plays = sport_config['s3_bucket_plays']
        self.s3_bucket_results = sport_config['s3_bucket_results']
        self.result_source = sport_config['result_source']
        self.sns_topic_arn = sport_config['sns_topic_arn']
        
        # Team name mappings (Odds API → Result APIs)
        self.team_name_map = config.get('team_name_mappings', {}).get(sport, {})
        
        # Daily report config
        daily_report_config = config.get('daily_report', {})
        self.daily_report_hour_et = daily_report_config.get('hour_et', 13)
        self.daily_report_minute_window_et = daily_report_config.get('minute_window_et', 10)
    
    def normalize_team_name(self, team_name):
        """Normalize team name from Odds API to match result API format."""
        return self.team_name_map.get(team_name, team_name)
    
    def get_s3_plays_key(self, date_str):
        """Get S3 key for plays file (date in ET timezone, YYYY-MM-DD)."""
        return f"data/04_output/plays/line-steam/{date_str}.csv"
    
    def get_s3_results_key(self, date_str):
        """Get S3 key for results file (date in ET timezone, YYYY-MM-DD)."""
        return f"data/04_output/results/line-steam/{date_str}.csv"
    
    def get_s3_snapshots_prefix(self):
        """Get S3 prefix for line movement snapshots."""
        return f"data/01_input/the-odds-api/{self.sport}/line_movement/"
    
    def __repr__(self):
        return f"SportConfig(sport='{self.sport}', name='{self.name}', threshold={self.threshold})"


# =============================================================================
# SNAPSHOT LOADING
# =============================================================================

def load_line_movement_snapshots(sport_config, days_back=None):
    """
    Load recent hourly line movement snapshots from S3.
    Only loads snapshot files from the last X days (efficient filtering by filename date).
    
    Args:
        sport_config: SportConfig instance
        days_back: Only load snapshots from the last X days (default from config)
    
    Returns:
        DataFrame with columns: game_id, game_time, away_team, home_team, 
                                bookmaker, away_spread, home_spread, etc.
    """
    if days_back is None:
        days_back = sport_config.days_back
    
    s3_prefix = sport_config.get_s3_snapshots_prefix()
    
    print(f"📥 Loading {sport_config.name} hourly snapshots from S3...")
    print(f"   S3 Bucket: {sport_config.s3_bucket_snapshots}")
    print(f"   Prefix: {s3_prefix}")
    print(f"   Days back: {days_back}")
    
    s3 = boto3.client('s3')
    
    # Calculate cutoff date (in UTC to match S3 file timestamps)
    cutoff_datetime = datetime.now(timezone.utc) - timedelta(days=days_back)
    cutoff_date_str = cutoff_datetime.strftime('%Y-%m-%d')
    print(f"   Cutoff date: {cutoff_date_str} (only loading files >= this date)")
    
    try:
        response = s3.list_objects_v2(
            Bucket=sport_config.s3_bucket_snapshots, 
            Prefix=s3_prefix
        )
    except Exception as e:
        print(f"❌ Error accessing S3 bucket: {e}")
        raise
    
    if 'Contents' not in response:
        raise ValueError(f"No snapshots found in S3 at {s3_prefix}")
    
    # Filter S3 objects by date BEFORE loading
    files_to_load = []
    total_files = 0
    
    for obj in response.get('Contents', []):
        key = obj['Key']
        total_files += 1
        
        # Only process snapshot CSV files
        if not key.endswith('.csv') or 'snapshot_' not in key:
            continue
        
        # Extract date from filename (e.g., snapshot_2026-01-14_16-00-46.csv)
        try:
            filename = key.split('/')[-1]
            if filename.startswith('snapshot_'):
                date_part = filename.split('_')[1]  # Get YYYY-MM-DD part
                
                # Compare filename date to cutoff
                if date_part >= cutoff_date_str:
                    files_to_load.append(key)
        except Exception as e:
            print(f"⚠️  Could not parse date from filename {key}: {e}")
            continue
    
    print(f"   Found {len(files_to_load)} snapshot files to load (out of {total_files} total files)")
    
    # Now load ONLY the filtered files
    all_dfs = []
    for key in files_to_load:
        try:
            response_obj = s3.get_object(Bucket=sport_config.s3_bucket_snapshots, Key=key)
            df = pd.read_csv(BytesIO(response_obj['Body'].read()))
            all_dfs.append(df)
        except Exception as e:
            print(f"⚠️  Error reading {key}: {e}")
    
    if not all_dfs:
        raise ValueError(f"No valid snapshot CSV files found for {sport_config.name}")
    
    # Combine all dataframes
    df = pd.concat(all_dfs, ignore_index=True)
    
    # Convert timestamps
    df['game_time'] = pd.to_datetime(df['game_time'])
    df['fetched_at'] = pd.to_datetime(df['fetched_at'])
    
    print(f"   Loaded {len(df):,} line records from {len(all_dfs)} snapshot files")
    print(f"   Date range: {df['fetched_at'].min()} to {df['fetched_at'].max()}")
    print(f"✅ Unique games: {df['game_id'].nunique():,}")
    
    return df


# =============================================================================
# CONSENSUS MOVEMENT CALCULATION
# =============================================================================

def calculate_consensus_movements(snapshots_df):
    """
    Calculate consensus line movements (opening → current).
    
    Args:
        snapshots_df: DataFrame with line snapshots
    
    Returns:
        DataFrame with one row per game containing:
        - Opening lines (first snapshot, averaged across books)
        - Current lines (latest snapshot, averaged across books)
        - Movement magnitude and direction
    """
    print("\n📊 Calculating consensus line movements...")
    
    # Sort by time
    snapshots_df = snapshots_df.sort_values('fetched_at')
    
    # Calculate consensus (median) at each snapshot time
    consensus = snapshots_df.groupby(['game_id', 'fetched_at']).agg({
        'game_time': 'first',
        'away_team': 'first',
        'home_team': 'first',
        'away_spread': 'median',
        'home_spread': 'median',
    }).reset_index()
    
    # Get opening (earliest) and current (latest) consensus lines
    grouped = consensus.groupby('game_id')
    opening_lines = grouped.first().reset_index()
    current_lines = grouped.last().reset_index()
    
    # Merge opening and current
    movements = opening_lines[['game_id', 'game_time', 'away_team', 'home_team', 
                                'away_spread', 'home_spread', 'fetched_at']].copy()
    movements.columns = ['game_id', 'game_time', 'away_team', 'home_team', 
                         'away_open', 'home_open', 'open_time']
    
    current_data = current_lines[['game_id', 'away_spread', 'home_spread', 'fetched_at']].copy()
    current_data.columns = ['game_id', 'away_current', 'home_current', 'current_time']
    
    movements = movements.merge(current_data, on='game_id', how='inner')
    
    # Calculate movement
    movements['away_movement'] = movements['away_current'] - movements['away_open']
    movements['home_movement'] = movements['home_current'] - movements['home_open']
    
    # Determine opening favorite/underdog
    movements['away_is_opening_favorite'] = movements['away_open'] < 0
    movements['opening_favorite'] = movements.apply(
        lambda row: row['away_team'] if row['away_is_opening_favorite'] else row['home_team'],
        axis=1
    )
    movements['opening_underdog'] = movements.apply(
        lambda row: row['home_team'] if row['away_is_opening_favorite'] else row['away_team'],
        axis=1
    )
    
    # Calculate movement from opening favorite's perspective
    movements['opening_favorite_spread_open'] = movements.apply(
        lambda row: row['away_open'] if row['away_is_opening_favorite'] else row['home_open'],
        axis=1
    )
    movements['opening_favorite_spread_current'] = movements.apply(
        lambda row: row['away_current'] if row['away_is_opening_favorite'] else row['home_current'],
        axis=1
    )
    
    # Movement formula: opening_spread - current_spread
    # Example: -7.5 - (-5.0) = -2.5 (line moved 2.5 pts TOWARD underdog)
    movements['opening_favorite_movement'] = (
        movements['opening_favorite_spread_open'] - movements['opening_favorite_spread_current']
    )
    
    # Determine steam direction and magnitude
    movements['steam_toward_opening_underdog'] = movements['opening_favorite_movement'] < 0
    movements['steam_magnitude'] = movements['opening_favorite_movement'].abs()
    
    print(f"✅ Calculated movements for {len(movements):,} games")
    
    return movements


# =============================================================================
# STEAM DETECTION
# =============================================================================

def detect_bidirectional_steam_in_plays(existing_plays_df, current_steam_direction_by_game):
    """
    Detect if a game has had steam in BOTH directions today.
    
    Args:
        existing_plays_df: DataFrame of all plays detected today (from S3)
        current_steam_direction_by_game: Dict mapping game_id to current steam_direction
    
    Returns:
        Set of game_ids that have bidirectional steam
    """
    if existing_plays_df is None or len(existing_plays_df) == 0:
        # No existing plays - check if any game appears in current with both directions
        return set()
    
    # Get all steam directions seen today (existing + current)
    all_directions = {}
    
    # Add existing detections
    for _, row in existing_plays_df.iterrows():
        game_id = row['game_id']
        direction = row['steam_direction']
        if game_id not in all_directions:
            all_directions[game_id] = set()
        all_directions[game_id].add(direction)
    
    # Add current detections
    for game_id, direction in current_steam_direction_by_game.items():
        if game_id not in all_directions:
            all_directions[game_id] = set()
        all_directions[game_id].add(direction)
    
    # Games with 2+ unique directions = bidirectional
    bidirectional_game_ids = {
        game_id for game_id, directions in all_directions.items()
        if len(directions) >= 2
    }
    
    return bidirectional_game_ids


def check_for_steam(movements_df, target_date_str, threshold, sport_name="", sport_config=None):
    """
    Check if any of today's games have threshold+ point steam (both directions).
    
    Args:
        movements_df: DataFrame with consensus movements
        target_date_str: Date string in YYYY-MM-DD format (ET timezone)
        threshold: Minimum movement in points (default from config)
        sport_name: Sport name for display (e.g., "NCAAB")
        sport_config: SportConfig instance (optional, for loading historical detections)
    
    Returns:
        tuple: (steam_detected: bool, steam_games: DataFrame or None)
    """
    print(f"\n🔍 Checking for {threshold}+ point steam (both directions)...")
    print(f"   Target date: {target_date_str} (ET timezone)")
    
    # Parse target date in ET timezone
    et_tz = ZoneInfo('America/New_York')
    target_date = datetime.strptime(target_date_str, '%Y-%m-%d').date()
    
    # Filter to games scheduled for target date (in ET)
    movements_df['game_date_et'] = movements_df['game_time'].dt.tz_convert(et_tz).dt.date
    today_games = movements_df[movements_df['game_date_et'] == target_date].copy()
    
    # Filter out games that have already started
    now_et = datetime.now(et_tz)
    today_games = today_games[today_games['game_time'] > now_et].copy()
    
    print(f"   Found {len(today_games)} games scheduled for {target_date_str}")
    
    if len(today_games) == 0:
        print(f"\n✅ STEAM_DETECTED: NO")
        print(f"   Reason: No games scheduled for {target_date_str}")
        print("\nSTEAM_DETECTED: NO")
        return False, None
    
    # Filter to ALL games with threshold+ movement (both directions)
    steam_games = today_games[
        (today_games['steam_magnitude'] >= threshold)
    ].copy()
    
    print(f"   Games with {threshold}+ pt steam: {len(steam_games)} total")
    
    if len(steam_games) == 0:
        # Show largest movement for context
        largest_movement = today_games['steam_magnitude'].max()
        largest_game = today_games.loc[today_games['steam_magnitude'].idxmax()]
        
        print(f"\n✅ STEAM_DETECTED: NO")
        print(f"   Checked {len(today_games)} games")
        print(f"   Largest movement: {largest_movement:.1f} pts")
        print(f"   Game: {largest_game['opening_favorite']} vs {largest_game['opening_underdog']}")
        print("\nSTEAM_DETECTED: NO")
        return False, None
    
    # Steam detected!
    print(f"\n🚨 STEAM_DETECTED: YES")
    print(f"   {len(steam_games)} game(s) with {threshold}+ point steam\n")
    
    # Load existing detections to track first/last detection times
    existing_plays = None
    if sport_config is not None:
        try:
            s3_key = sport_config.get_s3_plays_key(target_date_str)
            s3 = boto3.client('s3')
            response = s3.get_object(Bucket=sport_config.s3_bucket_plays, Key=s3_key)
            existing_plays = pd.read_csv(BytesIO(response['Body'].read()))
            existing_plays['detected_at'] = pd.to_datetime(existing_plays['detected_at'])
            print(f"📊 Loaded {len(existing_plays)} existing detections from today")
        except s3.exceptions.NoSuchKey:
            print(f"📊 No existing detections found for today (first check of the day)")
        except Exception as e:
            print(f"⚠️  Could not load existing detections: {e}")
    
    # Add detection tracking info to steam_games (BEFORE creating subsets)
    if existing_plays is not None and len(existing_plays) > 0:
        print(f"📊 Matching current steam to existing detections...")
        for idx, row in steam_games.iterrows():
            game_id = row['game_id']
            steam_direction = 'opening_underdog' if row['steam_toward_opening_underdog'] else 'opening_favorite'
            
            # Filter existing detections for this game + direction
            matching = existing_plays[
                (existing_plays['game_id'] == game_id) & 
                (existing_plays['steam_direction'] == steam_direction)
            ]
            
            if len(matching) > 0:
                first_detected = matching['detected_at'].min()
                steam_games.at[idx, 'first_detected_at'] = first_detected
                steam_games.at[idx, 'detection_count'] = len(matching)
                print(f"   ✅ {row['opening_favorite']} vs {row['opening_underdog']} ({steam_direction}): "
                      f"Found {len(matching)} previous detections, first at {first_detected}")
            else:
                steam_games.at[idx, 'first_detected_at'] = None
                steam_games.at[idx, 'detection_count'] = 0
                print(f"   🆕 {row['opening_favorite']} vs {row['opening_underdog']} ({steam_direction}): "
                      f"First detection (game_id: {game_id})")
    else:
        steam_games['first_detected_at'] = None
        steam_games['detection_count'] = 0
    
    # Detect bidirectional steam (steam in BOTH directions today)
    current_steam_directions = {
        row['game_id']: 'opening_underdog' if row['steam_toward_opening_underdog'] else 'opening_favorite'
        for _, row in steam_games.iterrows()
    }
    bidirectional_game_ids = detect_bidirectional_steam_in_plays(existing_plays, current_steam_directions)
    
    # Add bidirectional flag to steam_games
    steam_games['is_bidirectional_steam'] = steam_games['game_id'].isin(bidirectional_game_ids)
    
    if len(bidirectional_game_ids) > 0:
        print(f"\n⚠️  {len(bidirectional_game_ids)} game(s) with BIDIRECTIONAL STEAM detected:")
        for game_id in bidirectional_game_ids:
            game_row = steam_games[steam_games['game_id'] == game_id].iloc[0]
            print(f"   - {game_row['opening_favorite']} vs {game_row['opening_underdog']}")
        print()
    
    # Sort by steam magnitude (highest first)
    steam_games = steam_games.sort_values('steam_magnitude', ascending=False)
    
    # NOW create subsets for logging (after tracking info added)
    underdog_steam = steam_games[steam_games['steam_toward_opening_underdog']].copy()
    favorite_steam = steam_games[~steam_games['steam_toward_opening_underdog']].copy()
    
    print(f"   - Toward opening underdog: {len(underdog_steam)}")
    print(f"   - Toward opening favorite: {len(favorite_steam)}")
    
    # Output underdog steam games first
    if len(underdog_steam) > 0:
        print(f"{'='*80}")
        print(f"TOWARD OPENING UNDERDOG ({len(underdog_steam)} game{'s' if len(underdog_steam) != 1 else ''})")
        print(f"{'='*80}\n")
        
        for idx, (_, row) in enumerate(underdog_steam.sort_values('steam_magnitude', ascending=False).iterrows(), 1):
            game_time_et = row['game_time'].tz_convert(et_tz)
            
            # Add bidirectional warning emoji to game title
            bidir_warning = " ⚠️ BIDIRECTIONAL STEAM" if row.get('is_bidirectional_steam', False) else ""
            
            print(f"Game {idx}: {row['opening_favorite']} vs {row['opening_underdog']}{bidir_warning}")
            print(f"  Opening: {row['opening_favorite']} {row['opening_favorite_spread_open']:+.1f} | "
                  f"{row['opening_underdog']} {-row['opening_favorite_spread_open']:+.1f}")
            print(f"  Current: {row['opening_favorite']} {row['opening_favorite_spread_current']:+.1f} | "
                  f"{row['opening_underdog']} {-row['opening_favorite_spread_current']:+.1f}")
            print(f"  Steam: {row['steam_magnitude']:.1f} points toward opening underdog {row['opening_underdog']}")
            print(f"  Game time: {game_time_et.strftime('%I:%M %p ET')}")
            print(f"  Snapshots tracked: {(row['current_time'] - row['open_time']).total_seconds() / 3600:.1f} hours")
            
            # Show detection tracking info
            if 'detection_count' in row and pd.notna(row['detection_count']) and row['detection_count'] > 0:
                first_detected = pd.to_datetime(row['first_detected_at'])
                if first_detected.tz is None:
                    first_detected = first_detected.tz_localize('America/New_York')
                else:
                    first_detected = first_detected.tz_convert('America/New_York')
                detection_count = int(row['detection_count']) + 1  # +1 for current detection
                print(f"  Steam tracking: First detected at {first_detected.strftime('%I:%M %p ET')} | "
                      f"Detected {detection_count} times today")
            else:
                print(f"  Steam tracking: First detection of this steam pattern today")
            
            # Add bidirectional warning with historical context
            if row.get('is_bidirectional_steam', False):
                if sport_name.upper() == "NCAAB":
                    context = "Historical: 50/50 split in NCAAB (no edge)"
                elif sport_name.upper() == "NBA":
                    context = "Historical: 57.5% underdog cover in NBA (slight edge)"
                else:
                    context = "Market uncertainty - steam went both directions"
                
                print(f"  ⚠️  BIDIRECTIONAL STEAM WARNING")
                print(f"     This game had steam toward BOTH teams today")
                print(f"     {context}")
                print(f"     Confidence: LOW - Consider skipping or reducing stake")
            
            print()
    
    # Output favorite steam games
    if len(favorite_steam) > 0:
        print(f"{'='*80}")
        print(f"TOWARD OPENING FAVORITE ({len(favorite_steam)} game{'s' if len(favorite_steam) != 1 else ''})")
        print(f"{'='*80}\n")
        
        for idx, (_, row) in enumerate(favorite_steam.sort_values('steam_magnitude', ascending=False).iterrows(), 1):
            game_time_et = row['game_time'].tz_convert(et_tz)
            
            # Add bidirectional warning emoji to game title
            bidir_warning = " ⚠️ BIDIRECTIONAL STEAM" if row.get('is_bidirectional_steam', False) else ""
            
            print(f"Game {idx}: {row['opening_favorite']} vs {row['opening_underdog']}{bidir_warning}")
            print(f"  Opening: {row['opening_favorite']} {row['opening_favorite_spread_open']:+.1f} | "
                  f"{row['opening_underdog']} {-row['opening_favorite_spread_open']:+.1f}")
            print(f"  Current: {row['opening_favorite']} {row['opening_favorite_spread_current']:+.1f} | "
                  f"{row['opening_underdog']} {-row['opening_favorite_spread_current']:+.1f}")
            print(f"  Steam: {row['steam_magnitude']:.1f} points toward opening favorite {row['opening_favorite']}")
            print(f"  Game time: {game_time_et.strftime('%I:%M %p ET')}")
            print(f"  Snapshots tracked: {(row['current_time'] - row['open_time']).total_seconds() / 3600:.1f} hours")
            
            # Show detection tracking info
            if 'detection_count' in row and pd.notna(row['detection_count']) and row['detection_count'] > 0:
                first_detected = pd.to_datetime(row['first_detected_at'])
                if first_detected.tz is None:
                    first_detected = first_detected.tz_localize('America/New_York')
                else:
                    first_detected = first_detected.tz_convert('America/New_York')
                detection_count = int(row['detection_count']) + 1  # +1 for current detection
                print(f"  Steam tracking: First detected at {first_detected.strftime('%I:%M %p ET')} | "
                      f"Detected {detection_count} times today")
            else:
                print(f"  Steam tracking: First detection of this steam pattern today")
            
            # Add bidirectional warning with historical context
            if row.get('is_bidirectional_steam', False):
                if sport_name.upper() == "NCAAB":
                    context = "Historical: 50/50 split in NCAAB (no edge)"
                elif sport_name.upper() == "NBA":
                    context = "Historical: 57.5% underdog cover in NBA (slight edge)"
                else:
                    context = "Market uncertainty - steam went both directions"
                
                print(f"  ⚠️  BIDIRECTIONAL STEAM WARNING")
                print(f"     This game had steam toward BOTH teams today")
                print(f"     {context}")
                print(f"     Confidence: LOW - Consider skipping or reducing stake")
            
            print()
    
    # Add bidirectional summary before final return
    if len(bidirectional_game_ids) > 0:
        print(f"{'='*80}")
        print(f"⚠️  BIDIRECTIONAL STEAM SUMMARY")
        print(f"{'='*80}\n")
        print(f"{len(bidirectional_game_ids)} game(s) with steam in BOTH directions:")
        print(f"These games show market uncertainty - historical performance:")
        print(f"  • NCAAB: 50/50 coin flip (no edge)")
        print(f"  • NBA: 57.5% underdog cover (minimal edge)")
        print(f"\n💡 Recommendation: Skip or reduce stake size\n")
    
    # Add plays summary at the end (sorted by game time)
    print(f"{'='*80}")
    print("📋 PLAYS SUMMARY")
    print(f"{'='*80}\n")
    
    # Sort steam games by game time for the summary
    steam_games_sorted = steam_games.sort_values('game_time')
    
    for _, game in steam_games_sorted.iterrows():
        game_time_et = game['game_time'].tz_convert(et_tz)
        game_time_str = game_time_et.strftime('%I:%M%p ET').lstrip('0').lower()
        
        # Determine which team to display (steamed team)
        steamed_team = game['opening_underdog'] if game['steam_toward_opening_underdog'] else game['opening_favorite']
        current_spread = -game['opening_favorite_spread_current'] if game['steam_toward_opening_underdog'] else game['opening_favorite_spread_current']
        
        print(f"{game['opening_favorite']} vs {game['opening_underdog']} @ {game_time_str}")
        print(f"- {steamed_team} {current_spread:+.1f}")
        print()
    
    print(f"{'='*80}\n")
    
    print("STEAM_DETECTED: YES")
    return True, steam_games


# =============================================================================
# PLAYS PERSISTENCE
# =============================================================================

def save_plays_to_s3(steam_games, sport_config, target_date_str, season, threshold):
    """
    Save detected steam plays to S3.
    Appends all detections (no deduplication) to track steam evolution over time.
    
    Strategy:
    - PLAYS: Append every detection (tracks all signals throughout day)
    - RESULTS: Dedupe at (game_id, steam_direction) keeping largest steam magnitude
    
    Same game can have multiple rows in plays file:
    - Different steam directions (toward dog in AM, toward fav in PM)
    - Same direction but different magnitudes (2.5pts at 9am, 2.0pts at 11am, 3.0pts at 1pm)
    
    Results calculation will dedupe and use strongest signal for YTD tracking.
    
    Args:
        steam_games: DataFrame with games that have steam
        sport_config: SportConfig instance
        target_date_str: Date string in ET timezone (YYYY-MM-DD)
        season: Season string (e.g., '2025-26' for NBA/NCAAB, '2025' for NFL/NCAAF)
        threshold: Steam threshold used (stored in CSV, not in path)
    
    S3 Location: s3://{bucket}/data/04_output/plays/line-steam/{date_ET}.csv
    """
    print(f"\n💾 [save_plays_to_s3] Starting...")
    print(f"   Received {len(steam_games) if steam_games is not None else 0} steam games")
    print(f"   Target date: {target_date_str}")
    print(f"   Season: {season}")
    print(f"   Threshold: {threshold}")
    
    if steam_games is None or len(steam_games) == 0:
        print("💾 No plays to save")
        return
    
    # S3 path: threshold stored in CSV column, not in folder structure
    s3_key = sport_config.get_s3_plays_key(target_date_str)
    print(f"   S3 key: s3://{sport_config.s3_bucket_plays}/{s3_key}")
    s3 = boto3.client('s3')
    
    # Try to load existing plays for today
    existing_plays = None
    try:
        response = s3.get_object(Bucket=sport_config.s3_bucket_plays, Key=s3_key)
        existing_plays = pd.read_csv(BytesIO(response['Body'].read()))
        print(f"\n💾 Loaded {len(existing_plays)} existing plays from S3")
    except s3.exceptions.NoSuchKey:
        print(f"\n💾 No existing plays file - creating new one")
    except Exception as e:
        print(f"\n⚠️  Error loading existing plays: {e}")
    
    # Format current detections as play records
    et_tz = ZoneInfo('America/New_York')
    detected_at = datetime.now(et_tz).strftime('%Y-%m-%d %H:%M:%S')
    
    plays = []
    for _, game in steam_games.iterrows():
        game_time_et = game['game_time'].tz_convert(et_tz)
        
        play = {
            'detected_at': detected_at,
            'game_id': game['game_id'],
            'game_date': game_time_et.strftime('%Y-%m-%d'),
            'game_time': game_time_et.strftime('%Y-%m-%d %H:%M:%S'),
            'season': season,
            'opening_favorite': game['opening_favorite'],
            'opening_underdog': game['opening_underdog'],
            'opening_favorite_spread': game['opening_favorite_spread_open'],
            'current_favorite_spread': game['opening_favorite_spread_current'],
            'steam_magnitude': game['steam_magnitude'],
            'steam_direction': 'opening_underdog' if game['steam_toward_opening_underdog'] else 'opening_favorite',
            'steamed_team': game['opening_underdog'] if game['steam_toward_opening_underdog'] else game['opening_favorite'],
            'threshold': threshold,
            'play_team': game['opening_underdog'] if game['steam_toward_opening_underdog'] else game['opening_favorite'],
            'play_spread': -game['opening_favorite_spread_current'] if game['steam_toward_opening_underdog'] else game['opening_favorite_spread_current'],
            'status': 'pending',
            'actual_margin': None,
            'cover_margin': None,
            'is_bidirectional_steam': game.get('is_bidirectional_steam', False),
        }
        plays.append(play)
    
    new_plays_df = pd.DataFrame(plays)
    
    # Append all detections (no deduplication - keep steam evolution over time)
    if existing_plays is not None:
        combined_plays = pd.concat([existing_plays, new_plays_df], ignore_index=True)
        print(f"💾 Appended {len(new_plays_df)} new detections (total: {len(combined_plays)})")
    else:
        combined_plays = new_plays_df
        print(f"💾 Created {len(combined_plays)} new detections")
    
    # Save to S3
    csv_buffer = StringIO()
    combined_plays.to_csv(csv_buffer, index=False)
    
    s3.put_object(
        Bucket=sport_config.s3_bucket_plays,
        Key=s3_key,
        Body=csv_buffer.getvalue()
    )
    
    print(f"✅ Saved plays to s3://{sport_config.s3_bucket_plays}/{s3_key}")


def load_plays_from_s3(sport_config, date_str, threshold=None):
    """
    Load plays for a specific date (ET timezone).
    Dedupes at (game_id, steam_direction) level, keeping the detection with largest steam magnitude.
    
    Args:
        sport_config: SportConfig instance
        date_str: Date string in ET timezone (YYYY-MM-DD)
        threshold: Optional - filter to specific threshold after loading
    
    Returns:
        DataFrame with plays (optionally filtered by threshold, deduped by game/direction)
    """
    s3_key = sport_config.get_s3_plays_key(date_str)
    s3 = boto3.client('s3')
    
    try:
        response = s3.get_object(Bucket=sport_config.s3_bucket_plays, Key=s3_key)
        plays_df = pd.read_csv(BytesIO(response['Body'].read()))
        print(f"✅ Loaded {len(plays_df)} plays from S3: {s3_key}")
        
        # Filter by threshold if specified
        if threshold is not None:
            plays_df = plays_df[plays_df['threshold'] == threshold]
            print(f"   Filtered to threshold {threshold}: {len(plays_df)} plays")
        
        # Dedupe: keep detection with largest steam magnitude per (game_id, steam_direction)
        original_count = len(plays_df)
        plays_df = plays_df.sort_values('steam_magnitude', ascending=False).drop_duplicates(
            subset=['game_id', 'steam_direction'],
            keep='first'
        )
        deduped_count = len(plays_df)
        
        if deduped_count < original_count:
            print(f"   Deduped: {original_count} detections → {deduped_count} plays")
        
        return plays_df
    except s3.exceptions.NoSuchKey:
        print(f"⚠️  No plays file found: {s3_key}")
        return None
    except Exception as e:
        print(f"❌ Error loading plays: {e}")
        return None
