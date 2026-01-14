"""
Check NBA Line Steam - Alert Script for Lambda

Checks if any of TODAY'S NBA games (in ET timezone) have 2+ point consensus line 
movement toward the opening underdog. This script runs BEFORE games are played to 
identify sharp betting opportunities.

What it does:
1. Load hourly line movement snapshots from S3
2. Calculate consensus line movements (opening → current/latest)
3. Filter to games scheduled for TODAY (in ET timezone)
4. Check if any game has threshold+ movement toward opening underdog
5. Output "STEAM_DETECTED: YES" or "STEAM_DETECTED: NO" with game details

Key Logic:
- Opening line: First hourly snapshot per game
- Current line: Latest hourly snapshot per game (most recent check)
- Opening underdog: Team with positive spread at opening
- Steam toward underdog: Line moves TOWARD the underdog (away from favorite)
  Example: Opening LAL -7.5 → Current LAL -5.0 = 2.5pt steam toward DEN underdog

Usage:
    # Check today's games for 2+ point steam
    python scripts/check_nba_line_steam.py --date 2026-01-13 --threshold 2.0
    
    # Custom threshold
    python scripts/check_nba_line_steam.py --date 2026-01-13 --threshold 3.0
    
    # Show detailed breakdown of ALL today's games (sorted by start time)
    python scripts/check_nba_line_steam.py --date 2026-01-13 --threshold 2.0 --log-individual-games

Output Format:
    If steam detected:
        STEAM_DETECTED: YES
        <game details>
    
    If no steam:
        STEAM_DETECTED: NO
        Checked X games - largest movement was Y points

Context:
This script is called by lambda_function_nba_line_movement_alerts.py which runs
~5 minutes after the hourly line tracking script. The Lambda sends email alerts
only when STEAM_DETECTED: YES is found in the output.

Author: Thomas Myles
Date: 2026-01-13
"""

import pandas as pd
import boto3
from io import BytesIO
from datetime import datetime
from zoneinfo import ZoneInfo
import argparse
import sys
import os
from pathlib import Path

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

# Constants
S3_BUCKET_SNAPSHOTS = 'betting-line-movement-snapshots'  # Line movement tracking snapshots
S3_BUCKET_PLAYS = 'nba-betting-mt'  # Bucket for saving plays
TEAM_NAME_MAP = {
    'Los Angeles Clippers': 'LA Clippers',
}

def normalize_team_name(team_name):
    """Normalize team names from Odds API to match NBA API format."""
    return TEAM_NAME_MAP.get(team_name, team_name)


def load_line_movement_snapshots(days_back=3):
    """
    Load recent hourly line movement snapshots from S3.
    Only loads snapshot files from the last X days (efficient filtering by filename date).
    
    Args:
        days_back: Only load snapshots from the last X days (default 3)
    """
    s3_prefix = 'data/01_input/the-odds-api/nba/line_movement/'
    
    print(f"📥 Loading NBA hourly snapshots from S3...")
    print(f"   S3 Bucket: {S3_BUCKET_SNAPSHOTS}")
    print(f"   Prefix: {s3_prefix}")
    print(f"   Days back: {days_back}")
    
    s3 = boto3.client('s3')
    
    # Calculate cutoff date FIRST (in UTC to match S3 file timestamps)
    from datetime import timedelta, timezone
    cutoff_datetime = datetime.now(timezone.utc) - timedelta(days=days_back)
    cutoff_date_str = cutoff_datetime.strftime('%Y-%m-%d')
    print(f"   Cutoff date: {cutoff_date_str} (only loading files >= this date)")
    
    try:
        response = s3.list_objects_v2(Bucket=S3_BUCKET_SNAPSHOTS, Prefix=s3_prefix)
    except Exception as e:
        print(f"❌ Error accessing S3 bucket: {e}")
        raise
    
    if 'Contents' not in response:
        raise ValueError(f"No snapshots found in S3")
    
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
            filename = key.split('/')[-1]  # Get just the filename
            if filename.startswith('snapshot_'):
                # Parse date from filename: snapshot_YYYY-MM-DD_HH-MM-SS.csv
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
            response_obj = s3.get_object(Bucket=S3_BUCKET_SNAPSHOTS, Key=key)
            df = pd.read_csv(BytesIO(response_obj['Body'].read()))
            all_dfs.append(df)
        except Exception as e:
            print(f"⚠️  Error reading {key}: {e}")
    
    if not all_dfs:
        raise ValueError(f"No valid snapshot CSV files found")
    
    # Combine all dataframes
    df = pd.concat(all_dfs, ignore_index=True)
    
    # Convert timestamps
    df['game_time'] = pd.to_datetime(df['game_time'])
    df['fetched_at'] = pd.to_datetime(df['fetched_at'])
    
    print(f"   Loaded {len(df):,} line records from {len(all_dfs)} snapshot files")
    print(f"   Date range: {df['fetched_at'].min()} to {df['fetched_at'].max()}")
    print(f"✅ Unique games: {df['game_id'].nunique():,}")
    
    return df


def calculate_consensus_movements(snapshots_df):
    """
    Calculate consensus line movements (opening → current).
    
    Returns DataFrame with one row per game containing:
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
    # Positive = line moved toward favorite (favorite got more favored)
    # Negative = line moved toward underdog (favorite got less favored)
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


def log_individual_games(movements_df, target_date_str, threshold=2.0):
    """
    Log detailed breakdown of ALL today's games, sorted by start time (ET).
    Matches --log-individual-games convention from analyze_line_movement_predictiveness.py
    
    Args:
        movements_df: DataFrame with consensus movements
        target_date_str: Date string in YYYY-MM-DD format (ET timezone)
        threshold: Steam threshold for highlighting (default 2.0)
    """
    et_tz = ZoneInfo('America/New_York')
    target_date = datetime.strptime(target_date_str, '%Y-%m-%d').date()
    
    # Filter to games scheduled for target date (in ET)
    movements_df['game_date_et'] = movements_df['game_time'].dt.tz_convert(et_tz).dt.date
    today_games = movements_df[movements_df['game_date_et'] == target_date].copy()
    
    if len(today_games) == 0:
        print(f"\n📋 No games scheduled for {target_date_str}")
        return
    
    # Sort by game start time (ET)
    today_games['game_time_et'] = today_games['game_time'].dt.tz_convert(et_tz)
    today_games = today_games.sort_values('game_time_et')
    
    print(f"\n{'='*80}")
    print(f"📋 ALL GAMES TODAY ({target_date_str}) - Sorted by Start Time")
    print(f"{'='*80}")
    
    for idx, (_, row) in enumerate(today_games.iterrows(), 1):
        game_time_et = row['game_time_et']
        hours_tracked = (row['current_time'] - row['open_time']).total_seconds() / 3600
        
        # Check if this game has significant steam
        has_steam = row['steam_toward_opening_underdog'] and row['steam_magnitude'] >= threshold
        steam_indicator = "🚨 STEAM" if has_steam else ""
        
        print(f"\n{'─'*80}")
        print(f"#{idx} | {game_time_et.strftime('%I:%M %p ET')} {steam_indicator}")
        print(f"🏀 {row['opening_favorite']} (fav) vs {row['opening_underdog']} (dog)")
        
        # Opening and current lines
        print(f"📊 Opening: {row['opening_favorite']} {row['opening_favorite_spread_open']:+.1f} | "
              f"{row['opening_underdog']} {-row['opening_favorite_spread_open']:+.1f}")
        print(f"📊 Current: {row['opening_favorite']} {row['opening_favorite_spread_current']:+.1f} | "
              f"{row['opening_underdog']} {-row['opening_favorite_spread_current']:+.1f}")
        
        # Movement direction and magnitude
        if row['steam_toward_opening_underdog']:
            direction_text = f"toward opening UNDERDOG ({row['opening_underdog']})"
        else:
            direction_text = f"toward opening FAVORITE ({row['opening_favorite']})"
        
        print(f"🔥 Movement: {row['steam_magnitude']:.1f} pts {direction_text}")
        
        # Tracking metadata (convert UTC timestamps to ET)
        open_time_et = row['open_time'].tz_convert(et_tz)
        current_time_et = row['current_time'].tz_convert(et_tz)
        print(f"📈 Tracked: {hours_tracked:.1f} hrs | First: {open_time_et.strftime('%m/%d %I:%M%p')} ET → "
              f"Latest: {current_time_et.strftime('%m/%d %I:%M%p')} ET")
    
    print(f"\n{'='*80}")


def save_plays_to_s3(steam_games, target_date_str, season, threshold):
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
        target_date_str: Date string in ET timezone (YYYY-MM-DD)
        season: NBA season (e.g., '2025-26')
        threshold: Steam threshold used (stored in CSV, not in path)
    
    S3 Location: s3://nba-betting-mt/data/04_output/plays/line-steam/{date_ET}.csv
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
    s3_key = f"data/04_output/plays/line-steam/{target_date_str}.csv"
    print(f"   S3 key: s3://{S3_BUCKET_PLAYS}/{s3_key}")
    s3 = boto3.client('s3')
    
    # Try to load existing plays for today
    existing_plays = None
    try:
        response = s3.get_object(Bucket=S3_BUCKET_PLAYS, Key=s3_key)
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
        }
        plays.append(play)
    
    new_plays_df = pd.DataFrame(plays)
    
    # Append all detections (no deduplication - keep steam evolution over time)
    # Same game can have multiple detections as steam magnitude changes throughout day
    # Can dedupe later during analysis if needed
    if existing_plays is not None:
        combined_plays = pd.concat([existing_plays, new_plays_df], ignore_index=True)
        print(f"💾 Appended {len(new_plays_df)} new detections (total: {len(combined_plays)})")
    else:
        combined_plays = new_plays_df
        print(f"💾 Created {len(combined_plays)} new detections")
    
    # Save to S3
    from io import StringIO
    csv_buffer = StringIO()
    combined_plays.to_csv(csv_buffer, index=False)
    
    s3.put_object(
        Bucket=S3_BUCKET_PLAYS,
        Key=s3_key,
        Body=csv_buffer.getvalue()
    )
    
    print(f"✅ Saved plays to s3://{S3_BUCKET_PLAYS}/{s3_key}")


def check_for_steam(movements_df, target_date_str, threshold=2.0):
    """
    Check if any of today's games have threshold+ point steam (both directions).
    
    Args:
        movements_df: DataFrame with consensus movements
        target_date_str: Date string in YYYY-MM-DD format (ET timezone)
        threshold: Minimum movement in points (default 2.0)
    
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
    
    # Separate by direction for logging
    underdog_steam = steam_games[steam_games['steam_toward_opening_underdog']]
    favorite_steam = steam_games[~steam_games['steam_toward_opening_underdog']]
    
    print(f"   Games with {threshold}+ pt steam: {len(steam_games)} total")
    print(f"   - Toward opening underdog: {len(underdog_steam)}")
    print(f"   - Toward opening favorite: {len(favorite_steam)}")
    
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
    
    # Sort by steam magnitude (highest first)
    steam_games = steam_games.sort_values('steam_magnitude', ascending=False)
    
    # Output underdog steam games first
    if len(underdog_steam) > 0:
        print(f"{'='*80}")
        print(f"TOWARD OPENING UNDERDOG ({len(underdog_steam)} game{'s' if len(underdog_steam) != 1 else ''})")
        print(f"{'='*80}\n")
        
        for idx, (_, row) in enumerate(underdog_steam.sort_values('steam_magnitude', ascending=False).iterrows(), 1):
            game_time_et = row['game_time'].tz_convert(et_tz)
            
            print(f"Game {idx}: {row['opening_favorite']} vs {row['opening_underdog']}")
            print(f"  Opening: {row['opening_favorite']} {row['opening_favorite_spread_open']:+.1f} | "
                  f"{row['opening_underdog']} {-row['opening_favorite_spread_open']:+.1f}")
            print(f"  Current: {row['opening_favorite']} {row['opening_favorite_spread_current']:+.1f} | "
                  f"{row['opening_underdog']} {-row['opening_favorite_spread_current']:+.1f}")
            print(f"  Steam: {row['steam_magnitude']:.1f} points toward opening underdog {row['opening_underdog']}")
            print(f"  Game time: {game_time_et.strftime('%I:%M %p ET')}")
            print(f"  Snapshots tracked: {(row['current_time'] - row['open_time']).total_seconds() / 3600:.1f} hours")
            print()
    
    # Output favorite steam games
    if len(favorite_steam) > 0:
        print(f"{'='*80}")
        print(f"TOWARD OPENING FAVORITE ({len(favorite_steam)} game{'s' if len(favorite_steam) != 1 else ''})")
        print(f"{'='*80}\n")
        
        for idx, (_, row) in enumerate(favorite_steam.sort_values('steam_magnitude', ascending=False).iterrows(), 1):
            game_time_et = row['game_time'].tz_convert(et_tz)
            
            print(f"Game {idx}: {row['opening_favorite']} vs {row['opening_underdog']}")
            print(f"  Opening: {row['opening_favorite']} {row['opening_favorite_spread_open']:+.1f} | "
                  f"{row['opening_underdog']} {-row['opening_favorite_spread_open']:+.1f}")
            print(f"  Current: {row['opening_favorite']} {row['opening_favorite_spread_current']:+.1f} | "
                  f"{row['opening_underdog']} {-row['opening_favorite_spread_current']:+.1f}")
            print(f"  Steam: {row['steam_magnitude']:.1f} points toward opening favorite {row['opening_favorite']}")
            print(f"  Game time: {game_time_et.strftime('%I:%M %p ET')}")
            print(f"  Snapshots tracked: {(row['current_time'] - row['open_time']).total_seconds() / 3600:.1f} hours")
            print()
    
    print("STEAM_DETECTED: YES")
    return True, steam_games


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Check NBA line steam for today')
    parser.add_argument('--date', required=True, help='Date to check (YYYY-MM-DD format, ET timezone)')
    parser.add_argument('--threshold', type=float, default=2.0, help='Steam threshold in points (default: 2.0)')
    parser.add_argument('--days-back', type=int, default=3, help='Only load snapshots from last X days (default: 3)')
    parser.add_argument('--log-individual-games', action='store_true', 
                       help='Log detailed breakdown of ALL games today (sorted by start time)')
    parser.add_argument('--save-plays', action='store_true',
                       help='Save detected steam plays to S3 (s3://nba-betting-mt/data/04_output/plays/)')
    parser.add_argument('--season', type=str, help='NBA season (e.g., 2025-26) - required if --save-plays is used')
    args = parser.parse_args()
    
    # Validate args
    if args.save_plays and not args.season:
        print("❌ ERROR: --season is required when using --save-plays")
        sys.exit(1)
    
    try:
        # Load snapshots
        snapshots_df = load_line_movement_snapshots(days_back=args.days_back)
        
        # Calculate movements
        movements_df = calculate_consensus_movements(snapshots_df)
        
        # Log all games if requested (BEFORE checking for steam)
        if args.log_individual_games:
            log_individual_games(movements_df, args.date, args.threshold)
        
        # Check for steam
        steam_detected, steam_games = check_for_steam(movements_df, args.date, args.threshold)
        
        # Save plays to S3 if requested and steam detected
        if args.save_plays:
            print(f"\n💾 --save-plays flag detected")
            print(f"   Steam detected: {steam_detected}")
            print(f"   Steam games: {steam_games is not None}")
            if steam_games is not None:
                print(f"   Number of steam games: {len(steam_games)}")
            
            if steam_detected and steam_games is not None:
                print(f"   Calling save_plays_to_s3...")
                try:
                    save_plays_to_s3(steam_games, args.date, args.season, args.threshold)
                except Exception as e:
                    print(f"   ❌ Save failed: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"   Skipping save (no steam or no games)")
        
        # Exit code: 0 = success (steam or no steam), 1 = error
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

