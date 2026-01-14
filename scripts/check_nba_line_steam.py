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
S3_BUCKET_SNAPSHOTS = 'betting-line-movement-snapshots'
TEAM_NAME_MAP = {
    'Los Angeles Clippers': 'LA Clippers',
}

def normalize_team_name(team_name):
    """Normalize team names from Odds API to match NBA API format."""
    return TEAM_NAME_MAP.get(team_name, team_name)


def load_line_movement_snapshots(days_back=3):
    """
    Load recent hourly line movement snapshots from S3.
    
    Args:
        days_back: Only load snapshots from the last X days (default 3)
    """
    s3_prefix = 'data/01_input/the-odds-api/nba/line_movement/'
    
    print(f"📥 Loading NBA hourly snapshots from S3...")
    print(f"   S3 Bucket: {S3_BUCKET_SNAPSHOTS}")
    print(f"   Prefix: {s3_prefix}")
    print(f"   Days back: {days_back}")
    
    s3 = boto3.client('s3')
    
    try:
        response = s3.list_objects_v2(Bucket=S3_BUCKET_SNAPSHOTS, Prefix=s3_prefix)
    except Exception as e:
        print(f"❌ Error accessing S3 bucket: {e}")
        raise
    
    if 'Contents' not in response:
        raise ValueError(f"No snapshots found in S3")
    
    all_dfs = []
    
    for obj in response.get('Contents', []):
        key = obj['Key']
        
        # Only process snapshot CSV files
        if not key.endswith('.csv') or 'snapshot_' not in key:
            continue
        
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
    
    # Filter to only recent snapshots (last X days)
    et_tz = ZoneInfo('America/New_York')
    cutoff_date = datetime.now(et_tz) - pd.Timedelta(days=days_back)
    
    df_filtered = df[df['fetched_at'] >= cutoff_date].copy()
    
    print(f"   Filtered to last {days_back} days: {len(df_filtered):,} records ({len(df) - len(df_filtered):,} dropped)")
    print(f"   Date range: {df_filtered['fetched_at'].min()} to {df_filtered['fetched_at'].max()}")
    print(f"✅ Unique games: {df_filtered['game_id'].nunique():,}")
    
    return df_filtered


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


def check_for_steam(movements_df, target_date_str, threshold=2.0):
    """
    Check if any of today's games have threshold+ point steam toward opening underdog.
    
    Args:
        movements_df: DataFrame with consensus movements
        target_date_str: Date string in YYYY-MM-DD format (ET timezone)
        threshold: Minimum movement in points (default 2.0)
    
    Returns:
        tuple: (steam_detected: bool, steam_games: DataFrame or None)
    """
    print(f"\n🔍 Checking for {threshold}+ point steam toward opening underdog...")
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
    
    # Filter to games with threshold+ movement toward opening underdog
    steam_games = today_games[
        (today_games['steam_toward_opening_underdog']) & 
        (today_games['steam_magnitude'] >= threshold)
    ].copy()
    
    print(f"   Games with {threshold}+ pt steam toward opening underdog: {len(steam_games)}")
    
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
    print(f"   {len(steam_games)} game(s) with {threshold}+ point steam toward opening underdog\n")
    
    # Format output
    steam_games = steam_games.sort_values('steam_magnitude', ascending=False)
    
    for idx, row in steam_games.iterrows():
        game_time_et = row['game_time'].tz_convert(et_tz)
        
        print(f"Game {idx+1}: {row['opening_favorite']} vs {row['opening_underdog']}")
        print(f"  Opening: {row['opening_favorite']} {row['opening_favorite_spread_open']:+.1f} | "
              f"{row['opening_underdog']} {-row['opening_favorite_spread_open']:+.1f}")
        print(f"  Current: {row['opening_favorite']} {row['opening_favorite_spread_current']:+.1f} | "
              f"{row['opening_underdog']} {-row['opening_favorite_spread_current']:+.1f}")
        print(f"  Steam: {row['steam_magnitude']:.1f} points toward opening underdog {row['opening_underdog']}")
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
    args = parser.parse_args()
    
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
        
        # Exit code: 0 = success (steam or no steam), 1 = error
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

