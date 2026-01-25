"""
Calculate Line Steam Results - Multi-Sport Results Matching

Reads saved plays from S3, fetches game results, calculates outcomes (cover/miss),
and saves results back to S3 for YTD tracking.

Supports: NBA, NFL, NCAAB, NCAAF

What it does:
1. Load plays from S3 (all detections throughout the day)
2. Dedupe: Keep strongest signal per (game_id, steam_direction) - largest steam magnitude
3. Load game results (NBA API for NBA, ESPN API for NFL/NCAAB/NCAAF)
4. Match plays to results by game_id and date
5. Calculate: Did steam team cover? By how much?
6. Save updated results to S3

Deduplication Strategy:
- Plays file: Appends every detection (e.g., 9am 2.5pts, 11am 2.0pts, 1pm 3.0pts)
- Results file: Only keeps the strongest signal per game/direction (3.0pts in this example)
- This ensures YTD stats reflect the best betting opportunities, not multiple counts

Usage:
    # Calculate NCAAB results for specific date
    python scripts/calculate_line_steam_results.py --sport ncaab --date 2026-01-23
    
    # Calculate NFL results
    python scripts/calculate_line_steam_results.py --sport nfl --date 2026-01-26
    
    # Filter to specific threshold after loading
    python scripts/calculate_line_steam_results.py --sport nba --date 2026-01-23 --threshold 1.0

Output:
    - Updates plays CSV with results (status='won'/'lost'/'push')
    - Saves to: s3://{bucket}/data/04_output/results/line-steam/{date_ET}.csv

Author: Thomas Myles
Date: 2026-01-23
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import boto3
from io import BytesIO, StringIO

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

from line_steam_utils import SportConfig, load_plays_from_s3
from season_utils import (
    get_current_nba_season,
    get_current_nfl_season,
    get_current_ncaab_season,
    get_current_ncaaf_season
)


def get_current_season(sport):
    """Get current season for a sport."""
    season_funcs = {
        'nba': get_current_nba_season,
        'nfl': get_current_nfl_season,
        'ncaab': get_current_ncaab_season,
        'ncaaf': get_current_ncaaf_season
    }
    
    if sport not in season_funcs:
        raise ValueError(f"Unknown sport: {sport}")
    
    return season_funcs[sport]()


def load_game_results_nba_api(date_str, season, sport_config):
    """
    Load NBA game results from S3 (NBA API).
    
    Args:
        date_str: Game date (YYYY-MM-DD)
        season: NBA season (e.g., '2025-26')
        sport_config: SportConfig instance
    
    Returns:
        DataFrame with team-level game results
    """
    s3_key = f"player_game_logs/{season}/{date_str}.csv"
    s3 = boto3.client('s3')
    
    try:
        response = s3.get_object(Bucket=sport_config.s3_bucket_results, Key=s3_key)
        results_df = pd.read_csv(BytesIO(response['Body'].read()))
        
        # Aggregate to team level
        results_df['GAME_DATE'] = pd.to_datetime(results_df['GAME_DATE'])
        team_games = results_df.groupby(['GAME_DATE', 'TEAM_NAME', 'MATCHUP', 'WL']).agg({
            'PTS': 'sum'
        }).reset_index()
        
        print(f"✅ Loaded {len(team_games)} team-game records from S3: {s3_key}")
        return team_games
    except s3.exceptions.NoSuchKey:
        print(f"⚠️  No game results found for {date_str}")
        return None
    except Exception as e:
        print(f"❌ Error loading game results: {e}")
        return None


def load_game_results_espn_api(date_str, sport_config):
    """
    Load game results from S3 (ESPN API format).
    
    Used for: NFL, NCAAB, NCAAF
    
    Args:
        date_str: Game date (YYYY-MM-DD)
        sport_config: SportConfig instance
    
    Returns:
        DataFrame with game-level results (HOME/AWAY format)
    """
    s3_prefix = f"data/01_input/historical_game_results/"
    s3 = boto3.client('s3')
    
    # List all files in the results directory (timestamped files)
    try:
        response = s3.list_objects_v2(
            Bucket=sport_config.s3_bucket_results,
            Prefix=s3_prefix
        )
    except Exception as e:
        print(f"❌ Error accessing S3: {e}")
        return None
    
    if 'Contents' not in response:
        print(f"⚠️  No game results files found")
        return None
    
    # Load all available result files and filter by date
    all_results = []
    for obj in response['Contents']:
        key = obj['Key']
        if not key.endswith('.csv'):
            continue
        
        try:
            result_obj = s3.get_object(Bucket=sport_config.s3_bucket_results, Key=key)
            df = pd.read_csv(BytesIO(result_obj['Body'].read()))
            all_results.append(df)
        except Exception as e:
            print(f"⚠️  Error reading {key}: {e}")
    
    if not all_results:
        print(f"⚠️  No valid result files found")
        return None
    
    # Combine all results and filter to target date
    results_df = pd.concat(all_results, ignore_index=True)
    results_df['GAME_DATE'] = pd.to_datetime(results_df['GAME_DATE']).dt.date
    target_date = pd.to_datetime(date_str).date()
    
    results_df = results_df[results_df['GAME_DATE'] == target_date]
    
    if len(results_df) == 0:
        print(f"⚠️  No game results found for {date_str}")
        return None
    
    print(f"✅ Loaded {len(results_df)} games for {date_str}")
    return results_df


def calculate_outcomes_nba(plays_df, game_results_df, sport_config):
    """
    Calculate outcomes for NBA plays.
    
    Args:
        plays_df: DataFrame with plays
        game_results_df: DataFrame with NBA game results (team-level)
        sport_config: SportConfig instance
    
    Returns:
        DataFrame with updated status, actual_margin, cover_margin
    """
    if plays_df is None or game_results_df is None:
        return plays_df
    
    print(f"\n📊 Calculating outcomes for {len(plays_df)} plays...")
    
    updated_plays = []
    
    for idx, play in plays_df.iterrows():
        # Skip if already calculated
        if play['status'] != 'pending':
            updated_plays.append(play.to_dict())
            continue
        
        game_date = pd.to_datetime(play['game_date']).date()
        
        # Normalize team names
        opening_favorite = sport_config.normalize_team_name(play['opening_favorite'])
        opening_underdog = sport_config.normalize_team_name(play['opening_underdog'])
        
        # Get both teams' scores
        favorite_result = game_results_df[
            (game_results_df['TEAM_NAME'] == opening_favorite) &
            (game_results_df['GAME_DATE'].dt.date == game_date)
        ]
        
        underdog_result = game_results_df[
            (game_results_df['TEAM_NAME'] == opening_underdog) &
            (game_results_df['GAME_DATE'].dt.date == game_date)
        ]
        
        if favorite_result.empty or underdog_result.empty:
            # Game hasn't happened yet
            updated_plays.append(play.to_dict())
            continue
        
        favorite_score = favorite_result.iloc[0]['PTS']
        underdog_score = underdog_result.iloc[0]['PTS']
        
        # Calculate actual margin (from favorite's perspective)
        actual_margin = favorite_score - underdog_score
        
        # Determine which team we bet on
        bet_on_underdog = (play['steam_direction'] == 'opening_underdog')
        
        if bet_on_underdog:
            # Bet underdog - cover margin relative to spread
            cover_margin = play['play_spread'] - actual_margin
            covered = cover_margin > 0
        else:
            # Bet favorite
            cover_margin = actual_margin + play['play_spread']
            covered = cover_margin > 0
        
        # Update play
        play_dict = play.to_dict()
        play_dict['actual_margin'] = actual_margin
        play_dict['cover_margin'] = cover_margin
        play_dict['status'] = 'won' if covered else 'push' if cover_margin == 0 else 'lost'
        
        updated_plays.append(play_dict)
        
        # Log result
        result_emoji = '✅' if covered else '🟰' if cover_margin == 0 else '❌'
        print(f"{result_emoji} {play['play_team']} {play['play_spread']:+.1f}: "
              f"Score {favorite_score}-{underdog_score}, Margin: {cover_margin:+.1f}")
    
    updated_df = pd.DataFrame(updated_plays)
    
    # Summary
    completed = updated_df[updated_df['status'] != 'pending']
    if len(completed) > 0:
        wins = (completed['status'] == 'won').sum()
        losses = (completed['status'] == 'lost').sum()
        pushes = (completed['status'] == 'push').sum()
        print(f"\n📈 Results: {wins}-{losses}-{pushes} ({wins/len(completed)*100:.1f}% win rate)")
    
    return updated_df


def calculate_outcomes_espn(plays_df, game_results_df, sport_config):
    """
    Calculate outcomes for ESPN API sports (NFL/NCAAB/NCAAF).
    
    Args:
        plays_df: DataFrame with plays
        game_results_df: DataFrame with game results (HOME/AWAY format)
        sport_config: SportConfig instance
    
    Returns:
        DataFrame with updated status, actual_margin, cover_margin
    """
    if plays_df is None or game_results_df is None:
        return plays_df
    
    print(f"\n📊 Calculating outcomes for {len(plays_df)} plays...")
    
    updated_plays = []
    
    for idx, play in plays_df.iterrows():
        # Skip if already calculated
        if play['status'] != 'pending':
            updated_plays.append(play.to_dict())
            continue
        
        game_date = pd.to_datetime(play['game_date']).date()
        
        # Normalize team names
        opening_favorite = sport_config.normalize_team_name(play['opening_favorite'])
        opening_underdog = sport_config.normalize_team_name(play['opening_underdog'])
        
        # Find game in results (could be HOME or AWAY)
        game_result = game_results_df[
            (game_results_df['GAME_DATE'] == game_date) &
            (
                ((game_results_df['HOME_TEAM'] == opening_favorite) & (game_results_df['AWAY_TEAM'] == opening_underdog)) |
                ((game_results_df['HOME_TEAM'] == opening_underdog) & (game_results_df['AWAY_TEAM'] == opening_favorite))
            )
        ]
        
        if game_result.empty:
            # Game hasn't happened yet
            updated_plays.append(play.to_dict())
            continue
        
        game = game_result.iloc[0]
        
        # Determine who is home/away
        if game['HOME_TEAM'] == opening_favorite:
            favorite_score = game['HOME_SCORE']
            underdog_score = game['AWAY_SCORE']
        else:
            favorite_score = game['AWAY_SCORE']
            underdog_score = game['HOME_SCORE']
        
        # Calculate actual margin
        actual_margin = favorite_score - underdog_score
        
        # Determine which team we bet on
        bet_on_underdog = (play['steam_direction'] == 'opening_underdog')
        
        if bet_on_underdog:
            cover_margin = play['play_spread'] - actual_margin
            covered = cover_margin > 0
        else:
            cover_margin = actual_margin + play['play_spread']
            covered = cover_margin > 0
        
        # Update play
        play_dict = play.to_dict()
        play_dict['actual_margin'] = actual_margin
        play_dict['cover_margin'] = cover_margin
        play_dict['status'] = 'won' if covered else 'push' if cover_margin == 0 else 'lost'
        
        updated_plays.append(play_dict)
        
        # Log result
        result_emoji = '✅' if covered else '🟰' if cover_margin == 0 else '❌'
        print(f"{result_emoji} {play['play_team']} {play['play_spread']:+.1f}: "
              f"Score {favorite_score}-{underdog_score}, Margin: {cover_margin:+.1f}")
    
    updated_df = pd.DataFrame(updated_plays)
    
    # Summary
    completed = updated_df[updated_df['status'] != 'pending']
    if len(completed) > 0:
        wins = (completed['status'] == 'won').sum()
        losses = (completed['status'] == 'lost').sum()
        pushes = (completed['status'] == 'push').sum()
        print(f"\n📈 Results: {wins}-{losses}-{pushes} ({wins/len(completed)*100:.1f}% win rate)")
    
    return updated_df


def save_results_to_s3(results_df, sport_config, date_str):
    """
    Save calculated results to S3.
    
    Args:
        results_df: DataFrame with calculated outcomes
        sport_config: SportConfig instance
        date_str: Date string in ET timezone (YYYY-MM-DD)
    """
    # Save to results folder
    results_key = sport_config.get_s3_results_key(date_str)
    
    # Also update the plays file
    plays_key = sport_config.get_s3_plays_key(date_str)
    
    s3 = boto3.client('s3')
    csv_buffer = StringIO()
    results_df.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()
    
    # Save to both locations
    s3.put_object(Bucket=sport_config.s3_bucket_plays, Key=results_key, Body=csv_content)
    s3.put_object(Bucket=sport_config.s3_bucket_plays, Key=plays_key, Body=csv_content)
    
    print(f"\n✅ Saved results to:")
    print(f"   s3://{sport_config.s3_bucket_plays}/{results_key}")
    print(f"   s3://{sport_config.s3_bucket_plays}/{plays_key}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Calculate line steam results for NBA/NFL/NCAAB/NCAAF'
    )
    parser.add_argument('--sport', required=True,
                       choices=['nba', 'nfl', 'ncaab', 'ncaaf'],
                       help='Sport to calculate results for')
    parser.add_argument('--date', required=True, 
                       help='Date to calculate in ET timezone (YYYY-MM-DD)')
    parser.add_argument('--threshold', type=float, 
                       help='Optional: filter to specific threshold')
    parser.add_argument('--season', type=str,
                       help='Season (e.g., 2025-26) - auto-detected if not provided')
    args = parser.parse_args()
    
    # Auto-detect season if not provided
    if not args.season:
        args.season = get_current_season(args.sport)
        print(f"📅 Auto-detected season: {args.season}")
    
    # Load sport configuration
    sport_config = SportConfig(args.sport)
    
    print(f"\n{'='*80}")
    print(f"Calculating results for: {sport_config.name} - {args.date} (ET timezone)")
    if args.threshold:
        print(f"Filtering to threshold: {args.threshold}")
    print(f"{'='*80}")
    
    # Load game results
    print(f"\n📥 Loading game results for {args.date}...")
    
    if sport_config.result_source == 'nba_api':
        game_results_df = load_game_results_nba_api(args.date, args.season, sport_config)
    else:  # espn_api
        game_results_df = load_game_results_espn_api(args.date, sport_config)
    
    if game_results_df is None:
        print(f"⚠️  No game results available for {args.date} - games may not have finished yet")
        sys.exit(0)
    
    # Load plays
    plays_df = load_plays_from_s3(sport_config, args.date, threshold=args.threshold)
    
    if plays_df is None:
        print(f"⚠️  No plays found for {args.date}")
        sys.exit(0)
    
    print(f"\n📊 Processing {len(plays_df)} plays...")
    if args.threshold is None:
        # Show breakdown by threshold
        threshold_counts = plays_df['threshold'].value_counts().sort_index()
        print(f"   Breakdown by threshold:")
        for thresh, count in threshold_counts.items():
            print(f"   - {thresh}: {count} plays")
    
    # Calculate outcomes
    if sport_config.result_source == 'nba_api':
        results_df = calculate_outcomes_nba(plays_df, game_results_df, sport_config)
    else:  # espn_api
        results_df = calculate_outcomes_espn(plays_df, game_results_df, sport_config)
    
    # Save results
    if results_df is not None:
        save_results_to_s3(results_df, sport_config, args.date)
    
    print(f"\n{'='*80}")
    print("✅ Results calculation complete!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
