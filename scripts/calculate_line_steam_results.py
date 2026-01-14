"""
Calculate Line Steam Results - Match Plays to Game Outcomes

Reads saved plays from S3, fetches game results, calculates outcomes (cover/miss),
and saves results back to S3 for YTD tracking.

What it does:
1. Load plays from S3 (all detections throughout the day)
2. Dedupe: Keep strongest signal per (game_id, steam_direction) - largest steam magnitude
3. Load game results from NBA API S3 bucket
4. Match plays to results by game_id and date
5. Calculate: Did steam team cover? By how much?
6. Save updated results to S3

Deduplication Strategy:
    - Plays file: Appends every detection (e.g., 9am 2.5pts, 11am 2.0pts, 1pm 3.0pts)
    - Results file: Only keeps the strongest signal per game/direction (3.0pts in this example)
    - This ensures YTD stats reflect the best betting opportunities, not multiple counts

Usage:
    # Calculate results for specific date (processes all thresholds in file)
    python scripts/calculate_line_steam_results.py --date 2026-01-13 --season 2025-26
    
    # Filter to specific threshold after loading
    python scripts/calculate_line_steam_results.py --date 2026-01-13 --season 2025-26 --threshold 1.0

Output:
    - Updates plays CSV with results (status='won'/'lost'/'push')
    - Saves to: s3://nba-betting-mt/data/04_output/results/line-steam/{date_ET}.csv

Author: Thomas Myles
Date: 2026-01-14
"""

import pandas as pd
import boto3
from io import BytesIO, StringIO
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
S3_BUCKET_PLAYS = 'nba-betting-mt'
S3_BUCKET_GAME_RESULTS = 'nba-api-mt'

# Team name normalization
TEAM_NAME_MAP = {
    'Los Angeles Clippers': 'LA Clippers',
}

def normalize_team_name(team_name):
    """Normalize team names from Odds API to match NBA API format."""
    return TEAM_NAME_MAP.get(team_name, team_name)


def load_plays_from_s3(date_str, threshold=None):
    """
    Load plays for a specific date (ET timezone).
    Dedupes at (game_id, steam_direction) level, keeping the detection with largest steam magnitude.
    
    Args:
        date_str: Date string in ET timezone (YYYY-MM-DD)
        threshold: Optional - filter to specific threshold after loading
    
    Returns:
        DataFrame with plays (optionally filtered by threshold, deduped by game/direction)
    """
    s3_key = f"data/04_output/plays/line-steam/{date_str}.csv"
    s3 = boto3.client('s3')
    
    try:
        response = s3.get_object(Bucket=S3_BUCKET_PLAYS, Key=s3_key)
        plays_df = pd.read_csv(BytesIO(response['Body'].read()))
        print(f"✅ Loaded {len(plays_df)} plays from S3: {s3_key}")
        
        # Filter by threshold if specified
        if threshold is not None:
            plays_df = plays_df[plays_df['threshold'] == threshold]
            print(f"   Filtered to threshold {threshold}: {len(plays_df)} plays")
        
        # Dedupe: keep detection with largest steam magnitude per (game_id, steam_direction)
        # This selects the strongest signal we got for each game/direction throughout the day
        original_count = len(plays_df)
        plays_df = plays_df.sort_values('steam_magnitude', ascending=False).drop_duplicates(
            subset=['game_id', 'steam_direction'],
            keep='first'  # Keep the row with largest steam_magnitude (sorted desc)
        )
        deduped_count = len(plays_df)
        
        if deduped_count < original_count:
            print(f"   Deduped: {original_count} detections → {deduped_count} plays (kept largest steam per game/direction)")
        
        return plays_df
    except s3.exceptions.NoSuchKey:
        print(f"⚠️  No plays file found: {s3_key}")
        return None
    except Exception as e:
        print(f"❌ Error loading plays: {e}")
        return None


def load_game_results_for_date(game_date, season):
    """Load NBA game results from S3 for a specific date."""
    s3_key = f"player_game_logs/{season}/{game_date}.csv"
    s3 = boto3.client('s3')
    
    try:
        response = s3.get_object(Bucket=S3_BUCKET_GAME_RESULTS, Key=s3_key)
        results_df = pd.read_csv(BytesIO(response['Body'].read()))
        
        # Aggregate to team level
        results_df['GAME_DATE'] = pd.to_datetime(results_df['GAME_DATE'])
        team_games = results_df.groupby(['GAME_DATE', 'TEAM_NAME', 'MATCHUP', 'WL']).agg({
            'PTS': 'sum'
        }).reset_index()
        
        print(f"✅ Loaded {len(team_games)} team-game records from S3: {s3_key}")
        return team_games
    except s3.exceptions.NoSuchKey:
        print(f"⚠️  No game results found for {game_date}")
        return None
    except Exception as e:
        print(f"❌ Error loading game results: {e}")
        return None


def calculate_outcomes(plays_df, game_results_df):
    """
    Calculate outcomes for plays by matching to game results.
    
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
        opening_favorite = normalize_team_name(play['opening_favorite'])
        opening_underdog = normalize_team_name(play['opening_underdog'])
        
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
        
        # Determine which team we bet on (the steamed team)
        bet_on_underdog = (play['steam_direction'] == 'opening_underdog')
        
        if bet_on_underdog:
            # Bet underdog - they need to lose by less than the spread (or win)
            # ATS formula: (underdog_score + spread) > favorite_score
            # Equivalent: actual_margin < spread
            cover_margin = play['play_spread'] - actual_margin
            covered = cover_margin > 0
        else:
            # Bet favorite - they need to win by more than the spread
            # ATS formula: (favorite_score + spread) > underdog_score
            # Equivalent: actual_margin > -spread
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


def save_results_to_s3(results_df, date_str):
    """
    Save calculated results to S3.
    Threshold stored in CSV column (not in path).
    
    Args:
        results_df: DataFrame with calculated outcomes
        date_str: Date string in ET timezone (YYYY-MM-DD)
    """
    # Save to results folder
    results_key = f"data/04_output/results/line-steam/{date_str}.csv"
    
    # Also update the plays file
    plays_key = f"data/04_output/plays/line-steam/{date_str}.csv"
    
    s3 = boto3.client('s3')
    csv_buffer = StringIO()
    results_df.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()
    
    # Save to both locations
    s3.put_object(Bucket=S3_BUCKET_PLAYS, Key=results_key, Body=csv_content)
    s3.put_object(Bucket=S3_BUCKET_PLAYS, Key=plays_key, Body=csv_content)
    
    print(f"\n✅ Saved results to:")
    print(f"   s3://{S3_BUCKET_PLAYS}/{results_key}")
    print(f"   s3://{S3_BUCKET_PLAYS}/{plays_key}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Calculate line steam results')
    parser.add_argument('--date', required=True, help='Date to calculate in ET timezone (YYYY-MM-DD)')
    parser.add_argument('--threshold', type=float, help='Optional: filter to specific threshold')
    parser.add_argument('--season', required=True, help='NBA season (e.g., 2025-26)')
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"Calculating results for: {args.date} (ET timezone)")
    if args.threshold:
        print(f"Filtering to threshold: {args.threshold}")
    print(f"{'='*80}")
    
    # Load game results
    print(f"\n📥 Loading game results for {args.date}...")
    game_results_df = load_game_results_for_date(args.date, args.season)
    
    if game_results_df is None:
        print(f"⚠️  No game results available for {args.date} - games may not have finished yet")
        sys.exit(0)
    
    # Load plays (all thresholds)
    plays_df = load_plays_from_s3(args.date, threshold=args.threshold)
    
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
    results_df = calculate_outcomes(plays_df, game_results_df)
    
    # Save results
    if results_df is not None:
        save_results_to_s3(results_df, args.date)
    
    print(f"\n{'='*80}")
    print("✅ Results calculation complete!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

