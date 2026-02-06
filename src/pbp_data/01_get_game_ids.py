"""
Step 1: Collect game IDs for each date in the season.

Caches results in ~/Downloads/tmp/player_points_monte_carlo/game_ids/{date}.parquet

All dates use ET timezone to match NBA/ESPN game schedules.
Only fetches games up to yesterday (excludes today's incomplete games).

Usage:
    python src/pbp_data/01_get_game_ids.py [--verbose]
    
Loop usage:
    while true; do
    timeout --signal=INT 30 python -m src.pbp_data.01_get_game_ids --verbose
    rc=$?

    # timeout or SIGKILL → keep going
    if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then
        sleep 0.1
        continue
    fi

    # real error → stop
    if [ $rc -ne 0 ]; then
        echo "Error (rc=$rc) — stopping"
        break
    fi

    sleep 0.1
    done
"""

import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd

from .config import (
    GAME_IDS_DIR,
    SEASON,
    SEASON_CONFIG_PATH,
    RATE_LIMIT_BETWEEN_DATES,
)
from .utils import (
    load_season_dates,
    date_range,
    get_games_on_date,
    load_progress,
    add_to_progress,
)


def get_game_ids_for_date(date, verbose=False):
    """
    Get all game IDs for a specific date and cache to Parquet.
    
    Args:
        date: datetime.date object
        verbose: Print progress
    
    Returns:
        Number of games found
    """
    date_str = date.strftime('%Y%m%d')
    output_file = GAME_IDS_DIR / f"{date_str}.parquet"
    
    # Note: Skip check moved to main loop for better progress tracking
    
    try:
        # Get games from ESPN
        games = get_games_on_date(date)
        
        if games:
            # Save to Parquet
            df = pd.DataFrame(games)
            df.to_parquet(output_file, index=False)
            
            if verbose:
                print(f"  ✅ {date}: Found {len(games)} games")
        else:
            # Create empty parquet to mark as checked
            df = pd.DataFrame(columns=['game_id', 'home_team', 'away_team', 'date'])
            df.to_parquet(output_file, index=False)
            if verbose:
                print(f"  ⭕ {date}: No games")
        
        # Mark as completed
        add_to_progress('game_ids_progress.json', date_str)
        
        return len(games)
        
    except Exception as e:
        if verbose:
            print(f"  ❌ {date}: Error - {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description='Collect NBA game IDs')
    parser.add_argument('--verbose', action='store_true', help='Print progress')
    args = parser.parse_args()
    
    # Load season dates
    season_config = load_season_dates(SEASON_CONFIG_PATH, SEASON)
    start_date = season_config['season_start']
    # Use ET timezone and stop at yesterday (exclude today's incomplete games)
    yesterday_et = datetime.now(ZoneInfo('America/New_York')).date() - timedelta(days=1)
    end_date = yesterday_et.isoformat()
    
    if args.verbose:
        print(f"🏀 Collecting game IDs for {SEASON}")
        print(f"   Date range: {start_date} to {end_date}")
        print()
    
    # Load progress
    completed = load_progress('game_ids_progress.json')
    
    total_games = 0
    dates_processed = 0
    dates_skipped = 0
    
    # Process each date
    for date in date_range(start_date, end_date):
        date_str = date.strftime('%Y%m%d')
        output_file = GAME_IDS_DIR / f"{date_str}.parquet"
        
        # Check if already exists (show skip message)
        if output_file.exists():
            dates_skipped += 1
            if args.verbose:
                # Read to show game count
                df = pd.read_parquet(output_file)
                num_games = len(df)
                if num_games > 0:
                    print(f"  ⏭️  {date}: Already cached ({num_games} games)")
                else:
                    print(f"  ⏭️  {date}: Already cached (no games)")
            continue
        
        # Get games for this date
        num_games = get_game_ids_for_date(date, verbose=args.verbose)
        total_games += num_games
        dates_processed += 1
        
        # Rate limiting
        time.sleep(RATE_LIMIT_BETWEEN_DATES)
    
    if args.verbose:
        print()
        if dates_skipped > 0:
            print(f"⏭️  Skipped {dates_skipped} already-cached dates")
        print(f"✅ Processed {dates_processed} dates")
        print(f"✅ Found {total_games} games")
        print(f"📁 Cached in: {GAME_IDS_DIR}")
    
    # Exit with code 1 if no dates were processed (all done)
    if dates_processed == 0:
        if args.verbose:
            print()
            print("🎉 All dates processed - exiting")
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
