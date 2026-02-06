"""
Step 2: Download play-by-play data for each game.

Reads game IDs from ~/Downloads/tmp/player_points_monte_carlo/game_ids/*.parquet
Caches PBP data in ~/Downloads/tmp/player_points_monte_carlo/pbp_data/{date}_{game_id}.json

All dates are in ET timezone (inherited from game_ids parquet files).

Usage:
    python src/pbp_data/02_get_pbp_data.py [--verbose]
    
Loop usage:
    while true; do
    timeout --signal=INT 30 python -m src.pbp_data.02_get_pbp_data --verbose
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
import json
import time
from pathlib import Path
from collections import defaultdict
import pandas as pd

from .config import (
    GAME_IDS_DIR,
    PBP_DATA_DIR,
    RATE_LIMIT_BETWEEN_GAMES,
)
from .utils import (
    get_play_by_play,
    load_progress,
    add_to_progress,
)


def get_all_game_ids_by_date():
    """
    Load all game IDs from cached Parquet files, grouped by date.
    
    Returns:
        Dict of {date: [(game_id, date), ...]} sorted by date
    """
    games_by_date = defaultdict(list)
    
    for parquet_file in sorted(GAME_IDS_DIR.glob('*.parquet')):
        df = pd.read_parquet(parquet_file)
        for _, row in df.iterrows():
            games_by_date[row['date']].append((row['game_id'], row['date']))
    
    # Convert to sorted list
    return dict(sorted(games_by_date.items()))


def download_pbp_data(game_id, date, verbose=False):
    """
    Download and cache play-by-play data for a game.
    
    Saves full ESPN JSON response (includes plays + boxscore).
    
    Args:
        game_id: ESPN game ID
        date: Game date (YYYY-MM-DD)
        verbose: Print progress
    
    Returns:
        True if successful, False otherwise
    """
    date_str = date.replace('-', '')
    output_file = PBP_DATA_DIR / f"{date_str}_{game_id}.json"
    
    # Skip if already exists
    if output_file.exists():
        return True  # Don't print here, handle in main loop
    
    try:
        # Download from ESPN
        pbp_data = get_play_by_play(game_id)
        
        if pbp_data is None:
            if verbose:
                print(f"    ❌ Game {game_id}: No data")
            return False
        
        # Save full JSON response
        with open(output_file, 'w') as f:
            json.dump(pbp_data, f)
        
        if verbose:
            num_plays = len(pbp_data.get('plays', []))
            print(f"    ✅ Game {game_id}: Downloaded ({num_plays} plays)")
        
        # Mark as completed
        add_to_progress('pbp_progress.json', game_id)
        
        return True
        
    except Exception as e:
        if verbose:
            print(f"    ❌ Game {game_id}: Error - {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Download NBA play-by-play data')
    parser.add_argument('--verbose', action='store_true', help='Print progress')
    args = parser.parse_args()
    
    if args.verbose:
        print(f"🏀 Downloading play-by-play data")
        print()
    
    # Load all game IDs grouped by date
    games_by_date = get_all_game_ids_by_date()
    
    total_games = sum(len(games) for games in games_by_date.values())
    
    if args.verbose:
        print(f"   Found {total_games} total games across {len(games_by_date)} dates")
        print()
    
    games_downloaded = 0
    games_failed = 0
    games_skipped = 0
    
    # Iterate through each date
    for date in sorted(games_by_date.keys()):
        games_for_date = games_by_date[date]
        
        if args.verbose:
            print(f"\n--- {date} ({len(games_for_date)} games) ---")
        
        date_downloaded = 0
        date_failed = 0
        date_skipped = 0
        
        # Process each game for this date
        for game_id, game_date in games_for_date:
            date_str = game_date.replace('-', '')
            output_file = PBP_DATA_DIR / f"{date_str}_{game_id}.json"
            
            if output_file.exists():
                date_skipped += 1
                games_skipped += 1
                if args.verbose:
                    print(f"    ⏭️  Game {game_id}: Cached")
            else:
                success = download_pbp_data(game_id, game_date, verbose=args.verbose)
                
                if success:
                    date_downloaded += 1
                    games_downloaded += 1
                else:
                    date_failed += 1
                    games_failed += 1
                
                time.sleep(RATE_LIMIT_BETWEEN_GAMES)
        
        # Verify count matches expected
        if args.verbose:
            total_for_date = date_skipped + date_downloaded + date_failed
            expected = len(games_for_date)
            if total_for_date != expected:
                print(f"    ⚠️  Count mismatch: processed {total_for_date}, expected {expected}")
            else:
                print(f"    ✅ Verified: {total_for_date}/{expected} games accounted for")
    
    if args.verbose:
        print()
        print(f"⏭️  Skipped {games_skipped} already-cached games")
        print(f"✅ Downloaded {games_downloaded} games")
        if games_failed > 0:
            print(f"❌ Failed {games_failed} games")
        print(f"📁 Cached in: {PBP_DATA_DIR}")
    
    # Exit with code 1 if no games were downloaded (all done)
    if games_downloaded == 0 and games_failed == 0:
        if args.verbose:
            print()
            print("🎉 All games downloaded - exiting")
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
