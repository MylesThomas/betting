"""
Get NBA game results for all seasons dating back to 2014-15.

API of choice: nba_api
- https://github.com/swar/nba_api


Date: 2025-11-20
Author: Myles Thomas
"""

import pandas as pd
import os
from pathlib import Path
from datetime import datetime
import time

# ============================================================================
# SSL FIX - Must be done BEFORE importing nba_api
# See: api_setup/fixing_ssl.md for full explanation
# ============================================================================
import ssl
import urllib3
import requests

# Disable SSL verification warnings
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Monkey-patch requests Session to disable SSL verification
original_request = requests.Session.request

def patched_request(self, *args, **kwargs):
    kwargs['verify'] = False
    return original_request(self, *args, **kwargs)

requests.Session.request = patched_request

# NOW import nba_api (it will use the patched requests)
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams


def get_project_root():
    """Find project root by locating .gitignore file."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / '.gitignore').exists():
            return parent
    raise FileNotFoundError("Could not find project root (no .gitignore found)")


LST_SEASONS = [f"{year}-{str(year+1)[-2:]}" for year in range(2014, 2024 + 1)]

OUTPUT_DIR = os.path.join(get_project_root(), 'data', '01_input', 'nba_api', 'historical')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

print(f"get_project_root(): {get_project_root()}")
print(f"LST_SEASONS: {LST_SEASONS}")
print(f"OUTPUT_DIR: {OUTPUT_DIR}")

def fetch_games_for_season(season_str):
    """
    Fetch all NBA games for a given season.
    
    Args:
        season_str: Season string in format 'YYYY-YY' (e.g., '2014-15')
    
    Returns:
        DataFrame with game results
    """
    print(f"\nFetching games for {season_str} season...")
    
    # Fetch games using LeagueGameFinder
    gamefinder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season_str,
        league_id_nullable='00'  # NBA
    )
    
    games_df = gamefinder.get_data_frames()[0]
    
    print(f"  Retrieved {len(games_df)} game records (includes home + away)")
    
    # Add season column for easier filtering later
    games_df['SEASON'] = season_str
    
    # Sleep to respect API rate limits
    time.sleep(1)
    
    return games_df


def main():
    """Fetch NBA game results from 2014-15 season to current season."""
    
    print(f"Will fetch game results for {len(LST_SEASONS)} seasons: {LST_SEASONS[0]} to {LST_SEASONS[-1]}")
    
    all_games = []
    
    for season in LST_SEASONS:
        try:
            games_df = fetch_games_for_season(season)
            all_games.append(games_df)
            
            # Also save individual season file
            season_file = os.path.join(OUTPUT_DIR, f'nba_games_{season.replace("-", "_")}.csv')
            games_df.to_csv(season_file, index=False)
            print(f"  Saved to {os.path.basename(season_file)}")
            
        except Exception as e:
            print(f"  ERROR fetching {season}: {e}")
            continue
    
    # Combine all seasons
    if all_games:
        combined_df = pd.concat(all_games, ignore_index=True)
        
        # Sort by game date
        combined_df['GAME_DATE'] = pd.to_datetime(combined_df['GAME_DATE'])
        combined_df = combined_df.sort_values('GAME_DATE')
        
        # Save combined file
        combined_file = os.path.join(OUTPUT_DIR, 'nba_games_all_seasons.csv')
        combined_df.to_csv(combined_file, index=False)
        
        print(f"\n{'='*60}")
        print(f"SUCCESS!")
        print(f"Total game records: {len(combined_df):,}")
        print(f"Combined file saved: {os.path.basename(combined_file)}")
        print(f"Date range: {combined_df['GAME_DATE'].min()} to {combined_df['GAME_DATE'].max()}")
        print(f"{'='*60}")
        
        # Show sample of data
        print("\nSample of data (first 5 rows):")
        print(combined_df.head().to_string())
        
        print("\nColumns:")
        print(combined_df.columns.tolist())
        
    else:
        print("\nERROR: No games were successfully fetched")


if __name__ == '__main__':
    main()
