"""
Build player team history from career game logs.

Context:
========
After trade deadline, static roster caches become outdated. We need historical 
team assignments to correctly join player props data with team info for any game date.

This script:
1. Fetches career game logs for all active NBA players
2. Groups consecutive games by team
3. Creates date ranges (valid_from, valid_to) for each team stint
4. Outputs player_team_history.csv with temporal ranges

Data Source:
============
- NBA API PlayerGameLogs (career history for all players)
- Only includes games where player logged minutes (regular season + playoffs)
- No preseason games

Output Schema:
==============
player_normalized,team,valid_from,valid_to
Anthony Davis,LAL,2019-07-06,2026-02-05
Anthony Davis,DAL,2026-02-06,NULL

- valid_from: First game date with new team (or estimated trade date)
- valid_to: Last game date with team (NULL = current team)

Usage:
======
    python scripts/build_player_team_history_from_gamelogs.py
    
    # Optional: limit to current season only (faster for testing)
    python scripts/build_player_team_history_from_gamelogs.py --current-season-only

Output:
=======
Local + S3:
    - data/02_cache/player_team_history.csv
    - s3://nba-betting-mt/data/02_cache/player_team_history.csv
"""

import pandas as pd
import sys
from pathlib import Path
import time
import boto3
from io import StringIO
from datetime import datetime, timedelta
import argparse
import ssl
import urllib3
import requests

# Fix SSL certificate issues (must be done BEFORE importing nba_api)
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Monkey-patch requests Session to disable SSL verification
original_request = requests.Session.request

def patched_request(self, *args, **kwargs):
    kwargs['verify'] = False
    return original_request(self, *args, **kwargs)

requests.Session.request = patched_request

# Find repo root
current_dir = Path(__file__).resolve()
repo_root = current_dir
while not (repo_root / '.gitignore').exists():
    repo_root = repo_root.parent
    if repo_root == repo_root.parent:
        raise RuntimeError("Could not find repo root")

sys.path.append(str(repo_root))

from src.player_name_utils import normalize_player_name
from src.config import CURRENT_NBA_SEASON, EMOJI

# Output paths
OUTPUT_DIR = repo_root / 'data' / '02_cache'
OUTPUT_FILE = OUTPUT_DIR / 'player_team_history.csv'

# S3 Configuration
S3_BUCKET = 'nba-betting-mt'
S3_KEY = 'data/02_cache/player_team_history.csv'


def get_all_active_players():
    """
    Get list of all active NBA players from current roster cache.
    
    Returns:
        List of player names
    """
    roster_cache = OUTPUT_DIR / 'nba_full_roster_cache.csv'
    
    if not roster_cache.exists():
        print(f"{EMOJI['error']} Roster cache not found: {roster_cache}")
        print("   Run: python scripts/build_full_roster_cache.py")
        sys.exit(1)
    
    df = pd.read_csv(roster_cache)
    players = df['player_name_nba_api'].unique().tolist()
    
    print(f"{EMOJI['success']} Found {len(players)} active players in roster cache")
    return players


def fetch_player_career_gamelogs(player_name, current_season_only=False):
    """
    Fetch career game logs for a single player.
    
    Args:
        player_name: Player name (NBA API format)
        current_season_only: If True, only fetch current season
        
    Returns:
        DataFrame with columns: GAME_DATE, TEAM_ABBREVIATION, PLAYER_NAME
        Empty DataFrame if player not found or no games
    """
    try:
        from nba_api.stats.endpoints import playergamelogs
        from nba_api.stats.static import players
        
        # Find player ID
        all_players = players.get_players()
        player_info = [p for p in all_players if p['full_name'] == player_name]
        
        if not player_info:
            print(f"   {EMOJI['warning']} Player not found in NBA API: {player_name}")
            return pd.DataFrame()
        
        player_id = player_info[0]['id']
        
        # Fetch game logs
        if current_season_only:
            season_str = CURRENT_NBA_SEASON
        else:
            season_str = None  # All seasons
        
        # Regular season + Playoffs
        game_logs_list = []
        
        for season_type in ['Regular Season', 'Playoffs']:
            try:
                logs = playergamelogs.PlayerGameLogs(
                    player_id_nullable=player_id,
                    season_nullable=season_str,
                    season_type_nullable=season_type
                )
                
                df = logs.get_data_frames()[0]
                
                if not df.empty:
                    game_logs_list.append(df)
                
                time.sleep(0.6)  # Rate limit
                
            except Exception as e:
                print(f"   {EMOJI['warning']} Error fetching {season_type} for {player_name}: {e}")
                continue
        
        if not game_logs_list:
            return pd.DataFrame()
        
        # Combine regular season and playoffs
        all_games = pd.concat(game_logs_list, ignore_index=True)
        
        # Keep only needed columns
        cols_needed = ['GAME_DATE', 'TEAM_ABBREVIATION', 'PLAYER_NAME']
        all_games = all_games[cols_needed]
        
        # Convert GAME_DATE to datetime
        all_games['GAME_DATE'] = pd.to_datetime(all_games['GAME_DATE'])
        
        # Sort by date (oldest first)
        all_games = all_games.sort_values('GAME_DATE')
        
        return all_games
        
    except Exception as e:
        print(f"   {EMOJI['error']} Error fetching game logs for {player_name}: {e}")
        return pd.DataFrame()


def create_team_history_from_gamelogs(game_logs_df):
    """
    Convert game logs to team history with date ranges.
    
    Groups consecutive games by team and creates valid_from/valid_to ranges.
    
    Args:
        game_logs_df: DataFrame with GAME_DATE, TEAM_ABBREVIATION, PLAYER_NAME
        
    Returns:
        DataFrame with player_normalized, team, valid_from, valid_to
    """
    if game_logs_df.empty:
        return pd.DataFrame(columns=['player_normalized', 'team', 'valid_from', 'valid_to'])
    
    player_name = game_logs_df['PLAYER_NAME'].iloc[0]
    player_normalized = normalize_player_name(player_name)
    
    history = []
    
    # Group by team (consecutive games)
    game_logs_df['team_change'] = game_logs_df['TEAM_ABBREVIATION'] != game_logs_df['TEAM_ABBREVIATION'].shift()
    game_logs_df['team_stint'] = game_logs_df['team_change'].cumsum()
    
    for stint_id, stint_games in game_logs_df.groupby('team_stint'):
        team = stint_games['TEAM_ABBREVIATION'].iloc[0]
        
        # First game date with this team
        first_game = stint_games['GAME_DATE'].min()
        last_game = stint_games['GAME_DATE'].max()
        
        # Estimate trade date as day before first game (or keep first game date)
        # We'll use first game date as valid_from
        valid_from = first_game.date()
        
        # Check if this is the last stint (current team)
        is_last_stint = stint_id == game_logs_df['team_stint'].max()
        
        if is_last_stint:
            valid_to = None  # Current team
        else:
            # Set valid_to to last game date with this team
            valid_to = last_game.date()
        
        history.append({
            'player_normalized': player_normalized,
            'team': team,
            'valid_from': valid_from,
            'valid_to': valid_to
        })
    
    return pd.DataFrame(history)


def upload_to_s3(df, s3_key):
    """
    Upload DataFrame to S3 as CSV.
    
    Args:
        df: DataFrame to upload
        s3_key: S3 key path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        s3_client = boto3.client('s3')
        
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=csv_buffer.getvalue(),
            ContentType='text/csv'
        )
        
        print(f"{EMOJI['success']} Uploaded to S3: s3://{S3_BUCKET}/{s3_key}")
        return True
        
    except Exception as e:
        print(f"{EMOJI['warning']} S3 upload failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Build player team history from game logs')
    parser.add_argument('--current-season-only', action='store_true',
                       help='Only fetch current season (faster for testing)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Building Player Team History from Game Logs")
    print("=" * 70)
    print(f"Season: {CURRENT_NBA_SEASON if args.current_season_only else 'Full Career'}")
    print()
    
    # Step 1: Get all active players
    print("Step 1: Loading active players from roster cache...")
    players = get_all_active_players()
    print()
    
    # Step 2: Fetch game logs for each player
    print(f"Step 2: Fetching game logs for {len(players)} players...")
    print("   (This will take several minutes due to API rate limits)")
    print()
    
    all_history = []
    successful = 0
    failed = 0
    
    for i, player_name in enumerate(players, 1):
        print(f"   [{i}/{len(players)}] {player_name}...", end=' ')
        
        game_logs = fetch_player_career_gamelogs(player_name, args.current_season_only)
        
        if game_logs.empty:
            print(f"{EMOJI['warning']} No games found")
            failed += 1
            continue
        
        # Convert to team history
        player_history = create_team_history_from_gamelogs(game_logs)
        
        if not player_history.empty:
            all_history.append(player_history)
            num_stints = len(player_history)
            print(f"{EMOJI['success']} {num_stints} team stint(s)")
            successful += 1
        else:
            print(f"{EMOJI['warning']} Failed to create history")
            failed += 1
    
    print()
    print(f"{EMOJI['success']} Processed {successful}/{len(players)} players successfully")
    print(f"{EMOJI['warning']} {failed} players had no game logs or failed")
    print()
    
    # Step 3: Combine all histories
    if not all_history:
        print(f"{EMOJI['error']} No history data collected!")
        sys.exit(1)
    
    print("Step 3: Combining all player histories...")
    history_df = pd.concat(all_history, ignore_index=True)
    
    # Sort by player, then date
    history_df = history_df.sort_values(['player_normalized', 'valid_from'])
    
    print(f"{EMOJI['success']} Created {len(history_df)} team history records")
    print()
    
    # Step 4: Save locally
    print("Step 4: Saving to local file...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(OUTPUT_FILE, index=False)
    print(f"{EMOJI['success']} Saved: {OUTPUT_FILE}")
    print()
    
    # Step 5: Upload to S3
    print("Step 5: Uploading to S3...")
    upload_to_s3(history_df, S3_KEY)
    print()
    
    # Summary
    print("=" * 70)
    print(f"{EMOJI['success']} Player Team History Built!")
    print("=" * 70)
    print(f"Total players: {history_df['player_normalized'].nunique()}")
    print(f"Total team stints: {len(history_df)}")
    print(f"Date range: {history_df['valid_from'].min()} to {history_df['valid_to'].max()}")
    print()
    print("Sample history:")
    print(history_df.head(10).to_string(index=False))
    print()
    print(f"Local: {OUTPUT_FILE}")
    print(f"S3: s3://{S3_BUCKET}/{S3_KEY}")


if __name__ == '__main__':
    main()
