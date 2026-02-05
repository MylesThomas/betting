"""
Build player team history from career game logs.

Input:
======
- S3: s3://the-odds-api-mt/nba/historical_player_props/{2023-24,2024-25,2025-26}/*.csv
  (Scans to discover unique player names)

Output:
=======
- S3: s3://nba-betting-mt/data/02_cache/player_team_history.parquet
- Local: data/02_cache/player_team_history.parquet

Schema: player_normalized | team | valid_from | valid_to

Example:
  player_normalized | team | valid_from  | valid_to
  Anthony Davis     | LAL  | 2019-10-22  | 2026-02-05
  Anthony Davis     | DAL  | 2026-02-07  | NULL

Functions:
==========
- build_team_history(): Main orchestration function
- get_all_players_from_s3(): Discover players from S3 betting data
- fetch_player_career_gamelogs(): Fetch career game logs for one player
- create_team_history_from_gamelogs(): Convert game logs to date ranges
- upload_to_s3(): Upload parquet to S3

CLI Usage:
==========
    python -m src.player_team_history.builder
    python -m src.player_team_history.builder --current-season-only
    python -m src.player_team_history.builder --sample 100  # Test with sample
"""

import pandas as pd
import sys
from pathlib import Path
import time
import boto3
from io import BytesIO
from datetime import datetime
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
OUTPUT_FILE = OUTPUT_DIR / 'player_team_history.parquet'

# S3 Configuration
S3_BUCKET = 'nba-betting-mt'
S3_KEY = 'data/02_cache/player_team_history.parquet'

# Rate limit delay (can be overridden by --slow-mode)
RATE_LIMIT_DELAY = 1.0


def get_all_players_from_s3(sample_size=None):
    """
    Get list of all NBA players from S3 betting/props data.
    
    Input: s3://the-odds-api-mt/nba/historical_player_props/{2023-24,2024-25,2025-26}/*.csv
    
    Args:
        sample_size: Optional limit on number of S3 files to scan (for testing)
    
    Returns:
        List of player names
    """
    from src.player_team_history.discovery import discover_all_players
    
    players_set = discover_all_players(
        s3_sample_size=sample_size,
        verbose=True
    )
    
    players = list(players_set)
    return players


def fetch_player_career_gamelogs(player_name, current_season_only=False, max_retries=3):
    """
    Fetch career game logs for a single player.
    
    Uses playergamelog.PlayerGameLog (singular) - the working endpoint.
    
    Args:
        player_name: Player name (NBA API format)
        current_season_only: If True, only fetch current season
        max_retries: Number of retries on failure
        
    Returns:
        DataFrame with columns: GAME_DATE, TEAM_ABBREVIATION, PLAYER_NAME
    """
    from nba_api.stats.endpoints import playergamelog, commonplayerinfo
    from nba_api.stats.static import players
    
    # Find player ID with fuzzy matching
    all_players = players.get_players()
    
    # Try 1: Exact match
    player_info = [p for p in all_players if p['full_name'].lower() == player_name.lower()]
    
    # Try 2: Partial match
    if not player_info:
        player_info = [p for p in all_players if player_name.lower() in p['full_name'].lower()]
    
    # Try 3: Reversed name (handles "Caldwell Pope Kentavious" -> "Kentavious Caldwell-Pope")
    if not player_info:
        parts = player_name.split()
        if len(parts) >= 2:
            reversed_name = f"{parts[-1]} {' '.join(parts[:-1])}"
            player_info = [p for p in all_players if reversed_name.lower() in p['full_name'].lower()]
    
    if not player_info:
        raise ValueError(f"Player not found in NBA API: {player_name}")
    
    # If multiple matches, prefer active players
    if len(player_info) > 1:
        active = [p for p in player_info if p.get('is_active', False)]
        player_info = active if active else [player_info[0]]
    
    player_id = player_info[0]['id']
    
    # Get seasons to fetch
    if current_season_only:
        seasons = [CURRENT_NBA_SEASON]
    else:
        # Get all career seasons
        try:
            player_info_endpoint = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
            info_df = player_info_endpoint.get_data_frames()[0]
            
            if not info_df.empty:
                from_year = int(info_df['FROM_YEAR'].iloc[0])
                to_year = int(info_df['TO_YEAR'].iloc[0])
                
                seasons = []
                for year in range(from_year, to_year + 1):
                    season = f"{year}-{str(year + 1)[-2:]}"
                    seasons.append(season)
            else:
                seasons = [CURRENT_NBA_SEASON]
        except Exception:
            seasons = [CURRENT_NBA_SEASON]
    
    # Fetch game logs for each season
    game_logs_list = []
    
    for season in seasons:
        for attempt in range(max_retries):
            try:
                # USE WORKING ENDPOINT: playergamelog.PlayerGameLog (singular)
                logs = playergamelog.PlayerGameLog(
                    player_id=player_id,
                    season=season
                )
                
                df = logs.get_data_frames()[0]
                
                if not df.empty:
                    game_logs_list.append(df)
                
                # Success - break retry loop
                break
                
            except Exception as e:
                if attempt < max_retries - 1:
                    # Exponential backoff: 2s, 4s, 8s
                    wait_time = 2 ** (attempt + 1)
                    time.sleep(wait_time)
                    continue
                else:
                    # Last attempt failed - skip this season
                    pass
        
        # Rate limit between seasons
        time.sleep(RATE_LIMIT_DELAY)
    
    if not game_logs_list:
        return pd.DataFrame()
    
    # Combine all seasons
    all_games = pd.concat(game_logs_list, ignore_index=True)
    
    # Rename columns to match expected format
    if 'TEAM_ABBREVIATION' not in all_games.columns and 'MATCHUP' in all_games.columns:
        # Extract team from MATCHUP (e.g., "LAL @ GSW" -> "LAL")
        def extract_team(matchup):
            if pd.isna(matchup):
                return None
            if '@' in matchup:
                return matchup.split('@')[0].strip()
            elif 'vs.' in matchup:
                return matchup.split('vs.')[0].strip()
            return None
        
        all_games['TEAM_ABBREVIATION'] = all_games['MATCHUP'].apply(extract_team)
    
    # Add PLAYER_NAME if not present
    if 'PLAYER_NAME' not in all_games.columns:
        all_games['PLAYER_NAME'] = player_name
    
    # Keep only needed columns
    cols_needed = ['GAME_DATE', 'TEAM_ABBREVIATION', 'PLAYER_NAME']
    all_games = all_games[cols_needed]
    
    # Convert GAME_DATE to datetime
    all_games['GAME_DATE'] = pd.to_datetime(all_games['GAME_DATE'])
    
    # Sort by date (oldest first)
    all_games = all_games.sort_values('GAME_DATE')
    
    return all_games


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
        
        # First and last game date with this team
        first_game = stint_games['GAME_DATE'].min()
        last_game = stint_games['GAME_DATE'].max()
        
        valid_from = first_game.date()
        
        # Check if this is the last stint (current team)
        is_last_stint = stint_id == game_logs_df['team_stint'].max()
        
        valid_to = None if is_last_stint else last_game.date()
        
        history.append({
            'player_normalized': player_normalized,
            'team': team,
            'valid_from': valid_from,
            'valid_to': valid_to
        })
    
    return pd.DataFrame(history)


def upload_to_s3(df, s3_key):
    """
    Upload DataFrame to S3 as parquet.
    
    Args:
        df: DataFrame to upload
        s3_key: S3 key path
    """
    s3_client = boto3.client('s3')
    
    parquet_buffer = BytesIO()
    df.to_parquet(parquet_buffer, index=False, engine='pyarrow')
    parquet_buffer.seek(0)
    
    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key=s3_key,
        Body=parquet_buffer.getvalue(),
        ContentType='application/octet-stream'
    )
    
    print(f"{EMOJI['success']} Uploaded to S3: s3://{S3_BUCKET}/{s3_key}")


def build_team_history(current_season_only=False):
    """
    Build player team history from game logs.
    
    Main orchestration function that:
    1. Loads active players
    2. Fetches game logs for each
    3. Creates date ranges
    4. Saves to S3
    
    Args:
        current_season_only: If True, only fetch current season
        
    Returns:
        DataFrame with player team history
    """
    print("=" * 70)
    print("Building Player Team History from Game Logs")
    print("=" * 70)
    print(f"Season: {CURRENT_NBA_SEASON if current_season_only else 'Full Career'}")
    print()
    print("Input:  s3://the-odds-api-mt/nba/historical_player_props/{2023-24,2024-25,2025-26}/*.csv")
    print("Output: s3://nba-betting-mt/data/02_cache/player_team_history.parquet")
    print()
    
    # Step 1: Get all players from S3 betting data
    print("Step 1: Discovering players from S3 betting data...")
    players = get_all_players_from_s3()
    print()
    
    # Step 2: Fetch game logs for each player
    print(f"Step 2: Fetching game logs for {len(players)} players...")
    print("   (This will take several minutes due to API rate limits)")
    print()
    
    all_history = []
    successful = 0
    failed = 0
    
    for i, player_name in enumerate(players, 1):
        print(f"   [{i}/{len(players)}] {player_name}...", end=' ', flush=True)
        
        try:
            game_logs = fetch_player_career_gamelogs(player_name, current_season_only)
            
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
                
        except Exception as e:
            error_msg = str(e)
            # Truncate long error messages
            if len(error_msg) > 80:
                error_msg = error_msg[:77] + "..."
            print(f"{EMOJI['error']} {error_msg}")
            failed += 1
            continue
        
        # Checkpoint every 50 players
        if i % 50 == 0 and all_history:
            checkpoint_df = pd.concat(all_history, ignore_index=True)
            checkpoint_file = OUTPUT_DIR / f'player_team_history_checkpoint_{i}.parquet'
            checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_df.to_parquet(checkpoint_file, index=False)
            print(f"\n   {EMOJI['save']} Checkpoint saved: {successful} players ({i}/{len(players)})\n")
    
    print()
    print(f"{EMOJI['success']} Processed {successful}/{len(players)} players successfully")
    print(f"{EMOJI['warning']} {failed} players had no game logs or failed")
    print()
    
    # Step 3: Combine all histories
    if not all_history:
        raise RuntimeError("No history data collected!")
    
    print("Step 3: Combining all player histories...")
    history_df = pd.concat(all_history, ignore_index=True)
    
    # Sort by player, then date
    history_df = history_df.sort_values(['player_normalized', 'valid_from'])
    
    print(f"{EMOJI['success']} Created {len(history_df)} team history records")
    print()
    
    # Step 4: Save locally (backup)
    print("Step 4: Saving to local file...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    history_df.to_parquet(OUTPUT_FILE, index=False)
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
    
    return history_df


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Build player team history from game logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Input:  s3://the-odds-api-mt/nba/historical_player_props/{2023-24,2024-25,2025-26}/*.csv
Output: s3://nba-betting-mt/data/02_cache/player_team_history.parquet

Examples:
  # Full career history
  python -m src.player_team_history.builder
  
  # Current season only (faster)
  python -m src.player_team_history.builder --current-season-only
  
  # Test with sample
  python -m src.player_team_history.builder --sample 100
        """
    )
    parser.add_argument('--current-season-only', action='store_true',
                       help='Only fetch current season (faster for testing)')
    parser.add_argument('--slow-mode', action='store_true',
                       help='Add extra delays between requests (helps with rate limiting)')
    parser.add_argument('--sample', type=int,
                       help='Only scan this many S3 files (for testing)')
    args = parser.parse_args()
    
    # Set global rate limit delay if slow mode
    global RATE_LIMIT_DELAY
    if args.slow_mode:
        RATE_LIMIT_DELAY = 2.0
        print(f"{EMOJI['warning']} Slow mode enabled: 2s delay between requests\n")
    
    # Note: sample argument not yet implemented in builder
    # TODO: Pass sample_size to get_all_players_from_s3()
    if args.sample:
        print(f"{EMOJI['warning']} --sample not yet implemented in builder")
        print(f"   (Use discovery.py directly for testing)")
        print()
    
    build_team_history(current_season_only=args.current_season_only)


if __name__ == '__main__':
    main()
