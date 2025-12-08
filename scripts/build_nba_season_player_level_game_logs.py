"""
Build comprehensive NBA season game logs for all players.

This script fetches EVERY game log for EVERY player from the current season (2025-26).
Since the NBA API is free, we grab all available stats, not just 3PT data.

Data Organization:
    Individual player files stored in season directories for granular caching:
    
    data/01_input/nba_api/season_game_logs/2025_26/
        ├── LeBron_James.csv
        ├── Stephen_Curry.csv
        ├── Giannis_Antetokounmpo.csv
        └── ... (one file per player)
    
    Benefits:
    - Only refetch players with stale data (>12 hours old)
    - Failed players don't affect others
    - Parallel updates are resumable

This is the data foundation for:
1. Daily betting opportunity finder
2. Streak analysis
3. Trend detection
4. Player performance tracking

Usage:
    # Build 2025-26 season cache (recommended daily run)
    python scripts/build_season_game_logs.py
    
    # Same as above (explicit season)
    python scripts/build_season_game_logs.py --season 2025-26
    
    # Force rebuild all players (ignore 12-hour cache)
    python scripts/build_season_game_logs.py --force
    
    # Build previous season for backtesting
    python scripts/build_season_game_logs.py --season 2024-25

Author: Myles Thomas
Date: 2024-11-24
"""

import pandas as pd
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Fix SSL certificate issues with NBA API (must be done BEFORE importing nba_api)
import ssl
import urllib3
import requests

ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Monkey-patch requests Session to disable SSL verification
original_request = requests.Session.request

def patched_request(self, *args, **kwargs):
    kwargs['verify'] = False
    return original_request(self, *args, **kwargs)

requests.Session.request = patched_request

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from src.config import CURRENT_NBA_SEASON
from src.nba_gamelog_utils import parse_game_logs, get_empty_gamelog_dataframe
from src.config_loader import get_file_path

# Import NBA API
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog


# ============================================================================
# CONFIG
# ============================================================================

# Output path - organized under 01_input/nba_api/season_game_logs/SEASON/
# Each player gets their own file for granular caching
BASE_OUTPUT_DIR = Path(__file__).parent.parent / 'data' / '01_input' / 'nba_api' / 'season_game_logs'

# Cache settings
CACHE_MAX_AGE_HOURS = 12  # Refresh every 12 hours

# Parallel processing
MAX_WORKERS = 1  # Sequential processing only (NBA API is too flaky for parallel)
MAX_RETRIES = 3  # Number of attempts (3 = try up to 3 times)
RETRY_DELAY = 2  # Seconds to wait between retries
REQUEST_DELAY = 0.5  # Delay between each player request (avoid rate limits)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_all_active_players():
    """
    Get all active NBA players.
    
    Returns:
        List of player dictionaries with id, full_name
    """
    print("Fetching all active NBA players...")
    all_players = players.get_active_players()
    print(f"✅ Found {len(all_players)} active players")
    return all_players


def get_player_game_logs(player_id, player_name, season='2025-26', max_retries=MAX_RETRIES):
    """
    Fetch game logs for a specific player for the current season.
    
    Includes retry logic for connection errors.
    
    Args:
        player_id: NBA API player ID
        player_name: Player name (for logging)
        season: Season string (e.g., '2025-26')
        max_retries: Number of retry attempts (3 = 3 actual attempts)
    
    Returns:
        Tuple of (DataFrame or None, error_occurred: bool)
        - (df, False) if successful
        - (None, False) if player has no games
        - (None, True) if error occurred
    """
    for attempt in range(max_retries):
        try:
            # Fetch game logs with increased timeout
            gamelog = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                season_type_all_star='Regular Season',
                timeout=60  # Increased from default 30s
            )
            
            df = gamelog.get_data_frames()[0]
            
            if df.empty:
                return None, False  # No games, not an error
            
            # Add player name for reference
            df['PLAYER_NAME'] = player_name
            df['PLAYER_ID'] = player_id
            
            return df, False  # Success
        
        except Exception as e:
            if attempt < max_retries - 1:
                # Wait before retrying
                print(f"   ⏳ Retry {attempt + 1}/{max_retries - 1} for {player_name}...")
                time.sleep(RETRY_DELAY)
            else:
                # Final attempt failed - this is an error
                error_msg = str(e)
                if 'timeout' in error_msg.lower():
                    print(f"   ❌ {player_name}: Timeout after {max_retries} attempts")
                else:
                    print(f"   ❌ {player_name}: {error_msg[:80]}")
                return None, True  # Error occurred
    
    return None, True  # Should never reach here, but return error just in case


# parse_game_logs is now imported from src.nba_gamelog_utils


def get_player_file_path(player_name, season):
    """
    Get the file path for a specific player's game logs.
    
    Args:
        player_name: Player full name
        season: Season string (e.g., '2025-26')
    
    Returns:
        Path object for player's CSV file
    """
    # Create season directory
    season_dir = BASE_OUTPUT_DIR / season.replace('-', '_')
    season_dir.mkdir(parents=True, exist_ok=True)
    
    # Sanitize player name for filename (replace spaces with underscores)
    safe_name = player_name.replace(' ', '_').replace('.', '').replace("'", '')
    filename = f"{safe_name}.csv"
    
    return season_dir / filename


def is_player_data_fresh(player_name, season, max_age_hours=CACHE_MAX_AGE_HOURS):
    """
    Check if a player's data file exists and is fresh.
    
    Returns:
        True if data is fresh, False if stale or missing
    """
    file_path = get_player_file_path(player_name, season)
    
    if not file_path.exists():
        return False
    
    file_age = datetime.now() - datetime.fromtimestamp(file_path.stat().st_mtime)
    
    return file_age <= timedelta(hours=max_age_hours)


def save_player_game_logs(df, player_name, season, allow_empty=False):
    """
    Save a player's game logs to their individual file.
    
    Args:
        df: DataFrame with game logs (can be empty if allow_empty=True)
        player_name: Player full name
        season: Season string
        allow_empty: If True, saves empty dataframes (to mark "no games" players)
    """
    if df is None:
        return
    
    if df.empty and not allow_empty:
        return
    
    file_path = get_player_file_path(player_name, season)
    df.to_csv(file_path, index=False)


def load_all_player_game_logs(season):
    """
    Load and combine all player game log files for a season.
    
    Skips empty files (players with no games).
    
    Args:
        season: Season string (e.g., '2025-26')
    
    Returns:
        Combined DataFrame with all player game logs
    """
    season_dir = BASE_OUTPUT_DIR / season.replace('-', '_')
    
    if not season_dir.exists():
        return pd.DataFrame()
    
    all_files = list(season_dir.glob('*.csv'))
    
    if not all_files:
        return pd.DataFrame()
    
    dfs = []
    for file_path in all_files:
        try:
            df = pd.read_csv(file_path)
            
            # Skip empty files (players with no games)
            if df.empty:
                continue
            
            df['date'] = pd.to_datetime(df['date'])
            dfs.append(df)
        except Exception as e:
            print(f"⚠️  Error loading {file_path.name}: {e}")
    
    if not dfs:
        return pd.DataFrame()
    
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values('date', ascending=False).reset_index(drop=True)
    
    return combined


def check_missing_players(df_all, all_players, failed_players):
    """
    Check which players from the roster are missing from the final data.
    
    Distinguishes between:
    - Failed: We tried to fetch but got an error
    - Missing: Player exists but has no games (G-League, injured, etc.)
    
    Args:
        df_all: Combined DataFrame with game logs
        all_players: List of all player dictionaries from NBA API
        failed_players: List of player names that failed to fetch
    
    Returns:
        Tuple of (failed_list, missing_list)
    """
    if df_all.empty:
        all_names = [p['full_name'] for p in all_players]
        return failed_players, [n for n in all_names if n not in failed_players]
    
    # Get players we have data for
    players_in_data = set(df_all['player'].unique())
    
    # Get all players from roster
    all_player_names = set([p['full_name'] for p in all_players])
    
    # Find players without data
    no_data = all_player_names - players_in_data
    
    # Separate into failed vs missing
    failed_set = set(failed_players)
    missing = no_data - failed_set
    
    return sorted(failed_players), sorted(missing)


def log_failed_and_missing_players(failed_players, missing_players):
    """
    Log information about failed fetches and missing players.
    
    Args:
        failed_players: Players we tried to fetch but got errors
        missing_players: Players with no games (G-League, injured, rookies)
    """
    print()
    print("="*80)
    print("PLAYER DATA SUMMARY")
    print("="*80)
    
    # Failed players (errors during fetch)
    if failed_players:
        print()
        print(f"❌ FAILED TO FETCH ({len(failed_players)})")
        print("-"*80)
        print("These players encountered errors during fetch (retry recommended):")
        print()
        for i, player in enumerate(failed_players, 1):
            print(f"  {i:3}. {player}")
        print()
        print("💡 Tip: Run script again to retry failed players")
    
    # Missing players (no games)
    if missing_players:
        print()
        print(f"⚪ NO GAMES PLAYED ({len(missing_players)})")
        print("-"*80)
        print("These players are rostered but have no game logs:")
        print("(Likely reasons: G-League assignment, injured, rookies with no games)")
        print()
        for i, player in enumerate(missing_players, 1):
            print(f"  {i:3}. {player}")
        print()
        print("Note: This is normal - not all rostered players have played yet.")
    
    # All good
    if not failed_players and not missing_players:
        print()
        print("✅ All 531 players accounted for!")
    
    print()
    print("="*80)


# ============================================================================
# MAIN BUILD FUNCTION
# ============================================================================

def build_season_game_logs(season='2025-26', force=False):
    """
    Build comprehensive season game logs for all active players.
    
    Uses individual player files for granular caching:
    - Only fetches players with stale/missing data
    - Saves each player to separate file
    - Combines all files at the end
    
    Args:
        season: Season string (e.g., '2025-26')
        force: Force rebuild all players even if data is fresh
    
    Returns:
        DataFrame with all game logs
    """
    
    print()
    print("="*80)
    print(f"BUILDING SEASON GAME LOGS - {season}")
    print("="*80)
    print()
    
    # Get all active players
    all_players = get_all_active_players()
    
    # Filter to only players needing updates
    if not force:
        players_to_fetch = [
            p for p in all_players 
            if not is_player_data_fresh(p['full_name'], season)
        ]
        players_with_fresh_data = len(all_players) - len(players_to_fetch)
        
        print()
        print(f"Cache Status:")
        print(f"  ✅ Fresh data: {players_with_fresh_data} players")
        print(f"  🔄 Need update: {len(players_to_fetch)} players")
    else:
        players_to_fetch = all_players
        print()
        print(f"Force mode: Rebuilding all {len(players_to_fetch)} players")
    
    if not players_to_fetch:
        print()
        print("✅ All player data is fresh! Loading from individual files...")
        df_all = load_all_player_game_logs(season)
        print(f"✅ Loaded {len(df_all):,} game logs for {df_all['player'].nunique()} players")
        return df_all
    
    print()
    print(f"Fetching game logs for {len(players_to_fetch)} players...")
    if MAX_WORKERS == 1:
        print(f"⏱️  Processing sequentially (0.5s delay per player)")
        print(f"⏱️  Estimated time: ~{len(players_to_fetch) * (REQUEST_DELAY + 1) / 60:.1f} minutes")
    else:
        print(f"⚡ Using {MAX_WORKERS} parallel workers")
        print(f"⏱️  Estimated time: ~{len(players_to_fetch) * 0.6 / MAX_WORKERS / 60:.1f} minutes")
    print()
    
    all_game_logs = []
    successful = 0
    failed = 0
    no_games = 0
    failed_players = []  # Track which players failed
    
    # Thread-safe lock for updating counters
    lock = Lock()
    
    start_time = time.time()
    
    def fetch_player_logs(player):
        """Helper function to fetch logs for one player (for parallel execution)"""
        player_id = player['id']
        player_name = player['full_name']
        
        # Add delay to avoid overwhelming API
        time.sleep(REQUEST_DELAY)
        
        # Log start
        print(f"   🔄 Fetching {player_name}...")
        
        try:
            df, had_error = get_player_game_logs(player_id, player_name, season)
            
            # Check for error first
            if had_error:
                return ('failed', None, player_name)
            
            # Check if player has games
            if df is not None and not df.empty:
                parsed_df = parse_game_logs(df)
                
                if not parsed_df.empty:
                    # Save individual player file
                    save_player_game_logs(parsed_df, player_name, season)
                    
                    num_games = len(parsed_df)
                    print(f"   ✅ {player_name}: {num_games} games (saved)")
                    return ('success', parsed_df, player_name)
                else:
                    # Save empty file to mark "no games" and avoid re-checking
                    empty_df = get_empty_gamelog_dataframe()
                    save_player_game_logs(empty_df, player_name, season, allow_empty=True)
                    print(f"   ⚪ {player_name}: No valid games (cached)")
                    return ('no_games', None, player_name)
            else:
                # Save empty file to mark "no games" and avoid re-checking
                # Next run, this empty file will be treated as "fresh" and player will be skipped
                empty_df = get_empty_gamelog_dataframe()
                save_player_game_logs(empty_df, player_name, season, allow_empty=True)
                print(f"   ⚪ {player_name}: No games found (cached)")
                return ('no_games', None, player_name)
        except Exception as e:
            print(f"   ❌ {player_name}: Error - {str(e)[:50]}")
            return ('failed', None, player_name)
    
    # Process players in parallel
    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        futures = {executor.submit(fetch_player_logs, player): player for player in players_to_fetch}
        
        # Process results as they complete
        for future in as_completed(futures):
            completed += 1
            
            try:
                result_type, df, player_name = future.result()
                
                with lock:
                    if result_type == 'success':
                        all_game_logs.append(df)
                        successful += 1
                    elif result_type == 'no_games':
                        no_games += 1
                    else:
                        failed += 1
                        failed_players.append(player_name)
                
                # Progress indicator every 25 players (more frequent for sequential)
                if completed % 25 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    remaining = (len(players_to_fetch) - completed) / rate if rate > 0 else 0
                    remaining_min = remaining / 60
                    print(f"[{completed}/{len(players_to_fetch)}] ✅ {successful} success | ❌ {failed} failed | ⚪ {no_games} no games (ETA: {remaining_min:.1f}m)")
            
            except Exception as e:
                with lock:
                    failed += 1
    
    print()
    print("="*80)
    print("BUILD COMPLETE")
    print("="*80)
    print()
    print(f"✅ Successful: {successful} players (saved with game data)")
    print(f"⚠️  No games: {no_games} players (empty files saved - will be skipped next run)")
    print(f"❌ Failed: {failed} players (will retry next run)")
    print()
    
    # Load ALL player files (fresh + newly fetched)
    print("Loading all player data from individual files...")
    df_all = load_all_player_game_logs(season)
    
    if df_all.empty:
        print("❌ No game logs available!")
        return pd.DataFrame()
    
    print(f"✅ Combined data:")
    print(f"   Total game logs: {len(df_all):,}")
    print(f"   Total players: {df_all['player'].nunique()}")
    print(f"   Date range: {df_all['date'].min().strftime('%Y-%m-%d')} to {df_all['date'].max().strftime('%Y-%m-%d')}")
    print()
    
    # Save combined file for convenience (optional, since individual files exist)
    combined_file = BASE_OUTPUT_DIR / f'combined_{season.replace("-", "_")}.csv'
    print(f"💾 Saving combined file: {combined_file}")
    df_all.to_csv(combined_file, index=False)
    print("✅ Saved!")
    
    # Check for failed and missing players
    failed_list, missing_list = check_missing_players(df_all, all_players, failed_players)
    log_failed_and_missing_players(failed_list, missing_list)
    
    # Show sample stats
    print("="*80)
    print("SAMPLE STATS")
    print("="*80)
    print()
    
    # Top 3PT shooters
    top_3pt = df_all.groupby('player')['threes_made'].sum().sort_values(ascending=False).head(10)
    print("Top 10 3PT shooters (total makes):")
    for player, threes in top_3pt.items():
        games = len(df_all[df_all['player'] == player])
        avg = threes / games
        print(f"  {player:<30} {threes:.0f} 3PM ({games} games, {avg:.1f} avg)")
    
    print()
    
    # Sample data
    print("Sample game logs:")
    print(df_all[['player', 'date', 'opponent', 'minutes', 'threes_made', 'threes_attempted', 'pts']].head(10))
    print()
    
    elapsed_time = time.time() - start_time
    print(f"⏱️  Total time: {elapsed_time/60:.1f} minutes")
    print()
    
    return df_all


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Build comprehensive NBA season game logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build current season cache (default)
  python %(prog)s
  
  # Force rebuild (ignore cache)
  python %(prog)s --force
  
  # Specific season
  python %(prog)s --season 2024-25
        """
    )
    
    parser.add_argument('--season', type=str, default=CURRENT_NBA_SEASON,
                       help=f'Season to fetch (default: {CURRENT_NBA_SEASON})')
    parser.add_argument('--force', action='store_true',
                       help='Force rebuild (ignore existing cache)')
    
    args = parser.parse_args()
    
    # Validate season format
    if not args.season.count('-') == 1:
        print(f"❌ Invalid season format: {args.season}")
        print("   Use format: YYYY-YY (e.g., 2025-26)")
        return
    
    # Build
    df = build_season_game_logs(season=args.season, force=args.force)
    
    if df.empty:
        print("❌ Failed to build game logs")
        return
    
    print()
    print("="*80)
    print("✅ SUCCESS!")
    print("="*80)
    print()
    print("Next steps:")
    print("  1. Run daily betting finder: python scripts/find_aligned_betting_opportunities.py")
    print("  2. Set up cron job for automatic daily updates")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Script interrupted by user (Ctrl+C)")
        print("✅ All data fetched so far has been saved to individual player files")
        print("   Run the script again to continue from where you left off")
        sys.exit(0)

