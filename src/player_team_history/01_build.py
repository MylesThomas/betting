"""
STEP 1: Build player team history from S3 betting data.

This is the MAIN script for building player team history.
Discovers players from S3, fetches their game logs from NBA API,
and creates team history with date ranges.

Usage:
    # Full build (all players from S3)
    python src/player_team_history/01_build.py
    
    # Test with sample
    python src/player_team_history/01_build.py --sample 100
    
    # Resume from checkpoint
    python src/player_team_history/01_build.py --resume
    
    # Force fresh fetch (bypass cache)
    python src/player_team_history/01_build.py --no-cache
    
    # Verbose logging
    python src/player_team_history/01_build.py --verbose

Output:
    ~/Downloads/tmp/player_team_history/
    ├── history.parquet         # THE OUTPUT - final team history
    ├── checkpoint.parquet      # For resuming builds
    ├── failures.txt            # Detailed failure report
    └── cache/                  # Game log cache (speeds up subsequent runs)
        ├── Anthony_Davis.parquet
        └── ...

Next Steps:
    1. Run this script
    2. Analyze failures: python src/player_team_history/02_analyze_failures.py
    3. Fix name mappings in name_normalization.py
    4. Re-run this script
"""

import pandas as pd
from pathlib import Path
import time
from datetime import datetime
import argparse
import ssl
import urllib3
import requests
import sys

# Fix SSL - must be done BEFORE importing nba_api
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Patch requests globally
original_request = requests.Session.request
def patched_request(self, *args, **kwargs):
    kwargs['verify'] = False
    return original_request(self, *args, **kwargs)
requests.Session.request = patched_request

# Also patch the default Session
import requests.sessions
original_init = requests.sessions.Session.__init__
def patched_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    self.verify = False
requests.sessions.Session.__init__ = patched_init

# Add src to path
repo_root = Path(__file__).resolve()
while not (repo_root / '.gitignore').exists():
    repo_root = repo_root.parent
sys.path.insert(0, str(repo_root))

from src.player_team_history.name_normalization import normalize_from_odds_api, normalize_from_nba_api
from src.player_team_history.team_normalization import normalize_team_code
from src.config import CURRENT_NBA_SEASON, EMOJI

try:
    from nba_api.stats.endpoints import playergamelog, commonplayerinfo
    from nba_api.stats.static import players
except ImportError:
    print(f"{EMOJI['error']} nba_api not found. Install with: pip install nba_api")
    sys.exit(1)

# Output directory
OUTPUT_DIR = Path.home() / 'Downloads' / 'tmp' / 'player_team_history'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Output files
CHECKPOINT_FILE = OUTPUT_DIR / 'checkpoint.parquet'
FINAL_OUTPUT = OUTPUT_DIR / 'history.parquet'
FAILURE_REPORT = OUTPUT_DIR / 'failures.txt'
CACHE_DIR = OUTPUT_DIR / 'cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Two-tier cache: seasons and players
SEASON_CACHE_DIR = CACHE_DIR / 'seasons'
PLAYER_CACHE_DIR = CACHE_DIR / 'players'
SEASON_CACHE_DIR.mkdir(parents=True, exist_ok=True)
PLAYER_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Rate limiting
RATE_LIMIT = 0.1  # seconds between API requests


# =============================================================================
# HELPER FUNCTIONS (in execution order)
# =============================================================================

def get_player_cache_filename(player_name):
    """Get cache filename for complete player data."""
    safe_name = player_name.replace(' ', '_').replace("'", '').replace('.', '')
    return PLAYER_CACHE_DIR / f"{safe_name}.parquet"


def get_season_cache_filename(player_name, season):
    """Get cache filename for a specific season."""
    safe_name = player_name.replace(' ', '_').replace("'", '').replace('.', '')
    return SEASON_CACHE_DIR / f"{safe_name}_{season}.parquet"


def load_season_from_cache(player_name, season):
    """Load a specific season from cache."""
    cache_file = get_season_cache_filename(player_name, season)
    
    if cache_file.exists():
        try:
            return pd.read_parquet(cache_file)
        except Exception:
            # Corrupted cache file - delete it
            cache_file.unlink()
            return None
    
    return None


def save_season_to_cache(player_name, season, game_logs_df):
    """Save a specific season to cache with TEAM column."""
    if game_logs_df.empty:
        return
    
    # Add TEAM column from MATCHUP
    game_logs_df = game_logs_df.copy()
    game_logs_df['TEAM'] = game_logs_df['MATCHUP'].apply(extract_team_from_matchup)
    
    cache_file = get_season_cache_filename(player_name, season)
    game_logs_df.to_parquet(cache_file, index=False)


def load_from_cache(player_name):
    """
    Load complete player game logs from player-level cache only.
    
    Returns:
        DataFrame of game logs, or None if not cached
    """
    player_cache = get_player_cache_filename(player_name)
    
    if player_cache.exists():
        try:
            return pd.read_parquet(player_cache)
        except Exception:
            # Corrupted cache file - delete it
            player_cache.unlink()
            return None
    
    return None


def save_to_cache(player_name, game_logs_df):
    """Save complete player game logs to player-level cache (only when all seasons successful)."""
    if game_logs_df.empty:
        return
    
    cache_file = get_player_cache_filename(player_name)
    game_logs_df.to_parquet(cache_file, index=False)


def discover_players_from_s3(sample_size=None):
    """Get unique players from S3 betting data."""
    from src.player_team_history.discovery import discover_all_players
    
    players_set = discover_all_players(s3_sample_size=sample_size, verbose=True)
    players = sorted(list(players_set))
    
    return players


def find_player_id(player_name):
    """
    Find player ID in NBA API.
    
    The input player_name is already normalized from Odds API.
    We normalize NBA API names and compare.
    """
    all_players = players.get_players()
    
    search_name_normalized = player_name
    
    # Try exact match on normalized names
    for p in all_players:
        nba_name_normalized = normalize_from_nba_api(p['full_name'])
        if nba_name_normalized == search_name_normalized:
            return p['id']
    
    # Try partial match
    for p in all_players:
        nba_name_normalized = normalize_from_nba_api(p['full_name'])
        if nba_name_normalized and search_name_normalized in nba_name_normalized:
            if p.get('is_active', False):
                return p['id']
    
    # Try reversed name
    parts = search_name_normalized.split()
    if len(parts) >= 2:
        reversed_name = f"{parts[-1]} {' '.join(parts[:-1])}"
        for p in all_players:
            nba_name_normalized = normalize_from_nba_api(p['full_name'])
            if nba_name_normalized and reversed_name in nba_name_normalized:
                return p['id']
    
    return None


def get_career_seasons(player_id):
    """Get all seasons for a player (only called when NOT using cache)."""
    try:
        player_info = commonplayerinfo.CommonPlayerInfo(
            player_id=player_id,
            timeout=5  # Fail fast - 5 second timeout
        )
        df = player_info.get_data_frames()[0]
        
        if df.empty:
            return [CURRENT_NBA_SEASON]
        
        from_year = int(df['FROM_YEAR'].iloc[0])
        to_year = int(df['TO_YEAR'].iloc[0])
        
        seasons = []
        for year in range(from_year, to_year + 1):
            season = f"{year}-{str(year + 1)[-2:]}"
            seasons.append(season)
        
        return seasons
    except Exception:
        return []


def extract_team_from_matchup(matchup):
    """Extract player's team from MATCHUP string and normalize to modern codes."""
    if pd.isna(matchup):
        return None
    
    team_code = None
    if '@' in matchup:
        team_code = matchup.split('@')[0].strip()
    elif 'vs.' in matchup:
        team_code = matchup.split('vs.')[0].strip()
    
    # Normalize historical team codes
    return normalize_team_code(team_code)


def fetch_player_game_log(player_name, player_id, verbose=False, use_cache=True):
    """
    Fetch game logs with smart two-tier caching:
    1. Check player-level cache (complete data)
    2. If not found, check which seasons we have vs need, fetch missing ones
    3. Once all seasons obtained, save player-level cache
    
    Returns:
        Tuple of (DataFrame with game logs including TEAM column, bool indicating if from cache)
    """
    # Check player-level cache first (complete data)
    if use_cache:
        cached_logs = load_from_cache(player_name)
        if cached_logs is not None:
            if verbose:
                print(f"      [COMPLETE PLAYER CACHE]")
            return cached_logs, True
    
    # Not in player cache - get expected seasons
    if verbose:
        print(f"      [BUILDING FROM SEASONS]", flush=True)
    
    seasons = get_career_seasons(player_id)
    
    if verbose:
        print(f"      Expected: {len(seasons)} seasons...")
    
    all_games = []
    failed_seasons = []
    cached_count = 0
    fetched_count = 0
    
    for season in seasons:
        # Check season cache first
        season_df = load_season_from_cache(player_name, season)
        
        if season_df is not None:
            all_games.append(season_df)
            cached_count += 1
            if verbose:
                print(f"      💾 {season}: {len(season_df)} games [from season cache]")
            continue
        
        # Fetch from API
        try:
            gamelog = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                timeout=5
            )
            
            df = gamelog.get_data_frames()[0]
            
            if not df.empty:
                # Save this season immediately
                save_season_to_cache(player_name, season, df)
                all_games.append(df)
                fetched_count += 1
                if verbose:
                    print(f"      ✓ {season}: {len(df)} games [fetched & saved]")
            
            time.sleep(RATE_LIMIT)
            
        except Exception as e:
            failed_seasons.append((season, str(e)[:40]))
            if verbose:
                print(f"      ✗ {season}: {str(e)[:40]} [FAILED - try again later]")
    
    if verbose:
        print(f"      Summary: {cached_count} cached, {fetched_count} fetched, {len(failed_seasons)} failed")
    
    if not all_games:
        return pd.DataFrame(), False
    
    # Combine all seasons
    combined = pd.concat(all_games, ignore_index=True)
    
    # Add TEAM column from MATCHUP
    combined['TEAM'] = combined['MATCHUP'].apply(extract_team_from_matchup)
    
    # Only save player file if we got ALL seasons (no failures)
    if not failed_seasons:
        save_to_cache(player_name, combined)
        if verbose:
            print(f"      ✅ Saved complete player cache ({len(seasons)} seasons)")
    elif verbose:
        print(f"      ⚠️ NOT saving player cache - missing {len(failed_seasons)} seasons")
    
    return combined, False


def create_team_history_from_gamelogs(game_logs_df, player_name):
    """
    Convert game logs to team history with date ranges.
    
    Returns:
        DataFrame with columns: player_normalized, team, valid_from, valid_to
    """
    if game_logs_df.empty:
        return pd.DataFrame(columns=['player_normalized', 'team', 'valid_from', 'valid_to'])
    
    player_normalized = player_name
    
    # Convert dates
    game_logs_df['GAME_DATE'] = pd.to_datetime(game_logs_df['GAME_DATE'], format='mixed')
    game_logs_df = game_logs_df.sort_values('GAME_DATE')
    
    # Group consecutive games by team
    game_logs_df['team_change'] = game_logs_df['TEAM'] != game_logs_df['TEAM'].shift()
    game_logs_df['team_stint'] = game_logs_df['team_change'].cumsum()
    
    history = []
    
    for stint_id, stint_games in game_logs_df.groupby('team_stint'):
        team = stint_games['TEAM'].iloc[0]
        
        if pd.isna(team):
            continue
        
        # Normalize historical team codes to modern abbreviations
        team = normalize_team_code(team)
        
        first_game = stint_games['GAME_DATE'].min()
        last_game = stint_games['GAME_DATE'].max()
        
        valid_from = first_game.date()
        is_last_stint = stint_id == game_logs_df['team_stint'].max()
        valid_to = None if is_last_stint else last_game.date()
        
        history.append({
            'player_normalized': player_normalized,
            'team': team,
            'valid_from': valid_from,
            'valid_to': valid_to
        })
    
    return pd.DataFrame(history)


def load_checkpoint():
    """Load existing checkpoint if exists."""
    if not CHECKPOINT_FILE.exists():
        return pd.DataFrame(columns=['player_normalized', 'team', 'valid_from', 'valid_to'])
    
    df = pd.read_parquet(CHECKPOINT_FILE)
    print(f"{EMOJI['info']} Loaded checkpoint: {len(df)} records, {df['player_normalized'].nunique()} players")
    return df


def get_completed_players(checkpoint_df):
    """Get set of players already processed."""
    if checkpoint_df.empty:
        return set()
    return set(checkpoint_df['player_normalized'].unique())


def save_checkpoint(history_df):
    """Save current progress."""
    history_df.to_parquet(CHECKPOINT_FILE, index=False)


def generate_failure_report(not_found_in_nba, no_game_logs, no_history_created, processing_errors, total_processed, successful):
    """Generate detailed failure report."""
    failed = len(not_found_in_nba) + len(no_game_logs) + len(no_history_created) + len(processing_errors)
    
    with open(FAILURE_REPORT, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PLAYER TEAM HISTORY BUILD - FAILURE REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total players processed: {total_processed}\n")
        f.write(f"Successful: {successful}\n")
        f.write(f"Failed: {failed}\n\n")
        
        if not_found_in_nba:
            f.write("="*80 + "\n")
            f.write(f"NOT FOUND IN NBA API ({len(not_found_in_nba)} players)\n")
            f.write("="*80 + "\n")
            f.write("These players exist in Odds API but could not be matched to NBA API.\n")
            f.write("Action: Add name mappings to name_normalization.py\n\n")
            for player in sorted(not_found_in_nba):
                f.write(f"  - {player}\n")
            f.write("\n")
        
        if no_game_logs:
            f.write("="*80 + "\n")
            f.write(f"NO GAME LOGS ({len(no_game_logs)} players)\n")
            f.write("="*80 + "\n")
            f.write("Found in NBA API but no game logs available.\n")
            f.write("Possible reasons: rookies not yet played, retired players, etc.\n\n")
            for player in sorted(no_game_logs):
                f.write(f"  - {player}\n")
            f.write("\n")
        
        if no_history_created:
            f.write("="*80 + "\n")
            f.write(f"NO HISTORY CREATED ({len(no_history_created)} players)\n")
            f.write("="*80 + "\n")
            f.write("Game logs fetched but team history could not be created.\n")
            f.write("Possible reasons: MATCHUP parsing issues, all games have null team.\n\n")
            for player in sorted(no_history_created):
                f.write(f"  - {player}\n")
            f.write("\n")
        
        if processing_errors:
            f.write("="*80 + "\n")
            f.write(f"PROCESSING ERRORS ({len(processing_errors)} players)\n")
            f.write("="*80 + "\n")
            f.write("Unexpected errors during processing.\n\n")
            for player, error in sorted(processing_errors):
                f.write(f"  - {player}\n")
                f.write(f"    Error: {error}\n\n")


# =============================================================================
# MAIN BUILD FUNCTION
# =============================================================================

def build(resume=False, sample_size=None, verbose=False, use_cache=True):
    """
    Build team history incrementally with checkpoints and caching.
    
    Args:
        resume: Resume from checkpoint
        sample_size: Number of S3 files to sample (None = all)
        verbose: Show detailed logging
        use_cache: Use cached game logs if available
    """
    print("="*80)
    print(f"{EMOJI['nba']} BUILD PLAYER TEAM HISTORY")
    print("="*80)
    print()
    
    if use_cache:
        player_cache_count = len(list(PLAYER_CACHE_DIR.glob('*.parquet')))
        season_cache_count = len(list(SEASON_CACHE_DIR.glob('*.parquet')))
        if player_cache_count > 0 or season_cache_count > 0:
            print(f"{EMOJI['info']} Cache: {player_cache_count} complete players, {season_cache_count} individual seasons")
            print()
    
    # Load checkpoint if resuming
    checkpoint_df = load_checkpoint() if resume else pd.DataFrame()
    completed_players = get_completed_players(checkpoint_df)
    
    if resume and completed_players:
        print(f"{EMOJI['success']} Resuming from checkpoint: {len(completed_players)} players already done\n")
    
    # Discover players
    all_players = discover_players_from_s3(sample_size=sample_size)
    
    # Filter out completed
    if completed_players:
        players_to_process = [p for p in all_players if p not in completed_players]
    else:
        players_to_process = all_players
    
    print(f"\n{EMOJI['info']} Players to process: {len(players_to_process)}")
    print(f"{EMOJI['info']} Already completed: {len(completed_players)}")
    print()
    
    if not players_to_process:
        print(f"{EMOJI['success']} All players already processed!")
        return checkpoint_df
    
    # Process each player
    new_history = []
    successful = 0
    
    # Track failures
    not_found_in_nba = []
    no_game_logs = []
    no_history_created = []
    processing_errors = []
    
    # Track timing
    start_time = time.time()
    
    for i, player_name in enumerate(players_to_process, 1):
        player_start = time.time()
        print(f"[{i}/{len(players_to_process)}] {player_name}...", end=' ', flush=True)
        
        try:
            # Find player ID
            player_id = find_player_id(player_name)
            
            if not player_id:
                print(f"{EMOJI['warning']} Not found in NBA API")
                not_found_in_nba.append(player_name)
                continue
            
            # Fetch game logs
            game_logs, from_cache = fetch_player_game_log(
                player_name, 
                player_id, 
                verbose=verbose,
                use_cache=use_cache
            )
            
            if game_logs.empty:
                print(f"{EMOJI['warning']} No game logs")
                no_game_logs.append(player_name)
                continue
            
            # Create team history
            player_history = create_team_history_from_gamelogs(game_logs, player_name)
            
            t_total = time.time() - player_start
            
            if not player_history.empty:
                new_history.append(player_history)
                teams = ', '.join(player_history['team'].unique())
                
                # Show if from cache or API
                cache_indicator = "💾 CACHED" if from_cache else "🔄 API"
                print(f"{cache_indicator} {EMOJI['success']} {len(player_history)} stints [{teams}] ({t_total:.1f}s)")
                successful += 1
            else:
                print(f"{EMOJI['warning']} No history created ({t_total:.1f}s)")
                no_history_created.append(player_name)
        
        except Exception as e:
            t_total = time.time() - player_start
            error_msg = str(e)[:60]
            print(f"{EMOJI['error']} {error_msg} ({t_total:.1f}s)")
            processing_errors.append((player_name, str(e)))
            continue
        
        # Save checkpoint every 25 players
        if i % 25 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed
            remaining = len(players_to_process) - i
            eta_seconds = remaining / rate if rate > 0 else 0
            eta_minutes = eta_seconds / 60
            
            if new_history:
                new_df = pd.concat(new_history, ignore_index=True)
                combined = pd.concat([checkpoint_df, new_df], ignore_index=True)
                save_checkpoint(combined)
                print(f"\n{EMOJI['save']} Checkpoint saved: {successful + len(completed_players)} total players")
                print(f"   Progress: {i}/{len(players_to_process)} ({i/len(players_to_process)*100:.1f}%)")
                print(f"   Speed: {rate:.1f} players/sec")
                print(f"   ETA: {eta_minutes:.1f} min\n")
    
    # Final save
    if new_history:
        new_df = pd.concat(new_history, ignore_index=True)
        final_df = pd.concat([checkpoint_df, new_df], ignore_index=True)
        
        # Sort and dedupe
        final_df = final_df.sort_values(['player_normalized', 'valid_from'])
        final_df = final_df.drop_duplicates(subset=['player_normalized', 'team', 'valid_from'], keep='last')
        
        # Save final
        save_checkpoint(final_df)
        final_df.to_parquet(FINAL_OUTPUT, index=False)
        
        print()
        print("="*80)
        print(f"{EMOJI['success']} BUILD COMPLETE")
        print("="*80)
        print(f"Total players: {final_df['player_normalized'].nunique()}")
        print(f"Total stints: {len(final_df)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(not_found_in_nba) + len(no_game_logs) + len(no_history_created) + len(processing_errors)}")
        print()
        print(f"Output: {FINAL_OUTPUT}")
        print(f"Checkpoint: {CHECKPOINT_FILE}")
        print()
        
        # Generate failure report
        if any([not_found_in_nba, no_game_logs, no_history_created, processing_errors]):
            generate_failure_report(
                not_found_in_nba, no_game_logs, no_history_created, 
                processing_errors, len(players_to_process), successful
            )
            print(f"{EMOJI['warning']} Failure report: {FAILURE_REPORT}")
            print(f"   Analyze: python src/player_team_history/02_analyze_failures.py")
            print()
        
        # Display sample
        print("="*80)
        print(f"{EMOJI['chart']} SAMPLE - Top 10 Players by Stints")
        print("="*80)
        print()
        
        display_df = final_df.copy()
        display_df['valid_to'] = display_df['valid_to'].fillna('NULL')
        
        stint_counts = display_df.groupby('player_normalized').size().sort_values(ascending=False)
        top_players = stint_counts.head(10).index.tolist()
        
        for player in top_players:
            player_df = display_df[display_df['player_normalized'] == player].copy()
            player_df = player_df.sort_values('valid_from')
            
            print(f"{player} ({len(player_df)} stints):")
            for _, row in player_df.iterrows():
                print(f"  {row['team']:3} | {row['valid_from']} to {row['valid_to']}")
            print()
        
        print("="*80)
        print()
        
        return final_df
    else:
        return checkpoint_df


def main():
    parser = argparse.ArgumentParser(
        description='Build player team history from S3 betting data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/player_team_history/01_build.py
  python src/player_team_history/01_build.py --sample 100
  python src/player_team_history/01_build.py --resume
        """
    )
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint')
    parser.add_argument('--sample', type=int,
                       help='Only process sample of S3 files (for testing)')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed logging for each player')
    parser.add_argument('--no-cache', action='store_true',
                       help='Bypass cache and fetch fresh from NBA API')
    
    args = parser.parse_args()
    
    build(
        resume=args.resume, 
        sample_size=args.sample, 
        verbose=args.verbose,
        use_cache=not args.no_cache
    )


if __name__ == '__main__':
    main()
