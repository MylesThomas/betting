"""
STEP 3: Inspect and manage game log cache.

The cache stores game logs fetched from NBA API, making subsequent builds 100x+ faster.
Use this script to inspect cache contents, view specific players, or clear cache.

Usage:
    # Show cache statistics
    python src/player_team_history/03_cache.py --stats
    
    # Inspect specific player
    python src/player_team_history/03_cache.py --player "Anthony Davis"
    
    # Clear specific players
    python src/player_team_history/03_cache.py --clear "Player 1" "Player 2"
    
    # Clear all cache
    python src/player_team_history/03_cache.py --clear-all

Cache Location:
    ~/Downloads/tmp/player_team_history/cache/

Cache Benefits:
    - Speeds up subsequent builds by 100x+
    - Reduces NBA API load
    - Allows quick iteration on name normalization fixes
"""

import sys
from pathlib import Path
import pandas as pd
import argparse
import duckdb

# Add src to path
repo_root = Path(__file__).resolve()
while not (repo_root / '.gitignore').exists():
    repo_root = repo_root.parent
sys.path.insert(0, str(repo_root))

from src.config import EMOJI

CACHE_DIR = Path.home() / 'Downloads' / 'tmp' / 'player_team_history' / 'cache'
OUTPUT_DIR = Path.home() / 'Downloads' / 'tmp' / 'player_team_history'
BOX_SCORES_FILE = OUTPUT_DIR / 'box_scores.parquet'


def get_cache_stats():
    """Get cache statistics."""
    if not CACHE_DIR.exists():
        return {'players': 0, 'seasons': 0, 'player_info': 0, 'size_mb': 0}
    
    player_dir = CACHE_DIR / 'players'
    season_dir = CACHE_DIR / 'seasons'
    player_info_dir = CACHE_DIR / 'player_info'
    
    player_files = list(player_dir.glob('*.parquet')) if player_dir.exists() else []
    season_files = list(season_dir.glob('*.parquet')) if season_dir.exists() else []
    player_info_files = list(player_info_dir.glob('*.parquet')) if player_info_dir.exists() else []
    
    total_size = sum(f.stat().st_size for f in player_files + season_files + player_info_files)
    total_size_mb = total_size / (1024 * 1024)
    
    return {
        'players': len(player_files),
        'seasons': len(season_files),
        'player_info': len(player_info_files),
        'size_mb': total_size_mb
    }


def get_box_score_stats():
    """Read box score output stats with DuckDB."""
    if not BOX_SCORES_FILE.exists():
        return None
    return duckdb.sql(
        f"""
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT player_normalized) AS players,
            MIN(GAME_DATE) AS min_game_date,
            MAX(GAME_DATE) AS max_game_date
        FROM read_parquet('{BOX_SCORES_FILE.as_posix()}')
        """
    ).df().iloc[0].to_dict()


def show_stats():
    """Show cache statistics."""
    print("="*80)
    print(f"{EMOJI['chart']} GAME LOG CACHE STATISTICS")
    print("="*80)
    print()
    
    if not CACHE_DIR.exists():
        print(f"{EMOJI['info']} No cache directory found")
        print(f"   Cache will be created at: {CACHE_DIR}")
        return
    
    stats = get_cache_stats()
    
    print(f"Cache location: {CACHE_DIR}")
    print(f"Complete players: {stats['players']}")
    print(f"Individual seasons: {stats['seasons']}")
    print(f"Player info cache: {stats['player_info']}")
    print(f"Total size: {stats['size_mb']:.1f} MB")
    print()

    box_stats = get_box_score_stats()
    if box_stats:
        print(f"{EMOJI['info']} Box score artifact:")
        print(f"   Path: {BOX_SCORES_FILE}")
        print(f"   Rows: {int(box_stats['rows'])}")
        print(f"   Players: {int(box_stats['players'])}")
        print(f"   Date range: {str(box_stats['min_game_date'])[:10]} to {str(box_stats['max_game_date'])[:10]}")
        print()
    
    player_dir = CACHE_DIR / 'players'
    if stats['players'] > 0 and player_dir.exists():
        print(f"{EMOJI['info']} Sample of complete players:")
        cache_files = sorted(player_dir.glob('*.parquet'))[:10]
        for cache_file in cache_files:
            name = cache_file.stem.replace('_', ' ')
            size_kb = cache_file.stat().st_size / 1024
            print(f"   - {name} ({size_kb:.1f} KB)")
        
        if stats['players'] > 10:
            print(f"   ... and {stats['players'] - 10} more")
    print()


def inspect_player(player_name):
    """Inspect cached game logs for a specific player."""
    safe_name = player_name.replace(' ', '_').replace("'", '').replace('.', '')
    player_dir = CACHE_DIR / 'players'
    cache_file = player_dir / f"{safe_name}.parquet"
    
    if not cache_file.exists():
        print(f"{EMOJI['warning']} {player_name} not found in player cache")
        
        # Check season cache
        season_dir = CACHE_DIR / 'seasons'
        season_files = list(season_dir.glob(f"{safe_name}_*.parquet")) if season_dir.exists() else []
        if season_files:
            print(f"   Found {len(season_files)} individual season caches")
            print(f"   Run 01_build.py to complete this player")
        else:
            print(f"   No cache found for this player")
        return
    
    try:
        game_logs = duckdb.sql(f"SELECT * FROM read_parquet('{cache_file.as_posix()}')").df()
        
        print("="*80)
        print(f"{EMOJI['success']} {player_name.upper()}")
        print("="*80)
        print()
        
        print(f"Total games cached: {len(game_logs)}")
        print()
        
        if 'GAME_DATE' in game_logs.columns:
            game_logs['GAME_DATE'] = pd.to_datetime(game_logs['GAME_DATE'])
            first_game = game_logs['GAME_DATE'].min()
            last_game = game_logs['GAME_DATE'].max()
            print(f"Date range: {first_game.date()} to {last_game.date()}")
            print()
        
        if 'TEAM' in game_logs.columns:
            # Create team history
            game_logs = game_logs.sort_values('GAME_DATE')
            game_logs['team_change'] = game_logs['TEAM'] != game_logs['TEAM'].shift()
            game_logs['team_stint'] = game_logs['team_change'].cumsum()
            
            history = []
            for stint_id, stint_games in game_logs.groupby('team_stint'):
                team = stint_games['TEAM'].iloc[0]
                if pd.notna(team):
                    first_game = stint_games['GAME_DATE'].min()
                    last_game = stint_games['GAME_DATE'].max()
                    is_last = stint_id == game_logs['team_stint'].max()
                    
                    history.append({
                        'team': team,
                        'valid_from': first_game.date(),
                        'valid_to': None if is_last else last_game.date()
                    })
            
            print(f"Team History ({len(history)} stints):")
            print()
            print("TEAM | VALID_FROM  | VALID_TO")
            print("-" * 40)
            for stint in history:
                valid_to = stint['valid_to'] if stint['valid_to'] else 'NULL'
                print(f"{stint['team']:4} | {stint['valid_from']} | {valid_to}")
            print()
        
    except Exception as e:
        print(f"{EMOJI['error']} Error reading cache: {e}")


def clear_specific_players(player_names):
    """Clear cache for specific players."""
    if not CACHE_DIR.exists():
        print(f"{EMOJI['info']} No cache directory found")
        return
    
    player_dir = CACHE_DIR / 'players'
    season_dir = CACHE_DIR / 'seasons'
    player_info_dir = CACHE_DIR / 'player_info'
    
    cleared = []
    not_found = []
    
    for player_name in player_names:
        safe_name = player_name.replace(' ', '_').replace("'", '').replace('.', '')
        
        # Clear player-level cache
        player_file = player_dir / f"{safe_name}.parquet"
        # Clear all season-level caches for this player
        season_files = list(season_dir.glob(f"{safe_name}_*.parquet")) if season_dir.exists() else []
        player_info_file = player_info_dir / f"{safe_name}.parquet"
        
        found_any = False
        
        if player_file.exists():
            player_file.unlink()
            found_any = True
        if player_info_file.exists():
            player_info_file.unlink()
            found_any = True
        
        for season_file in season_files:
            season_file.unlink()
            found_any = True
        
        if found_any:
            cleared.append(player_name)
        else:
            not_found.append(player_name)
    
    if cleared:
        print(f"{EMOJI['success']} Cleared cache for {len(cleared)} players:")
        for name in cleared:
            print(f"   - {name}")
    
    if not_found:
        print(f"\n{EMOJI['warning']} Not found in cache ({len(not_found)} players):")
        for name in not_found:
            print(f"   - {name}")


def clear_all_cache():
    """Clear all cached game logs."""
    if not CACHE_DIR.exists():
        print(f"{EMOJI['info']} No cache directory found")
        return 0
    
    player_dir = CACHE_DIR / 'players'
    season_dir = CACHE_DIR / 'seasons'
    player_info_dir = CACHE_DIR / 'player_info'
    
    player_files = list(player_dir.glob('*.parquet')) if player_dir.exists() else []
    season_files = list(season_dir.glob('*.parquet')) if season_dir.exists() else []
    player_info_files = list(player_info_dir.glob('*.parquet')) if player_info_dir.exists() else []
    
    all_files = player_files + season_files + player_info_files
    
    if not all_files:
        print(f"{EMOJI['info']} Cache is already empty")
        return 0
    
    count = 0
    for cache_file in all_files:
        try:
            cache_file.unlink()
            count += 1
        except Exception:
            pass
    
    print(
        f"{EMOJI['success']} Cleared {count} cached files "
        f"({len(player_files)} players, {len(season_files)} seasons, {len(player_info_files)} player_info)"
    )
    return count


def main():
    parser = argparse.ArgumentParser(
        description='Inspect and manage game log cache',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/player_team_history/03_cache.py --stats
  python src/player_team_history/03_cache.py --player "Anthony Davis"
  python src/player_team_history/03_cache.py --clear "Player 1" "Player 2"
  python src/player_team_history/03_cache.py --clear-all
        """
    )
    parser.add_argument('--stats', action='store_true',
                       help='Show cache statistics')
    parser.add_argument('--player', type=str,
                       help='Inspect specific player cache')
    parser.add_argument('--clear', nargs='+',
                       help='Clear cache for specific players')
    parser.add_argument('--clear-all', action='store_true',
                       help='Clear all cache')
    
    args = parser.parse_args()
    
    if args.stats:
        show_stats()
    elif args.player:
        inspect_player(args.player)
    elif args.clear:
        print(f"{EMOJI['refresh']} Clearing cache for specific players...")
        print()
        clear_specific_players(args.clear)
        print()
    elif args.clear_all:
        print(f"{EMOJI['refresh']} Clearing ALL game log cache...")
        print()
        stats = get_cache_stats()
        num_players = stats['players']
        size_mb = stats['size_mb']
        
        if num_players > 0:
            print(f"   About to delete {num_players} cached players ({size_mb:.1f} MB)")
            response = input(f"   Continue? [y/N]: ")
            
            if response.lower() == 'y':
                cleared = clear_all_cache()
                print()
            else:
                print(f"{EMOJI['info']} Cancelled")
        else:
            clear_all_cache()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
