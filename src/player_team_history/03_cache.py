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

# Add src to path
repo_root = Path(__file__).resolve()
while not (repo_root / '.gitignore').exists():
    repo_root = repo_root.parent
sys.path.insert(0, str(repo_root))

from src.config import EMOJI

CACHE_DIR = Path.home() / 'Downloads' / 'tmp' / 'player_team_history' / 'cache'


def get_cache_stats():
    """Get cache statistics."""
    if not CACHE_DIR.exists():
        return 0, 0
    
    cache_files = list(CACHE_DIR.glob('*.parquet'))
    
    total_size = sum(f.stat().st_size for f in cache_files)
    total_size_mb = total_size / (1024 * 1024)
    
    return len(cache_files), total_size_mb


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
    
    num_players, size_mb = get_cache_stats()
    
    print(f"Cache location: {CACHE_DIR}")
    print(f"Cached players: {num_players}")
    print(f"Total size: {size_mb:.1f} MB")
    print()
    
    if num_players > 0:
        print(f"{EMOJI['info']} Sample of cached players:")
        cache_files = sorted(CACHE_DIR.glob('*.parquet'))[:10]
        for cache_file in cache_files:
            name = cache_file.stem.replace('_', ' ')
            size_kb = cache_file.stat().st_size / 1024
            print(f"   - {name} ({size_kb:.1f} KB)")
        
        if num_players > 10:
            print(f"   ... and {num_players - 10} more")
    print()


def inspect_player(player_name):
    """Inspect cached game logs for a specific player."""
    safe_name = player_name.replace(' ', '_').replace("'", '').replace('.', '')
    cache_file = CACHE_DIR / f"{safe_name}.parquet"
    
    if not cache_file.exists():
        print(f"{EMOJI['warning']} {player_name} not found in cache")
        print(f"   Expected file: {cache_file}")
        return
    
    try:
        game_logs = pd.read_parquet(cache_file)
        
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
    
    cleared = []
    not_found = []
    
    for player_name in player_names:
        safe_name = player_name.replace(' ', '_').replace("'", '').replace('.', '')
        cache_file = CACHE_DIR / f"{safe_name}.parquet"
        
        if cache_file.exists():
            try:
                cache_file.unlink()
                cleared.append(player_name)
            except Exception as e:
                print(f"{EMOJI['error']} Failed to clear {player_name}: {e}")
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
    
    cache_files = list(CACHE_DIR.glob('*.parquet'))
    
    if not cache_files:
        print(f"{EMOJI['info']} Cache is already empty")
        return 0
    
    count = 0
    for cache_file in cache_files:
        try:
            cache_file.unlink()
            count += 1
        except Exception:
            pass
    
    print(f"{EMOJI['success']} Cleared {count} cached player game logs")
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
        num_players, size_mb = get_cache_stats()
        
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
