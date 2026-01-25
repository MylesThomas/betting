"""
Check Line Steam - Multi-Sport Steam Detection Script

Detects significant line movement (steam) for NBA, NFL, NCAAB, or NCAAF games.
Checks if any of TODAY'S games have threshold+ point consensus line movement
toward the opening underdog or favorite.

What it does:
1. Load hourly line movement snapshots from S3
2. Calculate consensus line movements (opening → current/latest)
3. Filter to games scheduled for TODAY (in ET timezone)
4. Check if any game has threshold+ movement (both directions)
5. Output "STEAM_DETECTED: YES" or "STEAM_DETECTED: NO" with game details

Usage:
    # Check NCAAB steam for today
    python scripts/check_line_steam.py --sport ncaab --date 2026-01-23
    
    # Check NFL steam with custom threshold
    python scripts/check_line_steam.py --sport nfl --date 2026-01-26 --threshold 1.5
    
    # Save detected plays to S3
    python scripts/check_line_steam.py --sport ncaab --date 2026-01-23 --save-plays
    
    # Show detailed breakdown of ALL today's games
    python scripts/check_line_steam.py --sport nba --date 2026-01-23 --log-individual-games

Output Format:
    If steam detected:
        STEAM_DETECTED: YES
        <game details>
    
    If no steam:
        STEAM_DETECTED: NO
        Checked X games - largest movement was Y points

Context:
User request - "ideally we have it work for nba/nfl/ncaaf/ncaab, each would send 
out their own email summaries at the hour if there is steam etc"

Built modular system where sport is a parameter instead of separate scripts.
All sport-specific config in config/line_steam_config.yaml.

Author: Thomas Myles  
Date: 2026-01-23
"""

import argparse
import sys
from pathlib import Path
from zoneinfo import ZoneInfo
from datetime import datetime

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

from line_steam_utils import (
    SportConfig,
    load_line_movement_snapshots,
    calculate_consensus_movements,
    check_for_steam,
    save_plays_to_s3
)
from season_utils import (
    get_current_nba_season,
    get_current_nfl_season,
    get_current_ncaab_season,
    get_current_ncaaf_season
)


def get_current_season(sport):
    """Get current season for a sport."""
    season_funcs = {
        'nba': get_current_nba_season,
        'nfl': get_current_nfl_season,
        'ncaab': get_current_ncaab_season,
        'ncaaf': get_current_ncaaf_season
    }
    
    if sport not in season_funcs:
        raise ValueError(f"Unknown sport: {sport}")
    
    return season_funcs[sport]()


def log_individual_games(movements_df, target_date_str, threshold, sport_name=""):
    """
    Log detailed breakdown of ALL today's games, sorted by start time (ET).
    
    Args:
        movements_df: DataFrame with consensus movements
        target_date_str: Date string in YYYY-MM-DD format (ET timezone)
        threshold: Steam threshold for highlighting
        sport_name: Sport name for display (e.g., "NCAAB")
    """
    et_tz = ZoneInfo('America/New_York')
    target_date = datetime.strptime(target_date_str, '%Y-%m-%d').date()
    
    # Filter to games scheduled for target date (in ET)
    movements_df['game_date_et'] = movements_df['game_time'].dt.tz_convert(et_tz).dt.date
    today_games = movements_df[movements_df['game_date_et'] == target_date].copy()
    
    if len(today_games) == 0:
        print(f"\n📋 No games scheduled for {target_date_str}")
        return
    
    # Sort by game start time (ET)
    today_games['game_time_et'] = today_games['game_time'].dt.tz_convert(et_tz)
    today_games = today_games.sort_values('game_time_et')
    
    sport_display = f"{sport_name} " if sport_name else ""
    print(f"\n{'='*80}")
    print(f"📋 ALL {sport_display}GAMES TODAY ({target_date_str}) - Sorted by Start Time")
    print(f"{'='*80}")
    
    for idx, (_, row) in enumerate(today_games.iterrows(), 1):
        game_time_et = row['game_time_et']
        hours_tracked = (row['current_time'] - row['open_time']).total_seconds() / 3600
        
        # Check if this game has significant steam
        has_steam = row['steam_magnitude'] >= threshold
        steam_indicator = " 🚨 STEAM" if has_steam else ""
        
        print(f"\n{'─'*80}")
        print(f"#{idx} | {game_time_et.strftime('%I:%M %p ET')}{steam_indicator}")
        print(f"{row['away_team']} @ {row['home_team']}")
        print(f"   Favorite: {row['opening_favorite']} (opening)")
        
        # Opening and current lines
        print(f"📊 Opening: {row['opening_favorite']} {row['opening_favorite_spread_open']:+.1f} | "
              f"{row['opening_underdog']} {-row['opening_favorite_spread_open']:+.1f}")
        print(f"📊 Current: {row['opening_favorite']} {row['opening_favorite_spread_current']:+.1f} | "
              f"{row['opening_underdog']} {-row['opening_favorite_spread_current']:+.1f}")
        
        # Movement direction and magnitude
        if row['steam_toward_opening_underdog']:
            direction_text = f"toward opening UNDERDOG ({row['opening_underdog']})"
        else:
            direction_text = f"toward opening FAVORITE ({row['opening_favorite']})"
        
        print(f"🔥 Movement: {row['steam_magnitude']:.1f} pts {direction_text}")
        
        # Tracking metadata
        open_time_et = row['open_time'].tz_convert(et_tz)
        current_time_et = row['current_time'].tz_convert(et_tz)
        print(f"📈 Tracked: {hours_tracked:.1f} hrs | First: {open_time_et.strftime('%m/%d %I:%M%p')} ET → "
              f"Latest: {current_time_et.strftime('%m/%d %I:%M%p')} ET")
    
    print(f"\n{'='*80}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Check line steam for NBA/NFL/NCAAB/NCAAF',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check NCAAB steam for today
  python scripts/check_line_steam.py --sport ncaab --date 2026-01-23
  
  # Check NFL with custom threshold
  python scripts/check_line_steam.py --sport nfl --date 2026-01-26 --threshold 1.5
  
  # Save plays to S3
  python scripts/check_line_steam.py --sport ncaab --date 2026-01-23 --save-plays
  
  # Show all games (not just steam)
  python scripts/check_line_steam.py --sport nba --date 2026-01-23 --log-individual-games
        """
    )
    parser.add_argument('--sport', required=True, 
                       choices=['nba', 'nfl', 'ncaab', 'ncaaf'],
                       help='Sport to check (nba, nfl, ncaab, ncaaf)')
    parser.add_argument('--date', required=True, 
                       help='Date to check (YYYY-MM-DD format, ET timezone)')
    parser.add_argument('--threshold', type=float, 
                       help='Steam threshold in points (overrides config default)')
    parser.add_argument('--days-back', type=int, 
                       help='Only load snapshots from last X days (overrides config default)')
    parser.add_argument('--log-individual-games', action='store_true', 
                       help='Log detailed breakdown of ALL games today')
    parser.add_argument('--save-plays', action='store_true',
                       help='Save detected steam plays to S3')
    parser.add_argument('--season', type=str, 
                       help='Season string (e.g., 2025-26) - auto-detected if not provided')
    args = parser.parse_args()
    
    # Validate args
    if args.save_plays and not args.season:
        # Auto-detect season
        args.season = get_current_season(args.sport)
        print(f"📅 Auto-detected season: {args.season}")
    
    try:
        # Load sport configuration
        sport_config = SportConfig(args.sport)
        print(f"\n{'='*80}")
        print(f"{sport_config.icon} {sport_config.name} LINE STEAM CHECK - {args.date}")
        print(f"{'='*80}")
        print(f"Sport: {sport_config.name}")
        print(f"Season: {args.season if args.season else 'N/A (not saving plays)'}")
        
        # Get threshold (CLI override > config default)
        threshold = args.threshold if args.threshold is not None else sport_config.threshold
        print(f"Threshold: {threshold} points")
        
        # Load snapshots
        days_back = args.days_back if args.days_back is not None else sport_config.days_back
        snapshots_df = load_line_movement_snapshots(sport_config, days_back=days_back)
        
        # Calculate movements
        movements_df = calculate_consensus_movements(snapshots_df)
        
        # Log all games if requested (BEFORE checking for steam)
        if args.log_individual_games:
            log_individual_games(movements_df, args.date, threshold, sport_config.name)
        
        # Check for steam
        steam_detected, steam_games = check_for_steam(
            movements_df, 
            args.date, 
            threshold,
            sport_config.name
        )
        
        # Save plays to S3 if requested and steam detected
        if args.save_plays:
            print(f"\n💾 --save-plays flag detected")
            print(f"   Steam detected: {steam_detected}")
            print(f"   Steam games: {steam_games is not None}")
            if steam_games is not None:
                print(f"   Number of steam games: {len(steam_games)}")
            
            if steam_detected and steam_games is not None:
                print(f"   Calling save_plays_to_s3...")
                try:
                    save_plays_to_s3(
                        steam_games, 
                        sport_config,
                        args.date, 
                        args.season, 
                        threshold
                    )
                except Exception as e:
                    print(f"   ❌ Save failed: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"   Skipping save (no steam or no games)")
        
        # Exit code: 0 = success (steam or no steam), 1 = error
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
