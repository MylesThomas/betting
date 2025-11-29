"""
Fetch and Build Season Props - One-Stop Shop for Historical Props

This script handles the complete workflow:
1. Check what props exist for a season/market
2. Fetch missing props from The Odds API
3. Combine all props into consensus dataset
4. Output ready-to-use file for analysis

USAGE:
    # Fetch player_threes props for 2025-26 season (default market)
    python scripts/fetch_and_build_season_props.py --season 2025-26
    
    # Or explicitly specify the market
    python scripts/fetch_and_build_season_props.py --season 2025-26 --market player_threes
    
    # Search for any files containing "threes" (works with multi-market files)
    python scripts/fetch_and_build_season_props.py --season 2025-26 --market threes
    
    # Just combine existing files (no fetch)
    python scripts/fetch_and_build_season_props.py --season 2025-26 --combine-only
    
    # Force refetch all dates (ignore cache)
    python scripts/fetch_and_build_season_props.py --season 2025-26 --force

WORKFLOW:
    1. Load season calendar (which dates have games)
    2. Check which dates already have prop files
    3. Fetch missing dates from The Odds API
    4. Combine all individual files into consensus dataset
    5. Output: data/01_input/the-odds-api/historical_props/consensus_props_{market}.csv

OUTPUT FILES:
    Individual:  data/01_input/the-odds-api/historical_props/props_YYYY-MM-DD_{market}.csv
    Combined:    data/03_intermediate/combined_props_{season}_{market}.csv
    Consensus:   data/03_intermediate/consensus_props_{season}_{market}.csv

The consensus file is what you use for analysis - it has:
    - One row per player per game
    - Consensus line (penalty-based methodology)
    - Best/avg/worst odds for over and under
    - Arbitrage detection
    - Named by season and market for easy identification

Author: Myles Thomas
Date: 2024-11-24
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
from zoneinfo import ZoneInfo

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import CURRENT_NBA_SEASON

# Import from existing scripts
from fetch_historical_props import (
    fetch_date_props,
    setup_logging,
    API_KEY,
    credits_remaining,
    credits_used
)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Data paths
PROJECT_ROOT = Path(__file__).parent.parent
HISTORICAL_PROPS_DIR = PROJECT_ROOT / 'data' / '01_input' / 'the-odds-api' / 'nba' / 'historical_props'
INTERMEDIATE_DIR = PROJECT_ROOT / 'data' / '03_intermediate'
NBA_CALENDAR_DIR = PROJECT_ROOT / 'data' / '01_input' / 'nba_calendar'

# Defaults
DEFAULT_MARKET = 'player_threes'
ALL_MARKETS = 'player_points,player_rebounds,player_assists,player_threes,player_blocks,player_steals,player_double_double,player_triple_double,player_points_rebounds_assists'
CURRENT_NBA_SEASON = '2025-26'

# ============================================================================
# SEASON CALENDAR
# ============================================================================

def get_season_game_dates(season=CURRENT_NBA_SEASON):
    """
    Get all dates with games for a specific season.
    
    Scans nba_calendar directory for files matching the season.
    Only returns dates up to TODAY (can't fetch future props!)
    
    Args:
        season: Season string (e.g., '2025-26')
    
    Returns:
        Set of date objects for dates with games (up to today)
    """
    # Convert season format: 2025-26 -> 2025_26
    season_underscore = season.replace('-', '_')
    
    # Actually LOOK for files in the directory
    if not NBA_CALENDAR_DIR.exists():
        logging.error(f"Calendar directory doesn't exist: {NBA_CALENDAR_DIR}")
        return set()
    
    # Find ANY file with the season in it
    all_files = list(NBA_CALENDAR_DIR.glob(f'*{season_underscore}*'))
    
    if not all_files:
        logging.warning(f"No calendar files found for season {season}")
        logging.warning(f"Searched in: {NBA_CALENDAR_DIR}")
        logging.warning(f"Available files:")
        for f in NBA_CALENDAR_DIR.glob('*'):
            logging.warning(f"  - {f.name}")
        logging.warning("Falling back to date range estimation...")
        
        # Estimate based on typical NBA season
        start_year = int(season.split('-')[0])
        start_date = datetime(start_year, 10, 15).date()
        today = datetime.now().date()
        end_date = min(today, datetime(start_year + 1, 4, 15).date())
        
        game_dates = set()
        current = start_date
        while current <= end_date:
            game_dates.add(current)
            current += timedelta(days=1)
        
        logging.info(f"Estimated {len(game_dates)} possible game dates (up to today)")
        return game_dates
    
    # Use the first matching file found
    calendar_file = all_files[0]
    logging.info(f"Found calendar file: {calendar_file.name}")
    
    # Load based on file type
    game_dates = set()
    
    if calendar_file.suffix == '.json':
        import json
        with open(calendar_file) as f:
            data = json.load(f)
            # Try different key names
            date_strs = data.get('dates', data.get('game_dates', data.get('dates_list', [])))
            game_dates = set(datetime.strptime(d, '%Y-%m-%d').date() for d in date_strs)
    
    elif calendar_file.suffix == '.csv':
        df = pd.read_csv(calendar_file)
        # Try different column names
        date_col = None
        for col in ['GAME_DATE', 'game_date', 'date', 'Date']:
            if col in df.columns:
                date_col = col
                break
        
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col]).dt.date
            game_dates = set(df[date_col].unique())
        else:
            logging.error(f"Couldn't find date column in {calendar_file.name}")
            logging.error(f"Columns: {df.columns.tolist()}")
            return set()
    
    else:
        logging.error(f"Unknown file format: {calendar_file.suffix}")
        return set()
    
    # Filter to only dates up to today (can't fetch future props!)
    today = datetime.now().date()
    game_dates = {d for d in game_dates if d <= today}
    
    if len(game_dates) == 0:
        logging.error("No game dates found in calendar file!")
        return set()
    
    logging.info(f"Found {len(game_dates)} game dates for {season} (up to today)")
    logging.info(f"  First game: {min(game_dates)}")
    logging.info(f"  Last game: {max(game_dates)}")
    logging.info(f"  Today: {today}")
    
    return game_dates


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def date_to_season(date_obj):
    """
    Convert a date to its NBA season string.
    
    NBA seasons run October (year Y) through June (year Y+1):
    - Oct-Dec 2025 → 2025-26 season
    - Jan-Jun 2026 → 2025-26 season
    
    Args:
        date_obj: datetime.date object
    
    Returns:
        Season string (e.g., '2025-26')
    """
    year = date_obj.year
    month = date_obj.month
    
    # October onwards = start of season
    if month >= 10:
        return f"{year}-{str(year + 1)[-2:]}"
    # Jan-July = end of previous season (started in prior year)
    else:
        return f"{year - 1}-{str(year)[-2:]}"


# ============================================================================
# FILE CHECKING
# ============================================================================

def check_existing_props(season, market):
    """
    Check which dates already have prop files.
    
    Looks for files matching: props_YYYY-MM-DD_*{market}*.csv
    Examples:
        - props_2024-10-22_player_threes.csv
        - props_2024-10-23_player_points_player_rebounds_player_threes.csv
    
    Args:
        season: Season string (e.g., '2025-26')
        market: Market name to search for (e.g., 'threes', 'player_threes')
    
    Returns:
        Dict with:
            - existing_dates: Set of dates with files
            - existing_files: List of file paths
            - missing_dates: Set of dates without files
    """
    # Get game dates for season
    game_dates = get_season_game_dates(season)
    
    # Check which files exist - match any file with market in name
    # Pattern: props_*{market}*.csv
    pattern = f"props_*{market}*.csv"
    existing_files = list(HISTORICAL_PROPS_DIR.glob(pattern))
    
    logging.info(f"Looking for files matching pattern: {pattern}")
    logging.info(f"Found {len(existing_files)} files")
    
    # Extract dates from filenames and filter to only this season's files
    existing_dates = set()
    season_files = []  # Only files from this season
    
    for file in existing_files:
        # Filename: props_YYYY-MM-DD_{markets}.csv
        try:
            # Split by underscore and find the date part (YYYY-MM-DD)
            parts = file.stem.split('_')
            # Date is the part with dashes in YYYY-MM-DD format
            date_str = None
            for part in parts:
                if '-' in part and len(part) == 10:  # YYYY-MM-DD is 10 chars
                    date_str = part
                    break
            
            if date_str:
                date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
                file_season = date_to_season(date_obj)
                
                # Only include if file belongs to this season
                if file_season == season:
                    if date_obj in game_dates:
                        existing_dates.add(date_obj)
                        season_files.append(file)
        except (IndexError, ValueError) as e:
            logging.debug(f"Couldn't parse date from {file.name}: {e}")
            continue
    
    # Determine missing dates
    missing_dates = game_dates - existing_dates
    
    logging.info(f"Existing props: {len(existing_dates)}/{len(game_dates)} dates")
    logging.info(f"Missing props: {len(missing_dates)} dates")
    
    if len(existing_dates) > 0:
        # Show sample files found (from this season only)
        sample_files = [f.name for f in sorted(season_files)[:3]]
        logging.info(f"Sample files: {', '.join(sample_files)}")
    
    return {
        'game_dates': game_dates,
        'existing_dates': existing_dates,
        'existing_files': season_files,  # Only files from this season
        'missing_dates': missing_dates
    }


# ============================================================================
# FETCHING
# ============================================================================

def fetch_missing_props(missing_dates, market, force=False):
    """
    Fetch props for missing dates.
    
    Args:
        missing_dates: Set of date objects to fetch
        market: Market name
        force: Force refetch even if user cancels
    
    Returns:
        Stats dict with fetch results
    """
    if not missing_dates:
        logging.info("✅ No missing dates - all props already fetched!")
        return {'fetched': 0, 'skipped': 0, 'failed': 0}
    
    # Sort dates chronologically
    sorted_dates = sorted(missing_dates)
    
    logging.info("="*80)
    logging.info(f"FETCHING {len(sorted_dates)} MISSING DATES")
    logging.info("="*80)
    logging.info(f"Market: {market}")
    logging.info(f"Date range: {sorted_dates[0]} to {sorted_dates[-1]}")
    logging.info(f"Estimated cost: ~{len(sorted_dates) * 10} credits")
    logging.info("="*80)
    
    if not force:
        response = input(f"\nProceed with fetching {len(sorted_dates)} dates? (y/n): ")
        if response.lower() != 'y':
            logging.warning("User cancelled fetch")
            return {'fetched': 0, 'skipped': len(sorted_dates), 'failed': 0}
    
    logging.info("Starting fetch...")
    
    stats = {
        'fetched': 0,
        'skipped': 0,
        'failed': 0,
        'failed_dates': []
    }
    
    for i, date_obj in enumerate(sorted_dates, 1):
        date_str = date_obj.strftime('%Y-%m-%d')
        
        logging.info(f"\n[{i}/{len(sorted_dates)}] Fetching {date_str}...")
        
        try:
            df = fetch_date_props(date_str, markets=market, save=True)
            
            if df.empty:
                logging.warning(f"No data returned for {date_str}")
                stats['failed'] += 1
                stats['failed_dates'].append(date_str)
            else:
                logging.info(f"✅ Successfully fetched {len(df)} props")
                stats['fetched'] += 1
        
        except KeyboardInterrupt:
            logging.warning("\n⚠️  User interrupted (Ctrl+C)")
            break
        
        except Exception as e:
            logging.error(f"❌ Error fetching {date_str}: {e}")
            stats['failed'] += 1
            stats['failed_dates'].append(date_str)
    
    logging.info("="*80)
    logging.info("FETCH SUMMARY")
    logging.info("="*80)
    logging.info(f"Fetched: {stats['fetched']}")
    logging.info(f"Failed: {stats['failed']}")
    if stats['failed_dates']:
        logging.info(f"Failed dates: {', '.join(stats['failed_dates'])}")
    logging.info("="*80)
    
    return stats


# ============================================================================
# COMBINING
# ============================================================================

def combine_props_files(season, market):
    """
    Combine all individual prop files into one big file.
    
    Saves to: data/03_intermediate/combined_props_{season}_{market}.csv
    
    Args:
        season: Season string (e.g., '2025-26')
        market: Market name (e.g., 'player_threes')
    
    Returns:
        Combined DataFrame
    """
    logging.info("="*80)
    logging.info("COMBINING INDIVIDUAL PROP FILES")
    logging.info("="*80)
    
    # Check existing files
    check_result = check_existing_props(season, market)
    existing_files = check_result['existing_files']
    
    if not existing_files:
        logging.error("No prop files found to combine!")
        return pd.DataFrame()
    
    logging.info(f"Found {len(existing_files)} files to combine")
    
    # Load and combine
    all_dfs = []
    for file in sorted(existing_files):
        try:
            df = pd.read_csv(file)
            all_dfs.append(df)
        except Exception as e:
            logging.error(f"Error reading {file.name}: {e}")
    
    if not all_dfs:
        logging.error("No data loaded from files")
        return pd.DataFrame()
    
    combined = pd.concat(all_dfs, ignore_index=True)
    
    logging.info(f"✅ Combined {len(combined):,} total props")
    logging.info(f"   Unique players: {combined['player'].nunique()}")
    logging.info(f"   Date range: {combined['game_time'].min()} to {combined['game_time'].max()}")
    
    # Save combined file to intermediate with season in name
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    season_clean = season.replace('-', '_')
    combined_file = INTERMEDIATE_DIR / f'combined_props_{season_clean}_{market}.csv'
    combined.to_csv(combined_file, index=False)
    logging.info(f"💾 Saved: {combined_file}")
    
    return combined


def build_consensus_props(combined_df, season, market):
    """
    Build consensus props from combined data.
    
    Uses penalty-based methodology to find consensus lines.
    Saves to: data/03_intermediate/consensus_props_{season}_{market}.csv
    
    Args:
        combined_df: Combined props DataFrame
        season: Season string (e.g., '2025-26')
        market: Market name (e.g., 'player_threes')
    
    Returns:
        Consensus DataFrame
    """
    logging.info("="*80)
    logging.info("BUILDING CONSENSUS PROPS")
    logging.info("="*80)
    
    if combined_df.empty:
        logging.error("No data to build consensus from")
        return pd.DataFrame()
    
    # Import consensus logic (reusing from build_consensus_props.py)
    from build_consensus_props import (
        find_consensus_line_for_player,
        # calculate_vig
    )
    
    # Convert game_time to date
    combined_df['date'] = pd.to_datetime(combined_df['game_time']).dt.date
    
    # Group by player per game
    consensus_rows = []
    
    grouped = combined_df.groupby(['date', 'player', 'game'])
    total_groups = len(grouped)
    
    logging.info(f"Processing {total_groups} player-game combinations...")
    
    for i, ((date, player, game), player_df) in enumerate(grouped, 1):
        if i % 100 == 0:
            logging.info(f"  Processed {i}/{total_groups} ({i/total_groups*100:.1f}%)")
        
        try:
            consensus_data = find_consensus_line_for_player(player_df)
            consensus_data['date'] = date
            consensus_data['player'] = player
            consensus_data['game'] = game
            consensus_rows.append(consensus_data)
        except Exception as e:
            logging.error(f"Error processing {player} on {date}: {e}")
    
    if not consensus_rows:
        logging.error("No consensus data generated")
        return pd.DataFrame()
    
    consensus_df = pd.DataFrame(consensus_rows)
    
    # Sort by date and player
    consensus_df = consensus_df.sort_values(['date', 'player']).reset_index(drop=True)
    
    logging.info(f"✅ Built consensus for {len(consensus_df):,} player-games")
    
    # Save consensus file to intermediate with season in name
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    season_clean = season.replace('-', '_')
    consensus_file = INTERMEDIATE_DIR / f'consensus_props_{season_clean}_{market}.csv'
    consensus_df.to_csv(consensus_file, index=False)
    logging.info(f"💾 Saved: {consensus_file}")
    
    return consensus_df


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Fetch and build season props for analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch current season props
  python scripts/fetch_and_build_season_props.py
  
  # Fetch specific season
  python scripts/fetch_and_build_season_props.py --season 2025-26
  
  # Just combine existing files (no fetch)
  python scripts/fetch_and_build_season_props.py --combine-only
  
  # Force refetch all dates
  python scripts/fetch_and_build_season_props.py --force
        """
    )
    
    parser.add_argument('--season', type=str, default=CURRENT_NBA_SEASON,
                       help=f'Season to fetch (default: {CURRENT_NBA_SEASON})')
    parser.add_argument('--market', type=str, default=DEFAULT_MARKET,
                       help=f'Market to search/fetch - looks for files with this in name (default: {DEFAULT_MARKET}). '
                            'Examples: "threes", "player_threes", "points"')
    parser.add_argument('--combine-only', action='store_true',
                       help='Only combine existing files, no fetch')
    parser.add_argument('--force', action='store_true',
                       help='Force refetch all dates (ignore cache)')
    parser.add_argument('--skip-consensus', action='store_true',
                       help='Skip building consensus (just combine)')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(f'fetch_season_props_{args.season.replace("-", "_")}')
    
    logging.info("="*80)
    logging.info("FETCH AND BUILD SEASON PROPS")
    logging.info("="*80)
    logging.info(f"Season: {args.season}")
    logging.info(f"Market: {args.market}")
    logging.info(f"Mode: {'Combine only' if args.combine_only else 'Fetch + Combine'}")
    logging.info("="*80)
    print()
    
    # Check API key
    if not args.combine_only:
        if not API_KEY or API_KEY == 'your_api_key_here':
            logging.error("API key not configured!")
            logging.error("Add ODDS_API_KEY to .env file")
            return
    
    # STEP 1: Check existing props
    check_result = check_existing_props(args.season, args.market)
    
    # STEP 2: Fetch missing props (unless combine-only)
    if not args.combine_only:
        if check_result['missing_dates'] or args.force:
            if args.force:
                # Force mode: fetch ALL dates
                fetch_dates = check_result['game_dates']
                logging.info("Force mode: Will refetch all dates")
            else:
                fetch_dates = check_result['missing_dates']
            
            fetch_stats = fetch_missing_props(fetch_dates, args.market, force=args.force)
            
            if fetch_stats['fetched'] == 0:
                logging.warning("No new props fetched")
        else:
            logging.info("✅ All props already fetched!")
    
    # STEP 3: Combine individual files
    print()
    combined_df = combine_props_files(args.season, args.market)
    
    if combined_df.empty:
        logging.error("❌ Failed to combine props")
        return
    
    # STEP 4: Build consensus
    if not args.skip_consensus:
        print()
        consensus_df = build_consensus_props(combined_df, args.season, args.market)
        
        if consensus_df.empty:
            logging.error("❌ Failed to build consensus")
            return
    
    # FINAL SUMMARY
    print()
    logging.info("="*80)
    logging.info("✅ COMPLETE!")
    logging.info("="*80)
    logging.info(f"Season: {args.season}")
    logging.info(f"Market: {args.market}")
    logging.info("")
    season_clean = args.season.replace('-', '_')
    logging.info("Output files:")
    logging.info(f"  Combined:  data/03_intermediate/combined_props_{season_clean}_{args.market}.csv")
    if not args.skip_consensus:
        logging.info(f"  Consensus: data/03_intermediate/consensus_props_{season_clean}_{args.market}.csv")
    logging.info("")
    logging.info("Ready to use for analysis!")
    logging.info("="*80)


if __name__ == '__main__':
    main()

