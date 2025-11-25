"""
NBA Game Calendar Builder

This script:
1. Fetches all NBA games from any season
2. Creates a calendar of unique game dates
3. Saves game dates and event IDs for historical prop fetching

Usage:
    # Build calendar for current season
    python nba_calendar_builder.py
    
    # Build for specific season
    python nba_calendar_builder.py --season 2025-26

Note: This calendar only includes Regular Season games from the NBA API.
Special games like the NBA Cup Championship(e.g., Bucks vs. Thunder on Dec 17, 2024)
are not included as they fall under a different season type.
"""

import pandas as pd
from datetime import datetime, timedelta
import json
from pathlib import Path
import ssl
import urllib3
import requests
import argparse
import sys

# SSL fix for NBA API
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Patch Session.request
original_session_request = requests.Session.request
def patched_session_request(self, *args, **kwargs):
    kwargs['verify'] = False
    kwargs.setdefault('timeout', 120)  # Increase timeout to 120 seconds
    return original_session_request(self, *args, **kwargs)
requests.Session.request = patched_session_request

# Patch requests.get (nba_api uses this directly)
original_requests_get = requests.get
def patched_requests_get(*args, **kwargs):
    kwargs['verify'] = False
    kwargs.setdefault('timeout', 120)  # Increase timeout to 120 seconds
    return original_requests_get(*args, **kwargs)
requests.get = patched_requests_get

from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import teams
import time

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import CURRENT_NBA_SEASON


def get_all_nba_games(season='2025-26', max_retries=3):
    """
    Fetch all NBA games from specified season
    
    Args:
        season: Season string (e.g., '2025-26')
        max_retries: Number of times to retry on timeout
    
    Returns:
        DataFrame with game dates and info
    """
    print(f"🏀 Fetching all NBA games for {season} season...")
    print("⏳ This may take a moment...\n")
    
    # Retry logic for flaky NBA API
    for attempt in range(max_retries):
        try:
            # Get all games for specified season
            gamefinder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                season_type_nullable='Regular Season',
                league_id_nullable='00'
            )
            
            games = gamefinder.get_data_frames()[0]
            
            # Each game appears twice (once for each team), so we need to deduplicate
            # Use GAME_ID to get unique games
            unique_games = games.drop_duplicates(subset=['GAME_ID']).copy()
            
            print(f"✅ Found {len(unique_games)} unique games")
            
            return unique_games
            
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5  # 5, 10, 15 seconds
                print(f"⚠️  Request timed out (attempt {attempt + 1}/{max_retries})")
                print(f"   Waiting {wait_time} seconds before retry...\n")
                time.sleep(wait_time)
            else:
                print(f"❌ Failed after {max_retries} attempts")
                raise


def create_game_calendar(games_df):
    """
    Create a calendar of unique game dates
    Returns sorted list of dates and summary stats
    """
    print("\n📅 Creating game calendar...\n")
    
    # Convert GAME_DATE to datetime
    games_df['GAME_DATE_DT'] = pd.to_datetime(games_df['GAME_DATE'])
    games_df['DATE_ONLY'] = games_df['GAME_DATE_DT'].dt.date
    
    # Get unique dates
    unique_dates = sorted(games_df['DATE_ONLY'].unique())
    
    # Get opening day
    opening_day = unique_dates[0]
    
    # Count games per date
    games_per_date = games_df.groupby('DATE_ONLY').size().reset_index(name='num_games')
    
    print(f"🎯 Season Stats:")
    print(f"   Opening Day: {opening_day}")
    print(f"   Total Game Days: {len(unique_dates)}")
    print(f"   Total Games: {len(games_df)}")
    print(f"   Average Games per Day: {len(games_df) / len(unique_dates):.1f}")
    print(f"   Date Range: {unique_dates[0]} to {unique_dates[-1]}")
    
    return unique_dates, games_per_date, opening_day


def get_games_for_date(games_df, target_date):
    """
    Get all games for a specific date
    Returns DataFrame with game info
    """
    games_df['DATE_ONLY'] = pd.to_datetime(games_df['GAME_DATE']).dt.date
    
    date_games = games_df[games_df['DATE_ONLY'] == target_date].copy()
    
    # Keep only unique games (deduplicate by GAME_ID)
    date_games = date_games.drop_duplicates(subset=['GAME_ID'])
    
    return date_games[['GAME_ID', 'GAME_DATE', 'MATCHUP', 'TEAM_NAME']].sort_values('GAME_DATE')


def save_calendar(unique_dates, games_df, season='2025-26', output_dir=None):
    """
    Save game calendar and metadata to files
    
    Args:
        unique_dates: List of unique game dates
        games_df: DataFrame with all games
        season: Season string (e.g., '2025-26')
        output_dir: Output directory path (defaults to data/01_input/nba_calendar)
    """
    if output_dir is None:
        # Save to data/01_input/nba_calendar
        project_root = Path(__file__).parent.parent
        output_path = project_root / 'data' / '01_input' / 'nba_calendar'
    else:
        output_path = Path(output_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert season format for filenames: 2025-26 -> 2025_26
    season_underscore = season.replace('-', '_')
    
    # Save unique dates as list
    dates_list = [str(d) for d in unique_dates]
    json_file = output_path / f'game_dates_{season_underscore}.json'
    with open(json_file, 'w') as f:
        json.dump({
            'season': season,
            'total_game_days': len(unique_dates),
            'opening_day': str(unique_dates[0]),
            'last_game': str(unique_dates[-1]),
            'dates': dates_list
        }, f, indent=2)
    
    # Save full games dataset
    games_df['DATE_ONLY'] = pd.to_datetime(games_df['GAME_DATE']).dt.date
    csv_file = output_path / f'all_games_{season_underscore}.csv'
    games_df.to_csv(csv_file, index=False)
    
    # Create daily summary
    daily_summary = games_df.groupby('DATE_ONLY').agg({
        'GAME_ID': 'count',
        'MATCHUP': lambda x: ', '.join(x.unique()[:5]) + ('...' if len(x.unique()) > 5 else '')
    }).reset_index()
    daily_summary.columns = ['Date', 'Num_Games', 'Sample_Matchups']
    summary_file = output_path / f'daily_summary_{season_underscore}.csv'
    daily_summary.to_csv(summary_file, index=False)
    
    print(f"\n💾 Calendar saved to {output_path}/")
    print(f"   - {json_file.name} (list of dates)")
    print(f"   - {csv_file.name} (full game data)")
    print(f"   - {summary_file.name} (games per day)")
    
    return output_path


def estimate_api_costs(num_game_days, avg_games_per_day=12):
    """
    Estimate Odds API costs for fetching historical props
    """
    print("\n" + "="*60)
    print("💰 ESTIMATED ODDS API COSTS")
    print("="*60)
    
    # Cost per game for historical event odds with player props
    # 1 market (player_threes) × 1 region × 10 credits = 10 credits per game
    
    cost_per_game_1_market = 10
    cost_per_game_4_markets = 40  # player_points, player_rebounds, player_assists, player_threes
    
    total_games = num_game_days * avg_games_per_day
    
    print(f"\nScenario 1: Fetch player_threes only")
    print(f"  {num_game_days} days × {avg_games_per_day} games/day = {total_games:.0f} total games")
    print(f"  Cost: {total_games:.0f} games × 10 credits = {total_games * cost_per_game_1_market:.0f} credits")
    print(f"  Your budget: 20,000 credits")
    print(f"  ✅ Feasible!" if total_games * cost_per_game_1_market <= 20000 else "  ⚠️  Over budget!")
    
    print(f"\nScenario 2: Fetch 4 markets (points, rebounds, assists, 3s)")
    print(f"  Cost: {total_games:.0f} games × 40 credits = {total_games * cost_per_game_4_markets:.0f} credits")
    print(f"  Your budget: 20,000 credits")
    print(f"  ✅ Feasible!" if total_games * cost_per_game_4_markets <= 20000 else "  ⚠️  Over budget!")
    
    print(f"\nScenario 3: Strategic approach (recommended)")
    print(f"  - Fetch all games, 1 market: ~{total_games * cost_per_game_1_market:.0f} credits")
    print(f"  - Leaves {20000 - (total_games * cost_per_game_1_market):.0f} credits for:")
    print(f"    • Additional markets for interesting games")
    print(f"    • Re-fetching specific dates")
    print(f"    • Future analysis")
    
    print("="*60 + "\n")


def main(season=None):
    """
    Main function - builds NBA calendar and prepares for prop fetching
    
    Args:
        season: Season string (e.g., '2025-26'). Defaults to current season.
    """
    if season is None:
        season = CURRENT_NBA_SEASON
    
    print("="*60)
    print(f"NBA GAME CALENDAR BUILDER - {season} SEASON")
    print("="*60 + "\n")
    
    # Fetch all games
    games_df = get_all_nba_games(season)
    
    # Create calendar
    unique_dates, games_per_date, opening_day = create_game_calendar(games_df)
    
    # Show opening day games
    print("\n" + "="*60)
    print(f"🎉 OPENING DAY: {opening_day}")
    print("="*60)
    
    opening_games = get_games_for_date(games_df, opening_day)
    print(f"\nGames on opening day ({len(opening_games)}):")
    for idx, game in opening_games.iterrows():
        print(f"  {game['MATCHUP']}")
    
    # Save calendar
    output_path = save_calendar(unique_dates, games_df, season)
    
    # Show cost estimates
    avg_games = len(games_df) / len(unique_dates)
    estimate_api_costs(len(unique_dates), avg_games)
    
    # Show next steps
    print("="*60)
    print("🎯 NEXT STEPS")
    print("="*60)
    print(f"""
1. We've identified {len(unique_dates)} game days in {season} season

2. To fetch props for this season:
   python scripts/fetch_and_build_season_props.py --season {season} --market player_threes

This will:
- Check which dates already have props
- Fetch missing dates from The Odds API
- Combine all props into consensus dataset
""")
    
    return unique_dates, games_df, opening_day


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Build NBA game calendar for historical prop fetching',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build current season calendar
  python nba_calendar_builder.py
  
  # Build specific season
  python nba_calendar_builder.py --season 2025-26
        """
    )
    
    parser.add_argument('--season', type=str, default=None,
                       help=f'Season to build calendar for (default: {CURRENT_NBA_SEASON})')
    
    args = parser.parse_args()
    
    unique_dates, games_df, opening_day = main(args.season)

