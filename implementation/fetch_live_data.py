"""
Fetch live NBA 3pt prop data for today's games.

QUICK START:
    # Test mode (no API key needed)
    python implementation/fetch_live_data.py --test
    
    # Live mode (requires API key)
    export ODDS_API_KEY="your_key"
    python implementation/fetch_live_data.py --live

MODES:
    --live: Fetch live data from The Odds API (requires ODDS_API_KEY environment variable)
    --test: Use existing test data from data/01_input/the-odds-api/nba/all_markets/ (no API calls)

OUTPUT FILES (saved to data/live/):
    - props_today_YYYYMMDD.csv          # Tonight's 3pt prop odds
    - game_results_season_YYYYMMDD.csv  # Season-to-date game results
    - metadata_YYYYMMDD.json            # Fetch metadata

EXAMPLES:
    # Fetch live data
    export ODDS_API_KEY="your_key_here"
    python implementation/fetch_live_data.py --live
    
    # Use test data (no API key needed)
    python implementation/fetch_live_data.py --test
    
    # Custom output directory
    python implementation/fetch_live_data.py --live --output-dir data/custom/

NEXT STEP:
    After fetching data, run find_todays_plays.py to identify betting opportunities

DATA ORGANIZATION:
    data/
    ├── live/               # Today's live data (output)
    ├── 04_output/nba/arbs/     # Arbitrage data output
    ├── 01_input/the-odds-api/nba/all_markets/  # Raw props (used for test mode)
    ├── 03_intermediate/    # Analysis results
    └── 01_input/           # Reference data

Author: Myles Thomas
Date: 2025-11-21
"""

import os
import sys
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import requests

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Configuration
API_KEY = os.environ.get('ODDS_API_KEY', '')
SPORT = 'basketball_nba'
REGIONS = 'us'
MARKETS = 'player_threes'
ODDS_FORMAT = 'american'
DATE_FORMAT = '%Y-%m-%d'


def fetch_todays_props(api_key, output_path=None):
    """
    Fetch today's NBA 3pt prop odds from The Odds API.
    
    Args:
        api_key: The Odds API key
        output_path: Optional custom output path
        
    Returns:
        DataFrame with today's props
    """
    print("="*80)
    print("FETCHING TODAY'S 3PT PROP ODDS")
    print("="*80)
    
    if not api_key:
        raise ValueError("API key required. Set ODDS_API_KEY environment variable.")
    
    url = f'https://api.the-odds-api.com/v4/sports/{SPORT}/odds/'
    
    params = {
        'apiKey': api_key,
        'regions': REGIONS,
        'markets': MARKETS,
        'oddsFormat': ODDS_FORMAT,
    }
    
    print(f"Fetching odds from The Odds API...")
    print(f"Sport: {SPORT}")
    print(f"Markets: {MARKETS}")
    print()
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        # Check remaining requests
        remaining_requests = response.headers.get('x-requests-remaining')
        used_requests = response.headers.get('x-requests-used')
        print(f"API Usage: {used_requests} used, {remaining_requests} remaining")
        print()
        
        # Parse the response into a DataFrame
        props_data = []
        
        for game in data:
            game_id = game['id']
            commence_time = game['commence_time']
            home_team = game['home_team']
            away_team = game['away_team']
            
            for bookmaker in game.get('bookmakers', []):
                bookmaker_key = bookmaker['key']
                bookmaker_name = bookmaker['title']
                
                for market in bookmaker.get('markets', []):
                    if market['key'] == 'player_threes':
                        for outcome in market.get('outcomes', []):
                            props_data.append({
                                'game_id': game_id,
                                'commence_time': commence_time,
                                'home_team': home_team,
                                'away_team': away_team,
                                'bookmaker_key': bookmaker_key,
                                'bookmaker_name': bookmaker_name,
                                'player_name': outcome['description'],
                                'line': outcome['point'],
                                'over_under': outcome['name'],
                                'odds': outcome['price'],
                            })
        
        df = pd.DataFrame(props_data)
        
        if len(df) == 0:
            print("⚠️  No props found. There may be no games today.")
            return df
        
        # Add fetch timestamp
        df['fetch_time'] = datetime.now(timezone.utc).isoformat()
        
        # Save to file
        if output_path is None:
            today_str = datetime.now().strftime('%Y%m%d')
            output_path = f'data/live/props_today_{today_str}.csv'
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print(f"✅ Saved {len(df)} prop lines to: {output_path}")
        print(f"   Games: {df['game_id'].nunique()}")
        print(f"   Players: {df['player_name'].nunique()}")
        print(f"   Bookmakers: {df['bookmaker_key'].nunique()}")
        print()
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching data: {e}")
        raise


def fetch_season_game_results(output_path=None):
    """
    Load season-to-date game results from existing data.
    
    This uses the nba_game_results_2024_25.csv file which has:
    - Player 3PM/3PA per game
    - Minutes played
    - Game dates
    
    Args:
        output_path: Optional custom output path
        
    Returns:
        DataFrame with season game results
    """
    print("="*80)
    print("LOADING SEASON-TO-DATE GAME RESULTS")
    print("="*80)
    
    source_path = 'data/03_intermediate/nba_game_results_2024_25.csv'
    
    if not os.path.exists(source_path):
        print(f"❌ Error: {source_path} not found")
        print("   Run scripts/fetch_nba_game_results.py first to fetch game results")
        raise FileNotFoundError(f"Game results file not found: {source_path}")
    
    df = pd.read_csv(source_path)
    
    # Filter to games that have already happened (before today)
    today = datetime.now().strftime(DATE_FORMAT)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] < today].copy()
    
    # Save to live data folder
    if output_path is None:
        today_str = datetime.now().strftime('%Y%m%d')
        output_path = f'data/live/game_results_season_{today_str}.csv'
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✅ Loaded {len(df)} game results from: {source_path}")
    print(f"   Saved to: {output_path}")
    print(f"   Players: {df['player'].nunique()}")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print()
    
    return df


def use_test_data(output_dir='data/live'):
    """
    Use existing test data instead of fetching live data.
    
    Args:
        output_dir: Directory to save test data files
        
    Returns:
        Tuple of (props_df, game_results_df)
    """
    print("="*80)
    print("USING TEST DATA (No API calls)")
    print("="*80)
    print()
    
    # Use the most recent raw props file from data/01_input/the-odds-api/nba/all_markets/
    raw_props_dir = Path('data/01_input/the-odds-api/nba/all_markets')
    if not raw_props_dir.exists():
        raise FileNotFoundError("No test data found in data/01_input/the-odds-api/nba/all_markets/")
    
    # Find most recent raw props file
    raw_files = sorted(raw_props_dir.glob('raw_*.csv'))
    
    if not raw_files:
        raise FileNotFoundError("No raw props files found in data/01_input/the-odds-api/nba/all_markets/")
    
    props_file = raw_files[-1]
    print(f"Using props from: {props_file}")
    
    # Load and process props
    props_df = pd.read_csv(props_file)
    
    # Load game results
    game_results_file = 'data/03_intermediate/nba_game_results_2024_25.csv'
    if not os.path.exists(game_results_file):
        raise FileNotFoundError(f"Game results not found: {game_results_file}")
    
    game_results_df = pd.read_csv(game_results_file)
    
    # Filter game results to before today
    today = datetime.now().strftime('%Y-%m-%d')
    game_results_df['date'] = pd.to_datetime(game_results_df['date'])
    game_results_df = game_results_df[game_results_df['date'] < today].copy()
    
    print(f"Using game results from: {game_results_file}")
    print()
    print(f"✅ Test data loaded:")
    print(f"   Props: {len(props_df)} lines")
    print(f"   Game results: {len(game_results_df)} games")
    print()
    
    # Save to live directory for consistency
    today_str = datetime.now().strftime('%Y%m%d')
    
    props_output = f'{output_dir}/props_today_{today_str}.csv'
    results_output = f'{output_dir}/game_results_season_{today_str}.csv'
    
    os.makedirs(output_dir, exist_ok=True)
    props_df.to_csv(props_output, index=False)
    game_results_df.to_csv(results_output, index=False)
    
    print(f"📁 Saved test data to {output_dir}/")
    print()
    
    return props_df, game_results_df


def save_metadata(mode, props_count, games_count, output_dir='data/live'):
    """
    Save metadata about the data fetch.
    
    Args:
        mode: 'live' or 'test'
        props_count: Number of prop lines fetched
        games_count: Number of games in results
        output_dir: Directory to save metadata
    """
    today_str = datetime.now().strftime('%Y%m%d')
    metadata_path = f'{output_dir}/metadata_{today_str}.json'
    
    metadata = {
        'fetch_time': datetime.now(timezone.utc).isoformat(),
        'mode': mode,
        'props_count': props_count,
        'games_count': games_count,
        'season': '2024-25',
        'market': 'player_threes',
    }
    
    os.makedirs(output_dir, exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📄 Metadata saved to: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Fetch live NBA 3pt prop data for today\'s games',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch live betting data
  python implementation/fetch_live_data.py --live
  
  # Use test data (no API calls)
  python implementation/fetch_live_data.py --test
  
  # Specify custom output directory
  python implementation/fetch_live_data.py --live --output-dir data/custom/
        """
    )
    
    parser.add_argument('--live', action='store_true',
                       help='Fetch live data from The Odds API (requires API key)')
    parser.add_argument('--test', action='store_true',
                       help='Use test data from existing files (no API calls)')
    parser.add_argument('--output-dir', type=str, default='data/live',
                       help='Output directory for data files (default: data/live)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.live and args.test:
        parser.error("Cannot specify both --live and --test. Choose one.")
    
    if not args.live and not args.test:
        parser.error("Must specify either --live or --test mode.")
    
    print()
    print("🏀 NBA 3PT PROP DATA FETCHER")
    print("="*80)
    print()
    
    try:
        if args.live:
            # Fetch live data
            print("Mode: LIVE (fetching from API)")
            print()
            
            props_df = fetch_todays_props(API_KEY)
            game_results_df = fetch_season_game_results()
            
            save_metadata('live', len(props_df), len(game_results_df), args.output_dir)
            
        else:  # test mode
            # Use test data
            print("Mode: TEST (using existing data)")
            print()
            
            props_df, game_results_df = use_test_data(args.output_dir)
            
            save_metadata('test', len(props_df), len(game_results_df), args.output_dir)
        
        print()
        print("="*80)
        print("✅ DATA FETCH COMPLETE")
        print("="*80)
        print()
        print("Next step: Run find_todays_plays.py to identify betting opportunities")
        print()
        
    except Exception as e:
        print()
        print("="*80)
        print(f"❌ ERROR: {e}")
        print("="*80)
        sys.exit(1)


if __name__ == '__main__':
    main()

