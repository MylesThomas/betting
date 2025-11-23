"""
Fetch NBA Game Results for 2024-25 Season

Gets actual 3PM and 3PA for each player in each game.
Required for backtesting prop betting strategies.

Uses nba_api package to fetch player game logs.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import time

# ============================================================================
# SSL FIX - Must be done BEFORE importing nba_api
# See: api_setup/fixing_ssl.md for full explanation
# ============================================================================
import ssl
import urllib3
import requests

# Disable SSL verification warnings
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Monkey-patch requests Session to disable SSL verification
original_request = requests.Session.request

def patched_request(self, *args, **kwargs):
    kwargs['verify'] = False
    return original_request(self, *args, **kwargs)

requests.Session.request = patched_request

# NOW import nba_api (it will use the patched requests)
try:
    from nba_api.stats.endpoints import playergamelogs
    from nba_api.stats.static import players
except ImportError:
    print("❌ nba_api package not installed!")
    print("   Install with: pip install nba-api")
    print()
    exit(1)


def fetch_player_game_logs_2024_25():
    """
    Fetch all player game logs for 2024-25 season.
    
    Returns DataFrame with:
    - player: Player name
    - date: Game date (YYYY-MM-DD format)
    - team: Player's team
    - opponent: Opponent team
    - threes_made: 3-pointers made (FG3M)
    - threes_attempted: 3-pointers attempted (FG3A)
    - minutes: Minutes played
    """
    
    print("="*70)
    print("FETCHING NBA GAME LOGS FOR 2024-25 SEASON")
    print("="*70)
    print()
    
    print("Fetching data from NBA API...")
    print("This may take a minute...")
    print()
    
    try:
        # Fetch all player game logs for 2024-25 season
        game_logs = playergamelogs.PlayerGameLogs(
            season_nullable='2024-25',
            season_type_nullable='Regular Season'
        )
        
        df = game_logs.get_data_frames()[0]
        
        print(f"✅ Fetched {len(df):,} player game log entries")
        print()
        
        # Process and clean the data
        print("Processing data...")
        
        # Select and rename columns
        result_df = df[[
            'PLAYER_NAME',
            'GAME_DATE', 
            'TEAM_ABBREVIATION',
            'MATCHUP',
            'FG3M',  # 3-pointers made
            'FG3A',  # 3-pointers attempted
            'MIN'    # Minutes played
        ]].copy()
        
        result_df.columns = [
            'player',
            'game_date',
            'team',
            'matchup',
            'threes_made',
            'threes_attempted',
            'minutes'
        ]
        
        # Convert game_date to YYYY-MM-DD format
        result_df['game_date'] = pd.to_datetime(result_df['game_date']).dt.strftime('%Y-%m-%d')
        result_df = result_df.rename(columns={'game_date': 'date'})
        
        # Extract opponent from matchup (format: "LAL vs. BOS" or "LAL @ BOS")
        result_df['opponent'] = result_df['matchup'].apply(
            lambda x: x.split()[-1] if isinstance(x, str) else None
        )
        
        # Handle minutes (might be null for DNPs)
        result_df['minutes'] = pd.to_numeric(result_df['minutes'], errors='coerce')
        
        # Filter to NBA regular season (starts Oct 22, 2024)
        result_df = result_df[result_df['date'] >= '2024-10-22'].copy()
        
        # Sort by date and player
        result_df = result_df.sort_values(['date', 'player']).reset_index(drop=True)
        
        # Display summary
        print(f"✅ Processed {len(result_df):,} NBA game logs (2024-10-22 onwards)")
        print()
        print("Summary:")
        print(f"  Date range: {result_df['date'].min()} to {result_df['date'].max()}")
        print(f"  Unique players: {result_df['player'].nunique()}")
        print(f"  Total games: {len(result_df['date'].unique())}")
        print()
        
        # Show sample data
        print("Sample data:")
        print(result_df.head(10).to_string(index=False))
        print()
        
        return result_df
        
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return None


def save_results(df, output_file):
    """Save results to CSV"""
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"💾 Saved to: {output_path}")
    print()


def main():
    """Fetch and save NBA game results"""
    
    # Fetch data
    df = fetch_player_game_logs_2024_25()
    
    if df is None:
        return
    
    # Save to data directory
    output_file = Path(__file__).parent.parent / 'data' / 'nba_game_results_2024_25.csv'
    save_results(df, output_file)
    
    print("="*70)
    print("✅ Game results fetched successfully!")
    print("="*70)
    print()
    print("You can now run the backtest:")
    print("  python backtesting/20251120_nba_3pt_prop_miss_streaks.py")
    print()


if __name__ == "__main__":
    main()

