"""
Simple test script to fetch NFL and NBA championship futures from The Odds API.

Purpose:
- Fetch NFL Super Bowl championship odds
- Fetch NBA Championship odds
- Save timestamped files to track odds movement over time

Usage:
    cd /Users/thomasmyles/dev/betting/api_setup
    python3 test_futures_simple.py

Output:
- data/01_input/the-odds-api/nfl/futures/nfl_super_bowl_futures_YYYYMMDD_HHMMSS.csv
- data/01_input/the-odds-api/nba/futures/nba_championship_futures_YYYYMMDD_HHMMSS.csv

API docs: https://the-odds-api.com/liveapi/guides/v4/
"""

# SSL Fix for macOS - must be imported BEFORE requests
import ssl
import urllib3

ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import requests
import pandas as pd
import os
from datetime import datetime
from dotenv import load_dotenv

# Monkey-patch requests to disable SSL verification
original_request = requests.Session.request

def patched_request(self, *args, **kwargs):
    kwargs['verify'] = False
    return original_request(self, *args, **kwargs)

requests.Session.request = patched_request

# Load environment variables
load_dotenv()

API_KEY = os.getenv('ODDS_API_KEY')
BASE_URL = 'https://api.the-odds-api.com/v4'


def fetch_futures(sport_key):
    """Fetch futures odds for a given sport key"""
    url = f"{BASE_URL}/sports/{sport_key}/odds/"
    
    params = {
        'apiKey': API_KEY,
        'regions': 'us',
        'oddsFormat': 'american'
    }
    
    response = requests.get(url, params=params)
    
    # Print API usage
    remaining = response.headers.get('x-requests-remaining')
    used = response.headers.get('x-requests-used')
    if remaining:
        print(f"API Usage: {used} used, {remaining} remaining")
    
    response.raise_for_status()
    return response.json()


def parse_futures_to_df(data, sport_name):
    """
    Parse futures data into a DataFrame.
    
    Note: The Odds API with oddsFormat='american' should return proper American odds.
    Positive odds (underdogs) are returned as positive integers (e.g., 150 means +150).
    Negative odds (favorites) are returned as negative integers (e.g., -110).
    
    Some bookmakers may have data quality issues - we store odds as-is from the API.
    """
    futures_list = []
    
    for item in data:
        sport_key = item.get('sport_key')
        
        for bookmaker in item.get('bookmakers', []):
            bookmaker_name = bookmaker['key']
            
            for market in bookmaker.get('markets', []):
                market_key = market['key']
                
                for outcome in market.get('outcomes', []):
                    odds = outcome.get('price')
                    
                    futures_list.append({
                        'sport': sport_name,
                        'bookmaker': bookmaker_name,
                        'team': outcome.get('name'),
                        'odds': odds
                    })
    
    return pd.DataFrame(futures_list)


def main():
    """Main test function"""
    
    if not API_KEY:
        print("❌ ERROR: ODDS_API_KEY not found in .env file")
        print("Add your API key to .env: ODDS_API_KEY=your_key_here")
        return
    
    # Generate timestamp for this fetch
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("="*80)
    print("TESTING FUTURES MARKETS")
    print("="*80)
    print(f"Timestamp: {timestamp}\n")
    
    # Test NFL Super Bowl futures
    print("\n🏈 Fetching NFL Super Bowl futures...")
    try:
        nfl_data = fetch_futures('americanfootball_nfl_super_bowl_winner')
        df_nfl = parse_futures_to_df(nfl_data, 'NFL')
        
        if not df_nfl.empty:
            print(f"✅ Found {len(df_nfl)} odds from {df_nfl['bookmaker'].nunique()} bookmakers")
            
            # Show top 10 favorites by best available odds
            # Best odds = most favorable to bettor (least negative for favorites, most positive for dogs)
            best_odds_per_team = df_nfl.loc[df_nfl.groupby('team')['odds'].idxmax()]
            best_odds_per_team = best_odds_per_team.sort_values('odds', ascending=False)
            
            print("\nTop 10 Super Bowl Favorites (Best Available Odds):")
            print("-" * 70)
            for i, row in enumerate(best_odds_per_team.head(10).itertuples(), 1):
                odds_str = f"+{int(row.odds)}" if row.odds > 0 else f"{int(row.odds)}"
                print(f"{i:2d}. {row.team:<30} {odds_str:>7}  ({row.bookmaker})")
            
            # Save to CSV with timestamp
            output_file = f'../data/01_input/the-odds-api/nfl/futures/nfl_super_bowl_futures_{timestamp}.csv'
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            df_nfl.to_csv(output_file, index=False)
            print(f"\n💾 Saved to: {output_file}")
        else:
            print("⚠️  No NFL futures data found")
            
    except Exception as e:
        print(f"❌ Error fetching NFL futures: {e}")
    
    # Test NBA Championship futures
    print("\n\n🏀 Fetching NBA Championship futures...")
    try:
        nba_data = fetch_futures('basketball_nba_championship_winner')
        df_nba = parse_futures_to_df(nba_data, 'NBA')
        
        if not df_nba.empty:
            print(f"✅ Found {len(df_nba)} odds from {df_nba['bookmaker'].nunique()} bookmakers")
            
            # Show top 10 favorites by best available odds
            # Best odds = most favorable to bettor (least negative for favorites, most positive for dogs)
            best_odds_per_team = df_nba.loc[df_nba.groupby('team')['odds'].idxmax()]
            best_odds_per_team = best_odds_per_team.sort_values('odds', ascending=False)
            
            print("\nTop 10 NBA Championship Favorites (Best Available Odds):")
            print("-" * 70)
            for i, row in enumerate(best_odds_per_team.head(10).itertuples(), 1):
                odds_str = f"+{int(row.odds)}" if row.odds > 0 else f"{int(row.odds)}"
                print(f"{i:2d}. {row.team:<30} {odds_str:>7}  ({row.bookmaker})")
            
            # Save to CSV with timestamp
            output_file = f'../data/01_input/the-odds-api/nba/futures/nba_championship_futures_{timestamp}.csv'
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            df_nba.to_csv(output_file, index=False)
            print(f"\n💾 Saved to: {output_file}")
        else:
            print("⚠️  No NBA futures data found")
            
    except Exception as e:
        print(f"❌ Error fetching NBA futures: {e}")
    
    print("\n" + "="*80)
    print("✅ TEST COMPLETE")
    print("="*80)
    
    # Show available sport keys for other futures
    print("\n📋 Other futures sport keys to try:")
    print("   - baseball_mlb_world_series_winner")
    print("   - icehockey_nhl_championship_winner")
    print("   - basketball_ncaab_championship_winner")
    print("   - americanfootball_ncaaf_championship_winner")


if __name__ == "__main__":
    main()

