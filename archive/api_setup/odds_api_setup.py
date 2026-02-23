"""
The Odds API Setup and Testing Script

This script tests The Odds API for fetching NBA player prop betting lines.
Free tier: 500 requests/month

Sign up at: https://the-odds-api.com/
"""

import requests
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
import json


# Load environment variables
load_dotenv()


class OddsAPI:
    """Wrapper for The Odds API with credit management"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('ODDS_API_KEY')
        self.base_url = 'https://api.the-odds-api.com/v4'
        self.last_remaining = None
        self.last_used = None
        
        if not self.api_key or self.api_key == 'your_api_key_here':
            print("⚠️  WARNING: No valid API key found!")
            print("Get your free API key at: https://the-odds-api.com/")
            print("Then add it to .env file as: ODDS_API_KEY=your_key")
    
    def check_usage(self):
        """Check remaining API requests"""
        if not self.api_key or self.api_key == 'your_api_key_here':
            return None
        
        # Make a minimal request to check usage
        url = f"{self.base_url}/sports/"
        params = {'apiKey': self.api_key}
        
        try:
            response = requests.get(url, params=params)
            remaining = response.headers.get('x-requests-remaining')
            used = response.headers.get('x-requests-used')
            
            if remaining:
                print(f"API Usage: {used} used, {remaining} remaining")
                return int(remaining)
            return None
        except Exception as e:
            print(f"Error checking usage: {e}")
            return None
    
    def get_sports(self):
        """Get all available sports"""
        url = f"{self.base_url}/sports/"
        params = {'apiKey': self.api_key}
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        return response.json()
    
    def get_nba_odds(self, markets='h2h', regions='us', bookmakers=None):
        """
        Get NBA game odds
        
        Args:
            markets: 'h2h' (moneyline), 'spreads', 'totals'
            regions: 'us', 'uk', 'eu', 'au'
            bookmakers: Specific bookmakers (e.g., 'draftkings,fanduel')
        """
        url = f"{self.base_url}/sports/basketball_nba/odds/"
        
        params = {
            'apiKey': self.api_key,
            'regions': regions,
            'markets': markets,
        }
        
        if bookmakers:
            params['bookmakers'] = bookmakers
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        return response.json()
    
    def get_player_props(self, markets='player_points,player_rebounds,player_assists,player_threes'):
        """
        Get NBA player prop markets
        
        Available markets:
        - player_points
        - player_rebounds  
        - player_assists
        - player_threes
        - player_blocks
        - player_steals
        - player_turnovers
        - player_points_rebounds_assists (combo)
        """
        url = f"{self.base_url}/sports/basketball_nba/odds/"
        
        params = {
            'apiKey': self.api_key,
            'regions': 'us',
            'markets': markets,
            'oddsFormat': 'american'
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Check remaining requests
            remaining = response.headers.get('x-requests-remaining')
            if remaining:
                print(f"Requests remaining: {remaining}")
            
            return data
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                print("❌ Invalid API key!")
            elif e.response.status_code == 422:
                print("⚠️  Player props may require a paid plan")
                print("Free tier focuses on game odds (moneyline, spreads, totals)")
            else:
                print(f"Error: {e}")
            return None
    
    def parse_player_props(self, data):
        """Parse player prop data into readable format"""
        if not data:
            return pd.DataFrame()
        
        props_list = []
        
        for game in data:
            game_time = game.get('commence_time')
            home_team = game.get('home_team')
            away_team = game.get('away_team')
            
            for bookmaker in game.get('bookmakers', []):
                bookmaker_name = bookmaker['key']
                
                for market in bookmaker.get('markets', []):
                    market_key = market['key']
                    
                    for outcome in market.get('outcomes', []):
                        props_list.append({
                            'game_time': game_time,
                            'matchup': f"{away_team} @ {home_team}",
                            'bookmaker': bookmaker_name,
                            'market': market_key,
                            'player': outcome.get('description'),
                            'line': outcome.get('point'),
                            'over_odds': outcome.get('price') if outcome.get('name') == 'Over' else None,
                            'under_odds': outcome.get('price') if outcome.get('name') == 'Under' else None
                        })
        
        df = pd.DataFrame(props_list)
        return df


def test_odds_api():
    """Run comprehensive test of The Odds API"""
    print("="*60)
    print("THE ODDS API SETUP TEST")
    print("="*60)
    
    api = OddsAPI()
    
    # Check if API key is configured
    if not api.api_key or api.api_key == 'your_api_key_here':
        print("\n❌ API KEY NOT CONFIGURED")
        print("\nSetup Instructions:")
        print("1. Go to https://the-odds-api.com/")
        print("2. Sign up for a free account (500 requests/month)")
        print("3. Copy your API key")
        print("4. Create a .env file in this directory with:")
        print("   ODDS_API_KEY=your_actual_key_here")
        print("\nOnce configured, run this script again!")
        return False
    
    # Test 1: Check usage
    print("\n1️⃣  Checking API key and usage...")
    try:
        remaining = api.check_usage()
        if remaining is not None:
            print("✅ API key is valid!\n")
        else:
            print("⚠️  Could not verify usage\n")
    except Exception as e:
        print(f"❌ Error: {e}\n")
        return False
    
    # Test 2: Get available sports
    print("2️⃣  Fetching available sports...")
    try:
        sports = api.get_sports()
        nba_sport = [s for s in sports if 'nba' in s['key'].lower()]
        if nba_sport:
            print(f"✅ Found NBA: {nba_sport[0]['title']}\n")
        else:
            print("⚠️  NBA not found in available sports\n")
    except Exception as e:
        print(f"❌ Error: {e}\n")
        return False
    
    # Test 3: Get NBA odds
    print("3️⃣  Fetching NBA game odds...")
    try:
        odds = api.get_nba_odds(markets='h2h,spreads,totals')
        if odds:
            print(f"✅ Found {len(odds)} NBA games with odds")
            print(f"\nSample game:")
            if len(odds) > 0:
                game = odds[0]
                print(f"   {game['away_team']} @ {game['home_team']}")
                print(f"   Start: {game['commence_time']}\n")
        else:
            print("⚠️  No games found (may be off-season or no games today)\n")
    except Exception as e:
        print(f"❌ Error: {e}\n")
    
    # Test 4: Try to get player props (may require paid plan)
    print("4️⃣  Attempting to fetch player props...")
    try:
        props = api.get_player_props()
        if props:
            df = api.parse_player_props(props)
            if not df.empty:
                print(f"✅ Found player props!")
                print(f"\nSample props:")
                print(df[['player', 'market', 'line']].head())
            else:
                print("⚠️  No player props available")
        else:
            print("⚠️  Player props require paid plan or not available")
            print("    Free tier provides game odds (moneyline, spreads, totals)")
            print("    For player props, consider PrizePicks or other sources")
    except Exception as e:
        print(f"Note: {e}")
    
    print("\n" + "="*60)
    print("✅ THE ODDS API SETUP COMPLETE!")
    print("="*60)
    print("\nNote: The free tier focuses on game odds.")
    print("Player props often require paid plans or alternative sources.")
    print("Consider scraping PrizePicks, DraftKings, or FanDuel for props.")
    
    return True


def demo_usage():
    """Demonstrate practical usage"""
    print("\n" + "="*60)
    print("EXAMPLE USAGE")
    print("="*60)
    
    code_example = '''
# Basic usage example:
from odds_api_setup import OddsAPI

api = OddsAPI()

# Check remaining requests
api.check_usage()

# Get NBA game odds
games = api.get_nba_odds(markets='h2h,spreads,totals')

# Try player props (if available on your plan)
props = api.get_player_props(markets='player_points,player_threes')
df = api.parse_player_props(props)
'''
    
    print(code_example)


if __name__ == "__main__":
    success = test_odds_api()
    
    if success:
        demo_usage()

