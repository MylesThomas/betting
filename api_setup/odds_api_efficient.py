"""
The Odds API - Credit-Efficient Wrapper
For $30/month plan: 20,000 credits

Strategy:
- Track every request's credit cost
- Cache responses to avoid duplicate calls
- Batch requests when possible
- YOU control when requests fire (no auto-fetching)
"""

import requests
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
import json
from pathlib import Path

# Load environment variables
load_dotenv()


class OddsAPIEfficient:
    """Credit-conscious wrapper for The Odds API"""
    
    def __init__(self, api_key=None, cache_dir='odds_cache'):
        self.api_key = api_key or os.getenv('ODDS_API_KEY')
        self.base_url = 'https://api.the-odds-api.com/v4'
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Track usage
        self.credits_remaining = None
        self.credits_used = None
        self.request_count = 0
        
        if not self.api_key or self.api_key == 'your_api_key_here':
            raise ValueError("No valid API key found! Add it to .env file")
    
    def _make_request(self, endpoint, params, cache_key=None):
        """
        Make API request with credit tracking
        
        Args:
            endpoint: API endpoint path
            params: Request parameters
            cache_key: Optional cache key to avoid duplicate requests
        """
        # Check cache first
        if cache_key:
            cached = self._get_cached(cache_key)
            if cached:
                print(f"✅ Using cached data (saved credits!)")
                return cached
        
        # Show current credits before request
        if self.credits_remaining is not None:
            print(f"💰 Credits before request: {self.credits_remaining:,}")
        
        # Make request
        url = f"{self.base_url}/{endpoint}"
        params['apiKey'] = self.api_key
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            # Track credits
            self.credits_remaining = int(response.headers.get('x-requests-remaining', 0))
            self.credits_used = int(response.headers.get('x-requests-used', 0))
            self.request_count += 1
            
            # Calculate cost of this request
            if self.request_count == 1:
                cost = self.credits_used
            else:
                cost = "Unknown (tracking from first request)"
            
            print(f"📊 Request #{self.request_count} completed")
            print(f"💵 Credits used: {cost}")
            print(f"💰 Credits remaining: {self.credits_remaining:,} / 20,000")
            print(f"📈 Percentage used: {(self.credits_used/20000)*100:.2f}%\n")
            
            data = response.json()
            
            # Cache the response
            if cache_key:
                self._save_cache(cache_key, data)
            
            return data
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                print("❌ Invalid API key!")
            elif e.response.status_code == 429:
                print("❌ Rate limit exceeded!")
            else:
                print(f"❌ Error: {e}")
            return None
    
    def _get_cached(self, cache_key):
        """Get cached response if exists and recent"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                cached = json.load(f)
                cached_time = datetime.fromisoformat(cached['timestamp'])
                age_hours = (datetime.now() - cached_time).total_seconds() / 3600
                
                # Use cache if less than 4 hours old (adjust as needed)
                if age_hours < 4:
                    print(f"🗂️  Cache hit! (Saved from {age_hours:.1f} hours ago)")
                    return cached['data']
        return None
    
    def _save_cache(self, cache_key, data):
        """Save response to cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'data': data
            }, f, indent=2)
    
    def check_credits(self):
        """
        Check remaining credits (costs 1 credit)
        Returns: dict with usage info
        """
        print("🔍 Checking credit usage...\n")
        data = self._make_request('sports/', {})
        
        if data:
            usage = {
                'remaining': self.credits_remaining,
                'used': self.credits_used,
                'total': 20000,
                'percentage_used': (self.credits_used / 20000) * 100
            }
            return usage
        return None
    
    def get_nba_player_props(self, markets='player_points,player_rebounds,player_assists,player_threes', 
                             bookmakers=None, use_cache=True):
        """
        Get NBA player props (MAIN FUNCTION FOR YOUR STRATEGY)
        
        Args:
            markets: Comma-separated list of markets to fetch
                - player_points
                - player_rebounds
                - player_assists
                - player_threes
                - player_blocks
                - player_steals
                - player_turnovers
                - player_points_rebounds_assists
            bookmakers: Optional specific bookmakers (e.g., 'draftkings,fanduel')
            use_cache: Use cached data if available
        
        Returns:
            List of games with player prop data
        """
        print(f"🏀 Fetching NBA player props: {markets}")
        print(f"📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        
        cache_key = f"nba_props_{markets}_{datetime.now().strftime('%Y%m%d_%H')}" if use_cache else None
        
        params = {
            'regions': 'us',
            'markets': markets,
            'oddsFormat': 'american'
        }
        
        if bookmakers:
            params['bookmakers'] = bookmakers
        
        data = self._make_request('sports/basketball_nba/odds/', params, cache_key)
        return data
    
    def get_nba_game_odds(self, markets='h2h,spreads,totals', use_cache=True):
        """
        Get NBA game odds (spreads, totals, moneylines)
        Less useful for your strategy but good for context
        """
        print(f"🏀 Fetching NBA game odds: {markets}\n")
        
        cache_key = f"nba_games_{datetime.now().strftime('%Y%m%d_%H')}" if use_cache else None
        
        params = {
            'regions': 'us',
            'markets': markets,
            'oddsFormat': 'american'
        }
        
        data = self._make_request('sports/basketball_nba/odds/', params, cache_key)
        return data
    
    def parse_player_props_to_df(self, data):
        """
        Parse player prop data into a clean DataFrame
        
        Returns:
            DataFrame with columns: player, market, line, over_odds, under_odds, bookmaker, game
        """
        if not data:
            return pd.DataFrame()
        
        props_list = []
        
        for game in data:
            game_info = f"{game['away_team']} @ {game['home_team']}"
            game_time = game.get('commence_time')
            
            for bookmaker in game.get('bookmakers', []):
                bookmaker_name = bookmaker['key']
                
                for market in bookmaker.get('markets', []):
                    market_key = market['key']
                    
                    # Group outcomes by player (over/under pairs)
                    player_props = {}
                    for outcome in market.get('outcomes', []):
                        player = outcome.get('description', 'Unknown')
                        line = outcome.get('point')
                        odds = outcome.get('price')
                        bet_type = outcome.get('name')  # 'Over' or 'Under'
                        
                        if player not in player_props:
                            player_props[player] = {
                                'player': player,
                                'market': market_key,
                                'line': line,
                                'bookmaker': bookmaker_name,
                                'game': game_info,
                                'game_time': game_time
                            }
                        
                        if bet_type == 'Over':
                            player_props[player]['over_odds'] = odds
                        elif bet_type == 'Under':
                            player_props[player]['under_odds'] = odds
                    
                    props_list.extend(player_props.values())
        
        df = pd.DataFrame(props_list)
        
        if not df.empty:
            # Reorder columns
            cols = ['player', 'market', 'line', 'over_odds', 'under_odds', 'bookmaker', 'game', 'game_time']
            df = df[[c for c in cols if c in df.columns]]
        
        return df
    
    def get_usage_summary(self):
        """Display usage summary"""
        if self.credits_remaining is None:
            print("No requests made yet. Run check_credits() first.")
            return
        
        print("\n" + "="*60)
        print("📊 API USAGE SUMMARY")
        print("="*60)
        print(f"Plan: $30/month (20K credits)")
        print(f"Requests made this session: {self.request_count}")
        print(f"Credits used (total): {self.credits_used:,}")
        print(f"Credits remaining: {self.credits_remaining:,}")
        print(f"Percentage used: {(self.credits_used/20000)*100:.2f}%")
        print(f"Estimated requests left: ~{self.credits_remaining // 10} (assuming 10 credits/request)")
        print("="*60 + "\n")


def demo_usage():
    """
    Example usage - shows how to use the API efficiently
    """
    print("="*60)
    print("EFFICIENT ODDS API - DEMO")
    print("="*60 + "\n")
    
    print("""
This script shows you how to use the API WITHOUT making requests.
When you're ready to fetch real data, uncomment the sections below.

Key Features:
- 🔍 Tracks credits in real-time
- 💾 Caches responses (saves credits)
- 📊 Shows usage after each request
- 🎯 Optimized for your trend strategy
""")
    
    # Initialize API
    api = OddsAPIEfficient()
    
    print("\n" + "="*60)
    print("EXAMPLE 1: Check Your Credits (costs 1 credit)")
    print("="*60)
    print("""
# Uncomment to run:
# usage = api.check_credits()
# print(f"You have {usage['remaining']:,} credits remaining")
""")
    
    print("\n" + "="*60)
    print("EXAMPLE 2: Get Player Props (costs ~10-20 credits)")
    print("="*60)
    print("""
# Fetch multiple prop markets in ONE request (efficient!)
# props = api.get_nba_player_props(
#     markets='player_points,player_rebounds,player_assists,player_threes'
# )
# 
# # Convert to DataFrame for easy analysis
# df = api.parse_player_props_to_df(props)
# print(df.head(20))
# 
# # Save to CSV for combining with NBA stats
# df.to_csv('todays_props.csv', index=False)
""")
    
    print("\n" + "="*60)
    print("EXAMPLE 3: Cache Usage (saves credits)")
    print("="*60)
    print("""
# First call: Uses credits
# props1 = api.get_nba_player_props(use_cache=True)
# 
# Second call within 4 hours: Uses cache (FREE!)
# props2 = api.get_nba_player_props(use_cache=True)
# 
# Cache is automatic and timestamp-based
""")
    
    print("\n" + "="*60)
    print("CREDIT MANAGEMENT TIPS")
    print("="*60)
    print("""
With 20,000 credits/month:

1. Daily Usage Strategy:
   - 1 request/day for props: ~10-20 credits
   - 30 days x 20 credits = ~600 credits/month
   - You have PLENTY of room!

2. Efficient Practices:
   - Fetch multiple markets in ONE call (saves credits)
   - Use cache for repeated analysis (automatic)
   - Run once per day, analyze offline with NBA API

3. Credit Costs (approximate):
   - Sports list: 1 credit
   - Game odds (basic): 5-10 credits
   - Player props (multi-market): 10-30 credits
   - More bookmakers = more credits

4. Recommended Workflow:
   - Morning: Fetch today's props (1 request, ~20 credits)
   - Analyze with NBA API (FREE, unlimited)
   - Evening: Re-fetch if needed (uses cache if <4hrs)

Your 20K credits should last the entire month easily!
""")


if __name__ == "__main__":
    demo_usage()

