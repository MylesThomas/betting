"""
Test script to verify which player prop markets are available from The Odds API
"""

import ssl
import urllib3
import requests
from dotenv import load_dotenv
import os

# SSL fix
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

# Markets to test
MARKETS_TO_TEST = [
    'player_points',
    'player_rebounds',
    'player_assists',
    'player_threes',
    'player_blocks',
    'player_steals',
    'player_turnovers',
    'player_double_double',
    'player_triple_double',
    'player_points_rebounds_assists',
]

def test_market(api_key, event_id, market):
    """Test if a specific market is available"""
    url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{event_id}/odds"
    
    params = {
        'apiKey': api_key,
        'regions': 'us',
        'markets': market,
        'oddsFormat': 'american',
    }
    
    try:
        response = requests.get(url, params=params, verify=False)
        response.raise_for_status()
        data = response.json()
        
        # Check if we got any bookmakers with this market
        if data.get('bookmakers'):
            for bookmaker in data['bookmakers']:
                if bookmaker.get('markets'):
                    for mkt in bookmaker['markets']:
                        if mkt['key'] == market:
                            num_outcomes = len(mkt.get('outcomes', []))
                            return True, num_outcomes
        
        return False, 0
        
    except Exception as e:
        return None, str(e)


def main():
    api_key = os.getenv('ODDS_API_KEY')
    if not api_key:
        print("❌ No API key found!")
        return
    
    print("="*80)
    print("🏀 TESTING NBA PLAYER PROP MARKETS")
    print("="*80 + "\n")
    
    # First get today's events
    print("🔍 Fetching today's NBA events...\n")
    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/events"
    response = requests.get(url, params={'apiKey': api_key}, verify=False)
    
    if response.status_code != 200:
        print("❌ Failed to get events")
        return
    
    events = response.json()
    if not events:
        print("❌ No events found today")
        return
    
    # Use the first event for testing
    event = events[0]
    event_id = event['id']
    print(f"✅ Using event: {event['away_team']} @ {event['home_team']}")
    print(f"   Event ID: {event_id}\n")
    
    remaining = response.headers.get('x-requests-remaining')
    print(f"📊 Starting credits: {remaining}\n")
    print("-"*80 + "\n")
    
    # Test each market
    available_markets = []
    unavailable_markets = []
    error_markets = []
    
    for i, market in enumerate(MARKETS_TO_TEST, 1):
        print(f"Testing {i}/{len(MARKETS_TO_TEST)}: {market}...", end=" ")
        
        is_available, result = test_market(api_key, event_id, market)
        
        if is_available is True:
            print(f"✅ Available ({result} outcomes)")
            available_markets.append((market, result))
        elif is_available is False:
            print(f"❌ Not available")
            unavailable_markets.append(market)
        else:
            print(f"⚠️  Error: {result}")
            error_markets.append((market, result))
    
    # Final check on credits
    print("\n" + "-"*80 + "\n")
    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/events"
    response = requests.get(url, params={'apiKey': api_key}, verify=False)
    remaining = response.headers.get('x-requests-remaining')
    used = response.headers.get('x-requests-used')
    print(f"📊 Final credits: {remaining} (used {used} total this month)\n")
    
    # Summary
    print("="*80)
    print("📊 SUMMARY")
    print("="*80 + "\n")
    
    if available_markets:
        print(f"✅ AVAILABLE MARKETS ({len(available_markets)}):")
        for market, outcomes in available_markets:
            print(f"   - {market} ({outcomes} outcomes)")
        print()
    
    if unavailable_markets:
        print(f"❌ UNAVAILABLE MARKETS ({len(unavailable_markets)}):")
        for market in unavailable_markets:
            print(f"   - {market}")
        print()
    
    if error_markets:
        print(f"⚠️  ERROR MARKETS ({len(error_markets)}):")
        for market, error in error_markets:
            print(f"   - {market}: {error}")
        print()
    
    # Generate recommended config
    if available_markets:
        markets_string = ','.join([m for m, _ in available_markets])
        print("="*80)
        print("💡 RECOMMENDED CONFIGURATION")
        print("="*80)
        print(f"\nALL_PROP_MARKETS = '{markets_string}'")
        print()


if __name__ == "__main__":
    main()

