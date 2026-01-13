"""
Quick Test: Check if player_rush_yds data exists for known NFL playoff dates

Tests specific playoff game dates to see if The Odds API has historical 
player_rush_yds props available.

Usage:
------
python testing/quick_test_nfl_rush_props.py

Author: Thomas Myles
Date: 2026-01-12
"""

import requests
from datetime import datetime
import os
from dotenv import load_dotenv
import ssl
import urllib3
from pathlib import Path
import sys

# SSL fix
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load environment variables
load_dotenv()

API_KEY = os.getenv('ODDS_API_KEY') or os.getenv('THE_ODDS_API_KEY')
BASE_URL = 'https://api.the-odds-api.com/v4'
SPORT_KEY = 'americanfootball_nfl'
MARKET_KEY = 'player_rush_yds'

# Snapshot time (UTC) - 12pm ET = 5 PM UTC
EVENT_LIST_HOUR = 17  # 12pm ET (5 PM UTC)

if not API_KEY:
    print("❌ ERROR: API key not found!")
    sys.exit(1)

# Known playoff dates to test
TEST_DATES = [
    ('2024-01-13', '2023-24 Wild Card Saturday'),
    ('2024-01-14', '2023-24 Wild Card Sunday'),
    ('2024-01-20', '2023-24 Divisional Saturday'),
    ('2024-02-11', '2023-24 Super Bowl'),
    ('2025-01-11', '2024-25 Wild Card Saturday'),
    ('2025-01-12', '2024-25 Wild Card Sunday'),
]

def check_historical_data(date_str, description):
    """Check if events and props exist for a date"""
    print(f"\n{'='*80}")
    print(f"📅 Testing: {date_str} ({description})")
    print(f"{'='*80}")
    
    # Convert to timestamp at 12pm ET (5 PM UTC)
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    timestamp = date_obj.replace(hour=EVENT_LIST_HOUR, minute=0, second=0).isoformat() + 'Z'
    
    # Step 1: Check for events
    print(f"  1️⃣  Checking for events...")
    events_url = f"{BASE_URL}/historical/sports/{SPORT_KEY}/events"
    events_params = {
        'apiKey': API_KEY,
        'date': timestamp,
        'dateFormat': 'iso'
    }
    
    try:
        response = requests.get(events_url, params=events_params, verify=False, timeout=30)
        credits = int(float(response.headers.get('x-requests-remaining', 0)))
        
        if response.status_code == 200:
            events = response.json().get('data', [])
            if events:
                print(f"     ✅ Found {len(events)} games")
                games_list = [f"{e.get('away_team')} @ {e.get('home_team')}" for e in events[:3]]
                print(f"     Games: {games_list}")
                
                # Step 2: Check first event for props
                first_event = events[0]
                event_id = first_event.get('id')
                print(f"\n  2️⃣  Checking game for {MARKET_KEY} props...")
                print(f"     Game: {first_event.get('away_team')} @ {first_event.get('home_team')}")
                
                props_url = f"{BASE_URL}/historical/sports/{SPORT_KEY}/events/{event_id}/odds"
                props_params = {
                    'apiKey': API_KEY,
                    'date': timestamp,
                    'regions': 'us',
                    'markets': MARKET_KEY,
                    'oddsFormat': 'american',
                    'dateFormat': 'iso'
                }
                
                props_response = requests.get(props_url, params=props_params, verify=False, timeout=30)
                credits = int(float(props_response.headers.get('x-requests-remaining', 0)))
                
                if props_response.status_code == 200:
                    props_data = props_response.json().get('data', {})
                    bookmakers = props_data.get('bookmakers', [])
                    
                    if bookmakers:
                        # Count props
                        num_props = 0
                        for bm in bookmakers:
                            for market in bm.get('markets', []):
                                if market.get('key') == MARKET_KEY:
                                    num_props += len(market.get('outcomes', []))
                        
                        if num_props > 0:
                            print(f"     ✅ FOUND {num_props} rushing props!")
                            print(f"     Bookmakers: {[bm.get('key') for bm in bookmakers]}")
                            
                            # Show sample props
                            for bm in bookmakers[:1]:
                                for market in bm.get('markets', []):
                                    if market.get('key') == MARKET_KEY:
                                        outcomes = market.get('outcomes', [])
                                        for outcome in outcomes[:3]:
                                            player = outcome.get('description')
                                            line = outcome.get('point')
                                            print(f"        • {player}: {line} yards")
                            return True
                        else:
                            print(f"     ❌ No {MARKET_KEY} props found (bookmakers exist but no rushing props)")
                    else:
                        print(f"     ❌ No bookmakers with props")
                elif props_response.status_code == 422:
                    print(f"     ❌ HTTP 422 - Props not available for this event")
                else:
                    print(f"     ❌ HTTP {props_response.status_code}")
                
                print(f"     💰 Credits remaining: {credits:,}")
                return False
            else:
                print(f"     ❌ No events found")
                return False
        elif response.status_code == 422:
            print(f"     ❌ HTTP 422 - No historical data available")
            return False
        else:
            print(f"     ❌ HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"     ❌ Error: {e}")
        return False


def main():
    print("="*80)
    print("🔍 QUICK TEST: NFL Rushing Props Historical Availability")
    print("="*80)
    print(f"Market: {MARKET_KEY}")
    print(f"Testing {len(TEST_DATES)} known playoff dates...")
    
    results = []
    for date_str, description in TEST_DATES:
        has_props = check_historical_data(date_str, description)
        results.append((date_str, description, has_props))
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 SUMMARY")
    print(f"{'='*80}")
    
    dates_with_props = [r for r in results if r[2]]
    
    if dates_with_props:
        print(f"\n✅ Dates WITH {MARKET_KEY} props:")
        for date_str, desc, _ in dates_with_props:
            print(f"   • {date_str} - {desc}")
    else:
        print(f"\n❌ NO dates found with {MARKET_KEY} props")
    
    dates_without_props = [r for r in results if not r[2]]
    if dates_without_props:
        print(f"\n❌ Dates WITHOUT {MARKET_KEY} props:")
        for date_str, desc, _ in dates_without_props:
            print(f"   • {date_str} - {desc}")
    
    print(f"\n{'='*80}")
    if dates_with_props:
        print("✅ GOOD NEWS: Historical rushing props data EXISTS!")
        print(f"   Found on {len(dates_with_props)}/{len(TEST_DATES)} dates tested")
    else:
        print("❌ BAD NEWS: Historical rushing props NOT available")
        print("   The Odds API may not have historical player props in their API,")
        print("   or it may only be available for very recent dates (not playoffs).")
    print()


if __name__ == "__main__":
    main()

