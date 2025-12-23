"""
Test The Odds API Historical Data Coverage (2012-2021)

Quick test to see which NFL seasons have historical betting line data available
from The Odds API before spending API credits on full season fetches.

Tests Week 1 Thursday game from each season to verify data availability.

Usage:
    python scripts/test_odds_api_historical_coverage.py
"""

import requests
import os
from datetime import datetime
from dotenv import load_dotenv
import ssl
import urllib3

# SSL fix
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

API_KEY = os.getenv('ODDS_API_KEY')
BASE_URL = 'https://api.the-odds-api.com/v4'
SPORT = 'americanfootball_nfl'

# Week 1 Thursday games for each season (2012-2021)
# Format: (year, date, expected_matchup)
TEST_DATES = [
    (2012, '2012-09-05', 'NYG vs DAL'),  # Week 1 opener
    (2013, '2013-09-05', 'BAL vs DEN'),  # Week 1 Thursday
    (2014, '2014-09-04', 'GB vs SEA'),   # Week 1 Thursday
    (2015, '2015-09-10', 'NE vs PIT'),   # Week 1 Thursday
    (2016, '2016-09-08', 'CAR vs DEN'),  # Week 1 Thursday
    (2017, '2017-09-07', 'NE vs KC'),    # Week 1 Thursday
    (2018, '2018-09-06', 'ATL vs PHI'),  # Week 1 Thursday
    (2019, '2019-09-05', 'GB vs CHI'),   # Week 1 Thursday
    (2020, '2020-09-10', 'HOU vs KC'),   # Week 1 Thursday
    (2021, '2021-09-09', 'DAL vs TB'),   # Week 1 Thursday
]

def check_api_key():
    """Verify API key is loaded"""
    if not API_KEY or API_KEY == 'your_api_key_here':
        print("❌ ERROR: No valid API key found!")
        print("Make sure ODDS_API_KEY is set in your .env file")
        return False
    return True

def test_historical_date(year, date_str, expected_game):
    """
    Test if historical data exists for a specific date.
    
    Args:
        year: Season year
        date_str: Date in YYYY-MM-DD format
        expected_game: Expected matchup description
    
    Returns:
        dict with status and info
    """
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    timestamp = date_obj.replace(hour=17, minute=0, second=0).isoformat() + 'Z'
    
    url = f"{BASE_URL}/historical/sports/{SPORT}/events"
    
    params = {
        'apiKey': API_KEY,
        'date': timestamp,
        'dateFormat': 'iso'
    }
    
    try:
        response = requests.get(url, params=params, verify=False, timeout=10)
        
        # Get API usage
        credits_remaining = response.headers.get('x-requests-remaining', 'unknown')
        credits_used = response.headers.get('x-requests-last', '1')
        
        if response.status_code == 422:
            # No data available
            return {
                'year': year,
                'status': 'NO_DATA',
                'message': 'API returned 422 (no historical data)',
                'credits_used': 1,
                'credits_remaining': credits_remaining
            }
        
        response.raise_for_status()
        data = response.json()
        events = data.get('data', [])
        
        if not events:
            return {
                'year': year,
                'status': 'EMPTY',
                'message': 'API call succeeded but no events found',
                'credits_used': credits_used,
                'credits_remaining': credits_remaining
            }
        
        # Found events!
        return {
            'year': year,
            'status': 'SUCCESS',
            'message': f'Found {len(events)} event(s)',
            'events': events,
            'expected_game': expected_game,
            'credits_used': credits_used,
            'credits_remaining': credits_remaining
        }
        
    except requests.exceptions.Timeout:
        return {
            'year': year,
            'status': 'TIMEOUT',
            'message': 'Request timed out',
            'credits_used': 0,
            'credits_remaining': 'unknown'
        }
    except requests.exceptions.RequestException as e:
        return {
            'year': year,
            'status': 'ERROR',
            'message': str(e),
            'credits_used': 0,
            'credits_remaining': 'unknown'
        }

def main():
    """Run historical coverage test"""
    print("=" * 80)
    print("THE ODDS API HISTORICAL COVERAGE TEST")
    print("=" * 80)
    print("\nTesting NFL historical data availability (2012-2021)")
    print("Testing Week 1 Thursday game from each season\n")
    
    if not check_api_key():
        return
    
    print(f"API Key loaded: {API_KEY[:8]}...")
    print(f"\n{'Year':<8} {'Date':<12} {'Expected Game':<20} {'Status':<12} {'Message'}")
    print("-" * 80)
    
    results = []
    total_credits = 0
    
    for year, date_str, expected_game in TEST_DATES:
        result = test_historical_date(year, date_str, expected_game)
        results.append(result)
        
        # Update credits
        try:
            total_credits += int(result['credits_used'])
        except:
            pass
        
        # Print result
        status_emoji = {
            'SUCCESS': '✅',
            'NO_DATA': '❌',
            'EMPTY': '⚠️',
            'TIMEOUT': '⏱️',
            'ERROR': '❌'
        }.get(result['status'], '?')
        
        print(f"{year:<8} {date_str:<12} {expected_game:<20} {status_emoji} {result['status']:<10} {result['message']}")
        
        # Show first event if found
        if result['status'] == 'SUCCESS' and result.get('events'):
            first_event = result['events'][0]
            away = first_event.get('away_team', 'Unknown')
            home = first_event.get('home_team', 'Unknown')
            print(f"         → First game found: {away} @ {home}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
    no_data_count = sum(1 for r in results if r['status'] == 'NO_DATA')
    
    print(f"\nTotal seasons tested: {len(results)}")
    print(f"✅ Data available: {success_count}")
    print(f"❌ No data: {no_data_count}")
    print(f"💰 Credits used: {total_credits}")
    
    if results and results[0].get('credits_remaining') != 'unknown':
        print(f"💳 Credits remaining: {results[0]['credits_remaining']}")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    available_years = [r['year'] for r in results if r['status'] == 'SUCCESS']
    
    if available_years:
        print(f"\n✅ Historical data IS available for these seasons:")
        print(f"   {', '.join(map(str, available_years))}")
        print(f"\n💡 You can fetch full season data for these years")
        print(f"   Estimated cost per season: ~3,000 credits (~$150)")
    else:
        print(f"\n❌ No historical data found for 2012-2021")
        print(f"   The Odds API may only have data from 2022 onwards")
    
    unavailable_years = [r['year'] for r in results if r['status'] == 'NO_DATA']
    if unavailable_years:
        print(f"\n❌ No data for these seasons:")
        print(f"   {', '.join(map(str, unavailable_years))}")
        print(f"   These seasons cannot be backtested with The Odds API")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()

