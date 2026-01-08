"""
Test The Odds API Historical Data Availability - Events AND Props

Purpose:
--------
Query The Odds API directly to check:
1. Game events availability (for game lines: spreads, totals, ML)
2. Player props availability for those events

Context:
--------
- Game lines go back further in history than player props
- This script tests BOTH to understand what's available

Usage:
------
python testing/test_odds_api_historical_data_availability.py --seasons 2021-22 2023-24

Author: Thomas Myles
Date: 2026-01-08
"""

import argparse
import requests
from datetime import datetime, timedelta
from pathlib import Path
import sys
import time
import os
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from config_loader import get_config

# Load config
CONFIG = get_config()

# API Configuration
def get_api_key():
    """Try multiple methods to get API key"""
    # Method 1: Environment variable
    api_key = os.getenv('ODDS_API_KEY') or os.getenv('THE_ODDS_API_KEY')
    if api_key:
        return api_key
    
    # Method 2: Check if it's in a .env file
    env_file = Path(__file__).parent.parent / '.env'
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if line.startswith('ODDS_API_KEY=') or line.startswith('THE_ODDS_API_KEY='):
                    return line.split('=', 1)[1].strip()
    
    return None

API_KEY = get_api_key()
BASE_URL = CONFIG.get('odds_api', {}).get('base_url', 'https://api.the-odds-api.com/v4')

if not API_KEY:
    print("❌ ERROR: API key not found!")
    print("   Set environment variable: ODDS_API_KEY or THE_ODDS_API_KEY")
    print("   Or add it to .env file")
    sys.exit(1)


def get_season_date_range(season):
    """Get start and end dates for an NBA season."""
    start_year = int(season.split('-')[0])
    end_year = int('20' + season.split('-')[1])
    
    # NBA season typically runs October to April
    start_date = datetime(start_year, 10, 1).date()
    end_date = datetime(end_year, 6, 30).date()
    
    return start_date, end_date


def check_historical_events(date, sport='basketball_nba'):
    """
    Check if The Odds API has events for a specific date.
    
    Returns:
        dict: {'has_events': bool, 'num_events': int, 'events': list, 'error': str or None}
    """
    timestamp = datetime.combine(date, datetime.min.time()).replace(hour=12).isoformat() + 'Z'
    
    url = f"{BASE_URL}/historical/sports/{sport}/events"
    params = {
        'apiKey': API_KEY,
        'date': timestamp,
        'dateFormat': 'iso'
    }
    
    try:
        response = requests.get(url, params=params, timeout=30, verify=False)
        
        if response.status_code == 200:
            data = response.json()
            events = data.get('data', [])
            return {
                'has_events': len(events) > 0,
                'num_events': len(events),
                'events': events,
                'error': None
            }
        else:
            return {
                'has_events': False,
                'num_events': 0,
                'events': [],
                'error': f"HTTP {response.status_code}"
            }
    except Exception as e:
        return {
            'has_events': False,
            'num_events': 0,
            'events': [],
            'error': str(e)
        }


def check_historical_props(date, event_id, sport='basketball_nba'):
    """
    Check if The Odds API has player props for a specific event.
    
    Returns:
        dict: {'has_props': bool, 'num_bookmakers': int, 'error': str or None}
    """
    timestamp = datetime.combine(date, datetime.min.time()).replace(hour=12).isoformat() + 'Z'
    
    url = f"{BASE_URL}/historical/sports/{sport}/events/{event_id}/odds"
    params = {
        'apiKey': API_KEY,
        'date': timestamp,
        'regions': 'us',
        'markets': 'player_points',
        'oddsFormat': 'american',
        'dateFormat': 'iso'
    }
    
    try:
        response = requests.get(url, params=params, timeout=30, verify=False)
        
        if response.status_code == 200:
            data = response.json()
            bookmakers = data.get('data', {}).get('bookmakers', [])
            return {
                'has_props': len(bookmakers) > 0,
                'num_bookmakers': len(bookmakers),
                'error': None
            }
        elif response.status_code == 422:
            return {'has_props': False, 'num_bookmakers': 0, 'error': '422 - Not available'}
        else:
            return {'has_props': False, 'num_bookmakers': 0, 'error': f"HTTP {response.status_code}"}
    except Exception as e:
        return {'has_props': False, 'num_bookmakers': 0, 'error': str(e)}


def sample_season_dates(start_date, end_date, num_samples=10):
    """Get a sample of dates throughout the season."""
    total_days = (end_date - start_date).days
    interval = max(1, total_days // num_samples)
    
    sample_dates = []
    current_date = start_date
    
    while current_date <= end_date and len(sample_dates) < num_samples:
        if current_date <= datetime.now().date():
            sample_dates.append(current_date)
        current_date += timedelta(days=interval)
    
    return sample_dates


def analyze_season(season, num_samples=10):
    """Analyze API data availability for a season."""
    print(f"\n{'='*80}")
    print(f"🏀 TESTING THE ODDS API FOR SEASON: {season}")
    print(f"{'='*80}\n")
    
    start_date, end_date = get_season_date_range(season)
    print(f"📅 Season range: {start_date} to {end_date}")
    
    # Sample dates throughout the season
    sample_dates = sample_season_dates(start_date, end_date, num_samples)
    print(f"🎲 Testing {len(sample_dates)} sample dates...")
    print()
    
    events_found = 0
    props_found = 0
    dates_with_events = []
    dates_with_props = []
    
    for i, date in enumerate(sample_dates, 1):
        print(f"Testing {i}/{len(sample_dates)}: {date}...", end=" ")
        
        # Check if events exist for this date
        events_result = check_historical_events(date)
        
        if events_result['has_events']:
            events_found += 1
            dates_with_events.append(date)
            print(f"✅ {events_result['num_events']} events", end=" ")
            
            # Try to check props for the first event
            if events_result['events']:
                first_event = events_result['events'][0]
                event_id = first_event.get('id')
                
                if event_id:
                    props_result = check_historical_props(date, event_id)
                    
                    if props_result['has_props']:
                        props_found += 1
                        dates_with_props.append(date)
                        print(f"| ✅ Props available ({props_result['num_bookmakers']} books)")
                    else:
                        print(f"| ❌ No props ({props_result['error']})")
                else:
                    print(f"| ⚠️  No event ID")
            else:
                print()
        else:
            print(f"❌ No events ({events_result['error']})")
        
        # Rate limit: don't hammer the API
        time.sleep(0.5)
    
    # Summary
    print(f"\n{'─'*80}")
    print("📊 RESULTS:")
    print(f"{'─'*80}")
    
    events_pct = (events_found / len(sample_dates) * 100) if sample_dates else 0
    props_pct = (props_found / len(sample_dates) * 100) if sample_dates else 0
    
    print(f"\n🎲 Game Events (for game lines):")
    print(f"   Available: {events_found}/{len(sample_dates)} dates ({events_pct:.1f}%)")
    if dates_with_events:
        print(f"   First date: {dates_with_events[0]}")
        print(f"   Last date: {dates_with_events[-1]}")
    
    print(f"\n📊 Player Props:")
    print(f"   Available: {props_found}/{len(sample_dates)} dates ({props_pct:.1f}%)")
    if dates_with_props:
        print(f"   First date: {dates_with_props[0]}")
        print(f"   Last date: {dates_with_props[-1]}")
    
    print(f"\n{'─'*80}")
    
    if events_pct > 70:
        print(f"✅ Game lines: GOOD coverage for {season}")
    elif events_pct > 30:
        print(f"⚠️  Game lines: PARTIAL coverage for {season}")
    else:
        print(f"❌ Game lines: POOR coverage for {season}")
    
    if props_pct > 70:
        print(f"✅ Player props: GOOD coverage for {season}")
    elif props_pct > 30:
        print(f"⚠️  Player props: PARTIAL coverage for {season}")
    else:
        print(f"❌ Player props: POOR/NO coverage for {season}")
    
    return {
        'season': season,
        'events_found': events_found,
        'events_pct': events_pct,
        'props_found': props_found,
        'props_pct': props_pct,
        'dates_tested': len(sample_dates)
    }


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Test The Odds API historical data availability'
    )
    parser.add_argument(
        '--seasons',
        nargs='+',
        default=['2021-22', '2022-23', '2023-24', '2024-25', '2025-26'],
        help='Seasons to test (e.g., 2021-22 2022-23)'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=10,
        help='Number of dates to sample per season (default: 10)'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='The Odds API key (or set ODDS_API_KEY environment variable)'
    )
    args = parser.parse_args()
    
    # Override global API_KEY if provided
    global API_KEY
    if args.api_key:
        API_KEY = args.api_key
    
    if not API_KEY:
        print("❌ ERROR: API key not found!")
        print("   Provide with --api-key or set environment variable: ODDS_API_KEY")
        sys.exit(1)
    
    print("="*80)
    print("🔍 THE ODDS API HISTORICAL DATA AVAILABILITY TEST")
    print("   Testing: Game Events AND Player Props")
    print("="*80)
    print(f"\n⚠️  This will make API calls to The Odds API")
    print(f"   Testing {len(args.seasons)} season(s) with {args.samples} samples each")
    print(f"   Estimated API calls: ~{len(args.seasons) * args.samples * 2}")
    print()
    
    results = []
    for season in args.seasons:
        result = analyze_season(season, args.samples)
        results.append(result)
    
    # Final summary
    print(f"\n{'='*80}")
    print("📊 FINAL SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"{'Season':<12} {'Events':<15} {'Props':<15} {'Status'}")
    print(f"{'-'*12} {'-'*15} {'-'*15} {'-'*30}")
    
    for r in results:
        events_str = f"{r['events_found']}/{r['dates_tested']} ({r['events_pct']:.0f}%)"
        props_str = f"{r['props_found']}/{r['dates_tested']} ({r['props_pct']:.0f}%)"
        
        if r['props_pct'] > 50:
            status = "✅ Props + Events"
        elif r['events_pct'] > 70:
            status = "🎲 Events only (no props)"
        else:
            status = "❌ Insufficient"
        
        print(f"{r['season']:<12} {events_str:<15} {props_str:<15} {status}")
    
    print(f"\n{'='*80}")
    print("💡 RECOMMENDATIONS:")
    print(f"{'='*80}")
    
    props_seasons = [r for r in results if r['props_pct'] > 50]
    events_only_seasons = [r for r in results if r['events_pct'] > 70 and r['props_pct'] < 50]
    
    if props_seasons:
        print(f"\n✅ PLAYER PROPS backtesting viable for:")
        for r in props_seasons:
            print(f"   - {r['season']}")
    
    if events_only_seasons:
        print(f"\n🎲 GAME LINES backtesting viable for (but NO props):")
        for r in events_only_seasons:
            print(f"   - {r['season']}")
    
    print()


if __name__ == "__main__":
    main()
