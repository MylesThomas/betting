"""
Test NFL Rushing Props Historical Data Availability

Purpose:
--------
Check how far back we can find NFL rushing props (player_rush_yds) data from 
The Odds API. This queries the API directly to understand historical data 
coverage for rushing props specifically.

Context:
--------
User is researching QB first playoff game rushing props and wants to know:
- How far back does historical rushing props data exist?
- Which NFL seasons have rushing props available?
- Can we backtest strategies using historical rushing props?

Market Used:
-----------
player_rush_yds (from config/the-odds-api_config.yaml)

Usage:
------
python testing/test_nfl_rushing_props_availability.py --seasons 2021 2022 2023 2024
python testing/test_nfl_rushing_props_availability.py --start-date 2022-09-01 --end-date 2023-02-28

Author: Thomas Myles
Date: 2026-01-12
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
BASE_URL = CONFIG['odds_api']['base_url']
SPORT_KEY = 'americanfootball_nfl'
MARKET_KEY = 'player_rush_yds'

if not API_KEY:
    print("❌ ERROR: API key not found!")
    print("   Set environment variable: ODDS_API_KEY or THE_ODDS_API_KEY")
    print("   Or add it to .env file")
    sys.exit(1)


def get_season_date_range(season_year):
    """
    Get start and end dates for an NFL season.
    
    Args:
        season_year: Season year as int (e.g., 2023 for 2023-24 season)
    
    Returns:
        tuple: (start_date, end_date) as datetime.date objects
    """
    # NFL season runs September to February (next year)
    start_date = datetime(season_year, 9, 1).date()
    end_date = datetime(season_year + 1, 2, 28).date()
    
    return start_date, end_date


def check_historical_events(date, sport=SPORT_KEY):
    """
    Check if The Odds API has NFL events for a specific date.
    
    Args:
        date: datetime.date object
        sport: Sport key (default: americanfootball_nfl)
    
    Returns:
        dict: {'has_events': bool, 'num_events': int, 'event_ids': list, 'error': str or None}
    """
    # Format date as ISO timestamp at noon UTC
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
            event_ids = [e.get('id') for e in events if e.get('id')]
            return {
                'has_events': len(events) > 0,
                'num_events': len(events),
                'event_ids': event_ids,
                'error': None
            }
        else:
            return {
                'has_events': False,
                'num_events': 0,
                'event_ids': [],
                'error': f"HTTP {response.status_code}"
            }
    except Exception as e:
        return {
            'has_events': False,
            'num_events': 0,
            'event_ids': [],
            'error': str(e)
        }


def check_historical_rushing_props(date, event_id, sport=SPORT_KEY, market=MARKET_KEY):
    """
    Check if The Odds API has rushing props for a specific event.
    
    Args:
        date: datetime.date object
        event_id: Event ID from events endpoint
        sport: Sport key (default: americanfootball_nfl)
        market: Market key (default: player_rush_yds)
    
    Returns:
        dict: {'has_props': bool, 'num_props': int, 'error': str or None}
    """
    timestamp = datetime.combine(date, datetime.min.time()).replace(hour=12).isoformat() + 'Z'
    
    url = f"{BASE_URL}/historical/sports/{sport}/events/{event_id}/odds"
    params = {
        'apiKey': API_KEY,
        'date': timestamp,
        'regions': 'us',
        'markets': market,
        'oddsFormat': 'american',
        'dateFormat': 'iso'
    }
    
    try:
        response = requests.get(url, params=params, timeout=30, verify=False)
        
        if response.status_code == 200:
            data = response.json()
            # Count player props in bookmakers
            num_props = 0
            bookmakers = data.get('data', {}).get('bookmakers', [])
            for bookmaker in bookmakers:
                markets = bookmaker.get('markets', [])
                for market_data in markets:
                    if market_data.get('key') == market:
                        outcomes = market_data.get('outcomes', [])
                        num_props += len(outcomes)
            
            return {
                'has_props': num_props > 0,
                'num_props': num_props,
                'error': None
            }
        elif response.status_code == 422:
            return {
                'has_props': False,
                'num_props': 0,
                'error': '422 - Props not available'
            }
        else:
            return {
                'has_props': False,
                'num_props': 0,
                'error': f"HTTP {response.status_code}"
            }
    except Exception as e:
        return {
            'has_props': False,
            'num_props': 0,
            'error': str(e)
        }


def sample_season_dates(start_date, end_date, num_samples=15):
    """
    Get a sample of dates throughout the NFL season.
    
    Args:
        start_date: Season start date
        end_date: Season end date
        num_samples: Number of dates to sample
    
    Returns:
        list: List of datetime.date objects
    """
    total_days = (end_date - start_date).days
    interval = max(1, total_days // num_samples)
    
    sample_dates = []
    current_date = start_date
    
    while current_date <= end_date and len(sample_dates) < num_samples:
        # Only check dates up to today
        if current_date <= datetime.now().date():
            sample_dates.append(current_date)
        current_date += timedelta(days=interval)
    
    return sample_dates


def analyze_season(season_year, num_samples=15, check_props=True):
    """
    Analyze rushing props data availability for an NFL season.
    
    Args:
        season_year: Season year as int (e.g., 2023)
        num_samples: Number of dates to sample
        check_props: Whether to check individual events for props (slower)
    """
    print(f"\n{'='*80}")
    print(f"🏈 TESTING NFL RUSHING PROPS FOR SEASON: {season_year}-{str(season_year+1)[2:]}")
    print(f"{'='*80}\n")
    
    start_date, end_date = get_season_date_range(season_year)
    print(f"📅 Season range: {start_date} to {end_date}")
    print(f"🎯 Market: {MARKET_KEY} (Rushing Yards)")
    
    # Sample dates throughout the season
    sample_dates = sample_season_dates(start_date, end_date, num_samples)
    print(f"🎲 Testing {len(sample_dates)} sample dates...")
    print()
    
    events_found = 0
    props_found = 0
    dates_with_events = []
    dates_with_props = []
    total_events_checked = 0
    
    for i, date in enumerate(sample_dates, 1):
        print(f"Testing {i}/{len(sample_dates)}: {date}...", end=" ")
        
        # Check if events exist for this date
        events_result = check_historical_events(date)
        
        if events_result['has_events']:
            events_found += 1
            dates_with_events.append(date)
            print(f"✅ {events_result['num_events']} events", end=" ")
            
            # If requested, check first event for rushing props
            if check_props and events_result['event_ids']:
                first_event_id = events_result['event_ids'][0]
                props_result = check_historical_rushing_props(date, first_event_id)
                total_events_checked += 1
                
                if props_result['has_props']:
                    props_found += 1
                    dates_with_props.append(date)
                    print(f"| 🎯 Rush props: ✅ ({props_result['num_props']} props)")
                else:
                    print(f"| 🎯 Rush props: ❌ ({props_result['error']})")
                
                # Rate limit
                time.sleep(0.5)
            else:
                print()
        else:
            print(f"❌ No events ({events_result['error']})")
        
        # Rate limit: don't hammer the API
        time.sleep(0.3)
    
    # Summary
    print(f"\n{'─'*80}")
    print("📊 RESULTS:")
    print(f"{'─'*80}")
    
    events_pct = (events_found / len(sample_dates) * 100) if sample_dates else 0
    
    print(f"NFL games available: {events_found}/{len(sample_dates)} dates ({events_pct:.1f}%)")
    
    if dates_with_events:
        print(f"  First date with events: {dates_with_events[0]}")
        print(f"  Last date with events: {dates_with_events[-1]}")
    
    if check_props:
        props_pct = (props_found / total_events_checked * 100) if total_events_checked else 0
        print(f"\nRushing props available: {props_found}/{total_events_checked} events checked ({props_pct:.1f}%)")
        
        if dates_with_props:
            print(f"  First date with props: {dates_with_props[0]}")
            print(f"  Last date with props: {dates_with_props[-1]}")
    
    print(f"\n{'─'*80}")
    
    if check_props:
        if props_pct > 70:
            print(f"✅ Season {season_year}-{str(season_year+1)[2:]} HAS GOOD rushing props coverage")
        elif props_pct > 30:
            print(f"⚠️  Season {season_year}-{str(season_year+1)[2:]} HAS PARTIAL rushing props coverage")
        else:
            print(f"❌ Season {season_year}-{str(season_year+1)[2:]} HAS POOR rushing props coverage")
    else:
        if events_pct > 70:
            print(f"✅ Season {season_year}-{str(season_year+1)[2:]} HAS GOOD event coverage")
            print(f"   Run with --check-props to verify rushing props availability")
        else:
            print(f"❌ Season {season_year}-{str(season_year+1)[2:]} HAS POOR event coverage")
    
    return {
        'season': f"{season_year}-{str(season_year+1)[2:]}",
        'events_found': events_found,
        'events_pct': events_pct,
        'props_found': props_found if check_props else None,
        'props_pct': props_pct if check_props else None,
        'dates_tested': len(sample_dates),
        'events_checked': total_events_checked if check_props else None
    }


def analyze_date_range(start_date, end_date, check_props=True):
    """
    Analyze rushing props availability for a custom date range.
    
    Args:
        start_date: Start date as datetime.date
        end_date: End date as datetime.date
        check_props: Whether to check individual events for props
    """
    print(f"\n{'='*80}")
    print(f"🏈 TESTING NFL RUSHING PROPS FOR DATE RANGE")
    print(f"{'='*80}\n")
    
    print(f"📅 Date range: {start_date} to {end_date}")
    print(f"🎯 Market: {MARKET_KEY} (Rushing Yards)")
    
    # Get all dates in range
    all_dates = []
    current_date = start_date
    while current_date <= end_date:
        if current_date <= datetime.now().date():
            all_dates.append(current_date)
        current_date += timedelta(days=1)
    
    print(f"🎲 Testing {len(all_dates)} dates...")
    print()
    
    events_found = 0
    props_found = 0
    dates_with_events = []
    dates_with_props = []
    total_events_checked = 0
    
    for i, date in enumerate(all_dates, 1):
        # Skip dates that are unlikely to have NFL games (not Thu/Sun/Mon)
        weekday = date.weekday()
        # if weekday not in [3, 6, 0]:  # Thu, Sun, Mon
        #     continue
        
        if i % 10 == 0:
            print(f"Progress: {i}/{len(all_dates)} dates tested...")
        
        # Check if events exist for this date
        events_result = check_historical_events(date)
        
        if events_result['has_events']:
            events_found += 1
            dates_with_events.append(date)
            print(f"  {date}: ✅ {events_result['num_events']} events", end=" ")
            
            # If requested, check first event for rushing props
            if check_props and events_result['event_ids']:
                first_event_id = events_result['event_ids'][0]
                props_result = check_historical_rushing_props(date, first_event_id)
                total_events_checked += 1
                
                if props_result['has_props']:
                    props_found += 1
                    dates_with_props.append(date)
                    print(f"| Rush props: ✅")
                else:
                    print(f"| Rush props: ❌")
                
                time.sleep(0.5)
            else:
                print()
        
        time.sleep(0.3)
    
    # Summary
    print(f"\n{'─'*80}")
    print("📊 RESULTS:")
    print(f"{'─'*80}")
    
    events_pct = (events_found / len(all_dates) * 100) if all_dates else 0
    
    print(f"NFL games available: {events_found}/{len(all_dates)} dates ({events_pct:.1f}%)")
    
    if dates_with_events:
        print(f"  First date with events: {dates_with_events[0]}")
        print(f"  Last date with events: {dates_with_events[-1]}")
    
    if check_props and total_events_checked > 0:
        props_pct = (props_found / total_events_checked * 100)
        print(f"\nRushing props available: {props_found}/{total_events_checked} events ({props_pct:.1f}%)")
        
        if dates_with_props:
            print(f"  First date with props: {dates_with_props[0]}")
            print(f"  Last date with props: {dates_with_props[-1]}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Test NFL rushing props historical data availability'
    )
    parser.add_argument(
        '--seasons',
        nargs='+',
        type=int,
        help='Season years to test (e.g., 2021 2022 2023). Will test Sept-Feb of each season.'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date for custom range (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date for custom range (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=15,
        help='Number of dates to sample per season (default: 15)'
    )
    parser.add_argument(
        '--no-check-props',
        action='store_true',
        help='Skip checking individual events for props (faster, less API usage)'
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
    
    check_props = not args.no_check_props
    
    print("="*80)
    print("🔍 NFL RUSHING PROPS HISTORICAL DATA AVAILABILITY TEST")
    print("="*80)
    print(f"\n🎯 Market: {MARKET_KEY} (from config/the-odds-api_config.yaml)")
    
    # Custom date range mode
    if args.start_date and args.end_date:
        try:
            start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date()
            end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date()
            
            print(f"\n⚠️  Custom date range mode")
            print(f"   This will check every date in the range")
            if check_props:
                print(f"   Plus check events for rushing props")
            print()
            
            analyze_date_range(start_date, end_date, check_props)
            
        except ValueError as e:
            print(f"❌ ERROR: Invalid date format. Use YYYY-MM-DD")
            sys.exit(1)
    
    # Season mode (default)
    else:
        seasons = args.seasons or [2021, 2022, 2023, 2024]
        
        print(f"\n⚠️  Season testing mode")
        print(f"   Testing {len(seasons)} season(s) with {args.samples} samples each")
        estimated_calls = len(seasons) * args.samples
        if check_props:
            estimated_calls *= 2  # Double for props checks
        print(f"   Estimated API calls: {estimated_calls}")
        print()
        
        results = []
        for season_year in seasons:
            result = analyze_season(season_year, args.samples, check_props)
            results.append(result)
        
        # Final summary
        print(f"\n{'='*80}")
        print("📊 FINAL SUMMARY")
        print(f"{'='*80}\n")
        
        for r in results:
            if check_props and r['props_pct'] is not None:
                print(f"{r['season']}: {r['events_found']}/{r['dates_tested']} dates with events "
                      f"({r['events_pct']:.1f}%), {r['props_found']}/{r['events_checked']} "
                      f"with rushing props ({r['props_pct']:.1f}%)")
            else:
                print(f"{r['season']}: {r['events_found']}/{r['dates_tested']} dates "
                      f"({r['events_pct']:.1f}%) have events")
        
        if check_props:
            print(f"\n💡 Rushing props coverage indicates how far back {MARKET_KEY} data exists.")
        else:
            print(f"\n💡 Run with --check-props to verify {MARKET_KEY} availability on events.")
        print()


if __name__ == "__main__":
    main()

