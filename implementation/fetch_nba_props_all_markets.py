#!/usr/bin/env python3
"""
Fetch today's NBA props for all markets from The Odds API.

Saves raw API data to: data/01_input/the-odds-api/nba/live/
Output plays go to: data/04_output/

USAGE:
    export ODDS_API_KEY="your_key"
    python implementation/fetch_nba_props_all_markets.py
    
    # Then find opportunities:
    python implementation/find_nba_points_overs.py

Author: Myles Thomas
Date: 2025-12-05
"""

import os
import sys
import ssl
import json
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd
import requests
import urllib3

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Fix SSL certificate issues on macOS
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Monkey-patch requests to disable SSL verification globally
original_request = requests.Session.request
def patched_request(self, *args, **kwargs):
    kwargs['verify'] = False
    return original_request(self, *args, **kwargs)
requests.Session.request = patched_request

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
API_KEY = os.environ.get('ODDS_API_KEY', '')
SPORT = 'basketball_nba'
REGIONS = 'us'
ODDS_FORMAT = 'american'

# All markets we want to fetch
TARGET_MARKETS = [
    'player_points',
    'player_rebounds', 
    'player_assists',
    'player_threes',
    'player_points_rebounds_assists',
]

# Output paths
RAW_OUTPUT_DIR = PROJECT_ROOT / 'data' / '01_input' / 'the-odds-api' / 'nba' / 'live'
PLAYS_OUTPUT_DIR = PROJECT_ROOT / 'data' / '04_output'


# =============================================================================
# API FETCHING
# =============================================================================

def fetch_todays_events(api_key: str) -> list:
    """Fetch today's NBA events."""
    
    url = f'https://api.the-odds-api.com/v4/sports/{SPORT}/events/'
    
    params = {
        'apiKey': api_key,
    }
    
    response = requests.get(url, params=params, verify=False)
    response.raise_for_status()
    
    remaining = response.headers.get('x-requests-remaining', '?')
    events = response.json()
    print(f"   📡 Found {len(events)} events ({remaining} API calls remaining)")
    
    return events


def fetch_event_props(api_key: str, event_id: str, markets: str) -> dict:
    """Fetch props for a specific event."""
    
    url = f'https://api.the-odds-api.com/v4/sports/{SPORT}/events/{event_id}/odds'
    
    params = {
        'apiKey': api_key,
        'regions': REGIONS,
        'markets': markets,
        'oddsFormat': ODDS_FORMAT,
    }
    
    response = requests.get(url, params=params, verify=False)
    response.raise_for_status()
    
    return response.json()


def parse_event_props(data: dict, market: str) -> list:
    """Parse event-level props response into flat prop records."""
    
    props = []
    
    if not data:
        return props
    
    game_id = data.get('id', '')
    commence_time = data.get('commence_time', '')
    home_team = data.get('home_team', '')
    away_team = data.get('away_team', '')
    game_str = f"{away_team} @ {home_team}"
    
    for bookmaker in data.get('bookmakers', []):
        bookmaker_key = bookmaker['key']
        
        for mkt in bookmaker.get('markets', []):
            if mkt['key'] == market:
                # Group outcomes by player+line to get over/under pairs
                outcomes_by_player_line = {}
                
                for outcome in mkt.get('outcomes', []):
                    player = outcome.get('description', '')
                    line = outcome.get('point', 0)
                    side = outcome.get('name', '')  # 'Over' or 'Under'
                    odds = outcome.get('price', 0)
                    
                    key = (player, line)
                    if key not in outcomes_by_player_line:
                        outcomes_by_player_line[key] = {}
                    outcomes_by_player_line[key][side] = odds
                
                # Create records with both over and under odds
                for (player, line), sides in outcomes_by_player_line.items():
                    if 'Over' in sides and 'Under' in sides:
                        props.append({
                            'game_id': game_id,
                            'game': game_str,
                            'commence_time': commence_time,
                            'home_team': home_team,
                            'away_team': away_team,
                            'bookmaker': bookmaker_key,
                            'market': market,
                            'player': player,
                            'line': line,
                            'over_odds': sides['Over'],
                            'under_odds': sides['Under'],
                        })
    
    return props


def fetch_all_markets(api_key: str) -> pd.DataFrame:
    """Fetch all target markets and combine into one DataFrame."""
    
    print("\n📡 Fetching props from The Odds API...")
    print(f"   Markets: {', '.join(TARGET_MARKETS)}")
    print()
    
    # Step 1: Get today's events
    try:
        events = fetch_todays_events(api_key)
    except requests.exceptions.RequestException as e:
        print(f"   ❌ Error fetching events: {e}")
        return pd.DataFrame()
    
    if not events:
        print("   ⚠️  No events found for today")
        return pd.DataFrame()
    
    # Step 2: Fetch props for each event (all markets at once to save API calls)
    markets_str = ','.join(TARGET_MARKETS)
    all_props = []
    
    for i, event in enumerate(events):
        event_id = event['id']
        home = event.get('home_team', 'TBD')
        away = event.get('away_team', 'TBD')
        game_str = f"{away} @ {home}"
        
        try:
            data = fetch_event_props(api_key, event_id, markets_str)
            
            # Parse the response
            for market in TARGET_MARKETS:
                props = parse_event_props(data, market)
                all_props.extend(props)
            
            remaining = f"(event {i+1}/{len(events)})"
            print(f"   ✅ {game_str} {remaining}")
            
        except requests.exceptions.RequestException as e:
            print(f"   ❌ Error fetching {game_str}: {e}")
    
    if not all_props:
        print("\n⚠️  No props found. There may be no games today.")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_props)
    df['fetch_time'] = datetime.now(timezone.utc).isoformat()
    
    print(f"\n✅ Fetched {len(df):,} prop lines")
    print(f"   Games: {df['game_id'].nunique()}")
    print(f"   Players: {df['player'].nunique()}")
    print(f"   Bookmakers: {df['bookmaker'].nunique()}")
    print(f"   Markets: {df['market'].nunique()}")
    
    return df


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Fetch NBA props from The Odds API')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be fetched without calling API')
    args = parser.parse_args()
    
    print("=" * 80)
    print("🏀 FETCH NBA PROPS - ALL MARKETS")
    print("=" * 80)
    
    if args.dry_run:
        print("\n🔍 DRY RUN - would fetch:")
        for market in TARGET_MARKETS:
            print(f"   - {market}")
        print(f"\nOutput to: {RAW_OUTPUT_DIR}")
        return
    
    # Check API key
    if not API_KEY:
        print("\n❌ Error: ODDS_API_KEY environment variable not set")
        print("   export ODDS_API_KEY='your_key_here'")
        sys.exit(1)
    
    # Fetch all markets
    df = fetch_all_markets(API_KEY)
    
    if len(df) == 0:
        return
    
    # Save raw data to 01_input
    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    today_str = datetime.now().strftime('%Y-%m-%d')
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save with timestamp (for historical tracking)
    timestamped_file = RAW_OUTPUT_DIR / f'props_{timestamp_str}_all_markets.csv'
    df.to_csv(timestamped_file, index=False)
    print(f"\n💾 Saved raw data to: {timestamped_file}")
    
    # Also save as "latest" for easy access by find script
    latest_file = RAW_OUTPUT_DIR / f'props_latest.csv'
    df.to_csv(latest_file, index=False)
    print(f"💾 Saved as latest: {latest_file}")
    
    # Summary by market
    print("\n📊 Props by Market:")
    for market, group in df.groupby('market'):
        market_short = market.replace('player_', '')
        print(f"   {market_short:30} {len(group):5} lines, {group['player'].nunique():3} players")
    
    print("\n✅ Done! Now run:")
    print(f"   python implementation/find_nba_points_overs.py --data-dir {RAW_OUTPUT_DIR}")


if __name__ == '__main__':
    main()

