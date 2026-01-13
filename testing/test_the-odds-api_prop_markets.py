"""
Discover Available Prop Markets from The Odds API

Context:
--------
This script discovers what prop markets are actually available from The Odds API
for each sport (NFL, NBA, NCAAF, NCAAB) by testing all possible markets.
It then updates the config file with the discovered markets.

Goal:
-----
- Test all possible prop markets against the API
- Discover which markets are actually available for each sport
- Update config/the-odds-api_config.yaml with discovered markets
- Display market types, bookmakers, and sample data
- Notebook-friendly: can copy/paste sections
- Fail-fast: no defensive checks, let it break if data is wrong

Usage:
------
# In terminal:
python testing/test_the-odds-api_prop_markets.py --sport nfl
python testing/test_the-odds-api_prop_markets.py --sport nba
python testing/test_the-odds-api_prop_markets.py --sport ncaaf
python testing/test_the-odds-api_prop_markets.py --sport ncaab

# Update config after discovery:
python testing/test_the-odds-api_prop_markets.py --sport nfl --update-config

# In notebook (set SPORT variable manually):
SPORT = 'nfl'  # or 'nba', 'ncaaf', 'ncaab'
UPDATE_CONFIG = True  # Set to True to update config file
# Then copy/paste sections

Created: 2026-01-12
Author: Thomas Myles
"""

import os
import requests
import pandas as pd
from datetime import datetime
import json
import yaml
import urllib3
import argparse

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# For notebook usage - can be overridden
SPORT = None
UPDATE_CONFIG = False

# =============================================================================
# CONFIGURATION
# =============================================================================

def find_repo_root():
    """Find repo root by looking for .gitignore"""
    current = os.getcwd()
    while current != '/':
        if os.path.exists(os.path.join(current, '.gitignore')):
            print(f"Found repo root: {current}")
            return current
        current = os.path.dirname(current)
    raise ValueError("Could not find repo root (no .gitignore found)")

def load_base_config():
    """Load base config from yaml file - fail if not found"""
    repo_root = find_repo_root()
    config_path = os.path.join(repo_root, 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_sport_config():
    """Load sport-specific config from yaml file - fail if not found"""
    repo_root = find_repo_root()
    config_path = os.path.join(repo_root, 'config', 'the-odds-api_config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_env_file():
    """Load .env file from repo root"""
    repo_root = find_repo_root()
    env_path = os.path.join(repo_root, '.env')
    
    if not os.path.exists(env_path):
        raise ValueError(f".env file not found at {env_path}")
    
    # Load environment variables from .env file
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

def get_api_key():
    """Get API key from .env file - fail if not found"""
    # First try to load from .env file
    try:
        load_env_file()
    except ValueError as e:
        print(f"Warning: {e}")
    
    # Now try to get the key
    api_key = os.getenv('ODDS_API_KEY') or os.getenv('THE_ODDS_API_KEY')
    if not api_key:
        raise ValueError("API key not found! Set ODDS_API_KEY or THE_ODDS_API_KEY in .env file")
    return api_key

# Load configs and API key
BASE_CONFIG = load_base_config()
SPORT_CONFIG = load_sport_config()
API_KEY = get_api_key()
BASE_URL = BASE_CONFIG['odds_api']['base_url']

# =============================================================================
# API FUNCTIONS (in execution order)
# =============================================================================

def check_api_usage():
    """Check remaining API requests - fail if error"""
    url = f"{BASE_URL}/sports/"
    params = {'apiKey': API_KEY}
    
    response = requests.get(url, params=params, verify=False)
    response.raise_for_status()
    
    remaining = response.headers['x-requests-remaining']
    used = response.headers['x-requests-used']
    
    print(f"API Usage: {used} used, {remaining} remaining")
    return int(float(remaining))

def get_sport_info(sport):
    """Get sport configuration - fail if sport not found"""
    sport = sport.lower()
    sport_info = SPORT_CONFIG['sports'][sport]
    return {
        'key': sport,
        'api_key': sport_info['api_key'],
        'name': sport_info['name'],
        'icon': sport_info['icon']
    }

def get_events(sport_api_key, sport_name):
    """Get current events/games for sport"""
    url = f"{BASE_URL}/sports/{sport_api_key}/odds"
    params = {
        'apiKey': API_KEY,
        'regions': 'us',
        'markets': 'h2h',  # Just get basic data to see events
        'oddsFormat': 'american'
    }
    
    response = requests.get(url, params=params, verify=False)
    response.raise_for_status()
    
    events = response.json()
    print(f"\nFound {len(events)} {sport_name} events")
    
    return events

def try_prop_market(market_name, event_id, sport_api_key):
    """
    Try fetching a specific prop market
    
    Args:
        market_name: Name of market to try (e.g., 'player_pass_tds')
        event_id: Event ID (required)
        sport_api_key: Sport API key (e.g., 'americanfootball_nfl')
    """
    url = f"{BASE_URL}/sports/{sport_api_key}/events/{event_id}/odds"
    
    params = {
        'apiKey': API_KEY,
        'regions': 'us',
        'markets': market_name,
        'oddsFormat': 'american'
    }
    
    response = requests.get(url, params=params, verify=False)
    
    return {
        'market': market_name,
        'status_code': response.status_code,
        'success': response.status_code == 200,
        'data': response.json() if response.status_code == 200 else None,
        'error': response.text if response.status_code != 200 else None
    }

def get_all_possible_markets():
    """Get all possible markets to test from config - flatten nested structure"""
    possible_markets = SPORT_CONFIG['possible_markets']
    all_markets = []
    
    for category, market_list in possible_markets.items():
        all_markets.extend(market_list)
    
    return all_markets

def test_all_possible_markets(events, sport_info):
    """Test all possible prop market types to discover what's available"""
    
    # Get all possible markets to test
    potential_markets = get_all_possible_markets()
    
    print(f"\nTesting {len(potential_markets)} potential markets for {sport_info['name']}...")
    print("=" * 80)
    
    results = []
    available_markets = []
    
    # Use first event - fail if no events
    if not events:
        raise ValueError("No events found - cannot test markets without events")
    
    event_id = events[0]['id']
    sport_api_key = sport_info['api_key']
    print(f"Testing with event: {events[0]['away_team']} @ {events[0]['home_team']}\n")
    
    for market in potential_markets:
        print(f"Testing: {market}...", end=" ")
        
        result = try_prop_market(market, event_id, sport_api_key)
        results.append(result)
        
        if result['success']:
            # Fail fast - if success is True, data MUST exist
            # API returns a single event dict when querying specific event
            data = result['data']
            num_bookmakers = len(data['bookmakers']) if 'bookmakers' in data else 0
            if num_bookmakers > 0:
                print(f"✅ Available ({num_bookmakers} bookmakers)")
                available_markets.append(market)
            else:
                print(f"❌ No bookmakers")
        elif result['status_code'] == 422:
            print(f"❌ Not available")
        else:
            print(f"⚠️  Error: {result['status_code']}")
    
    return results, available_markets

def analyze_market_structure(results, available_markets):
    """Analyze the structure of available markets"""
    print(f"\n{'=' * 80}")
    print(f"AVAILABLE MARKETS SUMMARY")
    print(f"{'=' * 80}\n")
    
    print(f"Total markets tested: {len(results)}")
    print(f"Available markets: {len(available_markets)}\n")
    
    if not available_markets:
        print("⚠️  No prop markets available (may require paid plan or no games today)")
        return
    
    print("Available markets:")
    for market in available_markets:
        print(f"  - {market}")
    
    # Analyze structure of first available market
    print(f"\n{'=' * 80}")
    print(f"SAMPLE MARKET STRUCTURE")
    print(f"{'=' * 80}\n")
    
    for result in results:
        if result['success']:
            # Fail fast - if success, data MUST exist
            # API returns single event dict when querying specific event
            market_name = result['market']
            data = result['data']
            
            # Check if there are bookmakers
            if not data['bookmakers']:
                continue
            
            print(f"Market: {market_name}")
            print(f"Event: {data['away_team']} @ {data['home_team']}")
            print(f"Start time: {data['commence_time']}")
            print(f"\nBookmakers: {len(data['bookmakers'])}")
            
            # First bookmaker MUST exist since we checked len > 0
            bookmaker = data['bookmakers'][0]
            print(f"\nSample bookmaker: {bookmaker['key']}")
            print(f"Markets: {len(bookmaker['markets'])}")
            
            # First market MUST exist
            market = bookmaker['markets'][0]
            print(f"\nSample market structure:")
            print(f"  Key: {market['key']}")
            print(f"  Outcomes: {len(market['outcomes'])}")
            
            # First outcome MUST exist
            outcome = market['outcomes'][0]
            print(f"\n  Sample outcome:")
            print(f"    Name: {outcome['name']}")
            print(f"    Description: {outcome.get('description', 'N/A')}")  # Description only exists for player props
            print(f"    Price: {outcome['price']}")
            print(f"    Point: {outcome.get('point', 'N/A')}")  # Point may not exist for all markets
            
            break  # Only show first available market

def export_results_to_dataframe(results, available_markets):
    """Convert results to dataframe for analysis"""
    print(f"\n{'=' * 80}")
    print(f"EXPORT TO DATAFRAME")
    print(f"{'=' * 80}\n")
    
    # Create summary dataframe
    summary_data = []
    for result in results:
        summary_data.append({
            'market': result['market'],
            'available': result['success'],
            'status_code': result['status_code']
        })
    
    df = pd.DataFrame(summary_data)
    print("Market availability summary:")
    print(df.to_string(index=False))
    
    return df

def save_discovered_markets_to_config(sport, available_markets):
    """Update config file with discovered markets for a sport"""
    repo_root = find_repo_root()
    config_path = os.path.join(repo_root, 'config', 'the-odds-api_config.yaml')
    
    # Load current config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update the discovered markets for this sport
    config['sports'][sport]['discovered_markets'] = available_markets
    
    # Add timestamp comment at the top
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Write updated config
    with open(config_path, 'w') as f:
        # Add header comment
        f.write(f"# The Odds API Sport Configuration\n")
        f.write(f"# Auto-generated by test_the-odds-api_prop_markets.py\n")
        f.write(f"# Last updated: {timestamp}\n")
        f.write(f"# Sport '{sport}' discovered markets updated\n\n")
        
        # Write config
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✅ Updated config file with {len(available_markets)} discovered markets for {sport}")
    print(f"   Config file: {config_path}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main(sport, update_config=False):
    """Main execution function"""
    # Get sport configuration
    sport_info = get_sport_info(sport)
    
    print("=" * 80)
    print(f"{sport_info['icon']} {sport_info['name']} PROP MARKETS DISCOVERY")
    print("=" * 80)
    print(f"Sport: {sport_info['name']} ({sport_info['api_key']})")
    print(f"Timestamp: {datetime.now()}")
    print(f"Update config: {update_config}")
    
    # Step 1: Check API usage
    print("\n" + "=" * 80)
    print("STEP 1: Check API Usage")
    print("=" * 80)
    remaining = check_api_usage()
    
    if remaining < 50:
        print(f"\n⚠️  WARNING: Only {remaining} requests remaining!")
        print(f"   This script will test ~{len(get_all_possible_markets())} markets")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Step 2: Get events
    print("\n" + "=" * 80)
    print(f"STEP 2: Get {sport_info['name']} Events")
    print("=" * 80)
    events = get_events(sport_info['api_key'], sport_info['name'])
    
    # Fail fast - need events to test markets
    if not events:
        raise ValueError(f"No {sport_info['name']} events found. Cannot test markets without events.")
    
    print("\nUpcoming games:")
    for event in events[:5]:  # Show first 5
        print(f"  - {event['away_team']} @ {event['home_team']}")
        print(f"    Starts: {event['commence_time']}")
    
    # Step 3: Test all possible markets
    print("\n" + "=" * 80)
    print("STEP 3: Test All Possible Markets")
    print("=" * 80)
    results, available_markets = test_all_possible_markets(events, sport_info)
    
    # Step 4: Analyze results
    analyze_market_structure(results, available_markets)
    
    # Step 5: Export results
    df = export_results_to_dataframe(results, available_markets)
    
    # Step 6: Update config if requested
    if update_config:
        print(f"\n{'=' * 80}")
        print("STEP 6: Update Config File")
        print(f"{'=' * 80}")
        save_discovered_markets_to_config(sport, available_markets)
    
    # Final summary
    print(f"\n{'=' * 80}")
    print("COMPLETE")
    print(f"{'=' * 80}\n")
    print(f"Sport: {sport_info['name']}")
    print(f"Total markets tested: {len(results)}")
    print(f"Available markets: {len(available_markets)}")
    print(f"\nDataframe stored in variable: df")
    
    if not update_config and available_markets:
        print(f"\n💡 TIP: Run with --update-config to save these markets to the config file")
    
    return df, results, available_markets

# =============================================================================
# NOTEBOOK-FRIENDLY EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Discover available prop markets for a sport from The Odds API'
    )
    parser.add_argument(
        '--sport',
        type=str,
        choices=['nfl', 'nba', 'ncaaf', 'ncaab'],
        default='nfl',
        help='Sport to test (default: nfl)'
    )
    parser.add_argument(
        '--update-config',
        action='store_true',
        help='Update config file with discovered markets'
    )
    args = parser.parse_args()
    
    df, results, available_markets = main(args.sport, update_config=args.update_config)
else:
    # For notebook usage - check if SPORT variable is set
    if SPORT:
        print(f"Running for sport: {SPORT}")
        df, results, available_markets = main(SPORT, update_config=UPDATE_CONFIG)

