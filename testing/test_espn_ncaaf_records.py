"""
Test ESPN API for NCAAF team records.

Purpose:
Explore if ESPN has NCAAF (college football) team data available via their API.

Context:
Currently we're not fetching NCAAF records from ESPN. This script tests if it's possible.

Usage:
    cd /Users/thomasmyles/dev/betting
    python3 testing/test_espn_ncaaf_records.py
"""

import requests
import ssl
import urllib3
from pprint import pprint

# Disable SSL warnings
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

ESPN_API_BASE = 'https://sports.core.api.espn.com/v2/sports'

# Known college football teams with their ESPN IDs
# These IDs are from espn.com URLs (e.g., /cfb/team/_/id/52/georgia-bulldogs)
TEST_NCAAF_TEAMS = {
    'Georgia Bulldogs': 61,
    'Alabama Crimson Tide': 333,
    'Ohio State Buckeyes': 194,
    'Texas Longhorns': 251,
    'Michigan Wolverines': 130,
    'Oregon Ducks': 2483,
    'Penn State Nittany Lions': 213,
    'Notre Dame Fighting Irish': 87,
}

def test_ncaaf_record_fetch(team_name, team_id, season=2025):
    """
    Test fetching NCAAF team record from ESPN API.
    
    Args:
        team_name: Human-readable team name
        team_id: ESPN team ID
        season: Year (2025 = 2025-26 season, but NCAAF might use 2024 for 2024 season)
    """
    print(f"\n{'='*80}")
    print(f"Testing: {team_name} (ID: {team_id})")
    print(f"{'='*80}")
    
    # Try different URL patterns ESPN might use for college football
    url_patterns = [
        # Pattern 1: football/college-football
        f"{ESPN_API_BASE}/football/leagues/college-football/seasons/{season}/types/2/teams/{team_id}/record",
        
        # Pattern 2: americanfootball/college-football
        f"{ESPN_API_BASE}/americanfootball/leagues/college-football/seasons/{season}/types/2/teams/{team_id}/record",
        
        # Pattern 3: Try previous season (2024 for 2024 season)
        f"{ESPN_API_BASE}/football/leagues/college-football/seasons/2024/types/2/teams/{team_id}/record",
        
        # Pattern 4: Different type param
        f"{ESPN_API_BASE}/football/leagues/college-football/seasons/{season}/types/1/teams/{team_id}/record",
    ]
    
    for i, url in enumerate(url_patterns, 1):
        print(f"\nAttempt {i}: {url}")
        try:
            response = requests.get(url, timeout=5, verify=False)
            print(f"  Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"  ✅ SUCCESS! Found data:")
                
                # Extract record
                items = data.get('items', [])
                for item in items:
                    if item.get('type') == 'total' and item.get('name') == 'overall':
                        record = item.get('summary', 'N/A')
                        print(f"  📊 Record: {record}")
                        print(f"\n  Full item data:")
                        pprint(item, indent=4)
                        return True
                
                if not items:
                    print("  ⚠️  Response has no 'items' field")
                    print("  Full response:")
                    pprint(data, indent=4)
                
            elif response.status_code == 404:
                print(f"  ❌ Not Found (404)")
            else:
                print(f"  ❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ Exception: {e}")
    
    return False


def main():
    print("="*80)
    print("ESPN NCAAF RECORD API TEST")
    print("="*80)
    print(f"\nTesting {len(TEST_NCAAF_TEAMS)} teams...")
    
    success_count = 0
    
    for team_name, team_id in TEST_NCAAF_TEAMS.items():
        if test_ncaaf_record_fetch(team_name, team_id):
            success_count += 1
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"✅ Successful: {success_count}/{len(TEST_NCAAF_TEAMS)}")
    print(f"❌ Failed: {len(TEST_NCAAF_TEAMS) - success_count}/{len(TEST_NCAAF_TEAMS)}")
    
    if success_count > 0:
        print("\n🎉 ESPN NCAAF data IS available!")
        print("   We can add NCAAF record fetching to the main script.")
    else:
        print("\n⚠️  ESPN NCAAF data NOT found with current patterns.")
        print("   May need different URL structure or ESPN may not provide NCAAF records.")


if __name__ == '__main__':
    main()

