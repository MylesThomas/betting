"""
Test script to verify we can get all 2024 NFL playoff games.

2024 NFL Playoffs (should be 13 games):
- Wild Card: 6 games
- Divisional: 4 games  
- Conference Championships: 2 games
- Super Bowl: 1 game

Expected QBs:
- AFC: Patrick Mahomes, Josh Allen, Lamar Jackson, Justin Herbert, C.J. Stroud, Joe Flacco
- NFC: Jalen Hurts, Jared Goff, Jordan Love, Baker Mayfield, Matthew Stafford, Sam Darnold

Usage:
    python3 testing/test_2024_playoff_games.py
"""

import requests
from bs4 import BeautifulSoup
import ssl
import urllib3
import pandas as pd
from io import StringIO

# Disable SSL warnings
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def get_playoff_games_2024():
    """Get all playoff game IDs from 2024 season."""
    url = "https://www.espn.com/nfl/schedule/_/seasontype/3/year/2024"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    print("="*80)
    print("FETCHING 2024 NFL PLAYOFF SCHEDULE")
    print("="*80)
    print(f"\nURL: {url}")
    
    response = requests.get(url, headers=headers, timeout=10, verify=False)
    
    if response.status_code != 200:
        print(f"❌ Failed to fetch (status {response.status_code})")
        return []
    
    print(f"✅ Got page (status 200)")
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find all game links
    game_ids_set = set()
    for link in soup.find_all('a', href=True):
        href = link['href']
        if '/nfl/game/_/gameId/' in href:
            try:
                game_id = href.split('/gameId/')[1].split('/')[0].split('?')[0]
                if game_id.isdigit():
                    game_ids_set.add(game_id)
            except:
                continue
    
    game_ids = sorted(list(game_ids_set))
    
    print(f"\n{'='*80}")
    print(f"FOUND {len(game_ids)} PLAYOFF GAMES")
    print(f"{'='*80}")
    
    return game_ids


def get_known_2024_playoff_qbs():
    """
    Manually list known 2024 playoff QBs to test scraping.
    
    Wild Card QBs (based on actual 2024 playoffs):
    - HOU vs LAC: C.J. Stroud vs Justin Herbert
    - BAL vs PIT: Lamar Jackson vs Russell Wilson  
    - PHI vs GB: Jalen Hurts vs Jordan Love
    - BUF vs DEN: Josh Allen vs Bo Nix
    - LAR vs MIN: Matthew Stafford vs Sam Darnold
    - TB vs WAS: Baker Mayfield vs Jayden Daniels
    """
    return {
        'C.J. Stroud': 4432577,
        'Justin Herbert': 3915511,
        'Lamar Jackson': 3916387,
        'Russell Wilson': 14881,
        'Jalen Hurts': 4361741,
        'Jordan Love': 4361579,
        'Josh Allen': 3918298,
        'Bo Nix': 4430737,
        'Matthew Stafford': 12483,
        'Sam Darnold': 3917315,
        'Baker Mayfield': 3052587,
        'Jayden Daniels': 4431611,
    }


def get_qbs_from_game(game_id, game_num):
    """Extract QBs from a specific game using pandas to parse tables."""
    url = f"https://www.espn.com/nfl/boxscore/_/gameId/{game_id}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10, verify=False)
        if response.status_code != 200:
            return []
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Get game info
        game_info = soup.find('title')
        game_title = game_info.get_text().split(' - ')[0] if game_info else "Unknown"
        
        qbs = []
        
        # Use pandas to parse all tables
        try:
            tables_html = soup.find_all('table')
            
            for table in tables_html:
                try:
                    # Check if table contains passing stats
                    table_text = table.get_text().lower()
                    if 'passing' not in table_text and 'c/att' not in table_text:
                        continue
                    
                    # Parse with pandas
                    df = pd.read_html(StringIO(str(table)))[0]
                    
                    # Look for player names in the first column
                    if len(df) > 0 and len(df.columns) > 0:
                        # Find player links in this table
                        for link in table.find_all('a', href=True):
                            href = link['href']
                            
                            if '/nfl/player/_/id/' in href:
                                parts = href.split('/id/')[1].split('/')
                                if len(parts) >= 2:
                                    player_id = parts[0]
                                    player_name = link.get_text().strip()
                                    
                                    if player_name and player_id.isdigit() and len(player_name) > 2:
                                        if not any(q['id'] == player_id for q in qbs):
                                            qbs.append({
                                                'name': player_name,
                                                'id': player_id
                                            })
                except:
                    continue
                    
        except:
            pass
        
        print(f"\n[Game {game_num}] {game_title}")
        print(f"   Game ID: {game_id}")
        print(f"   URL: {url}")
        print(f"   QBs found: {', '.join([q['name'] for q in qbs]) if qbs else 'None'}")
        
        return qbs
        
    except Exception as e:
        print(f"\n[Game {game_num}] Error: {e}")
        return []


def main():
    """Main test function."""
    print("\n")
    
    # Step 1: Get all playoff games
    game_ids = get_playoff_games_2024()
    
    if not game_ids:
        print("\n❌ No games found!")
        return
    
    # Expected number
    print(f"\nExpected: 13 games (6 Wild Card + 4 Divisional + 2 Championships + 1 Super Bowl)")
    print(f"Found: {len(game_ids)} games")
    
    if len(game_ids) < 13:
        print(f"⚠️  Missing {13 - len(game_ids)} games!")
    elif len(game_ids) > 13:
        print(f"⚠️  Found extra {len(game_ids) - 13} games (may include Pro Bowl/All-Star)")
    else:
        print(f"✅ Correct number of games!")
    
    # Step 2: Extract QBs from each game
    print(f"\n{'='*80}")
    print(f"EXTRACTING QBs FROM EACH GAME")
    print(f"{'='*80}")
    
    all_qbs = {}
    
    for i, game_id in enumerate(game_ids, 1):
        qbs = get_qbs_from_game(game_id, i)
        
        for qb in qbs:
            qb_id = qb['id']
            qb_name = qb['name']
            
            if qb_id not in all_qbs:
                all_qbs[qb_id] = qb_name
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Total games: {len(game_ids)}")
    print(f"Unique QBs: {len(all_qbs)}")
    print(f"\nQBs found:")
    for qb_name in sorted(all_qbs.values()):
        print(f"  - {qb_name}")
    
    print(f"\n{'='*80}")
    print(f"VERIFICATION")
    print(f"{'='*80}")
    
    expected_qbs = [
        "Patrick Mahomes", "Josh Allen", "Lamar Jackson", "Justin Herbert",
        "C.J. Stroud", "Joe Flacco", "Jalen Hurts", "Jared Goff",
        "Jordan Love", "Baker Mayfield", "Matthew Stafford", "Sam Darnold"
    ]
    
    found_qb_names = [name.lower() for name in all_qbs.values()]
    
    print(f"\nExpected QBs (check if we got them):")
    for expected in expected_qbs:
        found = any(expected.lower() in name for name in found_qb_names)
        status = "✅" if found else "❌"
        print(f"  {status} {expected}")
    
    print(f"\n{'='*80}")


if __name__ == '__main__':
    main()

