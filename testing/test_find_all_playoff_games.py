"""
Test to verify we can find ALL 13 playoff games from ESPN schedule.

Expected for any playoff year:
- Wild Card: 6 games
- Divisional: 4 games
- Conference Championships: 2 games
- Super Bowl: 1 game
TOTAL: 13 games

Usage:
    python3 testing/test_find_all_playoff_games.py
"""

import requests
from bs4 import BeautifulSoup
import ssl
import urllib3

# Disable SSL warnings
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def test_schedule_page(year):
    """Test scraping ALL playoff weeks for a year."""
    print(f"\n{'='*80}")
    print(f"TESTING YEAR {year}")
    print(f"{'='*80}")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    all_game_ids = set()
    
    # Playoff weeks: 1 (Wild Card), 2 (Divisional), 3 (Conf Champ), 5 (Super Bowl)
    playoff_weeks = [1, 2, 3, 5]
    week_names = {1: "Wild Card", 2: "Divisional", 3: "Conf Champ", 5: "Super Bowl"}
    
    for week in playoff_weeks:
        url = f"https://www.espn.com/nfl/schedule/_/week/{week}/year/{year}/seasontype/3"
        print(f"\nWeek {week} ({week_names[week]}): {url}")
        
        try:
            response = requests.get(url, headers=headers, timeout=10, verify=False)
            
            if response.status_code != 200:
                print(f"  ❌ Failed (status {response.status_code})")
                continue
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            week_games = set()
            for link in soup.find_all('a', href=True):
                href = link['href']
                if '/nfl/game/_/gameId/' in href:
                    try:
                        game_id = href.split('/gameId/')[1].split('/')[0].split('?')[0]
                        if game_id.isdigit():
                            week_games.add(game_id)
                    except:
                        continue
            
            print(f"  ✅ Found {len(week_games)} games")
            all_game_ids.update(week_games)
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print(f"\n📊 Total unique game IDs: {len(all_game_ids)}")
    
    # Show all game IDs
    for i, gid in enumerate(sorted(all_game_ids), 1):
        print(f"  {i}. {gid}")
    
    return len(all_game_ids)


def main():
    """Test multiple years."""
    print("\n" + "="*80)
    print("TESTING ESPN PLAYOFF SCHEDULE SCRAPING")
    print("="*80)
    print("\nGoal: Find all 13 playoff games per season")
    print("  - Wild Card: 6 games")
    print("  - Divisional: 4 games")
    print("  - Championships: 2 games")
    print("  - Super Bowl: 1 game")
    
    # Test 2023 season playoffs (completed)
    count_2023 = test_schedule_page(2023)
    
    # Test 2024 season playoffs (in progress)
    count_2024 = test_schedule_page(2024)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"2023 playoffs: {count_2023} games found (expected 13)")
    print(f"2024 playoffs: {count_2024} games found (expected ~6-13, depends on progress)")
    
    if count_2023 < 13:
        print(f"\n⚠️  PROBLEM: Only finding {count_2023}/13 games for completed 2023 season!")
        print(f"   This means our scraping is missing games.")
        print(f"   Likely causes:")
        print(f"   1. ESPN loads games via JavaScript")
        print(f"   2. Different rounds are on different pages")
        print(f"   3. Need to scrape week-by-week instead of full schedule")
    else:
        print(f"\n✅ SUCCESS: Found all games!")
    
    print(f"\n💡 Next step: Check the saved HTML files to see page structure")
    print(f"   /tmp/espn_playoffs_2023.html")
    print(f"   /tmp/espn_playoffs_2024.html")


if __name__ == '__main__':
    main()

