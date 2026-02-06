"""
Explore NBA data sources to find game logs with timestamps.

Goal: Get play-by-play data with timestamps for a single game, including:
- Player points scored at each minute/play
- Quarter/time information
- Player minutes played by quarter

Data sources to try:
1. nba_api package (official NBA stats)
2. ESPN API
3. Basketball Reference web scraping
"""

import json
from datetime import datetime

# =============================================================================
# SOURCE 1: NBA API via nba_api package
# =============================================================================

def try_nba_api():
    """Try to get play-by-play data using nba_api package."""
    try:
        from nba_api.stats.endpoints import playbyplayv2, boxscoretraditionalv2
        
        # Recent game: Let's try a game from Feb 4, 2026
        # We need a game_id - format is typically "00XXXXXXXX"
        # Let's try to find a recent Mavericks game
        
        # Try Mavericks vs Celtics from Feb 5, 2026 (if it exists)
        # Game IDs are usually: 002XYYYYYY where X=season (6=2026), Y=game number
        
        # Let's try a known game ID format
        game_id = "0022600694"  # Placeholder, we'll need to find actual game IDs
        
        print("=" * 80)
        print("SOURCE 1: NBA API (nba_api package)")
        print("=" * 80)
        
        # Try to get play-by-play
        print("\n--- Attempting to get play-by-play data ---")
        pbp = playbyplayv2.PlayByPlayV2(game_id=game_id)
        pbp_df = pbp.get_data_frames()[0]
        
        print(f"✅ Success! Got {len(pbp_df)} plays")
        print(f"\nColumns available: {list(pbp_df.columns)}")
        print(f"\nFirst 5 plays:")
        print(pbp_df.head())
        
        # Try to get box score with quarter splits
        print("\n--- Attempting to get box score with quarters ---")
        box = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
        player_stats = box.get_data_frames()[0]
        
        print(f"✅ Success! Got stats for {len(player_stats)} players")
        print(f"\nColumns available: {list(player_stats.columns)}")
        print(f"\nSample player stats:")
        print(player_stats[['PLAYER_NAME', 'MIN', 'PTS', 'FGA']].head())
        
        # Save to file
        pbp_df.to_csv('/Users/thomasmyles/dev/betting/tmp/nba_api_playbyplay.csv', index=False)
        player_stats.to_csv('/Users/thomasmyles/dev/betting/tmp/nba_api_boxscore.csv', index=False)
        
        return True, pbp_df, player_stats
        
    except ImportError:
        print("❌ nba_api package not installed")
        print("Install with: pip install nba_api")
        return False, None, None
    except Exception as e:
        print(f"❌ Error: {e}")
        return False, None, None


# =============================================================================
# SOURCE 2: ESPN API
# =============================================================================

def try_espn_api():
    """Try to get play-by-play data from ESPN API."""
    try:
        import requests
        
        print("\n" + "=" * 80)
        print("SOURCE 2: ESPN API")
        print("=" * 80)
        
        # ESPN API endpoint for NBA play-by-play
        # Format: http://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event=GAME_ID
        
        # Let's try to find a recent game
        # First, get today's scoreboard
        scoreboard_url = "http://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
        
        print("\n--- Attempting to get recent games ---")
        response = requests.get(scoreboard_url)
        
        if response.status_code == 200:
            data = response.json()
            
            # Get first game
            if data.get('events') and len(data['events']) > 0:
                game = data['events'][0]
                game_id = game['id']
                
                home_team = game['competitions'][0]['competitors'][0]['team']['displayName']
                away_team = game['competitions'][0]['competitors'][1]['team']['displayName']
                
                print(f"✅ Found game: {away_team} @ {home_team} (ID: {game_id})")
                
                # Now get play-by-play for this game
                print(f"\n--- Attempting to get play-by-play for game {game_id} ---")
                pbp_url = f"http://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={game_id}"
                
                pbp_response = requests.get(pbp_url)
                
                if pbp_response.status_code == 200:
                    pbp_data = pbp_response.json()
                    
                    # Check if play-by-play exists
                    if 'plays' in pbp_data:
                        plays = pbp_data['plays']
                        print(f"✅ Success! Got {len(plays)} plays")
                        
                        # Show sample play
                        if len(plays) > 0:
                            print(f"\nSample play structure:")
                            print(json.dumps(plays[0], indent=2)[:500] + "...")
                        
                        # Save to file
                        with open('/Users/thomasmyles/dev/betting/tmp/espn_playbyplay.json', 'w') as f:
                            json.dump(pbp_data, f, indent=2)
                        
                        print(f"\n✅ Saved to tmp/espn_playbyplay.json")
                        
                        return True, pbp_data
                    else:
                        print("❌ No play-by-play data in response")
                        print(f"Available keys: {list(pbp_data.keys())}")
                        return False, None
                else:
                    print(f"❌ Failed to get play-by-play: {pbp_response.status_code}")
                    return False, None
            else:
                print("❌ No games found in scoreboard")
                return False, None
        else:
            print(f"❌ Failed to get scoreboard: {response.status_code}")
            return False, None
            
    except ImportError:
        print("❌ requests package not installed")
        return False, None
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None


# =============================================================================
# SOURCE 3: Basketball Reference scraping
# =============================================================================

def try_basketball_reference():
    """Try to scrape play-by-play from Basketball Reference."""
    try:
        import requests
        from bs4 import BeautifulSoup
        import pandas as pd
        
        print("\n" + "=" * 80)
        print("SOURCE 3: Basketball Reference")
        print("=" * 80)
        
        # Example URL format:
        # https://www.basketball-reference.com/boxscores/pbp/202602050DAL.html
        
        # Let's try a recent game (need to know the date and team code)
        # For now, let's try Feb 4, 2024 (format: YYYYMMDD)
        date = "20240204"
        team_code = "DAL"  # Dallas Mavericks
        
        url = f"https://www.basketball-reference.com/boxscores/pbp/{date}0{team_code}.html"
        
        print(f"\n--- Attempting to scrape: {url} ---")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            print(f"✅ Got response (status 200)")
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find play-by-play table
            pbp_table = soup.find('table', {'id': 'pbp'})
            
            if pbp_table:
                print(f"✅ Found play-by-play table")
                
                # Parse table to dataframe
                df = pd.read_html(str(pbp_table))[0]
                
                print(f"✅ Success! Got {len(df)} rows")
                print(f"\nColumns: {list(df.columns)}")
                print(f"\nFirst 10 rows:")
                print(df.head(10))
                
                # Save to file
                df.to_csv('/Users/thomasmyles/dev/betting/tmp/bbref_playbyplay.csv', index=False)
                print(f"\n✅ Saved to tmp/bbref_playbyplay.csv")
                
                return True, df
            else:
                print(f"❌ No play-by-play table found")
                print(f"Available table IDs: {[t.get('id') for t in soup.find_all('table')]}")
                return False, None
        else:
            print(f"❌ Failed to get page: {response.status_code}")
            return False, None
            
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Install with: pip install beautifulsoup4 lxml")
        return False, None
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n🏀 NBA Play-by-Play Data Source Explorer")
    print("=" * 80)
    print(f"Timestamp: {datetime.now()}")
    print("=" * 80)
    
    results = {}
    
    # Try each source
    print("\n\n🔍 Testing data sources...\n")
    
    # Source 1: NBA API
    try:
        success, pbp_df, box_df = try_nba_api()
        results['nba_api'] = success
    except Exception as e:
        print(f"NBA API crashed: {e}")
        results['nba_api'] = False
    
    # Source 2: ESPN
    try:
        success, espn_data = try_espn_api()
        results['espn'] = success
    except Exception as e:
        print(f"ESPN API crashed: {e}")
        results['espn'] = False
    
    # Source 3: Basketball Reference
    try:
        success, bbref_df = try_basketball_reference()
        results['basketball_reference'] = success
    except Exception as e:
        print(f"Basketball Reference crashed: {e}")
        results['basketball_reference'] = False
    
    # Summary
    print("\n\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    for source, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{source:25s}: {status}")
    
    print("\n" + "=" * 80)
    print("Next steps:")
    print("1. Check tmp/ folder for downloaded data")
    print("2. Inspect the structure of successful sources")
    print("3. Build parser for the best source")
    print("=" * 80)
