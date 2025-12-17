"""
NFL Game Context Scraper - Fragile Edition
Scrapes ESPN and other sites for game context using Selenium

WARNING: This is fragile and will break when sites change their HTML structure.
"""

import os
os.environ['WDM_SSL_VERIFY'] = '0'  # Disable SSL verification for webdriver-manager

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import json
import csv
from datetime import datetime
from typing import Dict, List
import time
import requests
import urllib3
import ssl

# Disable SSL warnings and verification (needed for macOS certificate issues)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context


class NFLContextScraper:
    """Scrape NFL game context using Selenium and BeautifulSoup"""
    
    def __init__(self, headless: bool = True):
        self.headless = headless
        self.driver = None
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        self.session.verify = False  # Disable SSL verification
        
    def setup_driver(self):
        """Setup Selenium driver"""
        print("    Setting up Chrome driver...")
        options = Options()
        if self.headless:
            options.add_argument('--headless=new')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('--ignore-certificate-errors')
        options.add_argument('--ignore-ssl-errors')
        options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36')
        
        # Use webdriver-manager to automatically handle ChromeDriver
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=options)
        self.driver.set_page_load_timeout(30)
        
    def close_driver(self):
        """Close Selenium driver"""
        if self.driver:
            self.driver.quit()
            self.driver = None
    
    def get_team_full_name(self, abbr: str) -> str:
        """Convert team abbreviation to full name"""
        team_map = {
            'CIN': 'Bengals', 'MIA': 'Dolphins',
            'NE': 'Patriots', 'BAL': 'Ravens',
            'LV': 'Raiders', 'HOU': 'Texans',
            'ATL': 'Falcons', 'ARI': 'Cardinals',
            'PIT': 'Steelers', 'DET': 'Lions',
            'MIN': 'Vikings', 'NYG': 'Giants',
            'TB': 'Buccaneers', 'CAR': 'Panthers',
        }
        return team_map.get(abbr, abbr)
    
    def scrape_espn_injuries(self, team_abbr: str) -> List[str]:
        """Scrape injury report from ESPN"""
        print(f"    Scraping ESPN injuries for {team_abbr}...")
        
        try:
            team_name = self.get_team_full_name(team_abbr).lower()
            url = f"https://www.espn.com/nfl/team/injuries/_/name/{team_abbr.lower()}"
            
            if not self.driver:
                self.setup_driver()
            
            self.driver.get(url)
            time.sleep(2)  # Wait for page load
            
            injuries = []
            
            # Try to find injury table
            try:
                injury_rows = self.driver.find_elements(By.CSS_SELECTOR, "tr.Table__TR")
                for row in injury_rows[:5]:  # Get top 5 injuries
                    try:
                        text = row.text
                        if text and any(status in text.upper() for status in ['OUT', 'QUESTIONABLE', 'DOUBTFUL']):
                            injuries.append(text[:200])
                    except:
                        continue
            except:
                pass
            
            # If no structured data, grab any text mentioning injuries
            if not injuries:
                try:
                    page_text = self.driver.find_element(By.TAG_NAME, "body").text
                    for line in page_text.split('\n'):
                        if any(status in line.upper() for status in ['OUT', 'QUESTIONABLE', 'DOUBTFUL']):
                            injuries.append(line[:200])
                            if len(injuries) >= 5:
                                break
                except:
                    pass
            
            return injuries[:5]
            
        except Exception as e:
            print(f"      Error scraping ESPN injuries: {e}")
            return []
    
    def scrape_team_recent_games(self, team_abbr: str) -> List[str]:
        """Scrape recent game results"""
        print(f"    Scraping recent games for {team_abbr}...")
        
        try:
            url = f"https://www.espn.com/nfl/team/schedule/_/name/{team_abbr.lower()}"
            
            if not self.driver:
                self.setup_driver()
            
            self.driver.get(url)
            time.sleep(2)
            
            games = []
            
            # Try to find schedule table
            try:
                game_rows = self.driver.find_elements(By.CSS_SELECTOR, "tr.Table__TR")
                for row in game_rows[-5:]:  # Get last 5 games (most recent)
                    try:
                        text = row.text
                        # Look for score patterns (numbers indicating games played)
                        if any(char.isdigit() for char in text) and ('W' in text or 'L' in text):
                            games.append(text[:200])
                    except:
                        continue
            except:
                pass
            
            return games[-3:] if len(games) >= 3 else games  # Return last 3
            
        except Exception as e:
            print(f"      Error scraping recent games: {e}")
            return []
    
    def scrape_game_preview_text(self, away_team: str, home_team: str) -> List[str]:
        """Scrape game preview/matchup text"""
        print(f"    Scraping game preview for {away_team} @ {home_team}...")
        
        try:
            # Try ESPN matchup page
            url = f"https://www.espn.com/nfl/preview?gameId=401671821"  # Generic, would need game ID
            
            # Alternative: Search ESPN for preview
            search_url = f"https://www.espn.com/search/_/q/{away_team}%20{home_team}%20preview"
            
            response = self.session.get(search_url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for preview text
            preview_text = []
            paragraphs = soup.find_all('p')
            for p in paragraphs[:10]:
                text = p.get_text(strip=True)
                if len(text) > 50 and (away_team in text or home_team in text):
                    preview_text.append(text[:300])
            
            return preview_text[:3]
            
        except Exception as e:
            print(f"      Error scraping preview: {e}")
            return []
    
    def scrape_game_context(self, away_team: str, home_team: str, week: int = 16) -> Dict:
        """Scrape all context for a game"""
        print(f"\n  Scraping {away_team} @ {home_team}...")
        
        context = {
            'game': f"{away_team} @ {home_team}",
            'week': week,
            'scraped_at': datetime.now().isoformat(),
        }
        
        try:
            # Get away team data
            context['away_team'] = away_team
            context['away_injuries'] = self.scrape_espn_injuries(away_team)
            time.sleep(2)  # Be nice to servers
            
            context['away_recent_games'] = self.scrape_team_recent_games(away_team)
            time.sleep(2)
            
            # Get home team data
            context['home_team'] = home_team
            context['home_injuries'] = self.scrape_espn_injuries(home_team)
            time.sleep(2)
            
            context['home_recent_games'] = self.scrape_team_recent_games(home_team)
            time.sleep(2)
            
            # Get game preview
            context['game_preview'] = self.scrape_game_preview_text(away_team, home_team)
            
        except Exception as e:
            print(f"    Error scraping game: {e}")
        
        return context


def save_to_json(data: List[Dict], filename: str):
    """Save data to JSON"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved to {filename}")


def save_to_text_summary(data: List[Dict], filename: str):
    """Save human-readable summary"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("NFL Week 16 Game Research - Scraped Context\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for game in data:
            f.write(f"\n{'='*60}\n")
            f.write(f"GAME: {game['game']}\n")
            f.write(f"{'='*60}\n\n")
            
            f.write(f"--- {game['away_team']} (Away) ---\n\n")
            
            f.write("Injuries:\n")
            if game.get('away_injuries'):
                for inj in game['away_injuries']:
                    f.write(f"  • {inj}\n")
            else:
                f.write("  (No data found)\n")
            
            f.write("\nRecent Games:\n")
            if game.get('away_recent_games'):
                for g in game['away_recent_games']:
                    f.write(f"  • {g}\n")
            else:
                f.write("  (No data found)\n")
            
            f.write(f"\n--- {game['home_team']} (Home) ---\n\n")
            
            f.write("Injuries:\n")
            if game.get('home_injuries'):
                for inj in game['home_injuries']:
                    f.write(f"  • {inj}\n")
            else:
                f.write("  (No data found)\n")
            
            f.write("\nRecent Games:\n")
            if game.get('home_recent_games'):
                for g in game['home_recent_games']:
                    f.write(f"  • {g}\n")
            else:
                f.write("  (No data found)\n")
            
            f.write("\n--- Game Preview ---\n")
            if game.get('game_preview'):
                for prev in game['game_preview']:
                    f.write(f"  {prev}\n\n")
            else:
                f.write("  (No preview data found)\n")
            
            f.write("\n")
    
    print(f"✅ Saved to {filename}")


def main():
    """Main execution"""
    print("\n" + "="*60)
    print("NFL Game Context Scraper - Fragile Edition")
    print("="*60)
    print("\n⚠️  WARNING: This scraper is fragile and may break!")
    print("⚠️  It will take ~5-10 minutes to scrape all games.")
    print("⚠️  Please be patient and don't kill the process.\n")
    
    # Week 16 games
    games = [
        ('CIN', 'MIA'),  # Bengals @ Dolphins
        ('NE', 'BAL'),   # Patriots @ Ravens
        ('LV', 'HOU'),   # Raiders @ Texans
        ('ATL', 'ARI'),  # Falcons @ Cardinals
        ('PIT', 'DET'),  # Steelers @ Lions
        ('MIN', 'NYG'),  # Vikings @ Giants
        ('TB', 'CAR'),   # Buccaneers @ Panthers
    ]
    
    week = 16
    scraper = NFLContextScraper(headless=True)
    all_game_data = []
    
    try:
        for away, home in games:
            try:
                game_data = scraper.scrape_game_context(away, home, week)
                all_game_data.append(game_data)
            except Exception as e:
                print(f"  ❌ Failed to scrape {away} @ {home}: {e}")
                continue
        
    finally:
        scraper.close_driver()
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_file = f"../data/nfl_research/nfl_week{week}_scraped_{timestamp}.json"
    text_file = f"../data/nfl_research/nfl_week{week}_scraped_{timestamp}.txt"
    
    save_to_json(all_game_data, json_file)
    save_to_text_summary(all_game_data, text_file)
    
    print("\n" + "="*60)
    print(f"✅ Scraping complete! Processed {len(all_game_data)} games.")
    print("="*60)
    print(f"\nOutput files:")
    print(f"  • {json_file}")
    print(f"  • {text_file} (human-readable)")
    print(f"\nNext: Review the scraped data and use it to write your post!")
    
    # Auto-open the text file
    print(f"\n📂 Opening {text_file}...")
    import subprocess
    subprocess.run(["open", text_file])


if __name__ == "__main__":
    main()

