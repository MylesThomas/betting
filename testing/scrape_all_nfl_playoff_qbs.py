"""
Scrape playoff game logs for ALL NFL quarterbacks who have played in playoffs.

Strategy:
1. Discovery Phase (use --start-year/--end-year for discovery):
   - Scrape ESPN's playoff stats pages for specified years
   - Find all players who appeared in playoff stats
   - Build master list: {player_id: {'name': str, 'seasons': [...]}}
   
2. Scraping Phase (scrapes history for each player):
   - For each player found, scrape years specified by --scrape-start/--scrape-end
   - By default, scrapes same years as discovery (--start-year/--end-year)
   - Example: --start-year 2025 --end-year 2026 scrapes only 2025-2026 games
   - To get full history: add --scrape-start 2001 to scrape back to 2001
   
3. Filtering Phase:
   - Only keep games with 10+ passing attempts (starters only)
   - This automatically filters out non-QBs (RBs, WRs, punters)
   
4. Export Phase:
   - Individual QB files: one per QB with ALL their playoff games
   - Master file: all QBs combined, sorted by season/date

Output:
  - Individual files: nfl_qb_playoff_gamelogs_{qb_name}_{player_id}.csv (one per QB)
  - Master file: nfl_all_qb_playoff_gamelogs_{start_year}_{end_year}_{date}.csv (all QBs combined)

Storage Locations:
  Test Mode (--test-mode):
    ~/Downloads/tmp/nfl_qb_playoff_gamelogs_tom_brady_12483.csv
    ~/Downloads/tmp/nfl_all_qb_playoff_gamelogs_*.csv
  
  Production (--export):
    s3://nfl-betting-mt/data/01_input/espn_web/playoffs/qb/gamelogs/nfl_qb_playoff_gamelogs_tom_brady_12483.csv
    s3://nfl-betting-mt/data/01_input/espn_web/playoffs/qb/gamelogs/nfl_all_qb_playoff_gamelogs_*.csv
  
  Local Override (--local-only):
    data/01_input/espn_web/playoffs/qb/gamelogs/nfl_qb_playoff_gamelogs_tom_brady_12483.csv
    data/01_input/espn_web/playoffs/qb/gamelogs/nfl_all_qb_playoff_gamelogs_*.csv
  
Filter:
  Only includes games where QB threw 10+ passes (excludes backup/garbage time appearances)

Usage:
    # Quick test: Find 2025-2026 playoff QBs, scrape only their 2025-2026 games (~1 min, ~24 requests)
    python3 testing/scrape_all_nfl_playoff_qbs.py --start-year 2025 --end-year 2026 --export --test-mode
    
    # Find 2025-2026 playoff QBs, get their FULL history back to 2001 (~5 min, ~300 requests)
    python3 testing/scrape_all_nfl_playoff_qbs.py --start-year 2025 --end-year 2026 --scrape-start 2001 --export --test-mode
    
    # Full historical: Find ALL ~500+ QBs from 2001-2026, complete history (~60+ min, ~12,000 requests)
    python3 testing/scrape_all_nfl_playoff_qbs.py --start-year 2001 --end-year 2026 --export

Flags:
  --start-year/--end-year: Discovery range (which years to find QBs from)
  --scrape-start/--scrape-end: History range (which years to scrape for each QB)
    - If not specified, defaults to same as --start-year/--end-year
  --export: Save files
  --test-mode: Save to ~/Downloads/tmp (for testing)
  --local-only: Save locally instead of S3
  --overwrite: Re-scrape and overwrite existing files (default: skip existing)

Key Points:
  - Automatically filters for games with 10+ passing attempts (starters only)
  - Adds playoff_round column (Wild Card, Divisional, Conference Championship, Super Bowl)
  - Individual files use stable naming: {name}_{player_id}.csv (enables incremental updates)
  - Master file includes timestamp for tracking different scrape runs
  - Skips existing files by default (fast incremental updates)
"""

import requests
import pandas as pd
from bs4 import BeautifulSoup
import argparse
import time
import ssl
import urllib3
from io import StringIO
from pathlib import Path
from datetime import datetime
import os
import json

# Disable SSL warnings
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# S3 configuration
S3_BUCKET = 'nfl-betting-mt'
S3_PREFIX = 'data/01_input/espn_web/playoffs/qb/gamelogs'

# Local data directories
LOCAL_DATA_DIR = 'data/01_input/espn_web/playoffs/qb/gamelogs'
TEST_DATA_DIR = os.path.expanduser('~/Downloads/tmp')

# Filter: Only include games where QB threw 10+ passes (starters only)
MIN_ATTEMPTS = 10


def get_playoff_games_for_year(year):
    """
    Get all playoff games for a specific year from ESPN.
    
    ESPN organizes playoffs by week (seasontype=3):
    - Week 1: Wild Card (6 games)
    - Week 2: Divisional (4 games)
    - Week 3: Conference Championships (2 games)
    - Week 5: Super Bowl (1 game)
    
    Total: 13 games per playoff season
    
    Returns:
        list: List of game IDs
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    all_game_ids = set()
    
    # Playoff weeks: 1 (Wild Card), 2 (Divisional), 3 (Conf Champ), 5 (Super Bowl)
    playoff_weeks = [1, 2, 3, 5]
    
    try:
        for week in playoff_weeks:
            url = f"https://www.espn.com/nfl/schedule/_/week/{week}/year/{year}/seasontype/3"
            
            response = requests.get(url, headers=headers, timeout=10, verify=False)
            
            if response.status_code != 200:
                continue
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all game links
            for link in soup.find_all('a', href=True):
                href = link['href']
                if '/nfl/game/_/gameId/' in href:
                    try:
                        game_id = href.split('/gameId/')[1].split('/')[0].split('?')[0]
                        # Validate it's a numeric ID
                        if game_id.isdigit():
                            all_game_ids.add(game_id)
                    except:
                        continue
            
            time.sleep(0.3)  # Be nice to servers
        
        return list(all_game_ids)
        
    except Exception as e:
        print(f"   ⚠️  Error getting games for {year}: {e}")
        return []


def get_qbs_from_game(game_id):
    """
    Extract QB info from a specific game box score.
    
    Only extracts players from the PASSING stats section to ensure we get QBs only.
    
    Returns:
        list: List of dicts with QB info: [{'name': str, 'id': int}, ...]
    """
    url = f"https://www.espn.com/nfl/boxscore/_/gameId/{game_id}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10, verify=False)
        if response.status_code != 200:
            return []
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        qbs = []
        
        # Find passing stats tables only
        # ESPN puts passing stats in tables with specific headers
        tables = soup.find_all('table')
        
        for table in tables:
            # Check if this is a passing stats table
            try:
                # Look for "Passing" header or C/ATT, YDS, TD columns
                header_text = table.get_text().lower()
                if 'passing' not in header_text and 'c/att' not in header_text:
                    continue
                
                # Find player links in this passing table
                for link in table.find_all('a', href=True):
                    href = link['href']
                    
                    # QB links look like: /nfl/player/_/id/3139477/patrick-mahomes
                    if '/nfl/player/_/id/' in href:
                        parts = href.split('/id/')[1].split('/')
                        if len(parts) >= 2:
                            player_id = parts[0]
                            
                            # Get player name from link text
                            player_name = link.get_text().strip()
                            
                            if player_name and player_id.isdigit() and len(player_name) > 2:
                                qb_info = {
                                    'name': player_name,
                                    'id': int(player_id)
                                }
                                
                                # Avoid duplicates
                                if not any(q['id'] == qb_info['id'] for q in qbs):
                                    qbs.append(qb_info)
            except:
                continue
        
        return qbs
        
    except Exception as e:
        return []


def find_all_playoff_qbs(start_year, end_year):
    """
    Find all QBs who played in playoffs during the year range.
    
    Uses ESPN's playoff stats page which lists all players.
    Much more reliable than scraping individual box scores.
    
    Returns:
        dict: {qb_id: {'name': str, 'seasons': [years]}}
    """
    print(f"\n{'='*80}")
    print(f"FINDING ALL PLAYOFF QBs ({start_year}-{end_year})")
    print(f"{'='*80}")
    
    all_qbs = {}
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    for year in range(start_year, end_year + 1):
        print(f"\n{year}:")
        print(f"  Scraping playoff stats page...", end=" ")
        
        # ESPN's playoff passing stats page - shows all players who played
        url = f"https://www.espn.com/nfl/stats/player/_/season/{year}/seasontype/3"
        
        try:
            response = requests.get(url, headers=headers, timeout=10, verify=False)
            
            if response.status_code != 200:
                print(f"❌ Failed")
                continue
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            year_qbs = set()
            
            # Find all player links on the page
            for link in soup.find_all('a', href=True):
                href = link['href']
                
                # Player links look like: /nfl/player/_/id/4432577/cj-stroud
                if '/nfl/player/_/id/' in href:
                    try:
                        parts = href.split('/id/')[1].split('/')
                        if len(parts) >= 2:
                            player_id = int(parts[0])
                            
                            # Get display name from link text
                            player_name = link.get_text().strip()
                            
                            if player_name and len(player_name) > 2:
                                if player_id not in all_qbs:
                                    all_qbs[player_id] = {
                                        'name': player_name,
                                        'seasons': []
                                    }
                                
                                if year not in all_qbs[player_id]['seasons']:
                                    all_qbs[player_id]['seasons'].append(year)
                                    year_qbs.add(player_name)
                    except:
                        continue
            
            print(f"✅ {len(year_qbs)} players")
            
            time.sleep(0.5)  # Be nice to servers
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n{'='*80}")
    print(f"TOTAL PLAYOFF PLAYERS FOUND: {len(all_qbs)}")
    print(f"{'='*80}")
    print(f"Note: Includes all players (QBs, RBs, WRs, etc.)")
    print(f"      Filtering to QBs only happens during game log scraping (10+ attempts)")
    
    return all_qbs


def scrape_qb_playoff_gamelogs(athlete_id, athlete_name, year):
    """Scrape playoff game logs for a QB for a specific year."""
    url = f"https://www.espn.com/nfl/player/gamelog/_/id/{athlete_id}/type/nfl/year/{year}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10, verify=False)
        if response.status_code != 200:
            return None
        
        soup = BeautifulSoup(response.content, 'html.parser')
        tables = soup.find_all('table')
        
        for table in tables:
            try:
                df = pd.read_html(StringIO(str(table)))[0]
                
                if len(df.columns) == 0:
                    continue
                
                first_col = df.columns[0]
                if isinstance(first_col, tuple) and 'Postseason' in str(first_col[0]):
                    # Process the table (same as scrape_espn_qb_playoff_gamelogs.py)
                    new_cols = []
                    for c in df.columns:
                        if isinstance(c, tuple):
                            category, stat = c
                            if category in ['Passing', 'Rushing']:
                                new_cols.append(f"{category.lower()}_{stat.lower()}")
                            else:
                                new_cols.append(stat)
                        else:
                            new_cols.append(c)
                    
                    df.columns = new_cols
                    df = df[~df['Date'].str.contains('Championship|Divisional|Wild Card', na=False, case=False)]
                    df = df[df['Date'].str.contains(r'\d', na=False)]
                    
                    df['season'] = year
                    df['athlete'] = athlete_name
                    df['athlete_id'] = athlete_id
                    
                    # Rename columns
                    column_mapping = {
                        'Date': 'date',
                        'OPP': 'opponent',
                        'Result': 'result',
                        'passing_cmp': 'completions',
                        'passing_att': 'attempts',
                        'passing_yds': 'passing_yards',
                        'passing_td': 'passing_tds',
                        'passing_int': 'interceptions',
                    }
                    df = df.rename(columns=column_mapping)
                    
                    # Filter for games with 10+ attempts (starters only)
                    if 'attempts' in df.columns:
                        df['attempts'] = pd.to_numeric(df['attempts'], errors='coerce')
                        # Drop rows where attempts is NaN or < 10
                        df = df[df['attempts'].notna() & (df['attempts'] >= MIN_ATTEMPTS)]
                    else:
                        # No attempts column = not a QB, return None
                        return None
                    
                    return df if len(df) > 0 else None
                    
            except:
                continue
        
        return None
        
    except:
        return None


def scrape_all_qbs(qbs_dict, start_year, end_year, playoff_schedule=None):
    """
    Scrape playoff history for all QBs.
    
    Strategy:
    - For each QB found, scrape years specified by start_year to end_year
    - By default, uses same years as discovery (can be overridden with --scrape-start/--scrape-end)
    - Adds playoff_round column using schedule mapping
    
    Args:
        qbs_dict: Dict from find_all_playoff_qbs()
        start_year: Earliest year to scrape
        end_year: Latest year to scrape
        playoff_schedule: Optional schedule mapping for rounds
        
    Returns:
        pd.DataFrame: All playoff games for all QBs
    """
    print(f"\n{'='*80}")
    print(f"SCRAPING PLAYOFF HISTORY FOR {len(qbs_dict)} PLAYERS")
    print(f"{'='*80}")
    print(f"Year range: {start_year}-{end_year}")
    
    all_games = []
    total_qbs = len(qbs_dict)
    
    for idx, (qb_id, qb_info) in enumerate(qbs_dict.items(), 1):
        qb_name = qb_info['name']
        
        print(f"\n[{idx}/{total_qbs}] {qb_name} (ID: {qb_id})", flush=True)
        
        qb_games = []
        years_data = {}  # Store year -> list of games with rounds
        
        # Scrape ALL years, not just discovery years
        # This gets their complete playoff history
        total_years = end_year - start_year + 1
        for year_idx, year in enumerate(range(start_year, end_year + 1), 1):
            # Show progress every 5 years
            if year_idx % 5 == 0:
                print(f"  Progress: {year_idx}/{total_years} years...", end="\r", flush=True)
            
            df = scrape_qb_playoff_gamelogs(qb_id, qb_name, year)
            
            if df is not None and len(df) > 0:
                qb_games.append(df)
                years_data[year] = df
            
            time.sleep(0.3)
        
        # Print years with games (detailed format with stats)
        if years_data:
            print("")  # New line after progress
            for year, df_year in years_data.items():
                # Add playoff rounds if schedule available
                if playoff_schedule:
                    df_year['playoff_round'] = df_year.apply(
                        lambda row: match_game_to_round(row['date'], row['season'], playoff_schedule),
                        axis=1
                    )
                    
                    print(f"  {year}:")
                    
                    # Sort by round order
                    round_order = ['Wild Card', 'Divisional', 'Conference Championship', 'Super Bowl']
                    for round_name in round_order:
                        round_games = df_year[df_year['playoff_round'] == round_name]
                        
                        for _, game in round_games.iterrows():
                            # Parse result for W/L
                            result = game.get('result', '')
                            win_loss = 'W' if result.startswith('W') else 'L' if result.startswith('L') else ''
                            
                            # Get stats (handle missing values)
                            pass_yds = int(game.get('passing_yards', 0)) if pd.notna(game.get('passing_yards')) else 0
                            pass_tds = int(game.get('passing_tds', 0)) if pd.notna(game.get('passing_tds')) else 0
                            ints = int(game.get('interceptions', 0)) if pd.notna(game.get('interceptions')) else 0
                            
                            # Get rushing yards (column is 'rushing_yds' not 'rushing_yards')
                            rush_yds = 0
                            if 'rushing_yds' in game:
                                rush_yds = int(game['rushing_yds']) if pd.notna(game['rushing_yds']) else 0
                            
                            print(f"    {round_name}: {win_loss} {result}, {pass_yds} pass yds, {pass_tds} pass TD, {ints} INT, {rush_yds} rush yds")
                else:
                    print(f"  {year}: {len(df_year)} games")
        
        if qb_games:
            qb_df = pd.concat(qb_games, ignore_index=True)
            
            # Add playoff round if schedule provided (if not already added)
            if playoff_schedule and 'playoff_round' not in qb_df.columns:
                qb_df['playoff_round'] = qb_df.apply(
                    lambda row: match_game_to_round(row['date'], row['season'], playoff_schedule),
                    axis=1
                )
            
            all_games.append(qb_df)
            print(f"\n  ✅ Total: {len(qb_df)} playoff games")
        else:
            print(f"\n  ❌ No playoff games found (filtered out: not a QB or <10 attempts)")
    
    if all_games:
        combined = pd.concat(all_games, ignore_index=True)
        print(f"\n{'='*80}")
        print(f"TOTAL PLAYOFF GAMES SCRAPED: {len(combined)}")
        print(f"{'='*80}")
        return combined
    
    return None


def build_playoff_schedule(start_year, end_year):
    """
    Build exact mapping of dates to playoff rounds by scraping ESPN schedule.
    
    For each year, scrapes each playoff week's schedule page:
    - Week 1: Wild Card
    - Week 2: Divisional
    - Week 3: Conference Championship
    - Week 4: Super Bowl (2001-2008) OR Pro Bowl (2009+)
    - Week 5: NA or Super Bowl (2009+)
    
    Returns:
        dict: {year: {date_str: round_name}}
        Example: {2024: {'1/11': 'Wild Card', '1/18': 'Divisional', ...}}
    """
    print(f"\n{'='*80}")
    print(f"BUILDING PLAYOFF SCHEDULE ({start_year}-{end_year})")
    print(f"{'='*80}")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    # Week to round mapping (changes in 2009)
    base_weeks = {
        1: 'Wild Card',
        2: 'Divisional',
        3: 'Conference Championship'
    }
    
    schedule = {}
    
    for year in range(start_year, end_year + 1):
        print(f"  {year}...", end=" ", flush=True)
        schedule[year] = {}
        total_games = 0
        
        # Super Bowl was Week 4 before 2009, Week 5 from 2009 onwards
        if year <= 2008:
            week_to_round = {**base_weeks, 4: 'Super Bowl'}
        else:
            week_to_round = {**base_weeks, 5: 'Super Bowl'}
        
        for week, round_name in week_to_round.items():
            url = f"https://www.espn.com/nfl/schedule/_/week/{week}/year/{year}/seasontype/3"
            
            try:
                response = requests.get(url, headers=headers, timeout=10, verify=False)
                if response.status_code != 200:
                    continue
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract dates from the schedule page
                # Look for date headers like "Saturday, January 11, 2025"
                date_headers = soup.find_all(['h2', 'div'], class_=lambda x: x and 'date' in x.lower() if x else False)
                
                # Also try finding dates in text
                import re
                page_text = soup.get_text()
                
                # Pattern: "January 11" or "February 9"
                date_patterns = re.findall(r'(January|February)\s+(\d{1,2})', page_text)
                
                month_map = {'January': 1, 'February': 2}
                
                for month_name, day in date_patterns:
                    if month_name in month_map:
                        date_key = f"{month_map[month_name]}/{int(day)}"
                        schedule[year][date_key] = round_name
                        total_games += 1
                
                time.sleep(0.2)
                
            except Exception as e:
                continue
        
        # Remove duplicates (same date might appear multiple times)
        unique_dates = len(set(schedule[year].keys()))
        print(f"✅ {unique_dates} unique dates")
    
    print(f"{'='*80}\n")
    return schedule


def match_game_to_round(date_str, season, playoff_schedule):
    """
    Match a game to its playoff round using the schedule mapping.
    
    Falls back to heuristics for older years where ESPN data is incomplete.
    
    Args:
        date_str: Date like "Sat 1/11" or "Sun 2/9"  
        season: Year
        playoff_schedule: Schedule dict from build_playoff_schedule()
        
    Returns:
        str: Round name or "Unknown"
    """
    try:
        # Parse date from format like "Sat 1/11"
        parts = date_str.split()
        if len(parts) < 2:
            return "Unknown"
        
        date_part = parts[1]  # "1/11"
        month, day = map(int, date_part.split('/'))
        
        # Look up in schedule first
        if season in playoff_schedule:
            if date_part in playoff_schedule[season]:
                return playoff_schedule[season][date_part]
        
        # Fallback heuristics for older years with incomplete ESPN data
        # Super Bowl is always in February (and not in schedule for 2001-2008)
        if month == 2:
            return "Super Bowl"
        
        # For January games without schedule data, use date ranges
        # These are historical patterns that hold true across all years
        if month == 1:
            if day <= 10:
                return "Wild Card"
            elif day <= 17:
                return "Divisional"
            elif day <= 31:
                return "Conference Championship"
        
        return "Unknown"
        
    except:
        return "Unknown"


def file_exists_in_s3(filename, bucket=S3_BUCKET, prefix=S3_PREFIX):
    """Check if a file exists in S3."""
    try:
        import boto3
        s3_client = boto3.client('s3')
        s3_key = f"{prefix}/{filename}"
        
        try:
            s3_client.head_object(Bucket=bucket, Key=s3_key)
            return True
        except:
            return False
    except ImportError:
        return False
    except:
        return False


def sanitize_filename(name):
    """Sanitize name for filename."""
    return name.lower().replace(' ', '_').replace('-', '_').replace('.', '')


def save_to_local(df, filename, test_mode=False):
    """Save DataFrame to local filesystem."""
    save_dir = Path(TEST_DATA_DIR if test_mode else LOCAL_DATA_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / filename
    df.to_csv(file_path, index=False)
    return file_path


def save_to_s3(df, filename, bucket=S3_BUCKET, prefix=S3_PREFIX):
    """Save DataFrame to S3."""
    try:
        import boto3
        from io import StringIO
        
        s3_client = boto3.client('s3')
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        
        s3_key = f"{prefix}/{filename}"
        s3_client.put_object(
            Bucket=bucket,
            Key=s3_key,
            Body=csv_buffer.getvalue(),
            ContentType='text/csv'
        )
        
        return f"s3://{bucket}/{s3_key}"
        
    except ImportError:
        raise ImportError("boto3 required for S3 uploads")
    except Exception as e:
        raise Exception(f"S3 upload failed: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Scrape ALL NFL playoff QB game logs',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--start-year', type=int, default=2001, help='Start year for QB discovery')
    parser.add_argument('--end-year', type=int, default=2026, help='End year for QB discovery')
    parser.add_argument('--scrape-start', type=int, help='Start year for scraping history (default: same as --start-year)')
    parser.add_argument('--scrape-end', type=int, help='End year for scraping history (default: same as --end-year)')
    parser.add_argument('--export', action='store_true', help='Export to CSV')
    parser.add_argument('--test-mode', action='store_true', help='Test mode: save to ~/Downloads/tmp')
    parser.add_argument('--local-only', action='store_true', help='Save local instead of S3')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files (default: skip existing)')
    parser.add_argument('--skip-discovery', action='store_true', help='Skip QB discovery (use existing QB list)')
    parser.add_argument('--qb-list-file', type=str, help='Path to JSON file with QB list')
    
    args = parser.parse_args()
    
    # Step 1: Find all playoff QBs (or load from file)
    if args.skip_discovery and args.qb_list_file:
        print(f"Loading QB list from {args.qb_list_file}")
        with open(args.qb_list_file, 'r') as f:
            qbs_dict = json.load(f)
    else:
        qbs_dict = find_all_playoff_qbs(args.start_year, args.end_year)
        
        # Save QB list for future use
        qb_list_file = f"playoff_qbs_{args.start_year}_{args.end_year}.json"
        
        if args.test_mode:
            # Save to test directory
            save_dir = Path(TEST_DATA_DIR)
            save_dir.mkdir(parents=True, exist_ok=True)
            qb_list_path = save_dir / qb_list_file
            with open(qb_list_path, 'w') as f:
                json.dump(qbs_dict, f, indent=2)
            print(f"\n💾 Saved QB list to: {qb_list_path}")
        elif args.local_only:
            # Save to local data directory
            save_dir = Path(LOCAL_DATA_DIR)
            save_dir.mkdir(parents=True, exist_ok=True)
            qb_list_path = save_dir / qb_list_file
            with open(qb_list_path, 'w') as f:
                json.dump(qbs_dict, f, indent=2)
            print(f"\n💾 Saved QB list to: {qb_list_path}")
        else:
            # Save to S3
            try:
                import boto3
                s3_client = boto3.client('s3')
                s3_key = f"{S3_PREFIX}/{qb_list_file}"
                s3_client.put_object(
                    Bucket=S3_BUCKET,
                    Key=s3_key,
                    Body=json.dumps(qbs_dict, indent=2),
                    ContentType='application/json'
                )
                print(f"\n☁️  Saved QB list to: s3://{S3_BUCKET}/{s3_key}")
            except Exception as e:
                print(f"\n⚠️  S3 upload failed: {e}")
                # Fallback to local
                qb_list_path = Path(LOCAL_DATA_DIR) / qb_list_file
                qb_list_path.parent.mkdir(parents=True, exist_ok=True)
                with open(qb_list_path, 'w') as f:
                    json.dump(qbs_dict, f, indent=2)
                print(f"💾 Saved QB list locally to: {qb_list_path}")
    
    if not qbs_dict:
        print("❌ No QBs found")
        return
    
    # Step 2: Determine scraping year range
    SCRAPE_START_YEAR = args.scrape_start if args.scrape_start else args.start_year
    SCRAPE_END_YEAR = args.scrape_end if args.scrape_end else args.end_year
    
    print(f"\n{'='*80}")
    print(f"SCRAPING CONFIGURATION")
    print(f"{'='*80}")
    print(f"Discovery years: {args.start_year}-{args.end_year} ({len(qbs_dict)} players found)")
    print(f"Scraping years: {SCRAPE_START_YEAR}-{SCRAPE_END_YEAR} ({SCRAPE_END_YEAR - SCRAPE_START_YEAR + 1} years per player)")
    print(f"  Note: By default, scraping years match discovery years")
    print(f"  Use --scrape-start 2001 to get full playoff history for each QB")
    print(f"Total requests: ~{len(qbs_dict)} players × {SCRAPE_END_YEAR - SCRAPE_START_YEAR + 1} years = {len(qbs_dict) * (SCRAPE_END_YEAR - SCRAPE_START_YEAR + 1)} requests")
    print(f"Estimated time: ~{int(len(qbs_dict) * (SCRAPE_END_YEAR - SCRAPE_START_YEAR + 1) * 0.5 / 60)} minutes")
    print(f"{'='*80}")
    
    # Step 3: Build playoff schedule (for round mapping)
    playoff_schedule = build_playoff_schedule(SCRAPE_START_YEAR, SCRAPE_END_YEAR)
    
    # Step 4: Scrape all QBs' game logs
    df = scrape_all_qbs(qbs_dict, SCRAPE_START_YEAR, SCRAPE_END_YEAR, playoff_schedule)
    
    if df is not None and len(df) > 0:
        print(f"\n{'='*80}")
        print(f"SAMPLE DATA")
        print(f"{'='*80}")
        
        # Show columns including playoff_round if it exists
        display_cols = ['athlete', 'season', 'date', 'opponent', 'result']
        if 'playoff_round' in df.columns:
            display_cols.append('playoff_round')
        display_cols.extend(['completions', 'attempts', 'passing_yards', 'passing_tds'])
        
        print(df[display_cols].head(20))
        
        # Export
        if args.export:
            
            print(f"\n{'='*80}")
            print(f"EXPORTING FILES")
            print(f"{'='*80}")
            
            # 1. Save individual QB files
            print(f"\n1️⃣  Saving individual QB files...")
            unique_qbs = df[['athlete', 'athlete_id']].drop_duplicates()
            
            skipped = 0
            saved = 0
            
            for _, row in unique_qbs.iterrows():
                qb_name = row['athlete']
                qb_id = row['athlete_id']
                qb_df = df[df['athlete'] == qb_name].copy()
                
                # Filename: name + player ID (no timestamp)
                qb_sanitized = sanitize_filename(qb_name)
                qb_filename = f"nfl_qb_playoff_gamelogs_{qb_sanitized}_{qb_id}.csv"
                
                # Check if file exists (unless overwrite flag)
                if not args.overwrite:
                    if args.test_mode or args.local_only:
                        # Check local filesystem
                        save_dir = Path(TEST_DATA_DIR if args.test_mode else LOCAL_DATA_DIR)
                        file_path = save_dir / qb_filename
                        if file_path.exists():
                            print(f"   ⏭️  {qb_name}: already exists, skipping")
                            skipped += 1
                            continue
                    else:
                        # Check S3
                        if file_exists_in_s3(qb_filename):
                            print(f"   ⏭️  {qb_name}: already exists in S3, skipping")
                            skipped += 1
                            continue
                
                # Save the file
                if args.test_mode:
                    qb_path = save_to_local(qb_df, qb_filename, test_mode=True)
                    print(f"   💾 {qb_name}: {qb_path.name}")
                elif args.local_only:
                    qb_path = save_to_local(qb_df, qb_filename, test_mode=False)
                    print(f"   💾 {qb_name}: {qb_path.name}")
                else:
                    try:
                        s3_uri = save_to_s3(qb_df, qb_filename)
                        print(f"   ☁️  {qb_name}: {qb_filename}")
                    except Exception as e:
                        print(f"   ⚠️  {qb_name}: S3 failed - {e}")
                        continue
                
                saved += 1
            
            # 2. Save master file (all QBs combined) - always with timestamp
            print(f"\n2️⃣  Saving master file (all QBs combined)...")
            date_str = datetime.now().strftime('%Y%m%d')
            master_filename = f"nfl_all_qb_playoff_gamelogs_{args.start_year}_{args.end_year}_{date_str}.csv"
            
            if args.test_mode:
                master_path = save_to_local(df, master_filename, test_mode=True)
                print(f"   💾 Master file: {master_path.absolute()}")
            elif args.local_only:
                master_path = save_to_local(df, master_filename, test_mode=False)
                print(f"   💾 Master file: {master_path.absolute()}")
            else:
                try:
                    s3_uri = save_to_s3(df, master_filename)
                    print(f"   ☁️  Master file: {s3_uri}")
                except Exception as e:
                    print(f"   ⚠️  Master file: S3 failed - {e}")
            
            print(f"\n{'='*80}")
            print(f"EXPORT COMPLETE")
            print(f"  Individual QB files: {saved} saved, {skipped} skipped")
            print(f"  Master file: {len(df)} total games")
            print(f"  Filter: Only games with {MIN_ATTEMPTS}+ attempts (starters)")
            if not args.overwrite and skipped > 0:
                print(f"\nℹ️  Tip: Use --overwrite to re-scrape existing files")
            print(f"{'='*80}")


if __name__ == '__main__':
    main()

