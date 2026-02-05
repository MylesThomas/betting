"""
Fetch championship futures for NFL, NBA, NCAAF, and NCAAB from The Odds API.

Context:
Unified futures data fetcher for all major sports championships:
- NFL Super Bowl
- NBA Championship
- NCAA Football (College Football Playoff)
- NCAA Basketball (March Madness)

Includes team records from ESPN API where available (NFL, NBA).

Purpose:
- Fetch championship futures odds for 4 major sports
- Fetch team records from ESPN API (NFL, NBA only)
- Save timestamped files to track odds movement over time
- Support weekly futures analysis workflow

Usage:
    cd /Users/thomasmyles/dev/betting
    python3 scripts/fetch_nfl_nba_championship_futures.py

Output:
- data/01_input/the-odds-api/nfl/futures/nfl_super_bowl_futures_YYYYMMDD_HHMMSS.csv
- data/01_input/the-odds-api/nba/futures/nba_championship_futures_YYYYMMDD_HHMMSS.csv
- data/01_input/the-odds-api/ncaaf/futures/ncaaf_championship_futures_YYYYMMDD_HHMMSS.csv
- data/01_input/the-odds-api/ncaab/futures/ncaab_championship_futures_YYYYMMDD_HHMMSS.csv

CSV columns:
- sport, bookmaker, team, odds, implied_prob, record (from ESPN API when available)

API docs: 
- The Odds API: https://the-odds-api.com/liveapi/guides/v4/
- ESPN API: https://gist.github.com/akeaswaran/b48b02f1c94f873c6655e7129910fc3b
"""

# SSL Fix for macOS - must be imported BEFORE requests
import ssl
import urllib3

ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import requests
import pandas as pd
import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
import boto3
from io import StringIO

# Add src to path for odds_utils
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root / 'src'))

from odds_utils import odds_to_implied_probability
from nfl_team_utils import NFL_TEAM_MAPPING, NFL_ABBR_TO_FULL
from ncaa_team_utils import ESPN_NCAAB_TEAM_IDS, ESPN_NCAAF_TEAM_IDS
from config_loader import get_config

# Monkey-patch requests to disable SSL verification
original_request = requests.Session.request

def patched_request(self, *args, **kwargs):
    kwargs['verify'] = False
    return original_request(self, *args, **kwargs)

requests.Session.request = patched_request

# Load environment variables
load_dotenv()

API_KEY = os.getenv('ODDS_API_KEY')
BASE_URL = 'https://api.the-odds-api.com/v4'
ESPN_API_BASE = 'https://sports.core.api.espn.com/v2/sports'

# Load config for season values
CONFIG = get_config()
ESPN_SEASONS = CONFIG['espn_seasons']

# S3 Configuration
S3_BUCKET = 'the-odds-api-mt'
AWS_REGION = os.getenv('AWS_REGION_NAME', 'us-east-2')


def fetch_futures(sport_key):
    """Fetch futures odds for a given sport key"""
    url = f"{BASE_URL}/sports/{sport_key}/odds/"
    
    params = {
        'apiKey': API_KEY,
        'regions': 'us',
        'oddsFormat': 'american'
    }
    
    response = requests.get(url, params=params)
    
    # Print API usage
    remaining = response.headers.get('x-requests-remaining')
    used = response.headers.get('x-requests-used')
    if remaining:
        print(f"API Usage: {used} used, {remaining} remaining")
    
    response.raise_for_status()
    return response.json()


def fetch_nfl_team_records_from_espn():
    """
    Fetch NFL team records from ESPN API (regular season + playoffs combined).
    
    Returns:
        dict: Team name (full) -> record string (e.g., "12-6" for 11-6 reg season + 1-0 playoffs)
    """
    print("📊 Fetching NFL team records from ESPN API...")
    
    # ESPN team IDs (mapping from abbreviation to ESPN team ID)
    espn_team_ids = {
        'ARI': 22, 'ATL': 1, 'BAL': 33, 'BUF': 2, 'CAR': 29, 'CHI': 3,
        'CIN': 4, 'CLE': 5, 'DAL': 6, 'DEN': 7, 'DET': 8, 'GB': 9,
        'HOU': 34, 'IND': 11, 'JAX': 30, 'KC': 12, 'LAR': 14, 'LAC': 24,
        'LV': 13, 'MIA': 15, 'MIN': 16, 'NE': 17, 'NO': 18, 'NYG': 19,
        'NYJ': 20, 'PHI': 21, 'PIT': 23, 'SEA': 26, 'SF': 25, 'TB': 27,
        'TEN': 10, 'WAS': 28
    }
    
    current_season = ESPN_SEASONS['nfl']  # From config: 2025-26 NFL season
    team_records = {}
    
    for abbr, team_id in espn_team_ids.items():
        try:
            # Fetch regular season record (types/2)
            reg_season_url = f"{ESPN_API_BASE}/football/leagues/nfl/seasons/{current_season}/types/2/teams/{team_id}/record"
            response = requests.get(reg_season_url, timeout=5, verify=False)
            
            reg_wins = 0
            reg_losses = 0
            reg_ties = 0
            
            if response.status_code == 200:
                data = response.json()
                items = data.get('items', [])
                
                for item in items:
                    if item.get('type') == 'total':
                        stats = item.get('stats', [])
                        for stat in stats:
                            if stat.get('name') == 'wins':
                                reg_wins = int(stat.get('value', 0))
                            elif stat.get('name') == 'losses':
                                reg_losses = int(stat.get('value', 0))
                            elif stat.get('name') == 'ties':
                                reg_ties = int(stat.get('value', 0))
            
            # Fetch playoff record (types/3)
            playoff_url = f"{ESPN_API_BASE}/football/leagues/nfl/seasons/{current_season}/types/3/teams/{team_id}/record"
            playoff_response = requests.get(playoff_url, timeout=5, verify=False)
            
            playoff_wins = 0
            playoff_losses = 0
            
            if playoff_response.status_code == 200:
                playoff_data = playoff_response.json()
                playoff_items = playoff_data.get('items', [])
                
                for item in playoff_items:
                    if item.get('type') == 'total':
                        stats = item.get('stats', [])
                        for stat in stats:
                            if stat.get('name') == 'wins':
                                playoff_wins = int(stat.get('value', 0))
                            elif stat.get('name') == 'losses':
                                playoff_losses = int(stat.get('value', 0))
            
            # Combine regular season + playoffs
            total_wins = reg_wins + playoff_wins
            total_losses = reg_losses + playoff_losses
            
            # Format record string
            if reg_ties > 0:
                record = f"{total_wins}-{total_losses}-{reg_ties}"
            else:
                record = f"{total_wins}-{total_losses}"
            
            # Get full team name
            full_name = NFL_ABBR_TO_FULL[abbr]
            team_records[full_name] = record
            
        except Exception as e:
            print(f"   ⚠️  Error fetching {abbr}: {e}")
            continue
    
    print(f"   ✅ Fetched records for {len(team_records)}/32 teams (regular season + playoffs)\n")
    return team_records


def fetch_nba_team_records_from_espn():
    """
    Fetch NBA team records from ESPN API.
    
    Returns:
        dict: Team name (full) -> record string (e.g., "25-10")
    """
    print("📊 Fetching NBA team records from ESPN API...")
    
    # ESPN team IDs for NBA
    espn_nba_team_ids = {
        'ATL': 1, 'BOS': 2, 'BKN': 17, 'CHA': 30, 'CHI': 4, 'CLE': 5,
        'DAL': 6, 'DEN': 7, 'DET': 8, 'GSW': 9, 'HOU': 10, 'IND': 11,
        'LAC': 12, 'LAL': 13, 'MEM': 29, 'MIA': 14, 'MIL': 15, 'MIN': 16,
        'NOP': 3, 'NYK': 18, 'OKC': 25, 'ORL': 19, 'PHI': 20, 'PHX': 21,
        'POR': 22, 'SAC': 23, 'SAS': 24, 'TOR': 28, 'UTA': 26, 'WAS': 27
    }
    
    # NBA team name mapping (ESPN abbreviation to full name)
    nba_team_names = {
        'ATL': 'Atlanta Hawks', 'BOS': 'Boston Celtics', 'BKN': 'Brooklyn Nets',
        'CHA': 'Charlotte Hornets', 'CHI': 'Chicago Bulls', 'CLE': 'Cleveland Cavaliers',
        'DAL': 'Dallas Mavericks', 'DEN': 'Denver Nuggets', 'DET': 'Detroit Pistons',
        'GSW': 'Golden State Warriors', 'HOU': 'Houston Rockets', 'IND': 'Indiana Pacers',
        'LAC': 'Los Angeles Clippers', 'LAL': 'Los Angeles Lakers', 'MEM': 'Memphis Grizzlies',
        'MIA': 'Miami Heat', 'MIL': 'Milwaukee Bucks', 'MIN': 'Minnesota Timberwolves',
        'NOP': 'New Orleans Pelicans', 'NYK': 'New York Knicks', 'OKC': 'Oklahoma City Thunder',
        'ORL': 'Orlando Magic', 'PHI': 'Philadelphia 76ers', 'PHX': 'Phoenix Suns',
        'POR': 'Portland Trail Blazers', 'SAC': 'Sacramento Kings', 'SAS': 'San Antonio Spurs',
        'TOR': 'Toronto Raptors', 'UTA': 'Utah Jazz', 'WAS': 'Washington Wizards'
    }
    
    current_season = ESPN_SEASONS['nba']  # From config: 2025-26 NBA season
    team_records = {}
    
    for abbr, team_id in espn_nba_team_ids.items():
        try:
            url = f"{ESPN_API_BASE}/basketball/leagues/nba/seasons/{current_season}/types/2/teams/{team_id}/record"
            response = requests.get(url, timeout=5, verify=False)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract wins and losses
                items = data.get('items', [])
                wins = 0
                losses = 0
                
                for item in items:
                    stat_type = item.get('type', '')
                    if stat_type == 'total':
                        stats = item.get('stats', [])
                        for stat in stats:
                            if stat.get('name') == 'wins':
                                wins = int(stat.get('value', 0))
                            elif stat.get('name') == 'losses':
                                losses = int(stat.get('value', 0))
                
                record = f"{wins}-{losses}"
                
                # Get full team name
                full_name = nba_team_names[abbr]
                team_records[full_name] = record
                
        except Exception as e:
            print(f"   ⚠️  Error fetching {abbr}: {e}")
            continue
    
    print(f"   ✅ Fetched records for {len(team_records)}/30 teams\n")
    return team_records


def fetch_ncaab_team_records_from_espn():
    """
    Fetch NCAAB team records from ESPN API.
    
    Note: ESPN team IDs for college basketball are different from other sports.
    This function includes major teams that typically appear in championship futures.
    
    Returns:
        dict: Team name (The Odds API format with mascots) -> record string (e.g., "25-3")
    """
    print("📊 Fetching NCAAB team records from ESPN API...")
    
    # Use team IDs from ncaa_team_utils (imported at top of file)
    # See docs/ESPN_TEAM_IDS.md for more team IDs to add
    
    current_season = ESPN_SEASONS['ncaab']  # From config: 2025-26 NCAAB season
    team_records = {}
    
    for team_name, team_id in ESPN_NCAAB_TEAM_IDS.items():
        try:
            url = f"{ESPN_API_BASE}/basketball/leagues/mens-college-basketball/seasons/{current_season}/types/2/teams/{team_id}/record"
            response = requests.get(url, timeout=5, verify=False)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract overall record from items
                items = data.get('items', [])
                for item in items:
                    if item.get('type') == 'total' and item.get('name') == 'overall':
                        # Use the summary field (e.g., "31-3")
                        record = item.get('summary', '')
                        if record:
                            team_records[team_name] = record
                            break
                
        except Exception as e:
            # Silently continue - not all teams may have data
            continue
    
    print(f"   ✅ Fetched records for {len(team_records)} NCAA teams\n")
    return team_records


def fetch_ncaaf_team_records_from_espn():
    """
    Fetch NCAAF team records from ESPN API.
    
    Note: Currently only fetching CFP playoff teams. Expand as needed.
    TODO 2027: Update team list for next season's playoff teams.
    
    Returns:
        dict: Team name (The Odds API format with mascots) -> record string (e.g., "12-1")
    """
    print("📊 Fetching NCAAF team records from ESPN API...")
    
    # Use team IDs from ncaa_team_utils (imported at top of file)
    # Currently only CFP playoff teams - expand as needed
    
    current_season = ESPN_SEASONS['ncaaf']  # From config: 2025-26 NCAAF season
    team_records = {}
    
    for team_name, team_id in ESPN_NCAAF_TEAM_IDS.items():
        try:
            url = f"{ESPN_API_BASE}/football/leagues/college-football/seasons/{current_season}/types/2/teams/{team_id}/record"
            response = requests.get(url, timeout=5, verify=False)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract overall record from items
                items = data.get('items', [])
                for item in items:
                    if item.get('type') == 'total' and item.get('name') == 'overall':
                        # Use the summary field (e.g., "12-1")
                        record = item.get('summary', '')
                        if record:
                            team_records[team_name] = record
                            break
                
        except Exception as e:
            # Silently continue - not all teams may have data
            continue
    
    print(f"   ✅ Fetched records for {len(team_records)} NCAAF teams\n")
    return team_records


def parse_futures_to_df(data, sport_name, team_records=None):
    """
    Parse futures data into a DataFrame.
    
    Note: The Odds API with oddsFormat='american' should return proper American odds.
    Positive odds (underdogs) are returned as positive integers (e.g., 150 means +150).
    Negative odds (favorites) are returned as negative integers (e.g., -110).
    
    Some bookmakers may have data quality issues - we store odds as-is from the API.
    
    Also calculates implied probability for easier sorting and analysis.
    
    Args:
        data: API response data
        sport_name: 'NFL' or 'NBA'
        team_records: Optional dict of team name -> record string
    """
    futures_list = []
    
    for item in data:
        sport_key = item.get('sport_key')
        
        for bookmaker in item.get('bookmakers', []):
            bookmaker_name = bookmaker['key']
            
            for market in bookmaker.get('markets', []):
                market_key = market['key']
                
                for outcome in market.get('outcomes', []):
                    odds = outcome.get('price')
                    implied_prob = odds_to_implied_probability(odds)
                    team = outcome.get('name')
                    
                    # Get record from API if available
                    record = team_records.get(team, '') if team_records else ''
                    
                    futures_list.append({
                        'sport': sport_name,
                        'bookmaker': bookmaker_name,
                        'team': team,
                        'odds': odds,
                        'implied_prob': implied_prob,
                        'record': record
                    })
    
    return pd.DataFrame(futures_list)


def get_s3_key(sport: str, timestamp: datetime) -> str:
    """
    Generate S3 key for futures data.
    
    Args:
        sport: 'nfl', 'nba', 'ncaaf', 'ncaab'
        timestamp: Datetime for filename
    
    Returns:
        S3 key like: nfl/futures/2026-01-10/nfl_super_bowl_futures_20260110_143052.csv
    """
    date_str = timestamp.strftime('%Y-%m-%d')
    timestamp_str = timestamp.strftime('%Y%m%d_%H%M%S')
    
    sport_filename_map = {
        'nfl': f'nfl_super_bowl_futures_{timestamp_str}.csv',
        'nba': f'nba_championship_futures_{timestamp_str}.csv',
        'ncaaf': f'ncaaf_championship_futures_{timestamp_str}.csv',
        'ncaab': f'ncaab_championship_futures_{timestamp_str}.csv'
    }
    
    filename = sport_filename_map[sport.lower()]
    return f"{sport}/futures/{date_str}/{filename}"


def save_to_s3(df: pd.DataFrame, sport: str, timestamp: datetime) -> str:
    """
    Save DataFrame to S3 as CSV.
    
    Args:
        df: DataFrame with futures data
        sport: 'nfl', 'nba', 'ncaaf', 'ncaab'
        timestamp: Datetime for filename
    
    Returns:
        S3 key where file was saved
    """
    s3_key = get_s3_key(sport, timestamp)
    
    # Convert DataFrame to CSV string
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    
    # Upload to S3
    s3_client = boto3.client('s3', region_name=AWS_REGION)
    
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=csv_buffer.getvalue(),
            ContentType='text/csv'
        )
        
        print(f"☁️  Saved to s3://{S3_BUCKET}/{s3_key}")
        return s3_key
        
    except Exception as e:
        print(f"⚠️  S3 upload failed: {e}")
        print("   (Local file still saved)")
        return None


def main():
    """Main test function"""
    
    if not API_KEY:
        print("❌ ERROR: ODDS_API_KEY not found in .env file")
        print("Add your API key to .env: ODDS_API_KEY=your_key_here")
        return
    
    # Generate single timestamp for this entire fetch session
    fetch_timestamp = datetime.now()
    timestamp_str = fetch_timestamp.strftime('%Y%m%d_%H%M%S')
    
    print("="*80)
    print("FETCHING CHAMPIONSHIP FUTURES (NFL, NBA, NCAAF, NCAAB)")
    print("="*80)
    print(f"Timestamp: {timestamp_str}\n")
    
    # Test NFL Super Bowl futures
    print("\n🏈 Fetching NFL Super Bowl futures...")
    try:
        # Fetch NFL team records from ESPN
        nfl_records = fetch_nfl_team_records_from_espn()
        
        nfl_data = fetch_futures('americanfootball_nfl_super_bowl_winner')
        df_nfl = parse_futures_to_df(nfl_data, 'NFL', nfl_records)
        
        if not df_nfl.empty:
            print(f"✅ Found {len(df_nfl)} odds from {df_nfl['bookmaker'].nunique()} bookmakers")
            
            # Show top 10 favorites - use best odds per team (highest implied prob = lowest odds)
            # Get best odds for each team (max odds = most favorable to bettor)
            best_odds_per_team = df_nfl.loc[df_nfl.groupby('team')['odds'].idxmax()]
            # Sort by implied probability descending (highest prob = biggest favorite)
            best_odds_per_team = best_odds_per_team.sort_values('implied_prob', ascending=False)
            
            print("\nTop 10 Super Bowl Favorites (Best Available Odds):")
            print("-" * 75)
            for i, row in enumerate(best_odds_per_team.head(10).itertuples(), 1):
                odds_str = f"+{int(row.odds)}" if row.odds > 0 else f"{int(row.odds)}"
                print(f"{i:2d}. {row.team:<30} {odds_str:>7}  ({row.implied_prob*100:>5.1f}% @ {row.bookmaker})")
            
            # Save to CSV with timestamp
            output_file = repo_root / f'data/01_input/the-odds-api/nfl/futures/nfl_super_bowl_futures_{timestamp_str}.csv'
            os.makedirs(output_file.parent, exist_ok=True)
            df_nfl.to_csv(output_file, index=False)
            print(f"\n💾 Saved locally: {output_file}")
            
            # Upload to S3
            save_to_s3(df_nfl, sport='nfl', timestamp=fetch_timestamp)
        else:
            print("⚠️  No NFL futures data found")
            
    except Exception as e:
        print(f"❌ Error fetching NFL futures: {e}")
    
    # Test NBA Championship futures
    print("\n\n🏀 Fetching NBA Championship futures...")
    try:
        # Fetch team records from ESPN API first
        nba_records = fetch_nba_team_records_from_espn()
        
        nba_data = fetch_futures('basketball_nba_championship_winner')
        df_nba = parse_futures_to_df(nba_data, 'NBA', nba_records)
        
        if not df_nba.empty:
            print(f"✅ Found {len(df_nba)} odds from {df_nba['bookmaker'].nunique()} bookmakers")
            
            # Show top 10 favorites - use best odds per team (highest implied prob = lowest odds)
            # Get best odds for each team (max odds = most favorable to bettor)
            best_odds_per_team = df_nba.loc[df_nba.groupby('team')['odds'].idxmax()]
            # Sort by implied probability descending (highest prob = biggest favorite)
            best_odds_per_team = best_odds_per_team.sort_values('implied_prob', ascending=False)
            
            print("\nTop 10 NBA Championship Favorites (Best Available Odds):")
            print("-" * 75)
            for i, row in enumerate(best_odds_per_team.head(10).itertuples(), 1):
                odds_str = f"+{int(row.odds)}" if row.odds > 0 else f"{int(row.odds)}"
                print(f"{i:2d}. {row.team:<30} {odds_str:>7}  ({row.implied_prob*100:>5.1f}% @ {row.bookmaker})")
            
            # Save to CSV with timestamp
            output_file = repo_root / f'data/01_input/the-odds-api/nba/futures/nba_championship_futures_{timestamp_str}.csv'
            os.makedirs(output_file.parent, exist_ok=True)
            df_nba.to_csv(output_file, index=False)
            print(f"\n💾 Saved locally: {output_file}")
            
            # Upload to S3
            save_to_s3(df_nba, sport='nba', timestamp=fetch_timestamp)
        else:
            print("⚠️  No NBA futures data found")
            
    except Exception as e:
        print(f"❌ Error fetching NBA futures: {e}")
    
    # Test NCAA Football Championship futures (College Football Playoff)
    print("\n\n🏈 Fetching NCAA Football Championship futures...")
    ncaaf_records = fetch_ncaaf_team_records_from_espn()
    try:
        ncaaf_data = fetch_futures('americanfootball_ncaaf_championship_winner')
        df_ncaaf = parse_futures_to_df(ncaaf_data, 'NCAAF', ncaaf_records)
        
        if not df_ncaaf.empty:
            print(f"✅ Found {len(df_ncaaf)} odds from {df_ncaaf['bookmaker'].nunique()} bookmakers")
            
            # Show top 10 favorites - use best odds per team (highest implied prob = lowest odds)
            # Get best odds for each team (max odds = most favorable to bettor)
            best_odds_per_team = df_ncaaf.loc[df_ncaaf.groupby('team')['odds'].idxmax()]
            # Sort by implied probability descending (highest prob = biggest favorite)
            best_odds_per_team = best_odds_per_team.sort_values('implied_prob', ascending=False)
            
            print("\nTop 10 CFP National Championship Favorites (Best Available Odds):")
            print("-" * 75)
            for i, row in enumerate(best_odds_per_team.head(10).itertuples(), 1):
                odds_str = f"+{int(row.odds)}" if row.odds > 0 else f"{int(row.odds)}"
                print(f"{i:2d}. {row.team:<30} {odds_str:>7}  ({row.implied_prob*100:>5.1f}% @ {row.bookmaker})")
            
            # Save to CSV with timestamp
            output_file = repo_root / f'data/01_input/the-odds-api/ncaaf/futures/ncaaf_championship_futures_{timestamp_str}.csv'
            os.makedirs(output_file.parent, exist_ok=True)
            df_ncaaf.to_csv(output_file, index=False)
            print(f"\n💾 Saved locally: {output_file}")
            
            # Upload to S3
            save_to_s3(df_ncaaf, sport='ncaaf', timestamp=fetch_timestamp)
        else:
            print("⚠️  No NCAA Football futures data found")
            
    except Exception as e:
        print(f"❌ Error fetching NCAA Football futures: {e}")
    
    # Test NCAA Basketball Championship futures (March Madness)
    print("\n\n🏀 Fetching NCAA Basketball Championship futures...")
    try:
        # Fetch team records from ESPN API first
        ncaab_records = fetch_ncaab_team_records_from_espn()
        
        ncaab_data = fetch_futures('basketball_ncaab_championship_winner')
        df_ncaab = parse_futures_to_df(ncaab_data, 'NCAAB', ncaab_records)
        
        if not df_ncaab.empty:
            print(f"✅ Found {len(df_ncaab)} odds from {df_ncaab['bookmaker'].nunique()} bookmakers")
            
            # Show top 10 favorites - use best odds per team (highest implied prob = lowest odds)
            # Get best odds for each team (max odds = most favorable to bettor)
            best_odds_per_team = df_ncaab.loc[df_ncaab.groupby('team')['odds'].idxmax()]
            # Sort by implied probability descending (highest prob = biggest favorite)
            best_odds_per_team = best_odds_per_team.sort_values('implied_prob', ascending=False)
            
            print("\nTop 10 March Madness Championship Favorites (Best Available Odds):")
            print("-" * 75)
            for i, row in enumerate(best_odds_per_team.head(10).itertuples(), 1):
                odds_str = f"+{int(row.odds)}" if row.odds > 0 else f"{int(row.odds)}"
                print(f"{i:2d}. {row.team:<30} {odds_str:>7}  ({row.implied_prob*100:>5.1f}% @ {row.bookmaker})")
            
            # Save to CSV with timestamp
            output_file = repo_root / f'data/01_input/the-odds-api/ncaab/futures/ncaab_championship_futures_{timestamp_str}.csv'
            os.makedirs(output_file.parent, exist_ok=True)
            df_ncaab.to_csv(output_file, index=False)
            print(f"\n💾 Saved locally: {output_file}")
            
            # Upload to S3
            save_to_s3(df_ncaab, sport='ncaab', timestamp=fetch_timestamp)
        else:
            print("⚠️  No NCAA Basketball futures data found")
            
    except Exception as e:
        print(f"❌ Error fetching NCAA Basketball futures: {e}")
    
    print("\n" + "="*80)
    print("✅ TEST COMPLETE")
    print("="*80)
    
    # Show available sport keys for other futures
    print("\n📋 Other futures sport keys to try:")
    print("   - baseball_mlb_world_series_winner")
    print("   - icehockey_nhl_championship_winner")


if __name__ == "__main__":
    main()

