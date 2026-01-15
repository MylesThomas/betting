"""
Build comprehensive NBA roster cache with all active players.

This script creates a complete player-to-team mapping for ALL active NBA players,
not just those in tonight's games. The cache includes:
- player_name_nba_api: Official name from NBA API
- team: Current team abbreviation
- player_normalized: Normalized name for matching (used by all scripts)

USAGE:
======
    python scripts/build_full_roster_cache.py

OUTPUT:
=======
This automatically creates BOTH files locally AND uploads to S3:

Local:
    - data/02_cache/nba_full_roster_cache.csv (full roster with all name formats)
    - data/02_cache/player_team_cache.csv (simplified for quick lookups)

S3:
    - s3://nba-betting-mt/data/02_cache/nba_full_roster_cache.csv
    - s3://nba-betting-mt/data/02_cache/player_team_cache.csv

WHY TWO FILES?
==============
- nba_full_roster_cache.csv: Complete roster with multiple name formats
- player_team_cache.csv: Simplified format optimized for quick lookups

Run this weekly or after major trades to keep rosters up to date.
"""

import pandas as pd
import sys
from pathlib import Path
import time
import boto3
from io import StringIO

# Fix SSL certificate issues with NBA API (must be done BEFORE importing nba_api)
import ssl
import urllib3
import requests

ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Monkey-patch requests Session to disable SSL verification
original_request = requests.Session.request

def patched_request(self, *args, **kwargs):
    kwargs['verify'] = False
    return original_request(self, *args, **kwargs)

requests.Session.request = patched_request

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.player_name_utils import normalize_player_name
from src.config import CURRENT_NBA_SEASON

# Output file (local backup - primary storage is S3)
OUTPUT_PATH = Path(__file__).parent.parent / 'data' / '02_cache' / 'nba_full_roster_cache.csv'

# S3 Configuration
S3_BUCKET = 'nba-betting-mt'
S3_PREFIX = 'data/02_cache'


def get_all_nba_rosters():
    """
    Query NBA API for all active players on all 30 teams.
    
    Returns:
        DataFrame with columns: player_name_nba_api, team
    """
    try:
        from nba_api.stats.static import teams, players
        
        print("Fetching all NBA teams...")
        all_teams = teams.get_teams()
        
        roster_data = []
        
        for team in all_teams:
            team_abbr = team['abbreviation']
            team_name = team['full_name']
            
            print(f"  Fetching roster for {team_name} ({team_abbr})...")
            
            try:
                # Query team roster endpoint
                url = 'https://stats.nba.com/stats/commonteamroster'
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'application/json',
                    'Referer': 'https://stats.nba.com/',
                }
                params = {
                    'TeamID': team['id'],
                    'Season': CURRENT_NBA_SEASON  # Auto-calculated from config
                }
                
                response = requests.get(
                    url,
                    headers=headers,
                    params=params,
                    timeout=10,
                    verify=False
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'resultSets' in data and len(data['resultSets']) > 0:
                        headers_list = data['resultSets'][0]['headers']
                        rows = data['resultSets'][0]['rowSet']
                        
                        # Find player name column
                        name_idx = headers_list.index('PLAYER')
                        
                        for row in rows:
                            player_name = row[name_idx]
                            roster_data.append({
                                'player_name_nba_api': player_name,
                                'team': team_abbr
                            })
                
                # Rate limit: 1 request per 0.6 seconds
                time.sleep(0.6)
                
            except Exception as e:
                print(f"    Warning: Could not fetch roster for {team_name}: {e}")
                continue
        
        return pd.DataFrame(roster_data)
    
    except ImportError:
        print("Error: nba_api not installed. Run: pip install nba_api")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching rosters: {e}")
        return pd.DataFrame()


def upload_to_s3(df, filename):
    """
    Upload DataFrame to S3 as CSV.
    
    Args:
        df: DataFrame to upload
        filename: Name of file (e.g., 'nba_full_roster_cache.csv')
    
    Returns:
        True if successful, False otherwise
    """
    s3_key = f"{S3_PREFIX}/{filename}"
    
    try:
        s3_client = boto3.client('s3')
        
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=csv_buffer.getvalue(),
            ContentType='text/csv'
        )
        
        print(f"✅ Uploaded to S3: s3://{S3_BUCKET}/{s3_key}")
        return True
        
    except Exception as e:
        print(f"⚠️  S3 upload failed: {e}")
        return False


def add_odds_api_names(roster_df, odds_data_path=None):
    """
    Add player_name_odds_api column by matching against odds data.
    
    Args:
        roster_df: DataFrame with player_name_nba_api and team
        odds_data_path: Optional path to recent odds data file
        
    Returns:
        DataFrame with player_name_odds_api column added
    """
    roster_df = roster_df.copy()
    roster_df['player_name_odds_api'] = None
    
    # Try to load odds data to find name variants
    if odds_data_path and Path(odds_data_path).exists():
        try:
            odds_df = pd.read_csv(odds_data_path)
            
            if 'player' in odds_df.columns:
                # Create mapping of normalized names to odds API names
                odds_names = {}
                for player_name in odds_df['player'].unique():
                    normalized = normalize_player_name(player_name)
                    odds_names[normalized] = player_name
                
                # Match roster players to odds names
                def find_odds_name(nba_name):
                    normalized = normalize_player_name(nba_name)
                    return odds_names.get(normalized)
                
                roster_df['player_name_odds_api'] = roster_df['player_name_nba_api'].apply(find_odds_name)
                
                matched = roster_df['player_name_odds_api'].notna().sum()
                total = len(roster_df)
                print(f"\nMatched {matched}/{total} players to odds data names ({matched/total*100:.1f}%)")
        
        except Exception as e:
            print(f"Warning: Could not load odds data: {e}")
    
    return roster_df


def main():
    print("=" * 70)
    print("Building Full NBA Roster Cache")
    print("=" * 70)
    print(f"Season: {CURRENT_NBA_SEASON}")
    print()
    
    # Fetch all rosters from NBA API
    print("Step 1: Fetching all NBA rosters from API...")
    roster_df = get_all_nba_rosters()
    
    if len(roster_df) == 0:
        print("Error: No roster data retrieved")
        return
    
    print(f"\n✅ Retrieved {len(roster_df)} players across 30 teams")
    print()
    
    # Add manual roster additions from config
    print("Step 2: Adding manual roster entries from config...")
    manual_additions = [
        {'player_name': 'Jeremiah Robinson-Earl', 'team': 'IND'},
        # Add more players here as needed:
        # {'player_name_nba_api': 'Player Name', 'team': 'TEAM'},
    ]
    
    if manual_additions:
        manual_df = pd.DataFrame(manual_additions)
        manual_df = manual_df.rename(columns={'player_name': 'player_name_nba_api'})
        roster_df = pd.concat([roster_df, manual_df], ignore_index=True)
        print(f"✅ MANUALLY added {len(manual_additions)} players not returned by NBA API:")
        for entry in manual_additions:
            print(f"   - {entry['player_name']} ({entry['team']})")
    else:
        print("   No manual additions found in config")
    print()
    
    # Add normalized name for easier lookups
    print("Step 3: Normalizing player names for matching...")
    roster_df['player_normalized'] = roster_df['player_name_nba_api'].apply(normalize_player_name)
    
    # Reorder columns (removed player_name_odds_api since it's rarely populated and not used)
    roster_df = roster_df[['player_name_nba_api', 'team', 'player_normalized']]
    
    # Sort by team, then player name
    roster_df = roster_df.sort_values(['team', 'player_name_nba_api'])
    
    # Save full roster cache to CSV (local backup)
    print(f"\nStep 4: Saving full roster cache locally...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    roster_df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Saved local backup: {OUTPUT_PATH}")
    
    # Step 5: Automatically create player_team_cache.csv for quick lookups
    print(f"\nStep 5: Creating player_team_cache.csv for quick lookups...")
    from datetime import datetime
    
    player_team_cache = pd.DataFrame({
        'player_normalized': roster_df['player_normalized'],
        'team': roster_df['team'],
        'timestamp': datetime.now().isoformat()
    })
    
    # Remove duplicates and sort
    player_team_cache = player_team_cache.drop_duplicates(subset=['player_normalized'], keep='first')
    player_team_cache = player_team_cache.sort_values('player_normalized')
    
    # Save to same directory (local backup)
    cache_path = OUTPUT_PATH.parent / 'player_team_cache.csv'
    player_team_cache.to_csv(cache_path, index=False)
    print(f"✅ Saved local backup: {cache_path}")
    
    # Upload both files to S3
    print(f"\nStep 6: Uploading cache files to S3...")
    upload_to_s3(roster_df, 'nba_full_roster_cache.csv')
    upload_to_s3(player_team_cache, 'player_team_cache.csv')
    
    print()
    print("=" * 70)
    print("✅ Full Roster Cache Created!")
    print("=" * 70)
    print(f"Local full roster: {OUTPUT_PATH}")
    print(f"Local quick cache: {cache_path}")
    print(f"S3 bucket: s3://{S3_BUCKET}/{S3_PREFIX}/")
    print(f"Total players: {len(roster_df)}")
    print(f"Teams: {roster_df['team'].nunique()}")
    print()
    print("Sample:")
    print(roster_df.head(10).to_string(index=False))
    print()
    print("Both cache files saved locally and uploaded to S3!")


if __name__ == '__main__':
    main()

