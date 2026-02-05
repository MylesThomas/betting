"""
Debug why Dejounte Murray only shows 1 stint.

From Basketball Reference, he should have:
- SAS (2016-2022)
- ATL (2022-2024)
- NOP (2024-present)
"""

import pandas as pd
import ssl
import urllib3
import requests
import sys
from pathlib import Path

# Fix SSL
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

original_request = requests.Session.request
def patched_request(self, *args, **kwargs):
    kwargs['verify'] = False
    return original_request(self, *args, **kwargs)
requests.Session.request = patched_request

# Add src to path
repo_root = Path(__file__).resolve()
while not (repo_root / '.gitignore').exists():
    repo_root = repo_root.parent
sys.path.insert(0, str(repo_root))

from nba_api.stats.endpoints import playergamelog, commonplayerinfo
from nba_api.stats.static import players
from src.player_team_history.name_normalization import normalize_player_name

def find_player():
    """Find Dejounte Murray in NBA API."""
    all_players = players.get_players()
    for p in all_players:
        if 'dejounte' in p['full_name'].lower() and 'murray' in p['full_name'].lower():
            return p
    return None

def get_career_info(player_id):
    """Get career seasons."""
    player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
    df = player_info.get_data_frames()[0]
    print("\n📋 Career Info:")
    print(df[['FROM_YEAR', 'TO_YEAR', 'ROSTERSTATUS']].to_string())
    return df

def extract_team_from_matchup(matchup):
    """Extract player's team from MATCHUP string."""
    if pd.isna(matchup):
        return None
    if '@' in matchup:
        return matchup.split('@')[0].strip()
    elif 'vs.' in matchup:
        return matchup.split('vs.')[0].strip()
    return None

def get_all_game_logs(player_id, from_year, to_year):
    """Fetch game logs for all seasons."""
    import time
    
    all_games = []
    
    for year in range(int(from_year), int(to_year) + 1):
        season = f"{year}-{str(year + 1)[-2:]}"
        print(f"\n🔄 Fetching season {season}...")
        
        try:
            gamelog = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season
            )
            
            df = gamelog.get_data_frames()[0]
            
            if not df.empty:
                print(f"   ✅ {len(df)} games")
                # Add TEAM column from MATCHUP
                if 'MATCHUP' in df.columns:
                    df['TEAM'] = df['MATCHUP'].apply(extract_team_from_matchup)
                    teams = df['TEAM'].unique()
                    print(f"   Teams: {', '.join([str(t) for t in teams if pd.notna(t)])}")
                all_games.append(df)
            else:
                print(f"   ⚠️ No games")
            
            time.sleep(0.7)
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:60]}")
            time.sleep(2)
    
    if not all_games:
        return pd.DataFrame()
    
    combined = pd.concat(all_games, ignore_index=True)
    return combined

def main():
    print("="*80)
    print("🔍 DEBUG: Dejounte Murray Team History")
    print("="*80)
    
    # Find player
    player = find_player()
    if not player:
        print("❌ Player not found in NBA API")
        return
    
    print(f"\n✅ Found: {player['full_name']}")
    print(f"   Player ID: {player['id']}")
    print(f"   Active: {player.get('is_active', 'Unknown')}")
    
    # Get career info
    career_info = get_career_info(player['id'])
    from_year = int(career_info['FROM_YEAR'].iloc[0])
    to_year = int(career_info['TO_YEAR'].iloc[0])
    
    print(f"\n📅 Career span: {from_year} to {to_year}")
    
    # Fetch all game logs
    print(f"\n🏀 Fetching game logs across all seasons...")
    game_logs = get_all_game_logs(player['id'], from_year, to_year)
    
    if game_logs.empty:
        print("\n❌ No game logs found!")
        return
    
    print(f"\n✅ Total games fetched: {len(game_logs)}")
    
    # Convert dates and sort
    game_logs['GAME_DATE'] = pd.to_datetime(game_logs['GAME_DATE'], format='mixed')
    game_logs = game_logs.sort_values('GAME_DATE')
    
    # Show team changes
    print("\n📊 Team changes detected:")
    print("-"*80)
    
    game_logs['team_change'] = game_logs['TEAM'] != game_logs['TEAM'].shift()
    game_logs['team_stint'] = game_logs['team_change'].cumsum()
    
    for stint_id, stint_games in game_logs.groupby('team_stint'):
        team = stint_games['TEAM'].iloc[0]
        first_game = stint_games['GAME_DATE'].min()
        last_game = stint_games['GAME_DATE'].max()
        num_games = len(stint_games)
        
        print(f"Stint {stint_id}: {team}")
        print(f"  From: {first_game.date()}")
        print(f"  To:   {last_game.date()}")
        print(f"  Games: {num_games}")
        print()
    
    # Show sample of game logs
    print("\n📋 Sample game logs:")
    print(game_logs[['GAME_DATE', 'MATCHUP', 'TEAM']].head(20).to_string())
    print("\n...")
    print(game_logs[['GAME_DATE', 'MATCHUP', 'TEAM']].tail(20).to_string())

if __name__ == '__main__':
    main()
