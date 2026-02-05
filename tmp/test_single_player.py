"""
Test fetching a single player to see what's actually happening.
"""

import pandas as pd
import ssl
import urllib3
import requests
import sys
from pathlib import Path
import time

# Fix SSL
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

original_request = requests.Session.request
def patched_request(self, *args, **kwargs):
    kwargs['verify'] = False
    return original_request(self, *args, **kwargs)
requests.Session.request = patched_request

import requests.sessions
original_init = requests.sessions.Session.__init__
def patched_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    self.verify = False
requests.sessions.Session.__init__ = patched_init

# Add src to path
repo_root = Path(__file__).resolve()
while not (repo_root / '.gitignore').exists():
    repo_root = repo_root.parent
sys.path.insert(0, str(repo_root))

from nba_api.stats.endpoints import playergamelog, commonplayerinfo
from nba_api.stats.static import players
from src.config import CURRENT_NBA_SEASON

# Import the actual functions from the build script
import importlib.util
spec = importlib.util.spec_from_file_location(
    "build_script",
    str(repo_root / "src/player_team_history/tmp/build_team_history_incremental.py")
)
build_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_module)

def main():
    print("="*80)
    print("Testing Dejounte Murray with build script functions")
    print("="*80)
    
    player_name = "Dejounte Murray"
    
    # Find player ID
    print(f"\n1. Finding player ID for: {player_name}")
    player_id = build_module.find_player_id(player_name)
    print(f"   Player ID: {player_id}")
    
    if not player_id:
        print("   ❌ Player not found!")
        return
    
    # Get career seasons
    print(f"\n2. Getting career seasons...")
    seasons = build_module.get_career_seasons(player_id)
    print(f"   Seasons ({len(seasons)}): {seasons}")
    
    # Fetch game logs
    print(f"\n3. Fetching game logs...")
    game_logs = build_module.fetch_player_game_log_working_endpoint(
        player_name, 
        player_id, 
        verbose=True
    )
    
    if game_logs.empty:
        print("   ❌ No game logs fetched!")
        return
    
    print(f"\n   ✅ Total games fetched: {len(game_logs)}")
    
    # Check TEAM column
    print(f"\n4. Checking TEAM column...")
    if 'TEAM' in game_logs.columns:
        teams = game_logs['TEAM'].unique()
        print(f"   Teams found: {list(teams)}")
        print(f"   Rows per team:")
        for team in teams:
            if pd.notna(team):
                count = (game_logs['TEAM'] == team).sum()
                print(f"      {team}: {count} games")
    else:
        print("   ❌ No TEAM column!")
    
    # Create team history
    print(f"\n5. Creating team history...")
    player_history = build_module.create_team_history_from_gamelogs(game_logs, player_name)
    
    if not player_history.empty:
        print(f"   ✅ Created {len(player_history)} stints:")
        print(player_history.to_string(index=False))
    else:
        print("   ❌ No history created!")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    main()
