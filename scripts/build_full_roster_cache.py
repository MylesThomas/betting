"""
Build comprehensive NBA roster cache with all active players.

This script creates a complete player-to-team mapping for ALL active NBA players,
not just those in tonight's games. The cache includes:
- player_name_nba_api: Official name from NBA API
- team: Current team abbreviation
- player_name_odds_api: Name variant from odds data (if available)
- player_normalized: Normalized name for matching

USAGE - FULL ORDER OF OPERATIONS:
==================================

Step 1: Build full roster cache from NBA API
    $ python scripts/build_full_roster_cache.py
    
    This creates: data/02_cache/nba_full_roster_cache.csv

Step 2: Convert to player_team_cache format (for Streamlit app)
    $ python3 << 'EOF'
    import pandas as pd
    from datetime import datetime
    
    # Load the full roster cache
    full_roster = pd.read_csv('data/02_cache/nba_full_roster_cache.csv')
    
    # Convert to player_team_cache format
    player_team_cache = pd.DataFrame({
        'player_normalized': full_roster['player_normalized'],
        'team': full_roster['team'],
        'timestamp': datetime.now().isoformat()
    })
    
    # Remove duplicates and sort
    player_team_cache = player_team_cache.drop_duplicates(subset=['player_normalized'], keep='first')
    player_team_cache = player_team_cache.sort_values('player_normalized')
    
    # Save to CSV
    player_team_cache.to_csv('data/02_cache/player_team_cache.csv', index=False)
    
    print(f"✅ Updated data/02_cache/player_team_cache.csv with {len(player_team_cache)} players")
    EOF

Step 3: Refresh Streamlit dashboard
    - Reload the page in your browser
    - The app will use the updated cache

WHY TWO FILES?
==============
- data/02_cache/nba_full_roster_cache.csv: Complete roster with multiple name formats
- data/02_cache/player_team_cache.csv: Simplified format optimized for quick lookups

Run this weekly or after major trades to keep rosters up to date.
"""

import pandas as pd
import sys
from pathlib import Path
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from src.player_name_utils import normalize_player_name
from src.config import CURRENT_NBA_SEASON
from src.config_loader import get_file_path

# Output file
OUTPUT_PATH = Path(__file__).parent.parent / get_file_path('nba_full_roster_cache')


def get_all_nba_rosters():
    """
    Query NBA API for all active players on all 30 teams.
    
    Returns:
        DataFrame with columns: player_name_nba_api, team
    """
    try:
        from nba_api.stats.static import teams, players
        import requests
        import urllib3
        
        # Disable SSL warnings
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
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
    
    # Find most recent odds data file
    print("Step 2: Matching player names to odds data...")
    data_dir = Path(__file__).parent.parent / "data" / "arbs"
    
    if data_dir.exists():
        arb_files = sorted(data_dir.glob("arb_*.csv"))
        if arb_files:
            most_recent = arb_files[-1]
            print(f"  Using: {most_recent.name}")
            roster_df = add_odds_api_names(roster_df, most_recent)
        else:
            print("  No odds data found - skipping name matching")
    
    # Add normalized name for easier lookups
    roster_df['player_normalized'] = roster_df['player_name_nba_api'].apply(normalize_player_name)
    
    # Add empty player_name_odds_api column if it doesn't exist
    if 'player_name_odds_api' not in roster_df.columns:
        roster_df['player_name_odds_api'] = ''
    
    # Reorder columns
    roster_df = roster_df[['player_name_nba_api', 'team', 'player_name_odds_api', 'player_normalized']]
    
    # Sort by team, then player name
    roster_df = roster_df.sort_values(['team', 'player_name_nba_api'])
    
    # Save to CSV
    print(f"\nStep 3: Saving to {OUTPUT_PATH}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    roster_df.to_csv(OUTPUT_PATH, index=False)
    
    print()
    print("=" * 70)
    print("✅ Full Roster Cache Created!")
    print("=" * 70)
    print(f"File: {OUTPUT_PATH}")
    print(f"Total players: {len(roster_df)}")
    print(f"Teams: {roster_df['team'].nunique()}")
    print(f"Players with odds names: {roster_df['player_name_odds_api'].notna().sum()}")
    print()
    print("Sample:")
    print(roster_df.head(10).to_string(index=False))
    print()
    print("You can now use this cache for instant team lookups!")


if __name__ == '__main__':
    main()

