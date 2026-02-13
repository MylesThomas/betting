"""
Debug script to investigate why live boxscore data is not retrieving correctly.

Context:
- Game 401810642 (Memphis Grizzlies @ Denver Nuggets)
- PBP data shows Nikola Jokic with 9 pts, 5 reb, 5 ast at Q2 3:28
- Need to verify boxscore API is returning player stats correctly
- This is for validating Monte Carlo simulation inputs

Usage:
    python src/pbp_data/tmp/debug_live_boxscore.py --game-id 401810642
"""

import os
import sys
import json
import argparse
from pathlib import Path
import requests
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Find project root (look for .gitignore as marker)
current = Path(__file__).resolve()
root = None
for parent in current.parents:
    if (parent / '.gitignore').exists():
        root = parent
        break

if root:
    sys.path.insert(0, str(root))
    os.chdir(root)
else:
    print("⚠️  Could not find project root (no .gitignore found)")
    sys.exit(1)

# =============================================================================
# CONFIG
# =============================================================================

ESPN_BOXSCORE_API = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={game_id}"

# =============================================================================
# MAIN DEBUG FUNCTION
# =============================================================================

def debug_boxscore(game_id: str, save_response: bool = True):
    """
    Fetch and debug the boxscore for a given game.
    
    Args:
        game_id: ESPN game ID
        save_response: If True, save raw JSON response to file
    """
    print("="*80)
    print(f"DEBUG: LIVE BOXSCORE API")
    print("="*80)
    print()
    
    url = ESPN_BOXSCORE_API.format(game_id=game_id)
    print(f"🔗 URL: {url}")
    print()
    
    # Fetch data
    print("🔄 Fetching boxscore data...")
    try:
        response = requests.get(url, timeout=10, verify=False)
        response.raise_for_status()
        data = response.json()
        print("✅ Response received")
    except Exception as e:
        print(f"❌ Error fetching boxscore: {e}")
        return
    
    # Save raw response
    if save_response:
        output_path = Path("src/pbp_data/tmp") / f"debug_boxscore_{game_id}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"💾 Saved raw response to: {output_path}")
    
    print()
    print("="*80)
    print("STRUCTURE ANALYSIS")
    print("="*80)
    print()
    
    # Check top-level structure
    print("📋 Top-level keys:")
    for key in data.keys():
        print(f"   - {key}")
    print()
    
    # Check for boxscore key
    if 'boxscore' not in data:
        print("❌ ERROR: 'boxscore' key not found in response!")
        print("   Available keys:", list(data.keys()))
        return
    
    boxscore = data['boxscore']
    print("✅ Found 'boxscore' key")
    print()
    
    # Check players structure (correct path for player stats)
    if 'players' not in boxscore:
        print("❌ ERROR: 'players' key not found in boxscore!")
        print("   Available keys:", list(boxscore.keys()))
        return
    
    players_data = boxscore['players']
    print(f"✅ Found {len(players_data)} team entries in boxscore.players")
    print()
    
    # Analyze each team
    print("="*80)
    print("TEAM & PLAYER DATA")
    print("="*80)
    print()
    
    all_players = []
    
    for team_idx, team_data in enumerate(players_data):
        team_name = team_data.get('team', {}).get('displayName', 'Unknown')
        print(f"📊 Team {team_idx + 1}: {team_name}")
        print("-" * 80)
        
        # Check statistics structure
        statistics = team_data.get('statistics', [])
        print(f"   Statistics groups: {len(statistics)}")
        
        for stat_idx, stat_group in enumerate(statistics):
            # Get labels
            labels = stat_group.get('labels', [])
            print(f"   \n   Group {stat_idx + 1} - Labels: {labels}")
            
            # Get athletes
            athletes = stat_group.get('athletes', [])
            print(f"   Athletes in group: {len(athletes)}")
            
            # Parse each athlete
            for athlete in athletes:
                athlete_info = athlete.get('athlete', {})
                athlete_name = athlete_info.get('displayName', 'Unknown')
                athlete_id = athlete_info.get('id', 'Unknown')
                
                stats = athlete.get('stats', [])
                
                # Parse stats
                points = 0
                minutes = 0
                rebounds = 0
                assists = 0
                
                for i, label in enumerate(labels):
                    if i >= len(stats):
                        break
                    
                    stat_val = stats[i]
                    
                    if label == 'PTS':
                        try:
                            points = float(stat_val) if stat_val != '--' else 0
                        except:
                            points = 0
                    elif label == 'MIN':
                        try:
                            if ':' in str(stat_val):
                                mins, secs = str(stat_val).split(':')
                                minutes = float(mins) + float(secs) / 60
                            else:
                                minutes = float(stat_val) if stat_val != '--' else 0
                        except:
                            minutes = 0
                    elif label == 'REB':
                        try:
                            rebounds = float(stat_val) if stat_val != '--' else 0
                        except:
                            rebounds = 0
                    elif label == 'AST':
                        try:
                            assists = float(stat_val) if stat_val != '--' else 0
                        except:
                            assists = 0
                
                # Only show players with minutes
                if minutes > 0:
                    player_info = {
                        'team': team_name,
                        'name': athlete_name,
                        'id': athlete_id,
                        'minutes': round(minutes, 1),
                        'points': points,
                        'rebounds': rebounds,
                        'assists': assists,
                    }
                    all_players.append(player_info)
                    
                    print(f"      - {athlete_name:25} | {minutes:5.1f} min | {points:4.0f} pts | {rebounds:3.0f} reb | {assists:3.0f} ast")
        
        print()
    
    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print()
    print(f"✅ Total active players (minutes > 0): {len(all_players)}")
    print()
    
    # Look for Nikola Jokic specifically
    jokic = [p for p in all_players if 'jokic' in p['name'].lower()]
    if jokic:
        print("🎯 Found Nikola Jokic:")
        for p in jokic:
            print(f"   Name: {p['name']}")
            print(f"   Team: {p['team']}")
            print(f"   Minutes: {p['minutes']}")
            print(f"   Points: {p['points']}")
            print(f"   Rebounds: {p['rebounds']}")
            print(f"   Assists: {p['assists']}")
    else:
        print("⚠️  Nikola Jokic not found in active players!")
    
    print()
    print("="*80)
    print("✅ DEBUG COMPLETE")
    print("="*80)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Debug live boxscore API for a specific game"
    )
    parser.add_argument(
        '--game-id',
        type=str,
        default='401810642',
        help='ESPN game ID (default: 401810642 - MEM @ DEN game)'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save raw JSON response to file'
    )
    
    args = parser.parse_args()
    
    debug_boxscore(args.game_id, save_response=not args.no_save)


if __name__ == '__main__':
    main()
