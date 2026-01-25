"""
Build player-team cache from The Odds API data.

This script extracts player-team mappings from recent Odds API game files,
which always have current and accurate roster information (unlike the NBA API).

Context:
The NBA API's roster endpoint returns stale/incorrect data after trades.
Sportsbooks (via The Odds API) always have current player-team info.

Usage:
    python scripts/build_player_team_cache_from_odds.py
"""

import boto3
import json
import pandas as pd
from datetime import datetime
from collections import defaultdict
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from player_name_utils import normalize_player_name
from team_utils import TEAM_NAME_TO_ABBR

# S3 Configuration
ODDS_BUCKET = 'the-odds-api-mt'
ODDS_PREFIX = 'nba/live_game_odds/'
OUTPUT_BUCKET = 'nba-betting-mt'
OUTPUT_PREFIX = 'data/02_cache'

def get_recent_odds_files(limit=100):
    """Get recent odds API files from S3"""
    s3 = boto3.client('s3')
    
    response = s3.list_objects_v2(
        Bucket=ODDS_BUCKET,
        Prefix=ODDS_PREFIX,
        MaxKeys=limit
    )
    
    if 'Contents' not in response:
        return []
    
    # Sort by last modified (newest first)
    files = sorted(response['Contents'], key=lambda x: x['LastModified'], reverse=True)
    return [f['Key'] for f in files[:limit]]


def extract_player_teams_from_odds_file(s3, bucket, key):
    """Extract player-team mappings from a single odds file"""
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        data = json.loads(obj['Body'].read().decode('utf-8'))
        
        player_teams = {}
        
        # Get team names from the game
        away_team_full = data.get('away_team', '')
        home_team_full = data.get('home_team', '')
        
        # Convert to abbreviations
        away_team = TEAM_NAME_TO_ABBR.get(away_team_full)
        home_team = TEAM_NAME_TO_ABBR.get(home_team_full)
        
        if not away_team or not home_team:
            return {}
        
        # Extract player props
        for bookmaker in data.get('bookmakers', []):
            for market in bookmaker.get('markets', []):
                if market.get('key') == 'player_points':
                    for outcome in market.get('outcomes', []):
                        player_name = outcome.get('description', '')
                        if player_name:
                            # Normalize player name
                            player_normalized = normalize_player_name(player_name)
                            
                            # Player must be on one of these two teams
                            # We'll track both possibilities and consolidate later
                            if player_normalized not in player_teams:
                                player_teams[player_normalized] = {
                                    'name': player_name,
                                    'possible_teams': set()
                                }
                            
                            player_teams[player_normalized]['possible_teams'].add(away_team)
                            player_teams[player_normalized]['possible_teams'].add(home_team)
        
        return player_teams
    
    except Exception as e:
        print(f"   ⚠️  Error processing {key}: {e}")
        return {}


def main():
    print("="*70)
    print("Building Player-Team Cache from The Odds API")
    print("="*70)
    print()
    
    s3 = boto3.client('s3')
    
    # Step 1: Get recent odds files
    print("Step 1: Fetching recent odds files...")
    recent_files = get_recent_odds_files(limit=100)
    print(f"   ✅ Found {len(recent_files)} recent odds files")
    print()
    
    # Step 2: Extract player-team mappings
    print("Step 2: Extracting player-team mappings...")
    all_player_teams = defaultdict(lambda: {'possible_teams': set(), 'game_count': 0})
    
    for i, key in enumerate(recent_files, 1):
        if i % 20 == 0:
            print(f"   Processing file {i}/{len(recent_files)}...")
        
        player_teams = extract_player_teams_from_odds_file(s3, ODDS_BUCKET, key)
        
        for player_norm, data in player_teams.items():
            all_player_teams[player_norm]['possible_teams'].update(data['possible_teams'])
            all_player_teams[player_norm]['game_count'] += 1
    
    print(f"   ✅ Extracted {len(all_player_teams)} unique players")
    print()
    
    # Step 3: Resolve team assignments
    print("Step 3: Resolving team assignments...")
    resolved_players = []
    unresolved_players = []
    
    for player_norm, data in all_player_teams.items():
        teams = list(data['possible_teams'])
        
        # If player only appeared with one team, that's their team
        if len(teams) == 1:
            resolved_players.append({
                'player_normalized': player_norm,
                'team': teams[0],
                'confidence': 'high',
                'games': data['game_count']
            })
        elif len(teams) == 2:
            # Player appeared in games with 2 teams
            # This is expected - every player appears in games between 2 teams
            # We need more sophisticated logic here
            # For now, mark as unresolved
            unresolved_players.append({
                'player_normalized': player_norm,
                'teams': teams,
                'games': data['game_count']
            })
        else:
            # Player appeared in games with 3+ teams (suspicious)
            unresolved_players.append({
                'player_normalized': player_norm,
                'teams': teams,
                'games': data['game_count']
            })
    
    print(f"   ✅ Resolved {len(resolved_players)} players (appeared in games with only 1 unique team)")
    print(f"   ⚠️  {len(unresolved_players)} players need manual resolution")
    print()
    
    # Step 4: For unresolved players, use frequency heuristic
    print("Step 4: Resolving ambiguous players using game tracking...")
    
    # Re-process files to track which team each player is ACTUALLY on
    player_actual_teams = defaultdict(lambda: defaultdict(int))
    
    for key in recent_files:
        try:
            obj = s3.get_object(Bucket=ODDS_BUCKET, Key=key)
            data = json.loads(obj['Body'].read().decode('utf-8'))
            
            away_team_full = data.get('away_team', '')
            home_team_full = data.get('home_team', '')
            away_team = TEAM_NAME_TO_ABBR.get(away_team_full)
            home_team = TEAM_NAME_TO_ABBR.get(home_team_full)
            
            if not away_team or not home_team:
                continue
            
            # Track players by checking which games they appear in
            players_in_game = set()
            for bookmaker in data.get('bookmakers', []):
                for market in bookmaker.get('markets', []):
                    if market.get('key') == 'player_points':
                        for outcome in market.get('outcomes', []):
                            player_name = outcome.get('description', '')
                            if player_name:
                                player_norm = normalize_player_name(player_name)
                                players_in_game.add(player_norm)
            
            # Each player in this game is on either away or home team
            # Count how many games each player appears in with each team
            for player_norm in players_in_game:
                player_actual_teams[player_norm][away_team] += 1
                player_actual_teams[player_norm][home_team] += 1
        
        except Exception:
            continue
    
    # For unresolved players, pick the team they appeared most often with
    for unresolved in unresolved_players:
        player_norm = unresolved['player_normalized']
        team_counts = player_actual_teams[player_norm]
        
        if team_counts:
            # Pick team with highest count
            best_team = max(team_counts.items(), key=lambda x: x[1])[0]
            resolved_players.append({
                'player_normalized': player_norm,
                'team': best_team,
                'confidence': 'medium',
                'games': unresolved['games']
            })
    
    print(f"   ✅ Resolved {len(resolved_players)} total players")
    print()
    
    # Step 5: Create cache DataFrame
    print("Step 5: Creating cache...")
    df_cache = pd.DataFrame(resolved_players)
    df_cache = df_cache[['player_normalized', 'team']]  # Only keep essential columns
    df_cache['timestamp'] = datetime.now().isoformat()
    df_cache = df_cache.sort_values('player_normalized')
    
    # Save locally
    local_path = Path(__file__).parent.parent / 'data' / '02_cache' / 'player_team_cache.csv'
    local_path.parent.mkdir(parents=True, exist_ok=True)
    df_cache.to_csv(local_path, index=False)
    print(f"   ✅ Saved locally: {local_path}")
    
    # Upload to S3
    from io import StringIO
    csv_buffer = StringIO()
    df_cache.to_csv(csv_buffer, index=False)
    
    s3_key = f"{OUTPUT_PREFIX}/player_team_cache.csv"
    s3.put_object(
        Bucket=OUTPUT_BUCKET,
        Key=s3_key,
        Body=csv_buffer.getvalue(),
        ContentType='text/csv'
    )
    print(f"   ✅ Uploaded to S3: s3://{OUTPUT_BUCKET}/{s3_key}")
    print()
    
    print("="*70)
    print("✅ Player-Team Cache Built!")
    print("="*70)
    print(f"Total players: {len(df_cache)}")
    print(f"Source: The Odds API (last {len(recent_files)} games)")
    print()
    
    # Show some sample mappings
    print("Sample mappings:")
    sample_players = ['Brook Lopez', 'Jake Laravia', 'Deandre Ayton', 'Al Horford', 'Marcus Smart']
    for player in sample_players:
        player_norm = normalize_player_name(player)
        match = df_cache[df_cache['player_normalized'] == player_norm]
        if not match.empty:
            team = match.iloc[0]['team']
            print(f"   {player}: {team}")
        else:
            print(f"   {player}: NOT FOUND")


if __name__ == '__main__':
    main()
