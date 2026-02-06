"""
Get play-by-play data for Bucks vs Pelicans game on Feb 4, 2026.

Goal: Extract minute-by-minute player scoring with timestamps.
"""

import json
import requests
from datetime import datetime
import pandas as pd

# =============================================================================
# GET GAMES FROM FEB 4, 2026
# =============================================================================

def get_games_on_date(date_str):
    """
    Get all NBA games on a specific date.
    
    Args:
        date_str: Date in YYYYMMDD format (e.g., "20260204")
    
    Returns:
        dict: Scoreboard data
    """
    # ESPN API endpoint with date parameter
    url = f"http://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_str}"
    
    print(f"Fetching games for {date_str}...")
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Error: {response.status_code}")
        return None


def find_game(scoreboard_data, team1, team2):
    """
    Find a specific game by team names.
    
    Args:
        scoreboard_data: Scoreboard JSON from ESPN
        team1, team2: Team names (e.g., "Bucks", "Pelicans")
    
    Returns:
        game_id, game_data
    """
    if not scoreboard_data or 'events' not in scoreboard_data:
        return None, None
    
    for event in scoreboard_data['events']:
        competitions = event.get('competitions', [])
        if not competitions:
            continue
        
        competition = competitions[0]
        competitors = competition.get('competitors', [])
        
        if len(competitors) >= 2:
            team_names = [comp['team']['displayName'] for comp in competitors]
            team_short = [comp['team']['shortDisplayName'] for comp in competitors]
            
            # Check if both teams match
            team_strings = ' '.join(team_names + team_short).lower()
            
            if team1.lower() in team_strings and team2.lower() in team_strings:
                game_id = event['id']
                home_team = competitors[0]['team']['displayName']
                away_team = competitors[1]['team']['displayName']
                
                print(f"\n✅ Found game: {away_team} @ {home_team}")
                print(f"   Game ID: {game_id}")
                
                return game_id, event
    
    return None, None


def get_play_by_play(game_id):
    """
    Get play-by-play data for a specific game.
    
    Args:
        game_id: ESPN game ID
    
    Returns:
        dict: Full game data including plays
    """
    url = f"http://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={game_id}"
    
    print(f"\nFetching play-by-play for game {game_id}...")
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        
        if 'plays' in data:
            print(f"✅ Got {len(data['plays'])} plays")
            return data
        else:
            print("❌ No plays data in response")
            print(f"Available keys: {list(data.keys())}")
            return None
    else:
        print(f"❌ Error: {response.status_code}")
        return None


def parse_play_by_play(game_data):
    """
    Parse play-by-play data into structured format.
    
    Returns:
        DataFrame with columns: play_num, quarter, time, description, score, scoring_player, points_value
    """
    plays = game_data.get('plays', [])
    
    parsed_plays = []
    
    for play in plays:
        play_dict = {
            'play_id': play.get('id'),
            'sequence_number': play.get('sequenceNumber'),
            'quarter': play.get('period', {}).get('number'),
            'time': play.get('clock', {}).get('displayValue'),
            'description': play.get('text', ''),
            'away_score': play.get('awayScore'),
            'home_score': play.get('homeScore'),
            'is_scoring_play': play.get('scoringPlay', False),
            'score_value': play.get('scoreValue', 0),
        }
        
        # Get team info
        team_info = play.get('team', {})
        play_dict['team_id'] = team_info.get('id')
        
        # Get player info (if available)
        participants = play.get('participants', [])
        if participants:
            # Usually first participant is the main player
            athlete = participants[0].get('athlete', {})
            play_dict['player_id'] = athlete.get('id')
            play_dict['player_name'] = athlete.get('displayName')
        else:
            play_dict['player_id'] = None
            play_dict['player_name'] = None
        
        parsed_plays.append(play_dict)
    
    df = pd.DataFrame(parsed_plays)
    return df


def extract_player_scoring_timeline(plays_df, game_data):
    """
    Extract minute-by-minute scoring for each player.
    
    Returns:
        DataFrame with player scoring timeline
    """
    # Get player info from boxscore
    boxscore = game_data.get('boxscore', {})
    players = boxscore.get('players', [])
    
    # Build player name -> team mapping
    player_team_map = {}
    
    for team_data in players:
        team_name = team_data.get('team', {}).get('displayName', '')
        for player in team_data.get('statistics', [{}])[0].get('athletes', []):
            player_name = player.get('athlete', {}).get('displayName')
            player_id = player.get('athlete', {}).get('id')
            if player_name:
                player_team_map[player_id] = {
                    'name': player_name,
                    'team': team_name
                }
    
    # Filter for scoring plays only
    scoring_plays = plays_df[plays_df['is_scoring_play'] == True].copy()
    
    # Add player name from mapping
    scoring_plays['player_name_mapped'] = scoring_plays['player_id'].map(
        lambda x: player_team_map.get(x, {}).get('name') if x else None
    )
    scoring_plays['team_name'] = scoring_plays['player_id'].map(
        lambda x: player_team_map.get(x, {}).get('team') if x else None
    )
    
    return scoring_plays


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("🏀 Bucks vs Pelicans Game Play-by-Play")
    print("=" * 80)
    print(f"Date: Feb 4, 2026")
    print("=" * 80)
    
    # Get games from Feb 4, 2026
    date_str = "20260204"
    scoreboard = get_games_on_date(date_str)
    
    if scoreboard:
        print(f"\n✅ Found {len(scoreboard.get('events', []))} games on {date_str}")
        
        # Find Bucks/Pelicans game
        game_id, game_info = find_game(scoreboard, "Bucks", "Pelicans")
        
        if game_id:
            # Get play-by-play
            game_data = get_play_by_play(game_id)
            
            if game_data:
                # Save raw data
                output_file = '/Users/thomasmyles/dev/betting/tmp/bucks_pelicans_pbp_20260204.json'
                with open(output_file, 'w') as f:
                    json.dump(game_data, f, indent=2)
                print(f"\n✅ Saved raw data to: tmp/bucks_pelicans_pbp_20260204.json")
                
                # Parse play-by-play
                print("\n" + "=" * 80)
                print("PARSING PLAY-BY-PLAY DATA")
                print("=" * 80)
                
                plays_df = parse_play_by_play(game_data)
                
                print(f"\n✅ Parsed {len(plays_df)} plays")
                print(f"\nColumns: {list(plays_df.columns)}")
                
                # Show sample plays
                print("\n--- Sample plays ---")
                print(plays_df[['quarter', 'time', 'description', 'away_score', 'home_score', 'player_name']].head(20).to_string())
                
                # Save parsed plays
                plays_csv = '/Users/thomasmyles/dev/betting/tmp/bucks_pelicans_plays_20260204.csv'
                plays_df.to_csv(plays_csv, index=False)
                print(f"\n✅ Saved parsed plays to: tmp/bucks_pelicans_plays_20260204.csv")
                
                # Extract scoring timeline
                print("\n" + "=" * 80)
                print("PLAYER SCORING TIMELINE")
                print("=" * 80)
                
                scoring_plays = extract_player_scoring_timeline(plays_df, game_data)
                
                print(f"\n✅ Found {len(scoring_plays)} scoring plays")
                print("\n--- Sample scoring plays ---")
                print(scoring_plays[['quarter', 'time', 'player_name_mapped', 'description', 
                                     'score_value', 'away_score', 'home_score']].head(30).to_string())
                
                # Save scoring timeline
                scoring_csv = '/Users/thomasmyles/dev/betting/tmp/bucks_pelicans_scoring_20260204.csv'
                scoring_plays.to_csv(scoring_csv, index=False)
                print(f"\n✅ Saved scoring timeline to: tmp/bucks_pelicans_scoring_20260204.csv")
                
                # Get boxscore data
                print("\n" + "=" * 80)
                print("BOXSCORE DATA")
                print("=" * 80)
                
                boxscore = game_data.get('boxscore', {})
                players = boxscore.get('players', [])
                
                for team_data in players:
                    team_name = team_data.get('team', {}).get('displayName', '')
                    print(f"\n{team_name}:")
                    
                    stats = team_data.get('statistics', [])
                    if stats:
                        athletes = stats[0].get('athletes', [])
                        
                        player_list = []
                        for player in athletes[:10]:  # First 10 players
                            athlete = player.get('athlete', {})
                            name = athlete.get('displayName', '')
                            
                            # Get stats
                            player_stats = {}
                            for stat in player.get('stats', []):
                                player_stats[stat] = player.get('stats', [])[player.get('stats', []).index(stat)]
                            
                            # Try to get basic stats (may need to adjust based on actual structure)
                            stats_list = player.get('stats', [])
                            if len(stats_list) >= 3:
                                mins = stats_list[0] if len(stats_list) > 0 else '-'
                                pts = stats_list[1] if len(stats_list) > 1 else '-'
                                
                                player_list.append(f"  {name}: {pts} pts, {mins} min")
                        
                        for p in player_list:
                            print(p)
                
                print("\n" + "=" * 80)
                print("✅ SUCCESS!")
                print("=" * 80)
                print("\nFiles created:")
                print(f"  1. tmp/bucks_pelicans_pbp_20260204.json (raw data)")
                print(f"  2. tmp/bucks_pelicans_plays_20260204.csv (all plays)")
                print(f"  3. tmp/bucks_pelicans_scoring_20260204.csv (scoring plays only)")
                
            else:
                print("\n❌ Failed to get play-by-play data")
        else:
            print("\n❌ Could not find Bucks/Pelicans game")
            print("\nAvailable games:")
            for event in scoreboard.get('events', []):
                comp = event.get('competitions', [{}])[0]
                competitors = comp.get('competitors', [])
                if len(competitors) >= 2:
                    away = competitors[1]['team']['displayName']
                    home = competitors[0]['team']['displayName']
                    print(f"  - {away} @ {home}")
    else:
        print("\n❌ Failed to get scoreboard data")
