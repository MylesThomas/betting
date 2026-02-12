"""
Test script to check if we can fetch live play-by-play data from ESPN API.

Purpose:
- Check the ESPN scoreboard for currently live NBA games
- Fetch the play-by-play data for any live games
- Display the game status, score, and recent plays
- Validate that we can get real-time data during games

Usage:
    python src/pbp_data/tmp/test_live_pbp_fetch.py [--fetch-pbp-data]
    
    --fetch-pbp-data: Fetch and display full play-by-play data for live games

Context:
- Thomas wants to test if ESPN API provides live play-by-play data during games
- This script will be run tonight when there are live games to verify functionality
- Uses same ESPN API endpoints as existing pbp_data pipeline
"""

import argparse
import json
import sys
import requests
from datetime import datetime
from pathlib import Path

# Find project root via .gitignore
current_path = Path(__file__).resolve()
while not (current_path / '.gitignore').exists():
    if current_path.parent == current_path:
        raise RuntimeError("Could not find project root (.gitignore not found)")
    current_path = current_path.parent
ROOT_DIR = current_path
sys.path.insert(0, str(ROOT_DIR / 'src'))

from pbp_data.config import ESPN_SCOREBOARD_URL, ESPN_SUMMARY_URL


def get_live_games():
    """
    Get all currently live NBA games from ESPN scoreboard.
    
    Returns:
        List of dicts with game_id, home_team, away_team, status, score
    """
    print("🔄 Checking ESPN scoreboard for live games...")
    
    # Get today's games
    today = datetime.now().strftime('%Y%m%d')
    url = f"{ESPN_SCOREBOARD_URL}?dates={today}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"❌ Error fetching scoreboard: {e}")
        return []
    
    live_games = []
    
    for event in data.get('events', []):
        competition = event['competitions'][0]
        status = competition['status']
        competitors = competition['competitors']
        
        # Check if game is in progress
        # status['type']['name'] can be: 'STATUS_SCHEDULED', 'STATUS_IN_PROGRESS', 'STATUS_FINAL', etc.
        is_live = status['type']['state'] == 'in'
        
        home_team = next(c for c in competitors if c['homeAway'] == 'home')
        away_team = next(c for c in competitors if c['homeAway'] == 'away')
        
        game_info = {
            'game_id': event['id'],
            'home_team': home_team['team']['displayName'],
            'away_team': away_team['team']['displayName'],
            'home_score': int(home_team.get('score', 0)),
            'away_score': int(away_team.get('score', 0)),
            'status_name': status['type']['name'],
            'status_detail': status['type']['detail'],
            'is_live': is_live,
            'period': status.get('period', 0),
            'clock': status.get('displayClock', 'N/A'),
        }
        
        live_games.append(game_info)
    
    return live_games


def get_live_play_by_play(game_id):
    """
    Get live play-by-play data for a specific game.
    
    Args:
        game_id: ESPN game ID
    
    Returns:
        Dict with play-by-play data or None if failed
    """
    url = f"{ESPN_SUMMARY_URL}?event={game_id}"
    
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        return data
    except Exception as e:
        print(f"❌ Error fetching play-by-play: {e}")
        return None


def validate_play_by_play_score(plays, game_info, team_id_map):
    """
    Validate that play-by-play scoring events sum to the reported score.
    
    Args:
        plays: List of play dictionaries
        game_info: Game info dict with away_team, home_team, away_score, home_score
        team_id_map: Dict mapping team ID to team name
    
    Returns:
        Dict with validation results
    """
    away_calculated = 0
    home_calculated = 0
    
    # Sum up all scoring plays by team ID
    for play in plays:
        score_value = play.get('scoreValue', 0)
        if score_value > 0:
            team = play.get('team', {})
            team_id = str(team.get('id', ''))
            team_name = team_id_map.get(team_id, '')
            
            if team_name == game_info['away_team']:
                away_calculated += score_value
            elif team_name == game_info['home_team']:
                home_calculated += score_value
    
    # Compare to API scores
    away_match = away_calculated == game_info['away_score']
    home_match = home_calculated == game_info['home_score']
    
    return {
        'away_team': game_info['away_team'],
        'home_team': game_info['home_team'],
        'away_api': game_info['away_score'],
        'away_calculated': away_calculated,
        'away_match': away_match,
        'home_api': game_info['home_score'],
        'home_calculated': home_calculated,
        'home_match': home_match,
        'both_match': away_match and home_match
    }


def display_game_summary(game_info, pbp_data):
    """
    Display a summary of the game and recent plays.
    
    Args:
        game_info: Game info dict from scoreboard
        pbp_data: Play-by-play data from summary endpoint
    """
    print("\n" + "="*80)
    print(f"🏀 {game_info['away_team']} @ {game_info['home_team']}")
    print(f"📊 Score: {game_info['away_team']} {game_info['away_score']}, {game_info['home_team']} {game_info['home_score']}")
    print(f"⏰ Status: {game_info['status_detail']}")
    print(f"🕐 Period: {game_info['period']}, Clock: {game_info['clock']}")
    print("="*80)
    
    # Check if play-by-play data exists
    if pbp_data is None:
        print("❌ No play-by-play data available")
        return
    
    # Get plays
    plays = pbp_data.get('plays', [])
    if not plays:
        print("❌ No plays found in response")
        return
    
    print(f"\n✅ Found {len(plays)} total plays\n")
    
    # Build team ID to name mapping from boxscore
    team_id_map = {}
    if 'boxscore' in pbp_data and 'players' in pbp_data['boxscore']:
        for player_group in pbp_data['boxscore']['players']:
            team = player_group.get('team', {})
            team_id = str(team.get('id', ''))
            team_name = team.get('displayName', '')
            if team_id and team_name:
                team_id_map[team_id] = team_name
    
    # Validate score by summing all scoring plays
    validation = validate_play_by_play_score(plays, game_info, team_id_map)
    print("🔢 Score Validation:")
    print(f"   {validation['away_team']}:")
    print(f"      API Score: {validation['away_api']}")
    print(f"      Calculated: {validation['away_calculated']}")
    print(f"      Match: {'✅' if validation['away_match'] else '❌'}")
    print(f"   {validation['home_team']}:")
    print(f"      API Score: {validation['home_api']}")
    print(f"      Calculated: {validation['home_calculated']}")
    print(f"      Match: {'✅' if validation['home_match'] else '❌'}")
    
    if not validation['both_match']:
        diff_away = validation['away_calculated'] - validation['away_api']
        diff_home = validation['home_calculated'] - validation['home_api']
        print(f"   ⚠️  WARNING: Score mismatch! Away diff: {diff_away:+d}, Home diff: {diff_home:+d}")
    else:
        print("   ✅ All scores validated!")
    
    # Show last 10 plays
    print("\n📝 Last 10 plays:")
    print("-" * 80)
    
    recent_plays = plays[-10:]
    for i, play in enumerate(recent_plays, 1):
        period = play.get('period', {}).get('number', '?')
        clock = play.get('clock', {}).get('displayValue', '??:??')
        text = play.get('text', 'No description')
        score_value = play.get('scoreValue', 0)
        
        # Get team name from ID
        team = play.get('team', {})
        team_id = str(team.get('id', ''))
        team_name = team_id_map.get(team_id, 'Unknown')
        
        play_type = play.get('type', {}).get('text', 'Unknown')
        
        score_marker = f"+{score_value}" if score_value > 0 else ""
        print(f"{i:2}. Q{period} {clock:>6} | {team_name:20} | {play_type:15} | {text}")
        if score_marker:
            print(f"    {score_marker} points")
    
    print("-" * 80)
    
    # Show boxscore summary if available
    if 'boxscore' in pbp_data and 'players' in pbp_data['boxscore']:
        print("\n📋 Boxscore - Top Performers:")
        
        for player_group in pbp_data['boxscore']['players']:
            team_name = player_group.get('team', {}).get('displayName', 'Unknown')
            print(f"\n{team_name}:")
            
            # Get statistics group (usually first one has main stats)
            if 'statistics' in player_group and player_group['statistics']:
                stat_group = player_group['statistics'][0]
                stat_names = stat_group.get('names', [])
                athletes = stat_group.get('athletes', [])
                
                # Find indices for MIN, PTS, REB, AST
                # stat_names example: ['MIN', 'PTS', 'FG', '3PT', 'FT', 'REB', 'AST', 'TO', 'STL', 'BLK', 'OREB', 'DREB', 'PF', '+/-']
                try:
                    min_idx = stat_names.index('MIN')
                    pts_idx = stat_names.index('PTS')
                    reb_idx = stat_names.index('REB')
                    ast_idx = stat_names.index('AST')
                except ValueError:
                    # Fallback if stat names are different
                    min_idx, pts_idx, reb_idx, ast_idx = 0, 1, 5, 6
                
                # Show top 5 active players by minutes (usually sorted by relevance)
                shown = 0
                for athlete_data in athletes:
                    if shown >= 5:
                        break
                    
                    # Skip DNP players
                    if athlete_data.get('didNotPlay'):
                        continue
                    
                    athlete = athlete_data.get('athlete', {})
                    stats = athlete_data.get('stats', [])
                    name = athlete.get('displayName', 'Unknown')
                    
                    # Get stats by index
                    if len(stats) > max(min_idx, pts_idx, reb_idx, ast_idx):
                        mins = stats[min_idx]
                        pts = stats[pts_idx]
                        reb = stats[reb_idx]
                        ast = stats[ast_idx]
                        print(f"  - {name:25} {mins:>5} min | {pts:>3} pts, {reb:>3} reb, {ast:>3} ast")
                        shown += 1
                    else:
                        # Fallback if structure is different
                        print(f"  - {name:25} {' | '.join(str(s) for s in stats[:5])}")
                        shown += 1


def main():
    """Main function to check for live games and fetch their play-by-play data."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Test fetching live play-by-play data from ESPN API'
    )
    parser.add_argument(
        '--fetch-pbp-data',
        action='store_true',
        help='Fetch and display full play-by-play data for live games'
    )
    args = parser.parse_args()
    
    print("🏀 ESPN Live Play-by-Play Test")
    print("="*80)
    print(f"🕐 Current time: {datetime.now().strftime('%Y-%m-%d %I:%M:%S %p EST')}")
    print("="*80)
    
    # Get all games (live and scheduled)
    all_games = get_live_games()
    
    if not all_games:
        print("❌ No games found on ESPN scoreboard today")
        return
    
    # Separate games by status
    live_games = [g for g in all_games if g['is_live']]
    final_games = [g for g in all_games if 'FINAL' in g['status_name']]
    scheduled_games = [g for g in all_games if not g['is_live'] and 'FINAL' not in g['status_name']]
    
    print(f"\n📊 Games today (ET timezone):")
    print(f"   🔴 {len(live_games)} LIVE")
    print(f"   ✅ {len(final_games)} Final")
    print(f"   📅 {len(scheduled_games)} Scheduled")
    print(f"   Total: {len(all_games)} games\n")
    
    # List all scheduled games
    if scheduled_games:
        print("📅 Scheduled games:")
        for game in scheduled_games:
            print(f"   {game['away_team']} @ {game['home_team']} - {game['status_detail']}")
    
    # List all final games
    if final_games:
        print(f"\n✅ Final games:")
        for game in final_games:
            print(f"   {game['away_team']} {game['away_score']}, {game['home_team']} {game['home_score']} - {game['status_detail']}")
    
    # List all live games
    if live_games:
        print(f"\n🔴 LIVE games:")
        for i, game in enumerate(live_games, 1):
            print(f"   {i}. {game['away_team']} {game['away_score']}, {game['home_team']} {game['home_score']}")
            print(f"      {game['status_detail']} - Q{game['period']} {game['clock']}")
        
        # Check if we should fetch play-by-play data
        if not args.fetch_pbp_data:
            print("\n" + "="*80)
            print("ℹ️  Use --fetch-pbp-data flag to fetch full play-by-play data")
            print("="*80)
            print("✅ Test complete!")
            return
        
        # Fetch play-by-play for each live game
        print("\n" + "="*80)
        print(f"📥 Fetching play-by-play data for {len(live_games)} live games...")
        print("="*80)
        
        for game in live_games:
            print(f"\n🔄 Fetching live play-by-play for game {game['game_id']}...")
            pbp_data = get_live_play_by_play(game['game_id'])
            
            if pbp_data:
                display_game_summary(game, pbp_data)
                
                # Save to file for inspection
                output_file = Path(__file__).parent / f"live_game_{game['game_id']}.json"
                with open(output_file, 'w') as f:
                    json.dump(pbp_data, f, indent=2)
                print(f"\n💾 Saved full response to: {output_file}")
            else:
                print(f"❌ Failed to fetch play-by-play for game {game['game_id']}")
    else:
        print("\nℹ️  No games currently in progress")
    
    print("\n" + "="*80)
    print("✅ Test complete!")


if __name__ == '__main__':
    main()
