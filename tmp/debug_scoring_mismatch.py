"""
Debug script to find missing scoring plays.

Compare cumulative points from play-by-play vs final boxscore.
"""

import pandas as pd
import json

# Load the scoring plays we parsed
scoring_df = pd.read_csv('/Users/thomasmyles/dev/betting/tmp/bucks_pelicans_scoring_20260204.csv')

# Load the raw JSON to check boxscore
with open('/Users/thomasmyles/dev/betting/tmp/bucks_pelicans_pbp_20260204.json', 'r') as f:
    raw_data = json.load(f)

print("=" * 80)
print("DEBUGGING: Why don't the numbers match?")
print("=" * 80)

# Get boxscore totals
boxscore = raw_data.get('boxscore', {})
players_data = boxscore.get('players', [])

print("\n--- BOXSCORE TOTALS ---")
boxscore_totals = {}

for team_data in players_data:
    team_name = team_data.get('team', {}).get('displayName', '')
    stats = team_data.get('statistics', [])
    
    if stats:
        athletes = stats[0].get('athletes', [])
        
        for player in athletes:
            athlete = player.get('athlete', {})
            name = athlete.get('displayName', '')
            
            # Get points from stats array (usually stats[1] is points)
            stats_list = player.get('stats', [])
            if len(stats_list) >= 2:
                pts_str = stats_list[1] if len(stats_list) > 1 else '0'
                try:
                    pts = int(pts_str)
                    if pts > 0:
                        boxscore_totals[name] = pts
                        print(f"{name:30s} {pts} pts ({team_name})")
                except:
                    pass

# Get cumulative totals from our parsed plays
print("\n--- CUMULATIVE FROM PLAY-BY-PLAY ---")
parsed_totals = {}

for player_name in scoring_df['player_name_mapped'].dropna().unique():
    player_plays = scoring_df[scoring_df['player_name_mapped'] == player_name]
    total_pts = player_plays['score_value'].sum()
    parsed_totals[player_name] = int(total_pts)
    print(f"{player_name:30s} {int(total_pts)} pts")

# Compare
print("\n--- DISCREPANCIES ---")
all_players = set(boxscore_totals.keys()) | set(parsed_totals.keys())

discrepancies = []
for player in sorted(all_players):
    boxscore_pts = boxscore_totals.get(player, 0)
    parsed_pts = parsed_totals.get(player, 0)
    
    if boxscore_pts != parsed_pts:
        diff = boxscore_pts - parsed_pts
        discrepancies.append({
            'player': player,
            'boxscore': boxscore_pts,
            'parsed': parsed_pts,
            'missing': diff
        })
        print(f"{player:30s} Boxscore: {boxscore_pts:2d}  Parsed: {parsed_pts:2d}  Missing: {diff:2d}")

# Check the raw plays to see if we're missing something
print("\n--- CHECKING RAW PLAYS ---")
all_plays = raw_data.get('plays', [])
print(f"Total plays in raw data: {len(all_plays)}")
print(f"Scoring plays we captured: {len(scoring_df)}")

# Count scoring plays in raw data
scoring_plays_raw = [p for p in all_plays if p.get('scoringPlay', False)]
print(f"Scoring plays in raw data: {len(scoring_plays_raw)}")

# Look for plays with the players who have discrepancies
if discrepancies:
    print(f"\n--- INVESTIGATING {discrepancies[0]['player']} ---")
    player_name = discrepancies[0]['player']
    
    # Find all plays involving this player
    player_plays_raw = []
    for play in all_plays:
        participants = play.get('participants', [])
        for p in participants:
            athlete = p.get('athlete', {})
            if athlete.get('displayName') == player_name:
                player_plays_raw.append(play)
                break
    
    print(f"Found {len(player_plays_raw)} total plays for {player_name}")
    
    # Show scoring plays
    scoring_for_player = [p for p in player_plays_raw if p.get('scoringPlay', False)]
    print(f"Scoring plays: {len(scoring_for_player)}")
    
    print("\nAll scoring plays:")
    total = 0
    for play in scoring_for_player:
        score_val = play.get('scoreValue', 0)
        total += score_val
        quarter = play.get('period', {}).get('number')
        time = play.get('clock', {}).get('displayValue')
        desc = play.get('text', '')
        print(f"  Q{quarter} {time:6s} +{score_val} pts: {desc[:60]}")
    
    print(f"\nTotal points: {total}")
    print(f"Boxscore says: {boxscore_totals.get(player_name, 0)}")
    print(f"We calculated: {parsed_totals.get(player_name, 0)}")
