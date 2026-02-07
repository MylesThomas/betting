"""
Debug script: Check if ESPN's play order is actually chronological.
"""

import json
from pathlib import Path

# Load the Luka game
game_id = "401809820"
pbp_dir = Path.home() / "Downloads" / "tmp" / "player_points_monte_carlo" / "pbp_data"
json_file = list(pbp_dir.glob(f"*_{game_id}.json"))[0]

with open(json_file, 'r') as f:
    data = json.load(f)

plays = data['plays']
player_name = "Luka Doncic"

print(f"Total plays: {len(plays)}")
print(f"\nChecking play order around Q1/Q2 boundary...\n")
print(f"{'='*100}")

# Find plays around the Q1/Q2 transition
for i in range(90, 230):
    if i >= len(plays):
        break
    
    play = plays[i]
    quarter = play.get('period', {}).get('number', 1)
    clock = play.get('clock', {}).get('displayValue', '12:00')
    description = play.get('text', '')
    
    # Check if Luka scored
    luka_scored = ""
    if player_name in description:
        if 'makes' in description.lower():
            if '3-pt' in description.lower():
                luka_scored = " ★ +3 PTS"
            elif '2-pt' in description.lower():
                luka_scored = " ★ +2 PTS"
        elif 'free throw' in description.lower() and 'makes' in description.lower():
            luka_scored = " ★ +1 PT"
    
    # Highlight Q1/Q2 boundary
    marker = ""
    if i == 96:
        print(f"\n{'='*100}")
        print("START OF Q2 PLAYS IN ESPN JSON")
        print(f"{'='*100}\n")
    
    print(f"Play {i:3d}: Q{quarter} {clock:>6s} {luka_scored:12s} | {description[:70]}")

print(f"\n{'='*100}\n")
