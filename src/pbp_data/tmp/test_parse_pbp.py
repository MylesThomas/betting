"""
Simple test script to debug play-by-play parsing and cumulative points.

Goal: Ensure cumulative_points only increases, never decreases.
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
print(f"\n{'='*80}")
print("TESTING DIFFERENT PARSING APPROACHES")
print(f"{'='*80}\n")

# Approach 1: Keep ESPN order, track cumulative as we go
print("APPROACH 1: Keep ESPN order (current)")
print("-" * 80)
player_points = 0
play_data = []

for i, play in enumerate(plays):
    description = play.get('text', '')
    quarter = play.get('period', {}).get('number', 1)
    clock = play.get('clock', {}).get('displayValue', '12:00')
    
    # Check if player scored
    if player_name in description:
        if 'makes' in description.lower() or 'free throw' in description.lower():
            if '3-pt' in description.lower():
                player_points += 3
            elif '2-pt' in description.lower():
                player_points += 2
            elif 'free throw' in description.lower() and 'makes' in description.lower():
                player_points += 1
    
    play_data.append({
        'index': i,
        'quarter': quarter,
        'clock': clock,
        'points': player_points,
        'description': description[:60] if player_name in description else ''
    })

# Check for decreases
print("\nChecking for point decreases...")
decreases = []
for i in range(1, len(play_data)):
    if play_data[i]['points'] < play_data[i-1]['points']:
        decreases.append((i, play_data[i-1], play_data[i]))

if decreases:
    print(f"❌ Found {len(decreases)} places where points DECREASED!")
    for idx, prev, curr in decreases[:5]:  # Show first 5
        print(f"\n  Play {prev['index']} (Q{prev['quarter']} {prev['clock']}): {prev['points']} pts")
        print(f"  Play {curr['index']} (Q{curr['quarter']} {curr['clock']}): {curr['points']} pts")
        print(f"  Decrease: {prev['points']} → {curr['points']}")
else:
    print(f"✅ No decreases found! Points only increase.")

# Show points at quarter boundaries
print(f"\n{'='*80}")
print("POINTS AT QUARTER BOUNDARIES")
print(f"{'='*80}\n")

for q in [1, 2, 3, 4]:
    start_clock = "12:00"
    end_clock = "0:00"
    
    q_plays = [p for p in play_data if p['quarter'] == q]
    
    if q_plays:
        start_pts = q_plays[0]['points']
        end_pts = q_plays[-1]['points']
        
        # Check 12:00 specifically
        at_1200 = [p for p in q_plays if p['clock'] == start_clock]
        
        print(f"Q{q}:")
        print(f"  Start: {start_pts} pts")
        print(f"  End: {end_pts} pts")
        print(f"  Plays at 12:00: {len(at_1200)}")
        
        if len(at_1200) > 1:
            pts_at_1200 = [p['points'] for p in at_1200]
            print(f"  Points at 12:00: {pts_at_1200}")
            if len(set(pts_at_1200)) > 1:
                print(f"  ⚠️  Multiple different point values at 12:00!")

print(f"\n{'='*80}")
print(f"FINAL RESULT: {play_data[-1]['points']} points")
print(f"{'='*80}")
