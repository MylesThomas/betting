"""
Test script to verify the get_active_players fix.

Context:
- Bug: get_active_players was using boxscore['teams'] instead of boxscore['players']
- Fix: Changed line 696 to use boxscore['players']
- This script tests that the fix works correctly

Usage:
    python src/pbp_data/tmp/test_get_active_players_fix.py
"""

import os
import sys
from pathlib import Path

# Find project root
current = Path(__file__).resolve()
root = None
for parent in current.parents:
    if (parent / '.gitignore').exists():
        root = parent
        break

if root:
    sys.path.insert(0, str(root / 'src'))
    os.chdir(root)

# Import the function using importlib to handle numeric filename
import importlib.util
spec = importlib.util.spec_from_file_location(
    "live_betting_signal_generator",
    root / 'src' / 'pbp_data' / '10_live_betting_signal_generator.py'
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

get_active_players = module.get_active_players

# =============================================================================
# TEST
# =============================================================================

print("="*80)
print("TEST: get_active_players fix")
print("="*80)
print()

# Test with game from last night (saved boxscore exists)
game_id = '401810642'
print(f"🧪 Testing game {game_id} (MEM @ DEN from last night)...")
print()

players = get_active_players(game_id)

if not players:
    print("❌ FAILED - No players returned")
    print("   The fix did not work!")
    sys.exit(1)

print(f"✅ SUCCESS! Found {len(players)} players")
print()

# Show top 10 players
print("Top 10 players by points:")
print(f"{'Player':<25} {'Team':<25} {'Pts':>5} {'Min':>6}")
print("-" * 80)

for p in players[:10]:
    print(f"{p['player_name']:<25} {p['team']:<25} {p['current_points']:>5.0f} {p['minutes_played']:>6.1f}")

print()

# Check for Nikola Jokic
jokic = [p for p in players if 'jokic' in p['player_name'].lower()]
if jokic:
    print("🎯 Found Nikola Jokic:")
    j = jokic[0]
    print(f"   Player: {j['player_name']}")
    print(f"   Team: {j['team']}")
    print(f"   Points: {j['current_points']}")
    print(f"   Minutes: {j['minutes_played']}")
else:
    print("⚠️  Nikola Jokic not found (might not be in top 20)")

print()
print("="*80)
print("✅ TEST PASSED")
print("="*80)
