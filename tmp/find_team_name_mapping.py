"""
Figure out team name mapping between ESPN game results and Odds API lines

Goal: Create a 1-to-1 mapping to get 100% match rate

Author: Thomas Myles  
Date: 2026-01-30
"""

import pandas as pd
from pathlib import Path

# Load cached 2025-26 data
cache_path = Path.home() / 'Downloads' / 'tmp' / 'nba_2025-26_merged.parquet'

if not cache_path.exists():
    print("❌ No cached data. Run optimizer first.")
    exit(1)

# Load the successfully merged games
merged_df = pd.read_parquet(cache_path)

print(f"✅ Loaded {len(merged_df)} successfully merged games")
print(f"\nTeam names from ESPN results (AWAY_TEAM, HOME_TEAM):")
espn_teams = sorted(set(merged_df['AWAY_TEAM'].unique()) | set(merged_df['HOME_TEAM'].unique()))
for team in espn_teams:
    print(f"  {team}")

print(f"\nTeam names from Odds API (away_team, home_team):")
odds_teams = sorted(set(merged_df['away_team'].unique()) | set(merged_df['home_team'].unique()))
for team in odds_teams:
    print(f"  {team}")

print(f"\n{'='*70}")
print("COMPARISON")
print(f"{'='*70}")

# Check if they're exactly the same
if espn_teams == odds_teams:
    print("✅ Team names are IDENTICAL between ESPN and Odds API!")
    print("   The join should work 100%")
else:
    print("❌ Team names differ!")
    
    espn_set = set(espn_teams)
    odds_set = set(odds_teams)
    
    only_espn = espn_set - odds_set
    only_odds = odds_set - espn_set
    
    if only_espn:
        print(f"\n📋 Only in ESPN ({len(only_espn)}):")
        for team in sorted(only_espn):
            print(f"  {team}")
    
    if only_odds:
        print(f"\n📋 Only in Odds API ({len(only_odds)}):")
        for team in sorted(only_odds):
            print(f"  {team}")
    
    # Try to find likely pairs
    print(f"\n🔍 Suggested mappings (based on similarity):")
    for espn_team in sorted(only_espn):
        for odds_team in sorted(only_odds):
            # Simple similarity check
            espn_words = set(espn_team.lower().split())
            odds_words = set(odds_team.lower().split())
            
            # If they share 50%+ words, they're probably the same team
            if espn_words & odds_words:
                shared = espn_words & odds_words
                print(f"  '{espn_team}' → '{odds_team}' (shared: {shared})")

print(f"\n{'='*70}")
print("CONCLUSION")
print(f"{'='*70}")

if espn_teams == odds_teams:
    print("Team names match perfectly!")
    print("The low join rate must be due to date/time mismatches.")
else:
    print(f"Need to create team name mapping for {len(only_espn)} teams")
    print("\nProposed mapping (add to config or code):")
    print("TEAM_NAME_MAP = {")
    for espn_team in sorted(only_espn):
        print(f"    '{espn_team}': 'TODO',  # Needs manual mapping")
    print("}")
