"""
Debug Player Name Mismatches

Use this script to diagnose player name matching issues between props and game results.

This script will:
1. Load props and game results data
2. Normalize player names in both datasets
3. Identify which players/dates don't match
4. Suggest possible matches for missing players

Date: 2025-11-20
Author: Myles Thomas
"""

import pandas as pd
import os
import sys
from pathlib import Path


def get_project_root():
    """Find project root by locating .gitignore file."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / '.gitignore').exists():
            return parent
    raise FileNotFoundError("Could not find project root (no .gitignore found)")


# Add src to path
sys.path.insert(0, str(get_project_root() / 'src'))

from player_name_utils import (
    normalize_player_names_df,
    print_name_mismatch_report,
    find_similar_player_names
)


def analyze_missing_player_dates(df_props, df_results):
    """
    Analyze which specific player-date combinations are missing.
    
    This is useful when a player exists in both datasets but specific dates are missing
    (e.g., player was injured/inactive on those dates).
    """
    print("="*70)
    print("ANALYZING MISSING PLAYER-DATE COMBINATIONS")
    print("="*70)
    print()
    
    # Get player-date combinations
    props_combos = set(zip(df_props['player'], df_props['date']))
    results_combos = set(zip(df_results['player'], df_results['date']))
    
    missing_combos = props_combos - results_combos
    
    print(f"Missing player-date combinations: {len(missing_combos):,}")
    print()
    
    if len(missing_combos) == 0:
        print("✅ No missing combinations!")
        return
    
    # Group by player to see which players have the most missing dates
    player_missing_counts = {}
    for player, date in missing_combos:
        player_missing_counts[player] = player_missing_counts.get(player, 0) + 1
    
    # Sort by count
    sorted_players = sorted(player_missing_counts.items(), key=lambda x: x[1], reverse=True)
    
    print("Top 20 players with most missing game dates:")
    print(f"{'Player':<30} {'Missing Dates':>15}")
    print("-"*50)
    for player, count in sorted_players[:20]:
        print(f"{player:<30} {count:>15}")
    print()
    
    # Check a specific player in detail
    if sorted_players:
        top_player = sorted_players[0][0]
        print(f"Detailed analysis for: {top_player}")
        print()
        
        # Get all props dates for this player
        props_dates = sorted(df_props[df_props['player'] == top_player]['date'].unique())
        results_dates = sorted(df_results[df_results['player'] == top_player]['date'].unique())
        
        print(f"  Props dates: {len(props_dates)} games")
        if props_dates:
            print(f"    First: {props_dates[0]}, Last: {props_dates[-1]}")
            print(f"    Sample: {props_dates[:5]}")
        
        print(f"  Game results dates: {len(results_dates)} games")
        if results_dates:
            print(f"    First: {results_dates[0]}, Last: {results_dates[-1]}")
            print(f"    Sample: {results_dates[:5]}")
        
        # Dates in props but not in results
        missing_dates = set(props_dates) - set(results_dates)
        if missing_dates:
            print(f"  Missing dates ({len(missing_dates)}): {sorted(missing_dates)[:10]}")
            print(f"    -> Player likely didn't play on these dates (injury/rest)")
    
    print()


def compare_raw_vs_normalized(props_file, results_file):
    """
    Compare player names before and after normalization.
    
    Shows which names changed during normalization.
    """
    print("="*70)
    print("RAW vs NORMALIZED PLAYER NAMES")
    print("="*70)
    print()
    
    # Load props
    df_props = pd.read_csv(props_file)
    original_props_names = df_props['player'].unique()
    
    # Normalize
    df_props_norm = normalize_player_names_df(df_props, 'player')
    normalized_props_names = df_props_norm['player'].unique()
    
    print("Props dataset - Names that changed after normalization:")
    print()
    
    changes = []
    for orig in original_props_names:
        # Find normalized version
        norm = df_props_norm[df_props['player'] == orig]['player'].iloc[0]
        if orig != norm:
            changes.append((orig, norm))
    
    if changes:
        print(f"{'Original':<30} -> {'Normalized':<30}")
        print("-"*65)
        for orig, norm in sorted(changes)[:30]:
            print(f"{orig:<30} -> {norm:<30}")
        print()
        if len(changes) > 30:
            print(f"... and {len(changes) - 30} more changes")
    else:
        print("No names changed during normalization")
    
    print()


def main():
    """Main debugging workflow."""
    
    ROOT = get_project_root()
    
    # File paths
    props_file = ROOT / 'data' / '01_input' / 'the-odds-api' / 'nba' / 'historical_props' / 'combined_props_player_threes.csv'
    results_file = ROOT / 'data' / 'nba_game_results_2024_25.csv'
    
    if not props_file.exists():
        print(f"❌ Props file not found: {props_file}")
        return
    
    if not results_file.exists():
        print(f"❌ Game results file not found: {results_file}")
        return
    
    print("Loading data...")
    df_props = pd.read_csv(props_file)
    df_results = pd.read_csv(results_file)
    print(f"  Props: {len(df_props):,} rows, {df_props['player'].nunique()} unique players")
    print(f"  Results: {len(df_results):,} rows, {df_results['player'].nunique()} unique players")
    print()
    
    # Show raw vs normalized names
    compare_raw_vs_normalized(props_file, results_file)
    
    # Normalize both datasets
    print("Normalizing player names...")
    df_props = normalize_player_names_df(df_props, 'player')
    df_results = normalize_player_names_df(df_results, 'player')
    print()
    
    # Print mismatch report
    print_name_mismatch_report(df_props, df_results)
    print()
    
    # Analyze missing player-date combinations
    analyze_missing_player_dates(df_props, df_results)
    
    print("="*70)
    print("DEBUGGING COMPLETE")
    print("="*70)
    print()
    print("To fix name mismatches:")
    print("1. Add mappings to src/player_name_utils.py in get_name_mappings()")
    print("2. Or update the normalize_player_name() function for pattern-based fixes")


if __name__ == '__main__':
    main()

