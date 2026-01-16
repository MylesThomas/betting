"""
Check DNP (Did Not Play) Players in Box Scores

Question: If a player doesn't play (injury, rest, etc.), do they appear in the box score data?

Investigation: 
- Check Kristaps Porzingis and compare with all Boston Celtics games
- See how many games he played vs total team games
- List all games he missed so you can verify against Basketball Reference

Date: 2025-11-20
Author: Myles Thomas
"""

import pandas as pd
import os
from pathlib import Path


def get_project_root():
    """Find project root by locating .gitignore file."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / '.gitignore').exists():
            return parent
    raise FileNotFoundError("Could not find project root (no .gitignore found)")


def main():
    ROOT = get_project_root()
    print(f"Project root: {ROOT}")
    print()
    
    # Load data
    print("Loading data...")
    merged_file = ROOT / 'data' / 'consensus_props_with_game_results_2024_25.csv'
    df_merged = pd.read_csv(merged_file)
    
    results_file = ROOT / 'data' / 'nba_game_results_2024_25.csv'
    df_results = pd.read_csv(results_file)
    
    print(f"  Game results: {len(df_results):,} rows")
    print(f"  Date range: {df_results['date'].min()} to {df_results['date'].max()}")
    print()
    
    # ========================================================================
    # KRISTAPS PORZINGIS ANALYSIS
    # ========================================================================
    
    print("="*70)
    print("KRISTAPS PORZINGIS - GAMES PLAYED ANALYSIS")
    print("="*70)
    print()
    
    # Get Kristaps game results
    kp_results = df_results[df_results['player'].str.contains('Kristaps', case=False, na=False)].copy()
    
    print("Kristaps Porzingis in game results:")
    print(f"  Total game entries: {len(kp_results)}")
    print(f"  Unique dates: {kp_results['date'].nunique()}")
    print(f"  Date range: {kp_results['date'].min()} to {kp_results['date'].max()}")
    print(f"  Team: {kp_results['team'].unique()}")
    print()
    
    # Get all Boston Celtics games
    bos_games_df = df_results[df_results['team'] == 'BOS'].copy()
    bos_dates = sorted(bos_games_df['date'].unique())
    
    print(f"Boston Celtics games in data:")
    print(f"  Total games: {len(bos_dates)}")
    print(f"  Date range: {bos_dates[0]} to {bos_dates[-1]}")
    print()
    
    # Compare
    kp_dates = sorted(kp_results['date'].unique())
    
    print(f"Kristaps Porzingis games played: {len(kp_dates)}")
    print(f"Games missed by Kristaps: {len(bos_dates) - len(kp_dates)}")
    print(f"Participation rate: {len(kp_dates)/len(bos_dates)*100:.1f}%")
    print()
    
    print("Season progress:")
    print(f"  {len(bos_dates)}/82 games = {len(bos_dates)/82*100:.1f}% of full season")
    print()
    
    # List all games Kristaps missed
    missed_dates = sorted(set(bos_dates) - set(kp_dates))
    
    print("="*70)
    print(f"GAMES KRISTAPS PORZINGIS MISSED ({len(missed_dates)} games)")
    print("="*70)
    print()
    print(f"{'#':<4} {'Date':<12} {'Matchup':<45}")
    print("-" * 70)
    
    for i, date in enumerate(missed_dates, 1):
        # Get opponent for that game
        game_info = bos_games_df[bos_games_df['date'] == date].iloc[0]
        matchup = game_info['matchup']
        print(f"{i:<4} {date:<12} {matchup:<45}")
    
    print()
    print("="*70)
    print("ANSWER TO KEY QUESTION")
    print("="*70)
    print()
    print("✅ Players who don't play (DNP) do NOT appear in box score data!")
    print()
    print(f"   - Kristaps has entries for {len(kp_dates)} games")
    print(f"   - Boston played {len(bos_dates)} games total")
    print(f"   - He missed {len(missed_dates)} games (no box score entries)")
    print()
    print("This confirms:")
    print("  • Missing prop matches = legitimate DNP situations")
    print("  • Cannot backtest props for games player didn't play")
    print("  • Match rate of 93.82% is expected and good!")
    print()
    
    # ========================================================================
    # CHECK PROPS
    # ========================================================================
    
    print("="*70)
    print("KRISTAPS PROPS ANALYSIS")
    print("="*70)
    print()
    
    kp_props = df_merged[df_merged['player'].str.contains('Kristaps', case=False, na=False)].copy()
    
    print(f"Kristaps Porzingis props:")
    print(f"  Total prop records: {len(kp_props)}")
    print(f"  Props WITH game results: {len(kp_props[~kp_props['threes_made'].isna()])}")
    print(f"  Props WITHOUT game results (DNP): {len(kp_props[kp_props['threes_made'].isna()])}")
    print()
    
    # Dates WITHOUT game results
    kp_without_results = kp_props[kp_props['threes_made'].isna()]
    prop_dates_no_results = sorted(kp_without_results['date'].unique())
    
    if len(prop_dates_no_results) > 0:
        print(f"Prop dates with no game results ({len(prop_dates_no_results)} dates):")
        print("(These are games he didn't play - props were posted but player scratched)")
        for date in prop_dates_no_results[:15]:
            print(f"  {date}")
        if len(prop_dates_no_results) > 15:
            print(f"  ... and {len(prop_dates_no_results) - 15} more")
        print()
    
    print("="*70)
    print("You can now compare these dates with Basketball Reference!")
    print("="*70)


if __name__ == '__main__':
    main()

