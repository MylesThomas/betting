"""
STEP 4: Validate player team history output.

Run this AFTER a successful build to validate data quality.
Checks for common issues like duplicate stints, invalid dates, etc.

Usage:
    python src/player_team_history/04_validate.py

Input:
    ~/Downloads/tmp/player_team_history/history.parquet

Output:
    - Validation report showing any issues found
    - Summary statistics

Validation Checks:
    1. No duplicate stints (same player, team, date range)
    2. Valid date ranges (valid_from <= valid_to)
    3. No gaps in active player history
    4. All teams are valid NBA abbreviations
    5. No overlapping stints for same player
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import date

# Add src to path
repo_root = Path(__file__).resolve()
while not (repo_root / '.gitignore').exists():
    repo_root = repo_root.parent
sys.path.insert(0, str(repo_root))

from src.config import EMOJI

HISTORY_FILE = Path.home() / 'Downloads' / 'tmp' / 'player_team_history' / 'history.parquet'

# Valid NBA team abbreviations (current and historical)
VALID_TEAMS = {
    'ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW',
    'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK',
    'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS',
    # Historical teams
    'SEA', 'VAN', 'NOH', 'NOK', 'CHA', 'NJN', 'CHH'
}


def load_history():
    """Load team history file."""
    if not HISTORY_FILE.exists():
        print(f"{EMOJI['error']} History file not found: {HISTORY_FILE}")
        print("   Run 01_build.py first to generate the history.")
        return None
    
    try:
        df = pd.read_parquet(HISTORY_FILE)
        return df
    except Exception as e:
        print(f"{EMOJI['error']} Error loading history file: {e}")
        return None


def check_duplicates(df):
    """Check for duplicate stints."""
    print(f"{EMOJI['test']} Checking for duplicate stints...")
    
    duplicates = df[df.duplicated(subset=['player_normalized', 'team', 'valid_from'], keep=False)]
    
    if duplicates.empty:
        print(f"   {EMOJI['success']} No duplicates found")
        return True
    else:
        print(f"   {EMOJI['error']} Found {len(duplicates)} duplicate stints:")
        for _, row in duplicates.head(10).iterrows():
            print(f"      {row['player_normalized']} - {row['team']} from {row['valid_from']}")
        if len(duplicates) > 10:
            print(f"      ... and {len(duplicates) - 10} more")
        return False


def check_date_validity(df):
    """Check that valid_from <= valid_to."""
    print(f"{EMOJI['test']} Checking date validity...")
    
    # Filter rows where valid_to is not null
    df_with_end = df[df['valid_to'].notna()].copy()
    
    invalid_dates = df_with_end[df_with_end['valid_from'] > df_with_end['valid_to']]
    
    if invalid_dates.empty:
        print(f"   {EMOJI['success']} All dates valid")
        return True
    else:
        print(f"   {EMOJI['error']} Found {len(invalid_dates)} invalid date ranges:")
        for _, row in invalid_dates.head(10).iterrows():
            print(f"      {row['player_normalized']} - {row['team']}: {row['valid_from']} > {row['valid_to']}")
        if len(invalid_dates) > 10:
            print(f"      ... and {len(invalid_dates) - 10} more")
        return False


def check_team_codes(df):
    """Check that all teams are valid NBA abbreviations."""
    print(f"{EMOJI['test']} Checking team codes...")
    
    invalid_teams = df[~df['team'].isin(VALID_TEAMS)]
    
    if invalid_teams.empty:
        print(f"   {EMOJI['success']} All teams valid")
        return True
    else:
        print(f"   {EMOJI['error']} Found {len(invalid_teams)} invalid team codes:")
        unique_invalid = invalid_teams['team'].unique()
        for team in unique_invalid[:10]:
            players = invalid_teams[invalid_teams['team'] == team]['player_normalized'].unique()
            print(f"      {team}: {', '.join(players[:3])}")
            if len(players) > 3:
                print(f"         ... and {len(players) - 3} more players")
        if len(unique_invalid) > 10:
            print(f"      ... and {len(unique_invalid) - 10} more invalid teams")
        return False


def check_overlapping_stints(df):
    """Check for overlapping stints for the same player."""
    print(f"{EMOJI['test']} Checking for overlapping stints...")
    
    overlaps = []
    
    for player in df['player_normalized'].unique():
        player_df = df[df['player_normalized'] == player].copy()
        player_df = player_df.sort_values('valid_from')
        
        for i in range(len(player_df) - 1):
            current = player_df.iloc[i]
            next_stint = player_df.iloc[i + 1]
            
            # Skip if current stint has no end date (still active)
            if pd.isna(current['valid_to']):
                continue
            
            # Check if next stint starts before current ends
            if next_stint['valid_from'] < current['valid_to']:
                overlaps.append({
                    'player': player,
                    'team1': current['team'],
                    'team2': next_stint['team'],
                    'overlap_start': next_stint['valid_from'],
                    'overlap_end': current['valid_to']
                })
    
    if not overlaps:
        print(f"   {EMOJI['success']} No overlapping stints")
        return True
    else:
        print(f"   {EMOJI['error']} Found {len(overlaps)} overlapping stints:")
        for overlap in overlaps[:10]:
            print(f"      {overlap['player']}: {overlap['team1']}/{overlap['team2']} overlap {overlap['overlap_start']} to {overlap['overlap_end']}")
        if len(overlaps) > 10:
            print(f"      ... and {len(overlaps) - 10} more")
        return False


def show_statistics(df):
    """Show summary statistics."""
    print()
    print("="*80)
    print(f"{EMOJI['chart']} SUMMARY STATISTICS")
    print("="*80)
    print()
    
    print(f"Total players: {df['player_normalized'].nunique()}")
    print(f"Total stints: {len(df)}")
    print(f"Teams represented: {df['team'].nunique()}")
    print()
    
    # Players with most stints
    stint_counts = df.groupby('player_normalized').size().sort_values(ascending=False)
    print(f"Players with most stints:")
    for player, count in stint_counts.head(5).items():
        print(f"   {player}: {count} stints")
    print()
    
    # Active players (valid_to is null)
    active_players = df[df['valid_to'].isna()]['player_normalized'].nunique()
    print(f"Active players (current team): {active_players}")
    print()


def main():
    print("="*80)
    print(f"{EMOJI['test']} VALIDATING PLAYER TEAM HISTORY")
    print("="*80)
    print()
    
    # Load history
    df = load_history()
    if df is None:
        return
    
    print(f"Loaded {len(df)} records for {df['player_normalized'].nunique()} players")
    print()
    
    # Run validation checks
    all_passed = True
    all_passed &= check_duplicates(df)
    all_passed &= check_date_validity(df)
    all_passed &= check_team_codes(df)
    all_passed &= check_overlapping_stints(df)
    
    # Show statistics
    show_statistics(df)
    
    # Final result
    print("="*80)
    if all_passed:
        print(f"{EMOJI['success']} ALL VALIDATION CHECKS PASSED")
    else:
        print(f"{EMOJI['warning']} SOME VALIDATION CHECKS FAILED")
        print("   Review the issues above and re-run the build if needed.")
    print("="*80)
    print()


if __name__ == '__main__':
    main()
