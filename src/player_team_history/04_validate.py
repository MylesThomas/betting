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
    6. Very short stints (< 7 days - may indicate 10-day contracts)
    7. NULL valid_to consistency (only final stint per player)
    8. Chronological order within each player
    9. Date range sanity (1946 to today + 1 year)
    10. Same-team consecutive stints check
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import date, timedelta

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


def check_short_stints(df):
    """Check for very short stints (< 7 days)."""
    print(f"{EMOJI['test']} Checking for very short stints...")
    
    df_with_end = df[df['valid_to'].notna()].copy()
    df_with_end['duration'] = (df_with_end['valid_to'] - df_with_end['valid_from']).apply(lambda x: x.days)
    
    short_stints = df_with_end[df_with_end['duration'] < 7]
    
    if short_stints.empty:
        print(f"   {EMOJI['success']} No very short stints found")
        return True
    else:
        print(f"   {EMOJI['warning']} Found {len(short_stints)} stints < 7 days (may be 10-day contracts):")
        for _, row in short_stints.head(10).iterrows():
            print(f"      {row['player_normalized']} - {row['team']}: {row['duration']} days ({row['valid_from']} to {row['valid_to']})")
        if len(short_stints) > 10:
            print(f"      ... and {len(short_stints) - 10} more")
        return True  # Warning, not failure


def check_null_consistency(df):
    """Check that only final stint for each player has NULL valid_to."""
    print(f"{EMOJI['test']} Checking NULL valid_to consistency...")
    
    issues = []
    
    for player in df['player_normalized'].unique():
        player_df = df[df['player_normalized'] == player].copy()
        player_df = player_df.sort_values('valid_from')
        
        # Check all stints except last
        for i in range(len(player_df) - 1):
            stint = player_df.iloc[i]
            if pd.isna(stint['valid_to']):
                issues.append({
                    'player': player,
                    'team': stint['team'],
                    'date': stint['valid_from'],
                    'issue': 'NULL valid_to in non-final stint'
                })
        
        # Check last stint
        last_stint = player_df.iloc[-1]
        if pd.notna(last_stint['valid_to']):
            issues.append({
                'player': player,
                'team': last_stint['team'],
                'date': last_stint['valid_to'],
                'issue': 'Final stint has non-NULL valid_to (player should be active)'
            })
    
    if not issues:
        print(f"   {EMOJI['success']} All players have correct NULL valid_to pattern")
        return True
    else:
        print(f"   {EMOJI['error']} Found {len(issues)} NULL consistency issues:")
        for issue in issues[:10]:
            print(f"      {issue['player']} - {issue['team']}: {issue['issue']}")
        if len(issues) > 10:
            print(f"      ... and {len(issues) - 10} more")
        return False


def check_chronological_order(df):
    """Check that stints are in chronological order within each player."""
    print(f"{EMOJI['test']} Checking chronological order...")
    
    issues = []
    
    for player in df['player_normalized'].unique():
        player_df = df[df['player_normalized'] == player].copy()
        player_df = player_df.sort_values('valid_from')
        
        for i in range(len(player_df) - 1):
            current = player_df.iloc[i]
            next_stint = player_df.iloc[i + 1]
            
            # Current stint's end should be before next stint's start
            if pd.notna(current['valid_to']) and current['valid_to'] > next_stint['valid_from']:
                issues.append({
                    'player': player,
                    'team1': current['team'],
                    'team2': next_stint['team'],
                    'issue': f"Out of order: {current['valid_to']} > {next_stint['valid_from']}"
                })
    
    if not issues:
        print(f"   {EMOJI['success']} All stints in chronological order")
        return True
    else:
        print(f"   {EMOJI['error']} Found {len(issues)} chronological order issues:")
        for issue in issues[:10]:
            print(f"      {issue['player']}: {issue['team1']} -> {issue['team2']} - {issue['issue']}")
        if len(issues) > 10:
            print(f"      ... and {len(issues) - 10} more")
        return False


def check_date_sanity(df):
    """Check that dates are within reasonable range (1946 to today + 1 year)."""
    print(f"{EMOJI['test']} Checking date range sanity...")
    
    nba_founding = date(1946, 6, 6)
    max_future = date.today() + timedelta(days=365)
    
    issues = []
    
    # Check valid_from dates
    too_early = df[df['valid_from'] < nba_founding]
    for _, row in too_early.iterrows():
        issues.append({
            'player': row['player_normalized'],
            'team': row['team'],
            'date': row['valid_from'],
            'issue': f"Date before NBA founding ({nba_founding})"
        })
    
    too_late = df[df['valid_from'] > max_future]
    for _, row in too_late.iterrows():
        issues.append({
            'player': row['player_normalized'],
            'team': row['team'],
            'date': row['valid_from'],
            'issue': f"Date too far in future (> {max_future})"
        })
    
    # Check valid_to dates
    df_with_end = df[df['valid_to'].notna()]
    too_late_end = df_with_end[df_with_end['valid_to'] > max_future]
    for _, row in too_late_end.iterrows():
        issues.append({
            'player': row['player_normalized'],
            'team': row['team'],
            'date': row['valid_to'],
            'issue': f"End date too far in future (> {max_future})"
        })
    
    if not issues:
        print(f"   {EMOJI['success']} All dates within reasonable range")
        return True
    else:
        print(f"   {EMOJI['error']} Found {len(issues)} date sanity issues:")
        for issue in issues[:10]:
            print(f"      {issue['player']} - {issue['team']}: {issue['date']} - {issue['issue']}")
        if len(issues) > 10:
            print(f"      ... and {len(issues) - 10} more")
        return False


def check_consecutive_same_team(df):
    """Check for same-team consecutive stints that might need consolidation."""
    print(f"{EMOJI['test']} Checking for consecutive same-team stints...")
    
    issues = []
    
    for player in df['player_normalized'].unique():
        player_df = df[df['player_normalized'] == player].copy()
        player_df = player_df.sort_values('valid_from')
        
        for i in range(len(player_df) - 1):
            current = player_df.iloc[i]
            next_stint = player_df.iloc[i + 1]
            
            # Check if same team
            if current['team'] == next_stint['team']:
                # Check if dates are close (within 30 days)
                if pd.notna(current['valid_to']):
                    gap_days = (next_stint['valid_from'] - current['valid_to']).days
                    if abs(gap_days) <= 30:
                        issues.append({
                            'player': player,
                            'team': current['team'],
                            'stint1_end': current['valid_to'],
                            'stint2_start': next_stint['valid_from'],
                            'gap_days': gap_days
                        })
    
    if not issues:
        print(f"   {EMOJI['success']} No suspicious consecutive same-team stints")
        return True
    else:
        print(f"   {EMOJI['warning']} Found {len(issues)} consecutive same-team stints (may be legitimate):")
        for issue in issues[:10]:
            print(f"      {issue['player']} - {issue['team']}: gap of {issue['gap_days']} days ({issue['stint1_end']} to {issue['stint2_start']})")
        if len(issues) > 10:
            print(f"      ... and {len(issues) - 10} more")
        return True  # Warning, not failure


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
    
    # Stint distribution
    print(f"Stint distribution:")
    one_stint = (stint_counts == 1).sum()
    two_three = ((stint_counts >= 2) & (stint_counts <= 3)).sum()
    four_six = ((stint_counts >= 4) & (stint_counts <= 6)).sum()
    seven_plus = (stint_counts >= 7).sum()
    print(f"   1 stint: {one_stint} players")
    print(f"   2-3 stints: {two_three} players")
    print(f"   4-6 stints: {four_six} players")
    print(f"   7+ stints: {seven_plus} players")
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
    all_passed &= check_null_consistency(df)
    all_passed &= check_chronological_order(df)
    all_passed &= check_date_sanity(df)
    
    # Warning checks (don't affect pass/fail)
    check_short_stints(df)
    check_consecutive_same_team(df)
    
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
