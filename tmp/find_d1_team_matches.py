"""
Find and display D1 NCAAB team matching status between ESPN and Odds API.

This script loads team matching data and displays statistics about:
- Total D1 teams
- Matched teams (ESPN → Odds API)
- Unmatched teams
- Win/loss records
- ATS (Against The Spread) statistics

Usage:
    # Run with default season (2024-25) and use cache
    python tmp/find_d1_team_matches.py
    
    # Run with different season
    python tmp/find_d1_team_matches.py --season 2023-24
    
    # Force reload without cache
    python tmp/find_d1_team_matches.py --no-cache
    
    # Show only unmatched teams
    python tmp/find_d1_team_matches.py --show-unmatched
    
    # Show only matched teams
    python tmp/find_d1_team_matches.py --show-matched

Author: Thomas Myles
Date: 2026-01-15
"""

import sys
from pathlib import Path
import argparse

# Setup paths
def find_project_root():
    """Find project root by looking for .gitignore file."""
    current = Path.cwd()
    while current != current.parent:
        if (current / '.gitignore').exists():
            return current
        current = current.parent
    return Path.cwd()

PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'tmp'))

from build_ncaab_team_name_mapping_v2 import get_d1_team_matching_df


def display_team_stats(df):
    """Display team matching statistics."""
    print(f"\n{'='*80}")
    print("TEAM MATCHING SUMMARY")
    print(f"{'='*80}")
    
    print(f"\nTotal teams: {len(df)}")
    print(f"Matched: {df['matched'].sum()} ({df['matched'].sum()/len(df)*100:.1f}%)")
    print(f"Unmatched: {(~df['matched']).sum()} ({(~df['matched']).sum()/len(df)*100:.1f}%)")
    
    # Win percentage stats
    avg_win_pct = df['espn_win_pct'].mean()
    print(f"\nAverage Win %: {avg_win_pct*100:.1f}%")
    
    # ATS stats (only for teams with ATS data)
    teams_with_ats = df[df['ats_games'] > 0]
    if len(teams_with_ats) > 0:
        avg_ats_pct = teams_with_ats['ats_win_pct'].mean()
        print(f"Average ATS Win % (teams with betting data): {avg_ats_pct*100:.1f}%")
        print(f"Teams with ATS data: {len(teams_with_ats)} ({len(teams_with_ats)/len(df)*100:.1f}%)")


def display_top_teams(df, n=10):
    """Display top teams by various metrics."""
    print(f"\n{'='*80}")
    print(f"TOP {n} TEAMS BY METRICS")
    print(f"{'='*80}")
    
    # Top by wins
    print(f"\n🏆 Top {n} by Win %:")
    top_wins = df.nlargest(n, 'espn_win_pct')
    for i, row in enumerate(top_wins.itertuples(), 1):
        print(f"   {i:2}. {row.team_name_espn:<40} {row.espn_win_pct*100:5.1f}% ({row.espn_wins}-{row.espn_losses})")
    
    # Top by ATS (only teams with ATS data)
    teams_with_ats = df[df['ats_games'] > 0]
    if len(teams_with_ats) >= n:
        print(f"\n💰 Top {n} by ATS Win %:")
        top_ats = teams_with_ats.nlargest(n, 'ats_win_pct')
        for i, row in enumerate(top_ats.itertuples(), 1):
            print(f"   {i:2}. {row.team_name_espn:<40} {row.ats_win_pct*100:5.1f}% ({row.ats_wins}-{row.ats_losses}-{row.ats_pushes})")


def display_unmatched_teams(df):
    """Display teams that need manual matching."""
    unmatched = df[~df['matched']].copy()
    
    if len(unmatched) == 0:
        print(f"\n✅ All teams matched!")
        return
    
    print(f"\n{'='*80}")
    print(f"UNMATCHED TEAMS ({len(unmatched)})")
    print(f"{'='*80}")
    print("\nThese teams exist in ESPN data but couldn't be automatically matched to Odds API:")
    
    # Sort by games played (most active teams first)
    unmatched = unmatched.sort_values('espn_games', ascending=False)
    
    for i, row in enumerate(unmatched.itertuples(), 1):
        print(f"\n{i:2}. {row.team_name_espn}")
        print(f"    Games: {row.espn_games} | Record: {row.espn_wins}-{row.espn_losses} ({row.espn_win_pct*100:.1f}%)")
        if row.ats_games > 0:
            print(f"    ATS: {row.ats_wins}-{row.ats_losses}-{row.ats_pushes} ({row.ats_win_pct*100:.1f}%)")


def display_matched_teams(df):
    """Display successfully matched teams."""
    matched = df[df['matched']].copy()
    
    print(f"\n{'='*80}")
    print(f"MATCHED TEAMS ({len(matched)})")
    print(f"{'='*80}")
    
    # Sort by games played
    matched = matched.sort_values('espn_games', ascending=False)
    
    print(f"\n{'Team Name (ESPN)':<45} {'Odds API':<45}")
    print(f"{'-'*90}")
    
    for row in matched.head(20).itertuples():
        print(f"{row.team_name_espn:<45} {row.team_name_odds_api or 'N/A':<45}")
    
    if len(matched) > 20:
        print(f"\n... and {len(matched) - 20} more teams")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Find and display D1 NCAAB team matching status'
    )
    parser.add_argument(
        '--season',
        type=str,
        default='2024-25',
        help='Season to analyze (e.g., 2024-25)'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Force reload data without using cache'
    )
    parser.add_argument(
        '--show-unmatched',
        action='store_true',
        help='Show only unmatched teams'
    )
    parser.add_argument(
        '--show-matched',
        action='store_true',
        help='Show only matched teams'
    )
    parser.add_argument(
        '--top',
        type=int,
        default=10,
        help='Number of top teams to display (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Get the matching dataframe
    use_cache = not args.no_cache
    df = get_d1_team_matching_df(season=args.season, use_cache=use_cache)
    
    # Display based on flags
    if args.show_unmatched:
        display_unmatched_teams(df)
    elif args.show_matched:
        display_matched_teams(df)
    else:
        # Default: show everything
        display_team_stats(df)
        display_top_teams(df, n=args.top)
        display_unmatched_teams(df)
    
    print(f"\n{'='*80}")
    print("✅ Analysis complete!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

