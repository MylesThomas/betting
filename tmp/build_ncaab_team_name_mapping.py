"""
Build NCAAB Team Name Mapping (ESPN → Odds API)

Does a left join of ESPN outcomes with Odds API lines, then extracts:
1. Matched teams (ESPN → Odds API mapping)
2. ESPN teams with no Odds API match
3. Odds API teams with no ESPN match

Returns 3 dataframes for manual review to complete the mapping.

Usage (in notebook or script):
    from tmp.build_ncaab_team_name_mapping import build_team_mapping
    
    df_mapping, df_espn_only, df_odds_only = build_team_mapping(season='2024-25', use_cache=True)
    
    # Review unmatched
    print(df_espn_only)  # ESPN teams missing from Odds API
    print(df_odds_only)  # Odds API teams missing from ESPN

Author: Thomas Myles
Date: 2026-01-15
"""

import sys
import pandas as pd
import boto3
from pathlib import Path
from io import StringIO

# Find project root
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

from config_loader import get_config
CONFIG = get_config()

# Import functions from join script
sys.path.insert(0, str(PROJECT_ROOT / 'tmp'))
from join_ncaab_outcomes_and_lines import (
    load_game_outcomes,
    load_game_lines,
    SEASON_DATES
)

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def build_team_mapping(season='2024-25', use_cache=True):
    """
    Build team name mapping from ESPN to Odds API.
    
    Args:
        season: Season string (e.g., '2024-25')
        use_cache: Whether to use cached data
    
    Returns:
        df_mapping: DataFrame with columns [espn_team, odds_team] for matched teams
        df_espn_only: ESPN teams with no Odds API match (sorted A-Z)
        df_odds_only: Odds API teams with no ESPN match (sorted A-Z)
    """
    
    if season not in SEASON_DATES:
        raise ValueError(f"Unknown season: {season}")
    
    start_date, end_date = SEASON_DATES[season]
    
    print(f"Building team mapping for {season}...")
    print(f"Date range: {start_date} to {end_date}")
    
    # Load data
    outcomes_df = load_game_outcomes(start_date, end_date, use_cache=use_cache)
    lines_df = load_game_lines(start_date, end_date, use_cache=use_cache)
    
    if outcomes_df.empty or lines_df.empty:
        raise ValueError("No data loaded")
    
    print(f"\nLoaded:")
    print(f"  Outcomes: {len(outcomes_df)} games")
    print(f"  Lines: {len(lines_df)} games")
    
    # Get all unique team names
    espn_home = outcomes_df[['GAME_DATE', 'HOME_TEAM']].rename(columns={'HOME_TEAM': 'espn_team'})
    espn_away = outcomes_df[['GAME_DATE', 'AWAY_TEAM']].rename(columns={'AWAY_TEAM': 'espn_team'})
    espn_teams_df = pd.concat([espn_home, espn_away]).drop_duplicates()
    espn_teams_df.columns = ['date', 'espn_team']
    
    odds_home = lines_df[['date', 'home_team']].rename(columns={'home_team': 'odds_team'})
    odds_away = lines_df[['date', 'away_team']].rename(columns={'away_team': 'odds_team'})
    odds_teams_df = pd.concat([odds_home, odds_away]).drop_duplicates()
    
    print(f"\nUnique teams:")
    print(f"  ESPN: {espn_teams_df['espn_team'].nunique()}")
    print(f"  Odds API: {odds_teams_df['odds_team'].nunique()}")
    
    # LEFT JOIN: ESPN teams on left
    joined = espn_teams_df.merge(
        odds_teams_df,
        left_on=['date', 'espn_team'],
        right_on=['date', 'odds_team'],
        how='left'
    )
    
    # 1. Matched teams (has both espn_team and odds_team)
    matched = joined[joined['odds_team'].notna()].copy()
    df_mapping = matched[['espn_team', 'odds_team']].drop_duplicates().sort_values('espn_team')
    
    # 2. ESPN teams with no Odds API match (odds_team is null)
    unmatched_espn = joined[joined['odds_team'].isna()].copy()
    df_espn_only = unmatched_espn[['espn_team']].drop_duplicates().sort_values('espn_team')
    
    # 3. Odds API teams with no ESPN match (do RIGHT join to find these)
    joined_right = espn_teams_df.merge(
        odds_teams_df,
        left_on=['date', 'espn_team'],
        right_on=['date', 'odds_team'],
        how='right'
    )
    unmatched_odds = joined_right[joined_right['espn_team'].isna()].copy()
    df_odds_only = unmatched_odds[['odds_team']].drop_duplicates().sort_values('odds_team')
    
    print(f"\nResults:")
    print(f"  Matched teams: {len(df_mapping)}")
    print(f"  ESPN only: {len(df_espn_only)}")
    print(f"  Odds API only: {len(df_odds_only)}")
    
    return df_mapping, df_espn_only, df_odds_only


# =============================================================================
# CLI EXECUTION
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Build NCAAB team name mapping')
    parser.add_argument('--season', type=str, default='2024-25',
                       help='Season to process (e.g., "2024-25")')
    parser.add_argument('--use-cache', action='store_true',
                       help='Load data from cache')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("BUILD NCAAB TEAM NAME MAPPING (ESPN → Odds API)")
    print("=" * 80)
    
    df_mapping, df_espn_only, df_odds_only = build_team_mapping(
        season=args.season,
        use_cache=args.use_cache
    )
    
    print(f"\n{'='*80}")
    print("MATCHED TEAMS (ESPN → Odds API)")
    print(f"{'='*80}")
    print(df_mapping.to_string(index=False))
    
    print(f"\n{'='*80}")
    print("ESPN TEAMS WITH NO ODDS API MATCH (sorted A-Z)")
    print(f"{'='*80}")
    print(df_espn_only.to_string(index=False))
    
    print(f"\n{'='*80}")
    print("ODDS API TEAMS WITH NO ESPN MATCH (sorted A-Z)")
    print(f"{'='*80}")
    print(df_odds_only.to_string(index=False))
    
    print(f"\n{'='*80}")
    print("✅ Mapping complete!")
    print(f"{'='*80}")

