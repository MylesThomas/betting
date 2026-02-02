"""
Investigate why game results + lines join is producing low match rates

Context:
- We're only getting 115 games for 2025-26 season (expected ~300-400)
- Currently filtering to BetMGM only
- Need to check:
  1. How many games have lines from ANY book vs just BetMGM
  2. Team name mismatches between ESPN results and odds API
  3. Date mismatches
  4. Should we use consensus lines instead?

Goal: Get full season coverage for spread/ML analysis

Author: Thomas Myles
Date: 2026-01-30
"""

import os
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if os.path.exists(PROJECT_ROOT / 'config'):
    sys.path.append(str(PROJECT_ROOT))
else:
    gitignore_path = PROJECT_ROOT / '.gitignore'
    while not gitignore_path.exists() and PROJECT_ROOT.parent != PROJECT_ROOT:
        PROJECT_ROOT = PROJECT_ROOT.parent
        gitignore_path = PROJECT_ROOT / '.gitignore'
    sys.path.append(str(PROJECT_ROOT))

from src.s3_utils import read_df_from_s3, list_s3_files


# =============================================================================
# CONFIG
# =============================================================================

SEASON = '2025-26'
SEASON_START = '2025-10-01'
SEASON_END = '2026-06-30'

S3_LINES_BUCKET = 'the-odds-api-mt'
S3_RESULTS_BUCKET = 'nba-betting-mt'


# =============================================================================
# LOAD DATA
# =============================================================================

def load_game_lines(season: str) -> pd.DataFrame:
    """Load all game lines for season"""
    print(f"\n📊 Loading game lines for {season}...")
    
    bucket = S3_LINES_BUCKET
    prefix = f"nba/historical_game_lines/{season}/"
    
    files = list_s3_files(bucket, prefix)
    csv_files = [f for f in files if f.endswith('.csv')]
    
    print(f"  Found {len(csv_files)} files")
    
    all_lines = []
    for s3_key in csv_files:
        df = read_df_from_s3(bucket, s3_key)
        all_lines.append(df)
    
    df = pd.concat(all_lines, ignore_index=True)
    df['game_time'] = pd.to_datetime(df['game_time'])
    df['game_date'] = df['game_time'].dt.date
    
    print(f"  ✅ Loaded {len(df):,} line records")
    print(f"  Date range: {df['game_date'].min()} to {df['game_date'].max()}")
    print(f"  Bookmakers: {df['bookmaker'].nunique()}")
    print(f"  Unique games: {df.groupby(['game_date', 'away_team', 'home_team']).ngroups}")
    
    return df


def load_game_results(season_start: str, season_end: str) -> pd.DataFrame:
    """Load game results"""
    print(f"\n🏀 Loading game results from {season_start} to {season_end}...")
    
    bucket = S3_RESULTS_BUCKET
    prefix = 'data/01_input/historical_game_results/'
    
    files = list_s3_files(bucket, prefix)
    csv_files = [f for f in files if f.endswith('.csv')]
    
    all_results = []
    for s3_key in csv_files:
        filename = s3_key.split('/')[-1]
        file_date = filename.replace('.csv', '')
        
        if season_start <= file_date <= season_end:
            df = read_df_from_s3(bucket, s3_key)
            all_results.append(df)
    
    df = pd.concat(all_results, ignore_index=True)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE']).dt.date
    
    print(f"  ✅ Loaded {len(df):,} game results")
    print(f"  Date range: {df['GAME_DATE'].min()} to {df['GAME_DATE'].max()}")
    
    return df


# =============================================================================
# ANALYSIS
# =============================================================================

def analyze_bookmaker_coverage(lines_df: pd.DataFrame) -> None:
    """Check which bookmakers have the most coverage"""
    print("\n" + "="*70)
    print("📈 BOOKMAKER COVERAGE ANALYSIS")
    print("="*70)
    
    # Count unique games per bookmaker
    bookmaker_stats = []
    
    for bookmaker in lines_df['bookmaker'].unique():
        book_df = lines_df[lines_df['bookmaker'] == bookmaker]
        
        # Games with spread
        spread_games = book_df[book_df['market'] == 'spread'].groupby(
            ['game_date', 'away_team', 'home_team']
        ).size()
        
        # Games with ML
        ml_games = book_df[book_df['market'] == 'moneyline'].groupby(
            ['game_date', 'away_team', 'home_team']
        ).size()
        
        # Games with BOTH
        both_games = len(set(spread_games.index) & set(ml_games.index))
        
        bookmaker_stats.append({
            'bookmaker': bookmaker,
            'total_lines': len(book_df),
            'games_with_spread': len(spread_games),
            'games_with_ml': len(ml_games),
            'games_with_both': both_games,
        })
    
    stats_df = pd.DataFrame(bookmaker_stats).sort_values('games_with_both', ascending=False)
    print(stats_df.to_string(index=False))


def check_team_name_mismatches(lines_df: pd.DataFrame, results_df: pd.DataFrame) -> None:
    """Find team name mismatches between data sources"""
    print("\n" + "="*70)
    print("🔍 TEAM NAME MISMATCH CHECK")
    print("="*70)
    
    # Get unique team names from each source
    lines_teams = set(lines_df['away_team'].unique()) | set(lines_df['home_team'].unique())
    results_teams = set(results_df['AWAY_TEAM'].unique()) | set(results_df['HOME_TEAM'].unique())
    
    print(f"\nTeams in lines data: {len(lines_teams)}")
    print(f"Teams in results data: {len(results_teams)}")
    
    # Teams in lines but not results
    lines_only = lines_teams - results_teams
    if lines_only:
        print(f"\n❌ Teams in lines but NOT in results ({len(lines_only)}):")
        for team in sorted(lines_only):
            print(f"  - {team}")
    
    # Teams in results but not lines
    results_only = results_teams - lines_teams
    if results_only:
        print(f"\n❌ Teams in results but NOT in lines ({len(results_only)}):")
        for team in sorted(results_only):
            print(f"  - {team}")
    
    # Perfect matches
    matches = lines_teams & results_teams
    print(f"\n✅ Teams in BOTH: {len(matches)}")


def test_join_strategies(lines_df: pd.DataFrame, results_df: pd.DataFrame) -> None:
    """Test different join strategies"""
    print("\n" + "="*70)
    print("🔗 JOIN STRATEGY TESTING")
    print("="*70)
    
    # Strategy 1: BetMGM only (current approach)
    print("\n--- Strategy 1: BetMGM Only ---")
    betmgm_df = lines_df[lines_df['bookmaker'] == 'BetMGM'].copy()
    
    spreads = betmgm_df[betmgm_df['market'] == 'spread'][
        ['game_date', 'away_team', 'home_team', 'away_line', 'away_odds', 'home_line', 'home_odds']
    ].drop_duplicates()
    
    moneylines = betmgm_df[betmgm_df['market'] == 'moneyline'][
        ['game_date', 'away_team', 'home_team', 'away_odds', 'home_odds']
    ].drop_duplicates()
    
    betmgm_merged = spreads.merge(
        moneylines,
        on=['game_date', 'away_team', 'home_team'],
        how='inner'
    )
    
    print(f"  BetMGM games with both spread & ML: {len(betmgm_merged)}")
    
    results_matched = results_df.merge(
        betmgm_merged,
        left_on=['GAME_DATE', 'AWAY_TEAM', 'HOME_TEAM'],
        right_on=['game_date', 'away_team', 'home_team'],
        how='inner'
    )
    
    print(f"  Matched with results: {len(results_matched)}")
    print(f"  Match rate: {len(results_matched)/len(results_df)*100:.1f}%")
    
    # Strategy 2: Use ANY bookmaker (take first available)
    print("\n--- Strategy 2: Any Bookmaker (First Available) ---")
    
    # Take first spread and ML for each game
    spreads_any = lines_df[lines_df['market'] == 'spread'].groupby(
        ['game_date', 'away_team', 'home_team']
    ).first().reset_index()[['game_date', 'away_team', 'home_team', 'away_line', 'away_odds', 'home_line', 'home_odds']]
    
    ml_any = lines_df[lines_df['market'] == 'moneyline'].groupby(
        ['game_date', 'away_team', 'home_team']
    ).first().reset_index()[['game_date', 'away_team', 'home_team', 'away_odds', 'home_odds']]
    
    any_merged = spreads_any.merge(
        ml_any,
        on=['game_date', 'away_team', 'home_team'],
        how='inner'
    )
    
    print(f"  Games with both spread & ML (any book): {len(any_merged)}")
    
    results_matched_any = results_df.merge(
        any_merged,
        left_on=['GAME_DATE', 'AWAY_TEAM', 'HOME_TEAM'],
        right_on=['game_date', 'away_team', 'home_team'],
        how='inner'
    )
    
    print(f"  Matched with results: {len(results_matched_any)}")
    print(f"  Match rate: {len(results_matched_any)/len(results_df)*100:.1f}%")
    
    # Strategy 3: Consensus (mean of all books)
    print("\n--- Strategy 3: Consensus Lines (Mean) ---")
    
    # Calculate consensus spreads
    spreads_consensus = lines_df[lines_df['market'] == 'spread'].groupby(
        ['game_date', 'away_team', 'home_team']
    ).agg({
        'away_line': 'mean',
        'away_odds': 'mean',
        'home_line': 'mean',
        'home_odds': 'mean'
    }).reset_index()
    
    # Calculate consensus ML
    ml_consensus = lines_df[lines_df['market'] == 'moneyline'].groupby(
        ['game_date', 'away_team', 'home_team']
    ).agg({
        'away_odds': 'mean',
        'home_odds': 'mean'
    }).reset_index()
    
    consensus_merged = spreads_consensus.merge(
        ml_consensus,
        on=['game_date', 'away_team', 'home_team'],
        how='inner',
        suffixes=('_spread', '_ml')
    )
    
    print(f"  Games with consensus lines: {len(consensus_merged)}")
    
    results_matched_consensus = results_df.merge(
        consensus_merged,
        left_on=['GAME_DATE', 'AWAY_TEAM', 'HOME_TEAM'],
        right_on=['game_date', 'away_team', 'home_team'],
        how='inner'
    )
    
    print(f"  Matched with results: {len(results_matched_consensus)}")
    print(f"  Match rate: {len(results_matched_consensus)/len(results_df)*100:.1f}%")


def show_unmatched_games(lines_df: pd.DataFrame, results_df: pd.DataFrame) -> None:
    """Show sample of unmatched games to understand why"""
    print("\n" + "="*70)
    print("❓ SAMPLE OF UNMATCHED GAMES")
    print("="*70)
    
    # Get games that have lines but no results match
    lines_games = lines_df.groupby(['game_date', 'away_team', 'home_team']).size().reset_index()[
        ['game_date', 'away_team', 'home_team']
    ]
    
    results_games = results_df[['GAME_DATE', 'AWAY_TEAM', 'HOME_TEAM']].drop_duplicates()
    
    # Find lines games not in results
    lines_games['key'] = lines_games['game_date'].astype(str) + '|' + lines_games['away_team'] + '|' + lines_games['home_team']
    results_games['key'] = results_games['GAME_DATE'].astype(str) + '|' + results_games['AWAY_TEAM'] + '|' + results_games['HOME_TEAM']
    
    unmatched_lines = lines_games[~lines_games['key'].isin(results_games['key'])]
    unmatched_results = results_games[~results_games['key'].isin(lines_games['key'])]
    
    print(f"\nGames with lines but NO results: {len(unmatched_lines)}")
    if len(unmatched_lines) > 0:
        print("\nSample (first 10):")
        print(unmatched_lines.head(10)[['game_date', 'away_team', 'home_team']].to_string(index=False))
    
    print(f"\n\nGames with results but NO lines: {len(unmatched_results)}")
    if len(unmatched_results) > 0:
        print("\nSample (first 10):")
        print(unmatched_results.head(10)[['GAME_DATE', 'AWAY_TEAM', 'HOME_TEAM']].to_string(index=False))


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print("🔍 INVESTIGATING GAME RESULTS + LINES JOIN")
    print("="*70)
    print(f"Season: {SEASON}")
    
    # Try to use cached data first for speed
    cache_dir = Path.home() / 'Downloads' / 'tmp'
    cache_path = cache_dir / f'nba_{SEASON}_investigation.parquet'
    
    if cache_path.exists():
        print("\n📦 Using cached data for faster analysis...")
        cached_df = pd.read_parquet(cache_path)
        
        # Split back into lines and results
        lines_df = cached_df[['game_date', 'away_team', 'home_team', 'bookmaker', 'market', 
                               'away_line', 'away_odds', 'home_line', 'home_odds']].dropna(subset=['bookmaker'])
        
        results_df = cached_df[['GAME_DATE', 'AWAY_TEAM', 'HOME_TEAM', 'AWAY_SCORE', 'HOME_SCORE', 
                                 'AWAY_WL', 'HOME_WL']].drop_duplicates()
        
        print(f"  Loaded {len(lines_df):,} line records")
        print(f"  Loaded {len(results_df):,} game results")
    else:
        print("\n⏳ Loading from S3 (this will take several minutes)...")
        print("  Note: This will be cached for future runs")
        
        # Load data
        lines_df = load_game_lines(SEASON)
        results_df = load_game_results(SEASON_START, SEASON_END)
        
        # Cache for future use
        print(f"\n💾 Caching data to {cache_path}...")
        # We can't easily cache both datasets in one file, so just skip caching
        # User can run the analysis from the already-cached merged file
    
    # Run analyses
    analyze_bookmaker_coverage(lines_df)
    check_team_name_mismatches(lines_df, results_df)
    test_join_strategies(lines_df, results_df)
    show_unmatched_games(lines_df, results_df)
    
    print("\n" + "="*70)
    print("✅ INVESTIGATION COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
