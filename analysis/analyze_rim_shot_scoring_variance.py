"""
Analyze Rim Shot Percentage as Proxy for Scoring Variance

Purpose:
    Explore whether % of shots within 6 feet and/or FG% within 6 feet correlates with
    scoring variance (over/under performance). High rim shot frequency might indicate:
    - More transition/rim-running opportunities (variance from pace)
    - More assisted looks vs self-created shots
    - Different scoring variance profiles
    
Context (from request):
    "lets look at % of shots that come within 6 feet to see if proxy for scoring variance
    (that, or % on shots within 6 feet)"

Metrics Explored:
    1. rim_shot_pct = rim_season_shots / total_season_shots * 100
    2. rim_fg_pct = FG% on shots 0-6 feet (already in data)
    3. pts_0_6_pct = % of total points from 0-6 feet (already in data)
    
Analysis:
    - Calculate scoring variance (std dev of points_diff)
    - Correlation between rim metrics and variance
    - Breakdown by player tier and team spread
    - Comparison across scorer types

Usage:
    # Run on 2025-26 season data
    python analysis/analyze_rim_shot_scoring_variance.py --season 2025-26 --rim-scorer-pct 40
    
    # Use local file
    python analysis/analyze_rim_shot_scoring_variance.py --input data/player_props_with_actuals_2025-26_rim40.csv

Author: Thomas Myles
Date: 2026-01-06
"""

import pandas as pd
import numpy as np
import sys
import argparse
from pathlib import Path
from io import BytesIO
import boto3

# =============================================================================
# PROJECT SETUP
# =============================================================================

current_dir = Path(__file__).parent
project_root = current_dir.parent
while not (project_root / '.gitignore').exists() and project_root != project_root.parent:
    project_root = project_root.parent

sys.path.insert(0, str(project_root))

# =============================================================================
# EMOJI MAP
# =============================================================================

EMOJI = {
    'success': '✅',
    'warning': '⚠️',
    'fire': '🔥',
    'chart': '📊',
    'target': '🎯',
    'light': '💡',
}

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(input_path=None, season='2025-26', rim_scorer_pct=40):
    """
    Load merged player props data with shot chart info
    
    Args:
        input_path: Path to CSV (local or S3), or None to auto-generate from season
        season: NBA season (used if input_path is None)
        rim_scorer_pct: Rim scorer threshold (used if input_path is None)
    
    Returns:
        DataFrame
    """
    if input_path is None:
        # Auto-generate S3 path
        s3_key = f"data/03_intermediate/player_props_with_actuals_{season}_rim{rim_scorer_pct}.csv"
        input_path = f"s3://nba-betting-mt/{s3_key}"
    
    print(f"\n{EMOJI['chart']} Loading data from: {input_path}")
    
    # Check if S3 or local
    if str(input_path).startswith('s3://'):
        # Load from S3
        s3_uri = str(input_path)
        parts = s3_uri.replace('s3://', '').split('/', 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ''
        
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(BytesIO(obj['Body'].read()))
        print(f"{EMOJI['success']} Loaded {len(df):,} player-game rows from S3")
    else:
        # Load from local file
        df = pd.read_csv(input_path)
        print(f"{EMOJI['success']} Loaded {len(df):,} player-game rows from local file")
    
    return df


# =============================================================================
# DATA PREP
# =============================================================================

def prepare_data(df):
    """
    Prepare data for variance analysis
    
    Adds:
        - rim_shot_pct: % of shots within 6 feet
        - points_diff: PTS - points_line (over/under margin)
    
    Filters:
        - Players with props (points_line not null)
        - Players with shot chart data
        - Players who played (MIN > 0)
    """
    print(f"\n{EMOJI['chart']} Preparing data...")
    
    # Filter to games with props
    df = df[df['points_line'].notna()].copy()
    print(f"   Rows with props: {len(df):,}")
    
    # Filter to players with shot chart data
    df = df[df['total_season_shots'].notna() & (df['total_season_shots'] > 0)].copy()
    print(f"   Rows with shot chart data: {len(df):,}")
    
    # Filter to players who played
    df = df[df['MIN'].notna() & (df['MIN'] > 0)].copy()
    print(f"   Rows where player played: {len(df):,}")
    
    # Calculate rim shot percentage
    df['rim_shot_pct'] = (df['rim_season_shots'] / df['total_season_shots'] * 100).fillna(0)
    
    # Calculate points differential (how much over/under the line they scored)
    df['points_diff'] = df['PTS'] - df['points_line']
    
    # Binary: did they go over?
    df['went_over'] = (df['points_diff'] > 0).astype(int)
    
    print(f"\n{EMOJI['success']} Data prepared:")
    print(f"   Total player-games: {len(df):,}")
    print(f"   Unique players: {df['PLAYER_NAME'].nunique():,}")
    print(f"   Date range: {df['game_date'].min()} to {df['game_date'].max()}")
    
    return df


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def print_section(title, emoji='chart'):
    """Print a formatted section header"""
    print(f"\n{'='*80}")
    print(f"{EMOJI[emoji]} {title}")
    print(f"{'='*80}")


def analyze_overall_correlations(df):
    """
    Calculate correlations between rim metrics and scoring variance
    """
    print_section("OVERALL CORRELATIONS: Rim Metrics vs Scoring Variance")
    
    # Calculate per-player variance metrics
    player_stats = df.groupby('PLAYER_NAME').agg({
        'points_diff': ['std', 'mean', 'count'],
        'rim_shot_pct': 'first',
        'rim_fg_pct': 'first',
        'pts_0_6_pct': 'first',
        'PTS': 'mean',
        'points_line': 'mean',
        'went_over': 'mean'
    }).reset_index()
    
    player_stats.columns = ['player', 'pts_diff_std', 'pts_diff_mean', 'games', 
                           'rim_shot_pct', 'rim_fg_pct', 'pts_0_6_pct', 
                           'avg_pts', 'avg_line', 'over_pct']
    
    # Filter to players with at least 10 games
    player_stats = player_stats[player_stats['games'] >= 10].copy()
    
    print(f"\nPlayers analyzed: {len(player_stats):,} (10+ games)")
    print(f"Average scoring variance (std dev): {player_stats['pts_diff_std'].mean():.2f} pts")
    
    # Calculate correlations
    print(f"\n{EMOJI['target']} Correlation with Scoring Variance (pts_diff_std):")
    
    corr_rim_shot = player_stats['rim_shot_pct'].corr(player_stats['pts_diff_std'])
    print(f"   rim_shot_pct:  {corr_rim_shot:+.3f}")
    
    corr_rim_fg = player_stats['rim_fg_pct'].corr(player_stats['pts_diff_std'])
    print(f"   rim_fg_pct:    {corr_rim_fg:+.3f}")
    
    corr_pts_06 = player_stats['pts_0_6_pct'].corr(player_stats['pts_diff_std'])
    print(f"   pts_0_6_pct:   {corr_pts_06:+.3f}")
    
    # Correlation with over rate
    print(f"\n{EMOJI['target']} Correlation with Over Rate (went_over %):")
    
    corr_rim_shot_over = player_stats['rim_shot_pct'].corr(player_stats['over_pct'])
    print(f"   rim_shot_pct:  {corr_rim_shot_over:+.3f}")
    
    corr_rim_fg_over = player_stats['rim_fg_pct'].corr(player_stats['over_pct'])
    print(f"   rim_fg_pct:    {corr_rim_fg_over:+.3f}")
    
    corr_pts_06_over = player_stats['pts_0_6_pct'].corr(player_stats['over_pct'])
    print(f"   pts_0_6_pct:   {corr_pts_06_over:+.3f}")
    
    # Show distribution of rim_shot_pct
    print(f"\n{EMOJI['chart']} Distribution of rim_shot_pct:")
    print(player_stats['rim_shot_pct'].describe())
    
    # Quartile analysis
    player_stats['rim_shot_quartile'] = pd.qcut(
        player_stats['rim_shot_pct'], 
        q=4, 
        labels=['Q1 (Low Rim)', 'Q2', 'Q3', 'Q4 (High Rim)']
    )
    
    print(f"\n{EMOJI['chart']} Variance by Rim Shot Quartile:")
    for quartile in ['Q1 (Low Rim)', 'Q2', 'Q3', 'Q4 (High Rim)']:
        q_data = player_stats[player_stats['rim_shot_quartile'] == quartile]
        if len(q_data) > 0:
            print(f"   {quartile:20s}: Avg std dev = {q_data['pts_diff_std'].mean():.2f} pts | "
                  f"Avg rim_shot_pct = {q_data['rim_shot_pct'].mean():.1f}% | "
                  f"Over rate = {q_data['over_pct'].mean()*100:.1f}%")
    
    return player_stats


def analyze_by_player_tier(df):
    """
    Analyze rim metrics and variance by player tier
    """
    print_section("ANALYSIS BY PLAYER TIER")
    
    # Bin players by their typical points line
    def bin_line(line):
        if line >= 30:
            return '30+ (Superstar)'
        elif line >= 25:
            return '25-30 (High Star)'
        elif line >= 20:
            return '20-25 (Star)'
        elif line >= 15:
            return '15-20 (High Role)'
        elif line >= 10:
            return '10-15 (Role Player)'
        else:
            return '<10 (Bench)'
    
    df['line_tier'] = df['points_line'].apply(bin_line)
    
    # Calculate metrics by tier
    tier_stats = df.groupby('line_tier').agg({
        'points_diff': ['std', 'mean'],
        'rim_shot_pct': 'mean',
        'rim_fg_pct': 'mean',
        'pts_0_6_pct': 'mean',
        'went_over': 'mean',
        'PLAYER_NAME': 'count'
    }).reset_index()
    
    tier_stats.columns = ['tier', 'pts_diff_std', 'pts_diff_mean', 
                         'avg_rim_shot_pct', 'avg_rim_fg_pct', 'avg_pts_0_6_pct',
                         'over_rate', 'games']
    
    # Sort by tier (descending points)
    tier_order = ['30+ (Superstar)', '25-30 (High Star)', '20-25 (Star)', 
                  '15-20 (High Role)', '10-15 (Role Player)', '<10 (Bench)']
    tier_stats['tier'] = pd.Categorical(tier_stats['tier'], categories=tier_order, ordered=True)
    tier_stats = tier_stats.sort_values('tier')
    
    print(f"\n{EMOJI['chart']} Variance and Rim Metrics by Player Tier:\n")
    print(f"{'Tier':<20} {'Games':>7} {'Variance':>9} {'Rim Shot%':>11} {'Rim FG%':>9} {'Pts 0-6%':>10} {'Over%':>8}")
    print("-" * 85)
    
    for _, row in tier_stats.iterrows():
        print(f"{row['tier']:<20} {row['games']:>7,} {row['pts_diff_std']:>8.2f}pts "
              f"{row['avg_rim_shot_pct']:>10.1f}% {row['avg_rim_fg_pct']:>8.1f}% "
              f"{row['avg_pts_0_6_pct']:>9.1f}% {row['over_rate']*100:>7.1f}%")
    
    return tier_stats


def analyze_by_rim_shot_buckets(df):
    """
    Bucket players by rim_shot_pct and analyze over/under performance
    """
    print_section("ANALYSIS BY RIM SHOT PERCENTAGE BUCKETS")
    
    # Create buckets
    df['rim_shot_bucket'] = pd.cut(
        df['rim_shot_pct'],
        bins=[0, 20, 30, 40, 50, 100],
        labels=['0-20% (Perimeter)', '20-30%', '30-40%', '40-50%', '50%+ (Heavy Rim)']
    )
    
    bucket_stats = df.groupby('rim_shot_bucket').agg({
        'points_diff': ['std', 'mean'],
        'went_over': 'mean',
        'rim_fg_pct': 'mean',
        'pts_0_6_pct': 'mean',
        'PTS': 'mean',
        'points_line': 'mean',
        'PLAYER_NAME': 'count'
    }).reset_index()
    
    bucket_stats.columns = ['bucket', 'pts_diff_std', 'pts_diff_mean', 
                           'over_rate', 'avg_rim_fg_pct', 'avg_pts_0_6_pct',
                           'avg_pts', 'avg_line', 'games']
    
    print(f"\n{EMOJI['chart']} Performance by Rim Shot % Bucket:\n")
    print(f"{'Bucket':<25} {'Games':>7} {'Over%':>8} {'Variance':>9} {'Avg Pts':>9} {'Avg Line':>10} {'Rim FG%':>9}")
    print("-" * 85)
    
    for _, row in bucket_stats.iterrows():
        print(f"{row['bucket']:<25} {row['games']:>7,} {row['over_rate']*100:>7.1f}% "
              f"{row['pts_diff_std']:>8.2f} {row['avg_pts']:>8.1f} "
              f"{row['avg_line']:>9.1f} {row['avg_rim_fg_pct']:>8.1f}%")
    
    return bucket_stats


def analyze_by_scorer_type_and_spread(df):
    """
    Analyze rim metrics by scorer type and team spread
    """
    print_section("ANALYSIS BY SCORER TYPE AND TEAM SPREAD")
    
    # Check if scorer_type exists
    if 'scorer_type' not in df.columns or df['scorer_type'].isna().all():
        print(f"\n{EMOJI['warning']} scorer_type not available in dataset - skipping this analysis")
        return None
    
    # Bin team spread
    def bin_spread(spread):
        if pd.isna(spread):
            return 'Unknown'
        elif spread < -10:
            return '10+ Fav'
        elif spread < -5:
            return '5-10 Fav'
        elif spread < -1:
            return '1-5 Fav'
        elif spread <= 1:
            return 'Pick em'
        elif spread <= 5:
            return '1-5 Dog'
        elif spread <= 10:
            return '5-10 Dog'
        else:
            return '10+ Dog'
    
    df['spread_bin'] = df['team_spread'].apply(bin_spread)
    
    # Filter to valid scorer types and spreads
    df_filtered = df[
        df['scorer_type'].notna() & 
        (df['spread_bin'] != 'Unknown')
    ].copy()
    
    print(f"\nFiltered to {len(df_filtered):,} games with scorer_type and spread data")
    
    # Aggregate by scorer type and spread
    combo_stats = df_filtered.groupby(['scorer_type', 'spread_bin']).agg({
        'points_diff': ['std', 'mean'],
        'went_over': 'mean',
        'rim_shot_pct': 'mean',
        'rim_fg_pct': 'mean',
        'PLAYER_NAME': 'count'
    }).reset_index()
    
    combo_stats.columns = ['scorer_type', 'spread_bin', 'pts_diff_std', 'pts_diff_mean',
                          'over_rate', 'avg_rim_shot_pct', 'avg_rim_fg_pct', 'games']
    
    # Filter to combos with 30+ games
    combo_stats = combo_stats[combo_stats['games'] >= 30].copy()
    
    # Sort by scorer type then spread
    spread_order = ['10+ Fav', '5-10 Fav', '1-5 Fav', 'Pick em', '1-5 Dog', '5-10 Dog', '10+ Dog']
    combo_stats['spread_bin'] = pd.Categorical(combo_stats['spread_bin'], categories=spread_order, ordered=True)
    combo_stats = combo_stats.sort_values(['scorer_type', 'spread_bin'])
    
    print(f"\n{EMOJI['chart']} Rim Metrics by Scorer Type and Spread (30+ games):\n")
    print(f"{'Scorer Type':<30} {'Spread':<12} {'Games':>7} {'Over%':>8} {'Variance':>9} {'Rim Shot%':>11} {'Rim FG%':>9}")
    print("-" * 95)
    
    for _, row in combo_stats.iterrows():
        print(f"{row['scorer_type']:<30} {row['spread_bin']:<12} {row['games']:>7,} "
              f"{row['over_rate']*100:>7.1f}% {row['pts_diff_std']:>8.2f} "
              f"{row['avg_rim_shot_pct']:>10.1f}% {row['avg_rim_fg_pct']:>8.1f}%")
    
    return combo_stats


def show_extreme_players(df, player_stats):
    """
    Show players with extreme rim shot percentages
    """
    print_section("PLAYERS WITH EXTREME RIM SHOT PERCENTAGES", "fire")
    
    # Top rim shooters (high rim_shot_pct)
    top_rim = player_stats.nlargest(15, 'rim_shot_pct')
    
    print(f"\n{EMOJI['fire']} TOP 15 RIM SHOOTERS (highest % of shots within 6 feet):\n")
    print(f"{'Player':<25} {'Rim Shot%':>11} {'Rim FG%':>9} {'Pts 0-6%':>10} {'Games':>7} {'Over%':>8} {'Variance':>9}")
    print("-" * 95)
    
    for _, row in top_rim.iterrows():
        print(f"{row['player']:<25} {row['rim_shot_pct']:>10.1f}% {row['rim_fg_pct']:>8.1f}% "
              f"{row['pts_0_6_pct']:>9.1f}% {row['games']:>7} {row['over_pct']*100:>7.1f}% "
              f"{row['pts_diff_std']:>8.2f}")
    
    # Bottom rim shooters (low rim_shot_pct)
    bottom_rim = player_stats.nsmallest(15, 'rim_shot_pct')
    
    print(f"\n{EMOJI['chart']} BOTTOM 15 RIM SHOOTERS (lowest % of shots within 6 feet):\n")
    print(f"{'Player':<25} {'Rim Shot%':>11} {'Rim FG%':>9} {'Pts 0-6%':>10} {'Games':>7} {'Over%':>8} {'Variance':>9}")
    print("-" * 95)
    
    for _, row in bottom_rim.iterrows():
        print(f"{row['player']:<25} {row['rim_shot_pct']:>10.1f}% {row['rim_fg_pct']:>8.1f}% "
              f"{row['pts_0_6_pct']:>9.1f}% {row['games']:>7} {row['over_pct']*100:>7.1f}% "
              f"{row['pts_diff_std']:>8.2f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Analyze rim shot percentage as proxy for scoring variance'
    )
    parser.add_argument('--input', type=str, default=None,
                       help='Input CSV path (local or S3). If not provided, uses --season and --rim-scorer-pct')
    parser.add_argument('--season', type=str, default='2025-26',
                       help='NBA season (used if --input not provided)')
    parser.add_argument('--rim-scorer-pct', type=int, default=40,
                       help='Rim scorer threshold used in dataset (default: 40)')
    
    args = parser.parse_args()
    
    # Load data
    df = load_data(
        input_path=args.input,
        season=args.season,
        rim_scorer_pct=args.rim_scorer_pct
    )
    
    # Prepare data
    df = prepare_data(df)
    
    # Run analyses
    player_stats = analyze_overall_correlations(df)
    analyze_by_player_tier(df)
    analyze_by_rim_shot_buckets(df)
    analyze_by_scorer_type_and_spread(df)
    show_extreme_players(df, player_stats)
    
    print(f"\n{'='*80}")
    print(f"{EMOJI['success']} Analysis complete!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()


