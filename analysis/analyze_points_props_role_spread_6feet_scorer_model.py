"""
NBA Player Props 3D Matrix Analysis - 2025-26 Season (Scorer Type Edition)

Purpose:
    Analyze player prop betting edges by creating a 3-dimensional matrix of:
    - Player tier (by points line: 30+, 20-30, 10-20, <10)
    - Team spread context (10+ Dog, 5-10 Dog, etc.)
    - Scorer type (50%+ points within 6 feet vs <50% within 6 feet)
    
    Goal: Find systematic pricing inefficiencies in the prop betting market
    by identifying which player types AND scoring styles in which game contexts 
    consistently beat or miss their lines.
    
    Hypothesis: The market may misprice props for rim-attacking players (50%+ within 6ft)
    vs perimeter-oriented players differently based on game context (spreads).

Key Features:
    1. Filters out injury/early exit games (≤25% usual minutes)
    2. Creates 3D tier × spread × scorer_type matrix with over/under hit rates
    3. Identifies best betting edges with sufficient sample size
    4. Shows which specific players drive each edge
    5. Compares rim attackers vs perimeter players in different contexts

Usage:
    # Quick: Use season flag (auto-generates S3 paths)
    python analysis/analyze_points_props_role_spread_6feet_scorer_model.py --season 2025-26 --granularity detailed --min-roi 5.0
    
    # Explicit input/output (local or S3)
    python analysis/analyze_points_props_role_spread_6feet_scorer_model.py \
        --input s3://nba-betting-mt/data/03_intermediate/player_props_with_actuals_2025-26.csv \
        --output s3://nba-betting-mt/data/03_intermediate/6feet_scorer_strategies.json \
        --granularity detailed \
        --min-roi 5.0
    
    # Local files
    python analysis/analyze_points_props_role_spread_6feet_scorer_model.py \
        --input data/merged_data.csv \
        --output strategies_6feet.json \
        --granularity standard \
        --min-roi 5.0

Output:
    - Console output with 3D matrices and scorer-type insights
    - JSON strategies file with scorer_type dimension included
    - Comparison of rim attackers vs perimeter players by context

Author: Extended from 2D matrix analysis on 2026-01-06
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from io import BytesIO

# =============================================================================
# CONFIGURATION
# =============================================================================

# Find project root (look for .gitignore)
current_dir = Path(__file__).parent
project_root = current_dir.parent
while not (project_root / '.gitignore').exists() and project_root != project_root.parent:
    project_root = project_root.parent

sys.path.insert(0, str(project_root))

# Data path - modify this to point to your merged dataset
# Note: When using --s3 flag, this path is ignored and data loads from S3
DATA_PATH = project_root / 'data' / '03_intermediate' / 'player_props_with_actuals_2025-26.csv'

# Emoji map for status/output
EMOJI = {
    'success': '✅',
    'warning': '⚠️',
    'fire': '🔥',
    'cold': '❄️',
    'chart': '📊',
    'target': '🎯',
    'light': '💡',
    'medical': '🏥',
}

# Analysis parameters
MIN_SAMPLE_SIZE = 50  # Minimum games per tier+spread combo
MIN_PLAYER_GAMES = 3  # Minimum games to show player in drill-down
EARLY_EXIT_THRESHOLD = 0.25  # Flag games with ≤25% usual minutes

# =============================================================================
# WHAT IS "EDGE"? (Read this first!)
# =============================================================================
"""
EDGE = How much better than baseline a betting opportunity performs

Simple example:
    - Baseline: Overs hit 49% of the time (roughly 50/50)
    - You find: Bench players on 5-10 pt favorites hit overs 53.5% of time
    - Edge: 53.5% - 49% = +4.5%
    
Why edge matters:
    - At -110 odds (typical), you need 52.4% win rate to break even
    - Baseline (49%) loses money: -1.9% ROI
    - +4.5% edge (53.5% win rate) makes money: +1.1% ROI
    
Rule of thumb:
    - Edge > +3%: Strong opportunity (if sample size is good)
    - Edge > +5%: Very strong (check for data issues/small sample)
    - Edge < +2%: Marginal (need large volume to profit)
    
The goal: Find combos with large positive edge and sufficient sample size
"""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def bin_points_line(line, granularity='standard'):
    """
    Categorize player by their points line tier
    
    Args:
        line: Points line value
        granularity: 'standard' (4 tiers) or 'detailed' (7 tiers)
    """
    if pd.isna(line):
        return 'Unknown'
    
    if granularity == 'detailed':
        # Detailed granularity: 7 tiers
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
        elif line >= 5:
            return '5-10 (Bench)'
        else:
            return '0-5 (Deep Bench)'
    else:
        # Standard granularity: 4 tiers (original)
        if line >= 30:
            return '30+ (Superstar)'
        elif line >= 20:
            return '20-30 (Star)'
        elif line >= 10:
            return '10-20 (Role Player)'
        else:
            return '<10 (Bench)'


def bin_team_spread(spread, granularity='standard'):
    """
    Categorize game by team spread
    
    Args:
        spread: Team spread value (positive = favorite, negative = underdog)
        granularity: 'standard' (6 bins) or 'detailed' (9 bins)
    """
    if pd.isna(spread):
        return 'Unknown'
    
    if granularity == 'detailed':
        # Detailed granularity: 9 bins
        if spread >= 15:
            return '15+ Fav'
        elif spread >= 10:
            return '10-15 Fav'
        elif spread >= 6:
            return '6-10 Fav'
        elif spread >= 2:
            return '2-6 Fav'
        elif spread >= -2:
            return "Pick'em (-2 to +2)"
        elif spread >= -6:
            return '2-6 Dog'
        elif spread >= -10:
            return '6-10 Dog'
        elif spread >= -15:
            return '10-15 Dog'
        else:
            return '15+ Dog'
    else:
        # Standard granularity: 6 bins (original)
        if spread >= 10:
            return '10+ Fav'
        elif spread >= 5:
            return '5-10 Fav'
        elif spread >= 0:
            return '0-5 Fav'
        elif spread >= -5:
            return '0-5 Dog'
        elif spread >= -10:
            return '5-10 Dog'
        else:
            return '10+ Dog'


def categorize_scorer_type(df):
    """
    Categorize each player as rim attacker (50%+ points within 6 feet) or perimeter player
    
    This uses the pre-calculated scorer_type column if it exists (from join script),
    otherwise calculates from scratch.
    
    Args:
        df: DataFrame with player game data
    
    Returns:
        dict: {player_name: 'Rim Attacker (≥50%)' or 'Perimeter (<50%)'}
    """
    # FIRST: Check if scorer_type already exists (from join script)
    if 'scorer_type' in df.columns:
        print("✅ Using pre-calculated scorer_type from data")
        
        # Create player -> scorer_type mapping from first occurrence of each player
        player_scorer = df.groupby('PLAYER_NAME')['scorer_type'].first().to_dict()
        
        # Print distribution
        scorer_values = list(player_scorer.values())
        rim_attackers = sum(1 for v in scorer_values if 'Rim' in str(v))
        perimeter = sum(1 for v in scorer_values if 'Perimeter' in str(v))
        
        print(f"\nScorer type distribution:")
        print(f"  Rim Attackers (≥50% within 6ft): {rim_attackers} players")
        print(f"  Perimeter (<50% within 6ft): {perimeter} players")
        
        return player_scorer
    
    # SECOND: Try to calculate from pts_0_6_pct if it exists
    if 'pts_0_6_pct' in df.columns:
        print("✅ Calculating scorer_type from pts_0_6_pct column")
        
        # Get each player's pts_0_6_pct (should be same for all their games)
        player_pct = df.groupby('PLAYER_NAME')['pts_0_6_pct'].first().to_dict()
        
        # Categorize based on 50% threshold
        scorer_type_map = {}
        for player, pct in player_pct.items():
            if pd.notna(pct):
                if pct >= 50.0:
                    scorer_type_map[player] = 'Rim Attacker (≥50%)'
                else:
                    scorer_type_map[player] = 'Perimeter (<50%)'
            else:
                scorer_type_map[player] = 'Unknown'
        
        # Print distribution
        rim_attackers = sum(1 for v in scorer_type_map.values() if 'Rim' in v)
        perimeter = sum(1 for v in scorer_type_map.values() if 'Perimeter' in v)
        
        print(f"\nScorer type distribution:")
        print(f"  Rim Attackers (≥50% within 6ft): {rim_attackers} players")
        print(f"  Perimeter (<50% within 6ft): {perimeter} players")
        
        return scorer_type_map
    
    # FALLBACK: Try legacy column names (shouldn't reach here with our data)
    print("⚠️  No scorer_type or pts_0_6_pct columns found!")
    print(f"Available columns: {sorted([c for c in df.columns if 'PT' in c.upper() or 'scorer' in c.lower()])}")
    print("Defaulting all players to 'Unknown' scorer type")
    return {player: 'Unknown' for player in df['PLAYER_NAME'].unique()}


def print_section(title, emoji=None):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    prefix = f"{EMOJI[emoji]} " if emoji else ""
    print(f"{prefix}{title}")
    print("=" * 80)


def print_subsection(title):
    """Print a formatted subsection"""
    print(f"\n{title}")
    print("-" * 80)


def get_tier_order(granularity='standard'):
    """Get the proper ordering for player tiers"""
    if granularity == 'detailed':
        return [
            '30+ (Superstar)',
            '25-30 (High Star)',
            '20-25 (Star)',
            '15-20 (High Role)',
            '10-15 (Role Player)',
            '5-10 (Bench)',
            '0-5 (Deep Bench)',
            'Total'
        ]
    else:
        return [
            '30+ (Superstar)',
            '20-30 (Star)',
            '10-20 (Role Player)',
            '<10 (Bench)',
            'Total'
        ]


def get_spread_order(granularity='standard'):
    """Get the proper ordering for spread bins"""
    if granularity == 'detailed':
        return [
            '15+ Dog',
            '10-15 Dog',
            '6-10 Dog',
            '2-6 Dog',
            "Pick'em (-2 to +2)",
            '2-6 Fav',
            '6-10 Fav',
            '10-15 Fav',
            '15+ Fav',
            'Total'
        ]
    else:
        return [
            '10+ Dog',
            '5-10 Dog',
            '0-5 Dog',
            '0-5 Fav',
            '5-10 Fav',
            '10+ Fav',
            'Total'
        ]


# =============================================================================
# MAIN ANALYSIS FUNCTIONS
# =============================================================================

def load_data(input_path):
    """
    Load and prepare the merged dataset from local file or S3
    
    Args:
        input_path: Local file path or S3 URI (s3://bucket/key)
    """
    print_section("Loading Data", "chart")
    
    # Check if S3 URI
    if str(input_path).startswith('s3://'):
        # Parse S3 URI
        s3_uri = str(input_path)
        parts = s3_uri.replace('s3://', '').split('/', 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ''
        
        print(f"Loading from S3: {s3_uri}")
        
        try:
            import boto3
            s3 = boto3.client('s3')
            obj = s3.get_object(Bucket=bucket, Key=key)
            df = pd.read_csv(BytesIO(obj['Body'].read()))
            print(f"{EMOJI['success']} Loaded {len(df):,} player-game rows from S3")
        except Exception as e:
            print(f"❌ Error loading from S3: {e}")
            raise
    else:
        # Load from local file
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Data file not found: {input_path}")
        
        print(f"Loading from local file: {input_path}")
        df = pd.read_csv(input_path)
        print(f"{EMOJI['success']} Loaded {len(df):,} player-game rows from local file")
    
    print(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")
    print(f"Unique players: {df['PLAYER_NAME'].nunique():,}")
    print(f"Unique games: {df['GAME_ID'].nunique():,}")
    
    # Show rim scorer threshold if it exists in data
    if 'rim_scorer_threshold' in df.columns:
        threshold = df['rim_scorer_threshold'].iloc[0]
        print(f"Rim scorer threshold: {threshold}%")
    
    return df


def filter_actionable_data(df):
    """Filter to games with both points line and actual points"""
    print_section("Filtering to Actionable Bets")
    
    df_bets = df[(df['points_line'].notna()) & (df['PTS'].notna())].copy()
    
    print(f"Filtered to {len(df_bets):,} actionable player-games")
    print(f"  Unique players: {df_bets['PLAYER_NAME'].nunique():,}")
    print(f"  Unique games: {df_bets['GAME_ID'].nunique():,}")
    print(f"  Coverage: {len(df_bets)/len(df)*100:.1f}% of all player-games")
    
    # Add calculated columns
    df_bets['points_diff'] = df_bets['PTS'] - df_bets['points_line']
    df_bets['over_hit'] = (df_bets['points_diff'] > 0).astype(int)
    df_bets['under_hit'] = (df_bets['points_diff'] < 0).astype(int)
    df_bets['push'] = (df_bets['points_diff'] == 0).astype(int)
    
    print(f"\nOverall hit rates:")
    print(f"  Overs:  {df_bets['over_hit'].sum():,} / {len(df_bets):,} ({df_bets['over_hit'].mean()*100:.1f}%)")
    print(f"  Unders: {df_bets['under_hit'].sum():,} / {len(df_bets):,} ({df_bets['under_hit'].mean()*100:.1f}%)")
    print(f"  Pushes: {df_bets['push'].sum():,} ({df_bets['push'].mean()*100:.1f}%)")
    
    return df_bets


def remove_early_exits(df_bets):
    """Remove games where player left early (injury, ejection, DNP-CD)"""
    print_section("Removing Injury/Early Exit Games", "medical")
    
    # Calculate each player's average minutes
    player_avg_min = df_bets.groupby('PLAYER_NAME')['MIN'].mean().to_dict()
    df_bets['player_avg_min'] = df_bets['PLAYER_NAME'].map(player_avg_min)
    
    # Calculate % of usual minutes played
    df_bets['min_pct_of_avg'] = df_bets['MIN'] / df_bets['player_avg_min']
    
    # Flag early exits
    df_bets['early_exit'] = df_bets['min_pct_of_avg'] <= EARLY_EXIT_THRESHOLD
    
    print(f"Total games: {len(df_bets):,}")
    print(f"Early exit games (≤{EARLY_EXIT_THRESHOLD*100:.0f}% usual minutes): {df_bets['early_exit'].sum():,} ({df_bets['early_exit'].mean()*100:.1f}%)")
    
    # Compare hit rates
    early_exit_over_rate = df_bets[df_bets['early_exit']]['over_hit'].mean() * 100
    full_game_over_rate = df_bets[~df_bets['early_exit']]['over_hit'].mean() * 100
    
    print(f"\nOver hit rates:")
    print(f"  Early exit games: {early_exit_over_rate:.1f}%")
    print(f"  Full games:       {full_game_over_rate:.1f}%")
    print(f"  Difference:       {full_game_over_rate - early_exit_over_rate:+.1f}%")
    
    # Show examples
    print_subsection("Example early exit games (likely injuries)")
    early_exits = df_bets[df_bets['early_exit']].nlargest(10, 'points_line')
    cols = ['PLAYER_NAME', 'game_date', 'MIN', 'player_avg_min', 'PTS', 'points_line', 'points_diff']
    print(early_exits[cols].to_string(index=False))
    
    # Create clean dataset
    df_clean = df_bets[~df_bets['early_exit']].copy()
    print(f"\n{EMOJI['success']} Clean dataset: {len(df_clean):,} games ({len(df_clean)/len(df_bets)*100:.1f}% of original)")
    print(f"Overall over hit rate (clean): {df_clean['over_hit'].mean()*100:.1f}%")
    
    return df_clean


def create_matrix_analysis(df_clean, granularity='standard'):
    """Create and display 3D tier × spread × scorer_type matrix"""
    print_section(f"3D Matrix Analysis: Player Tier × Team Spread × Scorer Type ({granularity.upper()})", "chart")
    
    # Create bins
    df_clean['line_tier'] = df_clean['points_line'].apply(lambda x: bin_points_line(x, granularity))
    df_clean['spread_bin'] = df_clean['team_spread'].apply(lambda x: bin_team_spread(x, granularity))
    
    # Add scorer type classification
    print_subsection("Classifying Scorer Types (50%+ within 6 feet)")
    scorer_type_map = categorize_scorer_type(df_clean)
    df_clean['scorer_type'] = df_clean['PLAYER_NAME'].map(scorer_type_map)
    
    # Filter out any None/NaN scorer types
    before_count = len(df_clean)
    df_clean = df_clean[df_clean['scorer_type'].notna()].copy()
    after_count = len(df_clean)
    if before_count != after_count:
        print(f"⚠️  Filtered out {before_count - after_count} games with missing scorer_type")
    
    # Get proper ordering
    col_order = get_spread_order(granularity)
    row_order = get_tier_order(granularity)
    scorer_types = sorted(df_clean['scorer_type'].unique())
    
    # Show distribution by scorer type
    print(f"\n{EMOJI['chart']} Overall distribution by scorer type:")
    for scorer_type in scorer_types:
        st_data = df_clean[df_clean['scorer_type'] == scorer_type]
        over_rate = st_data['over_hit'].mean() * 100
        print(f"  {scorer_type}: {len(st_data):,} games ({len(st_data)/len(df_clean)*100:.1f}%), {over_rate:.1f}% overs")
    
    # Create separate 2D matrices for each scorer type
    for scorer_type in scorer_types:
        st_data = df_clean[df_clean['scorer_type'] == scorer_type]
        
        if len(st_data) < 50:
            print(f"\n{EMOJI['warning']} Skipping {scorer_type} - insufficient data ({len(st_data)} games)")
            continue
        
        print_section(f"{scorer_type} - TIER × SPREAD MATRIX", "target")
        
        # 1. Sample size matrix
        print_subsection("1️⃣ SAMPLE SIZE MATRIX (number of games)")
        count_matrix = pd.crosstab(
            st_data['line_tier'],
            st_data['spread_bin'],
            margins=True,
            margins_name='Total'
        )
        count_matrix = count_matrix.reindex(columns=[c for c in col_order if c in count_matrix.columns])
        count_matrix = count_matrix.reindex([r for r in row_order if r in count_matrix.index])
        print(count_matrix.to_string())
        
        # 2. Over hit rate matrix
        print_subsection("\n2️⃣ OVER HIT RATE MATRIX (% of overs that hit)")
        over_matrix = pd.crosstab(
            st_data['line_tier'],
            st_data['spread_bin'],
            values=st_data['over_hit'],
            aggfunc='mean'
        ) * 100
        over_matrix = over_matrix.reindex(columns=[c for c in col_order[:-1] if c in over_matrix.columns])
        over_matrix = over_matrix.reindex([r for r in row_order[:-1] if r in over_matrix.index])
        print(over_matrix.round(1).to_string())
        
        # 3. Benchmarks for this scorer type
        print_subsection(f"\n3️⃣ BENCHMARKS - {scorer_type}")
        print(f"Overall over hit rate: {st_data['over_hit'].mean()*100:.1f}%")
        
        print("\nBy Player Tier:")
        for tier in row_order[:-1]:
            tier_data = st_data[st_data['line_tier'] == tier]
            if len(tier_data) > 0:
                print(f"  {tier:20s}: {tier_data['over_hit'].mean()*100:.1f}% ({len(tier_data):,} games)")
        
        print("\nBy Team Spread:")
        for spread in col_order[:-1]:
            spread_data = st_data[st_data['spread_bin'] == spread]
            if len(spread_data) > 0:
                print(f"  {spread:15s}: {spread_data['over_hit'].mean()*100:.1f}% ({len(spread_data):,} games)")
    
    # Comparison: Rim Attackers vs Perimeter Players
    print_section("🆚 RIM ATTACKERS vs PERIMETER PLAYERS COMPARISON", "fire")
    
    rim_data = df_clean[df_clean['scorer_type'].str.contains('Rim', na=False)]
    perim_data = df_clean[df_clean['scorer_type'].str.contains('Perimeter', na=False)]
    
    if len(rim_data) > 0 and len(perim_data) > 0:
        print(f"\nOverall hit rates:")
        print(f"  Rim Attackers: {rim_data['over_hit'].mean()*100:.1f}% overs ({len(rim_data):,} games)")
        print(f"  Perimeter:     {perim_data['over_hit'].mean()*100:.1f}% overs ({len(perim_data):,} games)")
        print(f"  Difference:    {(rim_data['over_hit'].mean() - perim_data['over_hit'].mean())*100:+.1f}%")
        
        # Compare by player tier
        print(f"\n{EMOJI['chart']} By Player Tier:")
        for tier in row_order[:-1]:
            rim_tier = rim_data[rim_data['line_tier'] == tier]
            perim_tier = perim_data[perim_data['line_tier'] == tier]
            
            if len(rim_tier) >= 20 and len(perim_tier) >= 20:
                rim_over = rim_tier['over_hit'].mean() * 100
                perim_over = perim_tier['over_hit'].mean() * 100
                diff = rim_over - perim_over
                
                print(f"  {tier:20s}: Rim {rim_over:5.1f}% vs Perim {perim_over:5.1f}% | Diff: {diff:+.1f}%")
        
        # Compare by spread
        print(f"\n{EMOJI['chart']} By Team Spread:")
        for spread in col_order[:-1]:
            rim_spread = rim_data[rim_data['spread_bin'] == spread]
            perim_spread = perim_data[perim_data['spread_bin'] == spread]
            
            if len(rim_spread) >= 20 and len(perim_spread) >= 20:
                rim_over = rim_spread['over_hit'].mean() * 100
                perim_over = perim_spread['over_hit'].mean() * 100
                diff = rim_over - perim_over
                
                print(f"  {spread:20s}: Rim {rim_over:5.1f}% vs Perim {perim_over:5.1f}% | Diff: {diff:+.1f}%")
    
    return df_clean


def find_edges(df_clean, baseline_over_rate, min_sample_size=50):
    """
    Identify betting edges from the 3D matrix (tier × spread × scorer_type)
    
    Edge explanation:
        Edge = how much better than baseline a combo performs
        
        Example: If baseline over rate is 49%, and a combo hits 53.5% overs:
            → Over edge = +4.5% (you win 4.5% more than expected)
            → To profit at -110 odds, you need ~52.4% hit rate
            → This combo has +1.1% edge over breakeven!
        
        Positive edge = opportunity to bet
        Larger edge = stronger signal
        
    Now includes scorer_type as 3rd dimension!
    """
    print_section("3D Edge Detection (Tier × Spread × Scorer Type)", "target")
    
    # Calculate baseline under rate (complement of over rate, excluding pushes)
    baseline_under_rate = 100 - baseline_over_rate
    
    print(f"Baseline rates (excluding pushes):")
    print(f"  Over:  {baseline_over_rate:.1f}%")
    print(f"  Under: {baseline_under_rate:.1f}%")
    print(f"\nNote: To profit at -110 odds, you need ~52.4% win rate")
    print(f"Edge = (Hit Rate - Baseline). Positive edge = betting opportunity\n")
    
    results = []
    for tier in df_clean['line_tier'].unique():
        for spread in df_clean['spread_bin'].unique():
            for scorer_type in df_clean['scorer_type'].unique():
                subset = df_clean[
                    (df_clean['line_tier'] == tier) & 
                    (df_clean['spread_bin'] == spread) &
                    (df_clean['scorer_type'] == scorer_type)
                ]
                
                if len(subset) >= min_sample_size:
                    # Calculate rates (as % of non-push outcomes)
                    total_non_push = subset['over_hit'].sum() + subset['under_hit'].sum()
                    over_rate = (subset['over_hit'].sum() / total_non_push) * 100 if total_non_push > 0 else 0
                    under_rate = (subset['under_hit'].sum() / total_non_push) * 100 if total_non_push > 0 else 0
                    push_rate = (subset['push'].sum() / len(subset)) * 100
                    
                    # Calculate strategy ROI at -110 odds
                    # At -110: to win $100 you risk $110, so profit per $1 wagered = 100/110
                    # ROI = (win_rate × profit_per_dollar) - loss_rate
                    over_roi = ((over_rate/100) * (100/110) - (under_rate/100)) * 100
                    under_roi = ((under_rate/100) * (100/110) - (over_rate/100)) * 100
                    
                    results.append({
                        'line_tier': tier,
                        'spread_bin': spread,
                        'scorer_type': scorer_type,
                        'games': len(subset),
                        'over_rate': over_rate,
                        'under_rate': under_rate,
                        'push_rate': push_rate,
                        'over_roi': over_roi,
                        'under_roi': under_roi,
                        'rate_sum': over_rate + under_rate,  # Should be 100%
                        'over_edge': over_rate - baseline_over_rate,
                        'under_edge': under_rate - baseline_under_rate,
                        'avg_line': subset['points_line'].mean(),
                        'avg_pts': subset['PTS'].mean(),
                        'avg_diff': subset['points_diff'].mean()
                    })
    
    df_edges = pd.DataFrame(results)
    
    # Verify rates sum to 100%
    print_subsection("Verification: Over + Under rates should = 100%")
    print(f"All combos sum correctly: {(df_edges['rate_sum'].round(1) == 100.0).all()}")
    print(f"Average rate_sum: {df_edges['rate_sum'].mean():.2f}%")
    print(f"Push rate range: {df_edges['push_rate'].min():.1f}% - {df_edges['push_rate'].max():.1f}%")
    
    # Top over opportunities (now with scorer type!)
    print_subsection(f"\n{EMOJI['fire']} TOP 10 OVER OPPORTUNITIES (highest edge vs baseline)")
    top_overs = df_edges.nlargest(10, 'over_edge')
    cols = ['line_tier', 'spread_bin', 'scorer_type', 'games', 'over_rate', 'under_rate', 'over_edge', 'push_rate', 'over_roi']
    print(top_overs[cols].to_string(index=False))
    
    # Top under opportunities (now with scorer type!)
    print_subsection(f"\n{EMOJI['cold']} TOP 10 UNDER OPPORTUNITIES (highest edge vs baseline)")
    top_unders = df_edges.nlargest(10, 'under_edge')
    cols = ['line_tier', 'spread_bin', 'scorer_type', 'games', 'over_rate', 'under_rate', 'under_edge', 'push_rate', 'under_roi']
    print(top_unders[cols].to_string(index=False))
    
    # Key insights
    print_subsection(f"\n{EMOJI['light']} KEY INSIGHTS")
    
    if len(df_edges) > 0:
        best_over = df_edges.loc[df_edges['over_edge'].idxmax()]
        print(f"Strongest OVER edge: {best_over['line_tier']} + {best_over['spread_bin']} + {best_over['scorer_type']}")
        print(f"  {EMOJI['target']} {best_over['over_rate']:.1f}% over / {best_over['under_rate']:.1f}% under (sum={best_over['rate_sum']:.1f}%)")
        print(f"  {EMOJI['target']} +{best_over['over_edge']:.1f}% edge | {int(best_over['games'])} games | {best_over['push_rate']:.1f}% pushes")
        print(f"  {EMOJI['target']} Strategy ROI at -110 odds: {best_over['over_roi']:+.1f}%")
        
        best_under = df_edges.loc[df_edges['under_edge'].idxmax()]
        print(f"\nStrongest UNDER edge: {best_under['line_tier']} + {best_under['spread_bin']} + {best_under['scorer_type']}")
        print(f"  {EMOJI['target']} {best_under['under_rate']:.1f}% under / {best_under['over_rate']:.1f}% over (sum={best_under['rate_sum']:.1f}%)")
        print(f"  {EMOJI['target']} +{best_under['under_edge']:.1f}% edge | {int(best_under['games'])} games | {best_under['push_rate']:.1f}% pushes")
        print(f"  {EMOJI['target']} Strategy ROI at -110 odds: {best_under['under_roi']:+.1f}%")
        
        weak_samples = df_edges[df_edges['games'] < 100]
        if len(weak_samples) > 0:
            print(f"\n{EMOJI['warning']} {len(weak_samples)} combinations have {min_sample_size}-100 games (use caution)")
        
        # Scorer type analysis
        print(f"\n{EMOJI['chart']} SCORER TYPE BREAKDOWN:")
        rim_edges = df_edges[df_edges['scorer_type'].str.contains('Rim', na=False)]
        perim_edges = df_edges[df_edges['scorer_type'].str.contains('Perimeter', na=False)]
        
        print(f"  Rim Attackers: {len(rim_edges)} combinations with {min_sample_size}+ games")
        if len(rim_edges) > 0:
            print(f"    Best ROI: {rim_edges['over_roi'].max():+.1f}% (OVER), {rim_edges['under_roi'].max():+.1f}% (UNDER)")
        
        print(f"  Perimeter: {len(perim_edges)} combinations with {min_sample_size}+ games")
        if len(perim_edges) > 0:
            print(f"    Best ROI: {perim_edges['over_roi'].max():+.1f}% (OVER), {perim_edges['under_roi'].max():+.1f}% (UNDER)")
    
    return df_edges


def drill_down_edge(df_clean, tier, spread_bin, scorer_type, top_n=15, bet_side='over'):
    """
    Show which players drive a specific edge (3D: tier + spread + scorer_type)
    
    Args:
        tier: Player tier (e.g., '20-30 (Star)')
        spread_bin: Team spread (e.g., '5-10 Fav')
        scorer_type: Scorer type (e.g., 'Rim Attacker (≥50%)')
        top_n: Number of top players to show
        bet_side: 'over' or 'under' - determines which ROI to calculate
    """
    subset = df_clean[
        (df_clean['line_tier'] == tier) & 
        (df_clean['spread_bin'] == spread_bin) &
        (df_clean['scorer_type'] == scorer_type)
    ]
    
    if len(subset) == 0:
        print(f"No data for {tier} + {spread_bin} + {scorer_type}")
        return
    
    player_stats = subset.groupby('PLAYER_NAME').agg({
        'over_hit': ['sum', 'count', 'mean'],
        'under_hit': ['sum'],
        'PTS': 'mean',
        'points_line': 'mean',
        'points_diff': 'mean',
        'MIN': 'mean'
    }).reset_index()
    
    player_stats.columns = ['PLAYER_NAME', 'overs_hit', 'games', 'over_rate', 'unders_hit', 'avg_pts', 'avg_line', 'avg_diff', 'avg_min']
    player_stats['under_rate'] = player_stats['unders_hit'] / player_stats['games']
    player_stats = player_stats[player_stats['games'] >= MIN_PLAYER_GAMES]
    
    # Calculate ROI at -110 odds based on which side we're betting
    # At -110: to win $100 you risk $110, so profit per $1 wagered = 100/110
    # ROI = (win_rate × profit_per_dollar) - loss_rate
    if bet_side == 'under':
        player_stats['roi'] = ((player_stats['under_rate'] * (100/110)) - (player_stats['over_rate'] * 1.0)) * 100
    else:
        player_stats['roi'] = ((player_stats['over_rate'] * (100/110)) - (player_stats['under_rate'] * 1.0)) * 100
    
    # Binary profitable flag
    player_stats['profitable'] = player_stats['roi'] > 0
    
    player_stats = player_stats.sort_values('games', ascending=False).head(top_n)
    
    # Round numeric columns for cleaner output
    player_stats['games'] = player_stats['games'].astype(int)  # No decimals for game count
    player_stats['over_rate'] = player_stats['over_rate'].round(3)  # Keep 3 decimals for rates (easier to read as 0.462 vs 46.2%)
    player_stats['under_rate'] = player_stats['under_rate'].round(3)
    player_stats['avg_pts'] = player_stats['avg_pts'].round(1)
    player_stats['avg_line'] = player_stats['avg_line'].round(1)
    player_stats['avg_diff'] = player_stats['avg_diff'].round(1)
    player_stats['roi'] = player_stats['roi'].round(1)
    
    overall_rate = subset[f'{bet_side}_hit'].mean() * 100
    print(f"\n{tier} + {spread_bin} + {scorer_type} | Total: {len(subset)} games, {overall_rate:.1f}% {bet_side}s")
    print("-" * 80)
    cols = ['PLAYER_NAME', 'games', 'over_rate', 'under_rate', 'avg_pts', 'avg_line', 'avg_diff', 'profitable', 'roi']
    print(player_stats[cols].to_string(index=False))


def show_top_edge_players(df_clean, df_edges):
    """Show player breakdown for best edges (including scorer type)"""
    print_section("Player Breakdown for Top Edges (3D: Tier × Spread × Scorer Type)")
    
    if len(df_edges) == 0:
        print("No edges found with sufficient sample size")
        return
    
    best_over = df_edges.loc[df_edges['over_edge'].idxmax()]
    print(f"\n{EMOJI['fire']} BEST OVER EDGE BREAKDOWN:")
    drill_down_edge(df_clean, best_over['line_tier'], best_over['spread_bin'], best_over['scorer_type'], bet_side='over')
    
    best_under = df_edges.loc[df_edges['under_edge'].idxmax()]
    print(f"\n\n{EMOJI['cold']} BEST UNDER EDGE BREAKDOWN:")
    drill_down_edge(df_clean, best_under['line_tier'], best_under['spread_bin'], best_under['scorer_type'], bet_side='under')


def compare_strategies(old_strategies, new_strategies, min_roi=5.0):
    """
    Compare old and new strategies to identify changes based on ROI threshold
    
    Reports:
    1. Still active: ROI >= threshold in BOTH old and new
    2. New strategies: ROI >= threshold in new but NOT in old
    3. Deactivated: ROI >= threshold in old but NOT in new (ROI dropped below threshold)
    
    Args:
        min_roi: Minimum ROI threshold for "active" strategies
    
    Note: Now includes scorer_type as 4th dimension in key
    """
    # Create lookup keys: (line_tier, spread_bin, scorer_type, bet_side)
    def get_key(s):
        # Handle both old 3-part keys and new 4-part keys (with scorer_type)
        scorer_type = s.get('scorer_type', 'All')  # Default to 'All' for old strategies
        return (s['line_tier'], s['spread_bin'], scorer_type, s['bet_side'])
    
    # Build dicts for ALL strategies (for comparison)
    old_dict = {get_key(s): s for s in old_strategies}
    new_dict = {get_key(s): s for s in new_strategies}
    
    # Filter to "active" strategies (ROI >= threshold)
    old_active = {k: v for k, v in old_dict.items() if v['roi'] >= min_roi}
    new_active = {k: v for k, v in new_dict.items() if v['roi'] >= min_roi}
    
    old_active_keys = set(old_active.keys())
    new_active_keys = set(new_active.keys())
    
    # Categorize strategies
    still_active_keys = old_active_keys & new_active_keys
    new_keys_only = new_active_keys - old_active_keys
    deactivated_keys = old_active_keys - new_active_keys
    
    # 1. Still active strategies
    print(f"\n{EMOJI['success']} STILL ACTIVE STRATEGIES (ROI >= {min_roi}% in both): {len(still_active_keys)}")
    print("-" * 80)
    
    if len(still_active_keys) > 0:
        print(f"{'Strategy':<50} {'Old ROI':>8} {'New ROI':>8} {'Change':>8} {'Old Games':>10} {'New Games':>10}")
        print("-" * 80)
        
        # Sort by abs(change in ROI) descending
        active_comparison = []
        for key in still_active_keys:
            old = old_active[key]
            new = new_active[key]
            change = new['roi'] - old['roi']
            active_comparison.append({
                'key': key,
                'name': f"{key[0]} + {key[1]} + {key[2]} {key[3]}",  # tier + spread + scorer_type + bet_side
                'old_roi': old['roi'],
                'new_roi': new['roi'],
                'change': change,
                'old_games': old['games'],
                'new_games': new['games']
            })
        
        active_comparison.sort(key=lambda x: abs(x['change']), reverse=True)
        
        # Show top 15 biggest changes
        for item in active_comparison[:15]:
            name = item['name'][:48]
            change_emoji = EMOJI['fire'] if item['change'] > 0 else EMOJI['cold']
            print(f"{name:<50} {item['old_roi']:>7.1f}% {item['new_roi']:>7.1f}% {change_emoji} {item['change']:>6.1f}% {item['old_games']:>10} {item['new_games']:>10}")
        
        if len(active_comparison) > 15:
            print(f"\n   ... and {len(active_comparison) - 15} more stable strategies")
    
    # 2. New strategies
    print(f"\n\n{EMOJI['fire']} NEW STRATEGIES (ROI >= {min_roi}% in new, <{min_roi}% or didn't exist in old): {len(new_keys_only)}")
    print("-" * 80)
    
    if len(new_keys_only) > 0:
        new_list_data = []
        for key in new_keys_only:
            new_strat = new_active[key]
            old_strat = old_dict.get(key)  # May not exist
            
            new_list_data.append({
                'strat': new_strat,
                'old_roi': old_strat['roi'] if old_strat else None,
                'old_games': old_strat['games'] if old_strat else 0
            })
        
        new_list_data.sort(key=lambda x: x['strat']['roi'], reverse=True)
        
        print(f"{'Strategy':<50} {'New ROI':>8} {'Old ROI':>8} {'Hit Rate':>10} {'Games':>8}")
        print("-" * 80)
        
        for item in new_list_data[:15]:
            strat = item['strat']
            scorer = strat.get('scorer_type', 'All')
            name = f"{strat['line_tier']} + {strat['spread_bin']} + {scorer} {strat['bet_side']}"[:48]
            old_roi_str = f"{item['old_roi']:>7.1f}%" if item['old_roi'] is not None else "    N/A"
            print(f"{name:<50} {strat['roi']:>7.1f}% {old_roi_str} {strat['hit_rate']:>9.1f}% {strat['games']:>8}")
        
        if len(new_list_data) > 15:
            print(f"\n   ... and {len(new_list_data) - 15} more new strategies")
    else:
        print("   (None)")
    
    # 3. Deactivated strategies
    print(f"\n\n❌ DEACTIVATED STRATEGIES (ROI >= {min_roi}% in old, <{min_roi}% in new): {len(deactivated_keys)}")
    print("-" * 80)
    
    if len(deactivated_keys) > 0:
        deactivated_list_data = []
        for key in deactivated_keys:
            old_strat = old_active[key]
            new_strat = new_dict.get(key)  # Should exist but may have dropped below threshold
            
            deactivated_list_data.append({
                'strat': old_strat,
                'new_roi': new_strat['roi'] if new_strat else None,
                'new_games': new_strat['games'] if new_strat else 0
            })
        
        deactivated_list_data.sort(key=lambda x: x['strat']['roi'], reverse=True)
        
        print(f"{'Strategy':<50} {'Old ROI':>8} {'New ROI':>8} {'ROI Drop':>9} {'Old Games':>10} {'New Games':>10}")
        print("-" * 80)
        
        for item in deactivated_list_data[:15]:
            strat = item['strat']
            scorer = strat.get('scorer_type', 'All')
            name = f"{strat['line_tier']} + {strat['spread_bin']} + {scorer} {strat['bet_side']}"[:48]
            new_roi_str = f"{item['new_roi']:>7.1f}%" if item['new_roi'] is not None else "    N/A"
            drop = (item['new_roi'] - strat['roi']) if item['new_roi'] is not None else 0
            print(f"{name:<50} {strat['roi']:>7.1f}% {new_roi_str} {drop:>8.1f}% {strat['games']:>10} {item['new_games']:>10}")
        
        if len(deactivated_list_data) > 15:
            print(f"\n   ... and {len(deactivated_list_data) - 15} more deactivated strategies")
        
        print(f"\n{EMOJI['warning']} Note: Deactivated means ROI dropped below {min_roi}% threshold")
    else:
        print("   (None)")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY (ROI Threshold = {min_roi}%):")
    print(f"  Old total strategies: {len(old_strategies)} (all), {len(old_active_keys)} active (>={min_roi}%)")
    print(f"  New total strategies: {len(new_strategies)} (all), {len(new_active_keys)} active (>={min_roi}%)")
    print(f"  Still active: {len(still_active_keys)} ({len(still_active_keys)/len(old_active_keys)*100 if len(old_active_keys) > 0 else 0:.1f}% of old active)")
    print(f"  New: {len(new_keys_only)}")
    print(f"  Deactivated: {len(deactivated_keys)}")
    print(f"{'='*80}")


def output_strategies_json(df_edges, output_path, df_original, granularity, min_roi=5.0, min_sample_size=50):
    """
    Output ALL strategies to JSON (no filtering)
    
    This creates a JSON file with all strategies that can be filtered at runtime
    by the daily plays script based on ROI threshold.
    
    Before overwriting, checks if file exists and:
    1. Creates timestamped backup
    2. Compares old vs new strategies
    3. Reports: still active, new, and deactivated strategies
    
    Args:
        output_path: Local file path or S3 URI (s3://bucket/key)
    """
    print_section("Exporting Strategies to JSON")
    
    # Check if output file already exists and load it for comparison
    old_strategies_data = None
    is_s3 = str(output_path).startswith('s3://')
    
    if is_s3:
        # Check S3
        s3_uri = str(output_path)
        parts = s3_uri.replace('s3://', '').split('/', 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ''
        
        try:
            import boto3
            s3 = boto3.client('s3')
            
            # Try to get existing file
            try:
                obj = s3.get_object(Bucket=bucket, Key=key)
                old_strategies_data = json.loads(obj['Body'].read().decode('utf-8'))
                print(f"\n📋 Found existing strategies file in S3")
                print(f"   Old file generated: {old_strategies_data.get('generated_at', 'Unknown')}")
                print(f"   Old data through: {old_strategies_data.get('data_through', 'Unknown')}")
                print(f"   Old strategies: {len(old_strategies_data.get('strategies', []))}")
                
                # Create timestamped backup
                old_date = old_strategies_data.get('data_through', datetime.now().strftime('%Y-%m-%d')).replace('-', '')
                backup_key = key.replace('.json', f'_{old_date}.json')
                
                print(f"\n💾 Creating backup: s3://{bucket}/{backup_key}")
                s3.copy_object(
                    Bucket=bucket,
                    CopySource={'Bucket': bucket, 'Key': key},
                    Key=backup_key
                )
                print(f"{EMOJI['success']} Backup created")
                
            except s3.exceptions.NoSuchKey:
                print(f"\n📝 No existing strategies file found (first run)")
                
        except Exception as e:
            print(f"⚠️  Error checking S3: {e}")
    else:
        # Check local file
        output_file = Path(output_path)
        if output_file.exists():
            try:
                with open(output_file, 'r') as f:
                    old_strategies_data = json.load(f)
                
                print(f"\n📋 Found existing strategies file locally")
                print(f"   Old file generated: {old_strategies_data.get('generated_at', 'Unknown')}")
                print(f"   Old data through: {old_strategies_data.get('data_through', 'Unknown')}")
                print(f"   Old strategies: {len(old_strategies_data.get('strategies', []))}")
                
                # Create timestamped backup
                old_date = old_strategies_data.get('data_through', datetime.now().strftime('%Y-%m-%d')).replace('-', '')
                backup_path = output_file.parent / f"{output_file.stem}_{old_date}.json"
                
                print(f"\n💾 Creating backup: {backup_path}")
                import shutil
                shutil.copy2(output_file, backup_path)
                print(f"{EMOJI['success']} Backup created")
                
            except Exception as e:
                print(f"⚠️  Error loading old file: {e}")
        else:
            print(f"\n📝 No existing strategies file found (first run)")
    
    # Create list of all strategies (both over and under for each combo)
    strategies = []
    
    for _, row in df_edges.iterrows():
        # OVER strategy
        strategies.append({
            'line_tier': row['line_tier'],
            'spread_bin': row['spread_bin'],
            'scorer_type': row['scorer_type'],
            'bet_side': 'OVER',
            'games': int(row['games']),
            'hit_rate': round(float(row['over_rate']), 1),
            'roi': round(float(row['over_roi']), 1),
            'edge': round(float(row['over_edge']), 1),
            'push_rate': round(float(row['push_rate']), 1),
        })
        
        # UNDER strategy
        strategies.append({
            'line_tier': row['line_tier'],
            'spread_bin': row['spread_bin'],
            'scorer_type': row['scorer_type'],
            'bet_side': 'UNDER',
            'games': int(row['games']),
            'hit_rate': round(float(row['under_rate']), 1),
            'roi': round(float(row['under_roi']), 1),
            'edge': round(float(row['under_edge']), 1),
            'push_rate': round(float(row['push_rate']), 1),
        })
    
    # Sort by ROI descending
    strategies.sort(key=lambda x: x['roi'], reverse=True)
    
    # Create output structure
    output_data = {
        'generated_at': datetime.now().isoformat(),
        'data_through': df_original['game_date'].max(),
        'total_games_analyzed': int(len(df_original)),
        'granularity': granularity,
        'min_sample_size': min_sample_size,
        'min_roi_threshold': min_roi,
        'total_strategies': len(strategies),
        'strategies': strategies
    }
    
    # Compare with old strategies if they exist
    if old_strategies_data:
        print_section("Strategy Comparison", "target")
        
        # Check if dimensions match (2D vs 3D analysis)
        old_strats = old_strategies_data.get('strategies', [])
        if old_strats:
            # Check if old strategies have scorer_type dimension
            sample_old = old_strats[0]
            sample_new = strategies[0] if strategies else None
            
            old_has_scorer = 'scorer_type' in sample_old and sample_old['scorer_type'] not in ['Unknown', 'All', None]
            new_has_scorer = sample_new and 'scorer_type' in sample_new and sample_new['scorer_type'] not in ['Unknown', 'All', None]
            
            if old_has_scorer != new_has_scorer:
                dimension = "3D (with scorer_type)" if new_has_scorer else "2D (without scorer_type)"
                old_dimension = "3D (with scorer_type)" if old_has_scorer else "2D (without scorer_type)"
                print(f"\n{EMOJI['warning']} Skipping comparison: Dimension mismatch")
                print(f"   Old file: {old_dimension}")
                print(f"   New file: {dimension}")
                print(f"   Tip: These analyses use different dimensions and cannot be compared.")
            else:
                compare_strategies(old_strats, strategies, min_roi=min_roi)
        else:
            compare_strategies(old_strats, strategies, min_roi=min_roi)
    
    # Show top strategies by ROI
    print(f"\n📊 Top 10 strategies by ROI (3D: Tier × Spread × Scorer Type):")
    for i, strat in enumerate(strategies[:10], 1):
        print(f"   {i}. {strat['line_tier']} + {strat['spread_bin']} + {strat['scorer_type']} {strat['bet_side']}: "
              f"{strat['roi']:+.1f}% ROI ({strat['hit_rate']:.1f}% hit, {strat['games']} games)")
    
    # Check if S3 URI
    if str(output_path).startswith('s3://'):
        # Parse S3 URI
        s3_uri = str(output_path)
        parts = s3_uri.replace('s3://', '').split('/', 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ''
        
        # Save to temp file first
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            json.dump(output_data, tmp, indent=2)
            tmp_path = tmp.name
        
        try:
            import boto3
            s3 = boto3.client('s3')
            
            print(f"\n📤 Uploading to S3: {s3_uri}")
            s3.upload_file(tmp_path, bucket, key)
            print(f"{EMOJI['success']} Uploaded {len(strategies)} strategies to S3")
            print(f"   Generated: {output_data['generated_at']}")
            print(f"   Data through: {output_data['data_through']}")
            print(f"   Total strategies: {len(strategies)} ({len(strategies)//2} combos × 2 sides)")
            
            # Clean up temp file
            Path(tmp_path).unlink()
            
        except Exception as e:
            print(f"❌ S3 upload failed: {e}")
            Path(tmp_path).unlink()
            raise
    else:
        # Save to local file
        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"{EMOJI['success']} Saved {len(strategies)} strategies to {output_path}")
        print(f"   Generated: {output_data['generated_at']}")
        print(f"   Data through: {output_data['data_through']}")
        print(f"   Total strategies: {len(strategies)} ({len(strategies)//2} combos × 2 sides)")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='NBA Player Props Matrix Analysis - Find betting edges by analyzing player tier × team spread combinations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick: Use season flag (auto-generates S3 paths)
  python analysis/analyze_points_props_role_spread_6feet_scorer_model.py --season 2025-26 --granularity detailed
  
  # Explicit: Specify exact input/output paths (local or S3)
  python analysis/analyze_points_props_role_spread_6feet_scorer_model.py \
    --input s3://nba-betting-mt/data/03_intermediate/player_props_with_actuals_2025-26.csv \
    --output s3://nba-betting-mt/data/03_intermediate/6feet_scorer_strategies.json \
    --granularity detailed
  
  # Local files
  python analysis/analyze_points_props_role_spread_6feet_scorer_model.py \
    --input data/03_intermediate/player_props_with_actuals_2025-26.csv \
    --output strategies_6feet.json \
    --granularity standard
  
Granularity levels:
  standard:  <10, 10-20, 20-30, 30+ pts  ×  10+ Dog, 5-10 Dog, 0-5 Dog, 0-5 Fav, 5-10 Fav, 10+ Fav  ×  2 scorer types
  detailed:  0-5, 5-10, 10-15, 15-20, 20-25, 25-30, 30+ pts  ×  15+ Dog, 10-15 Dog, ..., Pick'em, ..., 15+ Fav  ×  2 scorer types
        """
    )
    
    parser.add_argument(
        '--season',
        type=str,
        default=None,
        help='NBA season (e.g., 2025-26). Auto-generates S3 input/output paths.'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default=None,
        help='Input path: local file or S3 URI (s3://bucket/key). Overrides --season.'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path: local file or S3 URI (s3://bucket/key). Overrides --season.'
    )
    
    parser.add_argument(
        '--granularity',
        type=str,
        choices=['standard', 'detailed'],
        default='detailed',
        help='Level of granularity for bins (default: detailed)'
    )
    
    parser.add_argument(
        '--min-roi',
        type=float,
        default=5.0,
        help='Minimum ROI threshold for active strategies (default: 5.0%%)'
    )
    
    parser.add_argument(
        '--min-sample-size',
        type=int,
        default=50,
        help='Minimum games per tier+spread+scorer combo to include (default: 50)'
    )
    
    return parser.parse_args()


def main():
    """Run the full analysis pipeline"""
    args = parse_args()
    
    # Determine input/output paths
    if args.season and not args.input:
        # Auto-generate input from season only if no explicit input provided
        args.input = f's3://nba-betting-mt/data/03_intermediate/player_props_with_actuals_{args.season}.csv'
        # Also auto-generate output in this case
        if not args.output:
            args.output = f's3://nba-betting-mt/data/03_intermediate/points_by_role_gamespread_6feet_strategies_{args.season}.json'
    elif not args.input:
        # Default to local file if no season and no input specified
        args.input = DATA_PATH
    
    # If output is still None, that's okay - analysis will run but not save
    
    print_section("NBA Player Props 3D Matrix Analysis (With Scorer Type)", "chart")
    print(f"Dimensions: Player Tier × Team Spread × Scorer Type (50%+ within 6ft)")
    print(f"Input: {args.input}")
    print(f"Output: {args.output if args.output else 'None (analysis only, no save)'}")
    print(f"Granularity: {args.granularity.upper()}")
    print(f"Min sample size: {args.min_sample_size} games")
    print(f"Early exit threshold: {EARLY_EXIT_THRESHOLD*100:.0f}% usual minutes")
    
    # Load data
    df = load_data(args.input)
    
    # Filter to actionable bets
    df_bets = filter_actionable_data(df)
    baseline_over_rate = df_bets['over_hit'].mean() * 100
    
    # Remove injury games
    df_clean = remove_early_exits(df_bets)
    
    # Create matrix analysis
    df_clean = create_matrix_analysis(df_clean, args.granularity)
    
    # Find edges
    df_edges = find_edges(df_clean, baseline_over_rate, min_sample_size=args.min_sample_size)
    
    # Show player breakdowns
    show_top_edge_players(df_clean, df_edges)
    
    # Summary
    print_section("Analysis Complete", "success")
    print(f"Granularity: {args.granularity}")
    print(f"Clean games analyzed: {len(df_clean):,}")
    print(f"Tier × Spread × Scorer Type combinations with {args.min_sample_size}+ games: {len(df_edges)}")
    
    # Calculate total possible 3D combinations
    num_scorer_types = df_clean['scorer_type'].nunique()
    
    if args.granularity == 'detailed':
        total_combos = 7 * 9 * num_scorer_types
        print(f"Total possible combinations: {total_combos} (detailed: 7 tiers × 9 spreads × {num_scorer_types} scorer types)")
    else:
        total_combos = 4 * 6 * num_scorer_types
        print(f"Total possible combinations: {total_combos} (standard: 4 tiers × 6 spreads × {num_scorer_types} scorer types)")
    
    coverage_pct = (len(df_edges) / total_combos) * 100
    print(f"Coverage: {coverage_pct:.1f}% of 3D combinations have {args.min_sample_size}+ games")
    
    # Output strategies to JSON (NO filtering - output ALL strategies)
    if args.output:
        output_strategies_json(df_edges, args.output, df, args.granularity, min_roi=args.min_roi, min_sample_size=args.min_sample_size)
    
    print(f"\nKey variables available in memory:")
    print(f"  • df_clean: Clean dataset ({len(df_clean):,} rows)")
    print(f"  • df_edges: Edge opportunities ({len(df_edges)} combinations)")
    
    return df_clean, df_edges


if __name__ == '__main__':
    df_clean, df_edges = main()

