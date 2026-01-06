"""
NBA Player Props Matrix Analysis - 2025-26 Season

Purpose:
    Analyze player prop betting edges by creating a matrix of:
    - Player tier (by points line: 30+, 20-30, 10-20, <10)
    - Team spread context (10+ Dog, 5-10 Dog, etc.)
    
    Goal: Find systematic pricing inefficiencies in the prop betting market
    by identifying which player types in which game contexts consistently
    beat or miss their lines.

Key Features:
    1. Filters out injury/early exit games (≤25% usual minutes)
    2. Creates tier × spread matrix with over/under hit rates
    3. Identifies best betting edges with sufficient sample size
    4. Shows which specific players drive each edge

Usage:
    # Default (coarse granularity)
    python analysis/analyze_player_props_matrix.py
    
    # Coarse: 4 player tiers × 6 spread bins = 24 combinations
    python analysis/analyze_player_props_matrix.py --granularity coarse
    
    # Fine: 7 player tiers × 9 spread bins = 63 combinations
    python analysis/analyze_player_props_matrix.py --granularity fine
    
    Optional: Modify DATA_PATH at top of script to point to your merged dataset

Output:
    - Console output with all matrices and insights
    - Can be extended to save results to CSV/JSON

Author: Created from notebook analysis on 2026-01-05
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
from pathlib import Path

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
DATA_PATH = project_root / 'data' / '03_intermediate' / 'merged_data.csv'

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

def bin_points_line(line, granularity='coarse'):
    """
    Categorize player by their points line tier
    
    Args:
        line: Points line value
        granularity: 'coarse' (4 tiers) or 'fine' (7 tiers)
    """
    if pd.isna(line):
        return 'Unknown'
    
    if granularity == 'fine':
        # Fine granularity: 7 tiers
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
        # Coarse granularity: 4 tiers (original)
        if line >= 30:
            return '30+ (Superstar)'
        elif line >= 20:
            return '20-30 (Star)'
        elif line >= 10:
            return '10-20 (Role Player)'
        else:
            return '<10 (Bench)'


def bin_team_spread(spread, granularity='coarse'):
    """
    Categorize game by team spread
    
    Args:
        spread: Team spread value (positive = favorite, negative = underdog)
        granularity: 'coarse' (6 bins) or 'fine' (9 bins)
    """
    if pd.isna(spread):
        return 'Unknown'
    
    if granularity == 'fine':
        # Fine granularity: 9 bins
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
        # Coarse granularity: 6 bins (original)
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


def get_tier_order(granularity='coarse'):
    """Get the proper ordering for player tiers"""
    if granularity == 'fine':
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


def get_spread_order(granularity='coarse'):
    """Get the proper ordering for spread bins"""
    if granularity == 'fine':
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

def load_data(data_path):
    """Load and prepare the merged dataset"""
    print_section("Loading Data", "chart")
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"{EMOJI['success']} Loaded {len(df):,} player-game rows")
    print(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")
    print(f"Unique players: {df['PLAYER_NAME'].nunique():,}")
    print(f"Unique games: {df['GAME_ID'].nunique():,}")
    
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


def create_matrix_analysis(df_clean, granularity='coarse'):
    """Create and display tier × spread matrix"""
    print_section(f"Matrix Analysis: Player Tier × Team Spread ({granularity.upper()})", "chart")
    
    # Create bins
    df_clean['line_tier'] = df_clean['points_line'].apply(lambda x: bin_points_line(x, granularity))
    df_clean['spread_bin'] = df_clean['team_spread'].apply(lambda x: bin_team_spread(x, granularity))
    
    # Get proper ordering
    col_order = get_spread_order(granularity)
    row_order = get_tier_order(granularity)
    
    # 1. Sample size matrix
    print_subsection("1️⃣ SAMPLE SIZE MATRIX (number of games)")
    count_matrix = pd.crosstab(
        df_clean['line_tier'],
        df_clean['spread_bin'],
        margins=True,
        margins_name='Total'
    )
    count_matrix = count_matrix.reindex(columns=[c for c in col_order if c in count_matrix.columns])
    count_matrix = count_matrix.reindex([r for r in row_order if r in count_matrix.index])
    print(count_matrix.to_string())
    
    # 2. Over hit rate matrix
    print_subsection("\n2️⃣ OVER HIT RATE MATRIX (% of overs that hit)")
    over_matrix = pd.crosstab(
        df_clean['line_tier'],
        df_clean['spread_bin'],
        values=df_clean['over_hit'],
        aggfunc='mean'
    ) * 100
    over_matrix = over_matrix.reindex(columns=[c for c in col_order[:-1] if c in over_matrix.columns])
    over_matrix = over_matrix.reindex([r for r in row_order[:-1] if r in over_matrix.index])
    print(over_matrix.round(1).to_string())
    
    # 3. Benchmarks
    print_subsection("\n3️⃣ BENCHMARKS")
    print(f"Overall over hit rate: {df_clean['over_hit'].mean()*100:.1f}%")
    
    print("\nBy Player Tier:")
    for tier in row_order[:-1]:
        tier_data = df_clean[df_clean['line_tier'] == tier]
        if len(tier_data) > 0:
            print(f"  {tier:20s}: {tier_data['over_hit'].mean()*100:.1f}% ({len(tier_data):,} games)")
    
    print("\nBy Team Spread:")
    for spread in col_order[:-1]:
        spread_data = df_clean[df_clean['spread_bin'] == spread]
        if len(spread_data) > 0:
            print(f"  {spread:15s}: {spread_data['over_hit'].mean()*100:.1f}% ({len(spread_data):,} games)")
    
    return df_clean


def find_edges(df_clean, baseline_over_rate):
    """
    Identify betting edges from the matrix
    
    Edge explanation:
        Edge = how much better than baseline a combo performs
        
        Example: If baseline over rate is 49%, and a combo hits 53.5% overs:
            → Over edge = +4.5% (you win 4.5% more than expected)
            → To profit at -110 odds, you need ~52.4% hit rate
            → This combo has +1.1% edge over breakeven!
        
        Positive edge = opportunity to bet
        Larger edge = stronger signal
    """
    print_section("Edge Detection", "target")
    
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
            subset = df_clean[(df_clean['line_tier'] == tier) & (df_clean['spread_bin'] == spread)]
            
            if len(subset) >= MIN_SAMPLE_SIZE:
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
    
    # Top over opportunities
    print_subsection(f"\n{EMOJI['fire']} TOP 10 OVER OPPORTUNITIES (highest edge vs baseline)")
    top_overs = df_edges.nlargest(10, 'over_edge')
    cols = ['line_tier', 'spread_bin', 'games', 'over_rate', 'under_rate', 'over_edge', 'push_rate', 'over_roi']
    print(top_overs[cols].to_string(index=False))
    
    # Top under opportunities
    print_subsection(f"\n{EMOJI['cold']} TOP 10 UNDER OPPORTUNITIES (highest edge vs baseline)")
    top_unders = df_edges.nlargest(10, 'under_edge')
    cols = ['line_tier', 'spread_bin', 'games', 'over_rate', 'under_rate', 'under_edge', 'push_rate', 'under_roi']
    print(top_unders[cols].to_string(index=False))
    
    # Key insights
    print_subsection(f"\n{EMOJI['light']} KEY INSIGHTS")
    
    if len(df_edges) > 0:
        best_over = df_edges.loc[df_edges['over_edge'].idxmax()]
        print(f"Strongest OVER edge: {best_over['line_tier']} + {best_over['spread_bin']}")
        print(f"  {EMOJI['target']} {best_over['over_rate']:.1f}% over / {best_over['under_rate']:.1f}% under (sum={best_over['rate_sum']:.1f}%)")
        print(f"  {EMOJI['target']} +{best_over['over_edge']:.1f}% edge | {int(best_over['games'])} games | {best_over['push_rate']:.1f}% pushes")
        print(f"  {EMOJI['target']} Strategy ROI at -110 odds: {best_over['over_roi']:+.1f}%")
        
        best_under = df_edges.loc[df_edges['under_edge'].idxmax()]
        print(f"\nStrongest UNDER edge: {best_under['line_tier']} + {best_under['spread_bin']}")
        print(f"  {EMOJI['target']} {best_under['under_rate']:.1f}% under / {best_under['over_rate']:.1f}% over (sum={best_under['rate_sum']:.1f}%)")
        print(f"  {EMOJI['target']} +{best_under['under_edge']:.1f}% edge | {int(best_under['games'])} games | {best_under['push_rate']:.1f}% pushes")
        print(f"  {EMOJI['target']} Strategy ROI at -110 odds: {best_under['under_roi']:+.1f}%")
        
        weak_samples = df_edges[df_edges['games'] < 100]
        if len(weak_samples) > 0:
            print(f"\n{EMOJI['warning']} {len(weak_samples)} combinations have {MIN_SAMPLE_SIZE}-100 games (use caution)")
    
    return df_edges


def drill_down_edge(df_clean, tier, spread_bin, top_n=15, bet_side='over'):
    """
    Show which players drive a specific edge
    
    Args:
        bet_side: 'over' or 'under' - determines which ROI to calculate
    """
    subset = df_clean[(df_clean['line_tier'] == tier) & (df_clean['spread_bin'] == spread_bin)]
    
    if len(subset) == 0:
        print(f"No data for {tier} + {spread_bin}")
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
    print(f"\n{tier} + {spread_bin} | Total: {len(subset)} games, {overall_rate:.1f}% {bet_side}s")
    print("-" * 80)
    cols = ['PLAYER_NAME', 'games', 'over_rate', 'under_rate', 'avg_pts', 'avg_line', 'avg_diff', 'profitable', 'roi']
    print(player_stats[cols].to_string(index=False))


def show_top_edge_players(df_clean, df_edges):
    """Show player breakdown for best edges"""
    print_section("Player Breakdown for Top Edges")
    
    if len(df_edges) == 0:
        print("No edges found with sufficient sample size")
        return
    
    best_over = df_edges.loc[df_edges['over_edge'].idxmax()]
    print(f"\n{EMOJI['fire']} BEST OVER EDGE BREAKDOWN:")
    drill_down_edge(df_clean, best_over['line_tier'], best_over['spread_bin'], bet_side='over')
    
    best_under = df_edges.loc[df_edges['under_edge'].idxmax()]
    print(f"\n\n{EMOJI['cold']} BEST UNDER EDGE BREAKDOWN:")
    drill_down_edge(df_clean, best_under['line_tier'], best_under['spread_bin'], bet_side='under')


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
  # Run with coarse granularity (4 tiers × 6 spreads = 24 combos)
  python analysis/analyze_player_props_matrix.py --granularity coarse
  
  # Run with fine granularity (7 tiers × 9 spreads = 63 combos)
  python analysis/analyze_player_props_matrix.py --granularity fine
  
Granularity levels:
  coarse: <10, 10-20, 20-30, 30+ pts  ×  10+ Dog, 5-10 Dog, 0-5 Dog, 0-5 Fav, 5-10 Fav, 10+ Fav
  fine:   0-5, 5-10, 10-15, 15-20, 20-25, 25-30, 30+ pts  ×  15+ Dog, 10-15 Dog, ..., Pick'em, ..., 15+ Fav
        """
    )
    
    parser.add_argument(
        '--granularity',
        type=str,
        choices=['coarse', 'fine'],
        default='coarse',
        help='Level of granularity for bins (default: coarse)'
    )
    
    return parser.parse_args()


def main():
    """Run the full analysis pipeline"""
    args = parse_args()
    
    print_section("NBA Player Props Matrix Analysis - 2025-26 Season", "chart")
    print(f"Data source: {DATA_PATH}")
    print(f"Granularity: {args.granularity.upper()}")
    print(f"Min sample size: {MIN_SAMPLE_SIZE} games")
    print(f"Early exit threshold: {EARLY_EXIT_THRESHOLD*100:.0f}% usual minutes")
    
    # Load data
    df = load_data(DATA_PATH)
    
    # Filter to actionable bets
    df_bets = filter_actionable_data(df)
    baseline_over_rate = df_bets['over_hit'].mean() * 100
    
    # Remove injury games
    df_clean = remove_early_exits(df_bets)
    
    # Create matrix analysis
    df_clean = create_matrix_analysis(df_clean, args.granularity)
    
    # Find edges
    df_edges = find_edges(df_clean, baseline_over_rate)
    
    # Show player breakdowns
    show_top_edge_players(df_clean, df_edges)
    
    # Summary
    print_section("Analysis Complete", "success")
    print(f"Granularity: {args.granularity}")
    print(f"Clean games analyzed: {len(df_clean):,}")
    print(f"Tier × Spread combinations with {MIN_SAMPLE_SIZE}+ games: {len(df_edges)}")
    
    if args.granularity == 'fine':
        total_combos = 7 * 9
        print(f"Total possible combinations: {total_combos} (fine: 7 tiers × 9 spreads)")
    else:
        total_combos = 4 * 6
        print(f"Total possible combinations: {total_combos} (coarse: 4 tiers × 6 spreads)")
    
    coverage_pct = (len(df_edges) / total_combos) * 100
    print(f"Coverage: {coverage_pct:.1f}% of combinations have {MIN_SAMPLE_SIZE}+ games")
    
    print(f"\nKey variables available in memory:")
    print(f"  • df_clean: Clean dataset ({len(df_clean):,} rows)")
    print(f"  • df_edges: Edge opportunities ({len(df_edges)} combinations)")
    
    return df_clean, df_edges


if __name__ == '__main__':
    df_clean, df_edges = main()

