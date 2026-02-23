"""
NBA Props Player-Level Grid Search with Stability Analysis

Tests momentum/reversal strategies at the individual player level and measures
whether profitable patterns align with player stability/predictability.

Key Innovation:
Different players have different levels of predictability. This script:
1. Measures each player's stability (R²) in each market
2. Tests L1-L10 momentum/reversal strategies per player
3. Identifies alignment between stability and profitable strategy type

Theory:
- High Stability + Momentum = Strong edge (ride predictable streaks)
- Low Stability + Reversal = Good edge (fade noisy streaks)
- High Stability + Reversal = Conflict (investigate further)
- Low Stability + Momentum = Likely noise (skip)

Context from Graduate Research:
This approach is inspired by research showing that stability/predictability
of performance varies significantly across players, and this heterogeneity
can be exploited for betting strategies.

Output:
- ~/Downloads/tmp/prop_predictive_power_analysis/player_level_grid/
  - player_stability_by_market.csv (R² for each player+market)
  - player_strategies_all.csv (all L1-L10 results per player)
  - player_strategies_profitable.csv (only ROI > 0)
  - player_strategies_aligned.csv (strategies that align with stability)
  - summary_by_player.csv (best strategy per player+market)
  - visualizations/

Usage:
    # Top 50 players, all seasons
    python analysis/grid_search_player_level.py --top-n 50 --seasons 2023-24 2024-25 2025-26
    
    # Top 100, single season for testing
    python analysis/grid_search_player_level.py --top-n 100 --seasons 2024-25
    
    # Specific market only
    python analysis/grid_search_player_level.py --top-n 50 --market player_points --seasons 2023-24 2024-25 2025-26
    
    # Adjust thresholds
    python analysis/grid_search_player_level.py --top-n 50 --stability-threshold 0.4 --min-bets 20

Author: Myles Thomas
Date: 2026-02-10
"""

import sys
import os
import logging
from pathlib import Path
import argparse
import warnings
from collections import defaultdict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

ALL_MARKETS = [
    'player_points',
    'player_rebounds',
    'player_assists',
    'player_threes',
    'player_blocks',
    'player_steals',
    'player_double_double',
    'player_triple_double',
    'player_points_rebounds_assists'
]

OUTPUT_BASE = Path.home() / 'Downloads' / 'tmp' / 'prop_predictive_power_analysis' / 'player_level_grid'

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def american_to_decimal(american_odds):
    """Convert American odds to decimal odds"""
    if pd.isna(american_odds) or american_odds == 0:
        return None
    
    if american_odds > 0:
        return 1 + (american_odds / 100)
    else:
        return 1 + (100 / abs(american_odds))


def calculate_profit(odds, win, stake=100):
    """Calculate profit for a single bet"""
    if pd.isna(odds) or odds == 0:
        return None
    
    decimal_odds = american_to_decimal(odds)
    if decimal_odds is None:
        return None
    
    if win:
        return (decimal_odds - 1) * stake
    else:
        return -stake


def is_valid_american_odds(odds):
    """Check if odds value is valid American odds (not corrupted data)"""
    if pd.isna(odds) or odds == 0:
        return False
    
    # Valid American odds are either:
    # - >= 100 (positive odds)
    # - <= -110 (negative odds, using -110 as practical minimum)
    return (odds >= 100) or (odds <= -110)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_props_with_actuals(season):
    """Load props data with actuals"""
    data_file = OUTPUT_BASE.parent / season / '01_data' / 'props_with_actuals.csv'
    
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        return pd.DataFrame()
    
    df = pd.read_csv(data_file)
    return df


def get_top_n_players(df, n=50):
    """Get top N players by average PPG across seasons"""
    # Calculate average points per game for each player
    ppg = df[df['market'] == 'player_points'].groupby('player_normalized').agg({
        'actual_value': 'mean',
        'game_date': 'count'
    }).reset_index()
    
    ppg.columns = ['player_normalized', 'avg_ppg', 'games']
    
    # Filter to players with at least 20 games
    ppg = ppg[ppg['games'] >= 20]
    
    # Sort by PPG
    ppg = ppg.sort_values('avg_ppg', ascending=False)
    
    top_n = ppg.head(n)['player_normalized'].tolist()
    
    print(f"\n📊 Top {len(top_n)} Players by PPG:")
    for i, row in ppg.head(n).iterrows():
        print(f"   {row['player_normalized']:30s} {row['avg_ppg']:5.1f} PPG ({int(row['games'])} games)")
    
    return top_n


# =============================================================================
# STABILITY ANALYSIS
# =============================================================================

def calculate_stability_r2(df_player_market, bin_size=5):
    """
    Calculate stability R² for a player in a market using binning approach
    
    Measures how well previous bin performance predicts current bin performance.
    High R² = player is stable/predictable in this market
    
    Args:
        df_player_market: DataFrame filtered to one player+market
        bin_size: Size of rolling window bins (default: 5)
    
    Returns:
        float: R² value (0 to 1), or None if insufficient data
    """
    if len(df_player_market) < bin_size * 3:  # Need at least 3 bins
        return None
    
    df = df_player_market.copy().sort_values('game_date')
    
    # Create rolling bins
    df['bin_mean'] = df['actual_value'].rolling(window=bin_size, min_periods=bin_size).mean()
    df['prev_bin_mean'] = df['bin_mean'].shift(bin_size)
    
    # Remove rows without both values
    df_valid = df.dropna(subset=['bin_mean', 'prev_bin_mean'])
    
    if len(df_valid) < 10:  # Need reasonable sample
        return None
    
    X = df_valid['prev_bin_mean'].values.reshape(-1, 1)
    y = df_valid['bin_mean'].values
    
    try:
        r2 = r2_score(y, LinearRegression().fit(X, y).predict(X))
        return max(0, r2)  # Don't allow negative R²
    except:
        return None


def analyze_player_stability(df, players, markets):
    """Calculate stability for all player+market combinations"""
    print(f"\n{'='*80}")
    print(f"ANALYZING PLAYER STABILITY")
    print(f"{'='*80}")
    
    results = []
    
    for player in players:
        df_player = df[df['player_normalized'] == player]
        
        for market in markets:
            df_pm = df_player[df_player['market'] == market]
            
            if len(df_pm) < 15:  # Skip if too few observations
                continue
            
            r2 = calculate_stability_r2(df_pm, bin_size=5)
            
            if r2 is not None:
                results.append({
                    'player': player,
                    'market': market,
                    'stability_r2': r2,
                    'num_observations': len(df_pm),
                    'is_stable': r2 > 0.5  # Will make this configurable
                })
    
    df_stability = pd.DataFrame(results)
    
    print(f"   ✅ Calculated stability for {len(df_stability)} player+market combinations")
    print(f"   📊 High stability (R² > 0.5): {(df_stability['is_stable']).sum()}")
    print(f"   📊 Low stability (R² ≤ 0.5): {(~df_stability['is_stable']).sum()}")
    
    return df_stability


# =============================================================================
# SIGNAL GENERATION
# =============================================================================

def create_streak_signals(df, player, market, max_window=10):
    """Create L1-L10 streak signals for a single player+market"""
    df_pm = df[(df['player_normalized'] == player) & (df['market'] == market)].copy()
    
    if df_pm.empty:
        return pd.DataFrame()
    
    df_pm = df_pm.sort_values('game_date')
    
    # Create signals for each window
    for n in range(1, max_window + 1):
        # Count overs in last N games
        df_pm[f'overs_L{n}'] = df_pm['beat_line'].rolling(window=n, min_periods=n).sum()
        df_pm[f'overs_L{n}_lag'] = df_pm[f'overs_L{n}'].shift(1)
        
        # Binary signals
        df_pm[f'all_over_L{n}'] = (df_pm[f'overs_L{n}_lag'] == n).astype(int)
        df_pm[f'all_under_L{n}'] = (df_pm[f'overs_L{n}_lag'] == 0).astype(int)
    
    return df_pm


# =============================================================================
# STRATEGY TESTING
# =============================================================================

def test_strategy(df_signals, signal_col, bet_side, stake=100):
    """Test a single strategy for a player+market"""
    df_bets = df_signals[df_signals[signal_col] == 1].copy()
    
    if len(df_bets) == 0:
        return None
    
    # Set bet odds and outcome
    if bet_side == 'Over':
        df_bets['bet_odds'] = df_bets['over_odds']
        df_bets['win'] = (df_bets['actual_value'] > df_bets['prop_line']).astype(int)
    else:
        df_bets['bet_odds'] = df_bets['under_odds']
        df_bets['win'] = (df_bets['actual_value'] < df_bets['prop_line']).astype(int)
    
    # Filter to valid odds only
    df_bets = df_bets[df_bets['bet_odds'].apply(is_valid_american_odds)].copy()
    
    if len(df_bets) == 0:
        return None
    
    # Calculate profits
    df_bets['profit'] = df_bets.apply(
        lambda row: calculate_profit(row['bet_odds'], row['win'], stake),
        axis=1
    )
    
    df_bets = df_bets.dropna(subset=['profit'])
    
    if len(df_bets) == 0:
        return None
    
    # Metrics
    num_bets = len(df_bets)
    win_rate = df_bets['win'].mean()
    median_odds = df_bets['bet_odds'].median()
    total_profit = df_bets['profit'].sum()
    total_wagered = num_bets * stake
    roi = (total_profit / total_wagered) * 100
    
    return {
        'num_bets': num_bets,
        'win_rate': win_rate,
        'median_odds': median_odds,
        'total_profit': total_profit,
        'total_wagered': total_wagered,
        'roi': roi
    }


def grid_search_player_market(df, player, market, max_window=10, stake=100):
    """Run grid search for a single player+market combination"""
    df_signals = create_streak_signals(df, player, market, max_window)
    
    if df_signals.empty:
        return []
    
    results = []
    
    for n in range(1, max_window + 1):
        # Momentum Over
        m_over = test_strategy(df_signals, f'all_over_L{n}', 'Over', stake)
        if m_over:
            results.append({
                'player': player,
                'market': market,
                'window': f'L{n}',
                'strategy': 'M-Over',
                'strategy_type': 'Momentum',
                **m_over
            })
        
        # Momentum Under
        m_under = test_strategy(df_signals, f'all_under_L{n}', 'Under', stake)
        if m_under:
            results.append({
                'player': player,
                'market': market,
                'window': f'L{n}',
                'strategy': 'M-Under',
                'strategy_type': 'Momentum',
                **m_under
            })
        
        # Reversal Over (all over → bet under)
        r_over = test_strategy(df_signals, f'all_over_L{n}', 'Under', stake)
        if r_over:
            results.append({
                'player': player,
                'market': market,
                'window': f'L{n}',
                'strategy': 'R-Over',
                'strategy_type': 'Reversal',
                **r_over
            })
        
        # Reversal Under (all under → bet over)
        r_under = test_strategy(df_signals, f'all_under_L{n}', 'Over', stake)
        if r_under:
            results.append({
                'player': player,
                'market': market,
                'window': f'L{n}',
                'strategy': 'R-Under',
                'strategy_type': 'Reversal',
                **r_under
            })
    
    return results


# =============================================================================
# ALIGNMENT ANALYSIS
# =============================================================================

def analyze_alignment(df_strategies, df_stability, stability_threshold=0.5):
    """
    Determine alignment between stability and profitable strategies
    
    Rules:
    - High Stability (R² > threshold) + Momentum = STRONG (✅)
    - Low Stability (R² ≤ threshold) + Reversal = GOOD (✅)
    - High Stability + Reversal = CONFLICT (⚠️)
    - Low Stability + Momentum = NOISE (❌)
    """
    # Merge strategies with stability
    df_merged = df_strategies.merge(
        df_stability[['player', 'market', 'stability_r2', 'is_stable']],
        on=['player', 'market'],
        how='left'
    )
    
    # Define alignment
    def get_alignment(row):
        if pd.isna(row['stability_r2']):
            return 'UNKNOWN'
        
        is_stable = row['stability_r2'] > stability_threshold
        is_momentum = row['strategy_type'] == 'Momentum'
        
        if is_stable and is_momentum:
            return 'STRONG'
        elif not is_stable and not is_momentum:
            return 'GOOD'
        elif is_stable and not is_momentum:
            return 'CONFLICT'
        else:
            return 'NOISE'
    
    df_merged['alignment'] = df_merged.apply(get_alignment, axis=1)
    
    # Add emoji for display
    alignment_emoji = {
        'STRONG': '✅',
        'GOOD': '✅',
        'CONFLICT': '⚠️',
        'NOISE': '❌',
        'UNKNOWN': '❓'
    }
    df_merged['alignment_emoji'] = df_merged['alignment'].map(alignment_emoji)
    
    return df_merged


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Player-level grid search with stability analysis'
    )
    parser.add_argument('--seasons', type=str, nargs='+', required=True,
                        help='Seasons to analyze (e.g., 2023-24 2024-25 2025-26)')
    parser.add_argument('--top-n', type=int, default=50,
                        help='Number of top players by PPG (default: 50)')
    parser.add_argument('--market', type=str,
                        help='Specific market to analyze (default: all)')
    parser.add_argument('--max-window', type=int, default=10,
                        help='Maximum lookback window (default: 10)')
    parser.add_argument('--min-bets', type=int, default=10,
                        help='Minimum bets to report strategy (default: 10)')
    parser.add_argument('--stability-threshold', type=float, default=0.5,
                        help='R² threshold for high stability (default: 0.5)')
    parser.add_argument('--stake', type=int, default=100,
                        help='Bet size (default: 100)')
    
    args = parser.parse_args()
    
    markets = [args.market] if args.market else ALL_MARKETS
    
    print(f"\n{'='*80}")
    print(f"PLAYER-LEVEL GRID SEARCH WITH STABILITY ANALYSIS")
    print(f"{'='*80}")
    print(f"Seasons: {', '.join(args.seasons)}")
    print(f"Top N Players: {args.top_n}")
    print(f"Markets: {len(markets)}")
    print(f"Windows: L1 to L{args.max_window}")
    print(f"Stability Threshold: R² > {args.stability_threshold}")
    print(f"Min Bets: {args.min_bets}")
    
    # Load data from all seasons
    print(f"\n{'='*80}")
    print(f"LOADING DATA")
    print(f"{'='*80}")
    
    dfs = []
    for season in args.seasons:
        print(f"📊 Loading {season}...")
        df_season = load_props_with_actuals(season)
        if not df_season.empty:
            df_season['season'] = season
            dfs.append(df_season)
            print(f"   ✅ {len(df_season):,} rows")
    
    if not dfs:
        print("❌ No data loaded")
        return
    
    df_all = pd.concat(dfs, ignore_index=True)
    print(f"\n✅ Total: {len(df_all):,} rows across {len(args.seasons)} seasons")
    
    # Get top N players
    top_players = get_top_n_players(df_all, args.top_n)
    
    # Filter to top players
    df_filtered = df_all[df_all['player_normalized'].isin(top_players)]
    
    # Step 1: Calculate stability for all player+market combos
    df_stability = analyze_player_stability(df_filtered, top_players, markets)
    
    # Step 2: Run grid search for each player+market
    print(f"\n{'='*80}")
    print(f"RUNNING PLAYER-LEVEL GRID SEARCH")
    print(f"{'='*80}")
    
    all_strategies = []
    total_combos = len(top_players) * len(markets)
    completed = 0
    
    for player in top_players:
        for market in markets:
            completed += 1
            if completed % 10 == 0:
                print(f"   Progress: {completed}/{total_combos} ({completed/total_combos*100:.0f}%)")
            
            strategies = grid_search_player_market(df_filtered, player, market, args.max_window, args.stake)
            all_strategies.extend(strategies)
    
    print(f"\n   ✅ Completed {completed} player+market combinations")
    print(f"   ✅ Generated {len(all_strategies)} strategy results")
    
    if not all_strategies:
        print("❌ No strategies generated")
        return
    
    df_strategies = pd.DataFrame(all_strategies)
    
    # Step 3: Analyze alignment
    print(f"\n{'='*80}")
    print(f"ANALYZING ALIGNMENT")
    print(f"{'='*80}")
    
    df_aligned = analyze_alignment(df_strategies, df_stability, args.stability_threshold)
    
    # Filter to strategies with enough bets
    df_aligned = df_aligned[df_aligned['num_bets'] >= args.min_bets]
    
    # Summary statistics
    print(f"\n📊 Alignment Summary (strategies with {args.min_bets}+ bets):")
    alignment_counts = df_aligned['alignment'].value_counts()
    for alignment, count in alignment_counts.items():
        emoji = {'STRONG': '✅', 'GOOD': '✅', 'CONFLICT': '⚠️', 'NOISE': '❌', 'UNKNOWN': '❓'}.get(alignment, '')
        print(f"   {emoji} {alignment:10s}: {count:>5,} strategies")
    
    # Profitable strategies
    df_profitable = df_aligned[df_aligned['roi'] > 0].copy()
    print(f"\n💰 Profitable Strategies: {len(df_profitable)} ({len(df_profitable)/len(df_aligned)*100:.1f}%)")
    
    if len(df_profitable) > 0:
        profitable_by_alignment = df_profitable['alignment'].value_counts()
        print(f"\n   Breakdown by alignment:")
        for alignment, count in profitable_by_alignment.items():
            emoji = {'STRONG': '✅', 'GOOD': '✅', 'CONFLICT': '⚠️', 'NOISE': '❌', 'UNKNOWN': '❓'}.get(alignment, '')
            print(f"   {emoji} {alignment:10s}: {count:>5,}")
    
    # Best strategy per player+market
    df_best = df_aligned.loc[df_aligned.groupby(['player', 'market'])['roi'].idxmax()].copy()
    df_best = df_best.sort_values('roi', ascending=False)
    
    # Save results
    output_dir = OUTPUT_BASE
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"SAVING RESULTS")
    print(f"{'='*80}")
    
    df_stability.to_csv(output_dir / 'player_stability_by_market.csv', index=False)
    print(f"   ✅ {output_dir / 'player_stability_by_market.csv'}")
    
    df_aligned.to_csv(output_dir / 'player_strategies_all.csv', index=False)
    print(f"   ✅ {output_dir / 'player_strategies_all.csv'}")
    
    df_profitable.to_csv(output_dir / 'player_strategies_profitable.csv', index=False)
    print(f"   ✅ {output_dir / 'player_strategies_profitable.csv'}")
    
    df_best.to_csv(output_dir / 'summary_best_per_player_market.csv', index=False)
    print(f"   ✅ {output_dir / 'summary_best_per_player_market.csv'}")
    
    # Display top results
    print(f"\n{'='*80}")
    print(f"TOP 20 PROFITABLE STRATEGIES")
    print(f"{'='*80}")
    
    if len(df_profitable) > 0:
        top_20 = df_profitable.nlargest(20, 'roi')
        for _, row in top_20.iterrows():
            print(f"\n{row['alignment_emoji']} {row['player']:30s} | {row['market']:30s}")
            print(f"   Strategy: {row['strategy']} {row['window']} | ROI: {row['roi']:>6.1f}% ({int(row['num_bets'])} bets, {row['win_rate']*100:.1f}% win)")
            print(f"   Stability: R² = {row['stability_r2']:.3f} | Alignment: {row['alignment']}")
    else:
        print("   No profitable strategies found")
    
    print(f"\n✅ Analysis complete!")
    print(f"📁 Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
