"""
NBA Props Grid Search - L1 to L10 Lookback Windows

Tests simple momentum and reversal strategies with varying lookback windows
to find optimal signal for each market.

Context:
Previous analysis used arbitrary "3+ overs in last 5 games" threshold.
This script tests simpler, more interpretable signals:
- MOMENTUM: Last N games all went Over → Bet Over
- REVERSAL: Last N games all went Over → Bet Under

We test N from 1 to 10 to find optimal lookback window per market.

Strategies Tested per Market:
1. M-Over: Last N games Over → Bet Over (ride hot streak)
2. M-Under: Last N games Under → Bet Under (ride cold streak)  
3. R-Over: Last N games Over → Bet Under (fade hot streak)
4. R-Under: Last N games Under → Bet Over (fade cold streak)

Output:
- ~/Downloads/tmp/prop_predictive_power_analysis/grid_search/
  - grid_results_by_market/{market}_L1_L10.csv
  - best_strategy_per_market.csv
  - summary_heatmap.csv
  - visualizations/roi_heatmap.png

Usage:
    # All markets, all seasons
    python analysis/grid_search_L1_to_L10.py --seasons 2023-24 2024-25 2025-26
    
    # Single market to test
    python analysis/grid_search_L1_to_L10.py --market player_blocks --seasons 2023-24 2024-25 2025-26
    
    # Skip visualizations
    python analysis/grid_search_L1_to_L10.py --seasons 2023-24 2024-25 2025-26 --no-viz

Author: Myles Thomas
Date: 2026-02-10
"""

import sys
import os
from pathlib import Path
import argparse
from io import StringIO
import warnings

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import boto3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Markets to analyze
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

# Output directory
OUTPUT_BASE = Path.home() / 'Downloads' / 'tmp' / 'prop_predictive_power_analysis' / 'grid_search'

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


# =============================================================================
# DATA LOADING
# =============================================================================

def load_props_with_actuals(season):
    """Load props data with actuals from intermediate files"""
    data_file = OUTPUT_BASE.parent / season / '01_data' / 'props_with_actuals.csv'
    
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        print(f"   Run: python analysis/analyze_nba_props_stability_and_predictive_power.py --season {season}")
        return pd.DataFrame()
    
    print(f"📊 Loading {season}...")
    df = pd.read_csv(data_file)
    print(f"   ✅ {len(df):,} rows, {df['player_normalized'].nunique()} players")
    
    return df


# =============================================================================
# SIGNAL GENERATION
# =============================================================================

def create_streak_signals(df, market, max_window=10):
    """
    Create streak signals for L1 through L10
    
    Args:
        df: DataFrame with props and actuals
        market: Market to analyze
        max_window: Maximum lookback window
    
    Returns:
        DataFrame with all streak signals
    """
    # Filter to this market
    df_market = df[df['market'] == market].copy()
    
    # Sort by player and date
    df_market = df_market.sort_values(['player_normalized', 'game_date'])
    
    # For each window size, create streak indicators
    for n in range(1, max_window + 1):
        # Count overs in last N games
        df_market[f'overs_L{n}'] = df_market.groupby('player_normalized')['beat_line'].rolling(
            window=n, min_periods=n
        ).sum().reset_index(level=0, drop=True)
        
        # Shift to avoid lookahead
        df_market[f'overs_L{n}_lag'] = df_market.groupby('player_normalized')[f'overs_L{n}'].shift(1)
        
        # Create signals
        df_market[f'all_over_L{n}'] = (df_market[f'overs_L{n}_lag'] == n).astype(int)  # All N games over
        df_market[f'all_under_L{n}'] = (df_market[f'overs_L{n}_lag'] == 0).astype(int)  # All N games under
    
    return df_market


# =============================================================================
# STRATEGY TESTING
# =============================================================================

def test_strategy(df_signals, signal_col, bet_side, stake=100):
    """
    Test a single strategy
    
    Args:
        df_signals: DataFrame with signals
        signal_col: Column name for signal (e.g., 'all_over_L3')
        bet_side: 'Over' or 'Under'
        stake: Bet amount
    
    Returns:
        Dictionary with results
    """
    # Filter to where signal fires
    df_bets = df_signals[df_signals[signal_col] == 1].copy()
    
    if len(df_bets) == 0:
        return None
    
    # Set bet side and odds
    if bet_side == 'Over':
        df_bets['bet_odds'] = df_bets['over_odds']
        df_bets['win'] = (df_bets['actual_value'] > df_bets['prop_line']).astype(int)
    else:
        df_bets['bet_odds'] = df_bets['under_odds']
        df_bets['win'] = (df_bets['actual_value'] < df_bets['prop_line']).astype(int)
    
    # Filter to valid American odds only (exclude corrupted data)
    df_bets = df_bets[
        (df_bets['bet_odds'] <= -105) | (df_bets['bet_odds'] >= 100)
    ].copy()
    
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
    
    # Calculate metrics
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


# =============================================================================
# GRID SEARCH
# =============================================================================

def grid_search_market(df, market, max_window=10, stake=100):
    """
    Run grid search for a single market
    
    Args:
        df: DataFrame with props and actuals
        market: Market to test
        max_window: Maximum lookback window
        stake: Bet amount
    
    Returns:
        DataFrame with all results
    """
    print(f"\n{'='*80}")
    print(f"GRID SEARCH: {market}")
    print(f"{'='*80}")
    
    # Create all signals
    df_signals = create_streak_signals(df, market, max_window)
    
    if df_signals.empty:
        print(f"   ⚠️  No data for {market}")
        return pd.DataFrame()
    
    results = []
    
    # Test each window size
    for n in range(1, max_window + 1):
        print(f"\n   L{n}: Testing...")
        
        # Strategy 1: Momentum Over (all over → bet over)
        m_over = test_strategy(df_signals, f'all_over_L{n}', 'Over', stake)
        if m_over:
            print(f"      M-Over:  {m_over['num_bets']:>5,} bets | {m_over['win_rate']*100:>4.1f}% win | {m_over['median_odds']:>6.0f} odds | {m_over['roi']:>6.1f}% ROI")
            results.append({
                'market': market,
                'window': f'L{n}',
                'strategy': 'M-Over',
                'signal': f'Last {n} all Over',
                'bet_side': 'Over',
                **m_over
            })
        
        # Strategy 2: Momentum Under (all under → bet under)
        m_under = test_strategy(df_signals, f'all_under_L{n}', 'Under', stake)
        if m_under:
            print(f"      M-Under: {m_under['num_bets']:>5,} bets | {m_under['win_rate']*100:>4.1f}% win | {m_under['median_odds']:>6.0f} odds | {m_under['roi']:>6.1f}% ROI")
            results.append({
                'market': market,
                'window': f'L{n}',
                'strategy': 'M-Under',
                'signal': f'Last {n} all Under',
                'bet_side': 'Under',
                **m_under
            })
        
        # Strategy 3: Reversal Over (all over → bet under)
        r_over = test_strategy(df_signals, f'all_over_L{n}', 'Under', stake)
        if r_over:
            print(f"      R-Over:  {r_over['num_bets']:>5,} bets | {r_over['win_rate']*100:>4.1f}% win | {r_over['median_odds']:>6.0f} odds | {r_over['roi']:>6.1f}% ROI")
            results.append({
                'market': market,
                'window': f'L{n}',
                'strategy': 'R-Over',
                'signal': f'Last {n} all Over',
                'bet_side': 'Under',
                **r_over
            })
        
        # Strategy 4: Reversal Under (all under → bet over)
        r_under = test_strategy(df_signals, f'all_under_L{n}', 'Over', stake)
        if r_under:
            print(f"      R-Under: {r_under['num_bets']:>5,} bets | {r_under['win_rate']*100:>4.1f}% win | {r_under['median_odds']:>6.0f} odds | {r_under['roi']:>6.1f}% ROI")
            results.append({
                'market': market,
                'window': f'L{n}',
                'strategy': 'R-Under',
                'signal': f'Last {n} all Under',
                'bet_side': 'Over',
                **r_under
            })
    
    if not results:
        return pd.DataFrame()
    
    df_results = pd.DataFrame(results)
    
    # Find best strategy for this market
    best_idx = df_results['roi'].idxmax()
    best = df_results.loc[best_idx]
    
    print(f"\n   🏆 BEST: {best['strategy']} {best['window']} | ROI: {best['roi']:.1f}% | {best['num_bets']:,} bets")
    
    return df_results


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_heatmap(df_all, output_dir):
    """Create ROI heatmap across windows and markets"""
    print(f"\n{'='*80}")
    print(f"CREATING ROI HEATMAP")
    print(f"{'='*80}")
    
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # For each strategy type, create heatmap
    for strat_type in ['M-Over', 'M-Under', 'R-Over', 'R-Under']:
        df_strat = df_all[df_all['strategy'] == strat_type].copy()
        
        if df_strat.empty:
            continue
        
        # Pivot to create matrix
        pivot = df_strat.pivot_table(
            values='roi',
            index='market',
            columns='window',
            aggfunc='mean'
        )
        
        # Reorder columns (L1, L2, ..., L10)
        cols = [f'L{i}' for i in range(1, 11) if f'L{i}' in pivot.columns]
        pivot = pivot[cols]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 8))
        
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn',
            center=0,
            vmin=-20,
            vmax=20,
            cbar_kws={'label': 'ROI (%)'},
            ax=ax
        )
        
        ax.set_title(f'ROI Heatmap: {strat_type} Strategy', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Lookback Window', fontsize=12, fontweight='bold')
        ax.set_ylabel('Market', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        filename = f'roi_heatmap_{strat_type.lower().replace("-", "_")}.png'
        plt.savefig(viz_dir / filename, dpi=300, bbox_inches='tight')
        print(f"   ✅ Saved: {viz_dir / filename}")
        plt.close()


def create_summary_plot(df_best, output_dir):
    """Create summary plot of best strategies"""
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    df_plot = df_best.sort_values('roi', ascending=True)
    
    markets = df_plot['market'].values
    rois = df_plot['roi'].values
    labels = [f"{s} {w}" for s, w in zip(df_plot['strategy'], df_plot['window'])]
    
    colors = ['green' if roi > 0 else 'red' for roi in rois]
    
    bars = ax.barh(markets, rois, color=colors, alpha=0.7)
    
    ax.set_xlabel('ROI (%)', fontsize=12, fontweight='bold')
    ax.set_title('Best Strategy per Market (L1-L10 Grid Search)', fontsize=14, fontweight='bold', pad=20)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    
    # Add labels
    for i, (market, roi, label) in enumerate(zip(markets, rois, labels)):
        ax.text(roi + 0.3, i, f'{roi:.1f}% ({label})', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'best_strategy_summary.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {viz_dir / 'best_strategy_summary.png'}")
    plt.close()


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Grid search L1-L10 lookback windows for NBA props'
    )
    parser.add_argument('--season', type=str, default='2025-26',
                        help='NBA season (e.g., 2025-26)')
    parser.add_argument('--seasons', type=str, nargs='+',
                        help='Multiple NBA seasons (e.g., 2023-24 2024-25 2025-26)')
    parser.add_argument('--market', type=str,
                        help='Specific market to test (default: all)')
    parser.add_argument('--max-window', type=int, default=10,
                        help='Maximum lookback window (default: 10)')
    parser.add_argument('--stake', type=int, default=100,
                        help='Bet size (default: 100)')
    parser.add_argument('--no-viz', action='store_true',
                        help='Skip visualizations')
    
    args = parser.parse_args()
    
    # Determine seasons
    seasons = args.seasons if args.seasons else [args.season]
    
    # Determine markets
    markets = [args.market] if args.market else ALL_MARKETS
    
    print(f"\n{'='*80}")
    print(f"NBA PROPS GRID SEARCH - L1 TO L{args.max_window}")
    print(f"{'='*80}")
    print(f"Seasons: {', '.join(seasons)}")
    print(f"Markets: {len(markets)}")
    print(f"Windows: L1 to L{args.max_window}")
    print(f"Stake: ${args.stake}")
    print(f"\nStrategies per window:")
    print(f"  M-Over:  All N games Over → Bet Over (ride hot)")
    print(f"  M-Under: All N games Under → Bet Under (ride cold)")
    print(f"  R-Over:  All N games Over → Bet Under (fade hot)")
    print(f"  R-Under: All N games Under → Bet Over (fade cold)")
    
    # Process each season
    all_results = []
    
    for season in seasons:
        print(f"\n\n{'#'*80}")
        print(f"# SEASON: {season}")
        print(f"{'#'*80}")
        
        # Load data
        df = load_props_with_actuals(season)
        
        if df.empty:
            print(f"❌ Skipping {season}")
            continue
        
        # Grid search each market
        for market in markets:
            df_market_results = grid_search_market(df, market, args.max_window, args.stake)
            
            if not df_market_results.empty:
                df_market_results['season'] = season
                all_results.append(df_market_results)
    
    if not all_results:
        print(f"\n❌ No results")
        return
    
    # Combine all results
    df_all = pd.concat(all_results, ignore_index=True)
    
    # Aggregate across seasons
    df_agg = df_all.groupby(['market', 'window', 'strategy', 'signal', 'bet_side']).agg({
        'num_bets': 'sum',
        'win_rate': 'mean',
        'median_odds': 'median',
        'total_profit': 'sum',
        'total_wagered': 'sum'
    }).reset_index()
    
    df_agg['roi'] = (df_agg['total_profit'] / df_agg['total_wagered']) * 100
    
    # Find best strategy per market
    best_strategies = []
    for market in df_agg['market'].unique():
        market_results = df_agg[df_agg['market'] == market]
        best_idx = market_results['roi'].idxmax()
        best = market_results.loc[best_idx]
        best_strategies.append(best.to_dict())
    
    df_best = pd.DataFrame(best_strategies)
    df_best = df_best.sort_values('roi', ascending=False)
    
    # Display results
    print(f"\n\n{'='*80}")
    print(f"FINAL RESULTS - BEST STRATEGY PER MARKET (AGGREGATED)")
    print(f"{'='*80}")
    
    df_display = df_best.copy()
    df_display['win_rate'] = (df_display['win_rate'] * 100).round(1)
    df_display['roi'] = df_display['roi'].round(1)
    df_display['profitable'] = df_display['roi'] > 0
    
    print(df_display[['market', 'strategy', 'window', 'num_bets', 'win_rate', 'median_odds', 'roi', 'profitable']].to_string(index=False))
    
    # Count profitable
    num_profitable = (df_best['roi'] > 0).sum()
    print(f"\n✅ Profitable markets: {num_profitable} of {len(df_best)}")
    
    if num_profitable > 0:
        print(f"\n🏆 PROFITABLE STRATEGIES:")
        df_prof = df_display[df_display['profitable']]
        print(df_prof[['market', 'strategy', 'window', 'win_rate', 'roi', 'num_bets']].to_string(index=False))
    
    # Save results
    output_dir = OUTPUT_BASE
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results by market
    market_dir = output_dir / 'grid_results_by_market'
    market_dir.mkdir(parents=True, exist_ok=True)
    
    for market in df_agg['market'].unique():
        market_results = df_agg[df_agg['market'] == market].sort_values(['window', 'strategy'])
        market_file = market_dir / f'{market}_L1_L10.csv'
        market_results.to_csv(market_file, index=False)
    
    # Save summary files
    df_best.to_csv(output_dir / 'best_strategy_per_market.csv', index=False)
    df_agg.to_csv(output_dir / 'all_grid_results.csv', index=False)
    
    print(f"\n💾 Results saved to: {output_dir}")
    print(f"   - Individual market files: {market_dir}")
    print(f"   - Best per market: best_strategy_per_market.csv")
    print(f"   - All results: all_grid_results.csv")
    
    # Create visualizations
    if not args.no_viz:
        create_heatmap(df_agg, output_dir)
        create_summary_plot(df_best, output_dir)
    
    print(f"\n✅ Grid search complete!")


if __name__ == '__main__':
    main()
