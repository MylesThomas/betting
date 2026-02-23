"""
NBA Props Expected Value Analysis - All Markets

Tests both MOMENTUM and MEAN REVERSION strategies across all prop markets
using actual betting odds to calculate true ROI.

Context:
Previous analysis showed player_blocks has 67.7% accuracy with momentum signal.
BUT accuracy means nothing without odds! A 55% strategy at +120 beats 60% at -150.

This script:
1. Tests MOMENTUM strategy (bet Over when hot: 3+ overs in L5)
2. Tests MEAN REVERSION strategy (bet Under when hot: 3+ overs in L5)
3. Calculates actual EV and ROI using real odds from props data
4. Identifies which strategy + market combinations are profitable

Output:
- ~/Downloads/tmp/prop_predictive_power_analysis/ev_analysis/
  - momentum_strategy_results.csv (ROI for betting Over when hot)
  - mean_reversion_strategy_results.csv (ROI for betting Under when hot)
  - best_strategy_by_market.csv (Which strategy wins per market)
  - odds_distribution.csv (What odds we're actually getting)
  - player_breakdown_blocks.csv (Top players for blocks strategy)

Usage:
    # Single season
    python analysis/analyze_all_markets_odds_and_ev.py --season 2025-26
    
    # All 3 seasons (recommended - most robust)
    python analysis/analyze_all_markets_odds_and_ev.py --seasons 2023-24 2024-25 2025-26
    
    # Specific market only
    python analysis/analyze_all_markets_odds_and_ev.py --market player_blocks --seasons 2023-24 2024-25 2025-26

Author: Myles Thomas
Date: 2026-02-10
"""

import sys
import os
import logging
from pathlib import Path
import argparse
from io import StringIO
from datetime import datetime
import warnings

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import boto3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# S3 Configuration
S3_BUCKET_PROPS = 'the-odds-api-mt'
S3_BUCKET_NBA = 'nba-api-mt'

# Analysis Parameters
DEFAULT_WINDOW_SIZE = 5  # Look at last 5 games
DEFAULT_THRESHOLD = 3    # Need 3+ overs to trigger signal
DEFAULT_STAKE = 100      # Bet $100 per bet

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
OUTPUT_BASE = Path.home() / 'Downloads' / 'tmp' / 'prop_predictive_power_analysis' / 'ev_analysis'

# =============================================================================
# ODDS CONVERSION FUNCTIONS
# =============================================================================

def american_to_decimal(american_odds):
    """Convert American odds to decimal odds"""
    if pd.isna(american_odds) or american_odds == 0:
        return None
    
    if american_odds > 0:
        return 1 + (american_odds / 100)
    else:
        return 1 + (100 / abs(american_odds))


def american_to_implied_prob(american_odds):
    """Convert American odds to implied probability"""
    if pd.isna(american_odds):
        return None
    
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)


def calculate_ev(win_prob, odds, stake=100):
    """
    Calculate expected value for a bet
    
    Args:
        win_prob: Probability of winning (0-1)
        odds: American odds
        stake: Bet amount
    
    Returns:
        Expected value in dollars
    """
    if pd.isna(odds):
        return None
    
    decimal_odds = american_to_decimal(odds)
    profit_if_win = (decimal_odds - 1) * stake
    loss_if_lose = stake
    
    ev = (win_prob * profit_if_win) - ((1 - win_prob) * loss_if_lose)
    return ev


# =============================================================================
# DATA LOADING (Reuse from stability script)
# =============================================================================

def load_props_with_actuals(season):
    """
    Load props data with actuals from the intermediate files
    created by analyze_nba_props_stability_and_predictive_power.py
    """
    data_file = OUTPUT_BASE.parent / season / '01_data' / 'props_with_actuals.csv'
    
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        print(f"   Run this first: python analysis/analyze_nba_props_stability_and_predictive_power.py --season {season}")
        return pd.DataFrame()
    
    print(f"📊 Loading data from {data_file}...")
    df = pd.read_csv(data_file)
    
    print(f"   ✅ Loaded {len(df):,} rows")
    print(f"      Markets: {df['market'].nunique()}")
    print(f"      Players: {df['player_normalized'].nunique()}")
    
    return df


# =============================================================================
# STRATEGY TESTING FUNCTIONS
# =============================================================================

def create_momentum_signals(df, market, window_size=5, threshold=3):
    """
    Create momentum signals for a specific market
    
    Args:
        df: DataFrame with props and actuals
        market: Market to analyze
        window_size: Number of games to look back
        threshold: Number of overs needed to trigger signal
    
    Returns:
        DataFrame with signal indicators
    """
    # Filter to this market
    df_market = df[df['market'] == market].copy()
    
    # Sort by player and date
    df_market = df_market.sort_values(['player_normalized', 'game_date'])
    
    # Calculate rolling overs
    df_market['over_L5'] = df_market.groupby('player_normalized')['beat_line'].rolling(
        window=window_size, min_periods=1
    ).sum().reset_index(level=0, drop=True)
    
    # Shift to avoid lookahead bias
    df_market['over_L5_lag'] = df_market.groupby('player_normalized')['over_L5'].shift(1)
    
    # Create signal
    df_market['momentum_signal'] = (df_market['over_L5_lag'] >= threshold).astype(int)
    
    # Need at least window_size games of history
    df_market = df_market.dropna(subset=['over_L5_lag'])
    
    return df_market


def test_momentum_strategy(df_signals, stake=100):
    """
    Test momentum strategy: Bet Over when signal fires
    
    Args:
        df_signals: DataFrame with momentum signals
        stake: Bet amount
    
    Returns:
        Dictionary with results
    """
    # Filter to signal situations
    df_bets = df_signals[df_signals['momentum_signal'] == 1].copy()
    
    if len(df_bets) == 0:
        return {
            'num_bets': 0,
            'win_rate': None,
            'avg_over_odds': None,
            'total_ev': None,
            'roi': None
        }
    
    # Strategy: Bet Over
    df_bets['bet_side'] = 'Over'
    df_bets['bet_odds'] = df_bets['over_odds']
    df_bets['win'] = (df_bets['actual_value'] > df_bets['prop_line']).astype(int)
    
    # Filter to VALID American odds only (exclude corrupted data)
    # Valid odds: <= -105 OR >= 100 (standard American odds format)
    df_bets = df_bets[
        (df_bets['bet_odds'] <= -105) | (df_bets['bet_odds'] >= 100)
    ].copy()
    
    # Calculate profit for each bet
    df_bets['decimal_odds'] = df_bets['bet_odds'].apply(american_to_decimal)
    df_bets['profit'] = df_bets.apply(
        lambda row: (row['decimal_odds'] - 1) * stake if row['win'] else -stake,
        axis=1
    )
    
    # Remove bets with missing odds
    df_bets = df_bets.dropna(subset=['bet_odds', 'profit'])
    
    if len(df_bets) == 0:
        return {
            'num_bets': 0,
            'win_rate': None,
            'avg_over_odds': None,
            'total_ev': None,
            'roi': None
        }
    
    # Calculate metrics
    num_bets = len(df_bets)
    win_rate = df_bets['win'].mean()
    
    # Calculate average odds via implied probability (mathematically correct)
    df_bets['implied_prob'] = df_bets['bet_odds'].apply(american_to_implied_prob)
    avg_implied_prob = df_bets['implied_prob'].mean()
    
    # For display, show median odds (more representative than mean)
    median_over_odds = df_bets['bet_odds'].median()
    
    total_profit = df_bets['profit'].sum()
    total_wagered = num_bets * stake
    roi = (total_profit / total_wagered) * 100
    
    return {
        'num_bets': num_bets,
        'win_rate': win_rate,
        'avg_implied_prob': avg_implied_prob,
        'median_over_odds': median_over_odds,
        'total_profit': total_profit,
        'total_wagered': total_wagered,
        'roi': roi,
        'bets_df': df_bets
    }


def test_mean_reversion_strategy(df_signals, stake=100):
    """
    Test mean reversion strategy: Bet Under when signal fires
    
    Args:
        df_signals: DataFrame with momentum signals
        stake: Bet amount
    
    Returns:
        Dictionary with results
    """
    # Filter to signal situations (same signal, opposite bet)
    df_bets = df_signals[df_signals['momentum_signal'] == 1].copy()
    
    if len(df_bets) == 0:
        return {
            'num_bets': 0,
            'win_rate': None,
            'avg_under_odds': None,
            'total_ev': None,
            'roi': None
        }
    
    # Strategy: Bet Under (fade the streak)
    df_bets['bet_side'] = 'Under'
    df_bets['bet_odds'] = df_bets['under_odds']
    df_bets['win'] = (df_bets['actual_value'] < df_bets['prop_line']).astype(int)
    
    # Filter to VALID American odds only (exclude corrupted data)
    # Valid odds: <= -105 OR >= 100 (standard American odds format)
    df_bets = df_bets[
        (df_bets['bet_odds'] <= -105) | (df_bets['bet_odds'] >= 100)
    ].copy()
    
    # Calculate profit for each bet
    df_bets['decimal_odds'] = df_bets['bet_odds'].apply(american_to_decimal)
    df_bets['profit'] = df_bets.apply(
        lambda row: (row['decimal_odds'] - 1) * stake if row['win'] else -stake,
        axis=1
    )
    
    # Remove bets with missing odds
    df_bets = df_bets.dropna(subset=['bet_odds', 'profit'])
    
    if len(df_bets) == 0:
        return {
            'num_bets': 0,
            'win_rate': None,
            'avg_under_odds': None,
            'total_ev': None,
            'roi': None
        }
    
    # Calculate metrics
    num_bets = len(df_bets)
    win_rate = df_bets['win'].mean()
    
    # Calculate average odds via implied probability (mathematically correct)
    df_bets['implied_prob'] = df_bets['bet_odds'].apply(american_to_implied_prob)
    avg_implied_prob = df_bets['implied_prob'].mean()
    
    # For display, show median odds (more representative than mean)
    median_under_odds = df_bets['bet_odds'].median()
    
    total_profit = df_bets['profit'].sum()
    total_wagered = num_bets * stake
    roi = (total_profit / total_wagered) * 100
    
    return {
        'num_bets': num_bets,
        'win_rate': win_rate,
        'avg_implied_prob': avg_implied_prob,
        'median_under_odds': median_under_odds,
        'total_profit': total_profit,
        'total_wagered': total_wagered,
        'roi': roi,
        'bets_df': df_bets
    }


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_all_markets(df, markets, window_size=5, threshold=3, stake=100):
    """
    Analyze both strategies for all markets
    
    Args:
        df: DataFrame with props and actuals
        markets: List of markets to analyze
        window_size: Rolling window size
        threshold: Signal threshold
        stake: Bet size
    
    Returns:
        Tuple of (momentum_results, mean_reversion_results)
    """
    print(f"\n{'='*80}")
    print(f"TESTING BOTH STRATEGIES FOR ALL MARKETS")
    print(f"{'='*80}")
    print(f"Window: {window_size} games, Threshold: {threshold}+ overs, Stake: ${stake}")
    
    momentum_results = []
    mean_reversion_results = []
    
    for market in markets:
        print(f"\n📊 Analyzing {market}...")
        
        # Create signals
        df_signals = create_momentum_signals(df, market, window_size, threshold)
        
        if df_signals.empty:
            print(f"   ⚠️  Not enough data for {market}")
            continue
        
        print(f"   Signal fires: {df_signals['momentum_signal'].sum():,} times")
        
        # Test momentum strategy (bet Over when hot)
        momentum = test_momentum_strategy(df_signals, stake)
        
        if momentum['num_bets'] > 0:
            print(f"   MOMENTUM (Bet Over when hot):")
            print(f"      Bets: {momentum['num_bets']:,}")
            print(f"      Win Rate: {momentum['win_rate']*100:.1f}%")
            print(f"      Avg Implied Prob: {momentum['avg_implied_prob']*100:.1f}%")
            print(f"      Median Over Odds: {momentum['median_over_odds']:.0f}")
            print(f"      ROI: {momentum['roi']:.1f}%")
            
            momentum_results.append({
                'market': market,
                'strategy': 'MOMENTUM',
                'num_bets': momentum['num_bets'],
                'win_rate': momentum['win_rate'],
                'avg_implied_prob': momentum['avg_implied_prob'],
                'median_odds': momentum['median_over_odds'],
                'total_profit': momentum['total_profit'],
                'total_wagered': momentum['total_wagered'],
                'roi': momentum['roi']
            })
        
        # Test mean reversion strategy (bet Under when hot)
        mean_rev = test_mean_reversion_strategy(df_signals, stake)
        
        if mean_rev['num_bets'] > 0:
            print(f"   MEAN REVERSION (Bet Under when hot):")
            print(f"      Bets: {mean_rev['num_bets']:,}")
            print(f"      Win Rate: {mean_rev['win_rate']*100:.1f}%")
            print(f"      Avg Implied Prob: {mean_rev['avg_implied_prob']*100:.1f}%")
            print(f"      Median Under Odds: {mean_rev['median_under_odds']:.0f}")
            print(f"      ROI: {mean_rev['roi']:.1f}%")
            
            mean_reversion_results.append({
                'market': market,
                'strategy': 'MEAN_REVERSION',
                'num_bets': mean_rev['num_bets'],
                'win_rate': mean_rev['win_rate'],
                'avg_implied_prob': mean_rev['avg_implied_prob'],
                'median_odds': mean_rev['median_under_odds'],
                'total_profit': mean_rev['total_profit'],
                'total_wagered': mean_rev['total_wagered'],
                'roi': mean_rev['roi']
            })
    
    df_momentum = pd.DataFrame(momentum_results)
    df_mean_rev = pd.DataFrame(mean_reversion_results)
    
    return df_momentum, df_mean_rev


def combine_and_rank_strategies(df_momentum, df_mean_rev):
    """
    Combine both strategies and identify best per market
    
    Args:
        df_momentum: Momentum strategy results
        df_mean_rev: Mean reversion strategy results
    
    Returns:
        DataFrame with best strategy per market
    """
    # Combine both
    df_all = pd.concat([df_momentum, df_mean_rev], ignore_index=True)
    
    # For each market, pick best ROI strategy
    best_strategies = []
    
    for market in df_all['market'].unique():
        market_strategies = df_all[df_all['market'] == market].copy()
        
        # Pick highest ROI
        best = market_strategies.loc[market_strategies['roi'].idxmax()]
        
        best_strategies.append({
            'market': market,
            'best_strategy': best['strategy'],
            'num_bets': best['num_bets'],
            'win_rate': best['win_rate'],
            'avg_implied_prob': best['avg_implied_prob'],
            'median_odds': best['median_odds'],
            'roi': best['roi'],
            'profitable': best['roi'] > 0
        })
    
    df_best = pd.DataFrame(best_strategies)
    df_best = df_best.sort_values('roi', ascending=False)
    
    return df_best


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_visualizations(df_momentum, df_mean_rev, df_best, output_dir):
    """Create visualizations comparing strategies"""
    print(f"\n{'='*80}")
    print(f"CREATING VISUALIZATIONS")
    print(f"{'='*80}")
    
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. ROI Comparison
    print(f"\n📊 Creating ROI comparison chart...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    df_all = pd.concat([df_momentum, df_mean_rev])
    df_all = df_all.sort_values(['market', 'roi'], ascending=[True, False])
    
    markets = df_all['market'].unique()
    x = np.arange(len(markets))
    width = 0.35
    
    momentum_rois = []
    mean_rev_rois = []
    
    for market in markets:
        momentum_roi = df_momentum[df_momentum['market'] == market]['roi'].values
        mean_rev_roi = df_mean_rev[df_mean_rev['market'] == market]['roi'].values
        
        momentum_rois.append(momentum_roi[0] if len(momentum_roi) > 0 else 0)
        mean_rev_rois.append(mean_rev_roi[0] if len(mean_rev_roi) > 0 else 0)
    
    bars1 = ax.bar(x - width/2, momentum_rois, width, label='Momentum (Bet Over)', color='green', alpha=0.7)
    bars2 = ax.bar(x + width/2, mean_rev_rois, width, label='Mean Reversion (Bet Under)', color='red', alpha=0.7)
    
    ax.set_xlabel('Market', fontsize=12, fontweight='bold')
    ax.set_ylabel('ROI (%)', fontsize=12, fontweight='bold')
    ax.set_title('ROI Comparison: Momentum vs Mean Reversion Strategies', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(markets, rotation=45, ha='right')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=5, color='blue', linestyle='--', alpha=0.3, label='5% ROI Target')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'roi_comparison.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {viz_dir / 'roi_comparison.png'}")
    plt.close()
    
    # 2. Best Strategy by Market
    print(f"\n📊 Creating best strategy chart...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    df_best_sorted = df_best.sort_values('roi', ascending=True)
    markets = df_best_sorted['market'].values
    rois = df_best_sorted['roi'].values
    strategies = df_best_sorted['best_strategy'].values
    
    colors = ['green' if s == 'MOMENTUM' else 'red' for s in strategies]
    
    ax.barh(markets, rois, color=colors, alpha=0.7)
    ax.set_xlabel('ROI (%)', fontsize=12, fontweight='bold')
    ax.set_title('Best Strategy by Market (Highest ROI)', fontsize=14, fontweight='bold', pad=20)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.axvline(x=5, color='blue', linestyle='--', alpha=0.3, label='5% ROI Target')
    
    # Add ROI labels
    for i, (market, roi) in enumerate(zip(markets, rois)):
        ax.text(roi + 0.5, i, f'{roi:.1f}%', va='center', fontsize=9)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Momentum (Bet Over)'),
        Patch(facecolor='red', alpha=0.7, label='Mean Reversion (Bet Under)')
    ]
    ax.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'best_strategy_by_market.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {viz_dir / 'best_strategy_by_market.png'}")
    plt.close()
    
    print(f"\n✅ All visualizations created!")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Analyze expected value and ROI for NBA prop strategies'
    )
    parser.add_argument('--season', type=str, default='2025-26',
                        help='NBA season (e.g., 2025-26)')
    parser.add_argument('--seasons', type=str, nargs='+',
                        help='Multiple NBA seasons (e.g., 2023-24 2024-25 2025-26)')
    parser.add_argument('--market', type=str,
                        help='Specific market to analyze (default: all)')
    parser.add_argument('--window', type=int, default=5,
                        help='Rolling window size (default: 5)')
    parser.add_argument('--threshold', type=int, default=3,
                        help='Signal threshold (default: 3)')
    parser.add_argument('--stake', type=int, default=100,
                        help='Bet size in dollars (default: 100)')
    parser.add_argument('--no-viz', action='store_true',
                        help='Skip visualization generation')
    
    args = parser.parse_args()
    
    # Determine seasons to analyze
    seasons = args.seasons if args.seasons else [args.season]
    
    # Determine markets to analyze
    markets = [args.market] if args.market else ALL_MARKETS
    
    print(f"\n{'='*80}")
    print(f"NBA PROPS EXPECTED VALUE ANALYSIS")
    print(f"{'='*80}")
    print(f"Seasons: {', '.join(seasons)}")
    print(f"Markets: {len(markets)}")
    print(f"Window: {args.window} games")
    print(f"Threshold: {args.threshold}+ overs")
    print(f"Stake: ${args.stake} per bet")
    
    # Process each season
    all_momentum_results = []
    all_mean_rev_results = []
    
    for season in seasons:
        print(f"\n\n{'#'*80}")
        print(f"# PROCESSING SEASON: {season}")
        print(f"{'#'*80}")
        
        # Load data
        df = load_props_with_actuals(season)
        
        if df.empty:
            print(f"\n❌ Skipping {season} - no data")
            continue
        
        # Analyze
        df_momentum, df_mean_rev = analyze_all_markets(
            df, markets, args.window, args.threshold, args.stake
        )
        
        # Add season column
        if not df_momentum.empty:
            df_momentum['season'] = season
            all_momentum_results.append(df_momentum)
        
        if not df_mean_rev.empty:
            df_mean_rev['season'] = season
            all_mean_rev_results.append(df_mean_rev)
    
    # Combine all seasons
    if not all_momentum_results:
        print(f"\n❌ No results to analyze")
        return
    
    df_momentum_all = pd.concat(all_momentum_results, ignore_index=True)
    df_mean_rev_all = pd.concat(all_mean_rev_results, ignore_index=True)
    
    # Aggregate across seasons
    df_momentum_agg = df_momentum_all.groupby('market').agg({
        'num_bets': 'sum',
        'win_rate': 'mean',
        'avg_implied_prob': 'mean',
        'median_odds': 'median',
        'total_profit': 'sum',
        'total_wagered': 'sum'
    }).reset_index()
    df_momentum_agg['roi'] = (df_momentum_agg['total_profit'] / df_momentum_agg['total_wagered']) * 100
    df_momentum_agg['strategy'] = 'MOMENTUM'
    
    df_mean_rev_agg = df_mean_rev_all.groupby('market').agg({
        'num_bets': 'sum',
        'win_rate': 'mean',
        'avg_implied_prob': 'mean',
        'median_odds': 'median',
        'total_profit': 'sum',
        'total_wagered': 'sum'
    }).reset_index()
    df_mean_rev_agg['roi'] = (df_mean_rev_agg['total_profit'] / df_mean_rev_agg['total_wagered']) * 100
    df_mean_rev_agg['strategy'] = 'MEAN_REVERSION'
    
    # Find best strategy per market
    df_best = combine_and_rank_strategies(df_momentum_agg, df_mean_rev_agg)
    
    # Display results
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS - AGGREGATED ACROSS ALL SEASONS")
    print(f"{'='*80}")
    
    # Format for display
    df_momentum_display = df_momentum_agg.copy()
    df_momentum_display['win_rate'] = (df_momentum_display['win_rate'] * 100).round(1)
    df_momentum_display['avg_implied_prob'] = (df_momentum_display['avg_implied_prob'] * 100).round(1)
    df_momentum_display['roi'] = df_momentum_display['roi'].round(1)
    
    df_mean_rev_display = df_mean_rev_agg.copy()
    df_mean_rev_display['win_rate'] = (df_mean_rev_display['win_rate'] * 100).round(1)
    df_mean_rev_display['avg_implied_prob'] = (df_mean_rev_display['avg_implied_prob'] * 100).round(1)
    df_mean_rev_display['roi'] = df_mean_rev_display['roi'].round(1)
    
    df_best_display = df_best.copy()
    df_best_display['win_rate'] = (df_best_display['win_rate'] * 100).round(1)
    df_best_display['avg_implied_prob'] = (df_best_display['avg_implied_prob'] * 100).round(1)
    df_best_display['roi'] = df_best_display['roi'].round(1)
    
    print(f"\n1. MOMENTUM STRATEGY (Bet Over when hot):")
    print(df_momentum_display[['market', 'num_bets', 'win_rate', 'avg_implied_prob', 'median_odds', 'roi']].to_string(index=False))
    
    print(f"\n2. MEAN REVERSION STRATEGY (Bet Under when hot):")
    print(df_mean_rev_display[['market', 'num_bets', 'win_rate', 'avg_implied_prob', 'median_odds', 'roi']].to_string(index=False))
    
    print(f"\n3. BEST STRATEGY PER MARKET:")
    print(df_best_display.to_string(index=False))
    
    print(f"\n4. PROFITABLE MARKETS (ROI > 0):")
    df_profitable = df_best[df_best['profitable']].copy()
    if not df_profitable.empty:
        df_profitable_display = df_profitable.copy()
        df_profitable_display['win_rate'] = (df_profitable_display['win_rate'] * 100).round(1)
        df_profitable_display['roi'] = df_profitable_display['roi'].round(1)
        print(df_profitable_display[['market', 'best_strategy', 'win_rate', 'roi', 'num_bets']].to_string(index=False))
    else:
        print("   ❌ No profitable markets found")
    
    # Save results
    output_dir = OUTPUT_BASE
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df_momentum_agg.to_csv(output_dir / 'momentum_strategy_results.csv', index=False)
    df_mean_rev_agg.to_csv(output_dir / 'mean_reversion_strategy_results.csv', index=False)
    df_best.to_csv(output_dir / 'best_strategy_by_market.csv', index=False)
    
    # Save season-by-season detail
    df_momentum_all.to_csv(output_dir / 'momentum_by_season.csv', index=False)
    df_mean_rev_all.to_csv(output_dir / 'mean_reversion_by_season.csv', index=False)
    
    print(f"\n💾 Results saved to: {output_dir}")
    
    # Create visualizations
    if not args.no_viz:
        create_visualizations(df_momentum_agg, df_mean_rev_agg, df_best, output_dir)
    
    print(f"\n✅ Analysis complete!")
    print(f"📁 All files saved to: {output_dir}")


if __name__ == '__main__':
    main()
