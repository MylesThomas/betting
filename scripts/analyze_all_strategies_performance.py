"""
Analyze All NBA Points Props Strategies Performance

Context:
- User requested analysis of which strategies have the best edge/ROI over last 3 seasons
- Seasons: 2023-24, 2024-25, 2025-26
- Need a consistent script in scripts/ that can be run weekly to track performance
- Should analyze BOTH 2d and 3d strategies across all line tiers, spread bins, scorer types

This script:
1. Loads all backtest data from S3 for 2d and 3d strategies
2. Groups by strategy parameters (line_tier, spread_bin, bet_side, scorer_type)
3. Calculates edge metrics: ROI, total profit, win rate, consistency
4. Ranks strategies by overall performance
5. Outputs ranked results with season-by-season breakdowns

Output:
- JSON file with all strategies ranked by ROI/edge
- CSV with detailed season-by-season performance
- Summary report showing top 10 strategies

Author: Myles Thomas
Date: 2026-01-25
"""

import pandas as pd
import boto3
import json
import argparse
from io import StringIO
from typing import List, Dict, Tuple
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# =============================================================================
# GLOBAL CONSTANTS - Strategy Filtering Thresholds
# =============================================================================

BREAKEVEN_WIN_RATE = 52.38  # Win rate needed to beat -110 juice
MIN_ROI_FOR_EDGE = 0.01     # Minimum ROI to consider a strategy has edge (0.01%)
MIN_PLAYS_DEFAULT = 50      # Default minimum plays to include strategy
MIN_BETS_FOR_KELLY = 20     # Minimum bets before using Kelly (use flat bet until then)

# Paper Trading Defaults
DEFAULT_PAPER_TRADE_PLAYS = 0      # Default: start betting immediately
DEFAULT_MIN_EDGE_THRESHOLD = 52.38  # Default: just need to beat breakeven


def load_backtest_plays(s3_client, bucket: str, strategy_type: str, season: str) -> pd.DataFrame:
    """
    Load backtest plays CSV for a given season and strategy type.
    
    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        strategy_type: '2d' or '3d'
        season: Season string like '2023-24'
    
    Returns:
        DataFrame of plays
    """
    s3_key = f'data/04_output/backtests/{strategy_type}/{season}/plays.csv'
    
    try:
        response = s3_client.get_object(Bucket=bucket, Key=s3_key)
        df = pd.read_csv(StringIO(response['Body'].read().decode('utf-8')))
        df['season'] = season
        df['strategy_type'] = strategy_type
        return df
    except Exception as e:
        print(f"⚠️  Could not load {season} {strategy_type.upper()}: {e}")
        return pd.DataFrame()


def load_training_strategies(s3_client, bucket: str, strategy_type: str) -> Dict:
    """
    Load training ROI data from 2025-26 training strategies.
    
    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        strategy_type: '2d' or '3d'
    
    Returns:
        Dict mapping strategy key to training ROI
    """
    if strategy_type == '2d':
        s3_key = 'data/03_intermediate/points_by_role_gamespread_strategies_2025-26.json'
    else:
        s3_key = 'data/03_intermediate/points_by_role_gamespread_6feet_strategies_2025-26_rim40.json'
    
    try:
        response = s3_client.get_object(Bucket=bucket, Key=s3_key)
        data = json.loads(response['Body'].read().decode('utf-8'))
        
        if 'strategies' in data:
            strategies = data['strategies']
        else:
            strategies = data
        
        if isinstance(strategies, dict):
            strategies = list(strategies.values())
        
        # Build lookup dict
        training_roi = {}
        for strat in strategies:
            if strategy_type == '2d':
                key = (
                    strat['line_tier'],
                    strat['spread_bin'],
                    strat['bet_side']
                )
            else:
                key = (
                    strat['line_tier'],
                    strat['spread_bin'],
                    strat['bet_side'],
                    strat.get('scorer_type', 'N/A')
                )
            training_roi[key] = strat.get('roi', 0.0)
        
        return training_roi
        
    except Exception as e:
        print(f"⚠️  Could not load training data for {strategy_type}: {e}")
        return {}


def calculate_strategy_stats(df: pd.DataFrame, seasons: List[str]) -> Dict:
    """
    Calculate comprehensive statistics for a strategy.
    
    Args:
        df: DataFrame of plays for this strategy
        seasons: List of seasons analyzed
    
    Returns:
        Dict with strategy stats
    """
    stats = {}
    
    # Overall stats
    total_plays = len(df)
    if total_plays == 0:
        return None
    
    total_wins = (df['result'] == 'WIN').sum()
    total_losses = (df['result'] == 'LOSS').sum()
    total_profit = df['profit'].sum()
    win_rate = (total_wins / (total_wins + total_losses) * 100) if (total_wins + total_losses) > 0 else 0
    
    # ROI calculation (assuming $100 bets)
    total_wagered = total_plays * 100
    roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0
    
    # Season-by-season stats
    season_stats = {}
    profitable_seasons = 0
    
    for season in seasons:
        season_df = df[df['season'] == season]
        if len(season_df) == 0:
            season_stats[season] = {
                'plays': int(0),
                'profit': float(0.0),
                'win_rate': float(0.0),
                'roi': float(0.0),
                'profitable': bool(False)
            }
        else:
            season_wins = (season_df['result'] == 'WIN').sum()
            season_losses = (season_df['result'] == 'LOSS').sum()
            season_profit = season_df['profit'].sum()
            season_plays = len(season_df)
            season_win_rate = (season_wins / (season_wins + season_losses) * 100) if (season_wins + season_losses) > 0 else 0
            season_wagered = season_plays * 100
            season_roi = (season_profit / season_wagered * 100) if season_wagered > 0 else 0
            
            season_stats[season] = {
                'plays': int(season_plays),
                'profit': round(float(season_profit), 2),
                'win_rate': round(float(season_win_rate), 1),
                'roi': round(float(season_roi), 1),
                'profitable': bool(season_profit > 0)
            }
            
            if season_profit > 0:
                profitable_seasons += 1
    
    # Consistency score: how many seasons were profitable
    consistency = profitable_seasons / len(seasons)
    
    return {
        'total_plays': int(total_plays),
        'total_profit': round(float(total_profit), 2),
        'win_rate': round(float(win_rate), 1),
        'roi': round(float(roi), 1),
        'profitable_seasons': profitable_seasons,
        'consistency': round(consistency, 2),
        'season_stats': season_stats
    }


def analyze_all_strategies(
    strategy_types: List[str] = ['2d', '3d'],
    seasons: List[str] = ['2023-24', '2024-25', '2025-26'],
    min_plays: int = 50
) -> Tuple[List[Dict], pd.DataFrame]:
    """
    Analyze all strategies across seasons.
    
    Args:
        strategy_types: List of strategy types to analyze
        seasons: List of seasons to analyze
        min_plays: Minimum total plays required to include strategy
    
    Returns:
        Tuple of (list of strategy dicts, detailed DataFrame)
    """
    print("="*80)
    print("NBA POINTS PROPS STRATEGY PERFORMANCE ANALYSIS")
    print("="*80)
    print(f"Strategy Types: {', '.join(strategy_types)}")
    print(f"Seasons: {', '.join(seasons)}")
    print(f"Min Plays Filter: {min_plays}")
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Setup S3
    s3_client = boto3.client('s3')
    bucket = 'nba-betting-mt'
    
    # Load all backtest data
    all_plays = []
    
    for strategy_type in strategy_types:
        print(f"\nLoading {strategy_type.upper()} backtest data...")
        
        # Load training ROI
        training_roi = load_training_strategies(s3_client, bucket, strategy_type)
        print(f"  ✅ Loaded {len(training_roi)} training ROI values")
        
        for season in seasons:
            df = load_backtest_plays(s3_client, bucket, strategy_type, season)
            if len(df) > 0:
                print(f"  ✅ {season}: {len(df):,} plays")
                all_plays.append(df)
    
    if not all_plays:
        print("\n❌ No backtest data found!")
        return [], pd.DataFrame()
    
    # Combine all plays
    df_all = pd.concat(all_plays, ignore_index=True)
    print(f"\n✅ Total plays loaded: {len(df_all):,}")
    
    # Group by strategy parameters
    print("\nGrouping by strategy parameters...")
    
    strategies = []
    
    for strategy_type in strategy_types:
        df_type = df_all[df_all['strategy_type'] == strategy_type]
        
        # Load training ROI for this type
        training_roi = load_training_strategies(s3_client, bucket, strategy_type)
        
        if strategy_type == '2d':
            group_cols = ['line_tier', 'spread_bin', 'bet_side']
        else:
            group_cols = ['line_tier', 'spread_bin', 'bet_side', 'scorer_type']
        
        grouped = df_type.groupby(group_cols, dropna=False)
        
        for group_key, group_df in grouped:
            # Calculate stats
            stats = calculate_strategy_stats(group_df, seasons)
            
            if stats is None or stats['total_plays'] < min_plays:
                continue
            
            # Build strategy dict
            if strategy_type == '2d':
                line_tier, spread_bin, bet_side = group_key
                scorer_type = 'N/A'
                training_key = (line_tier, spread_bin, bet_side)
            else:
                line_tier, spread_bin, bet_side, scorer_type = group_key
                training_key = (line_tier, spread_bin, bet_side, scorer_type)
            
            # Generate strategy name with Fav/Dog distinction
            tier_slug = line_tier.split(' ')[0].replace('-', '_').lower()
            
            # Extract spread value and Fav/Dog
            spread_parts = spread_bin.split(' ')
            spread_value = spread_parts[0].replace('-', '_').replace("'", '').lower()
            spread_type = ''
            if len(spread_parts) > 1:
                if 'Fav' in spread_bin:
                    spread_type = '_fav'
                elif 'Dog' in spread_bin:
                    spread_type = '_dog'
            spread_slug = f"{spread_value}{spread_type}"
            
            bet_slug = bet_side.lower()
            
            if strategy_type == '3d' and scorer_type and scorer_type != 'N/A':
                scorer_slug = 'rim' if 'Rim' in scorer_type else 'perimeter'
                strategy_name = f"{tier_slug}_{spread_slug}_{scorer_slug}_{bet_slug}"
            else:
                strategy_name = f"{tier_slug}_{spread_slug}_{bet_slug}"
            
            # Get training ROI
            train_roi = training_roi.get(training_key, 0.0)
            
            strategy = {
                'strategy_name': strategy_name,
                'strategy_type': strategy_type,
                'line_tier': line_tier,
                'spread_bin': spread_bin,
                'bet_side': bet_side,
                'scorer_type': scorer_type,
                'training_roi': round(float(train_roi), 1),
                **stats
            }
            
            strategies.append(strategy)
    
    print(f"\n✅ Found {len(strategies)} unique strategies (min {min_plays} plays)")
    
    # Convert to DataFrame for analysis
    df_strategies = pd.DataFrame(strategies)
    
    # Sort by ROI descending
    df_strategies = df_strategies.sort_values('roi', ascending=False).reset_index(drop=True)
    
    return strategies, df_strategies


def print_top_strategies(df_strategies: pd.DataFrame, n: int = 10):
    """Print top N strategies by ROI."""
    print(f"\n{'='*80}")
    print(f"TOP {n} STRATEGIES BY ROI")
    print("="*80)
    print(f"{'Rank':<5} {'Strategy':<35} {'Type':<5} {'ROI':<8} {'Profit':<12} {'Plays':<7} {'Win%':<7} {'Prof/3':<8}")
    print("-"*80)
    
    for idx, row in df_strategies.head(n).iterrows():
        print(f"{idx+1:<5} {row['strategy_name']:<35} {row['strategy_type'].upper():<5} "
              f"{row['roi']:>6.1f}% ${row['total_profit']:>9,.0f} "
              f"{row['total_plays']:>6} {row['win_rate']:>5.1f}% "
              f"{row['profitable_seasons']}/3")


def print_summary_stats(df_strategies: pd.DataFrame):
    """Print summary statistics."""
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # Overall stats
    total_strategies = len(df_strategies)
    profitable_strategies = len(df_strategies[df_strategies['roi'] > 0])
    avg_roi = df_strategies['roi'].mean()
    median_roi = df_strategies['roi'].median()
    
    print(f"Total Strategies: {total_strategies}")
    print(f"Profitable Strategies: {profitable_strategies} ({profitable_strategies/total_strategies*100:.1f}%)")
    print(f"Average ROI: {avg_roi:+.1f}%")
    print(f"Median ROI: {median_roi:+.1f}%")
    
    # By strategy type
    print(f"\nBy Strategy Type:")
    for stype in df_strategies['strategy_type'].unique():
        df_type = df_strategies[df_strategies['strategy_type'] == stype]
        count = len(df_type)
        profitable = len(df_type[df_type['roi'] > 0])
        avg_roi_type = df_type['roi'].mean()
        print(f"  {stype.upper()}: {count} strategies | {profitable} profitable ({profitable/count*100:.1f}%) | Avg ROI: {avg_roi_type:+.1f}%")
    
    # By bet side
    print(f"\nBy Bet Side:")
    for side in df_strategies['bet_side'].unique():
        df_side = df_strategies[df_strategies['bet_side'] == side]
        count = len(df_side)
        profitable = len(df_side[df_side['roi'] > 0])
        avg_roi_side = df_side['roi'].mean()
        print(f"  {side}: {count} strategies | {profitable} profitable ({profitable/count*100:.1f}%) | Avg ROI: {avg_roi_side:+.1f}%")
    
    # Consistency analysis
    print(f"\nConsistency (Profitable Seasons):")
    for i in range(4):
        count = len(df_strategies[df_strategies['profitable_seasons'] == i])
        pct = (count / total_strategies * 100) if total_strategies > 0 else 0
        print(f"  {i}/3 seasons: {count} strategies ({pct:.1f}%)")


def save_results(strategies: List[Dict], df_strategies: pd.DataFrame, output_dir: str = 'data/04_output'):
    """Save results to JSON and CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save ranked strategies JSON
    json_path = f"{output_dir}/all_strategies_ranked_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump({
            'analysis_date': datetime.now().isoformat(),
            'seasons_analyzed': ['2023-24', '2024-25', '2025-26'],
            'total_strategies': len(strategies),
            'strategies': strategies
        }, f, indent=2)
    print(f"\n✅ Saved ranked strategies JSON: {json_path}")
    
    # Save detailed CSV
    csv_path = f"{output_dir}/all_strategies_ranked_{timestamp}.csv"
    
    # Flatten season stats for CSV
    df_export = df_strategies.copy()
    for season in ['2023-24', '2024-25', '2025-26']:
        season_key = season.replace('-', '_')
        df_export[f'{season_key}_plays'] = df_export['season_stats'].apply(lambda x: x.get(season, {}).get('plays', 0))
        df_export[f'{season_key}_profit'] = df_export['season_stats'].apply(lambda x: x.get(season, {}).get('profit', 0.0))
        df_export[f'{season_key}_roi'] = df_export['season_stats'].apply(lambda x: x.get(season, {}).get('roi', 0.0))
        df_export[f'{season_key}_win_rate'] = df_export['season_stats'].apply(lambda x: x.get(season, {}).get('win_rate', 0.0))
    
    df_export = df_export.drop(columns=['season_stats'])
    df_export.to_csv(csv_path, index=False)
    print(f"✅ Saved detailed CSV: {csv_path}")
    
    # Save latest version without timestamp for easy access
    json_latest = f"{output_dir}/all_strategies_ranked_latest.json"
    csv_latest = f"{output_dir}/all_strategies_ranked_latest.csv"
    
    with open(json_latest, 'w') as f:
        json.dump({
            'analysis_date': datetime.now().isoformat(),
            'seasons_analyzed': ['2023-24', '2024-25', '2025-26'],
            'total_strategies': len(strategies),
            'strategies': strategies
        }, f, indent=2)
    df_export.to_csv(csv_latest, index=False)
    
    print(f"✅ Saved latest versions: {json_latest}, {csv_latest}")


def calculate_kelly_fraction(win_rate: float, kelly_multiplier: float = 1.0) -> float:
    """
    Calculate Kelly criterion fraction for -110 odds.
    
    Args:
        win_rate: Win rate as decimal (e.g., 0.55 for 55%)
        kelly_multiplier: Multiplier (0.25=quarter, 0.5=half, 1.0=full)
    
    Returns:
        Kelly fraction to bet (0.0 if no edge)
    """
    b = 100 / 110  # Odds multiplier for -110
    p = win_rate
    q = 1 - p
    
    # Full Kelly fraction
    kelly_full = (p * b - q) / b if (p * b - q) > 0 else 0
    
    return kelly_full * kelly_multiplier


def simulate_kelly_betting(df_plays: pd.DataFrame, starting_bankroll: float, 
                          kelly_fraction: float, win_rate: float) -> pd.DataFrame:
    """
    Simulate Kelly criterion betting with STATIC edge (uses overall win rate).
    
    Args:
        df_plays: DataFrame with plays (must have 'result' and 'profit' columns)
        starting_bankroll: Starting bankroll amount
        kelly_fraction: Kelly fraction (1.0=full, 0.5=half, 0.25=quarter)
        win_rate: Win rate as decimal (e.g., 0.55 for 55%)
    
    Returns:
        DataFrame with added columns: bet_size, kelly_fraction_used, bankroll
    """
    df = df_plays.copy()
    
    # Calculate Kelly fraction based on win rate (STATIC - same for all bets)
    kelly_to_use = calculate_kelly_fraction(win_rate, kelly_fraction)
    
    # Simulate play-by-play
    bankroll = starting_bankroll
    bankrolls = []
    bet_sizes = []
    kelly_fractions_used = []
    
    for idx, row in df.iterrows():
        # Calculate bet size based on current bankroll and Kelly fraction
        bet_size = bankroll * kelly_to_use
        
        # Can't bet more than bankroll
        bet_size = min(bet_size, bankroll)
        
        # Record
        bet_sizes.append(bet_size)
        kelly_fractions_used.append(kelly_to_use)
        
        # Calculate actual profit based on bet size
        # The profit in the data is for a fixed unit (assume $100)
        # Scale it proportionally
        if row['result'] == 'WIN':
            # Win: get back (bet_size / 1.1) profit (for -110 odds)
            actual_profit = bet_size / 1.1
        elif row['result'] == 'LOSS':
            # Loss: lose the bet_size
            actual_profit = -bet_size
        else:
            actual_profit = 0
        
        # Update bankroll
        bankroll += actual_profit
        bankrolls.append(bankroll)
        
        # Prevent negative bankroll (bust)
        if bankroll <= 0:
            bankroll = 0
    
    df['bet_size'] = bet_sizes
    df['kelly_fraction_used'] = kelly_fractions_used
    df['bankroll'] = bankrolls
    df['play_number'] = range(1, len(df) + 1)
    
    return df


def simulate_kelly_betting_dynamic(df_plays: pd.DataFrame, starting_bankroll: float, 
                                   kelly_fraction: float, paper_trade_plays: int = 0,
                                   min_edge_threshold: float = 52.38) -> pd.DataFrame:
    """
    Simulate Kelly criterion betting with DYNAMIC edge calculation and optional paper trading.
    Recalculates win rate and Kelly % after each bet.
    
    Args:
        df_plays: DataFrame with plays (must have 'result' column, sorted chronologically)
        starting_bankroll: Starting bankroll amount
        kelly_fraction: Kelly fraction multiplier (1.0=full, 0.5=half, 0.25=quarter)
        paper_trade_plays: Number of plays to paper trade before betting (default: 0)
        min_edge_threshold: Minimum win rate % to start betting after paper trading (default: 52.38)
    
    Returns:
        DataFrame with added columns: bet_size, kelly_fraction_used, bankroll, running_win_rate, paper_trading
    """
    df = df_plays.copy()
    
    # Simulate play-by-play with dynamic edge
    bankroll = starting_bankroll
    running_wins = 0
    running_losses = 0
    
    bankrolls = []
    bet_sizes = []
    kelly_fractions_used = []
    running_win_rates = []
    paper_trading_flags = []
    
    for idx, row in df.iterrows():
        total_bets = running_wins + running_losses
        
        # Calculate current win rate from previous results
        if total_bets > 0:
            current_win_rate = running_wins / total_bets
        else:
            current_win_rate = 0.5
        
        # Determine if we're paper trading
        is_paper_trading = total_bets < paper_trade_plays
        
        # Determine if we should bet
        if is_paper_trading:
            # Paper trading period - no betting
            kelly_to_use = 0.0
            bet_size = 0.0
        elif total_bets < paper_trade_plays + MIN_BETS_FOR_KELLY:
            # Just finished paper trading, need more data for Kelly
            # Check if we have edge
            if current_win_rate * 100 >= min_edge_threshold:
                kelly_to_use = 0.01  # Small flat bet
                bet_size = bankroll * kelly_to_use
            else:
                kelly_to_use = 0.0
                bet_size = 0.0
        else:
            # Have enough data, use Kelly
            if current_win_rate * 100 >= min_edge_threshold:
                kelly_to_use = calculate_kelly_fraction(current_win_rate, kelly_fraction)
                bet_size = bankroll * kelly_to_use
            else:
                kelly_to_use = 0.0
                bet_size = 0.0
        
        bet_size = min(bet_size, bankroll)  # Can't bet more than bankroll
        
        # Record before updating
        bet_sizes.append(bet_size)
        kelly_fractions_used.append(kelly_to_use)
        running_win_rates.append(current_win_rate)
        paper_trading_flags.append(is_paper_trading)
        
        # Calculate actual profit/loss
        if row['result'] == 'WIN':
            if bet_size > 0:
                actual_profit = bet_size / 1.1  # Win at -110 odds
            else:
                actual_profit = 0  # Paper trading or no bet
            running_wins += 1
        elif row['result'] == 'LOSS':
            if bet_size > 0:
                actual_profit = -bet_size
            else:
                actual_profit = 0  # Paper trading or no bet
            running_losses += 1
        else:
            actual_profit = 0
        
        # Update bankroll
        bankroll += actual_profit
        bankrolls.append(max(bankroll, 0))  # Prevent negative
        
        if bankroll <= 0:
            bankroll = 0
    
    df['bet_size'] = bet_sizes
    df['kelly_fraction_used'] = kelly_fractions_used
    df['bankroll'] = bankrolls
    df['running_win_rate'] = running_win_rates
    df['paper_trading'] = paper_trading_flags
    df['play_number'] = range(1, len(df) + 1)
    
    return df


def filter_strategies_for_visualization(strategies_list: List[str], 
                                        df_strategies: pd.DataFrame,
                                        filter_mode: str,
                                        max_strats: int) -> List[str]:
    """
    Filter strategies based on specified criteria.
    
    Args:
        strategies_list: List of strategy names to filter from
        df_strategies: DataFrame with strategy stats
        filter_mode: 'edge', 'all', 'profitable', 'losing'
        max_strats: Maximum strategies to return
    
    Returns:
        Filtered list of strategy names
    """
    if filter_mode == 'all':
        return strategies_list[:max_strats]
    
    filtered = []
    
    for strategy in strategies_list:
        strategy_row = df_strategies[df_strategies['strategy_name'] == strategy].iloc[0]
        roi = strategy_row['roi']
        win_rate = strategy_row['win_rate']
        
        if filter_mode == 'edge':
            # Must have positive ROI AND beat breakeven win rate
            if roi >= MIN_ROI_FOR_EDGE and win_rate > BREAKEVEN_WIN_RATE:
                filtered.append(strategy)
        
        elif filter_mode == 'profitable':
            # Just positive ROI, don't care about win rate
            if roi >= MIN_ROI_FOR_EDGE:
                filtered.append(strategy)
        
        elif filter_mode == 'losing':
            # Negative ROI only
            if roi < 0:
                filtered.append(strategy)
    
    return filtered[:max_strats]


def create_visualizations(df_all_plays: pd.DataFrame, df_strategies: pd.DataFrame, 
                         output_dir: str = None, top_n: int = 10, show_plots: bool = True,
                         use_kelly: bool = False, starting_bankroll: float = 10000,
                         kelly_fraction: float = 0.5, max_viz_strats: int = 999,
                         viz_filter: str = 'edge', dynamic_edge: bool = False,
                         paper_trade_plays: int = 0, min_edge_threshold: float = 52.38):
    """
    Create visualizations of strategy performance.
    
    Args:
        df_all_plays: DataFrame with all plays across all strategies
        df_strategies: DataFrame with strategy-level stats
        output_dir: Directory to save plots (default: ~/Downloads/tmp)
        top_n: Number of top strategies to visualize
    """
    if output_dir is None:
        output_dir = str(Path.home() / 'Downloads' / 'tmp')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (14, 8)
    
    # Get top strategies
    top_strategies = df_strategies.head(top_n)['strategy_name'].tolist()
    
    print(f"\n{'='*80}")
    print(f"CREATING VISUALIZATIONS FOR TOP {top_n} STRATEGIES")
    print(f"Filter Mode: {viz_filter.upper()}")
    if use_kelly:
        edge_type = "DYNAMIC (recalculated each bet)" if dynamic_edge else "STATIC (overall win rate)"
        print(f"Kelly Criterion: {kelly_fraction}x Kelly | Edge Calculation: {edge_type}")
        print(f"Starting Bankroll: ${starting_bankroll:,.0f}")
        if paper_trade_plays > 0:
            print(f"Paper Trading: First {paper_trade_plays} plays (no betting)")
            print(f"Min Edge Threshold: {min_edge_threshold:.1f}% win rate to start betting")
    print("="*80)
    
    # Filter plays to top strategies
    df_top_plays = df_all_plays[df_all_plays['strategy_name'].isin(top_strategies)].copy()
    
    # Sort by date/time for chronological cumulative profit
    if 'game_date' in df_top_plays.columns:
        df_top_plays = df_top_plays.sort_values('game_date')
    
    # =========================================================================
    # 1. CUMULATIVE PROFIT OR BANKROLL GROWTH OVER TIME
    # =========================================================================
    if use_kelly:
        print("\n1. Creating Bankroll Growth Over Time chart (Kelly Criterion)...")
    else:
        print("\n1. Creating Cumulative Profit Over Time chart...")
    
    # Filter strategies based on mode
    strategies_to_plot = filter_strategies_for_visualization(
        top_strategies, 
        df_strategies, 
        viz_filter,
        max_viz_strats
    )
    
    # Build filter description for output
    filter_descriptions = {
        'edge': 'with Edge (ROI>0 & WinRate>52.38%)',
        'all': 'by ROI',
        'profitable': 'with ROI>0',
        'losing': 'with ROI<0'
    }
    filter_desc = filter_descriptions.get(viz_filter, '')
    
    print(f"   Strategies {filter_desc}: {len(strategies_to_plot)} | Plotting: {len(strategies_to_plot)}")
    
    if len(strategies_to_plot) == 0:
        print(f"   ⚠️  No strategies match filter '{viz_filter}', skipping growth chart")
        return
    
    # Log what we're about to plot
    print(f"\n   Strategies to plot:")
    for i, strat in enumerate(strategies_to_plot, 1):
        strat_row = df_strategies[df_strategies['strategy_name'] == strat].iloc[0]
        strat_plays = df_top_plays[df_top_plays['strategy_name'] == strat]
        print(f"     {i}. {strat}: ROI={strat_row['roi']:+.1f}%, WinRate={strat_row['win_rate']:.1f}%, Plays={len(strat_plays)}")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for strategy in strategies_to_plot:
        strategy_plays = df_top_plays[df_top_plays['strategy_name'] == strategy].copy()
        
        if len(strategy_plays) == 0:
            print(f"   ⚠️  Strategy '{strategy}' has no plays in dataset, skipping")
            continue
        
        if len(strategy_plays) == 0:
            continue
        
        if use_kelly:
            # Get strategy win rate (only needed for static Kelly)
            strategy_row = df_strategies[df_strategies['strategy_name'] == strategy].iloc[0]
            
            if dynamic_edge:
                # Dynamic Kelly - recalculates edge after each bet
                strategy_plays = simulate_kelly_betting_dynamic(
                    strategy_plays,
                    starting_bankroll,
                    kelly_fraction,
                    paper_trade_plays,
                    min_edge_threshold
                )
            else:
                # Static Kelly - uses overall win rate
                win_rate = strategy_row['win_rate'] / 100
                strategy_plays = simulate_kelly_betting(
                    strategy_plays,
                    starting_bankroll,
                    kelly_fraction,
                    win_rate
                )
            
            y_values = strategy_plays['bankroll']
            y_label = 'Bankroll ($)'
            edge_label = "Dynamic Edge" if dynamic_edge else "Static Edge"
            title = f'Bankroll Growth - {len(strategies_to_plot)} Strats {filter_desc} ({kelly_fraction}x Kelly, {edge_label})'
        else:
            strategy_plays['cumulative_profit'] = strategy_plays['profit'].cumsum()
            strategy_plays['play_number'] = range(1, len(strategy_plays) + 1)
            y_values = strategy_plays['cumulative_profit']
            y_label = 'Cumulative Profit ($)'
            title = f'Cumulative Profit - {len(strategies_to_plot)} Strats {filter_desc}'
        
        ax.plot(strategy_plays['play_number'], y_values, 
               label=strategy, linewidth=2, marker='o', markersize=2, alpha=0.7)
        
        # Get the color of the line we just plotted
        current_line_color = ax.get_lines()[-1].get_color()
        
        # Log detailed edge evolution (for dynamic Kelly)
        if use_kelly and dynamic_edge:
            print(f"\n   {'='*70}")
            print(f"   EDGE EVOLUTION: {strategy}")
            print(f"   {'='*70}")
            print(f"   {'Play':<6} {'RunWR%':<8} {'Kelly%':<8} {'BetSize':<10} {'Result':<7} {'Bankroll':<12} {'Paper?':<7}")
            print(f"   {'-'*76}")
            
            # Log ALL plays (no sampling)
            for idx, row in strategy_plays.iterrows():
                play_num = int(row['play_number'])
                run_wr = row['running_win_rate'] * 100
                kelly_pct = row['kelly_fraction_used'] * 100
                bet_size = row['bet_size']
                result = row['result']
                bankroll = row['bankroll']
                is_paper = 'YES' if row['paper_trading'] else 'NO'
                
                print(f"   {play_num:<6} {run_wr:>6.1f}% {kelly_pct:>6.2f}% ${bet_size:>8,.0f} {result:<7} ${bankroll:>10,.0f} {is_paper:<7}")
            
            print(f"   {'='*70}\n")
        
        # Add vertical lines for edge start/stop (only for dynamic Kelly)
        if use_kelly and dynamic_edge:
            # Find where edge first appears (kelly > 0)
            edge_starts = strategy_plays[strategy_plays['kelly_fraction_used'] > 0]['play_number'].values
            
            if len(edge_starts) > 0:
                # Edge start (first time kelly > 0 after MIN_BETS_FOR_KELLY)
                first_edge = edge_starts[0]
                ax.axvline(x=first_edge, color=current_line_color, linestyle='--', 
                          alpha=0.5, linewidth=1.5)
                
                # Find edge stops (kelly goes to 0 after having edge)
                had_edge = False
                for idx_play, row in strategy_plays.iterrows():
                    if row['kelly_fraction_used'] > 0:
                        had_edge = True
                    elif had_edge and row['kelly_fraction_used'] == 0:
                        # Edge just disappeared
                        ax.axvline(x=row['play_number'], color=current_line_color, linestyle=':',
                                  alpha=0.5, linewidth=1.5)
                        had_edge = False  # Reset to find next edge period
    
    if use_kelly:
        ax.axhline(y=starting_bankroll, color='black', linestyle='--', alpha=0.3, label='Starting Bankroll')
    else:
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    
    ax.set_xlabel('Play Number', fontsize=12, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    filepath = f"{output_dir}/01_{'bankroll_growth' if use_kelly else 'cumulative_profit'}_over_time.png"
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved: {filepath}")
    if show_plots:
        plt.show()
    plt.close()
    
    # =========================================================================
    # 2. ROI BY SEASON COMPARISON (GROUPED BAR CHART)
    # =========================================================================
    print("\n2. Creating ROI by Season Comparison chart...")
    
    # Prepare data for grouped bar chart
    roi_data = []
    top_10_strategies = df_strategies.head(10)
    
    for idx, row in top_10_strategies.iterrows():
        strategy_name = row['strategy_name']
        
        for season in ['2023-24', '2024-25', '2025-26']:
            season_key = season.replace('-', '_')
            col_name = f'{season_key}_roi'
            roi_value = row.get(col_name, 0)
            roi_data.append({
                'Strategy': strategy_name[:30],  # Truncate for readability
                'Season': season,
                'ROI': roi_value
            })
    
    df_roi = pd.DataFrame(roi_data)
    
    if len(df_roi) == 0:
        print("   ⚠️  No ROI data found for visualization, skipping chart 2")
    else:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create grouped bar chart
        strategies_unique = df_roi['Strategy'].unique()
        x = range(len(strategies_unique))
        width = 0.25
        
        seasons = ['2023-24', '2024-25', '2025-26']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, season in enumerate(seasons):
            season_data = df_roi[df_roi['Season'] == season]
            values = [season_data[season_data['Strategy'] == s]['ROI'].values[0] if len(season_data[season_data['Strategy'] == s]) > 0 else 0 for s in strategies_unique]
            ax.bar([xi + (i - 1) * width for xi in x], values, width, label=season, color=colors[i], alpha=0.8)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_xlabel('Strategy', fontsize=12, fontweight='bold')
        ax.set_ylabel('ROI (%)', fontsize=12, fontweight='bold')
        ax.set_title('ROI by Season Comparison - Top 10 Strategies', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(strategies_unique, rotation=45, ha='right', fontsize=9)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        filepath = f"{output_dir}/02_roi_by_season_comparison.png"
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"   ✅ Saved: {filepath}")
        if show_plots:
            plt.show()
        plt.close()
    
    # =========================================================================
    # 3. WIN RATE OVER TIME (ROLLING AVERAGE)
    # =========================================================================
    print("\n3. Creating Win Rate Over Time (Rolling Average) chart...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    window = 50  # 50-game rolling window
    
    for strategy in top_strategies[:5]:  # Top 5 for readability
        strategy_plays = df_top_plays[df_top_plays['strategy_name'] == strategy].copy()
        
        if len(strategy_plays) < window:
            continue
        
        # Convert result to binary (1=WIN, 0=LOSS)
        strategy_plays['win'] = (strategy_plays['result'] == 'WIN').astype(int)
        strategy_plays['rolling_win_rate'] = strategy_plays['win'].rolling(window=window).mean() * 100
        strategy_plays['play_number'] = range(1, len(strategy_plays) + 1)
        
        # Only plot after we have enough data
        valid_data = strategy_plays[strategy_plays['play_number'] >= window]
        
        ax.plot(valid_data['play_number'], valid_data['rolling_win_rate'], 
               label=strategy, linewidth=2, alpha=0.7)
    
    ax.axhline(y=52.38, color='red', linestyle='--', alpha=0.5, label='Breakeven (52.38%)')
    ax.set_xlabel('Play Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Win Rate (%) - 50 Game Rolling Avg', fontsize=12, fontweight='bold')
    ax.set_title(f'Win Rate Over Time - Top 5 Strategies (Rolling {window}-Game Avg)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([40, 70])
    
    filepath = f"{output_dir}/03_win_rate_over_time_rolling.png"
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved: {filepath}")
    if show_plots:
        plt.show()
    plt.close()
    
    # =========================================================================
    # 4. PROFIT DISTRIBUTION (BOX PLOT)
    # =========================================================================
    print("\n4. Creating Profit Distribution (Box Plot) chart...")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data for box plot
    profit_data = []
    for strategy in top_strategies[:10]:
        strategy_plays = df_top_plays[df_top_plays['strategy_name'] == strategy]
        for profit in strategy_plays['profit']:
            profit_data.append({
                'Strategy': strategy[:30],
                'Profit': profit
            })
    
    df_profit = pd.DataFrame(profit_data)
    
    # Create box plot
    strategies_for_plot = df_profit['Strategy'].unique()
    sns.boxplot(data=df_profit, x='Strategy', y='Profit', ax=ax, palette='Set2')
    
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Strategy', fontsize=12, fontweight='bold')
    ax.set_ylabel('Profit per Play ($)', fontsize=12, fontweight='bold')
    ax.set_title('Profit Distribution per Play - Top 10 Strategies', fontsize=14, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    filepath = f"{output_dir}/04_profit_distribution_boxplot.png"
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"   ✅ Saved: {filepath}")
    if show_plots:
        plt.show()
    plt.close()
    
    print(f"\n✅ All visualizations saved to: {output_dir}")
    print("="*80)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze all NBA points props strategies across multiple seasons'
    )
    
    parser.add_argument('--strategy-types', nargs='+', default=['2d', '3d'],
                       help='Strategy types to analyze (default: 2d 3d)')
    parser.add_argument('--seasons', nargs='+', default=['2023-24', '2024-25', '2025-26'],
                       help='Seasons to analyze')
    parser.add_argument('--min-plays', type=int, default=MIN_PLAYS_DEFAULT,
                       help=f'Minimum total plays required to include strategy (default: {MIN_PLAYS_DEFAULT})')
    parser.add_argument('--top-n', type=int, default=20,
                       help='Number of top strategies to display (default: 20)')
    parser.add_argument('--output-dir', default=str(Path.home() / 'Downloads' / 'tmp'),
                       help='Output directory for results (default: ~/Downloads/tmp)')
    parser.add_argument('--no-save', action='store_true',
                       help='Skip saving results to files')
    parser.add_argument('--viz', action='store_true',
                       help='Generate visualizations')
    parser.add_argument('--viz-output-dir', default=None,
                       help='Directory for visualization outputs (default: ~/Downloads/tmp)')
    parser.add_argument('--viz-top-n', type=int, default=10,
                       help='Number of top strategies to visualize (default: 10)')
    parser.add_argument('--viz-filter', choices=['edge', 'all', 'profitable', 'losing'], default='edge',
                       help='Filter mode for growth chart: edge (ROI>0 & WR>52.38%%), all (no filter), '
                            'profitable (ROI>0), losing (ROI<0) (default: edge)')
    parser.add_argument('--no-viz-display', action='store_true',
                       help='Save visualizations but do not display interactively')
    parser.add_argument('--kelly', action='store_true',
                       help='Use Kelly criterion for bet sizing (requires --bankroll)')
    parser.add_argument('--bankroll', type=float, default=10000,
                       help='Starting bankroll for Kelly criterion (default: 10000)')
    parser.add_argument('--kelly-fraction', type=float, default=0.5,
                       help='Kelly fraction: 1.0=full, 0.5=half, 0.25=quarter (default: 0.5)')
    parser.add_argument('--dynamically-calculate-edge', action='store_true',
                       help='Recalculate edge after each bet (vs using static overall win rate)')
    parser.add_argument('--paper-trade-plays', type=int, default=DEFAULT_PAPER_TRADE_PLAYS,
                       help=f'Number of plays to paper trade before betting (default: {DEFAULT_PAPER_TRADE_PLAYS})')
    parser.add_argument('--min-edge-threshold', type=float, default=DEFAULT_MIN_EDGE_THRESHOLD,
                       help=f'Minimum win rate %% to start betting after paper trading (default: {DEFAULT_MIN_EDGE_THRESHOLD:.1f})')
    parser.add_argument('--max-viz-strats', type=int, default=999,
                       help='Maximum strategies to show on growth chart (default: 999, shows all with edge)')
    
    args = parser.parse_args()
    
    # Run analysis
    strategies, df_strategies = analyze_all_strategies(
        strategy_types=args.strategy_types,
        seasons=args.seasons,
        min_plays=args.min_plays
    )
    
    if len(strategies) == 0:
        print("\n❌ No strategies found!")
        return
    
    # Print results
    print_top_strategies(df_strategies, n=args.top_n)
    print_summary_stats(df_strategies)
    
    # Save results
    if not args.no_save:
        save_results(strategies, df_strategies, output_dir=args.output_dir)
    
    # Generate visualizations if requested
    if args.viz:
        # Need to reload full play data with strategy names
        print(f"\n{'='*80}")
        print("LOADING PLAY DATA FOR VISUALIZATIONS")
        print("="*80)
        
        s3_client = boto3.client('s3')
        bucket = 'nba-betting-mt'
        
        all_plays_with_names = []
        
        for strategy_type in args.strategy_types:
            for season in args.seasons:
                df = load_backtest_plays(s3_client, bucket, strategy_type, season)
                if len(df) == 0:
                    continue
                
                # Add strategy name to each play
                if strategy_type == '2d':
                    group_cols = ['line_tier', 'spread_bin', 'bet_side']
                else:
                    group_cols = ['line_tier', 'spread_bin', 'bet_side', 'scorer_type']
                
                # Generate strategy names
                def get_strategy_name(row):
                    tier_slug = row['line_tier'].split(' ')[0].replace('-', '_').lower()
                    spread_parts = row['spread_bin'].split(' ')
                    spread_value = spread_parts[0].replace('-', '_').replace("'", '').lower()
                    spread_type = ''
                    if len(spread_parts) > 1:
                        if 'Fav' in row['spread_bin']:
                            spread_type = '_fav'
                        elif 'Dog' in row['spread_bin']:
                            spread_type = '_dog'
                    spread_slug = f"{spread_value}{spread_type}"
                    bet_slug = row['bet_side'].lower()
                    
                    if strategy_type == '3d' and 'scorer_type' in row and pd.notna(row['scorer_type']) and row['scorer_type'] != 'N/A':
                        scorer_slug = 'rim' if 'Rim' in row['scorer_type'] else 'perimeter'
                        return f"{tier_slug}_{spread_slug}_{scorer_slug}_{bet_slug}"
                    else:
                        return f"{tier_slug}_{spread_slug}_{bet_slug}"
                
                df['strategy_name'] = df.apply(get_strategy_name, axis=1)
                all_plays_with_names.append(df)
        
        if all_plays_with_names:
            df_all_plays = pd.concat(all_plays_with_names, ignore_index=True)
            print(f"✅ Loaded {len(df_all_plays):,} plays for visualization")
            
            create_visualizations(
                df_all_plays, 
                df_strategies, 
                output_dir=args.viz_output_dir,
                top_n=args.viz_top_n,
                show_plots=not args.no_viz_display,
                use_kelly=args.kelly,
                starting_bankroll=args.bankroll,
                kelly_fraction=args.kelly_fraction,
                max_viz_strats=args.max_viz_strats,
                viz_filter=args.viz_filter,
                dynamic_edge=args.dynamically_calculate_edge,
                paper_trade_plays=args.paper_trade_plays,
                min_edge_threshold=args.min_edge_threshold
            )
        else:
            print("❌ No play data found for visualizations")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
