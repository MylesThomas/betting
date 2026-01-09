"""
NBA Player Points Props Strategy Backtest
=========================================

Context:
--------
This script backtests the NBA player points prop betting strategies that were
developed using 2025-26 season data. The goal is to validate whether these
strategies would have been profitable on historical data from past seasons.

We have two strategies to backtest:
1. 2D Strategy: tier × spread (63 combinations in detailed mode)
   - 7 player tiers × 9 spread bins
   - Strategy file: points_by_role_gamespread_strategies_2025-26.json

2. 3D Strategy: tier × spread × scorer_type (126 combinations in detailed mode)
   - 7 player tiers × 9 spread bins × 2 scorer types (rim/non-rim)
   - Strategy file: points_by_role_gamespread_6feet_strategies_2025-26_rim40.json

Backtest Approach:
-----------------
1. Load the 2025-26 strategies JSON files from S3
2. For each historical season (2024-25, 2023-24, 2022-23, 2021-22):
   a. Load that season's player_props_with_actuals CSV from S3
   b. Apply the 2025-26 strategies to identify plays
   c. Calculate performance (wins, losses, ROI, profit)
   d. Store season-level results
3. Aggregate results across all seasons
4. Output summary statistics and visualizations

IMPORTANT - Data Prerequisites:
-------------------------------
Before running this backtest, you MUST fetch historical data for each season.
The Odds API only goes back to 2021-22 season.

For each season (2021-22, 2022-23, 2023-24, 2024-25), run this workflow:

Step 1: Fetch historical player props + game results
-----------------------------------------------------
python3 scripts/fetch_nba_player_props.py --mode 2 --fetch-games --s3 --season YYYY-YY

This fetches:
- s3://the-odds-api-mt/nba/historical_player_props/YYYY-YY/*.csv (props)
- s3://nba-api-mt/player_game_logs/YYYY-YY/*.csv (game results)

Step 2: Fetch historical game lines (spreads)
----------------------------------------------
python3 scripts/fetch_historical_nba_season_lines.py --season YYYY-YY --prod-run

This fetches:
- s3://the-odds-api-mt/nba/historical_game_lines/YYYY-YY/*.csv (game spreads)

Step 3: Fetch shot charts (for 3D strategy only)
------------------------------------------------
python3 scripts/fetch_all_nba_shot_charts.py --auto --seasons YYYY-YY

This fetches:
- s3://nba-api-mt/player_shot_charts/YYYY-YY/*.csv (shot distance data)

Step 4: Join all data sources into unified dataset
---------------------------------------------------
# For 2D strategy (no shot charts needed)
python3 scripts/join_nba_points_props_actuals_charts_gamelines.py --season YYYY-YY --s3

# For 3D strategy (includes shot charts + rim scorer classification)
python3 scripts/join_nba_points_props_actuals_charts_gamelines.py --season YYYY-YY --s3 --rim-scorer-pct 40

This creates:
- s3://nba-betting-mt/data/03_intermediate/player_props_with_actuals_YYYY-YY.csv (2D)
- s3://nba-betting-mt/data/03_intermediate/player_props_with_actuals_YYYY-YY_rim40.csv (3D)

Required Data Files in S3:
--------------------------
Strategies (from 2025-26):
- s3://nba-betting-mt/data/03_intermediate/points_by_role_gamespread_strategies_2025-26.json
- s3://nba-betting-mt/data/03_intermediate/points_by_role_gamespread_6feet_strategies_2025-26_rim40.json

Historical Data (for each season):
- s3://nba-betting-mt/data/03_intermediate/player_props_with_actuals_YYYY-YY.csv
- s3://nba-betting-mt/data/03_intermediate/player_props_with_actuals_YYYY-YY_rim40.csv

Output:
-------
The script outputs:
1. Season-by-season performance CSV
2. Aggregate statistics (total ROI, win rate, profit)
3. Strategy-level breakdown (which strategies performed best)
4. Time series visualizations (ROI over time)

Files saved to:
- s3://nba-betting-mt/data/04_output/backtests/points_props_YYYYMMDD/
  - 2d_strategy_backtest.csv
  - 3d_strategy_backtest.csv
  - aggregate_summary.json
  - visualizations/

Usage:
------
# Backtest both strategies on all available seasons
python3 backtesting/20260108_nba_points_props_strategy_backtest.py

# Backtest specific seasons
python3 backtesting/20260108_nba_points_props_strategy_backtest.py --seasons 2024-25 2023-24

# Backtest only 2D strategy
python3 backtesting/20260108_nba_points_props_strategy_backtest.py --strategy 2d

# Backtest only 3D strategy
python3 backtesting/20260108_nba_points_props_strategy_backtest.py --strategy 3d

# Backtest with different minimum ROI threshold
python3 backtesting/20260108_nba_points_props_strategy_backtest.py --min-roi 3.0

# Save results locally instead of S3
python3 backtesting/20260108_nba_points_props_strategy_backtest.py --local-only

Expected Output Example:
-----------------------
Season: 2024-25
  2D Strategy:
    Total Plays: 847
    Wins: 456 (53.8%)
    Losses: 391 (46.2%)
    Total Staked: $8,470
    Total Profit: +$1,234
    ROI: +14.6%
  
  3D Strategy:
    Total Plays: 623
    Wins: 348 (55.9%)
    Losses: 275 (44.1%)
    Total Staked: $6,230
    Total Profit: +$1,567
    ROI: +25.2%

Season: 2023-24
  [similar output]

Aggregate (2023-24 to 2024-25):
  2D Strategy: +$2,456 (+12.3% ROI) over 1,694 plays
  3D Strategy: +$3,891 (+21.8% ROI) over 1,246 plays

Notes:
------
- The strategies are fixed (from 2025-26 training data)
- We're testing out-of-sample performance
- This validates if the strategies generalize to other seasons
- If backtest shows poor performance, we may need to retrain strategies
  using multi-season data instead of just 2025-26

Author: Myles Thomas
Date: 2026-01-08
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import boto3
import pandas as pd
import numpy as np
from io import StringIO

# Find project root (look for .gitignore)
current_dir = Path(__file__).resolve().parent
while not (current_dir / '.gitignore').exists():
    if current_dir == current_dir.parent:
        raise FileNotFoundError("Could not find project root (.gitignore)")
    current_dir = current_dir.parent

PROJECT_ROOT = current_dir
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# Now import project modules
from config_loader import get_config

# Load config
CONFIG = get_config()

# AWS S3 configuration (from config.yaml)
S3_BUCKET_PROPS = CONFIG['aws']['buckets']['odds_api']
S3_BUCKET_NBA = CONFIG['aws']['buckets']['nba_api']
S3_BUCKET_OUTPUT = CONFIG['aws']['buckets']['nba_betting']
AWS_REGION = 'us-east-2'

# AWS S3 client
s3_client = boto3.client('s3', region_name=AWS_REGION)

# Emoji map
EMOJI = {
    'success': '✅',
    'error': '❌',
    'warning': '⚠️',
    'info': 'ℹ️',
    'chart': '📊',
    'calendar': '📅',
    'money': '💰',
    'up': '📈',
    'down': '📉',
    'basketball': '🏀',
    'test': '🧪',
    'upload': '⬆️',
    'star': '⭐',
}


def load_strategies_from_s3(strategy_type: str, season: str = '2025-26') -> Dict:
    """
    Load strategy JSON from S3.
    
    Args:
        strategy_type: '2d' or '3d'
        season: Season the strategies were trained on (default: 2025-26)
    
    Returns:
        dict: Strategy configurations
    """
    bucket = CONFIG['aws']['buckets']['nba_betting']
    
    if strategy_type == '2d':
        key = f"data/03_intermediate/points_by_role_gamespread_strategies_{season}.json"
    elif strategy_type == '3d':
        key = f"data/03_intermediate/points_by_role_gamespread_6feet_strategies_{season}_rim40.json"
    else:
        raise ValueError(f"Invalid strategy_type: {strategy_type}")
    
    print(f"{EMOJI['info']} Loading {strategy_type.upper()} strategies from s3://{bucket}/{key}")
    
    response = s3_client.get_object(Bucket=bucket, Key=key)
    strategies = json.loads(response['Body'].read().decode('utf-8'))
    
    print(f"{EMOJI['success']} Loaded {strategies.get('total_strategies', 0)} strategies")
    return strategies


def load_props_data_from_s3(season: str, strategy_type: str) -> pd.DataFrame:
    """
    Load player props with actuals CSV from S3.
    
    Args:
        season: NBA season (e.g., '2024-25')
        strategy_type: '2d' or '3d' (3d requires rim scorer data)
    
    Returns:
        pd.DataFrame: Props data with actuals (or None if not found)
    """
    if strategy_type == '2d':
        key = f"data/03_intermediate/player_props_with_actuals_{season}.csv"
    elif strategy_type == '3d':
        key = f"data/03_intermediate/player_props_with_actuals_{season}_rim40.csv"
    else:
        raise ValueError(f"Invalid strategy_type: {strategy_type}")
    
    print(f"{EMOJI['info']} Loading {season} data from s3://{S3_BUCKET_OUTPUT}/{key}")
    
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET_OUTPUT, Key=key)
        df = pd.read_csv(StringIO(response['Body'].read().decode('utf-8')))
        print(f"{EMOJI['success']} Loaded {len(df):,} rows for {season}")
        return df
    except s3_client.exceptions.NoSuchKey:
        print(f"{EMOJI['error']} Data file not found: s3://{S3_BUCKET_OUTPUT}/{key}")
        print(f"\n{EMOJI['warning']} To generate this file, run:")
        print(f"   python3 scripts/fetch_nba_player_props.py --mode 2 --fetch-games --s3 --season {season}")
        print(f"   python3 scripts/fetch_historical_nba_season_lines.py --season {season} --prod-run")
        if strategy_type == '3d':
            print(f"   python3 scripts/fetch_all_nba_shot_charts.py --auto --seasons {season}")
            print(f"   python3 scripts/join_nba_points_props_actuals_charts_gamelines.py --season {season} --s3 --rim-scorer-pct 40")
        else:
            print(f"   python3 scripts/join_nba_points_props_actuals_charts_gamelines.py --season {season} --s3")
        return None
    except Exception as e:
        print(f"{EMOJI['error']} Error loading data: {e}")
        return None


def bin_points_line(line: float, granularity: str = 'detailed') -> str:
    """
    Bin player points line into tiers.
    
    Must match the binning used in strategy generation.
    """
    if pd.isna(line):
        return 'Unknown'
    
    if granularity == 'standard':
        if line < 10:
            return '<10 (Bench)'
        elif line < 20:
            return '10-20 (Role)'
        elif line < 30:
            return '20-30 (Star)'
        else:
            return '30+ (Superstar)'
    else:  # detailed
        if line < 5:
            return '<5 (Deep Bench)'
        elif line < 10:
            return '5-10 (Bench)'
        elif line < 15:
            return '10-15 (Role Player)'
        elif line < 20:
            return '15-20 (High Role)'
        elif line < 25:
            return '20-25 (Star)'
        elif line < 30:
            return '25-30 (High Star)'
        else:
            return '30+ (Superstar)'


def bin_team_spread(spread: float, granularity: str = 'detailed') -> str:
    """
    Bin team spread into categories.
    
    Args:
        spread: Team spread (positive = underdog, negative = favorite)
    
    Must match the binning used in strategy generation.
    """
    if pd.isna(spread):
        return 'Unknown'
    
    if granularity == 'standard':
        if spread < -5:
            return 'Favorite'
        elif spread <= 5:
            return 'Pick\'em'
        else:
            return 'Underdog'
    else:  # detailed
        if spread < -15:
            return '15+ Fav'
        elif spread < -10:
            return '10-15 Fav'
        elif spread < -6:
            return '6-10 Fav'
        elif spread < -2:
            return '2-6 Fav'
        elif spread <= 2:
            return 'Pick\'em (-2 to +2)'
        elif spread <= 6:
            return '2-6 Dog'
        elif spread <= 10:
            return '6-10 Dog'
        elif spread <= 15:
            return '10-15 Dog'
        else:
            return '15+ Dog'


def apply_strategies_to_data(df: pd.DataFrame, strategies: Dict, granularity: str = 'detailed', strategy_type: str = '2d', min_roi: float = 5.0) -> pd.DataFrame:
    """
    Apply strategies to historical data to identify plays.
    
    Args:
        df: Player props with actuals
        strategies: Strategy dictionary from JSON
        granularity: 'standard' or 'detailed'
        strategy_type: '2d' or '3d'
        min_roi: Minimum ROI threshold to filter strategies (default: 5.0%)
    
    Returns:
        DataFrame with identified plays and their outcomes
    """
    print(f"\n{EMOJI['chart']} Applying strategies to {len(df):,} player-game records...")
    
    # Bin each row
    df['line_tier'] = df['points_line'].apply(lambda x: bin_points_line(x, granularity))
    df['spread_bin'] = df['team_spread'].apply(lambda x: bin_team_spread(x, granularity))
    
    # Extract strategies dict (handle nested structure)
    if 'strategies' in strategies:
        strat_list = strategies['strategies']
    else:
        strat_list = strategies
    
    # Convert to list if it's a dict (for uniform iteration)
    if isinstance(strat_list, dict):
        strat_list = list(strat_list.values())
    
    # Filter to strategies meeting minimum ROI threshold
    total_strats = len(strat_list)
    strat_list = [s for s in strat_list if s.get('roi', 0) >= min_roi]
    filtered_count = total_strats - len(strat_list)
    
    if filtered_count > 0:
        print(f"{EMOJI['info']} Filtered {filtered_count}/{total_strats} strategies below {min_roi}% ROI threshold")
        print(f"{EMOJI['success']} Using {len(strat_list)} winning strategies (ROI >= {min_roi}%)")
    
    if not strat_list:
        print(f"{EMOJI['warning']} No strategies meet ROI >= {min_roi}% threshold!")
        return pd.DataFrame()
    
    plays = []
    
    for idx, row in df.iterrows():
        line_tier = row['line_tier']
        spread_bin = row['spread_bin']
        scorer_type = row.get('scorer_type', None)
        
        # Try to match against each strategy
        for strat_idx, strat in enumerate(strat_list):
            # Check if this row matches the strategy
            line_match = strat['line_tier'] == line_tier
            spread_match = strat['spread_bin'] == spread_bin
            
            # For 3D strategies, also check scorer_type
            scorer_match = True
            if strategy_type == '3d' and 'scorer_type' in strat:
                if pd.isna(scorer_type):
                    continue
                scorer_match = strat['scorer_type'] == scorer_type
            
            if line_match and spread_match and scorer_match:
                # This row matches this strategy - would have been a play
                plays.append({
                    'game_date': row.get('game_date'),
                    'player_name': row.get('PLAYER_NAME'),
                    'team': row.get('TEAM_NAME'),
                    'opponent': row.get('MATCHUP'),
                    'points_line': row.get('points_line'),
                    'team_spread': row.get('team_spread'),
                    'line_tier': line_tier,
                    'spread_bin': spread_bin,
                    'scorer_type': scorer_type if strategy_type == '3d' else None,
                    'strategy_key': strat_idx,
                    'bet_side': strat['bet_side'],
                    'strategy_roi': strat['roi'],
                    'strategy_edge': strat['edge'],
                    'strategy_hit_rate': strat['hit_rate'],
                    'strategy_games': strat['games'],
                    'actual_points': row.get('PTS'),
                })
    
    if not plays:
        print(f"{EMOJI['warning']} No plays found matching strategies")
        return pd.DataFrame()
    
    df_plays = pd.DataFrame(plays)
    print(f"{EMOJI['success']} Found {len(df_plays):,} plays matching strategies")
    
    return df_plays


def calculate_outcomes(df_plays: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate win/loss/push for each play.
    
    Args:
        df_plays: DataFrame with plays and actual_points
    
    Returns:
        DataFrame with 'result' column added
    """
    print(f"\n{EMOJI['chart']} Calculating bet outcomes...")
    
    def determine_result(row):
        if pd.isna(row['actual_points']):
            return 'NO_DATA'
        
        actual = row['actual_points']
        line = row['points_line']
        
        if row['bet_side'] == 'OVER':
            if actual > line:
                return 'WIN'
            elif actual < line:
                return 'LOSS'
            else:
                return 'PUSH'
        else:  # UNDER
            if actual < line:
                return 'WIN'
            elif actual > line:
                return 'LOSS'
            else:
                return 'PUSH'
    
    df_plays['result'] = df_plays.apply(determine_result, axis=1)
    df_plays['margin'] = df_plays['actual_points'] - df_plays['points_line']
    
    # Calculate profit (assuming -110 odds, $100 bet)
    def calculate_profit(result):
        if result == 'WIN':
            return 90.91  # Win $90.91 on $100 bet at -110
        elif result == 'LOSS':
            return -100.0
        else:  # PUSH or NO_DATA
            return 0.0
    
    df_plays['profit'] = df_plays['result'].apply(calculate_profit)
    
    # Summary
    wins = (df_plays['result'] == 'WIN').sum()
    losses = (df_plays['result'] == 'LOSS').sum()
    pushes = (df_plays['result'] == 'PUSH').sum()
    no_data = (df_plays['result'] == 'NO_DATA').sum()
    
    print(f"   Wins: {wins} ({wins/len(df_plays)*100:.1f}%)")
    print(f"   Losses: {losses} ({losses/len(df_plays)*100:.1f}%)")
    print(f"   Pushes: {pushes} ({pushes/len(df_plays)*100:.1f}%)")
    if no_data > 0:
        print(f"   No Data: {no_data} ({no_data/len(df_plays)*100:.1f}%)")
    
    # Show sample plays for verification
    print(f"\n{EMOJI['test']} Sample Plays (random 5 for verification):")
    if len(df_plays) > 0:
        sample = df_plays.sample(min(5, len(df_plays)), random_state=42)
        for idx, row in sample.iterrows():
            bet = row['bet_side']
            line = row['points_line']
            actual = row['actual_points']
            result = row['result']
            margin = row['margin']
            print(f"   {row['player_name']:20s} | Bet {bet:5s} {line:4.1f} | Actual: {actual:4.1f} | Margin: {margin:+5.1f} → {result}")
    
    return df_plays


def analyze_per_strategy_performance(df_plays: pd.DataFrame, strategies: Dict, strategy_type: str) -> pd.DataFrame:
    """
    Break down performance by individual strategy.
    
    Args:
        df_plays: DataFrame with plays and outcomes
        strategies: Strategy dictionary
        strategy_type: '2d' or '3d'
    
    Returns:
        DataFrame with per-strategy stats
    """
    print(f"\n{EMOJI['chart']} Analyzing per-strategy performance...")
    
    # Extract strategies list
    if 'strategies' in strategies:
        strat_list = strategies['strategies']
    else:
        strat_list = strategies
    
    if isinstance(strat_list, dict):
        strat_list = list(strat_list.values())
    
    # Group plays by strategy
    strategy_stats = []
    
    for strat_idx, strat in enumerate(strat_list):
        strat_plays = df_plays[df_plays['strategy_key'] == strat_idx]
        
        if len(strat_plays) == 0:
            continue
        
        wins = (strat_plays['result'] == 'WIN').sum()
        losses = (strat_plays['result'] == 'LOSS').sum()
        pushes = (strat_plays['result'] == 'PUSH').sum()
        total = wins + losses  # Exclude pushes from win rate
        
        win_rate = (wins / total * 100) if total > 0 else 0
        total_profit = strat_plays['profit'].sum()
        total_staked = len(strat_plays) * 100
        roi = (total_profit / total_staked * 100) if total_staked > 0 else 0
        
        strategy_stats.append({
            'strategy_idx': strat_idx,
            'line_tier': strat['line_tier'],
            'spread_bin': strat['spread_bin'],
            'scorer_type': strat.get('scorer_type', 'N/A') if strategy_type == '3d' else 'N/A',
            'bet_side': strat['bet_side'],
            'training_roi': strat['roi'],
            'training_hit_rate': strat['hit_rate'],
            'training_games': strat['games'],
            'backtest_plays': len(strat_plays),
            'backtest_wins': wins,
            'backtest_losses': losses,
            'backtest_pushes': pushes,
            'backtest_win_rate': win_rate,
            'backtest_profit': total_profit,
            'backtest_roi': roi,
            'roi_delta': roi - strat['roi']  # How much worse/better than training
        })
    
    df_stats = pd.DataFrame(strategy_stats)
    
    # Sort by number of plays (most active strategies first)
    df_stats = df_stats.sort_values('backtest_plays', ascending=False)
    
    # Print top 10 most active strategies
    print(f"\n{EMOJI['star']} Top 10 Most Active Strategies:")
    print("="*120)
    top_10 = df_stats.head(10)
    for idx, row in top_10.iterrows():
        print(f"\nStrategy #{row['strategy_idx']}: {row['line_tier']} | {row['spread_bin']} | {row['bet_side']}")
        if strategy_type == '3d' and row['scorer_type'] != 'N/A':
            print(f"  Scorer Type: {row['scorer_type']}")
        print(f"  Training:  {row['training_games']:3.0f} games | {row['training_hit_rate']:5.1f}% hit | {row['training_roi']:+6.1f}% ROI")
        print(f"  Backtest:  {row['backtest_plays']:3.0f} plays | {row['backtest_win_rate']:5.1f}% win | {row['backtest_roi']:+6.1f}% ROI | Delta: {row['roi_delta']:+6.1f}%")
        print(f"  Results:   W:{row['backtest_wins']} L:{row['backtest_losses']} P:{row['backtest_pushes']} | Profit: ${row['backtest_profit']:,.2f}")
    
    # Show worst performing strategies
    print(f"\n{EMOJI['warning']} Worst 5 Performing Strategies (by ROI delta):")
    print("="*120)
    worst_5 = df_stats.nsmallest(5, 'roi_delta')
    for idx, row in worst_5.iterrows():
        print(f"\nStrategy #{row['strategy_idx']}: {row['line_tier']} | {row['spread_bin']} | {row['bet_side']}")
        if strategy_type == '3d' and row['scorer_type'] != 'N/A':
            print(f"  Scorer Type: {row['scorer_type']}")
        print(f"  Training:  {row['training_games']:3.0f} games | {row['training_hit_rate']:5.1f}% hit | {row['training_roi']:+6.1f}% ROI")
        print(f"  Backtest:  {row['backtest_plays']:3.0f} plays | {row['backtest_win_rate']:5.1f}% win | {row['backtest_roi']:+6.1f}% ROI | Delta: {row['roi_delta']:+6.1f}%")
    
    return df_stats


def backtest_season(season: str, strategies: Dict, strategy_type: str, granularity: str = 'detailed', min_roi: float = 5.0) -> Dict:
    """
    Backtest strategies on a single season.
    
    Args:
        season: NBA season (e.g., '2024-25')
        strategies: Strategy dictionary
        strategy_type: '2d' or '3d'
        granularity: 'standard' or 'detailed'
        min_roi: Minimum ROI threshold to filter strategies (default: 5.0%)
    
    Returns:
        dict: Season results
    """
    print(f"\n{'='*80}")
    print(f"{EMOJI['calendar']} Backtesting {strategy_type.upper()} Strategy on {season}")
    print(f"{'='*80}")
    
    # Load data
    df = load_props_data_from_s3(season, strategy_type)
    if df is None:
        return None
    
    # Apply strategies (filtering to min_roi threshold)
    df_plays = apply_strategies_to_data(df, strategies, granularity, strategy_type, min_roi)
    if df_plays.empty:
        return None
    
    # Calculate outcomes
    df_plays = calculate_outcomes(df_plays)
    
    # Analyze per-strategy performance
    df_strategy_stats = analyze_per_strategy_performance(df_plays, strategies, strategy_type)
    
    # Calculate summary statistics
    total_plays = len(df_plays)
    wins = (df_plays['result'] == 'WIN').sum()
    losses = (df_plays['result'] == 'LOSS').sum()
    pushes = (df_plays['result'] == 'PUSH').sum()
    no_data = (df_plays['result'] == 'NO_DATA').sum()
    
    # Calculate win rate (excluding pushes and no_data)
    decided_plays = wins + losses
    win_rate = (wins / decided_plays * 100) if decided_plays > 0 else 0.0
    
    # Calculate profit and ROI
    total_profit = df_plays['profit'].sum()
    total_staked = decided_plays * 100  # $100 per bet
    roi = (total_profit / total_staked * 100) if total_staked > 0 else 0.0
    
    results = {
        'season': season,
        'strategy_type': strategy_type,
        'total_plays': total_plays,
        'wins': wins,
        'losses': losses,
        'pushes': pushes,
        'no_data': no_data,
        'win_rate': win_rate,
        'total_profit': total_profit,
        'total_staked': total_staked,
        'roi': roi,
        'plays_df': df_plays,
        'strategy_stats_df': df_strategy_stats
    }
    
    # Print summary
    print(f"\n{EMOJI['chart']} Season Summary:")
    print(f"   Total Plays: {total_plays:,}")
    print(f"   Wins: {wins} ({win_rate:.1f}%)")
    print(f"   Losses: {losses}")
    print(f"   Pushes: {pushes}")
    if no_data > 0:
        print(f"   No Data: {no_data}")
    print(f"   Total Profit: ${total_profit:,.2f}")
    print(f"   ROI: {roi:+.2f}%")
    
    return results


def save_backtest_results(all_results: List[Dict], strategy_type: str, output_dir: str = None, upload_s3: bool = True):
    """
    Save backtest results to CSV and JSON (local + S3).
    Saves in two formats:
    1. Timestamped directory (for historical records)
    2. Per-season directories (for easy analysis)
    
    Args:
        all_results: List of season results
        strategy_type: '2d' or '3d'
        output_dir: Output directory (if None, uses default)
        upload_s3: Whether to upload to S3 (default: True)
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if output_dir is None:
        output_dir = PROJECT_ROOT / 'data' / '04_output' / 'backtests' / f'points_props_{timestamp}'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary CSV
    summary_data = []
    for result in all_results:
        if result is None:
            continue
        summary_data.append({
            'season': result['season'],
            'strategy_type': result['strategy_type'],
            'total_plays': result['total_plays'],
            'wins': result['wins'],
            'losses': result['losses'],
            'pushes': result['pushes'],
            'no_data': result['no_data'],
            'win_rate': result['win_rate'],
            'total_profit': result['total_profit'],
            'total_staked': result['total_staked'],
            'roi': result['roi']
        })
    
    df_summary = pd.DataFrame(summary_data)
    summary_file = output_dir / f'{strategy_type}_strategy_summary.csv'
    df_summary.to_csv(summary_file, index=False)
    print(f"\n{EMOJI['success']} Saved summary to {summary_file}")
    
    # Save detailed plays CSV
    all_plays = []
    for result in all_results:
        if result and 'plays_df' in result:
            all_plays.append(result['plays_df'])
    
    if all_plays:
        df_all_plays = pd.concat(all_plays, ignore_index=True)
        plays_file = output_dir / f'{strategy_type}_strategy_all_plays.csv'
        df_all_plays.to_csv(plays_file, index=False)
        print(f"{EMOJI['success']} Saved detailed plays to {plays_file}")
    
    # Save per-strategy stats CSV
    all_strategy_stats = []
    for result in all_results:
        if result and 'strategy_stats_df' in result:
            df_stats = result['strategy_stats_df'].copy()
            df_stats['season'] = result['season']
            all_strategy_stats.append(df_stats)
    
    if all_strategy_stats:
        df_all_strategy_stats = pd.concat(all_strategy_stats, ignore_index=True)
        strategy_stats_file = output_dir / f'{strategy_type}_per_strategy_performance.csv'
        df_all_strategy_stats.to_csv(strategy_stats_file, index=False)
        print(f"{EMOJI['success']} Saved per-strategy performance to {strategy_stats_file}")
    
    # Calculate aggregate statistics
    total_plays = df_summary['total_plays'].sum()
    total_wins = df_summary['wins'].sum()
    total_losses = df_summary['losses'].sum()
    total_profit = df_summary['total_profit'].sum()
    total_staked = df_summary['total_staked'].sum()
    
    aggregate = {
        'strategy_type': strategy_type,
        'seasons': df_summary['season'].tolist(),
        'total_plays': int(total_plays),
        'total_wins': int(total_wins),
        'total_losses': int(total_losses),
        'aggregate_win_rate': float(total_wins / (total_wins + total_losses) * 100) if (total_wins + total_losses) > 0 else 0.0,
        'total_profit': float(total_profit),
        'total_staked': float(total_staked),
        'aggregate_roi': float(total_profit / total_staked * 100) if total_staked > 0 else 0.0
    }
    
    aggregate_file = output_dir / f'{strategy_type}_aggregate.json'
    with open(aggregate_file, 'w') as f:
        json.dump(aggregate, f, indent=2)
    print(f"{EMOJI['success']} Saved aggregate stats to {aggregate_file}")
    
    # Also save per-season files in a structured format for easy analysis
    print(f"\n{EMOJI['info']} Saving per-season results in structured format...")
    base_dir = PROJECT_ROOT / 'data' / '04_output' / 'backtests' / strategy_type
    
    for result in all_results:
        if result is None:
            continue
        
        season = result['season']
        season_dir = base_dir / season
        season_dir.mkdir(parents=True, exist_ok=True)
        
        # Save season summary
        season_summary = pd.DataFrame([{
            'season': result['season'],
            'strategy_type': result['strategy_type'],
            'total_plays': result['total_plays'],
            'wins': result['wins'],
            'losses': result['losses'],
            'pushes': result['pushes'],
            'no_data': result['no_data'],
            'win_rate': result['win_rate'],
            'total_profit': result['total_profit'],
            'total_staked': result['total_staked'],
            'roi': result['roi']
        }])
        season_summary_file = season_dir / 'summary.csv'
        season_summary.to_csv(season_summary_file, index=False)
        
        # Save season plays
        if 'plays_df' in result and result['plays_df'] is not None and len(result['plays_df']) > 0:
            season_plays_file = season_dir / 'plays.csv'
            result['plays_df'].to_csv(season_plays_file, index=False)
        
        # Save per-strategy stats for this season
        if 'strategy_stats_df' in result and result['strategy_stats_df'] is not None:
            season_stats_file = season_dir / 'per_strategy.csv'
            result['strategy_stats_df'].to_csv(season_stats_file, index=False)
        
        print(f"   {EMOJI['success']} Saved {season} results to {season_dir}")
    
    # Upload to S3
    if upload_s3:
        print(f"\n{EMOJI['upload']} Uploading backtest results to S3...")
        
        # Upload timestamped directory
        relative_path = output_dir.relative_to(PROJECT_ROOT / 'data' / '04_output')
        s3_prefix = f"data/04_output/{relative_path}"
        
        files_to_upload = [
            (summary_file, f"{s3_prefix}/{strategy_type}_strategy_summary.csv"),
            (aggregate_file, f"{s3_prefix}/{strategy_type}_aggregate.json"),
        ]
        
        if all_plays:
            files_to_upload.append((plays_file, f"{s3_prefix}/{strategy_type}_strategy_all_plays.csv"))
        
        if all_strategy_stats:
            files_to_upload.append((strategy_stats_file, f"{s3_prefix}/{strategy_type}_per_strategy_performance.csv"))
        
        for local_file, s3_key in files_to_upload:
            try:
                s3_client.upload_file(str(local_file), S3_BUCKET_OUTPUT, s3_key)
                print(f"   {EMOJI['success']} Uploaded {local_file.name} to s3://{S3_BUCKET_OUTPUT}/{s3_key}")
            except Exception as e:
                print(f"   {EMOJI['warning']} Failed to upload {local_file.name}: {e}")
        
        # Upload per-season files
        print(f"\n{EMOJI['upload']} Uploading per-season results...")
        for result in all_results:
            if result is None:
                continue
            
            season = result['season']
            season_dir = base_dir / season
            s3_season_prefix = f"data/04_output/backtests/{strategy_type}/{season}"
            
            # Upload summary
            summary_path = season_dir / 'summary.csv'
            if summary_path.exists():
                try:
                    s3_client.upload_file(str(summary_path), S3_BUCKET_OUTPUT, f"{s3_season_prefix}/summary.csv")
                    print(f"   {EMOJI['success']} Uploaded {season} summary")
                except Exception as e:
                    print(f"   {EMOJI['warning']} Failed to upload {season} summary: {e}")
            
            # Upload plays
            plays_path = season_dir / 'plays.csv'
            if plays_path.exists():
                try:
                    s3_client.upload_file(str(plays_path), S3_BUCKET_OUTPUT, f"{s3_season_prefix}/plays.csv")
                    print(f"   {EMOJI['success']} Uploaded {season} plays")
                except Exception as e:
                    print(f"   {EMOJI['warning']} Failed to upload {season} plays: {e}")
            
            # Upload per-strategy stats
            stats_path = season_dir / 'per_strategy.csv'
            if stats_path.exists():
                try:
                    s3_client.upload_file(str(stats_path), S3_BUCKET_OUTPUT, f"{s3_season_prefix}/per_strategy.csv")
                    print(f"   {EMOJI['success']} Uploaded {season} per-strategy stats")
                except Exception as e:
                    print(f"   {EMOJI['warning']} Failed to upload {season} per-strategy stats: {e}")
        
        print(f"{EMOJI['success']} All backtest results uploaded to S3!")
    
    return output_dir, aggregate


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Backtest NBA player points prop strategies on historical data'
    )
    parser.add_argument(
        '--seasons',
        nargs='+',
        default=['2024-25', '2023-24', '2022-23'],
        help='Seasons to backtest (default: 2024-25 2023-24 2022-23)'
    )
    parser.add_argument(
        '--strategy',
        choices=['2d', '3d', 'both'],
        default='both',
        help='Which strategy to backtest (default: both)'
    )
    parser.add_argument(
        '--granularity',
        choices=['standard', 'detailed'],
        default='detailed',
        help='Binning granularity (default: detailed)'
    )
    parser.add_argument(
        '--min-roi',
        type=float,
        default=5.0,
        help='Minimum ROI threshold for filtering strategies (not yet implemented) (default: 5.0)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results (default: auto-generated timestamped dir)'
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"{EMOJI['basketball']} NBA Points Props Strategy Backtest")
    print(f"{'='*80}\n")
    
    print(f"Strategies to test: {args.strategy.upper()}")
    print(f"Seasons to backtest: {', '.join(args.seasons)}")
    print(f"Granularity: {args.granularity}")
    print(f"Minimum ROI threshold: {args.min_roi}%")
    print(f"Output directory: {args.output_dir or 'auto-generated'}\n")
    
    # Determine which strategies to test
    strategies_to_test = []
    if args.strategy in ['2d', 'both']:
        strategies_to_test.append('2d')
    if args.strategy in ['3d', 'both']:
        strategies_to_test.append('3d')
    
    # Run backtest for each strategy type
    all_aggregates = {}
    final_output_dir = None
    
    for strategy_type in strategies_to_test:
        print(f"\n{'='*80}")
        print(f"{EMOJI['basketball']} Starting {strategy_type.upper()} Strategy Backtest")
        print(f"{'='*80}")
        
        # Load strategies
        strategies = load_strategies_from_s3(strategy_type, season='2025-26')
        
        # Backtest each season
        season_results = []
        for season in args.seasons:
            result = backtest_season(season, strategies, strategy_type, args.granularity, args.min_roi)
            if result:
                season_results.append(result)
        
        # Save results
        if season_results:
            output_dir, aggregate = save_backtest_results(
                season_results,
                strategy_type,
                args.output_dir
            )
            final_output_dir = output_dir
            all_aggregates[strategy_type] = aggregate
        else:
            print(f"\n{EMOJI['warning']} No results to save for {strategy_type.upper()} strategy")
    
    # Print final summary
    print(f"\n{'='*80}")
    print(f"{EMOJI['chart']} FINAL AGGREGATE RESULTS")
    print(f"{'='*80}\n")
    
    for strategy_type, agg in all_aggregates.items():
        print(f"{strategy_type.upper()} Strategy:")
        print(f"   Seasons: {', '.join(agg['seasons'])}")
        print(f"   Total Plays: {agg['total_plays']:,}")
        print(f"   Win Rate: {agg['aggregate_win_rate']:.1f}%")
        print(f"   Total Profit: ${agg['total_profit']:,.2f}")
        print(f"   Total Staked: ${agg['total_staked']:,.2f}")
        print(f"   ROI: {agg['aggregate_roi']:+.2f}%")
        print()
    
    print(f"{EMOJI['success']} Backtest complete!")
    if final_output_dir:
        print(f"{EMOJI['success']} Results saved to: {final_output_dir}")


if __name__ == '__main__':
    main()

