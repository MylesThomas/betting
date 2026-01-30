"""
Optimal Spread/ML Weight Allocation for Underdog Bets
======================================================

Context:
--------
When betting on an underdog, you can split your bankroll between:
- Spread bet (e.g., +3 at -110)  
- Moneyline bet (e.g., +150)

This script finds the optimal weight allocation between these two bets
to maximize Kelly criterion log growth over historical data.

Example with $20 unit:
- w=0.25: $5 on ML (+150), $15 on spread (+3)
- w=0.50: $10 on ML, $10 on spread  
- w=0.75: $15 on ML, $5 on spread

Problem Statement:
------------------
Find weight w ∈ [0, 1] that maximizes:
    
    Σ log(1 + R_i(w))
    
where:
    R_i(w) = w * R_ml + (1-w) * R_spread
    
    R_ml = (decimal_odds - 1) if ML wins else -1
    R_spread = (decimal_odds - 1) if spread covers else -1

Three Optimization Methods:
----------------------------
1. Generalized: Single optimal weight across ALL underdog games
2. ML Bins: Optimal weight per ML price range (+100-150, +150-200, etc.)
3. Spread Bins: Optimal weight per spread range (+0-3, +3-6, etc.)

Usage:
------
# Analyze cover rates first
python analysis/optimize_spread_ml_weights.py --sport nba --analyze-only --seasons 2025-26

# Run optimization
python analysis/optimize_spread_ml_weights.py --sport nba --mode generalized --seasons 2025-26 2024-25
python analysis/optimize_spread_ml_weights.py --sport nba --mode ml_bins --seasons 2025-26 2024-25
python analysis/optimize_spread_ml_weights.py --sport nba --mode spread_bins --seasons 2025-26 2024-25
python analysis/optimize_spread_ml_weights.py --sport nba --mode all --seasons 2025-26 2024-25 2023-24

Output:
-------
Results saved to:
- S3: s3://nba-betting-mt/analysis/spread_ml_optimization/{sport}/
  - cover_analysis_{seasons}.csv
  - generalized_results_{seasons}.csv
  - ml_bins_results_{seasons}.csv
  - spread_bins_results_{seasons}.csv
- Local: ~/Downloads/tmp/spread_ml_optimization/{sport}/
  - Same files as S3 for quick access

Author: Thomas Myles
Date: 2026-01-26
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.optimize import minimize_scalar
from typing import Dict, List, Tuple
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.s3_utils import read_df_from_s3, list_s3_files, upload_df_to_s3


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    'min_sample_size': 20,
    'ml_bins': [
        (100, 150),
        (150, 200),
        (200, 300),
        (300, 500),
        (500, 2000),
    ],
    'spread_bins': [
        (0.0, 3.0),
        (3.0, 6.0),
        (6.0, 10.0),
        (10.0, 50.0),
    ],
    's3': {
        'nba_lines_bucket': 'the-odds-api-mt',
        'nba_results_bucket': 'nba-betting-mt',
        'output_bucket': 'nba-betting-mt',
        'output_prefix': 'analysis/spread_ml_optimization',
    }
}

# Season date ranges (for filtering)
SEASON_DATES = {
    '2025-26': ('2025-10-01', '2026-06-30'),
    '2024-25': ('2024-10-01', '2025-06-30'),
    '2023-24': ('2023-10-01', '2024-06-30'),
    '2022-23': ('2022-10-01', '2023-06-30'),
    '2021-22': ('2021-10-01', '2022-06-30'),
    '2020-21': ('2020-12-01', '2021-07-30'),
}


# =============================================================================
# CACHING
# =============================================================================

CACHE_DIR = Path.home() / 'Downloads' / 'tmp'

def get_cache_path(sport: str, season: str, data_type: str) -> Path:
    """
    Get cache file path for a season's data.
    
    Args:
        sport: Sport name (e.g., 'nba')
        season: Season (e.g., '2025-26')
        data_type: Type of data ('merged' for final merged dataset)
    
    Returns:
        Path to cache file
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{sport}_{season}_{data_type}.parquet"
    return CACHE_DIR / filename


def load_from_cache(cache_path: Path) -> pd.DataFrame:
    """Load DataFrame from parquet cache"""
    if cache_path.exists():
        print(f"  📦 Loading from cache: {cache_path.name}")
        return pd.read_parquet(cache_path)
    return None


def save_to_cache(df: pd.DataFrame, cache_path: Path) -> None:
    """Save DataFrame to parquet cache"""
    df.to_parquet(cache_path, index=False)
    print(f"  💾 Saved to cache: {cache_path.name}")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_nba_game_lines(seasons: List[str]) -> pd.DataFrame:
    """
    Load NBA game lines from S3.
    
    Args:
        seasons: List of seasons (e.g., ['2025-26', '2024-25'])
    
    Returns:
        DataFrame with game lines (spread and moneyline)
    """
    print("\n📊 Loading NBA game lines from S3...")
    
    all_lines = []
    bucket = CONFIG['s3']['nba_lines_bucket']
    
    for season in seasons:
        print(f"  Loading {season}...")
        prefix = f"nba/historical_game_lines/{season}/"
        
        # List all files for season
        files = list_s3_files(bucket, prefix)
        csv_files = [f for f in files if f.endswith('.csv')]
        
        print(f"    Found {len(csv_files)} files")
        
        # Load each file
        for s3_key in csv_files:
            df = read_df_from_s3(bucket, s3_key)
            all_lines.append(df)
    
    # Check if any files were found
    if not all_lines:
        raise ValueError(
            f"No game lines found for seasons {seasons} in "
            f"s3://{bucket}/nba/historical_game_lines/"
        )
    
    # Combine all data
    df = pd.concat(all_lines, ignore_index=True)
    
    # Convert dates
    df['game_time'] = pd.to_datetime(df['game_time'])
    df['game_date'] = df['game_time'].dt.date
    
    print(f"  ✅ Loaded {len(df):,} line records")
    print(f"  Date range: {df['game_date'].min()} to {df['game_date'].max()}")
    
    return df


def load_nba_game_results(seasons: List[str]) -> pd.DataFrame:
    """
    Load NBA game results from S3.
    
    Args:
        seasons: List of seasons (e.g., ['2025-26', '2024-25'])
    
    Returns:
        DataFrame with game results
    """
    print("\n🏀 Loading NBA game results from S3...")
    
    all_results = []
    bucket = CONFIG['s3']['nba_results_bucket']
    prefix = 'data/01_input/historical_game_results/'
    
    # List all result files
    files = list_s3_files(bucket, prefix)
    csv_files = [f for f in files if f.endswith('.csv')]
    
    # Filter to season date ranges
    season_start = min(SEASON_DATES[s][0] for s in seasons)
    season_end = max(SEASON_DATES[s][1] for s in seasons)
    
    print(f"  Looking for games from {season_start} to {season_end}")
    print(f"  Total CSV files found: {len(csv_files)}")
    
    for s3_key in csv_files:
        # Extract date from filename (e.g., 2025-04-19.csv)
        filename = s3_key.split('/')[-1]
        file_date = filename.replace('.csv', '')
        
        if season_start <= file_date <= season_end:
            df = read_df_from_s3(bucket, s3_key)
            all_results.append(df)
    
    print(f"  Matched {len(all_results)} files within date range")
    
    # Check if any files were found
    if not all_results:
        raise ValueError(
            f"No game results found for seasons {seasons}. "
            f"Expected files between {season_start} and {season_end} in "
            f"s3://{bucket}/{prefix}"
        )
    
    # Combine all data
    df = pd.concat(all_results, ignore_index=True)
    
    # Convert dates
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE']).dt.date
    
    print(f"  ✅ Loaded {len(df):,} game results")
    print(f"  Date range: {df['GAME_DATE'].min()} to {df['GAME_DATE'].max()}")
    
    return df


def merge_lines_and_results(lines_df: pd.DataFrame, results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge betting lines with game results.
    
    Args:
        lines_df: Game lines from the-odds-api
        results_df: Game results
    
    Returns:
        Merged DataFrame
    """
    print("\n🔗 Merging lines with results...")
    
    # Filter to BetMGM only for consistency
    lines_df = lines_df[lines_df['bookmaker'] == 'BetMGM'].copy()
    
    # Separate spreads and moneylines
    spreads = lines_df[lines_df['market'] == 'spread'][
        ['game_date', 'away_team', 'home_team', 'away_line', 'away_odds', 'home_line', 'home_odds']
    ].rename(columns={
        'away_line': 'away_spread',
        'away_odds': 'away_spread_odds',
        'home_line': 'home_spread',
        'home_odds': 'home_spread_odds'
    })
    
    moneylines = lines_df[lines_df['market'] == 'moneyline'][
        ['game_date', 'away_team', 'home_team', 'away_odds', 'home_odds']
    ].rename(columns={
        'away_odds': 'away_ml_odds',
        'home_odds': 'home_ml_odds'
    })
    
    # Merge spreads and moneylines
    lines_merged = spreads.merge(
        moneylines,
        on=['game_date', 'away_team', 'home_team'],
        how='inner'
    )
    
    print(f"  Merged {len(lines_merged):,} games with both spread and ML lines")
    
    # Merge with results
    # Match on date and teams
    merged = results_df.merge(
        lines_merged,
        left_on=['GAME_DATE', 'AWAY_TEAM', 'HOME_TEAM'],
        right_on=['game_date', 'away_team', 'home_team'],
        how='inner'
    )
    
    print(f"  ✅ Matched {len(merged):,} games with results")
    
    return merged


def create_underdog_dataset(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create dataset with underdog perspective (one row per underdog).
    
    Args:
        merged_df: Merged lines and results
    
    Returns:
        DataFrame with underdog games
    """
    print("\n🐕 Creating underdog dataset...")
    
    rows = []
    
    for _, row in merged_df.iterrows():
        # Determine underdog (positive ML odds)
        away_ml = row['away_ml_odds']
        home_ml = row['home_ml_odds']
        
        # Away team underdog
        if away_ml > 0:
            rows.append({
                'game_id': row['GAME_ID'],
                'game_date': row['GAME_DATE'],
                'team': row['AWAY_TEAM'],
                'opponent': row['HOME_TEAM'],
                'is_home': False,
                'spread_line': abs(row['away_spread']),  # Make positive
                'spread_odds': row['away_spread_odds'],
                'ml_odds': away_ml,
                'team_score': row['AWAY_SCORE'],
                'opp_score': row['HOME_SCORE'],
                'ml_won': row['AWAY_WL'] == 'W',
                'spread_covered': (row['AWAY_SCORE'] + row['away_spread']) > row['HOME_SCORE'],
            })
        
        # Home team underdog
        if home_ml > 0:
            rows.append({
                'game_id': row['GAME_ID'],
                'game_date': row['GAME_DATE'],
                'team': row['HOME_TEAM'],
                'opponent': row['AWAY_TEAM'],
                'is_home': True,
                'spread_line': abs(row['home_spread']),  # Make positive
                'spread_odds': row['home_spread_odds'],
                'ml_odds': home_ml,
                'team_score': row['HOME_SCORE'],
                'opp_score': row['AWAY_SCORE'],
                'ml_won': row['HOME_WL'] == 'W',
                'spread_covered': (row['HOME_SCORE'] + row['home_spread']) > row['AWAY_SCORE'],
            })
    
    df = pd.DataFrame(rows)
    
    print(f"  ✅ Created {len(df):,} underdog game records")
    print(f"  ML win rate: {df['ml_won'].mean():.1%}")
    print(f"  Spread cover rate: {df['spread_covered'].mean():.1%}")
    print(f"  Avg ML odds: +{df['ml_odds'].mean():.0f}")
    print(f"  Avg spread: +{df['spread_line'].mean():.1f}")
    
    return df


def analyze_cover_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze cover rates for favorites vs underdogs.
    
    Args:
        df: Underdog dataset
    
    Returns:
        DataFrame with cover rate analysis
    """
    print("\n📈 ANALYZING COVER RATES")
    print("="*60)
    
    results = []
    
    # Overall underdog rates
    results.append({
        'category': 'Underdog (All)',
        'sample_size': len(df),
        'spread_cover_rate': df['spread_covered'].mean(),
        'ml_win_rate': df['ml_won'].mean(),
        'both_win_rate': (df['ml_won'] & df['spread_covered']).mean(),
        'spread_win_ml_loss_rate': (df['spread_covered'] & ~df['ml_won']).mean(),
        'ml_win_spread_loss_rate': (df['ml_won'] & ~df['spread_covered']).mean(),
        'both_loss_rate': (~df['ml_won'] & ~df['spread_covered']).mean(),
    })
    
    # Conditional probability: P(ML win | spread covered)
    spread_covered = df[df['spread_covered']]
    if len(spread_covered) > 0:
        results.append({
            'category': 'P(ML win | Spread covered)',
            'sample_size': len(spread_covered),
            'spread_cover_rate': 1.0,
            'ml_win_rate': spread_covered['ml_won'].mean(),
            'both_win_rate': spread_covered['ml_won'].mean(),
            'spread_win_ml_loss_rate': (1 - spread_covered['ml_won'].mean()),
            'ml_win_spread_loss_rate': 0.0,
            'both_loss_rate': 0.0,
        })
    
    # By ML odds bins
    for low, high in CONFIG['ml_bins']:
        bin_df = df[(df['ml_odds'] >= low) & (df['ml_odds'] < high)]
        if len(bin_df) >= 10:
            results.append({
                'category': f'ML +{low}-{high}',
                'sample_size': len(bin_df),
                'spread_cover_rate': bin_df['spread_covered'].mean(),
                'ml_win_rate': bin_df['ml_won'].mean(),
                'both_win_rate': (bin_df['ml_won'] & bin_df['spread_covered']).mean(),
                'spread_win_ml_loss_rate': (bin_df['spread_covered'] & ~bin_df['ml_won']).mean(),
                'ml_win_spread_loss_rate': (bin_df['ml_won'] & ~bin_df['spread_covered']).mean(),
                'both_loss_rate': (~bin_df['ml_won'] & ~bin_df['spread_covered']).mean(),
            })
    
    # By spread bins
    for low, high in CONFIG['spread_bins']:
        bin_df = df[(df['spread_line'] >= low) & (df['spread_line'] < high)]
        if len(bin_df) >= 10:
            results.append({
                'category': f'Spread +{low}-{high}',
                'sample_size': len(bin_df),
                'spread_cover_rate': bin_df['spread_covered'].mean(),
                'ml_win_rate': bin_df['ml_won'].mean(),
                'both_win_rate': (bin_df['ml_won'] & bin_df['spread_covered']).mean(),
                'spread_win_ml_loss_rate': (bin_df['spread_covered'] & ~bin_df['ml_won']).mean(),
                'ml_win_spread_loss_rate': (bin_df['ml_won'] & ~bin_df['spread_covered']).mean(),
                'both_loss_rate': (~bin_df['ml_won'] & ~bin_df['spread_covered']).mean(),
            })
    
    results_df = pd.DataFrame(results)
    
    # Print results
    print("\nKey Metrics:")
    print(results_df.to_string(index=False))
    
    return results_df


# =============================================================================
# OPTIMIZATION ENGINE
# =============================================================================

def american_to_decimal(american_odds: float) -> float:
    """Convert American odds to decimal odds"""
    if american_odds < 0:
        return 1 + (100 / abs(american_odds))
    else:
        return 1 + (american_odds / 100)


def calculate_log_growth(df: pd.DataFrame, w_ml: float) -> float:
    """Calculate log growth rate for given ML weight"""
    w_spread = 1 - w_ml
    log_returns = []
    
    for _, row in df.iterrows():
        # ML return
        if row['ml_won']:
            r_ml = w_ml * (american_to_decimal(row['ml_odds']) - 1)
        else:
            r_ml = -w_ml
        
        # Spread return
        if row['spread_covered']:
            r_spread = w_spread * (american_to_decimal(row['spread_odds']) - 1)
        else:
            r_spread = -w_spread
        
        total_return = r_ml + r_spread
        
        if total_return <= -1:
            log_returns.append(-10)
        else:
            log_returns.append(np.log(1 + total_return))
    
    return np.mean(log_returns)


def calculate_performance_metrics(df: pd.DataFrame, w_ml: float) -> Dict:
    """Calculate performance metrics for given weight"""
    w_spread = 1 - w_ml
    
    bankroll = 1.0
    bankroll_history = [bankroll]
    
    for _, row in df.iterrows():
        if row['ml_won']:
            ml_profit = w_ml * bankroll * (american_to_decimal(row['ml_odds']) - 1)
        else:
            ml_profit = -w_ml * bankroll
        
        if row['spread_covered']:
            spread_profit = w_spread * bankroll * (american_to_decimal(row['spread_odds']) - 1)
        else:
            spread_profit = -w_spread * bankroll
        
        bankroll += ml_profit + spread_profit
        bankroll_history.append(bankroll)
    
    bankroll_history = np.array(bankroll_history)
    
    total_return = (bankroll - 1.0) * 100
    returns = np.diff(bankroll_history) / bankroll_history[:-1]
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0
    
    peak = np.maximum.accumulate(bankroll_history)
    drawdown = (bankroll_history - peak) / peak
    max_drawdown = abs(drawdown.min()) * 100
    
    return {
        'final_bankroll': bankroll,
        'total_return_pct': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown_pct': max_drawdown,
        'spread_win_rate': df['spread_covered'].mean(),
        'ml_win_rate': df['ml_won'].mean(),
    }


def optimize_generalized(df: pd.DataFrame) -> Dict:
    """Find single optimal weight across all games"""
    print("\n" + "="*60)
    print("🎯 OPTIMIZING GENERALIZED WEIGHT")
    print("="*60)
    
    def objective(w_ml):
        return -calculate_log_growth(df, w_ml)
    
    result = minimize_scalar(objective, bounds=(0.0, 1.0), method='bounded')
    
    optimal_w = result.x
    log_growth = -result.fun
    metrics = calculate_performance_metrics(df, optimal_w)
    
    results = {
        'method': 'generalized',
        'optimal_weight_ml': optimal_w,
        'optimal_weight_spread': 1 - optimal_w,
        'log_growth_rate': log_growth,
        'sample_size': len(df),
        **metrics
    }
    
    print_results(results)
    return results


def optimize_by_ml_bins(df: pd.DataFrame, bins: List[Tuple[int, int]]) -> Dict:
    """Find optimal weight for each ML price bin"""
    print("\n" + "="*60)
    print("💰 OPTIMIZING BY ML PRICE BINS")
    print("="*60)
    
    results = {}
    min_sample = CONFIG['min_sample_size']
    
    for low, high in bins:
        bin_name = f"+{low}-{high}"
        bin_df = df[(df['ml_odds'] >= low) & (df['ml_odds'] < high)]
        
        if len(bin_df) < min_sample:
            print(f"\n⏭️  Skipping {bin_name}: only {len(bin_df)} games (need {min_sample})")
            continue
        
        print(f"\n📈 Optimizing {bin_name} ({len(bin_df)} games)...")
        
        def objective(w_ml):
            return -calculate_log_growth(bin_df, w_ml)
        
        result = minimize_scalar(objective, bounds=(0.0, 1.0), method='bounded')
        
        optimal_w = result.x
        log_growth = -result.fun
        metrics = calculate_performance_metrics(bin_df, optimal_w)
        
        results[bin_name] = {
            'bin_range': (low, high),
            'optimal_weight_ml': optimal_w,
            'optimal_weight_spread': 1 - optimal_w,
            'log_growth_rate': log_growth,
            'sample_size': len(bin_df),
            **metrics
        }
        
        print(f"  ✅ Optimal ML weight: {optimal_w:.3f}")
    
    return results


def optimize_by_spread_bins(df: pd.DataFrame, bins: List[Tuple[float, float]]) -> Dict:
    """Find optimal weight for each spread bin"""
    print("\n" + "="*60)
    print("📏 OPTIMIZING BY SPREAD BINS")
    print("="*60)
    
    results = {}
    min_sample = CONFIG['min_sample_size']
    
    for low, high in bins:
        bin_name = f"+{low}-{high}"
        bin_df = df[(df['spread_line'] >= low) & (df['spread_line'] < high)]
        
        if len(bin_df) < min_sample:
            print(f"\n⏭️  Skipping {bin_name}: only {len(bin_df)} games (need {min_sample})")
            continue
        
        print(f"\n📈 Optimizing {bin_name} ({len(bin_df)} games)...")
        
        def objective(w_ml):
            return -calculate_log_growth(bin_df, w_ml)
        
        result = minimize_scalar(objective, bounds=(0.0, 1.0), method='bounded')
        
        optimal_w = result.x
        log_growth = -result.fun
        metrics = calculate_performance_metrics(bin_df, optimal_w)
        
        results[bin_name] = {
            'bin_range': (low, high),
            'optimal_weight_ml': optimal_w,
            'optimal_weight_spread': 1 - optimal_w,
            'log_growth_rate': log_growth,
            'sample_size': len(bin_df),
            **metrics
        }
        
        print(f"  ✅ Optimal ML weight: {optimal_w:.3f}")
    
    return results


def print_results(results: Dict) -> None:
    """Pretty print optimization results"""
    print(f"\n{'='*60}")
    print(f"✨ Optimal ML Weight: {results['optimal_weight_ml']:.3f}")
    print(f"✨ Optimal Spread Weight: {results['optimal_weight_spread']:.3f}")
    print(f"📊 Log Growth Rate: {results['log_growth_rate']:.6f}")
    print(f"🎲 Sample Size: {results['sample_size']} games")
    print(f"\n📈 Performance Metrics:")
    print(f"  Total Return: {results['total_return_pct']:+.2f}%")
    print(f"  Sharpe Ratio: {results['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown: {results['max_drawdown_pct']:.2f}%")
    print(f"  Spread Win Rate: {results['spread_win_rate']:.1%}")
    print(f"  ML Win Rate: {results['ml_win_rate']:.1%}")
    print(f"{'='*60}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Optimize spread/ML weight allocation')
    parser.add_argument('--sport', type=str, required=True, 
                       choices=['nba', 'nfl', 'ncaab', 'ncaaf'],
                       help='Sport to analyze')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['generalized', 'ml_bins', 'spread_bins', 'all'],
                       help='Optimization mode')
    parser.add_argument('--seasons', nargs='+', required=True,
                       help='Seasons to analyze (e.g., 2025-26 2024-25)')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze cover rates, skip optimization')
    parser.add_argument('--cache-check', type=lambda x: x.lower() == 'true', default=False,
                       help='Check cache before loading from S3 (true/false)')
    
    args = parser.parse_args()
    
    if args.sport != 'nba':
        raise NotImplementedError(f"{args.sport.upper()} not yet implemented. Use --sport nba")
    
    print("\n" + "="*60)
    print(f"🎲 SPREAD/ML WEIGHT OPTIMIZER - {args.sport.upper()}")
    print("="*60)
    print(f"Seasons: {', '.join(args.seasons)}")
    print(f"Cache: {'enabled' if args.cache_check else 'disabled'}")
    
    # Load data (with caching)
    all_season_data = []
    
    for season in args.seasons:
        cache_path = get_cache_path(args.sport, season, 'merged')
        
        # Try to load from cache
        if args.cache_check:
            cached_df = load_from_cache(cache_path)
            if cached_df is not None:
                all_season_data.append(cached_df)
                continue
        
        # Load from S3
        print(f"\n🔄 Loading {season} from S3...")
        lines_df = load_nba_game_lines([season])
        results_df = load_nba_game_results([season])
        merged_df = merge_lines_and_results(lines_df, results_df)
        
        # Save to cache
        if args.cache_check:
            save_to_cache(merged_df, cache_path)
        
        all_season_data.append(merged_df)
    
    # Combine all seasons
    print("\n🔗 Combining all seasons...")
    merged_df = pd.concat(all_season_data, ignore_index=True)
    print(f"  ✅ Total games: {len(merged_df):,}")
    
    underdog_df = create_underdog_dataset(merged_df)
    
    # Setup local temp directory for any local files
    local_tmp_dir = Path.home() / 'Downloads' / 'tmp' / 'spread_ml_optimization' / args.sport
    local_tmp_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup S3 output path
    s3_bucket = CONFIG['s3']['output_bucket']
    s3_prefix = f"{CONFIG['s3']['output_prefix']}/{args.sport}"
    season_suffix = '_'.join(args.seasons)
    
    # Analyze cover rates
    cover_analysis = analyze_cover_rates(underdog_df)
    
    # Save cover analysis to S3
    s3_key = f"{s3_prefix}/cover_analysis_{season_suffix}.csv"
    s3_uri = upload_df_to_s3(cover_analysis, s3_bucket, s3_key)
    print(f"\n💾 Saved cover analysis to S3: {s3_uri}")
    
    # Also save locally to tmp
    local_path = local_tmp_dir / f'cover_analysis_{season_suffix}.csv'
    cover_analysis.to_csv(local_path, index=False)
    print(f"💾 Saved locally: {local_path}")
    
    if args.analyze_only:
        print("\n✅ Analysis complete (--analyze-only flag set)")
        return
    
    # Run optimizations
    gen_results = None
    ml_results = {}
    spread_results = {}
    
    if args.mode in ['generalized', 'all']:
        gen_results = optimize_generalized(underdog_df)
    
    if args.mode in ['ml_bins', 'all']:
        ml_results = optimize_by_ml_bins(underdog_df, CONFIG['ml_bins'])
    
    if args.mode in ['spread_bins', 'all']:
        spread_results = optimize_by_spread_bins(underdog_df, CONFIG['spread_bins'])
    
    # Save results to S3 and local tmp
    print("\n" + "="*60)
    print("💾 SAVING RESULTS")
    print("="*60)
    
    if gen_results:
        gen_df = pd.DataFrame([gen_results])
        
        # Save to S3
        s3_key = f"{s3_prefix}/generalized_results_{season_suffix}.csv"
        s3_uri = upload_df_to_s3(gen_df, s3_bucket, s3_key)
        print(f"  ✅ S3: {s3_uri}")
        
        # Save locally
        local_path = local_tmp_dir / f'generalized_results_{season_suffix}.csv'
        gen_df.to_csv(local_path, index=False)
        print(f"  ✅ Local: {local_path}")
    
    if ml_results:
        ml_df = pd.DataFrame(ml_results).T
        
        # Save to S3
        s3_key = f"{s3_prefix}/ml_bins_results_{season_suffix}.csv"
        s3_uri = upload_df_to_s3(ml_df, s3_bucket, s3_key)
        print(f"  ✅ S3: {s3_uri}")
        
        # Save locally
        local_path = local_tmp_dir / f'ml_bins_results_{season_suffix}.csv'
        ml_df.to_csv(local_path)
        print(f"  ✅ Local: {local_path}")
    
    if spread_results:
        spread_df = pd.DataFrame(spread_results).T
        
        # Save to S3
        s3_key = f"{s3_prefix}/spread_bins_results_{season_suffix}.csv"
        s3_uri = upload_df_to_s3(spread_df, s3_bucket, s3_key)
        print(f"  ✅ S3: {s3_uri}")
        
        # Save locally
        local_path = local_tmp_dir / f'spread_bins_results_{season_suffix}.csv'
        spread_df.to_csv(local_path)
        print(f"  ✅ Local: {local_path}")
    
    # Summary
    print("\n" + "="*60)
    print("✅ OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"\n📁 S3: s3://{s3_bucket}/{s3_prefix}/")
    print(f"📁 Local: {local_tmp_dir}")
    
    if gen_results:
        print(f"\n🎯 Key Finding (Generalized):")
        print(f"  Optimal ML weight: {gen_results['optimal_weight_ml']:.3f}")
        print(f"  Optimal spread weight: {gen_results['optimal_weight_spread']:.3f}")
        print(f"  Expected return: {gen_results['total_return_pct']:+.2f}%")
        print(f"  Sharpe ratio: {gen_results['sharpe_ratio']:.3f}")


if __name__ == '__main__':
    main()
