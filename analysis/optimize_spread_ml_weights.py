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
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend - don't show plots on screen
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
    'line_source': 'consensus',  # Options: 'consensus', 'betmgm', 'draftkings', etc.
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
    },
    'team_name_mapping': {
        # ESPN → Odds API mapping
        'LA Clippers': 'Los Angeles Clippers',
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
    
    # Convert to ET timezone BEFORE extracting date
    df['game_time'] = pd.to_datetime(df['game_time'], utc=True)
    df['game_time'] = df['game_time'].dt.tz_convert('US/Eastern')
    df['game_date'] = df['game_time'].dt.date
    
    print(f"  ✅ Loaded {len(df):,} line records")
    print(f"  Date range (ET): {df['game_date'].min()} to {df['game_date'].max()}")
    
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
    
    # Filter out postponed/cancelled games (0-0 scores)
    df = df[(df['AWAY_SCORE'] > 0) | (df['HOME_SCORE'] > 0)].copy()
    
    # Convert to ET timezone (ESPN data is already in ET, but make it explicit)
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE']).dt.date
    
    # Normalize team names (ESPN → Odds API format)
    team_mapping = CONFIG['team_name_mapping']
    df['AWAY_TEAM'] = df['AWAY_TEAM'].replace(team_mapping)
    df['HOME_TEAM'] = df['HOME_TEAM'].replace(team_mapping)
    
    print(f"  ✅ Loaded {len(df):,} game results")
    print(f"  Date range (ET): {df['GAME_DATE'].min()} to {df['GAME_DATE'].max()}")
    
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
    
    line_source = CONFIG['line_source']
    print(f"  Line source: {line_source}")
    
    if line_source == 'consensus':
        # Calculate consensus spreads (mean across all books)
        spreads = lines_df[lines_df['market'] == 'spread'].groupby(
            ['game_date', 'away_team', 'home_team']
        ).agg({
            'away_line': 'mean',
            'away_odds': 'mean',
            'home_line': 'mean',
            'home_odds': 'mean',
            'bookmaker': 'count'
        }).rename(columns={
            'away_line': 'away_spread',
            'away_odds': 'away_spread_odds',
            'home_line': 'home_spread',
            'home_odds': 'home_spread_odds',
            'bookmaker': 'num_books_spread'
        }).reset_index()
        
        # Calculate consensus moneylines (mean across all books)
        moneylines = lines_df[lines_df['market'] == 'moneyline'].groupby(
            ['game_date', 'away_team', 'home_team']
        ).agg({
            'away_odds': 'mean',
            'home_odds': 'mean',
            'bookmaker': 'count'
        }).rename(columns={
            'away_odds': 'away_ml_odds',
            'home_odds': 'home_ml_odds',
            'bookmaker': 'num_books_ml'
        }).reset_index()
        
        # Show consensus stats
        print(f"  Avg books per game (spread): {spreads['num_books_spread'].mean():.1f}")
        print(f"  Avg books per game (ML): {moneylines['num_books_ml'].mean():.1f}")
        
    else:
        # Filter to specific bookmaker
        print(f"  Filtering to {line_source} only")
        lines_df = lines_df[lines_df['bookmaker'] == line_source].copy()
        
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
    
    # Check match rate
    match_rate = len(merged) / len(results_df) * 100
    if match_rate < 95:
        unmatched = len(results_df) - len(merged)
        print(f"  ⚠️  Warning: {unmatched} games ({100-match_rate:.1f}%) had no matching lines")
    
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
    
    # Add spread distribution summary
    print("\n📊 UNDERDOG SPREAD DISTRIBUTION")
    print("="*60)
    print(f"Total underdogs: {len(df)}")
    print(f"\nSpread statistics:")
    print(f"  Min:  +{df['spread_line'].min():.1f}")
    print(f"  Max:  +{df['spread_line'].max():.1f}")
    print(f"  Mean: +{df['spread_line'].mean():.1f}")
    print(f"  Median: +{df['spread_line'].median():.1f}")
    
    print(f"\nPercentiles:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        val = df['spread_line'].quantile(p/100)
        print(f"  {p}th: +{val:.1f}")
    
    # Show most common spreads
    spread_counts = df['spread_line'].value_counts().head(10)
    print(f"\nMost common spreads:")
    for spread, count in spread_counts.items():
        pct = count / len(df) * 100
        print(f"  +{spread:.1f}: {count:4d} games ({pct:5.1f}%)")
    
    # Check for any underdogs with tiny spreads (might indicate data issues)
    tiny_spreads = df[df['spread_line'] <= 1.0]
    if len(tiny_spreads) > 0:
        print(f"\n⚠️  {len(tiny_spreads)} underdogs with spreads ≤ +1.0")
    else:
        print(f"\n✓ No underdogs with spreads ≤ +1.0")
    
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
    
    # Conditional probability: P(ML win | spread NOT covered) - should be 0
    spread_not_covered = df[~df['spread_covered']]
    if len(spread_not_covered) > 0:
        results.append({
            'category': 'P(ML win | Spread NOT covered)',
            'sample_size': len(spread_not_covered),
            'spread_cover_rate': 0.0,
            'ml_win_rate': spread_not_covered['ml_won'].mean(),
            'both_win_rate': 0.0,
            'spread_win_ml_loss_rate': 0.0,
            'ml_win_spread_loss_rate': spread_not_covered['ml_won'].mean(),
            'both_loss_rate': (1 - spread_not_covered['ml_won'].mean()),
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
# PLOTTING FUNCTIONS
# =============================================================================

def analyze_underdog_spread_distribution(underdog_df: pd.DataFrame) -> None:
    """
    Print distribution statistics for underdog spreads.
    
    Args:
        underdog_df: Underdog dataset
    """
    print("\n📊 UNDERDOG SPREAD DISTRIBUTION")
    print("="*60)
    
    spreads = underdog_df['spread_line']
    
    print(f"Total underdogs: {len(spreads):,}")
    print(f"\nSpread statistics:")
    print(f"  Min:  +{spreads.min():.1f}")
    print(f"  Max:  +{spreads.max():.1f}")
    print(f"  Mean: +{spreads.mean():.1f}")
    print(f"  Median: +{spreads.median():.1f}")
    
    print(f"\nPercentiles:")
    for pct in [10, 25, 50, 75, 90, 95, 99]:
        val = spreads.quantile(pct/100)
        print(f"  {pct}th: +{val:.1f}")
    
    print(f"\nMost common spreads:")
    top_spreads = spreads.value_counts().head(10)
    for spread, count in top_spreads.items():
        pct = count / len(spreads) * 100
        print(f"  +{spread:.1f}: {count:4d} games ({pct:5.1f}%)")
    
    # Check for small spreads
    small_spreads = spreads[spreads <= 1.0]
    if len(small_spreads) > 0:
        print(f"\n⚠️  Found {len(small_spreads)} underdogs with spreads ≤ +1.0:")
        for spread in sorted(small_spreads.unique()):
            count = (spreads == spread).sum()
            print(f"  +{spread:.1f}: {count} games")
    else:
        print(f"\n✓ No underdogs with spreads ≤ +1.0")


def create_spread_line_plots(merged_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create 3 plots analyzing underdog performance by specific spread line.
    
    Args:
        merged_df: Merged game data with lines and results
        output_dir: Directory to save plots
    """
    print("\n📊 Creating spread line plots (underdog perspective)...")
    
    # Prepare data - UNDERDOG PERSPECTIVE ONLY
    rows = []
    
    for _, row in merged_df.iterrows():
        away_ml = row['away_ml_odds']
        home_ml = row['home_ml_odds']
        
        # Away team is underdog (positive ML)
        if away_ml > 0:
            rows.append({
                'team': row['AWAY_TEAM'],
                'spread': abs(row['away_spread']),  # Make positive
                'won': row['AWAY_WL'] == 'W',
                'covered': (row['AWAY_SCORE'] + row['away_spread']) > row['HOME_SCORE'],
            })
        
        # Home team is underdog (positive ML)
        if home_ml > 0:
            rows.append({
                'team': row['HOME_TEAM'],
                'spread': abs(row['home_spread']),  # Make positive
                'won': row['HOME_WL'] == 'W',
                'covered': (row['HOME_SCORE'] + row['home_spread']) > row['AWAY_SCORE'],
            })
    
    df = pd.DataFrame(rows)
    
    # Floor spread to whole numbers for binning (1.5 → 1, 2.5 → 2, etc.)
    df['spread_bin'] = df['spread'].astype(int)
    
    # Cap at 15 (bin larger spreads together)
    df.loc[df['spread_bin'] > 15, 'spread_bin'] = 15
    
    # Calculate stats by spread bin
    stats = df.groupby('spread_bin').agg({
        'won': ['sum', 'count', 'mean'],
        'covered': ['sum', 'mean']
    }).reset_index()
    
    stats.columns = ['spread_bin', 'wins', 'total_games', 'win_pct', 'covers', 'cover_pct']
    
    # Calculate P(win | covered) for each bin
    covered_df = df[df['covered']].groupby('spread_bin')['won'].agg(['sum', 'count']).reset_index()
    covered_df.columns = ['spread_bin', 'wins_when_covered', 'covers']
    covered_df['win_given_cover_pct'] = covered_df['wins_when_covered'] / covered_df['covers']
    
    stats = stats.merge(covered_df[['spread_bin', 'win_given_cover_pct']], on='spread_bin', how='left')
    
    # Filter to bins with at least 10 games
    stats = stats[stats['total_games'] >= 10].copy()
    
    # Sort by spread
    stats = stats.sort_values('spread_bin')
    
    print(f"  Spread bins with 10+ games: {len(stats)}")
    print(f"  Spread range: +{stats['spread_bin'].min():.1f} to +{stats['spread_bin'].max():.1f}")
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(16, 18))
    gs = fig.add_gridspec(3, 1, hspace=0.3)
    
    # Calculate overall ML W-L and Spread W-L records
    ml_wins = df['won'].sum()
    ml_losses = len(df) - ml_wins
    ml_win_pct = (ml_wins / len(df)) * 100
    
    spread_covers = df['covered'].sum()
    spread_losses = len(df) - spread_covers
    spread_cover_pct = (spread_covers / len(df)) * 100
    
    # Add overall title with sample info
    fig.suptitle(f'Underdog Spread Analysis - Aggregated Across All Seasons\n'
                 f'n={len(df):,} | ML: {ml_wins:,}-{ml_losses:,} ({ml_win_pct:.1f}%) | Spread: {spread_covers:,}-{spread_losses:,} ({spread_cover_pct:.1f}%) | 2021-22 through 2025-26',
                 fontsize=15, fontweight='bold', y=0.995)
    
    axes = [fig.add_subplot(gs[i]) for i in range(3)]
    
    # Plot 1: Win % by spread line (UNDERDOGS) - bars + line
    ax1 = axes[0]
    ax1.bar(stats['spread_bin'], stats['win_pct'] * 100, alpha=0.3, color='#1f77b4', width=0.4)
    ax1.plot(stats['spread_bin'], stats['win_pct'] * 100, marker='o', linewidth=2.5, markersize=8, color='#1f77b4')
    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
    ax1.set_xlabel('Underdog Spread Line (+points)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Win %', fontsize=12, fontweight='bold')
    ax1.set_title('Underdog Win % by Spread Line', fontsize=13, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend()
    ax1.set_ylim(0, 100)
    ax1.set_xlim(0, 16)
    
    # Set x-axis ticks every 1 unit
    ax1.set_xticks(range(0, 17, 1))
    
    # Add sample size annotations for significant bins
    for _, row in stats.iterrows():
        if row['total_games'] >= 30:
            ax1.annotate(f"n={int(row['total_games'])}", 
                        xy=(row['spread_bin'], row['win_pct'] * 100),
                        xytext=(0, 10), textcoords='offset points',
                        ha='center', fontsize=8, alpha=0.6)
    
    # Plot 2: Cover % by spread line (UNDERDOGS) - bars + line
    ax2 = axes[1]
    ax2.bar(stats['spread_bin'], stats['cover_pct'] * 100, alpha=0.3, color='#ff7f0e', width=0.4)
    ax2.plot(stats['spread_bin'], stats['cover_pct'] * 100, marker='s', linewidth=2.5, 
             markersize=8, color='#ff7f0e')
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
    ax2.set_xlabel('Underdog Spread Line (+points)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cover %', fontsize=12, fontweight='bold')
    ax2.set_title('Underdog Cover % by Spread Line', fontsize=13, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()
    ax2.set_ylim(0, 100)
    ax2.set_xlim(0, 16)
    
    # Set x-axis ticks every 1 unit
    ax2.set_xticks(range(0, 17, 1))
    
    # Plot 3: Win % given cover (UNDERDOGS) - bars + line
    ax3 = axes[2]
    ax3.bar(stats['spread_bin'], stats['win_given_cover_pct'] * 100, alpha=0.3, color='#2ca02c', width=0.4)
    ax3.plot(stats['spread_bin'], stats['win_given_cover_pct'] * 100, marker='^', 
             linewidth=2.5, markersize=8, color='#2ca02c')
    ax3.axhline(y=100, color='gray', linestyle='--', alpha=0.3, label='100% (would mean always win when cover)')
    ax3.set_xlabel('Underdog Spread Line (+points)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Win % (Given Cover)', fontsize=12, fontweight='bold')
    ax3.set_title('P(Underdog Wins | Covers Spread) by Spread Line', fontsize=13, fontweight='bold', pad=15)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend()
    ax3.set_ylim(0, 100)
    ax3.set_xlim(0, 16)
    
    # Set x-axis ticks every 1 unit
    ax3.set_xticks(range(0, 17, 1))
    
    # Add explanatory note
    ax3.text(0.02, 0.98, 'When underdogs cover:\n- Small spreads (+0.5 to +3): Usually won outright\n- Large spreads (+10+): Often just lost by less than the spread',
             transform=ax3.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / 'underdog_spread_line_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved plot: {plot_path}")
    
    # Also save the data
    data_path = output_dir / 'underdog_spread_line_stats.csv'
    stats.to_csv(data_path, index=False)
    print(f"  ✅ Saved data: {data_path}")
    
    plt.close()


def create_spread_line_plots_by_season(merged_df: pd.DataFrame, seasons: List[str], output_dir: Path) -> None:
    """
    Create 3 plots analyzing underdog performance by season (grouped bars + lines).
    
    Args:
        merged_df: Merged game data with lines and results
        seasons: List of seasons to analyze
        output_dir: Directory to save plots
    """
    print("\n📊 Creating spread line plots by season (grouped bars)...")
    
    # Prepare data - UNDERDOG PERSPECTIVE with season tracking
    rows = []
    
    for _, row in merged_df.iterrows():
        away_ml = row['away_ml_odds']
        home_ml = row['home_ml_odds']
        game_date = pd.to_datetime(row['GAME_DATE'])
        
        # Determine season from game date
        if game_date.month >= 10:
            season = f"{game_date.year}-{str(game_date.year + 1)[-2:]}"
        else:
            season = f"{game_date.year - 1}-{str(game_date.year)[-2:]}"
        
        # Away team is underdog (positive ML)
        if away_ml > 0:
            rows.append({
                'season': season,
                'spread': abs(row['away_spread']),
                'won': row['AWAY_WL'] == 'W',
                'covered': (row['AWAY_SCORE'] + row['away_spread']) > row['HOME_SCORE'],
            })
        
        # Home team is underdog (positive ML)
        if home_ml > 0:
            rows.append({
                'season': season,
                'spread': abs(row['home_spread']),
                'won': row['HOME_WL'] == 'W',
                'covered': (row['HOME_SCORE'] + row['home_spread']) > row['AWAY_SCORE'],
            })
    
    df = pd.DataFrame(rows)
    
    # Floor spread to whole numbers for binning (1.5 → 1, 2.5 → 2, etc.)
    df['spread_bin'] = df['spread'].astype(int)
    
    # Cap at 15
    df.loc[df['spread_bin'] > 15, 'spread_bin'] = 15
    
    # Calculate stats by spread bin AND season
    stats = df.groupby(['spread_bin', 'season']).agg({
        'won': ['sum', 'count', 'mean'],
        'covered': ['sum', 'mean']
    }).reset_index()
    
    stats.columns = ['spread_bin', 'season', 'wins', 'total_games', 'win_pct', 'covers', 'cover_pct']
    
    # Calculate P(win | covered) for each bin + season
    covered_df = df[df['covered']].groupby(['spread_bin', 'season'])['won'].agg(['sum', 'count']).reset_index()
    covered_df.columns = ['spread_bin', 'season', 'wins_when_covered', 'covers']
    covered_df['win_given_cover_pct'] = covered_df['wins_when_covered'] / covered_df['covers']
    
    stats = stats.merge(covered_df[['spread_bin', 'season', 'win_given_cover_pct']], 
                        on=['spread_bin', 'season'], how='left')
    
    # Filter to bins with at least 10 games PER SEASON
    stats = stats[stats['total_games'] >= 10].copy()
    
    print(f"  Total season-spread combinations with 10+ games: {len(stats)}")
    
    # Define colors for each season
    season_colors = {
        '2025-26': '#1f77b4',  # blue
        '2024-25': '#ff7f0e',  # orange
        '2023-24': '#2ca02c',  # green
        '2022-23': '#d62728',  # red
        '2021-22': '#9467bd',  # purple
    }
    
    # Get unique spread bins that appear in data
    spread_bins = sorted(stats['spread_bin'].unique())
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(18, 20))
    gs = fig.add_gridspec(3, 1, hspace=0.35)
    
    # Add clean title
    fig.suptitle('Underdog Spread Analysis - By Season Comparison',
                 fontsize=14, fontweight='bold', y=0.995)
    
    # Prepare season records for info box
    season_records = []
    for s in seasons:
        season_df = df[df['season'] == s]
        n = len(season_df)
        ml_w = season_df['won'].sum()
        ml_l = n - ml_w
        spread_w = season_df['covered'].sum()
        spread_l = n - spread_w
        season_records.append(f"{s}: n={n:,} ML:{ml_w}-{ml_l} ATS:{spread_w}-{spread_l}")
    
    axes = [fig.add_subplot(gs[i]) for i in range(3)]
    
    # Calculate bar positions
    bar_width = 0.15
    n_seasons = len(seasons)
    
    # Plot 1: Win % by spread line - grouped bars + lines
    ax1 = axes[0]
    for i, season in enumerate(seasons):
        season_data = stats[stats['season'] == season].copy()
        
        # Bar positions offset for grouping
        x_positions = [x + (i - n_seasons/2 + 0.5) * bar_width for x in season_data['spread_bin']]
        
        # Bars
        ax1.bar(x_positions, season_data['win_pct'] * 100, 
                width=bar_width, alpha=0.6, color=season_colors.get(season, 'gray'),
                label=season)
        
        # Line connecting points
        ax1.plot(season_data['spread_bin'], season_data['win_pct'] * 100,
                marker='o', linewidth=1.5, markersize=5, 
                color=season_colors.get(season, 'gray'))
    
    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax1.set_xlabel('Underdog Spread Line (+points)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Win %', fontsize=12, fontweight='bold')
    ax1.set_title('Underdog Win % by Spread Line (By Season)', fontsize=13, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_ylim(0, 100)
    ax1.set_xlim(0, 16)
    ax1.set_xticks(range(0, 17, 1))
    
    # Plot 2: Cover % by spread line - grouped bars + lines
    ax2 = axes[1]
    for i, season in enumerate(seasons):
        season_data = stats[stats['season'] == season].copy()
        
        x_positions = [x + (i - n_seasons/2 + 0.5) * bar_width for x in season_data['spread_bin']]
        
        ax2.bar(x_positions, season_data['cover_pct'] * 100,
                width=bar_width, alpha=0.6, color=season_colors.get(season, 'gray'),
                label=season)
        
        ax2.plot(season_data['spread_bin'], season_data['cover_pct'] * 100,
                marker='s', linewidth=1.5, markersize=5,
                color=season_colors.get(season, 'gray'))
    
    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax2.set_xlabel('Underdog Spread Line (+points)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cover %', fontsize=12, fontweight='bold')
    ax2.set_title('Underdog Cover % by Spread Line (By Season)', fontsize=13, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_ylim(0, 100)
    ax2.set_xlim(0, 16)
    ax2.set_xticks(range(0, 17, 1))
    
    # Plot 3: Win % given cover - grouped bars + lines
    ax3 = axes[2]
    for i, season in enumerate(seasons):
        season_data = stats[stats['season'] == season].copy()
        
        x_positions = [x + (i - n_seasons/2 + 0.5) * bar_width for x in season_data['spread_bin']]
        
        ax3.bar(x_positions, season_data['win_given_cover_pct'] * 100,
                width=bar_width, alpha=0.6, color=season_colors.get(season, 'gray'),
                label=season)
        
        ax3.plot(season_data['spread_bin'], season_data['win_given_cover_pct'] * 100,
                marker='^', linewidth=1.5, markersize=5,
                color=season_colors.get(season, 'gray'))
    
    ax3.axhline(y=100, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax3.set_xlabel('Underdog Spread Line (+points)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Win % (Given Cover)', fontsize=12, fontweight='bold')
    ax3.set_title('P(Underdog Wins | Covers Spread) by Spread Line (By Season)', fontsize=13, fontweight='bold', pad=15)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend(loc='upper right', fontsize=10)
    ax3.set_ylim(0, 100)
    ax3.set_xlim(0, 16)
    ax3.set_xticks(range(0, 17, 1))
    
    # Add season records info box to first plot (left side)
    info_text = 'Season Records:\n' + '\n'.join(season_records)
    ax1.text(0.02, 0.97, info_text,
             transform=ax1.transAxes, fontsize=8,
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    plt.tight_layout()
    
    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / 'underdog_spread_line_analysis_by_season.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved plot: {plot_path}")
    
    # Also save the data
    data_path = output_dir / 'underdog_spread_line_stats_by_season.csv'
    stats.to_csv(data_path, index=False)
    print(f"  ✅ Saved data: {data_path}")
    
    plt.close()


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
    
    # Analyze underdog spread distribution
    analyze_underdog_spread_distribution(underdog_df)
    
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
    
    # Create spread line plots
    create_spread_line_plots(merged_df, local_tmp_dir)
    
    # Create spread line plots by season (grouped bars)
    create_spread_line_plots_by_season(merged_df, args.seasons, local_tmp_dir)
    
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
