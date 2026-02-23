"""
NBA Props Stability & Predictive Power Analysis

Analyzes which NBA prop markets have the most predictive power for momentum
and reversal strategies.

Inspired by graduate school research on scoring stability and predictive power.
This script applies the same methodology to NBA betting markets:
- Stability Analysis: Does recent performance predict future performance?
- Line Prediction: Does beating the line predict future line results?

Context:
Based on grad school research comparing box score statistics and ShotQuality 
expectation metrics. Published article:
https://rpubs.com/mylesthomas/predictive-power-nba-statistics

Research showed:
- Box score stats (FG%, 3PT%) are highly variable game-to-game
- Expectation metrics show much better stability across 5-game windows
- More stable metrics = better for predictive models

This Analysis:
Applies same methodology to 9 NBA prop markets to identify which markets
are most predictable and suitable for momentum/reversal betting strategies.

Data Sources:
- Player props: s3://the-odds-api-mt/nba/historical_player_props/{season}/
- Game logs: s3://nba-api-mt/player_game_logs/{season}/

Output:
- ~/Downloads/tmp/prop_predictive_power_analysis/{season}/
  - 01_data/props_with_actuals.csv
  - 02_analysis/stability_by_market.csv
  - 02_analysis/line_prediction_by_market.csv
  - 02_analysis/combined_market_rankings.csv
  - 03_visualizations/stability_comparison.png
  - 03_visualizations/prediction_accuracy.png
  - 03_visualizations/momentum_vs_reversal.png

Usage:
    # Single season (default: 2025-26)
    python analysis/analyze_nba_props_stability_and_predictive_power.py
    
    # Specific season
    python analysis/analyze_nba_props_stability_and_predictive_power.py --season 2024-25
    
    # Multiple seasons
    python analysis/analyze_nba_props_stability_and_predictive_power.py --seasons 2023-24 2024-25 2025-26
    
    # Specific markets only
    python analysis/analyze_nba_props_stability_and_predictive_power.py --markets player_points player_rebounds
    
    # Custom bin size
    python analysis/analyze_nba_props_stability_and_predictive_power.py --bin-size 8
    
    # Skip visualization
    python analysis/analyze_nba_props_stability_and_predictive_power.py --no-viz

Author: Myles Thomas
Date: 2026-02-10
"""

import sys
import os
from pathlib import Path
import argparse
from io import StringIO
from datetime import datetime
import unicodedata
import warnings

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import boto3
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# S3 Configuration
S3_BUCKET_PROPS = 'the-odds-api-mt'
S3_BUCKET_NBA = 'nba-api-mt'

# Analysis Parameters
DEFAULT_BIN_SIZE = 5
DEFAULT_MIN_GAMES = 10
DEFAULT_SEASONS = ['2025-26']

# Market to NBA API stat mapping
MARKET_STAT_MAP = {
    'player_points': 'PTS',
    'player_rebounds': 'REB',
    'player_assists': 'AST',
    'player_threes': 'FG3M',
    'player_blocks': 'BLK',
    'player_steals': 'STL',
    'player_double_double': None,  # Calculated
    'player_triple_double': None,  # Calculated
    'player_points_rebounds_assists': None,  # Calculated
}

ALL_MARKETS = list(MARKET_STAT_MAP.keys())

# Output directory
OUTPUT_BASE = Path.home() / 'Downloads' / 'tmp' / 'prop_predictive_power_analysis'

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def remove_accents(text):
    """Remove accents from unicode string"""
    if pd.isna(text):
        return text
    nfd = unicodedata.normalize('NFD', text)
    return ''.join(char for char in nfd if unicodedata.category(char) != 'Mn')


def normalize_player_name(name):
    """
    Normalize player name for consistent matching across data sources.
    
    Rules:
    1. Remove periods (P.J. -> PJ)
    2. Title case
    3. Remove accents
    4. Remove generational suffixes (III, II, IV, V)
    5. Apply known mappings
    """
    if pd.isna(name):
        return name
    
    name = name.strip().replace('.', '').title()
    name = remove_accents(name)
    
    # Remove generational suffixes at end
    if name.endswith(' Iii'):
        name = name[:-4]
    elif name.endswith(' Ii'):
        name = name[:-3]
    elif name.endswith(' Iv'):
        name = name[:-3]
    elif name.endswith(' V'):
        name = name[:-2]
    
    name = ' '.join(name.split())
    
    # Known name mappings (Odds API -> NBA API)
    mappings = {
        'Herb Jones': 'Herbert Jones',
        'Moe Wagner': 'Moritz Wagner',
        'Nicolas Claxton': 'Nic Claxton',
        'Ron Holland': 'Ronald Holland',
        'Vincent Williams Jr': 'Vince Williams Jr',
        'Derrick Jones': 'Derrick Jones Jr',
        'Bruce Brown Jr': 'Bruce Brown',
        'Kenyon Martin Jr': 'Kj Martin',
        'Paul Reed Jr': 'Paul Reed',
        'Carlton Carrington': 'Bub Carrington',
        'Alfred Joel Horford Reynoso': 'Al Horford',
        'Anthony Davis Jr': 'Anthony Davis',
    }
    
    return mappings.get(name, name)


def calculate_actual_value(row, market):
    """Calculate actual value for a given market"""
    if market == 'player_points':
        return row.get('PTS')
    elif market == 'player_rebounds':
        return row.get('REB')
    elif market == 'player_assists':
        return row.get('AST')
    elif market == 'player_threes':
        return row.get('FG3M')
    elif market == 'player_blocks':
        return row.get('BLK')
    elif market == 'player_steals':
        return row.get('STL')
    elif market == 'player_double_double':
        # Count stats >= 10
        stats = [row.get('PTS', 0), row.get('REB', 0), row.get('AST', 0), 
                 row.get('STL', 0), row.get('BLK', 0)]
        return sum(1 for s in stats if s >= 10)
    elif market == 'player_triple_double':
        stats = [row.get('PTS', 0), row.get('REB', 0), row.get('AST', 0), 
                 row.get('STL', 0), row.get('BLK', 0)]
        return sum(1 for s in stats if s >= 10)
    elif market == 'player_points_rebounds_assists':
        pts = row.get('PTS', 0)
        reb = row.get('REB', 0)
        ast = row.get('AST', 0)
        return pts + reb + ast
    else:
        return None


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_props_from_s3(season):
    """Load player props for a season from S3"""
    print(f"\n📊 Loading props from S3 for {season}...")
    
    s3_client = boto3.client('s3')
    prefix = f"nba/historical_player_props/{season}/"
    
    try:
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_PROPS, Prefix=prefix)
        
        if 'Contents' not in response:
            print(f"   ❌ No props files found for {season}")
            return pd.DataFrame()
        
        all_props = []
        for obj in response['Contents']:
            if obj['Key'].endswith('.csv'):
                try:
                    obj_data = s3_client.get_object(Bucket=S3_BUCKET_PROPS, Key=obj['Key'])
                    df = pd.read_csv(StringIO(obj_data['Body'].read().decode('utf-8')))
                    
                    # Extract date from filename
                    filename = obj['Key'].split('/')[-1]
                    date_str = filename.replace('.csv', '')
                    df['game_date'] = date_str
                    
                    all_props.append(df)
                except Exception as e:
                    print(f"   ⚠️  Error loading {obj['Key']}: {e}")
        
        if not all_props:
            return pd.DataFrame()
        
        df_props = pd.concat(all_props, ignore_index=True)
        df_props['player_normalized'] = df_props['player'].apply(normalize_player_name)
        
        print(f"   ✅ Loaded {len(df_props):,} prop rows")
        print(f"      Dates: {df_props['game_date'].min()} to {df_props['game_date'].max()}")
        print(f"      Players: {df_props['player_normalized'].nunique():,}")
        print(f"      Markets: {df_props['market'].nunique()}")
        
        return df_props
        
    except Exception as e:
        print(f"   ❌ Error loading props: {e}")
        return pd.DataFrame()


def load_game_logs_from_s3(season):
    """Load game logs for a season from S3"""
    print(f"\n🏀 Loading game logs from S3 for {season}...")
    
    s3_client = boto3.client('s3')
    prefix = f"player_game_logs/{season}/"
    
    try:
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NBA, Prefix=prefix)
        
        if 'Contents' not in response:
            print(f"   ❌ No game log files found for {season}")
            return pd.DataFrame()
        
        all_game_logs = []
        for obj in response['Contents']:
            if obj['Key'].endswith('.csv'):
                try:
                    obj_data = s3_client.get_object(Bucket=S3_BUCKET_NBA, Key=obj['Key'])
                    df = pd.read_csv(StringIO(obj_data['Body'].read().decode('utf-8')))
                    all_game_logs.append(df)
                except Exception as e:
                    print(f"   ⚠️  Error loading {obj['Key']}: {e}")
        
        if not all_game_logs:
            return pd.DataFrame()
        
        df_games = pd.concat(all_game_logs, ignore_index=True)
        
        # Parse game date
        df_games['GAME_DATE'] = pd.to_datetime(df_games['GAME_DATE'])
        df_games['game_date'] = df_games['GAME_DATE'].dt.date.astype(str)
        
        # Normalize player names
        df_games['player_normalized'] = df_games['PLAYER_NAME'].apply(normalize_player_name)
        
        # Filter to players who actually played
        df_games = df_games[df_games['MIN'].notna() & (df_games['MIN'] > 0)].copy()
        
        print(f"   ✅ Loaded {len(df_games):,} player-game rows")
        print(f"      Dates: {df_games['game_date'].min()} to {df_games['game_date'].max()}")
        print(f"      Players: {df_games['player_normalized'].nunique():,}")
        
        return df_games
        
    except Exception as e:
        print(f"   ❌ Error loading game logs: {e}")
        return pd.DataFrame()


def join_props_and_actuals(df_props, df_games, markets_to_include=None):
    """
    Join props with actual game results
    
    Args:
        df_props: Props dataframe
        df_games: Game logs dataframe
        markets_to_include: List of markets to include (default: all)
    
    Returns:
        Joined dataframe with props and actuals
    """
    print(f"\n🔗 Joining props with actuals...")
    
    if markets_to_include:
        df_props = df_props[df_props['market'].isin(markets_to_include)].copy()
        print(f"   Filtering to {len(markets_to_include)} markets: {markets_to_include}")
    
    # Aggregate props by player/date/market (average across bookmakers)
    props_agg = df_props.groupby(['player_normalized', 'game_date', 'market']).agg({
        'prop_line': 'mean',
        'over_odds': 'median',
        'under_odds': 'median',
        'bookmaker': 'count'
    }).reset_index()
    
    props_agg.columns = ['player_normalized', 'game_date', 'market', 'prop_line', 
                          'over_odds', 'under_odds', 'num_bookmakers']
    
    # Join with game logs
    df_merged = df_games.merge(
        props_agg,
        on=['player_normalized', 'game_date'],
        how='inner'  # Only keep games with props
    )
    
    print(f"   ✅ Joined data")
    print(f"      Total rows: {len(df_merged):,}")
    print(f"      Players: {df_merged['player_normalized'].nunique():,}")
    print(f"      Markets: {df_merged['market'].nunique()}")
    
    # Calculate actual values for each market
    print(f"\n📈 Calculating actual values for each market...")
    df_merged['actual_value'] = df_merged.apply(
        lambda row: calculate_actual_value(row, row['market']), 
        axis=1
    )
    
    # Calculate line-relative metrics
    df_merged['margin'] = df_merged['actual_value'] - df_merged['prop_line']
    df_merged['beat_line'] = (df_merged['actual_value'] > df_merged['prop_line']).astype(int)
    
    # Remove rows with missing actuals
    before_len = len(df_merged)
    df_merged = df_merged[df_merged['actual_value'].notna()].copy()
    after_len = len(df_merged)
    
    if before_len > after_len:
        print(f"   ⚠️  Removed {before_len - after_len:,} rows with missing actuals")
    
    print(f"   ✅ Final dataset: {len(df_merged):,} rows")
    
    return df_merged


# =============================================================================
# STABILITY ANALYSIS (PART 1)
# =============================================================================

def create_rolling_bins(df, market, bin_size=5):
    """
    Create rolling bins for a specific market
    
    Args:
        df: Dataframe with props and actuals
        market: Market to analyze
        bin_size: Number of games per bin
    
    Returns:
        Dataframe with bin statistics
    """
    # Filter to this market
    df_market = df[df['market'] == market].copy()
    
    # Sort by player and date
    df_market = df_market.sort_values(['player_normalized', 'game_date'])
    
    # Create bins for each player
    all_bins = []
    
    for player in df_market['player_normalized'].unique():
        player_games = df_market[df_market['player_normalized'] == player].copy()
        
        # Need at least 2 bins worth of games
        if len(player_games) < bin_size * 2:
            continue
        
        # Create rolling bins
        player_games['game_num'] = range(len(player_games))
        player_games['bin_num'] = player_games['game_num'] // bin_size
        
        # Calculate mean for each bin
        bin_stats = player_games.groupby('bin_num').agg({
            'actual_value': 'mean',
            'player_normalized': 'first',
            'game_date': ['min', 'max', 'count']
        }).reset_index()
        
        bin_stats.columns = ['bin_num', 'bin_mean', 'player_normalized', 
                              'date_start', 'date_end', 'games_in_bin']
        
        # Need at least 2 bins
        if len(bin_stats) < 2:
            continue
        
        # Create lagged variable
        bin_stats['previous_bin_mean'] = bin_stats['bin_mean'].shift(1)
        bin_stats = bin_stats.dropna(subset=['previous_bin_mean'])
        
        all_bins.append(bin_stats)
    
    if not all_bins:
        return pd.DataFrame()
    
    df_bins = pd.concat(all_bins, ignore_index=True)
    df_bins['market'] = market
    
    return df_bins


def calculate_stability_r2(df_bins):
    """
    Calculate R² for stability (current bin ~ previous bin)
    
    Args:
        df_bins: Dataframe with bin statistics
    
    Returns:
        Dictionary with R² metrics
    """
    if len(df_bins) < 10:  # Need minimum sample size
        return {
            'r2_unweighted': None,
            'r2_weighted': None,
            'slope': None,
            'intercept': None,
            'n_bins': len(df_bins),
            'n_players': 0
        }
    
    X = df_bins['previous_bin_mean'].values.reshape(-1, 1)
    y = df_bins['bin_mean'].values
    weights = df_bins['games_in_bin'].values
    
    # Unweighted regression
    model_unweighted = LinearRegression()
    model_unweighted.fit(X, y)
    y_pred = model_unweighted.predict(X)
    r2_unweighted = r2_score(y, y_pred)
    
    # Weighted regression
    model_weighted = LinearRegression()
    model_weighted.fit(X, y, sample_weight=weights)
    y_pred_weighted = model_weighted.predict(X)
    r2_weighted = r2_score(y, y_pred_weighted, sample_weight=weights)
    
    return {
        'r2_unweighted': r2_unweighted,
        'r2_weighted': r2_weighted,
        'slope': model_weighted.coef_[0],
        'intercept': model_weighted.intercept_,
        'n_bins': len(df_bins),
        'n_players': df_bins['player_normalized'].nunique()
    }


def run_stability_analysis(df, markets, bin_size=5, min_games=10):
    """
    Run stability analysis for all markets
    
    Args:
        df: Dataframe with props and actuals
        markets: List of markets to analyze
        bin_size: Number of games per bin
        min_games: Minimum games required
    
    Returns:
        Dataframe with stability results by market
    """
    print(f"\n{'='*80}")
    print(f"PART 1: STABILITY ANALYSIS")
    print(f"{'='*80}")
    print(f"Bin size: {bin_size} games")
    print(f"Minimum games: {min_games}")
    
    results = []
    
    for market in markets:
        print(f"\n📊 Analyzing {market}...")
        
        # Create bins
        df_bins = create_rolling_bins(df, market, bin_size)
        
        if df_bins.empty:
            print(f"   ⚠️  Not enough data for {market}")
            continue
        
        print(f"   Bins created: {len(df_bins):,} bins from {df_bins['player_normalized'].nunique()} players")
        
        # Calculate R²
        stability_metrics = calculate_stability_r2(df_bins)
        
        if stability_metrics['r2_unweighted'] is not None:
            print(f"   ✅ R² (unweighted): {stability_metrics['r2_unweighted']:.3f}")
            print(f"   ✅ R² (weighted): {stability_metrics['r2_weighted']:.3f}")
        
        results.append({
            'market': market,
            **stability_metrics
        })
    
    df_stability = pd.DataFrame(results)
    
    # Rank by weighted R²
    df_stability = df_stability.sort_values('r2_weighted', ascending=False)
    df_stability['stability_rank'] = range(1, len(df_stability) + 1)
    
    print(f"\n{'='*80}")
    print(f"STABILITY RANKINGS")
    print(f"{'='*80}")
    print(df_stability[['market', 'r2_weighted', 'n_players', 'stability_rank']].to_string(index=False))
    
    return df_stability


# =============================================================================
# LINE PREDICTION ANALYSIS (PART 2)
# =============================================================================

def create_streak_features(df, market, window_size=5):
    """
    Create streak features for line prediction
    
    Args:
        df: Dataframe with props and actuals
        market: Market to analyze
        window_size: Size of rolling window
    
    Returns:
        Dataframe with streak features
    """
    # Filter to this market
    df_market = df[df['market'] == market].copy()
    
    # Sort by player and date
    df_market = df_market.sort_values(['player_normalized', 'game_date'])
    
    # Calculate rolling features for each player
    all_features = []
    
    for player in df_market['player_normalized'].unique():
        player_games = df_market[df_market['player_normalized'] == player].copy()
        
        # Need at least window_size + 1 games
        if len(player_games) <= window_size:
            continue
        
        # Calculate rolling features
        player_games['pct_over_L5'] = player_games['beat_line'].rolling(window=window_size, min_periods=1).mean()
        player_games['avg_margin_L5'] = player_games['margin'].rolling(window=window_size, min_periods=1).mean()
        player_games['streak_beat_line_L5'] = player_games['beat_line'].rolling(window=window_size, min_periods=1).sum()
        
        # Shift features to avoid lookahead bias
        player_games['pct_over_L5_lag'] = player_games['pct_over_L5'].shift(1)
        player_games['avg_margin_L5_lag'] = player_games['avg_margin_L5'].shift(1)
        
        # Drop first window_size rows (not enough history)
        player_games = player_games.iloc[window_size:].copy()
        
        all_features.append(player_games)
    
    if not all_features:
        return pd.DataFrame()
    
    df_features = pd.concat(all_features, ignore_index=True)
    
    # Drop rows with missing lagged features
    df_features = df_features.dropna(subset=['pct_over_L5_lag', 'avg_margin_L5_lag'])
    
    return df_features


def calculate_line_prediction_metrics(df_features):
    """
    Calculate prediction metrics for line results
    
    Args:
        df_features: Dataframe with streak features
    
    Returns:
        Dictionary with prediction metrics
    """
    if len(df_features) < 50:  # Need minimum sample size
        return {
            'accuracy': None,
            'roc_auc': None,
            'coef_binary': None,
            'pvalue_binary': None,
            'r2_margin': None,
            'coef_margin': None,
            'pvalue_margin': None,
            'n_samples': len(df_features),
            'n_players': 0
        }
    
    X_binary = df_features['pct_over_L5_lag'].values.reshape(-1, 1)
    y_binary = df_features['beat_line'].values
    
    X_margin = df_features['avg_margin_L5_lag'].values.reshape(-1, 1)
    y_margin = df_features['margin'].values
    
    # Binary prediction (beat line)
    model_binary = LogisticRegression(max_iter=1000)
    model_binary.fit(X_binary, y_binary)
    y_pred_binary = model_binary.predict(X_binary)
    y_pred_proba = model_binary.predict_proba(X_binary)[:, 1]
    
    accuracy = accuracy_score(y_binary, y_pred_binary)
    roc_auc = roc_auc_score(y_binary, y_pred_proba)
    
    # Calculate p-value for logistic regression coefficient
    from scipy.stats import chi2
    # Log likelihood ratio test (approximation)
    coef_binary = model_binary.coef_[0][0]
    
    # Margin prediction (continuous)
    model_margin = LinearRegression()
    model_margin.fit(X_margin, y_margin)
    y_pred_margin = model_margin.predict(X_margin)
    
    r2_margin = r2_score(y_margin, y_pred_margin)
    coef_margin = model_margin.coef_[0]
    
    # Calculate p-value for margin coefficient
    from scipy.stats import t as t_dist
    n = len(y_margin)
    residuals = y_margin - y_pred_margin
    mse = np.sum(residuals**2) / (n - 2)
    se = np.sqrt(mse / np.sum((X_margin - X_margin.mean())**2))
    t_stat = coef_margin / se
    pvalue_margin = 2 * (1 - t_dist.cdf(abs(t_stat), n - 2))
    
    # Determine strategy
    if pvalue_margin < 0.05:
        if coef_margin > 0:
            strategy = 'MOMENTUM'
        else:
            strategy = 'MEAN REVERSION'
    else:
        strategy = 'NEUTRAL'
    
    return {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'coef_binary': coef_binary,
        'pvalue_binary': None,  # Would need more complex calculation
        'r2_margin': r2_margin,
        'coef_margin': coef_margin,
        'pvalue_margin': pvalue_margin,
        'strategy': strategy,
        'n_samples': len(df_features),
        'n_players': df_features['player_normalized'].nunique()
    }


def run_line_prediction_analysis(df, markets, window_size=5):
    """
    Run line prediction analysis for all markets
    
    Args:
        df: Dataframe with props and actuals
        markets: List of markets to analyze
        window_size: Size of rolling window
    
    Returns:
        Dataframe with prediction results by market
    """
    print(f"\n{'='*80}")
    print(f"PART 2: LINE PREDICTION ANALYSIS")
    print(f"{'='*80}")
    print(f"Window size: {window_size} games")
    
    results = []
    
    for market in markets:
        print(f"\n📊 Analyzing {market}...")
        
        # Create features
        df_features = create_streak_features(df, market, window_size)
        
        if df_features.empty:
            print(f"   ⚠️  Not enough data for {market}")
            continue
        
        print(f"   Samples: {len(df_features):,} from {df_features['player_normalized'].nunique()} players")
        
        # Calculate metrics
        pred_metrics = calculate_line_prediction_metrics(df_features)
        
        if pred_metrics['accuracy'] is not None:
            print(f"   ✅ Accuracy: {pred_metrics['accuracy']:.1%}")
            print(f"   ✅ ROC AUC: {pred_metrics['roc_auc']:.3f}")
            print(f"   ✅ Margin Coef: {pred_metrics['coef_margin']:.3f} (p={pred_metrics['pvalue_margin']:.3f})")
            print(f"   ✅ Strategy: {pred_metrics['strategy']}")
        
        results.append({
            'market': market,
            **pred_metrics
        })
    
    df_prediction = pd.DataFrame(results)
    
    # Rank by accuracy
    df_prediction = df_prediction.sort_values('accuracy', ascending=False)
    df_prediction['prediction_rank'] = range(1, len(df_prediction) + 1)
    
    print(f"\n{'='*80}")
    print(f"PREDICTION RANKINGS")
    print(f"{'='*80}")
    print(df_prediction[['market', 'accuracy', 'strategy', 'prediction_rank']].to_string(index=False))
    
    return df_prediction


# =============================================================================
# COMBINED ANALYSIS (PART 3)
# =============================================================================

def combine_results(df_stability, df_prediction):
    """
    Combine stability and prediction results
    
    Args:
        df_stability: Stability analysis results
        df_prediction: Line prediction results
    
    Returns:
        Combined dataframe with overall rankings
    """
    print(f"\n{'='*80}")
    print(f"PART 3: COMBINED ANALYSIS")
    print(f"{'='*80}")
    
    df_combined = df_stability.merge(
        df_prediction,
        on='market',
        how='outer',
        suffixes=('_stability', '_prediction')
    )
    
    # Calculate overall score (normalize both metrics to 0-100 scale)
    max_r2 = df_combined['r2_weighted'].max()
    max_acc = df_combined['accuracy'].max()
    
    df_combined['stability_score'] = (df_combined['r2_weighted'] / max_r2 * 100).fillna(0)
    df_combined['prediction_score'] = (df_combined['accuracy'] / max_acc * 100).fillna(0)
    
    # Overall score (50/50 weight)
    df_combined['overall_score'] = (df_combined['stability_score'] + df_combined['prediction_score']) / 2
    
    # Overall rank
    df_combined = df_combined.sort_values('overall_score', ascending=False)
    df_combined['overall_rank'] = range(1, len(df_combined) + 1)
    
    print(f"\n{'='*80}")
    print(f"OVERALL RANKINGS")
    print(f"{'='*80}")
    
    display_cols = ['market', 'r2_weighted', 'accuracy', 'strategy', 'overall_score', 'overall_rank']
    print(df_combined[display_cols].to_string(index=False))
    
    return df_combined


# =============================================================================
# VISUALIZATION (PART 4)
# =============================================================================

def create_visualizations(df_stability, df_prediction, df_combined, output_dir):
    """
    Create visualizations for analysis results
    
    Args:
        df_stability: Stability results
        df_prediction: Prediction results
        df_combined: Combined results
        output_dir: Output directory for plots
    """
    print(f"\n{'='*80}")
    print(f"CREATING VISUALIZATIONS")
    print(f"{'='*80}")
    
    viz_dir = output_dir / '03_visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Stability Comparison
    print(f"\n📊 Creating stability comparison chart...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    df_plot = df_stability.sort_values('r2_weighted', ascending=True)
    markets = df_plot['market'].values
    r2_values = df_plot['r2_weighted'].values
    
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(markets)))
    ax.barh(markets, r2_values, color=colors)
    
    ax.set_xlabel('R² (Weighted)', fontsize=12, fontweight='bold')
    ax.set_title('Market Stability: Current Performance Predicts Future Performance', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlim(0, 1)
    
    for i, (market, r2) in enumerate(zip(markets, r2_values)):
        ax.text(r2 + 0.02, i, f'{r2:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'stability_comparison.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {viz_dir / 'stability_comparison.png'}")
    plt.close()
    
    # 2. Prediction Accuracy
    print(f"\n📊 Creating prediction accuracy chart...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    df_plot = df_prediction.sort_values('accuracy', ascending=True)
    markets = df_plot['market'].values
    accuracy_values = df_plot['accuracy'].values * 100
    
    colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(markets)))
    ax.barh(markets, accuracy_values, color=colors)
    
    ax.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Line Prediction Accuracy: Recent Results Predict Next Game', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.axvline(50, color='red', linestyle='--', alpha=0.5, label='Random (50%)')
    
    for i, (market, acc) in enumerate(zip(markets, accuracy_values)):
        ax.text(acc + 0.5, i, f'{acc:.1f}%', va='center', fontsize=10)
    
    ax.legend()
    plt.tight_layout()
    plt.savefig(viz_dir / 'prediction_accuracy.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {viz_dir / 'prediction_accuracy.png'}")
    plt.close()
    
    # 3. Momentum vs Reversal
    print(f"\n📊 Creating momentum vs reversal chart...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    df_plot = df_prediction.sort_values('coef_margin', ascending=True)
    markets = df_plot['market'].values
    coefs = df_plot['coef_margin'].values
    pvalues = df_plot['pvalue_margin'].values
    
    # Color by significance and direction
    colors = []
    for coef, pval in zip(coefs, pvalues):
        if pval < 0.05:
            if coef > 0:
                colors.append('green')
            else:
                colors.append('red')
        else:
            colors.append('gray')
    
    ax.barh(markets, coefs, color=colors)
    
    ax.set_xlabel('Coefficient (Margin Prediction)', fontsize=12, fontweight='bold')
    ax.set_title('Momentum vs Mean Reversion by Market', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    
    # Add legend
    momentum_patch = mpatches.Patch(color='green', label='Momentum (p<0.05)')
    reversal_patch = mpatches.Patch(color='red', label='Mean Reversion (p<0.05)')
    neutral_patch = mpatches.Patch(color='gray', label='Neutral (p≥0.05)')
    ax.legend(handles=[momentum_patch, reversal_patch, neutral_patch])
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'momentum_vs_reversal.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {viz_dir / 'momentum_vs_reversal.png'}")
    plt.close()
    
    print(f"\n✅ All visualizations created!")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Analyze NBA prop market stability and predictive power'
    )
    parser.add_argument('--season', type=str, default='2025-26',
                        help='NBA season (e.g., 2025-26)')
    parser.add_argument('--seasons', type=str, nargs='+',
                        help='Multiple NBA seasons (e.g., 2023-24 2024-25 2025-26)')
    parser.add_argument('--markets', type=str, nargs='+',
                        help='Specific markets to analyze (default: all)')
    parser.add_argument('--bin-size', type=int, default=DEFAULT_BIN_SIZE,
                        help=f'Bin size for rolling windows (default: {DEFAULT_BIN_SIZE})')
    parser.add_argument('--min-games', type=int, default=DEFAULT_MIN_GAMES,
                        help=f'Minimum games required (default: {DEFAULT_MIN_GAMES})')
    parser.add_argument('--no-viz', action='store_true',
                        help='Skip visualization generation')
    
    args = parser.parse_args()
    
    # Determine seasons to analyze
    seasons = args.seasons if args.seasons else [args.season]
    
    # Determine markets to analyze
    markets = args.markets if args.markets else ALL_MARKETS
    
    print(f"\n{'='*80}")
    print(f"NBA PROPS STABILITY & PREDICTIVE POWER ANALYSIS")
    print(f"{'='*80}")
    print(f"Seasons: {', '.join(seasons)}")
    print(f"Markets: {len(markets)}")
    print(f"Bin size: {args.bin_size}")
    print(f"Min games: {args.min_games}")
    
    # Process each season
    for season in seasons:
        print(f"\n\n{'#'*80}")
        print(f"# PROCESSING SEASON: {season}")
        print(f"{'#'*80}")
        
        # Load data
        df_props = load_props_from_s3(season)
        df_games = load_game_logs_from_s3(season)
        
        if df_props.empty or df_games.empty:
            print(f"\n❌ Skipping {season} - missing data")
            continue
        
        # Join data
        df_merged = join_props_and_actuals(df_props, df_games, markets)
        
        if df_merged.empty:
            print(f"\n❌ Skipping {season} - no joined data")
            continue
        
        # Create output directory
        output_dir = OUTPUT_BASE / season
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save joined data
        data_dir = output_dir / '01_data'
        data_dir.mkdir(parents=True, exist_ok=True)
        
        merged_path = data_dir / 'props_with_actuals.csv'
        df_merged.to_csv(merged_path, index=False)
        print(f"\n💾 Saved joined data: {merged_path}")
        
        # Run analyses
        df_stability = run_stability_analysis(df_merged, markets, args.bin_size, args.min_games)
        df_prediction = run_line_prediction_analysis(df_merged, markets, args.bin_size)
        
        # Combine results
        df_combined = combine_results(df_stability, df_prediction)
        
        # Save results
        analysis_dir = output_dir / '02_analysis'
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        df_stability.to_csv(analysis_dir / 'stability_by_market.csv', index=False)
        df_prediction.to_csv(analysis_dir / 'line_prediction_by_market.csv', index=False)
        df_combined.to_csv(analysis_dir / 'combined_market_rankings.csv', index=False)
        
        print(f"\n💾 Saved analysis results to: {analysis_dir}")
        
        # Create visualizations
        if not args.no_viz:
            create_visualizations(df_stability, df_prediction, df_combined, output_dir)
        
        print(f"\n✅ Analysis complete for {season}!")
        print(f"📁 Results saved to: {output_dir}")
    
    print(f"\n\n{'='*80}")
    print(f"ALL SEASONS COMPLETE")
    print(f"{'='*80}")
    print(f"Results location: {OUTPUT_BASE}")


if __name__ == '__main__':
    main()
