"""
Analyze whether current spread binning strategy is optimal.

Context:
--------
Current spread bins are arbitrary 4-point buckets:
- Pick'em: -2 to +2
- 2-6 Fav/Dog: -6 to -2, +2 to +6
- 6-10 Fav/Dog: -10 to -6, +6 to +10
- 10-15 Fav/Dog: -15 to -10, +10 to +15
- 15+ Fav/Dog: < -15, > +15

These bins may not reflect actual differences in player performance.

Questions to Answer:
--------------------
1. Is there a monotonic relationship between spread and hit rate?
2. Are the current 4-point bins meaningful or arbitrary?
3. Should we use continuous spread, percentiles, or data-driven cutoffs?
4. Do star vs role players need different binning strategies?
5. Does spread matter more for certain bet types (under vs over)?

Analysis Approaches:
--------------------
A. Continuous Analysis:
   - Plot hit rate vs continuous spread (smoothed)
   - Check for inflection points / thresholds
   - Test polynomial fits to find optimal degree

B. Current Bins Performance:
   - Compare hit rate across current bins
   - Statistical tests for differences between adjacent bins
   - Look for bins with similar performance that could merge

C. Alternative Binning:
   - Equal-frequency bins (percentiles)
   - Recursive partitioning (decision tree)
   - K-means clustering on spread + performance
   - Domain knowledge bins (e.g., key numbers like 3, 7 in NFL)

D. Interaction Effects:
   - Does line tier (star/role) interact with spread bins?
   - Does scorer type (rim/perimeter) interact with spread?
   - Should bins differ by bet side (under vs over)?

E. Predictive Power:
   - Compare model performance with:
     * Current bins
     * Continuous spread
     * Data-driven bins
     * No spread (baseline)

Usage:
------
python analysis/analyze_spread_binning_strategy.py --season 2025-26

Author: Myles Thomas
Date: 2026-01-30
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.interpolate import UnivariateSpline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# EMOJI MAP
# =============================================================================

EMOJI = {
    'success': '✅',
    'error': '❌',
    'warning': '⚠️',
    'info': 'ℹ️',
    'chart': '📊',
    'target': '🎯',
    'analysis': '🔬',
    'question': '❓',
    'light': '💡',
    'money': '💰',
}


# =============================================================================
# SPREAD BINNING FUNCTIONS
# =============================================================================

def bin_team_spread_current(spread):
    """Current production binning logic."""
    if pd.isna(spread):
        return 'Unknown'
    
    if spread < -15:
        return '15+ Fav'
    elif spread < -10:
        return '10-15 Fav'
    elif spread < -6:
        return '6-10 Fav'
    elif spread < -2:
        return '2-6 Fav'
    elif spread <= 2:
        return "Pick'em (-2 to +2)"
    elif spread <= 6:
        return '2-6 Dog'
    elif spread <= 10:
        return '6-10 Dog'
    elif spread <= 15:
        return '10-15 Dog'
    else:
        return '15+ Dog'


def bin_team_spread_percentile(spread, percentiles):
    """
    Bin spread by percentiles (equal-frequency bins).
    
    Args:
        spread: Team spread value
        percentiles: List of (pct, label) tuples
    
    Returns:
        Bin label
    """
    for pct, label in percentiles:
        if spread <= pct:
            return label
    return percentiles[-1][1]


def bin_team_spread_simple(spread):
    """Simplified 3-bin approach."""
    if pd.isna(spread):
        return 'Unknown'
    
    if spread < -5:
        return 'Favorite'
    elif spread <= 5:
        return "Pick'em"
    else:
        return 'Underdog'


# =============================================================================
# DATA LOADING
# =============================================================================

def load_backtest_data(season='2025-26'):
    """
    Load historical tracking results from S3.
    
    Returns:
        DataFrame with columns: player, team, opponent, team_spread, 
                                points_line, bet_side, hit, etc.
    """
    print(f"\n{EMOJI['chart']} Loading tracking results from S3 for {season}...")
    
    import boto3
    from io import StringIO
    from datetime import datetime, timedelta
    
    s3 = boto3.client('s3')
    bucket = 'nba-betting-mt'
    
    # Load from both 2D and 3D tracking results
    all_dfs = []
    
    for strategy in ['2d', '3d']:
        prefix = f'data/04_output/results/role_spread_points_model/{strategy}/'
        
        print(f"\n   Loading {strategy.upper()} tracking results...")
        
        # List all result files for this strategy
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
        
        file_count = 0
        for page in pages:
            if 'Contents' not in page:
                continue
                
            for obj in page['Contents']:
                key = obj['Key']
                
                # Skip non-CSV files
                if not key.endswith('.csv'):
                    continue
                
                # Skip _top3 suffix files (those are filtered subsets)
                if '_top3.csv' in key:
                    continue
                
                try:
                    # Load CSV
                    response = s3.get_object(Bucket=bucket, Key=key)
                    df = pd.read_csv(StringIO(response['Body'].read().decode('utf-8')))
                    
                    # Add strategy column
                    df['strategy_dimension'] = strategy
                    
                    all_dfs.append(df)
                    file_count += 1
                    
                except Exception as e:
                    print(f"      ⚠️  Failed to load {key}: {e}")
                    continue
        
        print(f"      {EMOJI['success']} Loaded {file_count} result files")
    
    if not all_dfs:
        raise ValueError(
            f"{EMOJI['error']} No tracking results found in S3\n"
            f"   Run the daily workflow to generate tracking results first"
        )
    
    # Combine all dataframes
    df = pd.concat(all_dfs, ignore_index=True)
    
    print(f"\n   {EMOJI['success']} Combined: {len(df):,} player-game records")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Strategies: 2D={len(df[df['strategy_dimension']=='2d']):,}, 3D={len(df[df['strategy_dimension']=='3d']):,}")
    
    return df


# =============================================================================
# ANALYSIS A: CONTINUOUS RELATIONSHIP
# =============================================================================

def analyze_continuous_relationship(df):
    """
    Analyze continuous relationship between spread and hit rate.
    Creates smoothed plots to identify natural breakpoints.
    """
    print(f"\n{'='*80}")
    print(f"{EMOJI['analysis']} ANALYSIS A: Continuous Spread Relationship")
    print(f"{'='*80}")
    
    # Create hit column from result
    df['hit'] = (df['result'] == 'WIN').astype(int)
    
    # Bin into small buckets for initial view
    df['spread_bucket_0.5'] = (df['spread'] // 0.5) * 0.5
    
    # Calculate hit rate by bucket
    hit_rate_by_spread = df.groupby('spread_bucket_0.5').agg({
        'hit': ['mean', 'count']
    }).reset_index()
    hit_rate_by_spread.columns = ['spread', 'hit_rate', 'count']
    
    # Filter to buckets with >= 30 samples for stability
    hit_rate_by_spread = hit_rate_by_spread[hit_rate_by_spread['count'] >= 30]
    
    print(f"\n{EMOJI['info']} Spread range: {df['spread'].min():.1f} to {df['spread'].max():.1f}")
    print(f"{EMOJI['info']} Buckets with >=30 samples: {len(hit_rate_by_spread)}")
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Raw scatter + smoothed line
    ax = axes[0]
    ax.scatter(hit_rate_by_spread['spread'], hit_rate_by_spread['hit_rate'], 
               s=hit_rate_by_spread['count']/5, alpha=0.5, color='steelblue')
    
    # Add smoothed spline
    if len(hit_rate_by_spread) >= 4:
        spline = UnivariateSpline(
            hit_rate_by_spread['spread'], 
            hit_rate_by_spread['hit_rate'],
            s=0.01,  # Smoothing factor
            k=3  # Cubic spline
        )
        x_smooth = np.linspace(hit_rate_by_spread['spread'].min(), 
                               hit_rate_by_spread['spread'].max(), 200)
        y_smooth = spline(x_smooth)
        ax.plot(x_smooth, y_smooth, 'r-', linewidth=2, label='Smoothed Trend')
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% (breakeven)')
    ax.set_xlabel('Team Spread (negative = favorite)', fontsize=12)
    ax.set_ylabel('Hit Rate', fontsize=12)
    ax.set_title('Hit Rate vs Continuous Spread\n(size = sample size)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Hit rate by bet side
    ax = axes[1]
    for bet_side in ['UNDER', 'OVER']:
        df_side = df[df['bet_side'] == bet_side].copy()
        df_side['spread_bucket_0.5'] = (df_side['spread'] // 0.5) * 0.5
        
        hit_rate_side = df_side.groupby('spread_bucket_0.5').agg({
            'hit': ['mean', 'count']
        }).reset_index()
        hit_rate_side.columns = ['spread', 'hit_rate', 'count']
        hit_rate_side = hit_rate_side[hit_rate_side['count'] >= 30]
        
        ax.scatter(hit_rate_side['spread'], hit_rate_side['hit_rate'],
                  s=hit_rate_side['count']/5, alpha=0.6, label=bet_side)
        
        # Add smoothed spline
        if len(hit_rate_side) >= 4:
            spline_side = UnivariateSpline(
                hit_rate_side['spread'], 
                hit_rate_side['hit_rate'],
                s=0.01,
                k=3
            )
            x_smooth_side = np.linspace(hit_rate_side['spread'].min(), 
                                       hit_rate_side['spread'].max(), 200)
            y_smooth_side = spline_side(x_smooth_side)
            ax.plot(x_smooth_side, y_smooth_side, linewidth=2)
    
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% (breakeven)')
    ax.set_xlabel('Team Spread (negative = favorite)', fontsize=12)
    ax.set_ylabel('Hit Rate', fontsize=12)
    ax.set_title('Hit Rate vs Spread by Bet Side\n(size = sample size)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path.cwd()
    while not (output_dir / '.gitignore').exists() and output_dir != output_dir.parent:
        output_dir = output_dir.parent
    output_dir = output_dir / 'analysis' / 'plots'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    output_path = output_dir / 'spread_binning_continuous_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n{EMOJI['success']} Plot saved: {output_path}")
    
    # Statistical test: is there a monotonic trend?
    correlation, p_value = stats.spearmanr(hit_rate_by_spread['spread'], 
                                           hit_rate_by_spread['hit_rate'])
    
    print(f"\n{EMOJI['target']} Spearman Correlation: {correlation:.3f} (p={p_value:.4f})")
    if p_value < 0.05:
        print(f"   {EMOJI['success']} Significant monotonic relationship detected!")
    else:
        print(f"   {EMOJI['warning']} No significant monotonic relationship (spread may not matter)")
    
    return hit_rate_by_spread


# =============================================================================
# ANALYSIS B: CURRENT BINS PERFORMANCE
# =============================================================================

def analyze_current_bins(df):
    """
    Evaluate performance of current spread bins.
    Test if adjacent bins are significantly different.
    """
    print(f"\n{'='*80}")
    print(f"{EMOJI['analysis']} ANALYSIS B: Current Bins Performance")
    print(f"{'='*80}")
    
    # Create hit column from result if not exists
    if 'hit' not in df.columns:
        df['hit'] = (df['result'] == 'WIN').astype(int)
    
    # Apply current binning
    df['spread_bin_current'] = df['spread'].apply(bin_team_spread_current)
    
    # Define bin order
    bin_order = [
        '15+ Fav', '10-15 Fav', '6-10 Fav', '2-6 Fav', 
        "Pick'em (-2 to +2)", 
        '2-6 Dog', '6-10 Dog', '10-15 Dog', '15+ Dog'
    ]
    
    # Calculate performance by bin
    bin_stats = df.groupby('spread_bin_current').agg({
        'hit': ['count', 'sum', 'mean'],
        'expected_roi': 'mean'
    }).reset_index()
    bin_stats.columns = ['spread_bin', 'count', 'wins', 'hit_rate', 'mean_expected_roi']
    
    # Reorder by bin order
    bin_stats['order'] = bin_stats['spread_bin'].map({b: i for i, b in enumerate(bin_order)})
    bin_stats = bin_stats.sort_values('order')
    
    # Calculate standard error
    bin_stats['std_err'] = np.sqrt(bin_stats['hit_rate'] * (1 - bin_stats['hit_rate']) / bin_stats['count'])
    
    print(f"\n{EMOJI['chart']} Current Bin Performance:")
    print(f"\n{'Spread Bin':<20} {'Count':>8} {'Wins':>8} {'Hit Rate':>10} {'StdErr':>8} {'Exp ROI':>10}")
    print("-" * 80)
    
    for _, row in bin_stats.iterrows():
        print(f"{row['spread_bin']:<20} {row['count']:>8.0f} {row['wins']:>8.0f} "
              f"{row['hit_rate']:>10.1%} {row['std_err']:>8.1%} {row['mean_expected_roi']:>10.1%}")
    
    # Statistical tests: Compare adjacent bins
    print(f"\n{EMOJI['target']} Statistical Tests (Adjacent Bins):")
    print("-" * 80)
    
    for i in range(len(bin_order) - 1):
        bin1 = bin_order[i]
        bin2 = bin_order[i + 1]
        
        data1 = df[df['spread_bin_current'] == bin1]['hit']
        data2 = df[df['spread_bin_current'] == bin2]['hit']
        
        if len(data1) >= 30 and len(data2) >= 30:
            # Two-proportion z-test
            n1, n2 = len(data1), len(data2)
            p1, p2 = data1.mean(), data2.mean()
            p_pooled = (data1.sum() + data2.sum()) / (n1 + n2)
            
            se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
            z = (p1 - p2) / se if se > 0 else 0
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))
            
            sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            
            print(f"{bin1:<20} vs {bin2:<20}: "
                  f"Δ={p1-p2:>+6.1%} | p={p_value:>6.4f} {sig_marker}")
        else:
            print(f"{bin1:<20} vs {bin2:<20}: Insufficient data")
    
    print(f"\n{EMOJI['info']} Legend: *** p<0.001, ** p<0.01, * p<0.05")
    
    # Visualize
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(len(bin_stats))
    ax.bar(x_pos, bin_stats['hit_rate'], yerr=bin_stats['std_err']*1.96, 
           capsize=5, color='steelblue', alpha=0.7)
    ax.axhline(y=0.5, color='red', linestyle='--', label='Breakeven (50%)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bin_stats['spread_bin'], rotation=45, ha='right')
    ax.set_ylabel('Hit Rate', fontsize=12)
    ax.set_xlabel('Spread Bin', fontsize=12)
    ax.set_title('Current Spread Bins Performance\n(error bars = 95% CI)', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_dir = Path.cwd()
    while not (output_dir / '.gitignore').exists() and output_dir != output_dir.parent:
        output_dir = output_dir.parent
    output_dir = output_dir / 'analysis' / 'plots'
    output_path = output_dir / 'spread_binning_current_bins.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n{EMOJI['success']} Plot saved: {output_path}")
    
    return bin_stats


# =============================================================================
# ANALYSIS C: ALTERNATIVE BINNING STRATEGIES
# =============================================================================

def analyze_alternative_bins(df):
    """
    Compare alternative binning strategies:
    1. Continuous (no binning)
    2. Simple 3-bin
    3. Percentile-based
    4. Decision tree (data-driven)
    """
    print(f"\n{'='*80}")
    print(f"{EMOJI['analysis']} ANALYSIS C: Alternative Binning Strategies")
    print(f"{'='*80}")
    
    # Create hit column from result if not exists
    if 'hit' not in df.columns:
        df['hit'] = (df['result'] == 'WIN').astype(int)
    
    # Strategy 1: Continuous spread (baseline)
    print(f"\n{EMOJI['target']} Strategy 1: Continuous Spread (no binning)")
    corr_continuous = df['spread'].corr(df['hit'])
    print(f"   Correlation with hit: {corr_continuous:.4f}")
    
    # Strategy 2: Simple 3-bin
    print(f"\n{EMOJI['target']} Strategy 2: Simple 3-Bin (Fav / Pick'em / Dog)")
    df['spread_bin_simple'] = df['spread'].apply(bin_team_spread_simple)
    
    simple_stats = df.groupby('spread_bin_simple').agg({
        'hit': ['count', 'mean']
    }).reset_index()
    simple_stats.columns = ['bin', 'count', 'hit_rate']
    
    for _, row in simple_stats.iterrows():
        print(f"   {row['bin']:<15}: {row['hit_rate']:.1%} (n={row['count']:,})")
    
    # Strategy 3: Percentile-based (5 equal-frequency bins)
    print(f"\n{EMOJI['target']} Strategy 3: Percentile-Based (quintiles)")
    
    percentiles = df['spread'].quantile([0.2, 0.4, 0.6, 0.8, 1.0])
    print(f"   Percentile cutoffs: {percentiles.values}")
    
    df['spread_bin_percentile'] = pd.qcut(df['spread'], q=5, 
                                          labels=['P1 (Strongest Fav)', 'P2', 'P3', 'P4', 'P5 (Strongest Dog)'],
                                          duplicates='drop')
    
    pct_stats = df.groupby('spread_bin_percentile').agg({
        'hit': ['count', 'mean']
    }).reset_index()
    pct_stats.columns = ['bin', 'count', 'hit_rate']
    
    for _, row in pct_stats.iterrows():
        print(f"   {row['bin']:<25}: {row['hit_rate']:.1%} (n={row['count']:,})")
    
    # Strategy 4: Decision Tree (data-driven optimal splits)
    print(f"\n{EMOJI['target']} Strategy 4: Decision Tree (data-driven)")
    
    X = df[['spread']].values
    y = df['hit'].values
    
    # Try different max_depth values
    for max_depth in [2, 3, 4]:
        clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=100, random_state=42)
        clf.fit(X, y)
        
        # Get split points
        tree = clf.tree_
        split_values = []
        
        def extract_splits(node_id=0):
            if tree.feature[node_id] != -2:  # Not a leaf
                split_values.append(tree.threshold[node_id])
                extract_splits(tree.children_left[node_id])
                extract_splits(tree.children_right[node_id])
        
        extract_splits()
        split_values = sorted(split_values)
        
        # Cross-validation score
        cv_score = cross_val_score(clf, X, y, cv=5, scoring='accuracy').mean()
        
        print(f"   Max Depth {max_depth}: CV Accuracy={cv_score:.3f}")
        print(f"      Split points: {[f'{s:.1f}' for s in split_values]}")
    
    return {
        'simple': simple_stats,
        'percentile': pct_stats,
    }


# =============================================================================
# ANALYSIS D: INTERACTION EFFECTS
# =============================================================================

def analyze_interactions(df):
    """
    Check if spread bins interact with:
    - Line tier (star vs role)
    - Scorer type (rim vs perimeter)
    - Bet side (under vs over)
    """
    print(f"\n{'='*80}")
    print(f"{EMOJI['analysis']} ANALYSIS D: Interaction Effects")
    print(f"{'='*80}")
    
    # Create hit column from result if not exists
    if 'hit' not in df.columns:
        df['hit'] = (df['result'] == 'WIN').astype(int)
    
    # Apply current binning
    df['spread_bin_current'] = df['spread'].apply(bin_team_spread_current)
    
    # Interaction 1: Line Tier × Spread
    print(f"\n{EMOJI['target']} Interaction 1: Line Tier × Spread Bin")
    
    interaction_tier = df.groupby(['line_tier', 'spread_bin_current']).agg({
        'hit': ['count', 'mean']
    }).reset_index()
    interaction_tier.columns = ['line_tier', 'spread_bin', 'count', 'hit_rate']
    
    # Pivot for easier reading
    pivot_tier = interaction_tier.pivot(index='spread_bin', columns='line_tier', values='hit_rate')
    
    print("\nHit Rate by Line Tier × Spread Bin:")
    print(pivot_tier.to_string())
    
    # Interaction 2: Bet Side × Spread
    print(f"\n{EMOJI['target']} Interaction 2: Bet Side × Spread Bin")
    
    interaction_side = df.groupby(['bet_side', 'spread_bin_current']).agg({
        'hit': ['count', 'mean']
    }).reset_index()
    interaction_side.columns = ['bet_side', 'spread_bin', 'count', 'hit_rate']
    
    pivot_side = interaction_side.pivot(index='spread_bin', columns='bet_side', values='hit_rate')
    
    print("\nHit Rate by Bet Side × Spread Bin:")
    print(pivot_side.to_string())
    
    # Check for scorer_type if available
    if 'scorer_type' in df.columns:
        print(f"\n{EMOJI['target']} Interaction 3: Scorer Type × Spread Bin")
        
        interaction_scorer = df.groupby(['scorer_type', 'spread_bin_current']).agg({
            'hit': ['count', 'mean']
        }).reset_index()
        interaction_scorer.columns = ['scorer_type', 'spread_bin', 'count', 'hit_rate']
        
        pivot_scorer = interaction_scorer.pivot(index='spread_bin', columns='scorer_type', values='hit_rate')
        
        print("\nHit Rate by Scorer Type × Spread Bin:")
        print(pivot_scorer.to_string())


# =============================================================================
# ANALYSIS E: PREDICTIVE POWER COMPARISON
# =============================================================================

def analyze_predictive_power(df):
    """
    Compare predictive power of different binning approaches.
    Uses logistic regression to measure information gain.
    """
    print(f"\n{'='*80}")
    print(f"{EMOJI['analysis']} ANALYSIS E: Predictive Power Comparison")
    print(f"{'='*80}")
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.metrics import roc_auc_score, log_loss
    
    # Create hit column from result if not exists
    if 'hit' not in df.columns:
        df['hit'] = (df['result'] == 'WIN').astype(int)
    
    # Prepare data
    X_continuous = df[['spread']].values
    y = df['hit'].values
    
    # Apply binning strategies
    df['spread_bin_current'] = df['spread'].apply(bin_team_spread_current)
    df['spread_bin_simple'] = df['spread'].apply(bin_team_spread_simple)
    
    # Model 1: No spread (baseline)
    print(f"\n{EMOJI['target']} Model 1: Baseline (no spread info)")
    baseline_pred = np.full(len(y), y.mean())
    baseline_logloss = log_loss(y, baseline_pred)
    print(f"   Log Loss: {baseline_logloss:.4f}")
    print(f"   AUC-ROC: 0.5000 (random)")
    
    # Model 2: Continuous spread
    print(f"\n{EMOJI['target']} Model 2: Continuous Spread")
    lr_continuous = LogisticRegression(random_state=42)
    lr_continuous.fit(X_continuous, y)
    pred_continuous = lr_continuous.predict_proba(X_continuous)[:, 1]
    
    continuous_logloss = log_loss(y, pred_continuous)
    continuous_auc = roc_auc_score(y, pred_continuous)
    print(f"   Log Loss: {continuous_logloss:.4f} (Δ{baseline_logloss - continuous_logloss:+.4f})")
    print(f"   AUC-ROC: {continuous_auc:.4f}")
    
    # Model 3: Simple bins
    print(f"\n{EMOJI['target']} Model 3: Simple 3-Bin")
    
    enc_simple = OneHotEncoder(sparse_output=False, drop='first')
    X_simple = enc_simple.fit_transform(df[['spread_bin_simple']])
    
    lr_simple = LogisticRegression(random_state=42)
    lr_simple.fit(X_simple, y)
    pred_simple = lr_simple.predict_proba(X_simple)[:, 1]
    
    simple_logloss = log_loss(y, pred_simple)
    simple_auc = roc_auc_score(y, pred_simple)
    print(f"   Log Loss: {simple_logloss:.4f} (Δ{baseline_logloss - simple_logloss:+.4f})")
    print(f"   AUC-ROC: {simple_auc:.4f}")
    
    # Model 4: Current detailed bins
    print(f"\n{EMOJI['target']} Model 4: Current Detailed Bins (9 bins)")
    
    enc_current = OneHotEncoder(sparse_output=False, drop='first')
    X_current = enc_current.fit_transform(df[['spread_bin_current']])
    
    lr_current = LogisticRegression(random_state=42)
    lr_current.fit(X_current, y)
    pred_current = lr_current.predict_proba(X_current)[:, 1]
    
    current_logloss = log_loss(y, pred_current)
    current_auc = roc_auc_score(y, pred_current)
    print(f"   Log Loss: {current_logloss:.4f} (Δ{baseline_logloss - current_logloss:+.4f})")
    print(f"   AUC-ROC: {current_auc:.4f}")
    
    # Summary comparison
    print(f"\n{EMOJI['chart']} Summary: Predictive Power")
    print("-" * 80)
    print(f"{'Model':<30} {'Log Loss':>12} {'Improvement':>15} {'AUC-ROC':>10}")
    print("-" * 80)
    
    models = [
        ('Baseline (no spread)', baseline_logloss, 0, 0.5),
        ('Continuous Spread', continuous_logloss, baseline_logloss - continuous_logloss, continuous_auc),
        ('Simple 3-Bin', simple_logloss, baseline_logloss - simple_logloss, simple_auc),
        ('Current Detailed 9-Bin', current_logloss, baseline_logloss - current_logloss, current_auc),
    ]
    
    for model_name, logloss, improvement, auc in models:
        print(f"{model_name:<30} {logloss:>12.4f} {improvement:>+15.4f} {auc:>10.4f}")
    
    print(f"\n{EMOJI['light']} Interpretation:")
    print(f"   - Lower log loss = better calibrated probabilities")
    print(f"   - Higher AUC-ROC = better discrimination")
    print(f"   - If continuous ~ detailed bins: binning adds no value (just noise)")
    print(f"   - If detailed >> simple: granular bins capture real patterns")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Analyze spread binning strategy for NBA props betting'
    )
    parser.add_argument(
        '--season',
        type=str,
        default='2025-26',
        help='NBA season (default: 2025-26)'
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"{EMOJI['chart']} NBA PROPS SPREAD BINNING ANALYSIS")
    print(f"{'='*80}")
    print(f"Season: {args.season}")
    print(f"Question: Are current spread bins optimal or arbitrary?")
    
    # Load data
    df = load_backtest_data(season=args.season)
    
    # Run analyses
    hit_rate_continuous = analyze_continuous_relationship(df)
    bin_stats_current = analyze_current_bins(df)
    alt_bin_stats = analyze_alternative_bins(df)
    analyze_interactions(df)
    analyze_predictive_power(df)
    
    # Final recommendations
    print(f"\n{'='*80}")
    print(f"{EMOJI['light']} RECOMMENDATIONS")
    print(f"{'='*80}")
    print(f"""
Based on the analyses above, consider these next steps:

1. If continuous spread has similar predictive power to bins:
   → Use continuous spread in model (simpler, no arbitrary cutoffs)

2. If certain adjacent bins are NOT significantly different:
   → Merge them to reduce overfitting

3. If decision tree suggests different splits:
   → Consider using data-driven cutoffs instead of round numbers

4. If interactions are strong (e.g., different bins for stars vs role players):
   → Create tier-specific binning strategies

5. If spread has weak predictive power overall:
   → Consider removing spread from model entirely
   → Focus on other features (line tier, scorer type, rest days, etc.)

Next steps:
-----------
1. Review plots in analysis/plots/spread_binning_*.png
2. Check statistical significance tables above
3. Decide on new binning strategy (if needed)
4. Re-run full backtest with new bins
5. Compare out-of-sample performance (validation set)
    """)
    
    print(f"\n{EMOJI['success']} Analysis complete!\n")


if __name__ == '__main__':
    main()
