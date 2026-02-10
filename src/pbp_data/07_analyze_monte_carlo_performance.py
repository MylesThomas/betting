"""
Script 07: Analyze Monte Carlo Performance

Purpose:
- Load predictions from validation run
- Calculate Brier scores (measures calibration)
- Generate analysis plots
- Print summary statistics

Brier Score:
- Measures accuracy of probabilistic predictions
- Score = mean((predicted_prob - actual_outcome)^2)
- Range: 0 (perfect) to 1 (worst)
- < 0.25 is generally considered good for binary predictions

Usage:
    # Basic analysis (summary only)
    python src/pbp_data/07_analyze_monte_carlo_performance.py
    
    # Full visualization suite
    python src/pbp_data/07_analyze_monte_carlo_performance.py --plot
    
Output:
    ~/Downloads/tmp/monte_carlo_validation/analysis/
        - brier_scores.csv
        - calibration_plot.png
        - performance_summary.txt
        - [with --plot] 8+ additional visualization plots
"""

import duckdb
import pandas as pd
import numpy as np
import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Import path functions
sys.path.insert(0, str(Path(__file__).parent.parent))
from pbp_data.monte_carlo_utils import get_project_root


# =============================================================================
# PATHS
# =============================================================================

VALIDATION_DIR = Path.home() / "Downloads" / "tmp" / "monte_carlo_validation"
PREDICTIONS_FILE = VALIDATION_DIR / "predictions.parquet"
CURRENT_PREDICTIONS_DIR = VALIDATION_DIR / "current_player_game_predictions"
ANALYSIS_DIR = VALIDATION_DIR / "analysis"
ANALYSIS_DIR.mkdir(exist_ok=True, parents=True)


# =============================================================================
# BRIER SCORE CALCULATION
# =============================================================================

def calculate_brier_score(predictions_df):
    """
    Calculate Brier score for predictions.
    
    Brier score = mean((predicted_prob - actual_outcome)^2)
    
    Returns:
        dict with overall score and breakdown by game stage
    """
    # Convert result to binary (1 = HIT, 0 = MISS)
    predictions_df['actual_outcome'] = (predictions_df['result'] == 'HIT').astype(int)
    
    # Calculate squared error
    predictions_df['squared_error'] = (predictions_df['prob_over'] - predictions_df['actual_outcome']) ** 2
    
    # Overall Brier score
    overall_brier = predictions_df['squared_error'].mean()
    
    # Brier score by game stage (quarters)
    brier_by_quarter = predictions_df.groupby('quarter')['squared_error'].mean()
    
    # Brier score by game minute bins
    predictions_df['minute_bin'] = pd.cut(
        predictions_df['game_minute'], 
        bins=[0, 12, 24, 36, 48],
        labels=['Q1 (0-12)', 'Q2 (12-24)', 'Q3 (24-36)', 'Q4 (36-48)']
    )
    brier_by_minute = predictions_df.groupby('minute_bin', observed=True)['squared_error'].mean()
    
    return {
        'overall': overall_brier,
        'by_quarter': brier_by_quarter.to_dict(),
        'by_minute': brier_by_minute.to_dict(),
    }


# =============================================================================
# CALIBRATION ANALYSIS
# =============================================================================

def analyze_calibration(predictions_df, n_bins=10):
    """
    Analyze calibration of probabilistic predictions.
    
    Calibration: Do events predicted at X% probability actually occur X% of the time?
    
    Returns:
        DataFrame with predicted_prob_bin, actual_freq, count
    """
    # Convert result to binary
    predictions_df['actual_outcome'] = (predictions_df['result'] == 'HIT').astype(int)
    
    # Bin predicted probabilities
    predictions_df['prob_bin'] = pd.cut(
        predictions_df['prob_over'], 
        bins=n_bins,
        labels=False
    )
    
    # Calculate actual frequency in each bin
    calibration = predictions_df.groupby('prob_bin').agg({
        'prob_over': 'mean',  # Average predicted probability in bin
        'actual_outcome': 'mean',  # Actual frequency of event
        'game_id': 'count'  # Number of predictions in bin
    }).reset_index()
    
    calibration.columns = ['prob_bin', 'predicted_prob', 'actual_freq', 'count']
    
    return calibration


# =============================================================================
# PLAYER PERFORMANCE ANALYSIS
# =============================================================================

def analyze_player_performance(predictions_df):
    """Analyze Brier scores by player (across all predictions, not just final)."""
    # Convert result to binary
    predictions_df['actual_outcome'] = (predictions_df['result'] == 'HIT').astype(int)
    predictions_df['squared_error'] = (predictions_df['prob_over'] - predictions_df['actual_outcome']) ** 2
    
    # Brier score by player (average across ALL predictions)
    player_brier = predictions_df.groupby('player_name').agg({
        'squared_error': 'mean',
        'game_id': lambda x: x.nunique(),  # Count unique games
        'actual_outcome': 'mean',  # Hit rate
    }).reset_index()
    
    player_brier.columns = ['player_name', 'brier_score', 'num_games', 'hit_rate']
    player_brier = player_brier.sort_values('brier_score')
    
    return player_brier


# =============================================================================
# ADDITIONAL VISUALIZATIONS (--plot flag)
# =============================================================================

def create_brier_over_time_plot(predictions_df):
    """Plot Brier score evolution over game time."""
    # Convert result to binary
    predictions_df['actual_outcome'] = (predictions_df['result'] == 'HIT').astype(int)
    predictions_df['squared_error'] = (predictions_df['prob_over'] - predictions_df['actual_outcome']) ** 2
    
    # Group by game minute
    brier_by_minute = predictions_df.groupby('game_minute')['squared_error'].agg(['mean', 'std', 'count']).reset_index()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(brier_by_minute['game_minute'], brier_by_minute['mean'], linewidth=2, color='steelblue')
    
    # Add confidence band (±1 std error)
    std_error = brier_by_minute['std'] / np.sqrt(brier_by_minute['count'])
    ax.fill_between(
        brier_by_minute['game_minute'],
        brier_by_minute['mean'] - std_error,
        brier_by_minute['mean'] + std_error,
        alpha=0.3,
        color='steelblue'
    )
    
    # Add quarter dividers
    for q_end in [12, 24, 36]:
        ax.axvline(x=q_end, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Game Minute', fontsize=12)
    ax.set_ylabel('Average Brier Score', fontsize=12)
    ax.set_title('Brier Score Evolution Over Game Time\n(Shaded area = ±1 SE)', fontsize=14, fontweight='bold')
    ax.axhline(y=0.25, color='red', linestyle='--', alpha=0.5, label='Good threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = ANALYSIS_DIR / "brier_over_time.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_file


def create_brier_distribution_plot(predictions_df):
    """Plot distribution of squared errors."""
    predictions_df['actual_outcome'] = (predictions_df['result'] == 'HIT').astype(int)
    predictions_df['squared_error'] = (predictions_df['prob_over'] - predictions_df['actual_outcome']) ** 2
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Histogram
    ax1.hist(predictions_df['squared_error'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(x=predictions_df['squared_error'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {predictions_df["squared_error"].mean():.4f}')
    ax1.axvline(x=predictions_df['squared_error'].median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {predictions_df["squared_error"].median():.4f}')
    ax1.set_xlabel('Squared Error', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Squared Errors', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot by quarter
    data_by_quarter = [predictions_df[predictions_df['quarter'] == q]['squared_error'].values for q in sorted(predictions_df['quarter'].unique())]
    ax2.boxplot(data_by_quarter, tick_labels=[f'Q{q}' for q in sorted(predictions_df['quarter'].unique())])
    ax2.set_xlabel('Quarter', fontsize=12)
    ax2.set_ylabel('Squared Error', fontsize=12)
    ax2.set_title('Squared Error Distribution by Quarter', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_file = ANALYSIS_DIR / "brier_distribution.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_file


def create_brier_by_probability_buckets_plot(predictions_df):
    """Plot Brier score by predicted probability buckets."""
    predictions_df['actual_outcome'] = (predictions_df['result'] == 'HIT').astype(int)
    predictions_df['squared_error'] = (predictions_df['prob_over'] - predictions_df['actual_outcome']) ** 2
    
    # Create probability buckets
    predictions_df['prob_bucket'] = pd.cut(
        predictions_df['prob_over'],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    )
    
    bucket_stats = predictions_df.groupby('prob_bucket', observed=True).agg({
        'squared_error': ['mean', 'count']
    }).reset_index()
    bucket_stats.columns = ['prob_bucket', 'brier_score', 'count']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(range(len(bucket_stats)), bucket_stats['brier_score'], color='steelblue', alpha=0.7)
    
    # Add sample counts on bars
    for i, (bar, count) in enumerate(zip(bars, bucket_stats['count'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'n={count:,}', ha='center', va='bottom', fontsize=10)
    
    ax.set_xticks(range(len(bucket_stats)))
    ax.set_xticklabels(bucket_stats['prob_bucket'])
    ax.set_xlabel('Predicted Probability Bucket', fontsize=12)
    ax.set_ylabel('Average Brier Score', fontsize=12)
    ax.set_title('Brier Score by Predicted Probability Range', fontsize=14, fontweight='bold')
    ax.axhline(y=0.25, color='red', linestyle='--', alpha=0.5, label='Good threshold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_file = ANALYSIS_DIR / "brier_by_probability_bucket.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_file


def create_overconfidence_analysis_plot(predictions_df):
    """Plot signed prediction errors to detect over/underconfidence."""
    predictions_df['actual_outcome'] = (predictions_df['result'] == 'HIT').astype(int)
    predictions_df['signed_error'] = predictions_df['prob_over'] - predictions_df['actual_outcome']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Scatter plot: predicted vs signed error
    sample_size = min(10000, len(predictions_df))
    sample = predictions_df.sample(n=sample_size, random_state=42)
    
    ax1.scatter(sample['prob_over'], sample['signed_error'], alpha=0.1, s=10, color='steelblue')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.set_xlabel('Predicted Probability', fontsize=12)
    ax1.set_ylabel('Signed Error (Predicted - Actual)', fontsize=12)
    ax1.set_title('Overconfidence Analysis\n(Above 0 = Overconfident, Below 0 = Underconfident)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1.1, 1.1)
    
    # Binned mean signed error
    predictions_df['prob_bin'] = pd.cut(predictions_df['prob_over'], bins=20)
    binned_errors = predictions_df.groupby('prob_bin', observed=True).agg({
        'signed_error': 'mean',
        'prob_over': 'mean'
    }).reset_index()
    
    ax2.plot(binned_errors['prob_over'], binned_errors['signed_error'], linewidth=2, marker='o', color='steelblue')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.fill_between(binned_errors['prob_over'], 0, binned_errors['signed_error'], 
                     where=(binned_errors['signed_error'] > 0), alpha=0.3, color='red', label='Overconfident')
    ax2.fill_between(binned_errors['prob_over'], 0, binned_errors['signed_error'], 
                     where=(binned_errors['signed_error'] <= 0), alpha=0.3, color='green', label='Underconfident')
    ax2.set_xlabel('Predicted Probability', fontsize=12)
    ax2.set_ylabel('Mean Signed Error', fontsize=12)
    ax2.set_title('Average Bias by Predicted Probability', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = ANALYSIS_DIR / "overconfidence_analysis.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_file


def create_quarter_probability_heatmap(predictions_df):
    """Heatmap of Brier scores by quarter and probability bucket."""
    predictions_df['actual_outcome'] = (predictions_df['result'] == 'HIT').astype(int)
    predictions_df['squared_error'] = (predictions_df['prob_over'] - predictions_df['actual_outcome']) ** 2
    
    predictions_df['prob_bucket'] = pd.cut(
        predictions_df['prob_over'],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    )
    
    heatmap_data = predictions_df.groupby(['quarter', 'prob_bucket'], observed=True)['squared_error'].mean().unstack()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='RdYlGn_r', 
                center=0.25, vmin=0, vmax=0.5, ax=ax, cbar_kws={'label': 'Brier Score'})
    ax.set_xlabel('Predicted Probability Bucket', fontsize=12)
    ax.set_ylabel('Quarter', fontsize=12)
    ax.set_title('Brier Score Heatmap: Quarter × Probability Bucket\n(Green = Good, Red = Bad)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plot_file = ANALYSIS_DIR / "quarter_probability_heatmap.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_file


def create_player_scatter_plot(player_brier_df):
    """Scatter plot of player Brier scores vs hit rates."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    scatter = ax.scatter(
        player_brier_df['hit_rate'],
        player_brier_df['brier_score'],
        s=player_brier_df['num_games'] * 5,
        alpha=0.6,
        c=player_brier_df['num_games'],
        cmap='viridis'
    )
    
    # Add reference lines
    ax.axhline(y=0.25, color='red', linestyle='--', alpha=0.5, label='Good Brier threshold')
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='50% hit rate')
    
    ax.set_xlabel('Hit Rate (Prop Overs)', fontsize=12)
    ax.set_ylabel('Brier Score', fontsize=12)
    ax.set_title('Player Performance: Brier Score vs Hit Rate\n(Bubble size = number of games)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Number of Games', fontsize=11)
    
    plt.tight_layout()
    plot_file = ANALYSIS_DIR / "player_scatter_brier_hitrate.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_file


def create_cumulative_brier_plot(predictions_df):
    """Plot cumulative Brier score over game progression."""
    predictions_df['actual_outcome'] = (predictions_df['result'] == 'HIT').astype(int)
    predictions_df['squared_error'] = (predictions_df['prob_over'] - predictions_df['actual_outcome']) ** 2
    
    # Group by minute and calculate mean (much faster than looping)
    minute_means = predictions_df.groupby('game_minute')['squared_error'].mean().sort_index()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot the mean Brier by minute (not truly cumulative, but shows progression)
    ax.plot(minute_means.index, minute_means.values, linewidth=2, color='steelblue')
    ax.axhline(y=0.25, color='red', linestyle='--', alpha=0.5, label='Good threshold')
    
    # Add quarter dividers
    for q_end in [12, 24, 36]:
        ax.axvline(x=q_end, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Game Minute', fontsize=12)
    ax.set_ylabel('Average Brier Score', fontsize=12)
    ax.set_title('Brier Score by Game Minute\n(Average performance at each minute mark)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = ANALYSIS_DIR / "cumulative_brier.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_file


def create_brier_decomposition_plot(predictions_df):
    """Decompose Brier score into reliability, resolution, and uncertainty."""
    predictions_df['actual_outcome'] = (predictions_df['result'] == 'HIT').astype(int)
    
    # Overall base rate
    base_rate = predictions_df['actual_outcome'].mean()
    
    # Bin predictions
    predictions_df['prob_bin'] = pd.cut(predictions_df['prob_over'], bins=10)
    
    binned = predictions_df.groupby('prob_bin').agg({
        'prob_over': 'mean',
        'actual_outcome': ['mean', 'count']
    }).reset_index()
    binned.columns = ['prob_bin', 'mean_forecast', 'mean_outcome', 'count']
    binned['weight'] = binned['count'] / len(predictions_df)
    
    # Brier components
    # Reliability: How far are forecasts from outcomes in each bin?
    reliability = np.sum(binned['weight'] * (binned['mean_forecast'] - binned['mean_outcome'])**2)
    
    # Resolution: How much do outcomes vary across bins?
    resolution = np.sum(binned['weight'] * (binned['mean_outcome'] - base_rate)**2)
    
    # Uncertainty: Variance of outcomes
    uncertainty = base_rate * (1 - base_rate)
    
    # Brier = Reliability - Resolution + Uncertainty
    brier_score = predictions_df['prob_over'].sub(predictions_df['actual_outcome']).pow(2).mean()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Component breakdown
    components = ['Reliability\n(Bad)', 'Resolution\n(Good)', 'Uncertainty\n(Baseline)']
    values = [reliability, -resolution, uncertainty]
    colors = ['red', 'green', 'gray']
    
    bars = ax1.bar(components, [reliability, resolution, uncertainty], color=colors, alpha=0.7)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title(f'Brier Score Decomposition\nBrier = {brier_score:.4f}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add values on bars
    for bar, val in zip(bars, [reliability, resolution, uncertainty]):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Formula visualization
    ax2.text(0.5, 0.7, 'Brier Score Decomposition Formula:', ha='center', fontsize=14, fontweight='bold')
    ax2.text(0.5, 0.5, f'Brier = Reliability - Resolution + Uncertainty', ha='center', fontsize=12)
    ax2.text(0.5, 0.35, f'{brier_score:.4f} = {reliability:.4f} - {resolution:.4f} + {uncertainty:.4f}', ha='center', fontsize=11, family='monospace')
    ax2.text(0.5, 0.15, 'Lower reliability = Better\nHigher resolution = Better\nUncertainty = Baseline variance', ha='center', fontsize=10, style='italic')
    ax2.axis('off')
    
    plt.tight_layout()
    plot_file = ANALYSIS_DIR / "brier_decomposition.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_file


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_calibration_plot(calibration_df):
    """Create calibration plot (predicted vs actual)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
    
    # Plot actual calibration
    ax.scatter(
        calibration_df['predicted_prob'], 
        calibration_df['actual_freq'],
        s=calibration_df['count'] / 10,  # Size by sample size
        alpha=0.6,
        color='blue',
        label='Model Calibration'
    )
    
    # Add error bars
    ax.errorbar(
        calibration_df['predicted_prob'],
        calibration_df['actual_freq'],
        yerr=np.sqrt(calibration_df['actual_freq'] * (1 - calibration_df['actual_freq']) / calibration_df['count']),
        fmt='none',
        ecolor='gray',
        alpha=0.3
    )
    
    ax.set_xlabel('Predicted Probability (Model)', fontsize=12)
    ax.set_ylabel('Actual Frequency (Observed)', fontsize=12)
    ax.set_title('Monte Carlo Model Calibration\n(Bubble size = sample size)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plot_file = ANALYSIS_DIR / "calibration_plot.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_file


def create_brier_by_quarter_plot(brier_by_quarter):
    """Create bar plot of Brier score by quarter."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    quarters = list(brier_by_quarter.keys())
    scores = list(brier_by_quarter.values())
    
    ax.bar(quarters, scores, color='steelblue', alpha=0.7)
    ax.set_xlabel('Quarter', fontsize=12)
    ax.set_ylabel('Brier Score', fontsize=12)
    ax.set_title('Monte Carlo Brier Score by Quarter\n(Lower is better)', fontsize=14, fontweight='bold')
    ax.axhline(y=0.25, color='red', linestyle='--', label='Good threshold (0.25)', linewidth=2)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plot_file = ANALYSIS_DIR / "brier_by_quarter.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_file


def create_player_brier_plot(player_brier_df, top_n=20):
    """Create horizontal bar plot of top/bottom players by Brier score."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    
    # Top performers (lowest Brier score)
    top_players = player_brier_df.head(top_n)
    ax1.barh(range(len(top_players)), top_players['brier_score'], color='green', alpha=0.6)
    ax1.set_yticks(range(len(top_players)))
    ax1.set_yticklabels(top_players['player_name'])
    ax1.set_xlabel('Brier Score', fontsize=11)
    ax1.set_title(f'Top {top_n} Players (Lowest Brier Score)', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    ax1.axvline(x=0.25, color='red', linestyle='--', alpha=0.5)
    ax1.grid(True, axis='x', alpha=0.3)
    
    # Bottom performers (highest Brier score)
    bottom_players = player_brier_df.tail(top_n)
    ax2.barh(range(len(bottom_players)), bottom_players['brier_score'], color='red', alpha=0.6)
    ax2.set_yticks(range(len(bottom_players)))
    ax2.set_yticklabels(bottom_players['player_name'])
    ax2.set_xlabel('Brier Score', fontsize=11)
    ax2.set_title(f'Bottom {top_n} Players (Highest Brier Score)', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()
    ax2.axvline(x=0.25, color='red', linestyle='--', alpha=0.5)
    ax2.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plot_file = ANALYSIS_DIR / "player_brier_scores.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_file


# =============================================================================
# SUMMARY REPORT
# =============================================================================

def generate_summary_report(predictions_df, brier_scores, calibration_df, player_brier_df):
    """Generate text summary report."""
    report_lines = []
    
    report_lines.append("="*80)
    report_lines.append("MONTE CARLO VALIDATION - PERFORMANCE SUMMARY")
    report_lines.append("="*80)
    report_lines.append("")
    
    # Dataset stats
    n_predictions = len(predictions_df)
    n_games = predictions_df['game_id'].nunique()
    n_players = predictions_df['player_name'].nunique()
    
    report_lines.append(f"Dataset Statistics:")
    report_lines.append(f"  - Total predictions: {n_predictions:,}")
    report_lines.append(f"  - Unique games: {n_games:,}")
    report_lines.append(f"  - Unique players: {n_players:,}")
    report_lines.append("")
    
    # Overall Brier score
    report_lines.append(f"Overall Brier Score: {brier_scores['overall']:.4f}")
    report_lines.append(f"  - Interpretation: {'GOOD ✅' if brier_scores['overall'] < 0.25 else 'NEEDS IMPROVEMENT ⚠️'}")
    report_lines.append("")
    
    # Brier by quarter
    report_lines.append("Brier Score by Quarter:")
    for quarter, score in sorted(brier_scores['by_quarter'].items()):
        report_lines.append(f"  - Q{quarter}: {score:.4f}")
    report_lines.append("")
    
    # Brier by minute bins
    report_lines.append("Brier Score by Game Stage:")
    for stage, score in brier_scores['by_minute'].items():
        report_lines.append(f"  - {stage}: {score:.4f}")
    report_lines.append("")
    
    # Calibration summary
    report_lines.append("Calibration Analysis:")
    report_lines.append(f"  - Number of bins: {len(calibration_df)}")
    report_lines.append(f"  - Mean absolute calibration error: {np.abs(calibration_df['predicted_prob'] - calibration_df['actual_freq']).mean():.4f}")
    report_lines.append("")
    
    # Top/bottom players
    report_lines.append("Top 5 Players (Lowest Brier Score):")
    for idx, row in player_brier_df.head(5).iterrows():
        report_lines.append(f"  - {row['player_name']}: {row['brier_score']:.4f} ({row['num_games']} games, {row['hit_rate']:.1%} hit rate)")
    report_lines.append("")
    
    report_lines.append("Bottom 5 Players (Highest Brier Score):")
    for idx, row in player_brier_df.tail(5).iterrows():
        report_lines.append(f"  - {row['player_name']}: {row['brier_score']:.4f} ({row['num_games']} games, {row['hit_rate']:.1%} hit rate)")
    report_lines.append("")
    
    report_lines.append("="*80)
    
    # Write to file
    report_file = ANALYSIS_DIR / "performance_summary.txt"
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    # Also print to console
    print('\n'.join(report_lines))
    
    return report_file


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze Monte Carlo performance")
    parser.add_argument("--plot", action="store_true", help="Generate full visualization suite")
    args = parser.parse_args()
    
    print("="*80)
    print("MONTE CARLO VALIDATION - PERFORMANCE ANALYSIS")
    print("="*80)
    print()
    
    # Check if combined predictions file exists
    if not PREDICTIONS_FILE.exists():
        print(f"❌ Combined predictions file not found: {PREDICTIONS_FILE}")
        print()
        print("This script requires a combined predictions.parquet file.")
        print()
        print("To fix this:")
        print("   1. Run script 06 to process games and generate predictions:")
        print("      python src/pbp_data/06_run_monte_carlo_validation.py --top-n 10 --n-sims 1000")
        print()
        print(f"   Expected output: {PREDICTIONS_FILE}")
        print(f"   Individual files: {CURRENT_PREDICTIONS_DIR}")
        return
    
    # Load combined prediction file
    print(f"📥 Loading predictions from: {PREDICTIONS_FILE}")
    predictions_df = pd.read_parquet(PREDICTIONS_FILE)
    print(f"   ✅ Loaded {len(predictions_df):,} predictions")
    print(f"   📊 {predictions_df['game_id'].nunique()} games, {predictions_df['player_name'].nunique()} players")
    print()
    
    # Calculate Brier scores
    print("📊 Calculating Brier scores...")
    brier_scores = calculate_brier_score(predictions_df)
    print(f"   Overall Brier Score: {brier_scores['overall']:.4f}")
    print()
    
    # Analyze calibration
    print("📈 Analyzing calibration...")
    calibration_df = analyze_calibration(predictions_df, n_bins=10)
    print(f"   ✅ Calibration analysis complete ({len(calibration_df)} bins)")
    print()
    
    # Analyze player performance
    print("👥 Analyzing player performance...")
    player_brier_df = analyze_player_performance(predictions_df)
    print(f"   ✅ Analyzed {len(player_brier_df)} players")
    print()
    
    # Save results
    print("💾 Saving results...")
    
    # Save Brier scores
    brier_df = pd.DataFrame([brier_scores['overall']], columns=['overall_brier_score'])
    brier_df.to_csv(ANALYSIS_DIR / "brier_scores.csv", index=False)
    
    # Save calibration
    calibration_df.to_csv(ANALYSIS_DIR / "calibration.csv", index=False)
    
    # Save player Brier scores
    player_brier_df.to_csv(ANALYSIS_DIR / "player_brier_scores.csv", index=False)
    
    print(f"   ✅ Saved CSV files to: {ANALYSIS_DIR}")
    print()
    
    # Generate plots
    print("📊 Generating plots...")
    plot_files = []
    
    # Basic plots (always generated)
    plot1 = create_calibration_plot(calibration_df)
    print(f"   ✅ Calibration plot: {plot1.name}")
    plot_files.append(plot1)
    
    plot2 = create_brier_by_quarter_plot(brier_scores['by_quarter'])
    print(f"   ✅ Brier by quarter: {plot2.name}")
    plot_files.append(plot2)
    
    plot3 = create_player_brier_plot(player_brier_df, top_n=20)
    print(f"   ✅ Player Brier scores: {plot3.name}")
    plot_files.append(plot3)
    
    # Additional plots (only with --plot flag)
    if args.plot:
        print()
        print("📊 Generating additional visualization suite...")
        
        plot4 = create_brier_over_time_plot(predictions_df)
        print(f"   ✅ Brier over time: {plot4.name}")
        plot_files.append(plot4)
        
        plot5 = create_brier_distribution_plot(predictions_df)
        print(f"   ✅ Brier distribution: {plot5.name}")
        plot_files.append(plot5)
        
        plot6 = create_brier_by_probability_buckets_plot(predictions_df)
        print(f"   ✅ Brier by probability bucket: {plot6.name}")
        plot_files.append(plot6)
        
        plot7 = create_overconfidence_analysis_plot(predictions_df)
        print(f"   ✅ Overconfidence analysis: {plot7.name}")
        plot_files.append(plot7)
        
        plot8 = create_quarter_probability_heatmap(predictions_df)
        print(f"   ✅ Quarter × probability heatmap: {plot8.name}")
        plot_files.append(plot8)
        
        plot9 = create_player_scatter_plot(player_brier_df)
        print(f"   ✅ Player scatter plot: {plot9.name}")
        plot_files.append(plot9)
        
        plot10 = create_cumulative_brier_plot(predictions_df)
        print(f"   ✅ Cumulative Brier plot: {plot10.name}")
        plot_files.append(plot10)
        
        plot11 = create_brier_decomposition_plot(predictions_df)
        print(f"   ✅ Brier decomposition: {plot11.name}")
        plot_files.append(plot11)
    
    print()
    
    # Generate summary report
    print("📝 Generating summary report...")
    report_file = generate_summary_report(predictions_df, brier_scores, calibration_df, player_brier_df)
    print()
    
    print("="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {ANALYSIS_DIR}")
    
    if args.plot:
        print(f"  Generated {len(plot_files)} plots")
        print()
        print("📂 To open all plots at once, run:")
        print()
        plot_paths = " ".join([f'"{p}"' for p in plot_files])
        print(f"  open {plot_paths}")
        print()
    else:
        print(f"  - Calibration plot: calibration_plot.png")
        print(f"  - Brier by quarter: brier_by_quarter.png")
        print(f"  - Player Brier scores: player_brier_scores.png")
        print(f"  - Summary report: performance_summary.txt")
        print()
        print("💡 Tip: Use --plot flag for full visualization suite")
        print()


if __name__ == "__main__":
    main()
