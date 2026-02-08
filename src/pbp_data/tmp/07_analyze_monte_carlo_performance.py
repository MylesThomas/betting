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
    python src/pbp_data/tmp/07_analyze_monte_carlo_performance.py
    
Output:
    ~/Downloads/tmp/monte_carlo_validation/analysis/
        - brier_scores.csv
        - calibration_plot.png
        - performance_summary.txt
"""

import duckdb
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Import path functions
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from pbp_data.monte_carlo_utils import get_project_root


# =============================================================================
# PATHS
# =============================================================================

VALIDATION_DIR = Path.home() / "Downloads" / "tmp" / "monte_carlo_validation"
PREDICTIONS_FILE = VALIDATION_DIR / "predictions.parquet"
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
    brier_by_minute = predictions_df.groupby('minute_bin')['squared_error'].mean()
    
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
    """Analyze Brier scores by player."""
    # Get one row per game (use final prediction)
    game_level = predictions_df.sort_values('game_minute').groupby(['player_name', 'game_id']).last().reset_index()
    
    # Convert result to binary
    game_level['actual_outcome'] = (game_level['result'] == 'HIT').astype(int)
    game_level['squared_error'] = (game_level['prob_over'] - game_level['actual_outcome']) ** 2
    
    # Brier score by player
    player_brier = game_level.groupby('player_name').agg({
        'squared_error': 'mean',
        'game_id': 'count',
        'result': lambda x: (x == 'HIT').mean(),  # Hit rate
    }).reset_index()
    
    player_brier.columns = ['player_name', 'brier_score', 'num_games', 'hit_rate']
    player_brier = player_brier.sort_values('brier_score')
    
    return player_brier


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
    print("="*80)
    print("MONTE CARLO VALIDATION - PERFORMANCE ANALYSIS")
    print("="*80)
    print()
    
    # Check if predictions file exists
    if not PREDICTIONS_FILE.exists():
        print(f"❌ Predictions file not found: {PREDICTIONS_FILE}")
        print("   Run script 06 first to generate predictions.")
        return
    
    # Load predictions
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
    
    plot1 = create_calibration_plot(calibration_df)
    print(f"   ✅ Calibration plot: {plot1}")
    
    plot2 = create_brier_by_quarter_plot(brier_scores['by_quarter'])
    print(f"   ✅ Brier by quarter: {plot2}")
    
    plot3 = create_player_brier_plot(player_brier_df, top_n=20)
    print(f"   ✅ Player Brier scores: {plot3}")
    
    print()
    
    # Generate summary report
    print("📝 Generating summary report...")
    report_file = generate_summary_report(predictions_df, brier_scores, calibration_df, player_brier_df)
    print()
    
    print("="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {ANALYSIS_DIR}")
    print(f"  - Calibration plot: calibration_plot.png")
    print(f"  - Brier by quarter: brier_by_quarter.png")
    print(f"  - Player Brier scores: player_brier_scores.png")
    print(f"  - Summary report: performance_summary.txt")
    print()


if __name__ == "__main__":
    main()
