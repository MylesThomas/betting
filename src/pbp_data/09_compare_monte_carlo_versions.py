"""
Script 09: Compare Monte Carlo Versions

Purpose:
- Compare performance metrics across different versions
- Generate comparison plots and tables
- Help understand impact of code changes

Usage:
    # Compare v1 vs v2
    python src/pbp_data/09_compare_monte_carlo_versions.py --versions v1 v2
    
    # Compare all versions
    python src/pbp_data/09_compare_monte_carlo_versions.py --all
"""

import argparse
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import pytz
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)


# =============================================================================
# PATHS
# =============================================================================

VALIDATION_DIR = Path.home() / "Downloads" / "tmp" / "monte_carlo_validation"
VERSIONS_SUMMARY_FILE = VALIDATION_DIR / "versions_summary.parquet"
VERSIONS_DIR = VALIDATION_DIR / "versions"
VERSION_COMPARISONS_DIR = VALIDATION_DIR / "version_comparisons"


# =============================================================================
# COMPARISON FUNCTIONS
# =============================================================================

def compare_versions(version_names=None):
    """Compare specified versions or all if None."""
    
    if not VERSIONS_SUMMARY_FILE.exists():
        print("❌ No versions found. Run script 08 first.")
        return
    
    summary_df = pd.read_parquet(VERSIONS_SUMMARY_FILE)
    
    if version_names:
        summary_df = summary_df[summary_df['version'].isin(version_names)]
        if len(summary_df) == 0:
            print(f"❌ No versions found matching: {version_names}")
            return
    
    print("="*80)
    print("MONTE CARLO VERSION COMPARISON")
    print("="*80)
    print()
    
    # Print summary table
    print("Summary:")
    print(summary_df[['version', 'timestamp_et', 'overall_brier_score', 'n_predictions']].to_string(index=False))
    print()
    
    # Calculate improvements
    if len(summary_df) > 1:
        print("Changes from baseline (v1):")
        baseline_brier = summary_df.iloc[0]['overall_brier_score']
        
        for idx, row in summary_df.iloc[1:].iterrows():
            version = row['version']
            current_brier = row['overall_brier_score']
            
            if pd.notna(baseline_brier) and pd.notna(current_brier):
                diff = current_brier - baseline_brier
                pct_change = (diff / baseline_brier) * 100
                
                emoji = "✅" if diff < 0 else "⚠️"
                print(f"  {emoji} {version}: {current_brier:.4f} ({diff:+.4f}, {pct_change:+.1f}%)")
        print()
    
    # Plot comparison
    if len(summary_df) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Brier score over versions
        ax1 = axes[0]
        summary_df_plot = summary_df.dropna(subset=['overall_brier_score'])
        ax1.plot(summary_df_plot['version'], summary_df_plot['overall_brier_score'], 
                 marker='o', linewidth=2, markersize=10, color='steelblue')
        ax1.axhline(y=0.25, color='red', linestyle='--', label='Good threshold (0.25)', linewidth=2)
        ax1.set_xlabel('Version', fontsize=12)
        ax1.set_ylabel('Overall Brier Score', fontsize=12)
        ax1.set_title('Brier Score Evolution\n(Lower is better)', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 1)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Dataset size
        ax2 = axes[1]
        ax2.bar(summary_df['version'], summary_df['n_predictions'], color='lightblue', alpha=0.7)
        ax2.set_xlabel('Version', fontsize=12)
        ax2.set_ylabel('Number of Predictions', fontsize=12)
        ax2.set_title('Dataset Size by Version', fontsize=14, fontweight='bold')
        ax2.grid(True, axis='y', alpha=0.3)
        
        # Format y-axis
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
        
        plt.tight_layout()
        
        # Save plot with descriptive filename
        VERSION_COMPARISONS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with versions and timestamp (ET)
        versions_str = "_".join(summary_df['version'].tolist())
        et_tz = pytz.timezone('America/New_York')
        timestamp = datetime.now(et_tz).strftime("%Y-%m-%d-%H%M%S")
        filename = f"mc_version_comparison_{versions_str}_{timestamp}.png"
        comparison_plot = VERSION_COMPARISONS_DIR / filename
        
        plt.savefig(comparison_plot, dpi=150, bbox_inches='tight')
        print(f"📊 Saved comparison plot: {comparison_plot}")
        plt.close()


def compare_detailed_metrics(version1, version2, bucket_size=10):
    """
    Compare detailed metrics between two versions.
    
    Args:
        version1: First version to compare
        version2: Second version to compare
        bucket_size: Probability bucket size (10 or 20)
    """
    
    print("="*80)
    print(f"DETAILED COMPARISON: {version1} vs {version2}")
    print("="*80)
    print()
    
    # Load metrics for both versions
    metrics1_file = VERSIONS_DIR / version1 / "metrics.parquet"
    metrics2_file = VERSIONS_DIR / version2 / "metrics.parquet"
    
    if not metrics1_file.exists() or not metrics2_file.exists():
        print("❌ Metrics not found for one or both versions")
        return
    
    metrics1 = pd.read_parquet(metrics1_file)
    metrics2 = pd.read_parquet(metrics2_file)
    
    # Compare overall
    overall1 = metrics1[metrics1['metric_type'] == 'overall']
    overall2 = metrics2[metrics2['metric_type'] == 'overall']
    
    if len(overall1) > 0 and len(overall2) > 0:
        brier1 = overall1.iloc[0]['value']
        brier2 = overall2.iloc[0]['value']
        diff = brier2 - brier1
        pct_change = (diff / brier1) * 100
        
        print(f"Overall Brier Score:")
        print(f"  {version1}: {brier1:.4f}")
        print(f"  {version2}: {brier2:.4f}")
        print(f"  Change: {diff:+.4f} ({pct_change:+.1f}%)")
        print()
    
    # Compare calibration
    cal1 = metrics1[metrics1['metric_type'] == 'calibration'].copy()
    cal2 = metrics2[metrics2['metric_type'] == 'calibration'].copy()
    
    if len(cal1) > 0 and len(cal2) > 0:
        print("Calibration Error by Probability Bin:")
        print("(Lower absolute error = better calibration)")
        print()
        
        # Extract bin number from metric_name (format: "prob_bin_0.0", "prob_bin_1.0", etc.)
        cal1['bin_idx'] = cal1['metric_name'].str.extract(r'prob_bin_([\d.]+)').astype(float).astype(int)
        cal2['bin_idx'] = cal2['metric_name'].str.extract(r'prob_bin_([\d.]+)').astype(float).astype(int)
        
        cal_comparison = pd.merge(
            cal1[['bin_idx', 'value']].rename(columns={'value': version1}),
            cal2[['bin_idx', 'value']].rename(columns={'value': version2}),
            on='bin_idx'
        ).sort_values('bin_idx')
        
        cal_comparison['change'] = cal_comparison[version2] - cal_comparison[version1]
        
        # Calculate percent change based on absolute values (since we care about magnitude)
        cal_comparison['pct_change'] = (cal_comparison['change'] / cal_comparison[version1].abs()) * 100
        
        # Format probability bin ranges (10% bins: 0-10%, 10-20%, etc.)
        cal_comparison['prob_range'] = cal_comparison['bin_idx'].apply(
            lambda x: f"{x*10:>2.0f}-{(x+1)*10:>2.0f}%"
        )
        
        # Print formatted table
        print(f"{'Bin':<10} {version1:>10} {version2:>10} {'Change':>10} {'% Chg':>8}  {'Status'}")
        print("-" * 70)
        
        for _, row in cal_comparison.iterrows():
            # Determine emoji based on absolute change (closer to 0 is better)
            abs_v1 = abs(row[version1])
            abs_v2 = abs(row[version2])
            abs_change = abs_v2 - abs_v1
            
            if abs_change < -0.005:
                emoji = "✅"
                status = "Better"
            elif abs_change > 0.005:
                emoji = "⚠️"
                status = "Worse"
            else:
                emoji = "➖"
                status = "Stable"
            
            print(f"{row['prob_range']:<10} {row[version1]:>10.4f} {row[version2]:>10.4f} "
                  f"{row['change']:>+10.4f} {row['pct_change']:>+7.1f}%  {emoji} {status}")
        print()
    
    # Compare heatmap squares (Quarter × Probability Bucket)
    # Generate heatmap data directly from predictions
    print(f"Heatmap Square Comparison (Quarter × Probability Bucket - {bucket_size}% bins):")
    print("(Brier Score by cell - Lower is better)")
    print()
    
    # Load predictions for both versions
    preds1_file = VERSIONS_DIR / version1 / "predictions.parquet"
    preds2_file = VERSIONS_DIR / version2 / "predictions.parquet"
    
    if not preds1_file.exists() or not preds2_file.exists():
        print("⚠️ Predictions not found for heatmap comparison")
        return
    
    def calculate_heatmap_data(preds_df, bucket_size=10):
        """Calculate Brier score by quarter and probability bucket."""
        preds_df = preds_df.copy()
        preds_df['actual_outcome'] = (preds_df['result'] == 'HIT').astype(int)
        preds_df['squared_error'] = (preds_df['prob_over'] - preds_df['actual_outcome']) ** 2
        
        if bucket_size == 10:
            # 10% buckets: 0-10%, 10-20%, ..., 90-100%
            bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            labels = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', 
                     '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
        else:
            # 20% buckets: 0-20%, 20-40%, ..., 80-100%
            bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
            labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
        
        preds_df['prob_bucket'] = pd.cut(
            preds_df['prob_over'],
            bins=bins,
            labels=labels
        )
        
        heatmap = preds_df.groupby(['quarter', 'prob_bucket'], observed=True)['squared_error'].mean().unstack()
        return heatmap
    
    preds1 = pd.read_parquet(preds1_file)
    preds2 = pd.read_parquet(preds2_file)
    
    heatmap1 = calculate_heatmap_data(preds1, bucket_size)
    heatmap2 = calculate_heatmap_data(preds2, bucket_size)
    
    # Compare each quarter
    all_quarters = sorted(set(heatmap1.index).union(set(heatmap2.index)))
    
    if bucket_size == 10:
        all_buckets = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', 
                      '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
    else:
        all_buckets = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    
    for quarter in all_quarters:
        quarter_label = f"Q{quarter}" if quarter <= 4 else f"OT{quarter-4}"
        print(f"{quarter_label}:")
        print(f"  {'Bucket':<12} {version1:>10} {version2:>10} {'Change':>10}  {'Status'}")
        print("  " + "-" * 60)
        
        for bucket in all_buckets:
            v1_val = heatmap1.loc[quarter, bucket] if quarter in heatmap1.index and bucket in heatmap1.columns else None
            v2_val = heatmap2.loc[quarter, bucket] if quarter in heatmap2.index and bucket in heatmap2.columns else None
            
            if pd.notna(v1_val) and pd.notna(v2_val):
                change = v2_val - v1_val
                
                # Determine status (lower Brier is better)
                if change < -0.01:
                    emoji = "✅"
                    status = "Better"
                elif change > 0.01:
                    emoji = "⚠️"
                    status = "Worse"
                else:
                    emoji = "➖"
                    status = "Stable"
                
                v1_str = f"{v1_val:.4f}"
                v2_str = f"{v2_val:.4f}"
                change_str = f"{change:+.4f}"
                
            elif pd.notna(v2_val) and pd.isna(v1_val):
                # New cell in v2
                emoji = "🆕"
                status = "New"
                v1_str = "N/A"
                v2_str = f"{v2_val:.4f}"
                change_str = "N/A"
                
            elif pd.notna(v1_val) and pd.isna(v2_val):
                # Cell removed in v2
                emoji = "❌"
                status = "Removed"
                v1_str = f"{v1_val:.4f}"
                v2_str = "N/A"
                change_str = "N/A"
                
            else:
                # Both N/A
                continue
            
            print(f"  {bucket:<12} {v1_str:>10} {v2_str:>10} {change_str:>10}  {emoji} {status}")
        print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compare Monte Carlo versions")
    parser.add_argument("--versions", nargs="+", help="Version names to compare (e.g., v1 v2)")
    parser.add_argument("--all", action="store_true", help="Compare all versions")
    parser.add_argument("--detailed", nargs=2, metavar=('V1', 'V2'), 
                       help="Detailed comparison between two versions")
    parser.add_argument("--bucket-size", type=int, choices=[10, 20], default=10,
                       help="Probability bucket size for heatmap (default: 10)")
    
    args = parser.parse_args()
    
    if args.detailed:
        compare_detailed_metrics(args.detailed[0], args.detailed[1], args.bucket_size)
    elif args.all:
        compare_versions()
    elif args.versions:
        compare_versions(args.versions)
    else:
        # Default: compare all
        compare_versions()


if __name__ == "__main__":
    main()
