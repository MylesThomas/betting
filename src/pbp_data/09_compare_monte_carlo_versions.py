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
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime


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
        
        # Generate filename with versions and timestamp
        versions_str = "_".join(summary_df['version'].tolist())
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        filename = f"mc_version_comparison_{versions_str}_{timestamp}.png"
        comparison_plot = VERSION_COMPARISONS_DIR / filename
        
        plt.savefig(comparison_plot, dpi=150, bbox_inches='tight')
        print(f"📊 Saved comparison plot: {comparison_plot}")
        plt.close()


def compare_detailed_metrics(version1, version2):
    """Compare detailed metrics between two versions."""
    
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
        cal1['bin'] = cal1['metric_name'].str.extract(r'(\d+)').astype(int)
        cal2['bin'] = cal2['metric_name'].str.extract(r'(\d+)').astype(int)
        
        cal_comparison = pd.merge(
            cal1[['bin', 'value']].rename(columns={'value': version1}),
            cal2[['bin', 'value']].rename(columns={'value': version2}),
            on='bin'
        )
        cal_comparison['change'] = cal_comparison[version2] - cal_comparison[version1]
        
        print(cal_comparison.to_string(index=False))
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
    
    args = parser.parse_args()
    
    if args.detailed:
        compare_detailed_metrics(args.detailed[0], args.detailed[1])
    elif args.all:
        compare_versions()
    elif args.versions:
        compare_versions(args.versions)
    else:
        # Default: compare all
        compare_versions()


if __name__ == "__main__":
    main()
