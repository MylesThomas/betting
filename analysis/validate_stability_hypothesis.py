"""
Validate Stability Hypothesis: Does R² Predict Strategy Performance?

Tests whether player stability (R²) actually correlates with momentum/reversal
strategy profitability, or if it's just theoretical noise.

Core Hypothesis:
- High R² (predictable) → Momentum should work better than reversal
- Low R² (erratic) → Reversal should work better than momentum

Validation Approach:
1. Bucket players by stability (R²)
2. Compare momentum vs reversal ROI within each bucket
3. Test if relationship holds statistically
4. Output actionable thresholds

Output:
- ~/Downloads/tmp/prop_predictive_power_analysis/stability_validation/
  - correlation_analysis.csv
  - bucket_performance.csv
  - actionable_thresholds.csv
  - visualizations/

Usage:
    # Analyze existing results
    python analysis/validate_stability_hypothesis.py
    
    # Specify different input directory
    python analysis/validate_stability_hypothesis.py --input-dir /path/to/results

Author: Myles Thomas
Date: 2026-02-10
"""

import sys
from pathlib import Path
import argparse
import warnings

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_BASE = Path.home() / 'Downloads' / 'tmp' / 'prop_predictive_power_analysis'
INPUT_DIR = OUTPUT_BASE / 'player_level_grid'
OUTPUT_DIR = OUTPUT_BASE / 'stability_validation'

# =============================================================================
# DATA LOADING
# =============================================================================

def load_results(input_dir):
    """Load player-level grid search results"""
    print(f"\n{'='*80}")
    print(f"LOADING RESULTS")
    print(f"{'='*80}")
    
    strategies_file = input_dir / 'player_strategies_all.csv'
    stability_file = input_dir / 'player_stability_by_market.csv'
    
    if not strategies_file.exists():
        print(f"❌ File not found: {strategies_file}")
        return None, None
    
    if not stability_file.exists():
        print(f"❌ File not found: {stability_file}")
        return None, None
    
    df_strat = pd.read_csv(strategies_file)
    df_stab = pd.read_csv(stability_file)
    
    print(f"✅ Loaded {len(df_strat):,} strategy results")
    print(f"✅ Loaded {len(df_stab):,} stability measurements")
    
    return df_strat, df_stab


# =============================================================================
# BUCKET ANALYSIS
# =============================================================================

def create_stability_buckets(df):
    """Create stability buckets and analyze strategy performance"""
    print(f"\n{'='*80}")
    print(f"STABILITY BUCKET ANALYSIS")
    print(f"{'='*80}")
    
    # Define buckets
    buckets = [
        (0.0, 0.2, 'Very Unstable (R² 0.0-0.2)'),
        (0.2, 0.4, 'Somewhat Stable (R² 0.2-0.4)'),
        (0.4, 0.6, 'Stable (R² 0.4-0.6)'),
        (0.6, 1.0, 'Highly Stable (R² 0.6+)')
    ]
    
    results = []
    
    for low, high, label in buckets:
        bucket_df = df[(df['stability_r2'] >= low) & (df['stability_r2'] < high)].copy()
        
        if len(bucket_df) == 0:
            print(f"\n{label}: No data")
            continue
        
        print(f"\n{label}:")
        print(f"   Total strategies: {len(bucket_df):,}")
        
        # Momentum strategies
        momentum_df = bucket_df[bucket_df['strategy_type'] == 'Momentum']
        momentum_roi = momentum_df['roi'].mean()
        momentum_count = len(momentum_df)
        momentum_profitable = (momentum_df['roi'] > 0).sum()
        momentum_profitable_pct = (momentum_profitable / momentum_count * 100) if momentum_count > 0 else 0
        
        print(f"   Momentum: {momentum_count:>5,} strategies | Avg ROI: {momentum_roi:>6.2f}% | Profitable: {momentum_profitable_pct:>4.1f}%")
        
        # Reversal strategies
        reversal_df = bucket_df[bucket_df['strategy_type'] == 'Reversal']
        reversal_roi = reversal_df['roi'].mean()
        reversal_count = len(reversal_df)
        reversal_profitable = (reversal_df['roi'] > 0).sum()
        reversal_profitable_pct = (reversal_profitable / reversal_count * 100) if reversal_count > 0 else 0
        
        print(f"   Reversal: {reversal_count:>5,} strategies | Avg ROI: {reversal_roi:>6.2f}% | Profitable: {reversal_profitable_pct:>4.1f}%")
        
        # Which is better?
        if momentum_roi > reversal_roi:
            winner = 'MOMENTUM'
            diff = momentum_roi - reversal_roi
        else:
            winner = 'REVERSAL'
            diff = reversal_roi - momentum_roi
        
        print(f"   Winner: {winner} by {diff:.2f}%")
        
        # Statistical test
        if momentum_count > 0 and reversal_count > 0:
            t_stat, p_value = stats.ttest_ind(momentum_df['roi'], reversal_df['roi'])
            sig = '✅ SIGNIFICANT' if p_value < 0.05 else '⚠️  NOT SIGNIFICANT'
            print(f"   T-test: p={p_value:.4f} {sig}")
        else:
            p_value = None
        
        results.append({
            'bucket': label,
            'r2_low': low,
            'r2_high': high,
            'r2_midpoint': (low + high) / 2,
            'momentum_count': momentum_count,
            'momentum_avg_roi': momentum_roi,
            'momentum_profitable_pct': momentum_profitable_pct,
            'reversal_count': reversal_count,
            'reversal_avg_roi': reversal_roi,
            'reversal_profitable_pct': reversal_profitable_pct,
            'winner': winner,
            'roi_difference': diff,
            'p_value': p_value
        })
    
    return pd.DataFrame(results)


# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

def analyze_correlations(df):
    """Analyze correlation between R² and strategy ROI"""
    print(f"\n{'='*80}")
    print(f"CORRELATION ANALYSIS")
    print(f"{'='*80}")
    
    # Overall correlation
    corr_all, p_all = stats.pearsonr(df['stability_r2'], df['roi'])
    print(f"\nOverall R² vs ROI:")
    print(f"   Correlation: {corr_all:.3f}")
    print(f"   P-value: {p_all:.4f}")
    print(f"   Significant: {'YES ✅' if p_all < 0.05 else 'NO ❌'}")
    
    # Momentum strategies
    momentum_df = df[df['strategy_type'] == 'Momentum']
    if len(momentum_df) > 10:
        corr_m, p_m = stats.pearsonr(momentum_df['stability_r2'], momentum_df['roi'])
        print(f"\nMomentum R² vs ROI:")
        print(f"   Correlation: {corr_m:.3f}")
        print(f"   P-value: {p_m:.4f}")
        print(f"   Interpretation: {'Higher R² → Higher ROI ✅' if corr_m > 0.1 else 'No clear relationship ❌'}")
    else:
        corr_m, p_m = None, None
    
    # Reversal strategies
    reversal_df = df[df['strategy_type'] == 'Reversal']
    if len(reversal_df) > 10:
        corr_r, p_r = stats.pearsonr(reversal_df['stability_r2'], reversal_df['roi'])
        print(f"\nReversal R² vs ROI:")
        print(f"   Correlation: {corr_r:.3f}")
        print(f"   P-value: {p_r:.4f}")
        print(f"   Interpretation: {'Higher R² → Lower ROI ✅' if corr_r < -0.1 else 'No clear relationship ❌'}")
    else:
        corr_r, p_r = None, None
    
    return {
        'overall_corr': corr_all,
        'overall_p': p_all,
        'momentum_corr': corr_m,
        'momentum_p': p_m,
        'reversal_corr': corr_r,
        'reversal_p': p_r
    }


# =============================================================================
# CROSSOVER ANALYSIS
# =============================================================================

def find_crossover_point(df_buckets):
    """Find R² threshold where momentum becomes better than reversal"""
    print(f"\n{'='*80}")
    print(f"CROSSOVER ANALYSIS")
    print(f"{'='*80}")
    
    # Find where momentum ROI > reversal ROI
    df_buckets['momentum_better'] = df_buckets['momentum_avg_roi'] > df_buckets['reversal_avg_roi']
    
    crossover = None
    for i in range(len(df_buckets) - 1):
        current = df_buckets.iloc[i]
        next_row = df_buckets.iloc[i + 1]
        
        if not current['momentum_better'] and next_row['momentum_better']:
            # Crossover found
            crossover = next_row['r2_low']
            print(f"\n✅ CROSSOVER FOUND at R² = {crossover:.2f}")
            print(f"   Below {crossover:.2f}: Use REVERSAL strategies")
            print(f"   Above {crossover:.2f}: Use MOMENTUM strategies")
            break
    
    if crossover is None:
        # Check which is consistently better
        momentum_wins = (df_buckets['momentum_avg_roi'] > df_buckets['reversal_avg_roi']).sum()
        reversal_wins = (df_buckets['reversal_avg_roi'] > df_buckets['momentum_avg_roi']).sum()
        
        if momentum_wins > reversal_wins:
            print(f"\n⚠️  NO CROSSOVER: Momentum consistently better across all R² ranges")
            recommendation = "Use MOMENTUM regardless of R²"
        elif reversal_wins > momentum_wins:
            print(f"\n⚠️  NO CROSSOVER: Reversal consistently better across all R² ranges")
            recommendation = "Use REVERSAL regardless of R²"
        else:
            print(f"\n⚠️  NO CLEAR PATTERN: Results are inconsistent")
            recommendation = "R² does not predict strategy performance"
    else:
        recommendation = f"Use REVERSAL if R² < {crossover:.2f}, MOMENTUM if R² > {crossover:.2f}"
    
    return crossover, recommendation


# =============================================================================
# ACTIONABLE RULES
# =============================================================================

def generate_actionable_rules(df, df_buckets, crossover):
    """Generate clear, actionable trading rules"""
    print(f"\n{'='*80}")
    print(f"ACTIONABLE TRADING RULES")
    print(f"{'='*80}")
    
    rules = []
    
    if crossover is not None:
        # Rule based on crossover
        rules.append({
            'rule_type': 'Crossover-Based',
            'condition': f'R² < {crossover:.2f}',
            'action': 'Use REVERSAL strategies',
            'reasoning': 'Player is unpredictable, fade streaks'
        })
        
        rules.append({
            'rule_type': 'Crossover-Based',
            'condition': f'R² >= {crossover:.2f}',
            'action': 'Use MOMENTUM strategies',
            'reasoning': 'Player is predictable, ride streaks'
        })
    else:
        # Rules based on bucket analysis
        for _, bucket in df_buckets.iterrows():
            if bucket['winner'] == 'MOMENTUM':
                action = 'Use MOMENTUM strategies'
                reasoning = f"Avg ROI: {bucket['momentum_avg_roi']:.2f}% vs {bucket['reversal_avg_roi']:.2f}%"
            else:
                action = 'Use REVERSAL strategies'
                reasoning = f"Avg ROI: {bucket['reversal_avg_roi']:.2f}% vs {bucket['momentum_avg_roi']:.2f}%"
            
            rules.append({
                'rule_type': 'Bucket-Based',
                'condition': f"{bucket['r2_low']:.2f} ≤ R² < {bucket['r2_high']:.2f}",
                'action': action,
                'reasoning': reasoning
            })
    
    df_rules = pd.DataFrame(rules)
    
    print("\n")
    for _, rule in df_rules.iterrows():
        print(f"{rule['condition']:25s} → {rule['action']:30s} | {rule['reasoning']}")
    
    return df_rules


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_visualizations(df, df_buckets, output_dir):
    """Create visualizations of R² vs ROI relationship"""
    print(f"\n{'='*80}")
    print(f"CREATING VISUALIZATIONS")
    print(f"{'='*80}")
    
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Scatter plot: R² vs ROI by strategy type
    fig, ax = plt.subplots(figsize=(12, 8))
    
    momentum = df[df['strategy_type'] == 'Momentum']
    reversal = df[df['strategy_type'] == 'Reversal']
    
    ax.scatter(momentum['stability_r2'], momentum['roi'], alpha=0.3, s=20, c='blue', label='Momentum')
    ax.scatter(reversal['stability_r2'], reversal['roi'], alpha=0.3, s=20, c='red', label='Reversal')
    
    # Add trend lines
    if len(momentum) > 10:
        z_m = np.polyfit(momentum['stability_r2'], momentum['roi'], 1)
        p_m = np.poly1d(z_m)
        x_m = np.linspace(momentum['stability_r2'].min(), momentum['stability_r2'].max(), 100)
        ax.plot(x_m, p_m(x_m), 'b--', alpha=0.8, linewidth=2, label=f'Momentum Trend')
    
    if len(reversal) > 10:
        z_r = np.polyfit(reversal['stability_r2'], reversal['roi'], 1)
        p_r = np.poly1d(z_r)
        x_r = np.linspace(reversal['stability_r2'].min(), reversal['stability_r2'].max(), 100)
        ax.plot(x_r, p_r(x_r), 'r--', alpha=0.8, linewidth=2, label=f'Reversal Trend')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_xlabel('Stability (R²)', fontsize=12, fontweight='bold')
    ax.set_ylabel('ROI (%)', fontsize=12, fontweight='bold')
    ax.set_title('Does Stability Predict Strategy Performance?', fontsize=14, fontweight='bold', pad=20)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'stability_vs_roi_scatter.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ {viz_dir / 'stability_vs_roi_scatter.png'}")
    plt.close()
    
    # 2. Bucket comparison chart
    fig, ax = plt.subplots(figsize=(14, 8))
    
    x = np.arange(len(df_buckets))
    width = 0.35
    
    momentum_roi = df_buckets['momentum_avg_roi'].values
    reversal_roi = df_buckets['reversal_avg_roi'].values
    
    bars1 = ax.bar(x - width/2, momentum_roi, width, label='Momentum', color='blue', alpha=0.7)
    bars2 = ax.bar(x + width/2, reversal_roi, width, label='Reversal', color='red', alpha=0.7)
    
    ax.set_xlabel('Stability Bucket', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average ROI (%)', fontsize=12, fontweight='bold')
    ax.set_title('Strategy Performance by Stability Bucket', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(df_buckets['bucket'], rotation=15, ha='right')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'bucket_performance_comparison.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ {viz_dir / 'bucket_performance_comparison.png'}")
    plt.close()
    
    # 3. Profitable percentage by bucket
    fig, ax = plt.subplots(figsize=(14, 8))
    
    momentum_prof = df_buckets['momentum_profitable_pct'].values
    reversal_prof = df_buckets['reversal_profitable_pct'].values
    
    bars1 = ax.bar(x - width/2, momentum_prof, width, label='Momentum', color='blue', alpha=0.7)
    bars2 = ax.bar(x + width/2, reversal_prof, width, label='Reversal', color='red', alpha=0.7)
    
    ax.set_xlabel('Stability Bucket', fontsize=12, fontweight='bold')
    ax.set_ylabel('% Profitable Strategies', fontsize=12, fontweight='bold')
    ax.set_title('Profitable Strategy Rate by Stability Bucket', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(df_buckets['bucket'], rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'profitable_rate_by_bucket.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ {viz_dir / 'profitable_rate_by_bucket.png'}")
    plt.close()


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Validate stability hypothesis: Does R² predict strategy performance?'
    )
    parser.add_argument('--input-dir', type=str,
                        help='Directory with player-level results')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir) if args.input_dir else INPUT_DIR
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"VALIDATE STABILITY HYPOTHESIS")
    print(f"{'='*80}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    
    # Load data
    df_strat, df_stab = load_results(input_dir)
    
    if df_strat is None or df_stab is None:
        print("\n❌ Failed to load data")
        return
    
    # Filter to strategies with stability measurements
    df = df_strat.dropna(subset=['stability_r2']).copy()
    print(f"\n✅ {len(df):,} strategies with stability measurements")
    
    # Bucket analysis
    df_buckets = create_stability_buckets(df)
    
    # Correlation analysis
    corr_results = analyze_correlations(df)
    
    # Find crossover point
    crossover, recommendation = find_crossover_point(df_buckets)
    
    # Generate actionable rules
    df_rules = generate_actionable_rules(df, df_buckets, crossover)
    
    # Create visualizations
    create_visualizations(df, df_buckets, output_dir)
    
    # Save results
    print(f"\n{'='*80}")
    print(f"SAVING RESULTS")
    print(f"{'='*80}")
    
    df_buckets.to_csv(output_dir / 'bucket_performance.csv', index=False)
    print(f"   ✅ {output_dir / 'bucket_performance.csv'}")
    
    df_rules.to_csv(output_dir / 'actionable_rules.csv', index=False)
    print(f"   ✅ {output_dir / 'actionable_rules.csv'}")
    
    # Save correlation results
    df_corr = pd.DataFrame([corr_results])
    df_corr.to_csv(output_dir / 'correlation_analysis.csv', index=False)
    print(f"   ✅ {output_dir / 'correlation_analysis.csv'}")
    
    # Save summary
    with open(output_dir / 'summary.txt', 'w') as f:
        f.write("STABILITY VALIDATION SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Crossover Point: {crossover if crossover else 'None found'}\n")
        f.write(f"Recommendation: {recommendation}\n\n")
        f.write("Correlation Results:\n")
        f.write(f"  Overall R² vs ROI: {corr_results['overall_corr']:.3f} (p={corr_results['overall_p']:.4f})\n")
        f.write(f"  Momentum R² vs ROI: {corr_results['momentum_corr']:.3f} (p={corr_results['momentum_p']:.4f})\n")
        f.write(f"  Reversal R² vs ROI: {corr_results['reversal_corr']:.3f} (p={corr_results['reversal_p']:.4f})\n\n")
        f.write("Interpretation:\n")
        if crossover:
            f.write(f"  R² DOES predict strategy performance. Use threshold of {crossover:.2f}\n")
        else:
            f.write(f"  R² does NOT clearly predict strategy performance.\n")
    
    print(f"   ✅ {output_dir / 'summary.txt'}")
    
    # Final verdict
    print(f"\n{'='*80}")
    print(f"FINAL VERDICT")
    print(f"{'='*80}")
    print(f"\n{recommendation}")
    
    if corr_results['momentum_corr'] and corr_results['momentum_corr'] > 0.1 and corr_results['momentum_p'] < 0.05:
        print(f"\n✅ MOMENTUM: Higher R² correlates with higher ROI (validated)")
    else:
        print(f"\n❌ MOMENTUM: No clear relationship between R² and ROI")
    
    if corr_results['reversal_corr'] and corr_results['reversal_corr'] < -0.1 and corr_results['reversal_p'] < 0.05:
        print(f"✅ REVERSAL: Higher R² correlates with lower ROI (validated)")
    else:
        print(f"❌ REVERSAL: No clear relationship between R² and ROI")
    
    print(f"\n✅ Validation complete!")
    print(f"📁 Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
