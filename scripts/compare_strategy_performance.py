"""
Compare Strategy Performance Over Time

Context:
- User needs to track how strategy rankings change week-over-week
- This script compares the latest analysis to a previous timestamped version
- Shows which strategies improved/declined in ROI, rank, and profitability

Usage:
    # Compare latest to a specific previous run
    python scripts/compare_strategy_performance.py --previous data/04_output/all_strategies_ranked_20260125_224217.csv
    
    # Compare latest to most recent previous run (auto-detect)
    python scripts/compare_strategy_performance.py --auto

Author: Myles Thomas
Date: 2026-01-25
"""

import pandas as pd
import argparse
import glob
import os
from datetime import datetime


def load_analysis(filepath: str) -> pd.DataFrame:
    """Load a strategy analysis CSV."""
    df = pd.read_csv(filepath)
    df['rank'] = range(1, len(df) + 1)
    return df


def compare_analyses(df_current: pd.DataFrame, df_previous: pd.DataFrame) -> pd.DataFrame:
    """
    Compare two strategy analyses.
    
    Args:
        df_current: Current analysis DataFrame
        df_previous: Previous analysis DataFrame
    
    Returns:
        DataFrame with comparison metrics
    """
    # Merge on strategy name
    df_merged = df_current.merge(
        df_previous[['strategy_name', 'roi', 'total_profit', 'rank', 'win_rate']],
        on='strategy_name',
        how='outer',
        suffixes=('_current', '_previous')
    )
    
    # Calculate changes
    df_merged['roi_change'] = df_merged['roi_current'] - df_merged['roi_previous']
    df_merged['profit_change'] = df_merged['total_profit_current'] - df_merged['total_profit_previous']
    df_merged['rank_change'] = df_merged['rank_previous'] - df_merged['rank_current']  # Positive = moved up
    df_merged['win_rate_change'] = df_merged['win_rate_current'] - df_merged['win_rate_previous']
    
    # Sort by current rank
    df_merged = df_merged.sort_values('rank_current').reset_index(drop=True)
    
    return df_merged


def print_comparison_report(df_comparison: pd.DataFrame, n: int = 20):
    """Print a comparison report."""
    print("="*100)
    print("STRATEGY PERFORMANCE COMPARISON")
    print("="*100)
    
    print(f"\n{'='*100}")
    print(f"TOP {n} STRATEGIES (CURRENT)")
    print("="*100)
    print(f"{'Rank':<5} {'Strategy':<35} {'ROI':<12} {'Δ ROI':<10} {'Δ Rank':<10} {'Profit':<15} {'Δ Profit':<12}")
    print("-"*100)
    
    for idx, row in df_comparison.head(n).iterrows():
        rank_current = int(row['rank_current']) if pd.notna(row['rank_current']) else '?'
        strategy = row['strategy_name']
        
        roi_current = row['roi_current']
        roi_prev = row['roi_previous']
        roi_change = row['roi_change']
        
        rank_change = row['rank_change']
        
        profit_current = row['total_profit_current']
        profit_change = row['profit_change']
        
        # Format ROI
        if pd.notna(roi_current):
            roi_str = f"{roi_current:+6.1f}%"
        else:
            roi_str = "NEW"
        
        # Format ROI change
        if pd.notna(roi_change) and roi_change != 0:
            roi_change_str = f"{roi_change:+6.1f}%"
        else:
            roi_change_str = "-"
        
        # Format rank change
        if pd.notna(rank_change) and rank_change != 0:
            if rank_change > 0:
                rank_change_str = f"↑{int(rank_change)}"
            else:
                rank_change_str = f"↓{abs(int(rank_change))}"
        else:
            rank_change_str = "-"
        
        # Format profit
        if pd.notna(profit_current):
            profit_str = f"${profit_current:>9,.0f}"
        else:
            profit_str = "NEW"
        
        # Format profit change
        if pd.notna(profit_change) and profit_change != 0:
            profit_change_str = f"${profit_change:+9,.0f}"
        else:
            profit_change_str = "-"
        
        print(f"{rank_current:<5} {strategy:<35} {roi_str:<12} {roi_change_str:<10} {rank_change_str:<10} {profit_str:<15} {profit_change_str:<12}")
    
    # Biggest movers
    print(f"\n{'='*100}")
    print("BIGGEST MOVERS")
    print("="*100)
    
    # Biggest ROI improvements
    df_roi_gains = df_comparison[pd.notna(df_comparison['roi_change'])].nlargest(5, 'roi_change')
    if len(df_roi_gains) > 0:
        print("\nTop 5 ROI Improvements:")
        for idx, row in df_roi_gains.iterrows():
            print(f"  {row['strategy_name']:<35} {row['roi_change']:+6.1f}% (from {row['roi_previous']:+6.1f}% to {row['roi_current']:+6.1f}%)")
    
    # Biggest ROI declines
    df_roi_losses = df_comparison[pd.notna(df_comparison['roi_change'])].nsmallest(5, 'roi_change')
    if len(df_roi_losses) > 0:
        print("\nTop 5 ROI Declines:")
        for idx, row in df_roi_losses.iterrows():
            print(f"  {row['strategy_name']:<35} {row['roi_change']:+6.1f}% (from {row['roi_previous']:+6.1f}% to {row['roi_current']:+6.1f}%)")
    
    # Biggest rank improvements
    df_rank_gains = df_comparison[pd.notna(df_comparison['rank_change'])].nlargest(5, 'rank_change')
    if len(df_rank_gains) > 0:
        print("\nBiggest Rank Improvements:")
        for idx, row in df_rank_gains.iterrows():
            print(f"  {row['strategy_name']:<35} Moved from #{int(row['rank_previous'])} to #{int(row['rank_current'])} ({int(row['rank_change'])} spots)")
    
    # New strategies
    df_new = df_comparison[pd.isna(df_comparison['roi_previous'])]
    if len(df_new) > 0:
        print(f"\nNew Strategies (not in previous analysis): {len(df_new)}")
        for idx, row in df_new.iterrows():
            print(f"  {row['strategy_name']:<35} ROI: {row['roi_current']:+6.1f}%")
    
    # Removed strategies
    df_removed = df_comparison[pd.isna(df_comparison['roi_current'])]
    if len(df_removed) > 0:
        print(f"\nRemoved Strategies (not in current analysis): {len(df_removed)}")
        for idx, row in df_removed.iterrows():
            print(f"  {row['strategy_name']:<35} Previous ROI: {row['roi_previous']:+6.1f}%")


def find_most_recent_analysis(output_dir: str, exclude_latest: bool = True) -> str:
    """
    Find the most recent timestamped analysis file.
    
    Args:
        output_dir: Directory containing analysis files
        exclude_latest: If True, exclude the *_latest.csv files
    
    Returns:
        Path to most recent analysis file
    """
    pattern = f"{output_dir}/all_strategies_ranked_*.csv"
    files = glob.glob(pattern)
    
    if exclude_latest:
        files = [f for f in files if not f.endswith('_latest.csv')]
    
    if not files:
        raise ValueError(f"No analysis files found matching {pattern}")
    
    # Sort by modification time
    files.sort(key=os.path.getmtime, reverse=True)
    
    return files[0]


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Compare strategy performance between two analysis runs'
    )
    
    parser.add_argument('--current', default='data/04_output/all_strategies_ranked_latest.csv',
                       help='Path to current analysis CSV (default: latest)')
    parser.add_argument('--previous', default=None,
                       help='Path to previous analysis CSV')
    parser.add_argument('--auto', action='store_true',
                       help='Auto-detect most recent previous run')
    parser.add_argument('--output-dir', default='data/04_output',
                       help='Output directory (default: data/04_output)')
    parser.add_argument('--top-n', type=int, default=20,
                       help='Number of top strategies to display (default: 20)')
    
    args = parser.parse_args()
    
    # Load current analysis
    if not os.path.exists(args.current):
        print(f"❌ Current analysis file not found: {args.current}")
        print("Run analyze_all_strategies_performance.py first!")
        return
    
    df_current = load_analysis(args.current)
    print(f"✅ Loaded current analysis: {args.current}")
    print(f"   Date: {os.path.getmtime(args.current)}")
    
    # Load previous analysis
    if args.auto:
        try:
            previous_path = find_most_recent_analysis(args.output_dir)
            print(f"✅ Auto-detected previous analysis: {previous_path}")
        except ValueError as e:
            print(f"❌ {e}")
            return
    elif args.previous:
        previous_path = args.previous
        if not os.path.exists(previous_path):
            print(f"❌ Previous analysis file not found: {previous_path}")
            return
    else:
        print("❌ Must specify --previous or --auto")
        return
    
    df_previous = load_analysis(previous_path)
    print(f"   Date: {os.path.getmtime(previous_path)}")
    
    # Compare
    df_comparison = compare_analyses(df_current, df_previous)
    
    # Print report
    print_comparison_report(df_comparison, n=args.top_n)
    
    print(f"\n{'='*100}")
    print("COMPARISON COMPLETE")
    print("="*100)


if __name__ == '__main__':
    main()
