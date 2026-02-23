"""
Compare Kelly Fractions for Strategy Performance

Quickly compare how different Kelly fractions affect bankroll growth for top strategies.

Usage:
    python scripts/compare_kelly_fractions.py --strategy "20_25_2_6_dog_rim_under" --bankroll 10000

Author: Myles Thomas  
Date: 2026-01-25
"""

import pandas as pd
import logging
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)


def load_latest_results():
    """Load the latest strategy analysis results."""
    csv_path = Path.home() / 'Downloads' / 'tmp' / 'all_strategies_ranked_latest.csv'
    return pd.read_csv(csv_path)


def main():
    parser = argparse.ArgumentParser(description='Compare Kelly fractions for strategy performance')
    parser.add_argument('--strategy', required=True, help='Strategy name to analyze')
    parser.add_argument('--bankroll', type=float, default=10000, help='Starting bankroll (default: 10000)')
    parser.add_argument('--output-dir', default=str(Path.home() / 'Downloads' / 'tmp'),
                       help='Output directory (default: ~/Downloads/tmp)')
    
    args = parser.parse_args()
    
    # Load results
    df = load_latest_results()
    
    # Find strategy
    strategy_row = df[df['strategy_name'] == args.strategy]
    
    if len(strategy_row) == 0:
        print(f"❌ Strategy '{args.strategy}' not found!")
        print("\nAvailable strategies:")
        for idx, row in df.head(10).iterrows():
            print(f"  - {row['strategy_name']}")
        return
    
    strategy_row = strategy_row.iloc[0]
    
    print("="*80)
    print(f"KELLY FRACTION COMPARISON: {args.strategy}")
    print("="*80)
    print(f"Win Rate: {strategy_row['win_rate']:.1f}%")
    print(f"ROI: {strategy_row['roi']:+.1f}%")
    print(f"Total Plays: {int(strategy_row['total_plays'])}")
    print(f"Starting Bankroll: ${args.bankroll:,.0f}")
    print()
    
    # Calculate optimal Kelly
    win_rate = strategy_row['win_rate'] / 100
    b = 100 / 110
    kelly_full = (win_rate * b - (1 - win_rate)) / b
    
    kelly_fractions = [0.25, 0.5, 1.0]
    
    print("Kelly Calculations:")
    print(f"  Full Kelly: {kelly_full:.3f} ({kelly_full*100:.1f}% of bankroll per bet)")
    for frac in kelly_fractions:
        print(f"  {frac}x Kelly: {kelly_full * frac:.3f} ({kelly_full*frac*100:.1f}% of bankroll per bet)")
    
    print(f"\nRun full analysis with each Kelly fraction:")
    print("="*80)
    for frac in kelly_fractions:
        print(f"\npython scripts/analyze_all_strategies_performance.py \\")
        print(f"    --viz --kelly \\")
        print(f"    --bankroll {args.bankroll:.0f} \\")
        print(f"    --kelly-fraction {frac} \\")
        print(f"    --no-viz-display")


if __name__ == '__main__':
    main()
