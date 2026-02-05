"""
Quick Futures Summary Script

Quickly displays mean odds and implied probabilities for championship futures
across bookmakers for a given timestamp and sport(s).

Usage:
    python scripts/quick_futures_summary.py --timestamp 20260204_163703 --sports nba ncaab
    python scripts/quick_futures_summary.py -t 20260204_163703 -s nba
    python scripts/quick_futures_summary.py -t 20260204_163703 -s nba ncaab ncaaf nfl

Output:
    Prints to console: team, mean_implied_prob, mean_odds (sorted by mean_implied_prob desc)
"""

import argparse
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='Quick summary of futures odds by team'
    )
    parser.add_argument(
        '--timestamp', '-t',
        required=True,
        help='Timestamp for the futures file (e.g., 20260204_163703)'
    )
    parser.add_argument(
        '--sports', '-s',
        nargs='+',
        required=True,
        help='Sport(s) to analyze: nba, ncaab, nfl, ncaaf'
    )
    
    args = parser.parse_args()
    
    repo_root = Path(__file__).parent.parent
    timestamp = args.timestamp
    
    print(f"\n{'='*80}")
    print(f"Championship Futures Summary - {timestamp}")
    print(f"{'='*80}\n")
    
    for sport in args.sports:
        sport_lower = sport.lower()
        futures_file = (
            repo_root / 
            'data' / 
            '01_input' / 
            'the-odds-api' / 
            sport_lower / 
            'futures' / 
            f'{sport_lower}_championship_futures_{timestamp}.csv'
        )
        
        if not futures_file.exists():
            print(f"❌ File not found: {futures_file}")
            continue
        
        print(f"\n{sport_lower.upper()} Championship Futures")
        print(f"{'-'*80}")
        
        df = pd.read_csv(futures_file)
        
        # Calculate min, max, median, mean for odds and implied prob by team
        summary = df.groupby('team').agg({
            'implied_prob': ['min', 'max', 'median', 'mean'],
            'odds': ['min', 'max', 'median', 'mean']
        }).reset_index()
        
        # Flatten column names
        summary.columns = [
            'team',
            'implied_min', 'implied_max', 'implied_median', 'implied_mean',
            'odds_min', 'odds_max', 'odds_median', 'odds_mean'
        ]
        
        # Sort by mean implied prob descending
        summary = summary.sort_values('implied_mean', ascending=False)
        
        # Format output
        print(f"\n{'Team':<35} {'Implied %':<45} {'Odds':<50}")
        print(f"{'':35} {'Min     Max     Med     Mean':<45} {'Min      Max      Med      Mean':<50}")
        print(f"{'-'*130}")
        
        for _, row in summary.iterrows():
            team = row['team']
            
            # Implied prob stats
            impl_min = row['implied_min'] * 100
            impl_max = row['implied_max'] * 100
            impl_med = row['implied_median'] * 100
            impl_mean = row['implied_mean'] * 100
            
            # Odds stats
            odds_min = row['odds_min']
            odds_max = row['odds_max']
            odds_med = row['odds_median']
            odds_mean = row['odds_mean']
            
            # Format odds with + for positive
            def fmt_odds(val):
                return f"+{int(val)}" if val > 0 else f"{int(val)}"
            
            print(
                f"{team:<35} "
                f"{impl_min:>5.1f}%  {impl_max:>5.1f}%  {impl_med:>5.1f}%  {impl_mean:>5.1f}%   "
                f"{fmt_odds(odds_min):>7}  {fmt_odds(odds_max):>7}  {fmt_odds(odds_med):>7}  {fmt_odds(odds_mean):>7}"
            )
        
        print(f"\nTotal teams: {len(summary)}")
    
    print(f"\n{'='*80}\n")


if __name__ == '__main__':
    main()
