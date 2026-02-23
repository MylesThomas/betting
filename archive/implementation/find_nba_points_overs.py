#!/usr/bin/env python3
"""
Find +EV NBA Points OVER opportunities based on vig structure and line position.

================================================================================
STRATEGY OVERVIEW
================================================================================

Based on analysis of 34,239 props over 42 days (2025-26 season), we found
systematic mispricing in NBA points markets. Books consistently underprice
OVERS when:

1. VIG STRUCTURE is "under_heavy" (books charge more vig on UNDER)
2. LINE POSITION is "min" (lowest line offered across books)

When books charge more vig on UNDER, they're signaling they think UNDER is
more likely. But historically, they've been WRONG - OVERS hit 68% of the time
in these spots.

================================================================================
STRATEGY DETAILS
================================================================================

STRATEGY 1: Points OVER on Under-Heavy Vig Lines
    Filter:
        - Market: player_points
        - Vig Structure: under_heavy (under_vig > over_vig + 1%)
        - Bet Side: OVER
    
    Historical Performance (N=2,682):
        - Hit Rate: 68.6%
        - Implied: 34.6%
        - Edge: +34.0%
        - ROI: ~+90% (at avg +180 odds)

STRATEGY 2: Points OVER on Min Lines
    Filter:
        - Market: player_points
        - Line Position: min (lowest line offered)
        - Bet Side: OVER
    
    Historical Performance (N=2,945):
        - Hit Rate: 67.8%
        - Implied: 36.6%
        - Edge: +31.2%
        - ROI: ~+90% (at avg +180 odds)

STRATEGY 3: PRA OVER on Under-Heavy Vig Lines
    Filter:
        - Market: player_points_rebounds_assists
        - Vig Structure: under_heavy
        - Bet Side: OVER
    
    Historical Performance (N=1,228):
        - Hit Rate: 66.3%
        - Edge: +26.4%

STRATEGY 4: Rebounds OVER on Under-Heavy or Min Lines
    Filter:
        - Market: player_rebounds
        - Vig Structure: under_heavy OR Line Position: min
        - Bet Side: OVER
    
    Historical Performance:
        - Under-heavy: N=1,846, Hit 58.3%, Edge +19.1%
        - Min line: N=2,117, Hit 62.2%, Edge +18.1%

================================================================================
HOW IT WORKS
================================================================================

1. Fetch live props from multiple books (via The Odds API)
2. For each player/market, calculate:
   - min_line: lowest line across all books
   - max_line: highest line across all books
   - over_vig: vig attributed to over side
   - under_vig: vig attributed to under side
3. Flag opportunities where:
   - under_vig > over_vig + 0.01 (under_heavy)
   - OR line = min_line AND multiple lines available
4. Output qualifying OVER bets

================================================================================
VIG STRUCTURE LOGIC
================================================================================

Vig = implied_over + implied_under - 1

For -105/-125 (over/under):
    implied_over = 105/205 = 51.2%
    implied_under = 125/225 = 55.6%
    total_vig = 51.2% + 55.6% - 100% = 6.8%
    
Vig attribution (proportional):
    fair_over = 51.2% / 106.8% = 47.9%
    fair_under = 55.6% / 106.8% = 52.1%
    over_vig = 51.2% - 47.9% = 3.3%
    under_vig = 55.6% - 52.1% = 3.5%

If under_vig > over_vig + 1%: "under_heavy" → bet OVER

================================================================================
MARKETS TO TARGET (by edge)
================================================================================

| Rank | Market                  | Strategy     | Edge    | N     |
|------|-------------------------|--------------|---------|-------|
| 1    | points                  | under_heavy  | +34.0%  | 2,682 |
| 2    | points                  | min line     | +31.2%  | 2,945 |
| 3    | points_rebounds_assists | under_heavy  | +26.4%  | 1,228 |
| 4    | rebounds                | under_heavy  | +19.1%  | 1,846 |
| 5    | rebounds                | min line     | +18.1%  | 2,117 |
| 6    | assists                 | min line     | +14.3%  | 1,401 |

================================================================================
MARKETS TO AVOID
================================================================================

- blocks: Negative edge across all strategies
- steals: Mixed/small sample, negative edge on most combos
- max lines: Negative edge (-3% to -4%)
- only_line props: No line shopping possible, negative edge

================================================================================
USAGE
================================================================================

    # Fetch live props first
    python implementation/fetch_live_data.py --live
    
    # Find opportunities (reads from data/live/)
    python implementation/find_nba_points_overs.py
    
    # Use custom data directory
    python implementation/find_nba_points_overs.py --data-dir data/custom/
    
    # Filter to specific markets
    python implementation/find_nba_points_overs.py --markets points,rebounds
    
    # Adjust vig threshold
    python implementation/find_nba_points_overs.py --vig-threshold 0.015

================================================================================
OUTPUT
================================================================================

Console: Detailed betting opportunities with:
    - Player name and matchup
    - Line and odds
    - Vig structure analysis
    - Strategy match and historical edge

CSV: data/04_output/nba_points_overs_YYYYMMDD.csv

================================================================================
IMPORTANT NOTES
================================================================================

⚠️  Risk Disclaimer:
    - This is based on historical backtesting (42 days, 2025-26 season)
    - Past performance does not guarantee future results
    - Always bet responsibly within your limits
    - Verify all data before placing bets
    - Track your results to validate the strategy
    
⚠️  Data Quality:
    - Requires multiple books to calculate min/max lines
    - Single-book props ("only_line") don't qualify
    - Stale odds can create false signals

================================================================================
BACKTEST REFERENCE
================================================================================

Analysis script: analysis/nba_props_vs_actuals.py --mode game-player-market-line --verbose
Output data: data/03_intermediate/nba_props_analysis/top_strategies_game_player_market_line.csv

Author: Myles Thomas
Date: 2025-12-05
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from odds_utils import (
    odds_to_implied_probability, 
    calculate_vig_attribution,
    calculate_bet_amount,
    calculate_profit
)
from player_name_utils import normalize_player_name

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / 'data' / '01_input' / 'the-odds-api' / 'nba' / 'live'
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / 'data' / '04_output'

# Strategy parameters
VIG_ASYMMETRY_THRESHOLD = 0.01  # 1% difference to be "under_heavy"

# Markets to analyze (in priority order)
TARGET_MARKETS = [
    'player_points',
    'player_points_rebounds_assists', 
    'player_rebounds',
    'player_assists',
    'player_threes',
]

# Historical edge by market × strategy (from backtest)
HISTORICAL_EDGE = {
    ('player_points', 'under_heavy'): 0.340,
    ('player_points', 'min'): 0.312,
    ('player_points_rebounds_assists', 'under_heavy'): 0.264,
    ('player_rebounds', 'under_heavy'): 0.191,
    ('player_rebounds', 'min'): 0.181,
    ('player_assists', 'min'): 0.143,
    ('player_threes', 'min'): 0.103,
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_todays_props(data_dir: Path) -> pd.DataFrame:
    """Load today's props from live data directory."""
    
    # Find the most recent props file
    props_files = list(data_dir.glob('props_*.csv'))
    
    if not props_files:
        raise FileNotFoundError(f"No props files found in {data_dir}")
    
    # Sort by modification time, get most recent
    latest_file = max(props_files, key=lambda p: p.stat().st_mtime)
    
    print(f"📂 Loading props from: {latest_file.name}")
    df = pd.read_csv(latest_file)
    print(f"   ✅ Loaded {len(df):,} prop lines")
    
    return df


# =============================================================================
# VIG & LINE ANALYSIS
# =============================================================================

def calculate_prop_metrics(props_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate vig structure and line position metrics for each prop.
    
    Groups by player/game/market to find:
    - min_line, max_line, line_spread
    - vig attribution (over_vig, under_vig)
    - line position category (min, max, middle, only_line)
    """
    print("\n📊 Calculating prop metrics...")
    
    # Filter to target markets
    props_df = props_df[props_df['market'].isin(TARGET_MARKETS)].copy()
    
    if len(props_df) == 0:
        print("   ⚠️ No props found for target markets")
        return pd.DataFrame()
    
    # Calculate implied probabilities
    props_df['implied_over'] = props_df['over_odds'].apply(odds_to_implied_probability)
    props_df['implied_under'] = props_df['under_odds'].apply(odds_to_implied_probability)
    
    # Calculate vig attribution for each line
    vig_attrs = props_df.apply(
        lambda row: calculate_vig_attribution(row['implied_over'], row['implied_under']),
        axis=1
    )
    props_df['total_vig'] = vig_attrs.apply(lambda x: x['total_vig'])
    props_df['over_vig'] = vig_attrs.apply(lambda x: x['over_vig'])
    props_df['under_vig'] = vig_attrs.apply(lambda x: x['under_vig'])
    props_df['fair_over'] = vig_attrs.apply(lambda x: x['fair_over'])
    props_df['fair_under'] = vig_attrs.apply(lambda x: x['fair_under'])
    
    # Group by player/game/market to get line-level stats
    group_cols = ['player', 'game', 'market']
    
    # Calculate min/max lines per prop
    line_stats = props_df.groupby(group_cols).agg(
        min_line=('line', 'min'),
        max_line=('line', 'max'),
        n_books=('bookmaker', 'nunique'),
        n_lines=('line', 'nunique'),
    ).reset_index()
    
    line_stats['line_spread'] = line_stats['max_line'] - line_stats['min_line']
    
    # Merge back to props
    props_df = props_df.merge(
        line_stats[group_cols + ['min_line', 'max_line', 'n_books', 'n_lines', 'line_spread']],
        on=group_cols,
        how='left'
    )
    
    # Calculate line position
    def get_line_position(row):
        if row['line_spread'] == 0:
            return 'only_line'
        elif row['line'] == row['min_line']:
            return 'min'
        elif row['line'] == row['max_line']:
            return 'max'
        else:
            return 'middle'
    
    props_df['line_position'] = props_df.apply(get_line_position, axis=1)
    
    # Calculate vig structure
    def get_vig_structure(row):
        if pd.isna(row['over_vig']) or pd.isna(row['under_vig']):
            return 'unknown'
        diff = row['under_vig'] - row['over_vig']
        if diff > VIG_ASYMMETRY_THRESHOLD:
            return 'under_heavy'
        elif diff < -VIG_ASYMMETRY_THRESHOLD:
            return 'over_heavy'
        else:
            return 'symmetric'
    
    props_df['vig_structure'] = props_df.apply(get_vig_structure, axis=1)
    
    print(f"   ✅ Calculated metrics for {len(props_df):,} lines")
    print(f"   📊 Markets: {props_df['market'].unique().tolist()}")
    print(f"   📊 Vig structure: {props_df['vig_structure'].value_counts().to_dict()}")
    print(f"   📊 Line position: {props_df['line_position'].value_counts().to_dict()}")
    
    return props_df


# =============================================================================
# OPPORTUNITY FINDING
# =============================================================================

def find_opportunities(props_df: pd.DataFrame, vig_threshold: float = VIG_ASYMMETRY_THRESHOLD) -> pd.DataFrame:
    """
    Find qualifying OVER opportunities based on vig structure and line position.
    
    Qualifies if:
    - vig_structure == 'under_heavy' (any market)
    - OR line_position == 'min' AND n_lines > 1 (points, rebounds, assists)
    """
    print("\n🎯 Finding opportunities...")
    
    opportunities = []
    
    for _, row in props_df.iterrows():
        market = row['market']
        
        # Strategy 1: Under-heavy vig (any target market)
        if row['vig_structure'] == 'under_heavy':
            strategy = 'under_heavy'
            edge = HISTORICAL_EDGE.get((market, strategy), 0.10)
            opportunities.append({
                **row.to_dict(),
                'strategy': strategy,
                'historical_edge': edge,
                'bet_side': 'OVER',
                'bet_odds': row['over_odds'],
                'bet_implied': row['implied_over'],
            })
        
        # Strategy 2: Min line (when multiple lines available)
        if row['line_position'] == 'min' and row['n_lines'] > 1:
            # Skip if already captured by under_heavy (avoid duplicates)
            if row['vig_structure'] == 'under_heavy':
                continue
            strategy = 'min'
            edge = HISTORICAL_EDGE.get((market, strategy), 0.05)
            opportunities.append({
                **row.to_dict(),
                'strategy': strategy,
                'historical_edge': edge,
                'bet_side': 'OVER',
                'bet_odds': row['over_odds'],
                'bet_implied': row['implied_over'],
            })
    
    if not opportunities:
        print("   ⚠️ No qualifying opportunities found")
        return pd.DataFrame()
    
    opps_df = pd.DataFrame(opportunities)
    
    # Sort by: 1) historical edge (best first), 2) market (points first), 3) strategy
    opps_df['market_priority'] = opps_df['market'].map({
        'player_points': 1,
        'player_points_rebounds_assists': 2,
        'player_rebounds': 3,
        'player_assists': 4,
        'player_threes': 5,
    }).fillna(99)
    
    opps_df = opps_df.sort_values(
        ['historical_edge', 'market_priority', 'strategy'],
        ascending=[False, True, True]
    )
    
    opps_df = opps_df.drop(columns=['market_priority'])
    
    print(f"   ✅ Found {len(opps_df):,} opportunities")
    
    # Summary by market × strategy
    summary = opps_df.groupby(['market', 'strategy']).agg(
        count=('player', 'count'),
        edge=('historical_edge', 'first')
    ).reset_index()
    print("\n   📊 Summary by Market × Strategy:")
    for _, row in summary.sort_values('edge', ascending=False).iterrows():
        market_short = row['market'].replace('player_', '')
        print(f"      {market_short:30} {row['strategy']:12} N={row['count']:4} Edge={row['edge']:.1%}")
    
    return opps_df


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def print_opportunities(opps_df: pd.DataFrame):
    """Print formatted opportunities to console."""
    
    if len(opps_df) == 0:
        print("\n" + "=" * 80)
        print("❌ NO QUALIFYING OPPORTUNITIES FOUND")
        print("=" * 80)
        print("\nThis can happen when:")
        print("  - Market pricing is symmetric (no vig asymmetry)")
        print("  - All props have only one line (no min/max comparison)")
        print("  - No games tonight")
        return
    
    print("\n" + "=" * 80)
    print(f"🎯 {len(opps_df)} QUALIFYING OPPORTUNITIES")
    print("=" * 80)
    
    for i, (_, opp) in enumerate(opps_df.iterrows(), 1):
        market_short = opp['market'].replace('player_', '').upper()
        
        print(f"\n{'─' * 80}")
        print(f"OPPORTUNITY #{i}")
        print(f"{'─' * 80}")
        
        print(f"\n🏀 Player:     {opp['player']}")
        print(f"🎯 Game:       {opp['game']}")
        
        print(f"\n📊 Bet:        {opp['bet_side']} {opp['line']} {market_short}")
        print(f"💰 Odds:       {opp['bet_odds']:+.0f}")
        print(f"📈 Implied:    {opp['bet_implied']:.1%}")
        
        bet_amount = calculate_bet_amount(opp['bet_odds'], 100)
        print(f"💵 Bet Amount: ${bet_amount:.2f} (to win $100)")
        
        print(f"\n📚 Bookmaker:  {opp['bookmaker']}")
        print(f"📊 Strategy:   {opp['strategy'].upper()}")
        print(f"📈 Hist. Edge: {opp['historical_edge']:.1%}")
        
        print(f"\n🔬 Vig Analysis:")
        print(f"   Total Vig:  {opp['total_vig']:.1%}")
        print(f"   Over Vig:   {opp['over_vig']:.1%}")
        print(f"   Under Vig:  {opp['under_vig']:.1%}")
        print(f"   Structure:  {opp['vig_structure']}")
        
        if opp['n_lines'] > 1:
            print(f"\n📏 Line Position:")
            print(f"   This Line:  {opp['line']}")
            print(f"   Min Line:   {opp['min_line']}")
            print(f"   Max Line:   {opp['max_line']}")
            print(f"   Position:   {opp['line_position']}")
    
    print(f"\n{'=' * 80}")
    print(f"✅ {len(opps_df)} opportunities found")
    print(f"{'=' * 80}")


def save_opportunities(opps_df: pd.DataFrame, output_dir: Path):
    """Save opportunities to CSV."""
    
    if len(opps_df) == 0:
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    today = datetime.now().strftime('%Y%m%d')
    output_file = output_dir / f'nba_points_overs_{today}.csv'
    
    # Select and order columns for output
    output_cols = [
        'player', 'game', 'market', 'line', 
        'bet_side', 'bet_odds', 'bet_implied',
        'strategy', 'historical_edge',
        'bookmaker', 'over_odds', 'under_odds',
        'vig_structure', 'total_vig', 'over_vig', 'under_vig',
        'line_position', 'min_line', 'max_line', 'n_books', 'n_lines',
    ]
    
    # Filter to columns that exist
    output_cols = [c for c in output_cols if c in opps_df.columns]
    
    opps_df[output_cols].to_csv(output_file, index=False)
    print(f"\n💾 Saved to: {output_file}")


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Find +EV NBA Points OVER opportunities',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python implementation/find_nba_points_overs.py
    python implementation/find_nba_points_overs.py --data-dir data/custom/
    python implementation/find_nba_points_overs.py --markets points,rebounds
    python implementation/find_nba_points_overs.py --vig-threshold 0.015
        """
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help=f'Directory containing props data (default: {DEFAULT_DATA_DIR})'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f'Directory to save output (default: {DEFAULT_OUTPUT_DIR})'
    )
    
    parser.add_argument(
        '--markets',
        type=str,
        default=None,
        help='Comma-separated list of markets to analyze (default: all target markets)'
    )
    
    parser.add_argument(
        '--vig-threshold',
        type=float,
        default=VIG_ASYMMETRY_THRESHOLD,
        help=f'Vig asymmetry threshold for under_heavy (default: {VIG_ASYMMETRY_THRESHOLD})'
    )
    
    parser.add_argument(
        '--top',
        type=int,
        default=None,
        help='Only show top N opportunities (default: all)'
    )
    
    parser.add_argument(
        '--focus',
        type=str,
        choices=['points', 'all'],
        default='all',
        help='Focus on specific markets: "points" for highest edge, "all" for all markets'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    print("=" * 80)
    print("🏀 NBA POINTS OVERS - OPPORTUNITY FINDER")
    print("=" * 80)
    print(f"Strategy: Bet OVER on under_heavy vig + min lines")
    print(f"Based on: 42 days backtest, +31-34% historical edge")
    print("=" * 80)
    
    # Update target markets if specified
    global TARGET_MARKETS
    if args.focus == 'points':
        TARGET_MARKETS = ['player_points']
        print(f"\n🎯 Focus mode: POINTS only (highest edge: +34%)")
    elif args.markets:
        TARGET_MARKETS = [f'player_{m}' if not m.startswith('player_') else m 
                         for m in args.markets.split(',')]
        print(f"\n📊 Targeting markets: {TARGET_MARKETS}")
    
    # Load data
    data_dir = Path(args.data_dir)
    props_df = load_todays_props(data_dir)
    
    # Calculate metrics
    props_df = calculate_prop_metrics(props_df)
    
    if len(props_df) == 0:
        print("\n❌ No props data to analyze")
        return
    
    # Find opportunities
    opps_df = find_opportunities(props_df, vig_threshold=args.vig_threshold)
    
    # Limit to top N if specified
    if args.top and len(opps_df) > args.top:
        print(f"\n📋 Limiting to top {args.top} opportunities")
        opps_df = opps_df.head(args.top)
    
    # Output
    print_opportunities(opps_df)
    save_opportunities(opps_df, Path(args.output_dir))
    
    print("\n✅ Done!")


if __name__ == '__main__':
    main()

