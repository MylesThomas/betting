"""
Analyze NCAAB Book Coverage

Analyzes how many bookmakers provide data for each NCAAB game from the fetched
historical game lines data. Shows per-game and aggregate statistics.

Context:
User wants to see how many books have data on NCAAB games being fetched by
fetch_historical_ncaab_season_lines.py. This helps understand data quality
and coverage across different bookmakers.

Usage:
    # Analyze a specific date (from S3)
    python tmp/analyze_ncaab_book_coverage.py --date 2025-11-04
    
    # Analyze a specific date (from local file)
    python tmp/analyze_ncaab_book_coverage.py --date 2025-11-04 --local
    
    # Analyze multiple dates
    python tmp/analyze_ncaab_book_coverage.py --start-date 2025-11-04 --end-date 2025-11-10
    
    # Show detailed per-game breakdown
    python tmp/analyze_ncaab_book_coverage.py --date 2025-11-04 --verbose

Output:
    - Per-game: Shows each matchup with spread/total book counts
    - Summary stats: Min, max, average, median books per game
    - Bookmaker breakdown: Which books provide most consistent coverage

Author: Thomas Myles
Date: 2026-01-15
"""

import sys
import os
import pandas as pd
import boto3
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
import argparse
from io import StringIO

# Find project root
def find_project_root():
    """Find project root by looking for .gitignore file."""
    current = Path.cwd()
    while current != current.parent:
        if (current / '.gitignore').exists():
            return current
        current = current.parent
    return Path.cwd()

PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from config_loader import get_config

# Load config
CONFIG = get_config()
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

S3_BUCKET = 'ncaab-betting-mt'
S3_PREFIX = 'data/01_input/the-odds-api/ncaab/game_lines'
LOCAL_DIR = PROJECT_ROOT / 'data' / '01_input' / 'the-odds-api' / 'ncaab' / 'game_lines'

# =============================================================================
# EMOJI MAP
# =============================================================================

EMOJI = {
    'success': '✅',
    'error': '❌',
    'warning': '⚠️',
    'info': 'ℹ️',
    'chart': '📊',
    'book': '📚',
    'game': '🏀',
    'calendar': '📅',
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def read_game_lines_from_s3(date_str):
    """Read game lines CSV from S3."""
    s3 = boto3.client('s3')
    s3_key = f"{S3_PREFIX}/{date_str}.csv"
    
    try:
        response = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
        csv_content = response['Body'].read().decode('utf-8')
        
        if not csv_content.strip():
            return None
        
        df = pd.read_csv(StringIO(csv_content))
        return df
        
    except s3.exceptions.NoSuchKey:
        return None
    except Exception as e:
        print(f"{EMOJI['error']} Error reading from S3: {e}")
        return None


def read_game_lines_from_local(date_str):
    """Read game lines CSV from local file."""
    file_path = LOCAL_DIR / f"{date_str}.csv"
    
    if not file_path.exists():
        return None
    
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"{EMOJI['error']} Error reading local file: {e}")
        return None


def get_bookmaker_columns(df):
    """Extract bookmaker columns from dataframe."""
    spread_cols = [col for col in df.columns if col.endswith('_spread') and col != 'consensus_spread']
    total_cols = [col for col in df.columns if col.endswith('_total') and col != 'consensus_total']
    
    # Extract bookmaker names (exclude num_books which is a count column, not a bookmaker)
    spread_books = [col.replace('_spread', '') for col in spread_cols if col != 'num_books_spread']
    total_books = [col.replace('_total', '') for col in total_cols if col != 'num_books_total']
    
    return {
        'spread_cols': spread_cols,
        'total_cols': total_cols,
        'spread_books': spread_books,
        'total_books': total_books,
        'all_books': sorted(set(spread_books + total_books))
    }


def analyze_game_coverage(df, verbose=False):
    """Analyze book coverage for games in dataframe."""
    
    if df is None or df.empty:
        print(f"{EMOJI['warning']} No games found")
        return
    
    # Get bookmaker info
    book_info = get_bookmaker_columns(df)
    
    print(f"\n{EMOJI['info']} Found {len(df)} games")
    print(f"{EMOJI['book']} Total unique bookmakers: {len(book_info['all_books'])}")
    
    # Per-game analysis
    if verbose:
        print(f"\n{EMOJI['game']} PER-GAME BREAKDOWN")
        print("=" * 80)
        
        for idx, row in df.iterrows():
            home = row['home_team']
            away = row['away_team']
            num_spread = row.get('num_books_spread', 0)
            num_total = row.get('num_books_total', 0)
            
            print(f"{away} @ {home}")
            print(f"  Spreads: {num_spread} books | Totals: {num_total} books")
            
            # Show which books have data
            if num_spread > 0:
                spread_books = [book for book in book_info['spread_books'] 
                               if pd.notna(row.get(f"{book}_spread"))]
                print(f"  Spread books: {', '.join(spread_books)}")
            
            if num_total > 0:
                total_books = [book for book in book_info['total_books'] 
                              if pd.notna(row.get(f"{book}_total"))]
                print(f"  Total books: {', '.join(total_books)}")
            
            print()
    
    # Summary statistics
    print(f"\n{EMOJI['chart']} COVERAGE STATISTICS")
    print("=" * 80)
    
    if 'num_books_spread' in df.columns and 'num_books_total' in df.columns:
        spread_stats = df['num_books_spread'].describe()
        total_stats = df['num_books_total'].describe()
        
        print("\nSPREAD COVERAGE:")
        print(f"  Min:    {int(spread_stats['min'])} books")
        print(f"  Max:    {int(spread_stats['max'])} books")
        print(f"  Mean:   {spread_stats['mean']:.1f} books")
        print(f"  Median: {spread_stats['50%']:.1f} books")
        
        print("\nTOTAL COVERAGE:")
        print(f"  Min:    {int(total_stats['min'])} books")
        print(f"  Max:    {int(total_stats['max'])} books")
        print(f"  Mean:   {total_stats['mean']:.1f} books")
        print(f"  Median: {total_stats['50%']:.1f} books")
    
    # Bookmaker consistency analysis
    print(f"\n{EMOJI['book']} BOOKMAKER CONSISTENCY")
    print("=" * 80)
    print(f"(Shows % of games each bookmaker provided lines for)")
    print()
    
    total_games = len(df)
    
    # Count games each bookmaker appears in
    book_coverage = {}
    for book in book_info['all_books']:
        spread_col = f"{book}_spread"
        total_col = f"{book}_total"
        
        spread_count = df[spread_col].notna().sum() if spread_col in df.columns else 0
        total_count = df[total_col].notna().sum() if total_col in df.columns else 0
        
        book_coverage[book] = {
            'spreads': spread_count,
            'totals': total_count,
            'spread_pct': (spread_count / total_games) * 100,
            'total_pct': (total_count / total_games) * 100
        }
    
    # Sort by average coverage
    sorted_books = sorted(book_coverage.items(), 
                         key=lambda x: (x[1]['spread_pct'] + x[1]['total_pct']) / 2,
                         reverse=True)
    
    for book, stats in sorted_books:
        avg_pct = (stats['spread_pct'] + stats['total_pct']) / 2
        print(f"{book:20} Spreads: {stats['spread_pct']:5.1f}% | Totals: {stats['total_pct']:5.1f}% | Avg: {avg_pct:5.1f}%")


def analyze_multiple_dates(start_date, end_date, use_local=False):
    """Analyze book coverage across multiple dates."""
    
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    all_data = []
    dates_analyzed = 0
    
    print(f"\n{EMOJI['calendar']} Analyzing dates from {start_date} to {end_date}")
    print("=" * 80)
    
    current_dt = start_dt
    while current_dt <= end_dt:
        date_str = current_dt.strftime('%Y-%m-%d')
        
        if use_local:
            df = read_game_lines_from_local(date_str)
        else:
            df = read_game_lines_from_s3(date_str)
        
        if df is not None and not df.empty:
            all_data.append(df)
            dates_analyzed += 1
            print(f"{EMOJI['success']} {date_str}: {len(df)} games")
        else:
            print(f"{EMOJI['warning']} {date_str}: No data")
        
        current_dt += timedelta(days=1)
    
    if not all_data:
        print(f"\n{EMOJI['error']} No data found for date range")
        return
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    print(f"\n{EMOJI['info']} Combined Analysis")
    print(f"  Dates analyzed: {dates_analyzed}")
    print(f"  Total games: {len(combined_df)}")
    
    # Run combined analysis
    analyze_game_coverage(combined_df, verbose=False)


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description='Analyze bookmaker coverage for NCAAB game lines'
    )
    
    parser.add_argument('--date', type=str,
                       help='Single date to analyze (YYYY-MM-DD)')
    parser.add_argument('--start-date', type=str,
                       help='Start date for range analysis (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str,
                       help='End date for range analysis (YYYY-MM-DD)')
    parser.add_argument('--local', action='store_true',
                       help='Read from local files instead of S3')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed per-game breakdown')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("NCAAB BOOKMAKER COVERAGE ANALYSIS")
    print("=" * 80)
    
    # Validate arguments
    if args.date and (args.start_date or args.end_date):
        print(f"{EMOJI['error']} Cannot specify both --date and --start-date/--end-date")
        return
    
    if (args.start_date and not args.end_date) or (args.end_date and not args.start_date):
        print(f"{EMOJI['error']} Must specify both --start-date and --end-date")
        return
    
    # Default to today if no date specified
    if not args.date and not args.start_date:
        args.date = datetime.now().strftime('%Y-%m-%d')
        print(f"{EMOJI['info']} No date specified, using today: {args.date}")
    
    # Execute analysis
    if args.date:
        print(f"\n{EMOJI['calendar']} Analyzing date: {args.date}")
        print(f"{EMOJI['info']} Data source: {'Local files' if args.local else 'S3'}")
        
        if args.local:
            df = read_game_lines_from_local(args.date)
        else:
            df = read_game_lines_from_s3(args.date)
        
        if df is None:
            print(f"\n{EMOJI['error']} No data found for {args.date}")
            return
        
        analyze_game_coverage(df, verbose=args.verbose)
    
    elif args.start_date and args.end_date:
        analyze_multiple_dates(args.start_date, args.end_date, use_local=args.local)
    
    print(f"\n{EMOJI['success']} Analysis complete!")


if __name__ == '__main__':
    main()

