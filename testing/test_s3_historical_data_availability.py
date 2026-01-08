"""
Test Historical Data Availability for NBA Seasons

Purpose:
--------
Check if props data and game results exist in S3 for specified NBA seasons.
The Odds API doesn't have historical props before ~mid-2022, so this helps
verify what data is actually available before attempting backtests.

Context:
--------
- The user tried fetching 2021-22 season data and got 422 errors (props not available)
- This script checks S3 to see what dates actually have data
- Helps identify which seasons can be backtested

Usage:
------
python testing/test_historical_data_availability.py --seasons 2021-22 2022-23 2023-24

Author: Thomas Myles
Date: 2026-01-08
"""

import argparse
import boto3
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from config_loader import get_config

# Load config
CONFIG = get_config()

# S3 Configuration
S3_BUCKET_PROPS = CONFIG['aws']['buckets']['odds_api']
S3_BUCKET_NBA = CONFIG['aws']['buckets']['nba_api']


def get_season_date_range(season):
    """
    Get start and end dates for an NBA season.
    
    Args:
        season: Season string like '2021-22'
    
    Returns:
        tuple: (start_date, end_date) as datetime.date objects
    """
    start_year = int(season.split('-')[0])
    end_year = int('20' + season.split('-')[1])
    
    # NBA season typically runs October to April
    start_date = datetime(start_year, 10, 1).date()
    end_date = datetime(end_year, 6, 30).date()
    
    return start_date, end_date


def check_s3_files(bucket, prefix, start_date, end_date):
    """
    Check which dates have files in S3 for a given date range.
    
    Args:
        bucket: S3 bucket name
        prefix: S3 prefix (folder path)
        start_date: Start date to check
        end_date: End date to check
    
    Returns:
        dict: {
            'total_dates': int,
            'dates_with_data': list,
            'dates_missing': list,
            'coverage_pct': float
        }
    """
    s3_client = boto3.client('s3')
    
    # Get all objects in the prefix
    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        
        if 'Contents' not in response:
            return {
                'total_dates': 0,
                'dates_with_data': [],
                'dates_missing': [],
                'coverage_pct': 0.0
            }
        
        # Extract dates from filenames (formats: YYYY-MM-DD.csv OR prefix_YYYY-MM-DD.csv)
        existing_dates = set()
        for obj in response['Contents']:
            filename = Path(obj['Key']).name
            if filename.endswith('.csv'):
                # Try to extract date from filename
                # Handle formats like: 2021-10-20.csv OR nba_game_lines_2021-10-20.csv
                date_str = filename.replace('.csv', '')
                
                # If filename has underscore, try to get date from the end
                if '_' in date_str:
                    # Split and get last part which should be the date
                    parts = date_str.split('_')
                    date_str = parts[-1]
                
                try:
                    date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
                    existing_dates.add(date_obj)
                except ValueError:
                    continue
        
        # Generate all dates in range
        all_dates = []
        current_date = start_date
        while current_date <= end_date:
            # Only check dates up to today
            if current_date <= datetime.now().date():
                all_dates.append(current_date)
            current_date += timedelta(days=1)
        
        # Find missing dates
        dates_with_data = sorted([d for d in all_dates if d in existing_dates])
        dates_missing = sorted([d for d in all_dates if d not in existing_dates])
        
        coverage_pct = (len(dates_with_data) / len(all_dates) * 100) if all_dates else 0
        
        return {
            'total_dates': len(all_dates),
            'dates_with_data': dates_with_data,
            'dates_missing': dates_missing,
            'coverage_pct': coverage_pct
        }
        
    except Exception as e:
        print(f"❌ Error checking S3: {e}")
        return {
            'total_dates': 0,
            'dates_with_data': [],
            'dates_missing': [],
            'coverage_pct': 0.0
        }


def analyze_season(season):
    """
    Analyze data availability for a specific season.
    
    Args:
        season: Season string like '2021-22'
    """
    print(f"\n{'='*80}")
    print(f"🏀 ANALYZING SEASON: {season}")
    print(f"{'='*80}\n")
    
    # Get date range
    start_date, end_date = get_season_date_range(season)
    print(f"📅 Date range: {start_date} to {end_date}")
    
    # Check game lines (spreads, totals, moneylines)
    game_lines_prefix = f"nba/historical_game_lines/{season}/"
    print(f"\n🎲 Checking game lines (spreads/totals/ML): s3://{S3_BUCKET_PROPS}/{game_lines_prefix}")
    game_lines_data = check_s3_files(S3_BUCKET_PROPS, game_lines_prefix, start_date, end_date)
    
    print(f"   Total days in range: {game_lines_data['total_dates']}")
    print(f"   Days with data: {len(game_lines_data['dates_with_data'])}")
    print(f"   Days missing: {len(game_lines_data['dates_missing'])}")
    print(f"   Coverage: {game_lines_data['coverage_pct']:.1f}%")
    
    if game_lines_data['dates_with_data']:
        print(f"   First data: {game_lines_data['dates_with_data'][0]}")
        print(f"   Last data: {game_lines_data['dates_with_data'][-1]}")
    else:
        print(f"   ❌ No game lines data found!")
    
    # Check player props data
    props_prefix = f"nba/historical_player_props/{season}/"
    print(f"\n📊 Checking player props data: s3://{S3_BUCKET_PROPS}/{props_prefix}")
    props_data = check_s3_files(S3_BUCKET_PROPS, props_prefix, start_date, end_date)
    
    print(f"   Total days in range: {props_data['total_dates']}")
    print(f"   Days with data: {len(props_data['dates_with_data'])}")
    print(f"   Days missing: {len(props_data['dates_missing'])}")
    print(f"   Coverage: {props_data['coverage_pct']:.1f}%")
    
    if props_data['dates_with_data']:
        print(f"   First data: {props_data['dates_with_data'][0]}")
        print(f"   Last data: {props_data['dates_with_data'][-1]}")
    else:
        print(f"   ❌ No player props data found!")
    
    # Check game results
    games_prefix = f"player_game_logs/{season}/"
    print(f"\n🏆 Checking game results: s3://{S3_BUCKET_NBA}/{games_prefix}")
    games_data = check_s3_files(S3_BUCKET_NBA, games_prefix, start_date, end_date)
    
    print(f"   Total days in range: {games_data['total_dates']}")
    print(f"   Days with data: {len(games_data['dates_with_data'])}")
    print(f"   Days missing: {len(games_data['dates_missing'])}")
    print(f"   Coverage: {games_data['coverage_pct']:.1f}%")
    
    if games_data['dates_with_data']:
        print(f"   First data: {games_data['dates_with_data'][0]}")
        print(f"   Last data: {games_data['dates_with_data'][-1]}")
    
    # Summary
    print(f"\n{'─'*80}")
    print("📋 SUMMARY:")
    print(f"{'─'*80}")
    
    # Props backtest viable if we have props and game results
    props_backtest_viable = props_data['coverage_pct'] > 50 and games_data['coverage_pct'] > 50
    
    # Game lines backtest viable if we have game lines
    game_lines_backtest_viable = game_lines_data['coverage_pct'] > 50
    
    if props_backtest_viable:
        print(f"✅ Season {season} is VIABLE for PLAYER PROPS backtesting")
        print(f"   Player props coverage: {props_data['coverage_pct']:.1f}%")
        print(f"   Games coverage: {games_data['coverage_pct']:.1f}%")
    else:
        print(f"❌ Season {season} is NOT VIABLE for PLAYER PROPS backtesting")
        print(f"   Player props coverage: {props_data['coverage_pct']:.1f}% (need >50%)")
        print(f"   Games coverage: {games_data['coverage_pct']:.1f}% (need >50%)")
    
    if game_lines_backtest_viable:
        print(f"\n✅ Season {season} HAS GAME LINES DATA (spreads/totals/ML)")
        print(f"   Game lines coverage: {game_lines_data['coverage_pct']:.1f}%")
    else:
        print(f"\n❌ Season {season} has NO/INSUFFICIENT GAME LINES DATA")
        print(f"   Game lines coverage: {game_lines_data['coverage_pct']:.1f}%")
    
    # Show sample of missing dates if substantial
    if len(props_data['dates_missing']) > 10:
        print(f"\n⚠️  Props data missing for {len(props_data['dates_missing'])} dates")
    
    if len(games_data['dates_missing']) > 10:
        print(f"\n⚠️  Game results missing for {len(games_data['dates_missing'])} dates")
    
    return {
        'season': season,
        'props_viable': props_backtest_viable,
        'game_lines_viable': game_lines_backtest_viable,
        'game_lines_coverage': game_lines_data['coverage_pct'],
        'props_coverage': props_data['coverage_pct'],
        'games_coverage': games_data['coverage_pct'],
        'game_lines_dates': len(game_lines_data['dates_with_data']),
        'props_dates': len(props_data['dates_with_data']),
        'games_dates': len(games_data['dates_with_data'])
    }


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Check data availability for NBA seasons'
    )
    parser.add_argument(
        '--seasons',
        nargs='+',
        default=['2021-22', '2022-23', '2023-24', '2024-25', '2025-26'],
        help='Seasons to check (e.g., 2021-22 2022-23)'
    )
    args = parser.parse_args()
    
    print("="*80)
    print("🔍 NBA HISTORICAL DATA AVAILABILITY CHECKER")
    print("="*80)
    print(f"\nChecking {len(args.seasons)} season(s)...")
    
    results = []
    for season in args.seasons:
        result = analyze_season(season)
        results.append(result)
    
    # Final summary
    print(f"\n{'='*80}")
    print("📊 FINAL SUMMARY")
    print(f"{'='*80}\n")
    
    props_viable_seasons = [r for r in results if r['props_viable']]
    game_lines_viable_seasons = [r for r in results if r['game_lines_viable']]
    
    print(f"🎯 PLAYER PROPS BACKTESTING:")
    print(f"{'─'*80}")
    if props_viable_seasons:
        print(f"✅ Viable seasons ({len(props_viable_seasons)}):")
        for r in props_viable_seasons:
            print(f"   - {r['season']}: Props {r['props_coverage']:.1f}%, "
                  f"Games {r['games_coverage']:.1f}% "
                  f"({r['props_dates']} days)")
    else:
        print(f"❌ No seasons with sufficient player props data")
        print(f"   The Odds API historical props may not go back far enough.")
    
    print(f"\n🎲 GAME LINES BACKTESTING (spreads/totals/ML):")
    print(f"{'─'*80}")
    if game_lines_viable_seasons:
        print(f"✅ Viable seasons ({len(game_lines_viable_seasons)}):")
        for r in game_lines_viable_seasons:
            print(f"   - {r['season']}: Game lines {r['game_lines_coverage']:.1f}% "
                  f"({r['game_lines_dates']} days)")
    else:
        print(f"❌ No seasons with sufficient game lines data")
    
    print(f"\n{'='*80}")
    print("💡 KEY FINDINGS:")
    print(f"{'='*80}")
    
    # Check if we have game lines but not props for older seasons
    older_seasons_with_lines = [r for r in results 
                                if r['game_lines_viable'] and not r['props_viable']]
    
    if older_seasons_with_lines:
        print(f"\n🔍 IMPORTANT: These seasons have GAME LINES but NO PLAYER PROPS:")
        for r in older_seasons_with_lines:
            print(f"   - {r['season']}: Game lines {r['game_lines_coverage']:.1f}%, "
                  f"Player props {r['props_coverage']:.1f}%")
        print(f"\n   This confirms: The Odds API has historical GAME-LEVEL betting data")
        print(f"   going back further than PLAYER-LEVEL props data.")
    
    if props_viable_seasons:
        print(f"\n✅ Focus PLAYER PROPS backtesting on:")
        for r in props_viable_seasons:
            print(f"   - {r['season']}")
    
    if game_lines_viable_seasons and not older_seasons_with_lines:
        print(f"\n✅ Focus GAME LINES backtesting on:")
        for r in game_lines_viable_seasons:
            print(f"   - {r['season']}")
    
    print()


if __name__ == "__main__":
    main()

