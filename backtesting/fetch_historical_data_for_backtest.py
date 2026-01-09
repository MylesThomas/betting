"""
Fetch Historical NBA Data for Backtesting
==========================================

This script fetches all required historical data for backtesting NBA player
points prop strategies across multiple seasons.

The Odds API has historical data going back to 2021-22 season.

What this script does for each season:
1. Fetches player props from The Odds API
2. Fetches game results from NBA API
3. Fetches game lines (spreads) from The Odds API
4. Fetches shot charts from NBA API (for 3D strategy)
5. Joins all data into unified analysis datasets (2D + 3D)

Usage:
    # Fetch all seasons (includes 2025-26)
    python3 backtesting/fetch_historical_data_for_backtest.py
    
    # Fetch specific seasons only
    python3 backtesting/fetch_historical_data_for_backtest.py --seasons 2024-25 2023-24
    
    # Skip 2025-26 if you already have it (common use case)
    python3 backtesting/fetch_historical_data_for_backtest.py --seasons 2024-25 2023-24 2022-23 2021-22
    
    # Skip confirmation prompt (use with caution!)
    python3 backtesting/fetch_historical_data_for_backtest.py --yes

Author: Myles Thomas
Date: 2026-01-08
"""

import argparse
import subprocess
import sys
import boto3
from pathlib import Path
from io import StringIO

# Find project root
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
while not (project_root / '.gitignore').exists():
    if project_root == project_root.parent:
        raise FileNotFoundError("Could not find project root (.gitignore)")
    project_root = project_root.parent

# AWS S3 configuration
S3_BUCKET_PROPS = 'the-odds-api-mt'
S3_BUCKET_NBA = 'nba-api-mt'
S3_BUCKET_OUTPUT = 'nba-betting-mt'
AWS_REGION = 'us-east-2'

# Initialize S3 client
s3_client = boto3.client('s3', region_name=AWS_REGION)

# Emoji map
EMOJI = {
    'success': '✅',
    'error': '❌',
    'warning': '⚠️',
    'info': 'ℹ️',
    'basketball': '🏀',
    'chart': '📊',
    'download': '⬇️',
    'upload': '⬆️',
    'hourglass': '⏳',
}


def run_command(cmd, description):
    """
    Run a command and print output in real-time.
    
    Args:
        cmd: Command list (e.g., ['python3', 'script.py', '--arg', 'value'])
        description: Human-readable description of what's happening
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n{EMOJI['hourglass']} {description}...")
    print(f"   Command: {' '.join(cmd)}")
    print()
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=project_root
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        
        if process.returncode == 0:
            print(f"\n{EMOJI['success']} {description} complete")
            return True
        else:
            print(f"\n{EMOJI['error']} {description} failed (exit code: {process.returncode})")
            return False
            
    except Exception as e:
        print(f"\n{EMOJI['error']} Error running command: {e}")
        return False


def verify_s3_files(bucket, prefix, description, expected=None):
    """
    Verify files exist in S3 and count them.
    
    Args:
        bucket: S3 bucket name
        prefix: S3 prefix (folder path)
        description: What we're checking
        expected: Optional expected count to show as "actual/expected"
    
    Returns:
        int: Number of files found
    """
    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        if 'Contents' not in response:
            count = 0
        else:
            count = len([obj for obj in response['Contents'] if obj['Key'].endswith('.csv')])
        
        if expected is not None:
            print(f"   {EMOJI['success']} {description}: {count}/{expected} files")
        else:
            print(f"   {EMOJI['success']} {description}: {count} files")
        return count
        
    except Exception as e:
        print(f"   {EMOJI['warning']} Could not verify {description}: {e}")
        return 0


def get_expected_game_dates(season):
    """
    Get expected number of game dates for a season from the NBA calendar in S3.
    
    Args:
        season: NBA season (e.g., "2024-25")
    
    Returns:
        int: Number of expected game dates, or None if can't determine
    """
    try:
        import pandas as pd
        season_underscore = season.replace('-', '_')
        key = f"data/01_input/nba_calendar/{season_underscore}/daily_summary_{season_underscore}.csv"
        
        obj = s3_client.get_object(Bucket=S3_BUCKET_OUTPUT, Key=key)
        df = pd.read_csv(obj['Body'])
        
        return len(df)
    except Exception as e:
        print(f"   {EMOJI['warning']} Could not determine expected game dates: {e}")
        return None


def verify_csv_rows(bucket, key, description):
    """
    Verify CSV exists and count rows.
    
    Args:
        bucket: S3 bucket name
        key: S3 key (file path)
        description: What we're checking
    
    Returns:
        int: Number of rows (excluding header)
    """
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read().decode('utf-8')
        rows = len(content.strip().split('\n')) - 1  # Exclude header
        
        # Get file size
        size_bytes = response['ContentLength']
        if size_bytes < 1024:
            size_str = f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            size_str = f"{size_bytes / 1024:.1f} KB"
        else:
            size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
        
        print(f"   {EMOJI['success']} {description}: {rows:,} rows ({size_str})")
        return rows
        
    except s3_client.exceptions.NoSuchKey:
        print(f"   {EMOJI['error']} {description}: File not found")
        return 0
    except Exception as e:
        print(f"   {EMOJI['warning']} Could not verify {description}: {e}")
        return 0


def check_season_complete(season):
    """
    Check if all final output files exist for a season.
    
    Args:
        season: Season string (e.g., '2024-25')
    
    Returns:
        tuple: (bool: all_exist, list: missing_files)
    """
    required_files = [
        f"data/03_intermediate/player_props_with_actuals_{season}.csv",  # 2D
        f"data/03_intermediate/player_props_with_actuals_{season}_rim40.csv",  # 3D
    ]
    
    missing = []
    for key in required_files:
        try:
            s3_client.head_object(Bucket=S3_BUCKET_OUTPUT, Key=key)
        except:
            missing.append(key)
    
    return len(missing) == 0, missing


def fetch_season_data(season):
    """
    Fetch all required data for a single season.
    
    Args:
        season: Season string (e.g., '2024-25')
    
    Returns:
        dict: Results of each step
    """
    print(f"\n{'='*80}")
    print(f"{EMOJI['basketball']} Fetching Historical Data for Season: {season}")
    print(f"{'='*80}")
    
    # Check if season already complete
    all_exist, missing = check_season_complete(season)
    if all_exist:
        print(f"\n{EMOJI['success']} Season {season} already complete!")
        print(f"   Found both final output files:")
        print(f"   - player_props_with_actuals_{season}.csv (2D)")
        print(f"   - player_props_with_actuals_{season}_rim40.csv (3D)")
        print(f"\n{EMOJI['info']} Skipping entire season (all data already exists)")
        
        return {
            'season': season,
            'skipped': True,
            'reason': 'All final output files already exist'
        }
    else:
        print(f"\n{EMOJI['info']} Missing {len(missing)} final output file(s), will fetch data:")
        for f in missing:
            print(f"   - {f.split('/')[-1]}")
    
    results = {
        'season': season,
        'skipped': False,
        'steps': {}
    }
    
    # Step 0b: Build NBA Calendar for this season
    print(f"\n{'─'*80}")
    print(f"Step 0b: Building NBA Calendar for {season}")
    print(f"{'─'*80}")
    
    cmd = [
        'python3', 'scripts/nba_calendar_builder.py',
        '--season', season,
        '--s3'
    ]
    success = run_command(cmd, f"Build NBA calendar for {season}")
    results['steps']['calendar'] = success
    
    if success:
        print(f"\n{EMOJI['chart']} Verifying calendar upload...")
        season_underscore = season.replace('-', '_')
        verify_s3_files(S3_BUCKET_OUTPUT, f"data/01_input/nba_calendar/{season_underscore}/", f"NBA calendar ({season})")
    
    # Get expected game dates from calendar for verification
    expected_dates = get_expected_game_dates(season)
    
    # Step 1: Fetch closing betting lines (game spreads)
    print(f"\n{'─'*80}")
    print(f"Step 1: Fetching Closing Betting Lines (Game Spreads)")
    print(f"{'─'*80}")
    
    cmd = [
        'python3', 'scripts/fetch_historical_nba_season_lines.py',
        '--season', season,
        '--prod-run'
    ]
    success = run_command(cmd, f"Fetch game lines for {season}")
    results['steps']['game_lines'] = success
    
    if success:
        print(f"\n{EMOJI['chart']} Verifying uploads...")
        verify_s3_files(S3_BUCKET_PROPS, f"nba/historical_game_lines/{season}/", "Game lines", expected=expected_dates)
    
    # Step 2: Fetch shot charts
    print(f"\n{'─'*80}")
    print(f"Step 2: Fetching Shot Charts (for 3D Strategy)")
    print(f"{'─'*80}")
    
    cmd = [
        'python3', 'scripts/fetch_all_nba_shot_charts.py',
        '--auto',
        '--seasons', season
    ]
    success = run_command(cmd, f"Fetch shot charts for {season}")
    results['steps']['shot_charts'] = success
    
    if success:
        print(f"\n{EMOJI['chart']} Verifying uploads...")
        verify_s3_files(S3_BUCKET_NBA, f"player_shot_charts/{season}/", "Shot charts")
    
    # Step 3: Fetch player props + game results
    print(f"\n{'─'*80}")
    print(f"Step 3: Fetching Player Props + Game Results")
    print(f"{'─'*80}")
    print(f"Note: Using fetch_nba_player_props.py (has S3 support + game results)")
    print(f"      fetch_historical_nba_prop_markets.py is the older local-only version")
    
    cmd = [
        'python3', 'scripts/fetch_nba_player_props.py',
        '--mode', '2',
        '--fetch-games',
        '--s3',
        '--season', season
    ]
    success = run_command(cmd, f"Fetch props and game results for {season}")
    results['steps']['props_and_results'] = success
    
    if success:
        print(f"\n{EMOJI['chart']} Verifying uploads...")
        verify_s3_files(S3_BUCKET_PROPS, f"nba/historical_player_props/{season}/", "Player props", expected=expected_dates)
        verify_s3_files(S3_BUCKET_NBA, f"player_game_logs/{season}/", "Game results", expected=expected_dates)
    
    # Step 4: Join data for 2D strategy
    print(f"\n{'─'*80}")
    print(f"Step 4: Joining Data for 2D Strategy (tier × spread)")
    print(f"{'─'*80}")
    
    cmd = [
        'python3', 'scripts/join_nba_points_props_actuals_charts_gamelines.py',
        '--season', season,
        '--s3'
    ]
    success = run_command(cmd, f"Join data for 2D strategy ({season})")
    results['steps']['join_2d'] = success
    
    if success:
        print(f"\n{EMOJI['chart']} Verifying 2D dataset...")
        verify_csv_rows(
            S3_BUCKET_OUTPUT,
            f"data/03_intermediate/player_props_with_actuals_{season}.csv",
            f"2D dataset ({season})"
        )
    
    # Step 5: Join data for 3D strategy
    print(f"\n{'─'*80}")
    print(f"Step 5: Joining Data for 3D Strategy (tier × spread × scorer_type)")
    print(f"{'─'*80}")
    
    cmd = [
        'python3', 'scripts/join_nba_points_props_actuals_charts_gamelines.py',
        '--season', season,
        '--s3',
        '--rim-scorer-pct', '40'
    ]
    success = run_command(cmd, f"Join data for 3D strategy ({season})")
    results['steps']['join_3d'] = success
    
    if success:
        print(f"\n{EMOJI['chart']} Verifying 3D dataset...")
        verify_csv_rows(
            S3_BUCKET_OUTPUT,
            f"data/03_intermediate/player_props_with_actuals_{season}_rim40.csv",
            f"3D dataset ({season})"
        )
    
    # Summary
    all_success = all(results['steps'].values())
    if all_success:
        print(f"\n{EMOJI['success']} Season {season} complete - all steps successful!")
    else:
        failed_steps = [k for k, v in results['steps'].items() if not v]
        print(f"\n{EMOJI['warning']} Season {season} complete with failures: {', '.join(failed_steps)}")
    
    return results


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Fetch historical NBA data for backtesting strategies'
    )
    parser.add_argument(
        '--seasons',
        nargs='+',
        default=['2025-26', '2024-25', '2023-24', '2022-23', '2021-22'],
        help='Seasons to fetch (default: 2025-26 2024-25 2023-24 2022-23 2021-22)'
    )
    parser.add_argument(
        '--yes', '-y',
        action='store_true',
        help='Skip confirmation prompt'
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"{EMOJI['basketball']} Historical Data Fetching for Backtest")
    print(f"{'='*80}\n")
    
    print(f"Seasons to fetch: {', '.join(args.seasons)}")
    
    # Estimate API costs
    num_seasons = len(args.seasons)
    estimated_credits = num_seasons * 22000  # ~22k credits per season
    estimated_cost = estimated_credits / 500000 * 100  # $100 for 500k credits
    
    print(f"\nEstimated API costs:")
    print(f"  - Credits: ~{estimated_credits:,} ({num_seasons} seasons × 22,000 credits)")
    print(f"  - Cost: ~${estimated_cost:.2f} (based on $100/500k plan)")
    
    # Confirmation
    if not args.yes:
        print(f"\n{EMOJI['warning']} This will fetch data from The Odds API (will consume API credits)")
        response = input("\nContinue? (y/n): ")
        if response.lower() != 'y':
            print(f"{EMOJI['error']} Aborted")
            sys.exit(0)
    
    # Fetch data for each season
    all_results = []
    for season in args.seasons:
        result = fetch_season_data(season)
        all_results.append(result)
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"{EMOJI['chart']} FINAL SUMMARY")
    print(f"{'='*80}\n")
    
    for result in all_results:
        season = result['season']
        
        # Handle skipped seasons
        if result.get('skipped'):
            print(f"{EMOJI['success']} {season}: Skipped (already complete)")
            continue
        
        steps = result['steps']
        all_success = all(steps.values())
        status = EMOJI['success'] if all_success else EMOJI['warning']
        
        print(f"{status} {season}:")
        for step_name, success in steps.items():
            step_status = EMOJI['success'] if success else EMOJI['error']
            print(f"     {step_status} {step_name}")
    
    # Check if all successful (including skipped = success)
    all_successful = all(
        result.get('skipped') or all(result['steps'].values())
        for result in all_results
    )
    
    if all_successful:
        print(f"\n{EMOJI['success']} All seasons fetched successfully!")
        print(f"\nYou can now run the backtest:")
        print(f"  python3 backtesting/20260108_nba_points_props_strategy_backtest.py --seasons {' '.join(args.seasons)}")
    else:
        print(f"\n{EMOJI['warning']} Some steps failed - review errors above")
        print(f"You may need to re-run failed steps manually")


if __name__ == '__main__':
    main()

