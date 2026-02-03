"""
Refresh Strategy Statistics - Daily Multi-Season Backtest Update

Lambda function: nba-strategy-stats-refresher

Context:
This script updates strategy JSONs daily with fresh multi-season statistics.
It combines backtest results from 2023-24, 2024-25, and 2025-26 (through yesterday)
to provide robust, up-to-date win rates and ROI figures.

Steps:
1. Clone GitHub repo to get latest code
2. Re-run 2025-26 backtest (includes yesterday's games)
3. Load 2023-24 and 2024-25 backtests from S3 (static)
4. Combine all 3 seasons
5. Calculate aggregate strategy stats
6. Generate updated strategy JSON files
7. Upload to S3

Lambda Deployment:
This script is self-contained and can be copied directly into the Lambda editor.
It clones the GitHub repo at runtime to access backtest scripts and dependencies.

Required Lambda Environment Variables:
- GITHUB_TOKEN: Personal access token for cloning private repo

Usage (CLI):
    python scripts/lambda_function_refresh_strategy_statistics.py --season 2025-26

Usage (Lambda):
    Event payload:
    {
        "season": "2025-26",           # optional, default: "2025-26"
        "strategy": "both",            # optional, default: "both" (choices: "2d", "3d", "both")
        "skip_backtest": false         # optional, default: false
    }

Author: Myles Thomas
Date: 2026-01-30
Updated: 2026-02-01 (self-contained with git clone)
"""

import sys
import os
import json
import subprocess
import shutil
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import List, Dict, Optional
from io import StringIO

# Only import stdlib and boto3 (available in Lambda by default)
import boto3

# Try to import pandas (may need to be in Lambda layer)
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("⚠️  pandas not available")


# =============================================================================
# CONFIGURATION
# =============================================================================

S3_BUCKET = 'nba-betting-mt'
BACKTEST_PREFIX = 'data/04_output/backtests'
STRATEGIES_PREFIX = 'data/03_intermediate'

# Multi-season analysis (hardcoded for stability)
BACKTEST_SEASONS = ['2023-24', '2024-25', '2025-26']

# Minimum plays to include strategy
MIN_PLAYS_THRESHOLD = 50

# GitHub repo
GITHUB_REPO = 'https://github.com/MylesThomas/betting.git'


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_yesterday_et() -> str:
    """Get yesterday's date in ET timezone."""
    et_tz = ZoneInfo('America/New_York')
    now_et = datetime.now(et_tz)
    yesterday = (now_et - timedelta(days=1)).strftime('%Y-%m-%d')
    return yesterday


def run_cmd(cmd: List[str], cwd: Optional[str] = None) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=300
    )
    return result


def send_sns(subject: str, message: str) -> None:
    """
    Send SNS notification.
    
    Args:
        subject: Email subject
        message: Email body
    """
    try:
        sns_client = boto3.client('sns')
        topic_arn = os.environ.get('SNS_TOPIC_ARN')
        
        if not topic_arn:
            print("   ⚠️  SNS_TOPIC_ARN not set - skipping notification")
            return
        
        sns_client.publish(
            TopicArn=topic_arn,
            Subject=subject,
            Message=message
        )
        print(f"   ✅ SNS notification sent")
    except Exception as e:
        print(f"   ⚠️  Failed to send SNS: {e}")


def clone_repo(token: Optional[str] = None) -> str:
    """
    Clone GitHub repo to /tmp directory.
    
    Args:
        token: GitHub personal access token (optional for public repos)
    
    Returns:
        Path to cloned repo
    """
    print("📥 Cloning GitHub repo...")
    
    repo_dir = '/tmp/betting_repo'
    
    # Clean up existing repo if it exists
    if os.path.exists(repo_dir):
        shutil.rmtree(repo_dir)
    
    # Build clone URL with token if provided
    if token:
        repo_url = GITHUB_REPO.replace('https://', f'https://{token}@')
    else:
        repo_url = GITHUB_REPO
    
    # Clone repo
    result = run_cmd(['git', 'clone', repo_url, repo_dir])
    
    if result.returncode != 0:
        raise Exception(f"Failed to clone repo: {result.stderr}")
    
    print(f"   ✅ Repo cloned to {repo_dir}")
    
    # Add to Python path
    sys.path.insert(0, repo_dir)
    sys.path.insert(0, str(Path(repo_dir) / 'src'))
    
    return repo_dir


# =============================================================================
# BACKTEST FUNCTIONS
# =============================================================================

def run_backtest_for_season(season: str, strategy_type: str, repo_dir: str) -> bool:
    """
    Run backtest for a specific season and strategy type.
    
    Args:
        season: NBA season (e.g., '2025-26')
        strategy_type: '2d' or '3d'
        repo_dir: Path to cloned repo
    
    Returns:
        bool: True if successful
    """
    print(f"   Running {strategy_type.upper()} backtest for {season}...")
    
    backtest_script = Path(repo_dir) / 'backtesting' / '20260108_nba_points_props_strategy_backtest.py'
    
    if not backtest_script.exists():
        print(f"   ❌ Backtest script not found: {backtest_script}")
        return False
    
    # Use sys.executable to ensure subprocess uses same Python as Lambda runtime
    cmd = [
        sys.executable,  # Use Lambda's Python (has access to layers)
        str(backtest_script),
        '--seasons', season,
        '--strategy', strategy_type,
        '--granularity', 'detailed',
        '--min-roi', '-1000.0'  # Set very low to capture ALL strategies (filter later by min plays)
    ]
    
    try:
        # Create environment with Lambda layer paths
        env = os.environ.copy()
        
        # Add Lambda layer paths to PYTHONPATH
        layer_paths = [
            '/opt/python',  # Lambda layer for Python packages
            '/opt/python/lib/python3.12/site-packages',
            repo_dir,
            str(Path(repo_dir) / 'src')
        ]
        current_pythonpath = env.get('PYTHONPATH', '')
        env['PYTHONPATH'] = ':'.join(layer_paths + ([current_pythonpath] if current_pythonpath else []))
        print(f"   ✅ PYTHONPATH: {env['PYTHONPATH']}")
        
        # Run with modified environment
        result = subprocess.run(
            cmd,
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=300,
            env=env
        )
        
        if result.returncode == 0:
            print(f"   ✅ {strategy_type.upper()} backtest complete")
            return True
        else:
            print(f"   ❌ {strategy_type.upper()} backtest failed:")
            print(f"      {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"   ❌ {strategy_type.upper()} backtest timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"   ❌ {strategy_type.upper()} backtest error: {e}")
        return False


def load_backtest_plays(s3_client, bucket: str, strategy_type: str, season: str) -> 'pd.DataFrame':
    """
    Load backtest plays CSV from S3.
    
    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        strategy_type: '2d' or '3d'
        season: Season string (e.g., '2023-24')
    
    Returns:
        DataFrame of plays
    """
    if not PANDAS_AVAILABLE:
        print(f"   ⚠️  pandas not available - cannot load backtest plays")
        return None
    
    s3_key = f'{BACKTEST_PREFIX}/{strategy_type}/{season}/plays.csv'
    
    try:
        response = s3_client.get_object(Bucket=bucket, Key=s3_key)
        df = pd.read_csv(StringIO(response['Body'].read().decode('utf-8')))
        df['season'] = season
        print(f"   Loaded {len(df)} plays from {season} {strategy_type.upper()}")
        return df
    except Exception as e:
        print(f"   ⚠️  Could not load {season} {strategy_type.upper()}: {e}")
        return None


def calculate_aggregate_strategy_stats(
    df_all: 'pd.DataFrame',
    strategy_type: str,
    seasons: List[str],
    min_plays: int = MIN_PLAYS_THRESHOLD
) -> List[Dict]:
    """
    Calculate aggregate statistics for each strategy across all seasons.
    
    Args:
        df_all: Combined DataFrame with all plays from all seasons
        strategy_type: '2d' or '3d'
        seasons: List of seasons included
        min_plays: Minimum total plays to include strategy
    
    Returns:
        List of strategy dicts ready for JSON export
    """
    if not PANDAS_AVAILABLE or df_all is None:
        print("   ⚠️  pandas not available or no data - cannot calculate stats")
        return []
    
    print(f"\n   Calculating aggregate stats for {strategy_type.upper()} strategies...")
    
    # Group by strategy parameters
    if strategy_type == '2d':
        group_cols = ['line_tier', 'spread_bin', 'bet_side']
    else:  # 3d
        group_cols = ['line_tier', 'spread_bin', 'bet_side', 'scorer_type']
    
    strategies = []
    
    for group_key, group_df in df_all.groupby(group_cols):
        # Calculate stats
        total_plays = len(group_df)
        
        if total_plays < min_plays:
            continue
        
        total_wins = (group_df['result'] == 'WIN').sum()
        total_losses = (group_df['result'] == 'LOSS').sum()
        total_ties = (group_df['result'] == 'PUSH').sum()
        total_profit = group_df['profit'].sum()
        
        if (total_wins + total_losses) == 0:
            continue
        
        hit_rate = (total_wins / (total_wins + total_losses) * 100)
        
        # ROI calculation (assuming $100 bets)
        total_wagered = total_plays * 100
        roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0
        
        # Edge vs baseline (assume 50% baseline)
        edge = hit_rate - 50.0
        
        # Build strategy dict
        if strategy_type == '2d':
            line_tier, spread_bin, bet_side = group_key
            strat = {
                'line_tier': line_tier,
                'spread_bin': spread_bin,
                'bet_side': bet_side,
                'hit_rate': round(hit_rate, 1),
                'roi': round(roi, 1),
                'edge': round(edge, 1),
                'games': total_plays,
                'wins': int(total_wins),
                'losses': int(total_losses),
                'ties': int(total_ties)
            }
        else:  # 3d
            line_tier, spread_bin, bet_side, scorer_type = group_key
            strat = {
                'line_tier': line_tier,
                'spread_bin': spread_bin,
                'bet_side': bet_side,
                'scorer_type': scorer_type,
                'hit_rate': round(hit_rate, 1),
                'roi': round(roi, 1),
                'edge': round(edge, 1),
                'games': total_plays,
                'wins': int(total_wins),
                'losses': int(total_losses),
                'ties': int(total_ties)
            }
        
        strategies.append(strat)
    
    print(f"   ✅ Found {len(strategies)} strategies with >= {min_plays} plays")
    return strategies


def log_strategy_results(
    strategies: List[Dict],
    strategy_type: str,
    seasons: List[str],
    df_all: 'pd.DataFrame'
) -> None:
    """
    Log detailed backtest results for each strategy with per-season breakdown.
    
    Args:
        strategies: List of strategy dicts with performance metrics
        strategy_type: '2d' or '3d'
        seasons: List of seasons included in backtest
        df_all: Full dataframe with all plays for per-season breakdown
    """
    if not PANDAS_AVAILABLE or df_all is None:
        print("   ⚠️  pandas not available - cannot show detailed results")
        return
    
    print(f"\n{'='*80}")
    print(f"📊 {strategy_type.upper()} STRATEGY BACKTEST RESULTS ({', '.join(seasons)})")
    print(f"{'='*80}\n")
    
    # Sort strategies by win rate descending
    sorted_strategies = sorted(strategies, key=lambda x: x['hit_rate'], reverse=True)
    
    # Group columns for filtering
    if strategy_type == '2d':
        group_cols = ['line_tier', 'spread_bin', 'bet_side']
    else:  # 3d
        group_cols = ['line_tier', 'spread_bin', 'bet_side', 'scorer_type']
    
    for i, strat in enumerate(sorted_strategies, 1):
        # Build strategy description
        if strategy_type == '2d':
            desc = f"{strat['line_tier']} | {strat['spread_bin']} | {strat['bet_side']}"
        else:  # 3d
            desc = f"{strat['line_tier']} | {strat['spread_bin']} | {strat['bet_side']} | {strat['scorer_type']}"
        
        # Format aggregate metrics
        hit_rate = strat['hit_rate']
        roi = strat['roi']
        edge = strat['edge']
        total_games = strat['games']
        total_wins = strat['wins']
        total_losses = strat['losses']
        total_ties = strat['ties']
        
        # Determine emoji based on win rate
        if hit_rate >= 60:
            emoji = '🔥'
        elif hit_rate >= 55:
            emoji = '✅'
        elif hit_rate >= 50:
            emoji = '➖'
        else:
            emoji = '❌'
        
        print(f"{emoji} #{i:2d}. {desc}")
        print(f"        AGGREGATE: {total_wins}W-{total_losses}L-{total_ties}T | Hit Rate: {hit_rate:5.1f}% | ROI: {roi:6.1f}% | Edge: {edge:+5.1f}%")
        
        # Filter dataframe for this strategy
        if strategy_type == '2d':
            mask = (
                (df_all['line_tier'] == strat['line_tier']) &
                (df_all['spread_bin'] == strat['spread_bin']) &
                (df_all['bet_side'] == strat['bet_side'])
            )
        else:  # 3d
            mask = (
                (df_all['line_tier'] == strat['line_tier']) &
                (df_all['spread_bin'] == strat['spread_bin']) &
                (df_all['bet_side'] == strat['bet_side']) &
                (df_all['scorer_type'] == strat['scorer_type'])
            )
        
        strat_df = df_all[mask]
        
        # Calculate per-season stats
        for season in seasons:
            season_df = strat_df[strat_df['season'] == season]
            
            if len(season_df) == 0:
                continue
            
            season_wins = (season_df['result'] == 'WIN').sum()
            season_losses = (season_df['result'] == 'LOSS').sum()
            season_ties = (season_df['result'] == 'PUSH').sum()
            season_plays = len(season_df)
            
            if (season_wins + season_losses) > 0:
                season_hit_rate = (season_wins / (season_wins + season_losses) * 100)
            else:
                season_hit_rate = 0.0
            
            season_profit = season_df['profit'].sum()
            season_wagered = season_plays * 100
            season_roi = (season_profit / season_wagered * 100) if season_wagered > 0 else 0
            
            print(f"          {season}: {season_wins}W-{season_losses}L-{season_ties}T | Hit Rate: {season_hit_rate:5.1f}% | ROI: {season_roi:6.1f}%")
        
        print()
    
    # Summary statistics
    total_plays = sum(s['games'] for s in strategies)
    total_wins = sum(s['wins'] for s in strategies)
    total_losses = sum(s['losses'] for s in strategies)
    total_ties = sum(s['ties'] for s in strategies)
    avg_roi = sum(s['roi'] for s in strategies) / len(strategies) if strategies else 0
    avg_hit_rate = sum(s['hit_rate'] for s in strategies) / len(strategies) if strategies else 0
    
    print(f"{'='*80}")
    print(f"SUMMARY:")
    print(f"  Total Strategies: {len(strategies)}")
    print(f"  Total Plays: {total_plays}")
    print(f"  Overall Record: {total_wins}W-{total_losses}L-{total_ties}T ({avg_hit_rate:.1f}% avg hit rate)")
    print(f"  Average ROI: {avg_roi:.1f}%")
    print(f"  Profitable Strategies: {sum(1 for s in strategies if s['roi'] > 0)}/{len(strategies)}")
    print(f"{'='*80}\n")


def generate_strategy_json(
    strategies: List[Dict],
    output_path: str,
    metadata: Dict
) -> None:
    """
    Generate strategy JSON file.
    
    Args:
        strategies: List of strategy dicts
        output_path: Where to save JSON
        metadata: Metadata to include in JSON
    """
    data = {
        'generated_at': metadata['generated_at'],
        'data_through': metadata['data_through'],
        'seasons_included': metadata['seasons_included'],
        'total_plays': metadata['total_plays'],
        'strategies': strategies
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"   💾 Saved {len(strategies)} strategies to {output_path}")


# =============================================================================
# MAIN REFRESH FUNCTION
# =============================================================================

def refresh_strategy_statistics(
    season: str = '2025-26',
    strategy_types: List[str] = ['2d', '3d'],
    skip_backtest: bool = False,
    github_token: Optional[str] = None
) -> Dict:
    """
    Main function to refresh strategy statistics.
    
    Args:
        season: Current NBA season
        strategy_types: List of strategy types to update
        skip_backtest: If True, skip regenerating current season backtest
        github_token: GitHub personal access token for cloning repo
    
    Returns:
        Dict with results summary
    """
    yesterday = get_yesterday_et()
    et_tz = ZoneInfo('America/New_York')
    now_et = datetime.now(et_tz)
    
    print("="*80)
    print("🔄 REFRESHING STRATEGY STATISTICS")
    print("="*80)
    print(f"Current Season: {season}")
    print(f"Strategy Types: {', '.join(strategy_types)}")
    print(f"Backtest Seasons: {', '.join(BACKTEST_SEASONS)}")
    print(f"Data Through: {yesterday}")
    print(f"Timestamp: {now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print("="*80)
    
    # Step 0: Clone repo if we need to run backtests
    repo_dir = None
    if not skip_backtest:
        try:
            repo_dir = clone_repo(github_token)
        except Exception as e:
            error_msg = f"Failed to clone repo: {e}"
            print(f"   ❌ {error_msg}")
            return {
                '2d': {'success': False, 'error': error_msg},
                '3d': {'success': False, 'error': error_msg}
            }
    
    results = {}
    s3_client = boto3.client('s3')
    
    for strategy_type in strategy_types:
        print(f"\n{'='*80}")
        print(f"Processing {strategy_type.upper()} Strategy")
        print(f"{'='*80}\n")
        
        # Step 1: Re-run current season backtest (if not skipped)
        if not skip_backtest and repo_dir:
            print(f"Step 1: Updating {season} backtest...")
            success = run_backtest_for_season(season, strategy_type, repo_dir)
            if not success:
                error_msg = f"Backtest failed for {strategy_type.upper()} - cannot proceed"
                print(f"   ❌ {error_msg}")
                results[strategy_type] = {'success': False, 'error': error_msg}
                continue  # Skip to next strategy type
        else:
            print(f"Step 1: Skipping backtest regeneration (using existing data)")
        
        # Step 2: Load all seasons from S3
        print(f"\nStep 2: Loading multi-season backtest data...")
        
        dfs = []
        for s in BACKTEST_SEASONS:
            df = load_backtest_plays(s3_client, S3_BUCKET, strategy_type, s)
            if df is not None and not df.empty:
                dfs.append(df)
        
        if not dfs:
            print(f"   ❌ No backtest data found for any season!")
            results[strategy_type] = {'success': False, 'error': 'No data'}
            continue
        
        if not PANDAS_AVAILABLE:
            print(f"   ❌ pandas not available - cannot proceed!")
            results[strategy_type] = {'success': False, 'error': 'pandas not available'}
            continue
        
        df_all = pd.concat(dfs, ignore_index=True)
        print(f"   ✅ Loaded {len(df_all)} total plays across {len(dfs)} seasons")
        
        # Step 3: Calculate aggregate stats
        print(f"\nStep 3: Calculating aggregate strategy statistics...")
        strategies = calculate_aggregate_strategy_stats(
            df_all,
            strategy_type,
            BACKTEST_SEASONS
        )
        
        if not strategies:
            print(f"   ❌ No strategies met minimum threshold!")
            results[strategy_type] = {'success': False, 'error': 'No strategies qualified'}
            continue
        
        # Step 4: Generate updated JSON
        print(f"\nStep 4: Generating updated strategy file...")
        
        if strategy_type == '2d':
            filename = f'points_by_role_gamespread_strategies_{season}.json'
        else:
            filename = f'points_by_role_gamespread_6feet_strategies_{season}_rim40.json'
        
        local_path = f'/tmp/{filename}'
        
        generate_strategy_json(
            strategies=strategies,
            output_path=local_path,
            metadata={
                'generated_at': now_et.isoformat(),
                'data_through': yesterday,
                'seasons_included': BACKTEST_SEASONS,
                'total_plays': len(df_all)
            }
        )
        
        # Log detailed results for this strategy type
        log_strategy_results(strategies, strategy_type, BACKTEST_SEASONS, df_all)
        
        # Step 5: Upload to S3
        print(f"\nStep 5: Uploading to S3...")
        s3_key = f'{STRATEGIES_PREFIX}/{filename}'
        
        try:
            s3_client.upload_file(local_path, S3_BUCKET, s3_key)
            print(f"   ✅ Uploaded to s3://{S3_BUCKET}/{s3_key}")
            
            results[strategy_type] = {
                'success': True,
                'strategies_count': len(strategies),
                'total_plays': len(df_all),
                's3_path': f's3://{S3_BUCKET}/{s3_key}'
            }
            
        except Exception as e:
            print(f"   ❌ Upload failed: {e}")
            results[strategy_type] = {'success': False, 'error': str(e)}
    
    # Summary
    print(f"\n{'='*80}")
    print("✅ REFRESH COMPLETE")
    print(f"{'='*80}")
    
    summary_lines = []
    all_success = all(r.get('success', False) for r in results.values())
    
    for strategy_type, result in results.items():
        if result['success']:
            line = f"{strategy_type.upper()}: ✅ {result['strategies_count']} strategies, {result['total_plays']} plays"
            print(line)
            summary_lines.append(line)
        else:
            line = f"{strategy_type.upper()}: ❌ {result.get('error', 'Failed')}"
            print(line)
            summary_lines.append(line)
    
    print(f"{'='*80}\n")
    
    # Send SNS notification
    if all_success:
        subject = f"✅ Strategy Statistics Refresh Complete - {season}"
        message = f"""Strategy Statistics Refresh Completed Successfully

Season: {season}
Data Through: {yesterday}
Backtest Seasons: {', '.join(BACKTEST_SEASONS)}
Timestamp: {now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}

Results:
{chr(10).join(summary_lines)}

Total Strategies: {sum(r.get('strategies_count', 0) for r in results.values() if r.get('success'))}
Total Plays: {sum(r.get('total_plays', 0) for r in results.values() if r.get('success'))}

All strategy JSON files have been updated in S3.
"""
    else:
        subject = f"❌ Strategy Statistics Refresh Failed - {season}"
        message = f"""Strategy Statistics Refresh Failed

Season: {season}
Timestamp: {now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}

Errors:
{chr(10).join(summary_lines)}

Please check CloudWatch logs for details.
"""
    
    send_sns(subject, message)
    
    return results


# =============================================================================
# LAMBDA HANDLER
# =============================================================================

def lambda_handler(event, context):
    """
    AWS Lambda handler function.
    
    Args:
        event: Lambda event (can contain 'season', 'strategy', 'skip_backtest')
        context: Lambda context
    
    Returns:
        Dict with execution results
    """
    # Extract parameters from event (with defaults)
    season = event.get('season', '2025-26')
    strategy = event.get('strategy', 'both')
    skip_backtest = event.get('skip_backtest', False)  # Default: run backtest
    
    # Get GitHub token from environment
    github_token = os.environ.get('GITHUB_TOKEN')
    
    # Determine strategy types
    if strategy == 'both':
        strategy_types = ['2d', '3d']
    else:
        strategy_types = [strategy]
    
    # Run refresh
    results = refresh_strategy_statistics(
        season=season,
        strategy_types=strategy_types,
        skip_backtest=skip_backtest,
        github_token=github_token
    )
    
    # Format response
    all_success = all(r.get('success', False) for r in results.values())
    
    return {
        'statusCode': 200 if all_success else 500,
        'body': json.dumps({
            'success': all_success,
            'results': results
        })
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI entry point for local execution."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Refresh strategy statistics with multi-season backtest data'
    )
    parser.add_argument(
        '--season',
        default='2025-26',
        help='Current NBA season (default: 2025-26)'
    )
    parser.add_argument(
        '--strategy',
        choices=['2d', '3d', 'both'],
        default='both',
        help='Which strategy type to update (default: both)'
    )
    parser.add_argument(
        '--skip-backtest',
        action='store_true',
        help='Skip regenerating current season backtest (use existing data)'
    )
    parser.add_argument(
        '--github-token',
        type=str,
        default=None,
        help='GitHub personal access token (for private repos)'
    )
    
    args = parser.parse_args()
    
    # Determine strategy types
    if args.strategy == 'both':
        strategy_types = ['2d', '3d']
    else:
        strategy_types = [args.strategy]
    
    # Run refresh
    results = refresh_strategy_statistics(
        season=args.season,
        strategy_types=strategy_types,
        skip_backtest=args.skip_backtest,
        github_token=args.github_token
    )
    
    # Exit with error code if any failed
    all_success = all(r.get('success', False) for r in results.values())
    sys.exit(0 if all_success else 1)


if __name__ == '__main__':
    main()
