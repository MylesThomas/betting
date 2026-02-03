"""
Refresh Strategy Statistics - Daily Multi-Season Backtest Update

Lambda function: nba-strategy-stats-refresher

Context:
This script updates strategy JSONs daily with fresh multi-season statistics.
It combines backtest results from 2023-24, 2024-25, and 2025-26 (through yesterday)
to provide robust, up-to-date win rates and ROI figures.

Self-Contained Approach:
Instead of relying on external backtest scripts and historical JSON files,
this implementation generates ALL possible strategy combinations (98 for 2D,
196 for 3D) and tests them directly. This eliminates the circular dependency
where only historically profitable strategies get tested.

Steps:
1. Generate all possible strategy combinations (line tier × spread bin × bet side × scorer type)
2. Load player props data from S3 for current season
3. Match props to strategies and calculate WIN/LOSS/PUSH outcomes
4. Save plays.csv to S3
5. Load plays from all 3 seasons (2023-24, 2024-25, 2025-26)
6. Calculate aggregate statistics across all seasons
7. Filter strategies by minimum plays (50+) not historical ROI
8. Generate updated strategy JSON files
9. Upload to S3
10. Send SNS email notification

Lambda Deployment:
This script is fully self-contained and can be copied directly into the Lambda editor.
No git clone or external dependencies required beyond boto3 and pandas (Lambda layer).

Required Lambda Environment Variables:
- SNS_TOPIC_ARN: SNS topic for email notifications (optional)

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
Updated: 2026-02-03 (fully self-contained, tests all 98/196 combinations)
"""

import sys
import os
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import List, Dict
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


# =============================================================================
# BINNING FUNCTIONS
# =============================================================================

def bin_points_line(line: float) -> str:
    """
    Bin player points line into tiers (detailed granularity).
    
    Args:
        line: Player points line
    
    Returns:
        Line tier string
    """
    if PANDAS_AVAILABLE and pd.isna(line):
        return 'Unknown'
    
    if line < 5:
        return '<5 (Deep Bench)'
    elif line < 10:
        return '5-10 (Bench)'
    elif line < 15:
        return '10-15 (Role Player)'
    elif line < 20:
        return '15-20 (High Role)'
    elif line < 25:
        return '20-25 (Star)'
    elif line < 30:
        return '25-30 (High Star)'
    elif line < 35:
        return '30-35 (Superstar)'
    elif line < 40:
        return '35-40 (Elite)'
    else:
        return '40+ (MVP)'


def bin_team_spread(spread: float) -> str:
    """
    Bin team spread into categories (detailed granularity).
    
    Args:
        spread: Team spread (positive = underdog, negative = favorite)
    
    Returns:
        Spread bin string
    """
    if PANDAS_AVAILABLE and pd.isna(spread):
        return 'Unknown'
    
    if spread < -15:
        return '15+ Fav'
    elif spread < -10:
        return '10-15 Fav'
    elif spread < -6:
        return '6-10 Fav'
    elif spread < -2:
        return '2-6 Fav'
    elif spread <= 2:
        return "Pick'em (-2 to +2)"
    elif spread <= 6:
        return '2-6 Dog'
    elif spread <= 10:
        return '6-10 Dog'
    elif spread <= 15:
        return '10-15 Dog'
    else:
        return '15+ Dog'


def generate_all_strategy_combinations(strategy_type: str) -> List[Dict]:
    """
    Generate all possible strategy combinations to test.
    
    Args:
        strategy_type: '2d' or '3d'
    
    Returns:
        List of all strategy dictionaries
    """
    # All possible line tiers
    line_tiers = [
        '5-10 (Bench)',
        '10-15 (Role Player)',
        '15-20 (High Role)',
        '20-25 (Star)',
        '25-30 (High Star)',
        '30-35 (Superstar)',
        '35-40 (Elite)'
    ]
    
    # All possible spread bins
    spread_bins = [
        "Pick'em (-2 to +2)",
        '2-6 Fav',
        '2-6 Dog',
        '6-10 Fav',
        '6-10 Dog',
        '10-15 Fav',
        '10-15 Dog'
    ]
    
    # Both bet sides
    bet_sides = ['OVER', 'UNDER']
    
    combinations = []
    
    if strategy_type == '2d':
        for line_tier in line_tiers:
            for spread_bin in spread_bins:
                for bet_side in bet_sides:
                    combinations.append({
                        'line_tier': line_tier,
                        'spread_bin': spread_bin,
                        'bet_side': bet_side
                    })
    
    elif strategy_type == '3d':
        scorer_types = ['Rim Attacker (≥40.0%)', 'Perimeter (<40.0%)']
        for line_tier in line_tiers:
            for spread_bin in spread_bins:
                for bet_side in bet_sides:
                    for scorer_type in scorer_types:
                        combinations.append({
                            'line_tier': line_tier,
                            'spread_bin': spread_bin,
                            'bet_side': bet_side,
                            'scorer_type': scorer_type
                        })
    
    return combinations


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_yesterday_et() -> str:
    """Get yesterday's date in ET timezone."""
    et_tz = ZoneInfo('America/New_York')
    now_et = datetime.now(et_tz)
    yesterday = (now_et - timedelta(days=1)).strftime('%Y-%m-%d')
    return yesterday


def load_player_props_from_s3(s3_client, season: str, strategy_type: str) -> 'pd.DataFrame':
    """
    Load player props with actuals CSV from S3.
    
    Args:
        s3_client: Boto3 S3 client
        season: NBA season (e.g., '2024-25')
        strategy_type: '2d' or '3d' (3d requires rim scorer data)
    
    Returns:
        DataFrame with player props and actuals (or None if not found)
    """
    if not PANDAS_AVAILABLE:
        print(f"   ⚠️  pandas not available - cannot load props data")
        return None
    
    if strategy_type == '2d':
        key = f"data/03_intermediate/player_props_with_actuals_{season}.csv"
    elif strategy_type == '3d':
        key = f"data/03_intermediate/player_props_with_actuals_{season}_rim40.csv"
    else:
        raise ValueError(f"Invalid strategy_type: {strategy_type}")
    
    print(f"   Loading props from s3://{S3_BUCKET}/{key}...")
    
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        df = pd.read_csv(StringIO(response['Body'].read().decode('utf-8')))
        print(f"   ✅ Loaded {len(df):,} player-game records")
        return df
    except s3_client.exceptions.NoSuchKey:
        print(f"   ❌ Data file not found: s3://{S3_BUCKET}/{key}")
        return None
    except Exception as e:
        print(f"   ❌ Error loading data: {e}")
        return None


def match_and_calculate_plays(df: 'pd.DataFrame', strategies: List[Dict], strategy_type: str) -> 'pd.DataFrame':
    """
    Match player props to strategies and calculate outcomes.
    
    Args:
        df: Player props with actuals
        strategies: List of all strategy combinations
        strategy_type: '2d' or '3d'
    
    Returns:
        DataFrame with all plays and their outcomes
    """
    if not PANDAS_AVAILABLE or df is None or df.empty:
        print(f"   ⚠️  No data to process")
        return None
    
    print(f"   Matching {len(df):,} records to {len(strategies)} strategies...")
    
    # Bin the data
    df['line_tier'] = df['points_line'].apply(bin_points_line)
    df['spread_bin'] = df['team_spread'].apply(bin_team_spread)
    
    plays = []
    
    for idx, row in df.iterrows():
        line_tier = row['line_tier']
        spread_bin = row['spread_bin']
        scorer_type = row.get('scorer_type', None)
        
        # Try to match against each strategy
        for strat in strategies:
            # Check if this row matches the strategy
            line_match = strat['line_tier'] == line_tier
            spread_match = strat['spread_bin'] == spread_bin
            
            # For 3D strategies, also check scorer_type
            scorer_match = True
            if strategy_type == '3d':
                if pd.isna(scorer_type):
                    continue
                scorer_match = strat['scorer_type'] == scorer_type
            
            if line_match and spread_match and scorer_match:
                # This row matches this strategy - create a play
                actual_points = row.get('PTS')
                line = row['points_line']
                bet_side = strat['bet_side']
                
                # Determine result
                if pd.isna(actual_points):
                    result = 'NO_DATA'
                    profit = 0.0
                elif bet_side == 'OVER':
                    if actual_points > line:
                        result = 'WIN'
                        profit = 100.0
                    elif actual_points < line:
                        result = 'LOSS'
                        profit = -110.0
                    else:
                        result = 'PUSH'
                        profit = 0.0
                else:  # UNDER
                    if actual_points < line:
                        result = 'WIN'
                        profit = 100.0
                    elif actual_points > line:
                        result = 'LOSS'
                        profit = -110.0
                    else:
                        result = 'PUSH'
                        profit = 0.0
                
                plays.append({
                    'game_date': row.get('game_date'),
                    'player_name': row.get('PLAYER_NAME'),
                    'team': row.get('TEAM_NAME'),
                    'opponent': row.get('MATCHUP'),
                    'points_line': line,
                    'team_spread': row.get('team_spread'),
                    'line_tier': line_tier,
                    'spread_bin': spread_bin,
                    'scorer_type': scorer_type if strategy_type == '3d' else None,
                    'bet_side': bet_side,
                    'actual_points': actual_points,
                    'result': result,
                    'profit': profit,
                    'season': row.get('season', '')
                })
    
    if not plays:
        print(f"   ⚠️  No plays found")
        return pd.DataFrame()
    
    df_plays = pd.DataFrame(plays)
    print(f"   ✅ Generated {len(df_plays):,} plays")
    
    # Filter out NO_DATA results
    df_plays = df_plays[df_plays['result'] != 'NO_DATA']
    print(f"   ✅ {len(df_plays):,} plays with valid results")
    
    return df_plays


def save_plays_to_s3(s3_client, df_plays: 'pd.DataFrame', strategy_type: str, season: str) -> bool:
    """
    Save plays DataFrame to S3 as CSV.
    
    Args:
        s3_client: Boto3 S3 client
        df_plays: DataFrame with plays
        strategy_type: '2d' or '3d'
        season: NBA season
    
    Returns:
        bool: True if successful
    """
    if not PANDAS_AVAILABLE or df_plays is None or df_plays.empty:
        print(f"   ⚠️  No plays to save")
        return False
    
    s3_key = f'{BACKTEST_PREFIX}/{strategy_type}/{season}/plays.csv'
    
    try:
        # Convert to CSV
        csv_buffer = StringIO()
        df_plays.to_csv(csv_buffer, index=False)
        
        # Upload to S3
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=csv_buffer.getvalue()
        )
        
        print(f"   ✅ Saved {len(df_plays)} plays to s3://{S3_BUCKET}/{s3_key}")
        return True
    except Exception as e:
        print(f"   ❌ Failed to save plays: {e}")
        return False


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


# =============================================================================
# BACKTEST FUNCTIONS
# =============================================================================

def run_backtest_for_season(s3_client, season: str, strategy_type: str) -> bool:
    """
    Run self-contained backtest for a specific season and strategy type.
    
    This generates ALL possible strategy combinations and tests them,
    eliminating the circular dependency on historical JSON files.
    
    Args:
        s3_client: Boto3 S3 client
        season: NBA season (e.g., '2025-26')
        strategy_type: '2d' or '3d'
    
    Returns:
        bool: True if successful
    """
    print(f"   Running {strategy_type.upper()} backtest for {season}...")
    
    # Step 1: Generate all possible strategy combinations
    all_strategies = generate_all_strategy_combinations(strategy_type)
    print(f"   ✅ Generated {len(all_strategies)} strategy combinations to test")
    
    # Step 1a: Validate strategy pairs (every OVER has an UNDER)
    validate_strategy_pairs(all_strategies, strategy_type)
    
    # Step 2: Load player props data from S3
    df_props = load_player_props_from_s3(s3_client, season, strategy_type)
    if df_props is None or df_props.empty:
        print(f"   ❌ No props data found for {season}")
        return False
    
    # Add season column if not present
    if 'season' not in df_props.columns:
        df_props['season'] = season
    
    # Step 3: Match props to strategies and calculate outcomes
    df_plays = match_and_calculate_plays(df_props, all_strategies, strategy_type)
    if df_plays is None or df_plays.empty:
        print(f"   ❌ No plays generated")
        return False
    
    # Step 3a: Validate inverse results (OVER losses ≈ UNDER wins)
    validate_inverse_results(df_plays, strategy_type)
    
    # Step 4: Save plays to S3
    success = save_plays_to_s3(s3_client, df_plays, strategy_type, season)
    
    if success:
        print(f"   ✅ {strategy_type.upper()} backtest complete")
    
    return success


def validate_strategy_pairs(strategies: List[Dict], strategy_type: str) -> bool:
    """
    Validate that every OVER strategy has a corresponding UNDER strategy.
    
    Args:
        strategies: List of strategy dictionaries
        strategy_type: '2d' or '3d'
    
    Returns:
        bool: True if validation passes
    
    Raises:
        AssertionError: If validation fails
    """
    print(f"\n   🔍 Validating strategy pairs...")
    
    # Group strategies by everything except bet_side
    if strategy_type == '2d':
        group_keys = ['line_tier', 'spread_bin']
    else:  # 3d
        group_keys = ['line_tier', 'spread_bin', 'scorer_type']
    
    # Create dictionary: (key_tuple) -> [list of strategies]
    strategy_groups = {}
    for strat in strategies:
        key = tuple(strat[k] for k in group_keys)
        if key not in strategy_groups:
            strategy_groups[key] = []
        strategy_groups[key].append(strat)
    
    # Validate each group has exactly 2 strategies (OVER and UNDER)
    errors = []
    for key, group in strategy_groups.items():
        if len(group) != 2:
            errors.append(f"   ❌ Strategy {key} has {len(group)} variations (expected 2)")
            continue
        
        bet_sides = {s['bet_side'] for s in group}
        if bet_sides != {'OVER', 'UNDER'}:
            errors.append(f"   ❌ Strategy {key} has bet_sides {bet_sides} (expected OVER and UNDER)")
    
    if errors:
        print(f"\n   ❌ VALIDATION FAILED:")
        for error in errors[:10]:  # Show first 10 errors
            print(error)
        if len(errors) > 10:
            print(f"   ... and {len(errors) - 10} more errors")
        raise AssertionError(f"Strategy pair validation failed: {len(errors)} issues found")
    
    print(f"   ✅ Validated {len(strategy_groups)} strategy pairs ({len(strategies)} total strategies)")
    print(f"   ✅ Each combination has both OVER and UNDER")
    
    return True


def validate_inverse_results(df_plays: 'pd.DataFrame', strategy_type: str) -> bool:
    """
    Validate that OVER and UNDER strategies have approximately inverse results.
    
    Args:
        df_plays: DataFrame with all plays
        strategy_type: '2d' or '3d'
    
    Returns:
        bool: True if validation passes
    """
    if not PANDAS_AVAILABLE or df_plays is None or df_plays.empty:
        print(f"   ⚠️  Cannot validate inverse results - no data")
        return True
    
    print(f"\n   🔍 Validating inverse results (OVER vs UNDER)...")
    
    # Group by everything except bet_side
    if strategy_type == '2d':
        group_cols = ['line_tier', 'spread_bin']
    else:  # 3d
        group_cols = ['line_tier', 'spread_bin', 'scorer_type']
    
    issues = []
    
    for group_key, group_df in df_plays.groupby(group_cols):
        over_df = group_df[group_df['bet_side'] == 'OVER']
        under_df = group_df[group_df['bet_side'] == 'UNDER']
        
        if len(over_df) == 0 or len(under_df) == 0:
            continue
        
        # Check that total plays are equal
        if len(over_df) != len(under_df):
            issues.append(f"   ⚠️  {group_key}: OVER has {len(over_df)} plays, UNDER has {len(under_df)} plays")
            continue
        
        # Calculate win rates
        over_wins = (over_df['result'] == 'WIN').sum()
        over_losses = (over_df['result'] == 'LOSS').sum()
        under_wins = (under_df['result'] == 'WIN').sum()
        under_losses = (under_df['result'] == 'LOSS').sum()
        
        # OVER wins should approximately equal UNDER losses (and vice versa)
        # Allow small discrepancy due to pushes
        if abs(over_wins - under_losses) > 5 or abs(over_losses - under_wins) > 5:
            over_wr = over_wins / (over_wins + over_losses) * 100 if (over_wins + over_losses) > 0 else 0
            under_wr = under_wins / (under_wins + under_losses) * 100 if (under_wins + under_losses) > 0 else 0
            issues.append(
                f"   ⚠️  {group_key}: OVER {over_wins}W-{over_losses}L ({over_wr:.1f}%), "
                f"UNDER {under_wins}W-{under_losses}L ({under_wr:.1f}%) - not inverse"
            )
    
    if issues:
        print(f"\n   ⚠️  Found {len(issues)} potential issues (showing first 5):")
        for issue in issues[:5]:
            print(issue)
        # Don't fail - just warn (pushes can cause legitimate differences)
        print(f"   Note: Small differences are OK due to pushes")
    else:
        print(f"   ✅ All OVER/UNDER pairs have inverse results")
    
    return True


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
    skip_backtest: bool = False
) -> Dict:
    """
    Main function to refresh strategy statistics.
    
    Self-contained implementation that:
    - Generates ALL possible strategy combinations (98 for 2D, 196 for 3D)
    - Loads player props directly from S3
    - Matches props to strategies and calculates outcomes
    - Aggregates statistics across multiple seasons
    - Filters by minimum plays (50+) not historical ROI
    
    Args:
        season: Current NBA season
        strategy_types: List of strategy types to update
        skip_backtest: If True, skip regenerating current season backtest
    
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
    print(f"Mode: Self-contained (tests all {98 if '2d' in strategy_types else 0}/{196 if '3d' in strategy_types else 0} combinations)")
    print("="*80)
    
    results = {}
    s3_client = boto3.client('s3')
    
    for strategy_type in strategy_types:
        print(f"\n{'='*80}")
        print(f"Processing {strategy_type.upper()} Strategy")
        print(f"{'='*80}\n")
        
        # Step 1: Re-run current season backtest (if not skipped)
        if not skip_backtest:
            print(f"Step 1: Updating {season} backtest...")
            success = run_backtest_for_season(s3_client, season, strategy_type)
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
        
        # Validate final strategy pairs before logging
        try:
            validate_strategy_pairs(strategies, strategy_type)
        except AssertionError as e:
            print(f"   ⚠️  Warning: {e}")
        
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
    
    # Determine strategy types
    if strategy == 'both':
        strategy_types = ['2d', '3d']
    else:
        strategy_types = [strategy]
    
    # Run refresh
    results = refresh_strategy_statistics(
        season=season,
        strategy_types=strategy_types,
        skip_backtest=skip_backtest
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
        skip_backtest=args.skip_backtest
    )
    
    # Exit with error code if any failed
    all_success = all(r.get('success', False) for r in results.values())
    sys.exit(0 if all_success else 1)


if __name__ == '__main__':
    main()
