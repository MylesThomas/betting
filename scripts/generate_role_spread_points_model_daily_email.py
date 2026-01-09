"""
Generate Daily Email for Role-Spread Points Model (2D/3D Strategies)

Combines yesterday's performance results with today's plays into a formatted email.
Supports both 2D (tier × spread) and 3D (tier × spread × scorer_type) strategies.
Outputs to stdout or sends via AWS SNS.

Context:
This is Step 8 of the daily workflow. It loads:
1. Yesterday's results from S3 (win/loss tracking from Step 7)
2. Today's plays from S3 (betting recommendations from Step 6)

And generates a formatted email containing both sections.

Usage:
    # Generate text email for both strategies (default)
    python scripts/generate_role_spread_points_model_daily_email.py --season 2025-26
    
    # Only 2D strategy
    python scripts/generate_role_spread_points_model_daily_email.py --season 2025-26 --strategy 2d
    
    # Only 3D strategy
    python scripts/generate_role_spread_points_model_daily_email.py --season 2025-26 --strategy 3d
    
    # Generate HTML email
    python scripts/generate_role_spread_points_model_daily_email.py --season 2025-26 --format html
    
    # Send via AWS SNS
    python scripts/generate_role_spread_points_model_daily_email.py --season 2025-26 --sns-topic arn:aws:sns:us-east-2:232692785472:nba-props-alerts
    
    # Specify dates (defaults to today for plays, yesterday for results)
    python scripts/generate_role_spread_points_model_daily_email.py --plays-date 2026-01-05 --results-date 2026-01-04 --strategy both

Output:
    - Console: Formatted email text or HTML
    - SNS: Published to specified topic (if --sns-topic provided)

Author: Thomas Myles
Date: 2026-01-06
"""

import pandas as pd
import boto3
from io import StringIO
import sys
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# =============================================================================
# EMOJI MAP
# =============================================================================

EMOJI = {
    'success': '✅',
    'error': '❌',
    'push': '🟰',
    'unknown': '❓',
    'fire': '🔥',
    'target': '🎯',
    'chart': '📊',
    'calendar': '📅',
    'clock': '⏰',
    'basketball': '🏀',
    'arrow_up': '📈',
    'arrow_down': '📉',
}

# =============================================================================
# S3 PATHS
# =============================================================================

STRATEGY_NAME = 'role_spread_points_model'
S3_BUCKET = 'nba-betting-mt'
S3_PREFIX_PLAYS = f'data/04_output/plays/{STRATEGY_NAME}'
S3_PREFIX_RESULTS = f'data/04_output/results/{STRATEGY_NAME}'

ET_TZ = ZoneInfo('America/New_York')


# =============================================================================
# DATA LOADING
# =============================================================================

def load_plays_from_s3(date_str, strategy='both', plays_suffix=''):
    """
    Load today's plays from S3 (supports 2d/, 3d/, or both)
    
    Args:
        date_str: Date string (YYYY-MM-DD)
        strategy: '2d', '3d', or 'both'
        plays_suffix: Suffix for filename (e.g., '_top3')
    
    Returns:
        DataFrame with plays (combined if strategy='both'), with overlap detection
    """
    s3 = boto3.client('s3')
    results = {}
    
    strategies_to_load = []
    if strategy in ['2d', 'both']:
        strategies_to_load.append('2d')
    if strategy in ['3d', 'both']:
        strategies_to_load.append('3d')
    
    for strat in strategies_to_load:
        key = f"{S3_PREFIX_PLAYS}/{strat}/{date_str}{plays_suffix}.csv"
        
        try:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
            df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
            df['strategy_dimension'] = strat.upper()  # Tag with 2D or 3D
            results[strat] = df
            print(f"   ✅ Loaded {len(df)} {strat.upper()} plays for {date_str}", file=sys.stderr)
        except s3.exceptions.NoSuchKey:
            print(f"   ⚠️  No {strat.upper()} plays found for {date_str}", file=sys.stderr)
            print(f"      Expected: s3://{S3_BUCKET}/{key}", file=sys.stderr)
        except Exception as e:
            print(f"   ❌ Error loading {strat.upper()} plays: {e}", file=sys.stderr)
    
    if not results:
        return None
    
    # If we have both 2D and 3D, detect overlaps
    if '2d' in results and '3d' in results:
        df_2d = results['2d']
        df_3d = results['3d']
        
        # Find overlaps (same player, bet_side, and line)
        overlaps = pd.merge(
            df_2d[['player', 'bet_side', 'line']],
            df_3d[['player', 'bet_side', 'line']],
            on=['player', 'bet_side', 'line'],
            how='inner'
        )
        
        if len(overlaps) > 0:
            print(f"   🔄 Found {len(overlaps)} overlapping plays between 2D and 3D", file=sys.stderr)
            
            # Mark overlaps in both dataframes
            for _, overlap in overlaps.iterrows():
                mask = (
                    (df_2d['player'] == overlap['player']) &
                    (df_2d['bet_side'] == overlap['bet_side']) &
                    (df_2d['line'] == overlap['line'])
                )
                df_2d.loc[mask, 'strategy_dimension'] = '2D AND 3D'
                
                mask = (
                    (df_3d['player'] == overlap['player']) &
                    (df_3d['bet_side'] == overlap['bet_side']) &
                    (df_3d['line'] == overlap['line'])
                )
                df_3d.loc[mask, 'strategy_dimension'] = '2D AND 3D'
            
            # For overlapping plays, keep only the 3D version (higher ROI)
            # Remove overlaps from 2D
            for _, overlap in overlaps.iterrows():
                mask = (
                    (df_2d['player'] == overlap['player']) &
                    (df_2d['bet_side'] == overlap['bet_side']) &
                    (df_2d['line'] == overlap['line'])
                )
                df_2d = df_2d[~mask]
            
            # Update results
            results['2d'] = df_2d
            results['3d'] = df_3d
    
    # Combine all results
    return pd.concat(list(results.values()), ignore_index=True)


def load_results_from_s3(date_str, strategy='both', tracking_suffix=''):
    """
    Load yesterday's results from S3 (supports 2d/, 3d/, or both)
    
    Args:
        date_str: Date string (YYYY-MM-DD)
        strategy: '2d', '3d', or 'both'
        tracking_suffix: Suffix for filename (e.g., '_top3')
    
    Returns:
        DataFrame with results (combined if strategy='both')
    """
    s3 = boto3.client('s3')
    results = []
    
    strategies_to_load = []
    if strategy in ['2d', 'both']:
        strategies_to_load.append('2d')
    if strategy in ['3d', 'both']:
        strategies_to_load.append('3d')
    
    for strat in strategies_to_load:
        key = f"{S3_PREFIX_RESULTS}/{strat}/{date_str}{tracking_suffix}.csv"
        
        try:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
            df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
            df['strategy_dimension'] = strat.upper()  # Tag with 2D or 3D
            results.append(df)
            print(f"   ✅ Loaded {len(df)} {strat.upper()} results for {date_str}", file=sys.stderr)
        except s3.exceptions.NoSuchKey:
            print(f"   ⚠️  No {strat.upper()} results found for {date_str}", file=sys.stderr)
            print(f"      Expected: s3://{S3_BUCKET}/{key}", file=sys.stderr)
        except Exception as e:
            print(f"   ❌ Error loading {strat.upper()} results: {e}", file=sys.stderr)
    
    if not results:
        return None
    
    # Combine all results
    return pd.concat(results, ignore_index=True)


def load_season_ytd_results(season, tracking_suffix='_top3'):
    """
    Load ALL tracking results for the season to calculate YTD stats.
    Uses parallel workers for fast loading of 100+ CSV files.
    
    Args:
        season: NBA season (e.g., '2025-26')
        tracking_suffix: Suffix for tracking files (e.g., '_top3')
    
    Returns:
        dict with overall and per-strategy stats, or None if no data
    """
    print(f"📊 Loading season YTD stats (suffix: '{tracking_suffix}')...", file=sys.stderr)
    
    s3 = boto3.client('s3')
    
    # List all tracking result files for the season
    all_files = []
    for dimension in ['2d', '3d']:
        prefix = f"{S3_PREFIX_RESULTS}/{dimension}/"
        
        response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
        if 'Contents' in response:
            files = [
                obj['Key'] for obj in response['Contents']
                if obj['Key'].endswith(f'{tracking_suffix}.csv')
            ]
            all_files.extend(files)
    
    if not all_files:
        print(f"   ⚠️  No YTD tracking files found", file=sys.stderr)
        return None
    
    print(f"   Found {len(all_files)} tracking files to load", file=sys.stderr)
    
    # Parallel load function
    def load_single_file(key):
        try:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
            df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
            return df
        except Exception as e:
            print(f"   ⚠️  Failed to load {key}: {e}", file=sys.stderr)
            return None
    
    # Load all files in parallel (100 workers for speed)
    all_data = []
    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = {executor.submit(load_single_file, key): key for key in all_files}
        for future in as_completed(futures):
            df = future.result()
            if df is not None and not df.empty:
                all_data.append(df)
    
    if not all_data:
        print(f"   ⚠️  No valid YTD data loaded", file=sys.stderr)
        return None
    
    # Combine all tracking results
    df_all = pd.concat(all_data, ignore_index=True)
    print(f"   ✅ Loaded {len(df_all)} total plays", file=sys.stderr)
    
    # Calculate overall stats
    wins = (df_all['result'] == 'win').sum()
    losses = (df_all['result'] == 'loss').sum()
    total = wins + losses
    win_pct = (wins / total * 100) if total > 0 else 0
    total_profit = df_all['profit'].sum()
    
    overall_stats = {
        'wins': wins,
        'losses': losses,
        'total': total,
        'win_pct': win_pct,
        'profit': total_profit
    }
    
    # Calculate per-strategy stats (if strategy_name column exists)
    strategy_stats = []
    if 'strategy_name' in df_all.columns:
        for strategy_name in df_all['strategy_name'].dropna().unique():
            df_strat = df_all[df_all['strategy_name'] == strategy_name]
            strat_wins = (df_strat['result'] == 'win').sum()
            strat_losses = (df_strat['result'] == 'loss').sum()
            strat_total = strat_wins + strat_losses
            strat_win_pct = (strat_wins / strat_total * 100) if strat_total > 0 else 0
            strat_profit = df_strat['profit'].sum()
            
            strategy_stats.append({
                'name': strategy_name,
                'wins': strat_wins,
                'losses': strat_losses,
                'total': strat_total,
                'win_pct': strat_win_pct,
                'profit': strat_profit
            })
        
        # Sort by profit descending
        strategy_stats = sorted(strategy_stats, key=lambda x: x['profit'], reverse=True)
    
    print(f"   ✅ YTD: {wins}-{losses} ({win_pct:.1f}%) | ${total_profit:,.2f} profit", file=sys.stderr)
    
    return {
        'overall': overall_stats,
        'strategies': strategy_stats
    }


# =============================================================================
# TEXT FORMATTING
# =============================================================================

def format_ytd_stats(ytd_stats):
    """Format YTD season stats for email"""
    if not ytd_stats:
        return ""
    
    overall = ytd_stats['overall']
    strategies = ytd_stats['strategies']
    
    text = f"""
{'='*80}
{EMOJI['chart']} 2025-26 SEASON PERFORMANCE (YTD)
{'='*80}

Overall: {overall['wins']}-{overall['losses']} ({overall['win_pct']:.1f}%) | ${overall['profit']:,.2f} profit

"""
    
    if strategies:
        text += "Strategy Breakdown:\n"
        for i, strat in enumerate(strategies, 1):
            text += f"  {i}. {strat['name']:30s} {strat['wins']:3d}-{strat['losses']:2d} ({strat['win_pct']:4.1f}%) | ${strat['profit']:>10,.2f}\n"
        text += "\n"
    
    text += f"{'='*80}\n\n"
    
    return text


def format_results_text(df_results, date_str):
    """Format yesterday's results as text"""
    if df_results is None or df_results.empty:
        return f"""
{'='*80}
{EMOJI['chart']} YESTERDAY'S RESULTS ({date_str})
{'='*80}

No results available for yesterday.
This might be your first day, or yesterday had no plays.

"""
    
    # Calculate summary stats
    total = len(df_results)
    wins = (df_results['result'] == 'WIN').sum()
    losses = (df_results['result'] == 'LOSS').sum()
    pushes = (df_results['result'] == 'PUSH').sum()
    dnp = (df_results['result'] == 'DNP').sum()
    
    win_pct = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
    
    # Calculate ROI
    total_wagered = (wins + losses) * 110
    profit = (wins * 100) - (losses * 110)
    actual_roi = (profit / total_wagered * 100) if total_wagered > 0 else 0
    expected_roi = df_results['expected_roi'].mean()
    
    # Build text
    text = f"""
{'='*80}
{EMOJI['chart']} YESTERDAY'S RESULTS ({date_str})
{'='*80}

{EMOJI['success']} {wins} WINS | {EMOJI['error']} {losses} LOSSES"""
    
    if pushes > 0:
        text += f" | {EMOJI['push']} {pushes} PUSHES"
    
    text += f"""
Win Rate: {win_pct:.1f}% | Actual ROI: {actual_roi:+.1f}% | Expected ROI: {expected_roi:+.1f}%

"""
    
    # Strategy dimension breakdown (if both 2D and 3D present)
    if 'strategy_dimension' in df_results.columns:
        dimensions = df_results['strategy_dimension'].unique()
        if len(dimensions) > 1:
            text += "BREAKDOWN BY STRATEGY:\n"
            text += "─" * 80 + "\n"
            
            for dim in sorted(dimensions):
                dim_data = df_results[df_results['strategy_dimension'] == dim]
                dim_wins = (dim_data['result'] == 'WIN').sum()
                dim_losses = (dim_data['result'] == 'LOSS').sum()
                dim_win_pct = (dim_wins / (dim_wins + dim_losses) * 100) if (dim_wins + dim_losses) > 0 else 0
                dim_profit = (dim_wins * 100) - (dim_losses * 110)
                
                text += f"{dim} Strategy: {dim_wins}-{dim_losses} ({dim_win_pct:.1f}%) | Profit: ${dim_profit:+.2f}\n"
            
            text += "\n"
    
    # Individual bets
    text += "INDIVIDUAL BETS:\n"
    text += "─" * 80 + "\n"
    
    for _, row in df_results.iterrows():
        if row['result'] == 'WIN':
            emoji = EMOJI['success']
        elif row['result'] == 'LOSS':
            emoji = EMOJI['error']
        elif row['result'] == 'PUSH':
            emoji = EMOJI['push']
        else:
            emoji = EMOJI['unknown']
        
        strat_label = f"[{row['strategy_dimension']}]" if 'strategy_dimension' in row else ""
        text += f"{emoji} {row['result']} {strat_label}: {row['player']} {row['bet_side']} {row['line']} pts\n"
        text += f"   Actual: {row['actual_pts']:.0f} pts | Margin: {row['margin']:+.1f}\n"
        text += f"   {row['team']} vs {row['opponent']} | Expected ROI: {row['expected_roi']:+.1f}%\n\n"
    
    return text


def format_plays_text(df_plays, date_str):
    """Format today's plays as text"""
    if df_plays is None or df_plays.empty:
        return f"""
{'='*80}
{EMOJI['target']} TODAY'S PLAYS ({date_str})
{'='*80}

No plays found for today.
Either no games match our strategies, or plays haven't been generated yet.

"""
    
    # Calculate summary
    total = len(df_plays)
    avg_roi = df_plays['expected_roi'].mean()
    
    text = f"""
{'='*80}
{EMOJI['target']} TODAY'S PLAYS ({date_str})
{'='*80}

Total Plays: {total} | Avg Expected ROI: {avg_roi:+.1f}%

"""
    
    # Strategy dimension breakdown (if both 2D and 3D present)
    if 'strategy_dimension' in df_plays.columns:
        dimensions = df_plays['strategy_dimension'].unique()
        if len(dimensions) > 1 or '2D AND 3D' in dimensions:
            text += "BREAKDOWN BY STRATEGY:\n"
            text += "─" * 80 + "\n"
            
            # Show 2D Only, 3D Only, and Both separately
            if '2D' in dimensions or '2D AND 3D' in dimensions:
                dim_2d_only = df_plays[df_plays['strategy_dimension'] == '2D']
                if len(dim_2d_only) > 0:
                    dim_avg_roi = dim_2d_only['expected_roi'].mean()
                    text += f"2D Only: {len(dim_2d_only)} plays | Avg Expected ROI: {dim_avg_roi:+.1f}%\n"
            
            if '3D' in dimensions or '2D AND 3D' in dimensions:
                dim_3d_only = df_plays[df_plays['strategy_dimension'] == '3D']
                if len(dim_3d_only) > 0:
                    dim_avg_roi = dim_3d_only['expected_roi'].mean()
                    text += f"3D Only: {len(dim_3d_only)} plays | Avg Expected ROI: {dim_avg_roi:+.1f}%\n"
            
            if '2D AND 3D' in dimensions:
                dim_both = df_plays[df_plays['strategy_dimension'] == '2D AND 3D']
                if len(dim_both) > 0:
                    dim_avg_roi = dim_both['expected_roi'].mean()
                    text += f"Both (2D+3D): {len(dim_both)} plays | Avg Expected ROI: {dim_avg_roi:+.1f}%\n"
            
            # Calculate total unique plays
            total_unique = len(df_plays)
            text += f"Total Unique: {total_unique} plays\n"
            
            text += "\n"
    
    # Group plays by game (team + opponent)
    # Create a sortable game identifier
    df_plays['game_key'] = df_plays.apply(
        lambda r: tuple(sorted([r['team'], r['opponent']])), axis=1
    )
    
    # Get unique games and their first occurrence for sorting
    games = df_plays.groupby('game_key').first().reset_index()
    
    game_num = 1
    for _, game in games.iterrows():
        game_teams = game['game_key']
        team1, team2 = game_teams
        
        # Get all plays for this game
        game_plays = df_plays[df_plays['game_key'] == game_teams].copy()
        
        # Sort by ROI descending
        game_plays = game_plays.sort_values('expected_roi', ascending=False)
        
        # Format game time if available
        game_time_str = ""
        if 'game_time' in game.index and pd.notna(game['game_time']):
            try:
                # Parse game_time - it might be a string or datetime
                if isinstance(game['game_time'], str):
                    game_time_dt = pd.to_datetime(game['game_time'])
                else:
                    game_time_dt = game['game_time']
                
                # Ensure it's timezone-aware (ET)
                if game_time_dt.tzinfo is None:
                    game_time_dt = game_time_dt.tz_localize(ET_TZ)
                else:
                    game_time_dt = game_time_dt.astimezone(ET_TZ)
                
                # Format as "6pm ET" (no minutes if on the hour)
                if game_time_dt.minute == 0:
                    time_formatted = game_time_dt.strftime('%I%p ET').lstrip('0').lower()
                else:
                    time_formatted = game_time_dt.strftime('%I:%M%p ET').lstrip('0').lower()
                game_time_str = f" ({time_formatted})"
            except Exception:
                # If parsing fails, just skip the time
                pass
        
        text += f"""{'─'*80}
{EMOJI['basketball']} GAME {game_num}: {team1} vs {team2}{game_time_str}
{'─'*80}

"""
        
        for _, play in game_plays.iterrows():
            strat_label = f"[{play['strategy_dimension']}]" if 'strategy_dimension' in play else ""
            text += f"{EMOJI['fire']} {strat_label} {play['bet_side']}: {play['player']} {play['line']} pts\n"
            text += f"   Team: {play['team']} (Spread: {play['spread']:+.1f})\n"
            text += f"   Strategy: {play['strategy_name']}\n"
            text += f"   Expected ROI: {play['expected_roi']:+.1f}% | Hit Rate: {play['hit_rate']:.1f}% ({play['games_in_sample']} games)\n"
            text += f"   Edge vs Baseline: {play['edge_vs_baseline']:+.1f}% | Edge vs Breakeven: {play['edge_vs_breakeven']:+.1f}%\n\n"
        
        game_num += 1
    
    return text


def generate_email_text(df_results, results_date, df_plays, plays_date, custom_title=None, ytd_stats=None):
    """Generate complete email body in text format"""
    
    if custom_title:
        subject = f"{custom_title} - {plays_date}"
    else:
        subject = f"NBA Role-Spread Model: Plays for {plays_date}"
        if df_results is not None:
            subject += f" + {results_date} Results"
    
    body = f"""
{'='*80}
{EMOJI['basketball']} NBA ROLE-SPREAD POINTS MODEL - DAILY UPDATE
{'='*80}
{EMOJI['calendar']} Generated: {datetime.now(ET_TZ).strftime('%Y-%m-%d %I:%M %p ET')}
{'='*80}
"""
    
    # Add YTD stats first (if provided)
    if ytd_stats:
        body += format_ytd_stats(ytd_stats)
    
    # Add results (yesterday's performance)
    body += format_results_text(df_results, results_date)
    
    # Add today's plays
    body += format_plays_text(df_plays, plays_date)
    
    body += f"""
{'='*80}
Strategy: Role-Spread Points Model (Detailed Granularity)
Generated by: /betting/scripts/generate_role_spread_points_model_daily_email.py
{'='*80}
"""
    
    return subject, body


# =============================================================================
# HTML FORMATTING (Optional)
# =============================================================================

def generate_email_html(df_results, results_date, df_plays, plays_date, custom_title=None, ytd_stats=None):
    """Generate complete email body in HTML format"""
    # TODO: Implement HTML formatting if needed
    # For now, just wrap text in <pre> tags
    subject, text_body = generate_email_text(df_results, results_date, df_plays, plays_date, custom_title, ytd_stats)
    html_body = f"<html><body><pre>{text_body}</pre></body></html>"
    return subject, html_body


# =============================================================================
# SNS PUBLISHING
# =============================================================================

def publish_to_sns(subject, body, topic_arn, format='text'):
    """Publish email to AWS SNS topic"""
    # Extract region from ARN: arn:aws:sns:REGION:ACCOUNT:TOPIC
    region = topic_arn.split(':')[3] if ':' in topic_arn else 'us-east-2'
    sns = boto3.client('sns', region_name=region)
    
    try:
        response = sns.publish(
            TopicArn=topic_arn,
            Subject=subject,
            Message=body,
            MessageStructure='string'
        )
        print(f"{EMOJI['success']} Published to SNS: {topic_arn}", file=sys.stderr)
        print(f"   Message ID: {response['MessageId']}", file=sys.stderr)
        return True
    except Exception as e:
        print(f"{EMOJI['error']} Failed to publish to SNS: {e}", file=sys.stderr)
        return False


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate daily email for Role-Spread Points Model (2D/3D)'
    )
    parser.add_argument('--season', type=str, default='2025-26',
                       help='NBA season (e.g., 2025-26)')
    parser.add_argument('--plays-date', type=str, default=None,
                       help='Date for plays (YYYY-MM-DD). Defaults to today (ET).')
    parser.add_argument('--results-date', type=str, default=None,
                       help='Date for results (YYYY-MM-DD). Defaults to yesterday (ET).')
    parser.add_argument('--strategy', type=str, default='both', choices=['2d', '3d', 'both'],
                       help='Which strategy to include: 2d, 3d, or both (default: both)')
    parser.add_argument('--format', choices=['text', 'html'], default='text',
                       help='Email format (default: text)')
    parser.add_argument('--sns-topic', type=str, default=None,
                       help='AWS SNS topic ARN to publish to (optional)')
    parser.add_argument('--output', type=str, default=None,
                       help='Save email body to file (optional)')
    parser.add_argument('--plays-suffix', type=str, default='',
                       help='Suffix for plays filename (e.g., "_top3")')
    parser.add_argument('--tracking-suffix', type=str, default='',
                       help='Suffix for tracking filename (e.g., "_top3")')
    parser.add_argument('--email-title', type=str, default='NBA Daily Props Report',
                       help='Custom email subject line')
    parser.add_argument('--load-ytd', action='store_true', default=False,
                       help='Load and display YTD season stats (default: False)')
    
    args = parser.parse_args()
    
    # Calculate dates in ET
    now_et = datetime.now(ET_TZ)
    
    if args.plays_date:
        plays_date = args.plays_date
    else:
        plays_date = now_et.strftime('%Y-%m-%d')
    
    if args.results_date:
        results_date = args.results_date
    else:
        yesterday_et = now_et - timedelta(days=1)
        results_date = yesterday_et.strftime('%Y-%m-%d')
    
    print(f"{EMOJI['calendar']} Generating email...", file=sys.stderr)
    print(f"   Plays date: {plays_date}", file=sys.stderr)
    print(f"   Results date: {results_date}", file=sys.stderr)
    print(f"   Strategy: {args.strategy}", file=sys.stderr)
    print(f"   Format: {args.format}", file=sys.stderr)
    print(f"   Season: {args.season}\n", file=sys.stderr)
    
    # Load data
    df_plays = load_plays_from_s3(plays_date, strategy=args.strategy, plays_suffix=args.plays_suffix)
    df_results = load_results_from_s3(results_date, strategy=args.strategy, tracking_suffix=args.tracking_suffix)
    
    if df_plays is None:
        print(f"{EMOJI['error']} Warning: No plays found for {plays_date} (strategy: {args.strategy})", file=sys.stderr)
        print(f"   Expected: s3://{S3_BUCKET}/{S3_PREFIX_PLAYS}/{{2d,3d}}/{plays_date}.csv\n", file=sys.stderr)
    else:
        print(f"{EMOJI['success']} Loaded {len(df_plays)} plays for {plays_date}", file=sys.stderr)
    
    if df_results is None:
        print(f"{EMOJI['error']} Warning: No results found for {results_date} (strategy: {args.strategy})", file=sys.stderr)
        print(f"   Expected: s3://{S3_BUCKET}/{S3_PREFIX_RESULTS}/{{2d,3d}}/{results_date}.csv\n", file=sys.stderr)
    else:
        print(f"{EMOJI['success']} Loaded {len(df_results)} results for {results_date}\n", file=sys.stderr)
    
    # Load YTD stats if requested
    ytd_stats = None
    if args.load_ytd:
        ytd_stats = load_season_ytd_results(args.season, tracking_suffix=args.tracking_suffix)
    
    # Generate email
    if args.format == 'html':
        subject, body = generate_email_html(df_results, results_date, df_plays, plays_date, args.email_title, ytd_stats)
    else:
        subject, body = generate_email_text(df_results, results_date, df_plays, plays_date, args.email_title, ytd_stats)
    
    # Output
    if args.output:
        # Save to file
        with open(args.output, 'w') as f:
            f.write(f"Subject: {subject}\n\n")
            f.write(body)
        print(f"{EMOJI['success']} Saved email to: {args.output}\n", file=sys.stderr)
    
    if args.sns_topic:
        # Publish to SNS
        publish_to_sns(subject, body, args.sns_topic, args.format)
    
    # Always print to stdout
    print(f"Subject: {subject}\n")
    print(body)


if __name__ == '__main__':
    main()

