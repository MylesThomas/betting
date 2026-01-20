"""
Update Kelly Criterion Bankroll Tracker

This script updates the bankroll based on yesterday's Top3 plays results.
It reads the current bankroll from S3, calculates PnL from yesterday's bets,
and writes updated bankroll back to S3 with history.

Context:
Run this daily BEFORE generating today's plays, so that today's Kelly
calculations use the updated bankroll from yesterday's results.

Workflow:
1. Load current bankroll from S3 config
2. Load yesterday's Top3 plays (what we recommended)
3. Load yesterday's Top3 tracking results (what actually happened)
4. Calculate PnL for each bet using Kelly sizing
5. Update bankroll and save to S3

Usage:
    # Update bankroll for today based on yesterday's results
    python tmp/update_kelly_bankroll_tracker.py --date 2026-01-19
    
    # Initialize new tracker (one-time setup)
    python scripts/update_kelly_bankroll_tracker.py --initialize --starting-bankroll 10000 --starting-date 2026-01-19
    
    # Dry run (don't save to S3)
    python scripts/update_kelly_bankroll_tracker.py --date 2026-01-19 --dry-run

Author: Thomas Myles
Date: 2026-01-19
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import boto3
import json
import argparse
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from kelly_criterion import calculate_kelly_criterion, kelly_bet_size

# =============================================================================
# CONSTANTS
# =============================================================================

S3_BUCKET = 'nba-betting-mt'
S3_CONFIG_PATH = 'config/kelly_bankroll_tracker.json'
ET_TZ = ZoneInfo('America/New_York')

# Default Kelly fraction (0.5 = half Kelly for reduced variance)
DEFAULT_FRACTIONAL_KELLY = 0.5
MAX_KELLY = 0.10

# =============================================================================
# S3 FUNCTIONS
# =============================================================================

def load_config_from_s3():
    """Load Kelly bankroll tracker config from S3"""
    s3_client = boto3.client('s3')
    
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=S3_CONFIG_PATH)
        config = json.loads(response['Body'].read().decode('utf-8'))
        print(f"✅ Loaded config from s3://{S3_BUCKET}/{S3_CONFIG_PATH}")
        return config
    except s3_client.exceptions.NoSuchKey:
        print(f"⚠️  Config not found at s3://{S3_BUCKET}/{S3_CONFIG_PATH}")
        print(f"   Run with --initialize to create initial config")
        return None


def save_config_to_s3(config):
    """Save Kelly bankroll tracker config to S3"""
    s3_client = boto3.client('s3')
    
    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key=S3_CONFIG_PATH,
        Body=json.dumps(config, indent=2),
        ContentType='application/json'
    )
    print(f"✅ Saved config to s3://{S3_BUCKET}/{S3_CONFIG_PATH}")


def load_top3_plays_from_s3(date_str):
    """Load Top3 plays CSV from S3 for a given date"""
    s3_client = boto3.client('s3')
    
    plays_2d = []
    plays_3d = []
    
    # Load 2D plays
    try:
        s3_path_2d = f'data/04_output/plays/role_spread_points_model/2d/{date_str}_top3.csv'
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_path_2d)
        df_2d = pd.read_csv(response['Body'])
        df_2d['dimension'] = '2D'
        plays_2d.append(df_2d)
        print(f"✅ Loaded {len(df_2d)} 2D plays from s3://{S3_BUCKET}/{s3_path_2d}")
    except s3_client.exceptions.NoSuchKey:
        print(f"⚠️  No 2D plays found for {date_str}")
    
    # Load 3D plays
    try:
        s3_path_3d = f'data/04_output/plays/role_spread_points_model/3d/{date_str}_top3.csv'
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_path_3d)
        df_3d = pd.read_csv(response['Body'])
        df_3d['dimension'] = '3D'
        plays_3d.append(df_3d)
        print(f"✅ Loaded {len(df_3d)} 3D plays from s3://{S3_BUCKET}/{s3_path_3d}")
    except s3_client.exceptions.NoSuchKey:
        print(f"⚠️  No 3D plays found for {date_str}")
    
    if not plays_2d and not plays_3d:
        print(f"❌ No Top3 plays found for {date_str}")
        return None
    
    # Combine and dedupe (some plays appear in both 2D and 3D)
    df_plays = pd.concat(plays_2d + plays_3d, ignore_index=True)
    
    # Dedupe by player + line + bet_side (keep first occurrence)
    df_plays = df_plays.drop_duplicates(subset=['player', 'line', 'bet_side'], keep='first')
    
    print(f"✅ Total unique Top3 plays: {len(df_plays)}")
    return df_plays


def load_top3_tracking_from_s3(date_str):
    """Load Top3 tracking results CSV from S3 for a given date (both 2D and 3D)"""
    s3_client = boto3.client('s3')
    
    results_2d = []
    results_3d = []
    
    # Load 2D tracking results
    try:
        s3_path_2d = f'data/04_output/results/role_spread_points_model/2d/{date_str}_top3.csv'
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_path_2d)
        df_2d = pd.read_csv(response['Body'])
        df_2d['dimension'] = '2D'
        results_2d.append(df_2d)
        print(f"✅ Loaded {len(df_2d)} 2D tracking results from s3://{S3_BUCKET}/{s3_path_2d}")
    except s3_client.exceptions.NoSuchKey:
        print(f"⚠️  No 2D tracking results found for {date_str}")
    
    # Load 3D tracking results
    try:
        s3_path_3d = f'data/04_output/results/role_spread_points_model/3d/{date_str}_top3.csv'
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_path_3d)
        df_3d = pd.read_csv(response['Body'])
        df_3d['dimension'] = '3D'
        results_3d.append(df_3d)
        print(f"✅ Loaded {len(df_3d)} 3D tracking results from s3://{S3_BUCKET}/{s3_path_3d}")
    except s3_client.exceptions.NoSuchKey:
        print(f"⚠️  No 3D tracking results found for {date_str}")
    
    if not results_2d and not results_3d:
        print(f"❌ No tracking results found for {date_str} (checked both 2D and 3D)")
        return None
    
    # Combine and dedupe (some plays appear in both 2D and 3D)
    df_tracking = pd.concat(results_2d + results_3d, ignore_index=True)
    
    # Dedupe by player + line + bet_side (keep first occurrence)
    df_tracking = df_tracking.drop_duplicates(subset=['player', 'line', 'bet_side'], keep='first')
    
    print(f"✅ Total unique tracking results: {len(df_tracking)}")
    return df_tracking


# =============================================================================
# KELLY CALCULATION
# =============================================================================

def extract_best_odds_from_bookmakers(bookmaker_details_json, default_odds=-110):
    """
    Extract best odds from bookmaker details JSON.
    Falls back to average if multiple, then default if none available.
    
    Args:
        bookmaker_details_json: JSON string with bookmaker details
        default_odds: Default odds if no bookmakers available
    
    Returns:
        Best American odds as integer
    """
    try:
        details = json.loads(bookmaker_details_json)
        if not details:
            return default_odds
        
        # Extract all non-null odds
        all_odds = [book['odds'] for book in details if book.get('odds') is not None]
        
        if not all_odds:
            return default_odds
        
        # Return average of available odds (rounded to nearest integer)
        avg_odds = sum(all_odds) / len(all_odds)
        return int(round(avg_odds))
    except (json.JSONDecodeError, KeyError, TypeError):
        return default_odds


def calculate_bet_pnl(row, fractional_kelly, bankroll):
    """
    Calculate PnL for a single bet using Kelly sizing.
    
    Args:
        row: DataFrame row with bet details (player, line, bet_side, result, etc.)
        fractional_kelly: Kelly fraction to use (e.g., 0.5 for half Kelly)
        bankroll: Current bankroll
    
    Returns:
        dict with bet details and PnL
    """
    # Calculate Kelly %
    win_prob = row['hit_rate'] / 100
    
    # Extract odds from bookmaker JSON
    bet_side = row['bet_side']
    if bet_side == 'OVER':
        bookmaker_json = row.get('bookmaker_details_over', '[]')
    else:
        bookmaker_json = row.get('bookmaker_details_under', '[]')
    
    odds = extract_best_odds_from_bookmakers(bookmaker_json, default_odds=-110)
    
    kelly_result = calculate_kelly_criterion(win_prob, odds, max_kelly=MAX_KELLY)
    kelly_pct = kelly_result['kelly_pct']
    
    # Apply fractional Kelly
    actual_kelly_pct = kelly_pct * fractional_kelly
    
    # Calculate bet size
    bet_amount = kelly_bet_size(actual_kelly_pct, bankroll)
    
    # Calculate PnL based on result
    result = row.get('result', 'DNP')
    
    if result == 'WIN':
        # Calculate profit using odds
        if odds < 0:
            profit = bet_amount * (100 / abs(odds))
        else:
            profit = bet_amount * (odds / 100)
        pnl = profit
    elif result == 'LOSS':
        pnl = -bet_amount
    elif result == 'PUSH':
        pnl = 0
    else:  # DNP or unknown
        pnl = 0
        bet_amount = 0
    
    return {
        'player': row['player'],
        'line': row['line'],
        'bet_side': row['bet_side'],
        'result': result,
        'odds': odds,
        'win_prob': win_prob,
        'kelly_pct': kelly_pct * 100,
        'fractional_kelly_pct': actual_kelly_pct * 100,
        'bet_amount': bet_amount,
        'pnl': pnl
    }


# =============================================================================
# MAIN UPDATE LOGIC
# =============================================================================

def update_bankroll(date_str, dry_run=False):
    """
    Update bankroll based on yesterday's results.
    
    Args:
        date_str: Today's date (YYYY-MM-DD)
        dry_run: If True, don't save to S3
    """
    print(f"\n{'='*80}")
    print(f"Kelly Bankroll Tracker - Update for {date_str}")
    print(f"{'='*80}\n")
    
    # Load config
    config = load_config_from_s3()
    if not config:
        return
    
    # Get yesterday's date
    today = datetime.strptime(date_str, '%Y-%m-%d').date()
    yesterday = (today - timedelta(days=1)).strftime('%Y-%m-%d')
    
    print(f"Today: {date_str}")
    print(f"Yesterday: {yesterday}\n")
    
    # Get current bankroll
    current_bankroll = config['current_bankroll']
    fractional_kelly = config.get('fractional_kelly', DEFAULT_FRACTIONAL_KELLY)
    
    print(f"Current bankroll: ${current_bankroll:,.2f}")
    print(f"Fractional Kelly: {fractional_kelly}x\n")
    
    # Load yesterday's Top3 plays
    df_plays = load_top3_plays_from_s3(yesterday)
    if df_plays is None:
        print(f"⚠️  No plays found for {yesterday}, skipping update")
        return
    
    # Load yesterday's tracking results
    df_tracking = load_top3_tracking_from_s3(yesterday)
    if df_tracking is None:
        print(f"⚠️  No tracking results found for {yesterday}, skipping update")
        return
    
    # Merge plays with results
    df_merged = df_plays.merge(
        df_tracking[['player', 'line', 'bet_side', 'result', 'actual_pts']],
        on=['player', 'line', 'bet_side'],
        how='left'
    )
    
    # Calculate PnL for each bet
    print(f"\n{'='*80}")
    print(f"Calculating PnL for {len(df_merged)} bets")
    print(f"{'='*80}\n")
    
    bets = []
    total_wagered = 0
    total_pnl = 0
    
    for _, row in df_merged.iterrows():
        bet_pnl = calculate_bet_pnl(row, fractional_kelly, current_bankroll)
        bets.append(bet_pnl)
        
        if bet_pnl['bet_amount'] > 0:
            total_wagered += bet_pnl['bet_amount']
            total_pnl += bet_pnl['pnl']
            
            result_emoji = {'WIN': '✅', 'LOSS': '❌', 'PUSH': '🟰', 'DNP': '❓'}.get(bet_pnl['result'], '❓')
            print(f"{result_emoji} {bet_pnl['player']} {row['bet_side']} {bet_pnl['line']}")
            print(f"   Kelly: {bet_pnl['fractional_kelly_pct']:.1f}% | Bet: ${bet_pnl['bet_amount']:.0f} | PnL: ${bet_pnl['pnl']:+.0f}")
    
    # Calculate new bankroll
    new_bankroll = current_bankroll + total_pnl
    roi_pct = (total_pnl / total_wagered * 100) if total_wagered > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"Summary for {yesterday}")
    print(f"{'='*80}")
    print(f"Starting bankroll: ${current_bankroll:,.2f}")
    print(f"Total wagered: ${total_wagered:,.2f}")
    print(f"Total PnL: ${total_pnl:+,.2f}")
    print(f"ROI: {roi_pct:+.1f}%")
    print(f"Ending bankroll: ${new_bankroll:,.2f}")
    print(f"{'='*80}\n")
    
    # Update config
    history_entry = {
        'date': yesterday,
        'starting_bankroll': current_bankroll,
        'plays': bets,
        'total_wagered': total_wagered,
        'total_pnl': total_pnl,
        'ending_bankroll': new_bankroll,
        'roi_pct': roi_pct
    }
    
    config['history'].append(history_entry)
    config['current_bankroll'] = new_bankroll
    config['current_date'] = date_str
    
    # Save to S3
    if dry_run:
        print("🔍 DRY RUN - Not saving to S3")
        print(f"\nWould save config:")
        print(json.dumps(config, indent=2))
    else:
        save_config_to_s3(config)
        print(f"✅ Bankroll updated to ${new_bankroll:,.2f} for {date_str}")


def initialize_tracker(starting_bankroll, starting_date, fractional_kelly=DEFAULT_FRACTIONAL_KELLY):
    """Initialize Kelly bankroll tracker (one-time setup)"""
    print(f"\n{'='*80}")
    print(f"Initializing Kelly Bankroll Tracker")
    print(f"{'='*80}\n")
    
    config = {
        'starting_bankroll': starting_bankroll,
        'starting_date': starting_date,
        'current_bankroll': starting_bankroll,
        'current_date': starting_date,
        'fractional_kelly': fractional_kelly,
        'max_kelly': MAX_KELLY,
        'history': [
            {
                'date': starting_date,
                'starting_bankroll': starting_bankroll,
                'plays': [],
                'total_wagered': 0,
                'total_pnl': 0,
                'ending_bankroll': starting_bankroll,
                'roi_pct': 0.0
            }
        ]
    }
    
    print(f"Starting bankroll: ${starting_bankroll:,.2f}")
    print(f"Starting date: {starting_date}")
    print(f"Fractional Kelly: {fractional_kelly}x")
    print(f"Max Kelly: {MAX_KELLY*100:.0f}%\n")
    
    save_config_to_s3(config)
    print(f"✅ Initialized tracker at s3://{S3_BUCKET}/{S3_CONFIG_PATH}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Update Kelly Criterion Bankroll Tracker')
    
    parser.add_argument('--date', type=str,
                       help='Today\'s date (YYYY-MM-DD). Defaults to today ET.')
    
    parser.add_argument('--initialize', action='store_true',
                       help='Initialize new tracker (one-time setup)')
    
    parser.add_argument('--starting-bankroll', type=float, default=10000,
                       help='Starting bankroll for initialization (default: 10000)')
    
    parser.add_argument('--starting-date', type=str,
                       help='Starting date for initialization (YYYY-MM-DD)')
    
    parser.add_argument('--fractional-kelly', type=float, default=DEFAULT_FRACTIONAL_KELLY,
                       help=f'Fractional Kelly multiplier (default: {DEFAULT_FRACTIONAL_KELLY})')
    
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run - don\'t save to S3')
    
    args = parser.parse_args()
    
    # Get today's date (ET timezone)
    if args.date:
        date_str = args.date
    else:
        date_str = datetime.now(ET_TZ).strftime('%Y-%m-%d')
    
    if args.initialize:
        # Initialize tracker
        starting_date = args.starting_date or date_str
        initialize_tracker(args.starting_bankroll, starting_date, args.fractional_kelly)
    else:
        # Update bankroll
        update_bankroll(date_str, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
