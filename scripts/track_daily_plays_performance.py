"""
Track Daily Plays Performance (2D/3D Strategies)

Loads yesterday's plays from S3, fetches actual game results from S3,
calculates win/loss for each bet, and generates performance report.

Supports both 2D (tier × spread) and 3D (tier × spread × scorer_type) strategies.

Note: Uses Eastern Time (ET) for "yesterday" since NBA operates on ET.
If run at 2 AM ET, it tracks the previous day's games (which just finished).

Usage:
    # Track both 2D and 3D strategies (default)
    python scripts/track_daily_plays_performance.py
    
    # Track only 2D strategy
    python scripts/track_daily_plays_performance.py --strategy 2d
    
    # Track only 3D strategy
    python scripts/track_daily_plays_performance.py --strategy 3d
    
    # Track specific date
    python scripts/track_daily_plays_performance.py --date 2026-01-04 --strategy both
    
    # Specify season
    python scripts/track_daily_plays_performance.py --date 2026-01-04 --season 2025-26

Output:
    - Console: Detailed performance report (separate sections for each strategy)
    - S3: Results CSV saved to data/04_output/results/role_spread_points_model/{strategy}/{date}.csv

Requirements:
    - Plays must exist in S3 (from play finder scripts)
    - Game results must exist in S3 (from fetch_nba_player_props.py --fetch-games)

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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from player_name_utils import normalize_player_name

# S3 paths
S3_BUCKET_PLAYS = 'nba-betting-mt'
S3_PREFIX_PLAYS = 'data/04_output/plays/role_spread_points_model'
S3_PREFIX_RESULTS = 'data/04_output/results/role_spread_points_model'

S3_BUCKET_GAME_LOGS = 'nba-api-mt'
S3_PREFIX_GAME_LOGS = 'player_game_logs'


def load_plays_from_s3(date_str, season, strategy='both'):
    """
    Load plays CSV from S3 (supports 2d/, 3d/, or both)
    
    Args:
        date_str: Date string (YYYY-MM-DD)
        season: NBA season
        strategy: '2d', '3d', or 'both'
    
    Returns:
        dict: {'2d': df_2d, '3d': df_3d, 'combined': df_combined}
    """
    print(f"📥 Loading plays for {date_str} (strategy: {strategy})...")
    
    s3 = boto3.client('s3')
    results = {}
    
    strategies_to_load = []
    if strategy in ['2d', 'both']:
        strategies_to_load.append('2d')
    if strategy in ['3d', 'both']:
        strategies_to_load.append('3d')
    
    for strat in strategies_to_load:
        key = f"{S3_PREFIX_PLAYS}/{strat}/{date_str}.csv"
        
        try:
            obj = s3.get_object(Bucket=S3_BUCKET_PLAYS, Key=key)
            df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
            df['strategy_dimension'] = strat  # Tag with which strategy it came from
            results[strat] = df
            print(f"   ✅ Loaded {len(df)} {strat.upper()} plays from S3")
        except s3.exceptions.NoSuchKey:
            print(f"   ⚠️  No {strat.upper()} plays found for {date_str}")
            print(f"      Expected: s3://{S3_BUCKET_PLAYS}/{key}")
            results[strat] = pd.DataFrame()
        except Exception as e:
            print(f"   ❌ Error loading {strat.upper()} plays: {e}")
            results[strat] = pd.DataFrame()
    
    # Combine if loading both
    if strategy == 'both':
        if not results.get('2d', pd.DataFrame()).empty or not results.get('3d', pd.DataFrame()).empty:
            results['combined'] = pd.concat([results.get('2d', pd.DataFrame()), 
                                            results.get('3d', pd.DataFrame())], 
                                           ignore_index=True)
        else:
            results['combined'] = pd.DataFrame()
    
    return results


def load_game_results_from_s3(date_str, season):
    """Load actual game results from S3"""
    print(f"📥 Loading game results for {date_str}...")
    
    s3 = boto3.client('s3')
    key = f"{S3_PREFIX_GAME_LOGS}/{season}/{date_str}.csv"
    
    try:
        obj = s3.get_object(Bucket=S3_BUCKET_GAME_LOGS, Key=key)
        df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
        
        # Normalize player names for joining
        df['player_normalized'] = df['PLAYER_NAME'].apply(normalize_player_name)
        
        print(f"   ✅ Loaded stats for {len(df)} players from S3")
        return df
    except s3.exceptions.NoSuchKey:
        print(f"   ❌ No game results found for {date_str}")
        print(f"      Expected: s3://{S3_BUCKET_GAME_LOGS}/{key}")
        print(f"      Run: python scripts/fetch_nba_player_props.py --mode 2 --fetch-games --s3 --season {season}")
        return None
    except Exception as e:
        print(f"   ❌ Error loading game results: {e}")
        return None


def calculate_results(plays_df, results_df):
    """Calculate win/loss for each play"""
    print(f"\n🧮 Calculating bet results...")
    
    # Normalize player names in plays
    plays_df['player_normalized'] = plays_df['player'].apply(normalize_player_name)
    
    # Join plays with actual stats
    df = plays_df.merge(
        results_df[['player_normalized', 'PTS']],
        on='player_normalized',
        how='left'
    )
    
    # Calculate win/loss
    def determine_result(row):
        if pd.isna(row['PTS']):
            return 'DNP'
        
        actual = row['PTS']
        line = row['line']
        
        if row['bet_side'] == 'OVER':
            if actual > line:
                return 'WIN'
            elif actual < line:
                return 'LOSS'
            else:
                return 'PUSH'
        else:  # UNDER
            if actual < line:
                return 'WIN'
            elif actual > line:
                return 'LOSS'
            else:
                return 'PUSH'
    
    df['actual_pts'] = df['PTS']
    df['result'] = df.apply(determine_result, axis=1)
    df['margin'] = df['actual_pts'] - df['line']
    
    # Drop intermediate columns
    df = df.drop(['player_normalized', 'PTS'], axis=1)
    
    missing_data = (df['result'] == 'DNP').sum()
    if missing_data > 0:
        print(f"   ⚠️  {missing_data} plays with DNP (Did Not Play)")
    
    print(f"   ✅ Calculated results for {len(df)} plays")
    
    return df


def generate_report(df_results):
    """Generate detailed performance report"""
    print(f"\n{'='*80}")
    print(f"📊 PERFORMANCE REPORT: {df_results['date'].iloc[0]}")
    print(f"{'='*80}\n")
    
    # Individual bets
    print("INDIVIDUAL BETS:")
    print("─" * 80)
    
    for _, row in df_results.iterrows():
        if row['result'] == 'WIN':
            emoji = '✅'
        elif row['result'] == 'LOSS':
            emoji = '❌'
        elif row['result'] == 'PUSH':
            emoji = '🟰'
        else:
            emoji = '❓'
        
        print(f"{emoji} {row['result']}: {row['player']} {row['bet_side']} {row['line']} pts")
        print(f"   Actual: {row['actual_pts']:.0f} pts | Line: {row['line']} | Margin: {row['margin']:+.1f}")
        print(f"   Team: {row['team']} vs {row['opponent']}")
        print(f"   Strategy: {row['strategy_name']} (Expected ROI: {row['expected_roi']:+.1f}%)")
        print()
    
    # Summary stats
    print("─" * 80)
    print("SUMMARY:")
    print("─" * 80)
    
    total = len(df_results)
    wins = (df_results['result'] == 'WIN').sum()
    losses = (df_results['result'] == 'LOSS').sum()
    pushes = (df_results['result'] == 'PUSH').sum()
    dnp = (df_results['result'] == 'DNP').sum()
    
    win_pct = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
    
    print(f"Total Bets: {total}")
    print(f"Wins: {wins} ({wins/total*100:.1f}%)")
    print(f"Losses: {losses} ({losses/total*100:.1f}%)")
    print(f"Pushes: {pushes} ({pushes/total*100:.1f}%)")
    if dnp > 0:
        print(f"DNP: {dnp} ({dnp/total*100:.1f}%)")
    print()
    print(f"Win Rate (excluding pushes): {win_pct:.1f}%")
    
    # Calculate ROI at -110 odds
    # Win: +$100, Loss: -$110, Push: $0
    total_wagered = (wins + losses) * 110  # $110 per bet (to win $100)
    profit = (wins * 100) - (losses * 110)
    actual_roi = (profit / total_wagered * 100) if total_wagered > 0 else 0
    
    expected_roi = df_results['expected_roi'].mean()
    
    print()
    print(f"Expected ROI (avg): {expected_roi:+.1f}%")
    print(f"Actual ROI (at -110): {actual_roi:+.1f}%")
    print(f"Difference: {actual_roi - expected_roi:+.1f}%")
    
    # Strategy Dimension breakdown (2D vs 3D)
    if 'strategy_dimension' in df_results.columns:
        dimensions = df_results['strategy_dimension'].unique()
        if len(dimensions) > 1:
            print("\n─" * 80)
            print("BREAKDOWN BY STRATEGY DIMENSION:")
            print("─" * 80)
            
            for dim in sorted(dimensions):
                dim_data = df_results[df_results['strategy_dimension'] == dim]
                dim_wins = (dim_data['result'] == 'WIN').sum()
                dim_losses = (dim_data['result'] == 'LOSS').sum()
                dim_pushes = (dim_data['result'] == 'PUSH').sum()
                dim_win_pct = (dim_wins / (dim_wins + dim_losses) * 100) if (dim_wins + dim_losses) > 0 else 0
                
                # Calculate profit
                dim_profit = (dim_wins * 100) - (dim_losses * 110)
                
                print(f"{dim.upper()} Strategy: {dim_wins}-{dim_losses} ({dim_win_pct:.1f}%) | Profit: ${dim_profit:+.2f}")
    
    # Strategy breakdown
    print("\n─" * 80)
    print("STRATEGY BREAKDOWN:")
    print("─" * 80)
    
    strategy_stats = df_results.groupby('strategy_name').apply(
        lambda x: pd.Series({
            'wins': (x['result'] == 'WIN').sum(),
            'losses': (x['result'] == 'LOSS').sum(),
            'total': len(x),
            'win_pct': (x['result'] == 'WIN').sum() / ((x['result'] == 'WIN').sum() + (x['result'] == 'LOSS').sum()) * 100 if ((x['result'] == 'WIN').sum() + (x['result'] == 'LOSS').sum()) > 0 else 0
        })
    ).reset_index()
    
    for _, row in strategy_stats.iterrows():
        print(f"{row['strategy_name']}: {int(row['wins'])}-{int(row['losses'])} ({row['win_pct']:.1f}%)")
    
    print(f"{'='*80}\n")
    
    return {
        'total': total,
        'wins': wins,
        'losses': losses,
        'pushes': pushes,
        'win_pct': win_pct,
        'expected_roi': expected_roi,
        'actual_roi': actual_roi
    }


def save_results_to_s3(df_results, date_str, strategy='2d'):
    """
    Save results CSV to S3 (in strategy-specific subfolder)
    
    Args:
        df_results: DataFrame with tracking results
        date_str: Date string (YYYY-MM-DD)
        strategy: '2d' or '3d' (determines subfolder)
    """
    print(f"💾 Saving {strategy.upper()} results to S3...")
    
    s3 = boto3.client('s3')
    key = f"{S3_PREFIX_RESULTS}/{strategy}/{date_str}.csv"
    
    try:
        csv_buffer = StringIO()
        df_results.to_csv(csv_buffer, index=False)
        
        s3.put_object(
            Bucket=S3_BUCKET_PLAYS,
            Key=key,
            Body=csv_buffer.getvalue(),
            ContentType='text/csv'
        )
        
        print(f"   ✅ Saved {strategy.upper()} results to s3://{S3_BUCKET_PLAYS}/{key}")
        
    except Exception as e:
        print(f"   ❌ Failed to save {strategy.upper()} results: {e}")


def main():
    parser = argparse.ArgumentParser(description='Track daily plays performance (2D/3D)')
    parser.add_argument('--date', type=str, default=None,
                       help='Date to track (YYYY-MM-DD). Defaults to yesterday.')
    parser.add_argument('--season', type=str, default='2025-26',
                       help='NBA season (e.g., 2025-26)')
    parser.add_argument('--strategy', type=str, default='both', choices=['2d', '3d', 'both'],
                       help='Which strategy to track: 2d, 3d, or both (default: both)')
    
    args = parser.parse_args()
    
    # Default to yesterday (in ET) if not specified
    if args.date:
        date_str = args.date
    else:
        # Use Eastern Time (NBA operates on ET)
        et_tz = ZoneInfo('America/New_York')
        now_et = datetime.now(et_tz)
        yesterday_et = now_et - timedelta(days=1)
        date_str = yesterday_et.strftime('%Y-%m-%d')
        print(f"💡 No date specified, using yesterday (ET): {date_str}")
        print(f"   Current time ET: {now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}\n")
    
    # Load plays (returns dict with 2d, 3d, and/or combined)
    plays_dict = load_plays_from_s3(date_str, args.season, args.strategy)
    
    # Check if we got any plays
    if args.strategy == 'both':
        if plays_dict.get('combined', pd.DataFrame()).empty:
            print(f"\n❌ Cannot track performance - no plays found for {date_str}")
            return
        plays_to_track = plays_dict['combined']
    else:
        if plays_dict.get(args.strategy, pd.DataFrame()).empty:
            print(f"\n❌ Cannot track performance - no {args.strategy.upper()} plays found for {date_str}")
            return
        plays_to_track = plays_dict[args.strategy]
    
    # Load game results
    results_df = load_game_results_from_s3(date_str, args.season)
    if results_df is None or results_df.empty:
        print(f"\n❌ Cannot track performance - no game results found for {date_str}")
        return
    
    # Calculate results
    df_results = calculate_results(plays_to_track, results_df)
    
    # Generate report (now includes strategy_dimension column)
    summary = generate_report(df_results)
    
    # Save results (separate files for 2d/3d if tracking both)
    if args.strategy == 'both':
        # Save separate results for each strategy
        for strat in ['2d', '3d']:
            strat_results = df_results[df_results['strategy_dimension'] == strat]
            if not strat_results.empty:
                save_results_to_s3(strat_results, date_str, strategy=strat)
    else:
        save_results_to_s3(df_results, date_str, strategy=args.strategy)
    
    # Return summary for programmatic access
    return df_results, summary


if __name__ == '__main__':
    main()

