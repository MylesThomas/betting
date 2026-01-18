"""
Backtest Strategy Analyzer

Given strategy parameters, load backtest data for multiple seasons and return
performance metrics in JSON format ready for strategy configs.

Usage:
    python scripts/backtest_strategy_analyzer.py \
        --strategy-type 3d \
        --line-tier "5-10 (Bench)" \
        --spread-bin "Pick'em (-2 to +2)" \
        --bet-side UNDER \
        --scorer-type "Rim Attacker (≥40.0%)" \
        --seasons 2023-24 2024-25 2025-26

Or use as a module:
    from scripts.backtest_strategy_analyzer import analyze_strategy
    
    result = analyze_strategy(
        strategy_type='3d',
        line_tier='5-10 (Bench)',
        spread_bin='Pick\'em (-2 to +2)',
        bet_side='UNDER',
        scorer_type='Rim Attacker (≥40.0%)',
        strategy_name='bench_pickem_rim_under',
        seasons=['2023-24', '2024-25', '2025-26']
    )

Author: Myles Thomas
Date: 2026-01-18
"""

import pandas as pd
import boto3
import json
import argparse
from io import StringIO
from typing import List, Dict, Optional


def load_backtest_plays(s3_client, bucket: str, strategy_type: str, season: str) -> pd.DataFrame:
    """
    Load backtest plays CSV for a given season and strategy type.
    
    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        strategy_type: '2d' or '3d'
        season: Season string like '2023-24'
    
    Returns:
        DataFrame of plays
    """
    s3_key = f'data/04_output/backtests/{strategy_type}/{season}/plays.csv'
    
    try:
        response = s3_client.get_object(Bucket=bucket, Key=s3_key)
        df = pd.read_csv(StringIO(response['Body'].read().decode('utf-8')))
        df['season'] = season
        return df
    except Exception as e:
        print(f"⚠️  Could not load {season} {strategy_type.upper()}: {e}")
        return pd.DataFrame()


def load_training_roi(s3_client, bucket: str, strategy_type: str, 
                     line_tier: str, spread_bin: str, bet_side: str,
                     scorer_type: Optional[str] = None) -> float:
    """
    Load training ROI from 2025-26 training strategies.
    
    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket name
        strategy_type: '2d' or '3d'
        line_tier: e.g., '5-10 (Bench)'
        spread_bin: e.g., 'Pick\'em (-2 to +2)'
        bet_side: 'UNDER' or 'OVER'
        scorer_type: e.g., 'Rim Attacker (≥40.0%)' (for 3D only)
    
    Returns:
        Training ROI as float
    """
    if strategy_type == '2d':
        s3_key = 'data/03_intermediate/points_by_role_gamespread_strategies_2025-26.json'
    else:
        s3_key = 'data/03_intermediate/points_by_role_gamespread_6feet_strategies_2025-26_rim40.json'
    
    try:
        response = s3_client.get_object(Bucket=bucket, Key=s3_key)
        data = json.loads(response['Body'].read().decode('utf-8'))
        
        if 'strategies' in data:
            strategies = data['strategies']
        else:
            strategies = data
        
        if isinstance(strategies, dict):
            strategies = list(strategies.values())
        
        # Find matching strategy
        for strat in strategies:
            match = (
                strat.get('line_tier') == line_tier and
                strat.get('spread_bin') == spread_bin and
                strat.get('bet_side') == bet_side
            )
            
            if strategy_type == '3d' and scorer_type:
                match = match and (strat.get('scorer_type') == scorer_type)
            
            if match:
                return round(strat.get('roi', 0.0), 1)
        
        print(f"⚠️  Strategy not found in training data")
        return 0.0
        
    except Exception as e:
        print(f"⚠️  Could not load training data: {e}")
        return 0.0


def analyze_strategy(
    strategy_type: str,
    line_tier: str,
    spread_bin: str,
    bet_side: str,
    strategy_name: Optional[str] = None,
    seasons: List[str] = ['2023-24', '2024-25', '2025-26'],
    scorer_type: Optional[str] = None,
    notes: str = ""
) -> Dict:
    """
    Analyze strategy performance across multiple seasons.
    
    Args:
        strategy_type: '2d' or '3d'
        line_tier: e.g., '5-10 (Bench)'
        spread_bin: e.g., 'Pick\'em (-2 to +2)'
        bet_side: 'UNDER' or 'OVER'
        strategy_name: Name for the strategy (optional - auto-generated if not provided)
        seasons: List of seasons to analyze
        scorer_type: e.g., 'Rim Attacker (≥40.0%)' (for 3D only)
        notes: Optional notes about the strategy
    
    Returns:
        Dict with strategy performance data
    """
    # Auto-generate strategy name if not provided
    if strategy_name is None:
        tier_slug = line_tier.split(' ')[0].replace('-', '_').lower()  # "5-10" -> "5_10"
        spread_slug = spread_bin.split(' ')[0].replace('-', '_').replace('\'', '').lower()  # "Pick'em" -> "pickem"
        bet_slug = bet_side.lower()
        
        if strategy_type == '3d' and scorer_type:
            scorer_slug = 'rim' if 'Rim' in scorer_type else 'perimeter'
            strategy_name = f"{tier_slug}_{spread_slug}_{scorer_slug}_{bet_slug}"
        else:
            strategy_name = f"{tier_slug}_{spread_slug}_{bet_slug}"
    
    print("="*80)
    print(f"ANALYZING STRATEGY: {strategy_name}")
    print("="*80)
    print(f"Type: {strategy_type.upper()}")
    print(f"Line Tier: {line_tier}")
    print(f"Spread Bin: {spread_bin}")
    print(f"Bet Side: {bet_side}")
    if scorer_type:
        print(f"Scorer Type: {scorer_type}")
    print(f"Seasons: {', '.join(seasons)}")
    
    # Setup S3
    s3_client = boto3.client('s3')
    bucket = 'nba-betting-mt'
    
    # Load training ROI
    training_roi = load_training_roi(
        s3_client, bucket, strategy_type, 
        line_tier, spread_bin, bet_side, scorer_type
    )
    print(f"\n✅ Training ROI (2025-26): {training_roi:+.1f}%")
    
    # Load backtest data for each season
    all_plays = []
    season_stats = {}
    
    print(f"\n{'='*80}")
    print("LOADING BACKTEST DATA")
    print("="*80)
    
    for season in seasons:
        df = load_backtest_plays(s3_client, bucket, strategy_type, season)
        
        if len(df) == 0:
            season_stats[season] = {
                'profit': 0.0,
                'plays': 0,
                'profitable': False
            }
            print(f"⚠️  {season}: No data")
            continue
        
        # Filter to strategy
        strategy_plays = df[
            (df['line_tier'] == line_tier) &
            (df['spread_bin'] == spread_bin) &
            (df['bet_side'] == bet_side)
        ].copy()
        
        # For 3D, also filter by scorer_type
        if strategy_type == '3d' and scorer_type:
            strategy_plays = strategy_plays[
                strategy_plays['scorer_type'] == scorer_type
            ]
        
        if len(strategy_plays) == 0:
            season_stats[season] = {
                'profit': 0.0,
                'plays': 0,
                'profitable': False
            }
            print(f"⚠️  {season}: No matching plays")
            continue
        
        # Calculate stats
        wins = (strategy_plays['result'] == 'WIN').sum()
        losses = (strategy_plays['result'] == 'LOSS').sum()
        total_profit = strategy_plays['profit'].sum()
        
        season_stats[season] = {
            'profit': round(float(total_profit), 2),
            'plays': int(len(strategy_plays)),
            'profitable': bool(total_profit > 0)
        }
        
        all_plays.append(strategy_plays)
        
        print(f"✅ {season}: {len(strategy_plays)} plays | ${total_profit:,.2f} profit | {wins}-{losses}")
    
    # Calculate overall stats
    print(f"\n{'='*80}")
    print("OVERALL PERFORMANCE")
    print("="*80)
    
    if all_plays:
        df_all = pd.concat(all_plays, ignore_index=True)
        total_plays = len(df_all)
        total_wins = (df_all['result'] == 'WIN').sum()
        total_losses = (df_all['result'] == 'LOSS').sum()
        total_profit = df_all['profit'].sum()
        win_rate = (total_wins / (total_wins + total_losses) * 100) if (total_wins + total_losses) > 0 else 0
        profitable_seasons = sum(1 for s in season_stats.values() if s['profitable'])
    else:
        total_plays = 0
        total_profit = 0.0
        win_rate = 0.0
        profitable_seasons = 0
    
    print(f"Total Plays: {total_plays}")
    print(f"Total Profit: ${total_profit:,.2f}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Profitable Seasons: {profitable_seasons}/{len(seasons)}")
    
    # Build result dict
    result = {
        'strategy_name': strategy_name,
        'strategy_type': strategy_type,
        'line_tier': line_tier,
        'spread_bin': spread_bin,
        'bet_side': bet_side,
        'training_roi': training_roi
    }
    
    if strategy_type == '3d' and scorer_type:
        result['scorer_type'] = scorer_type
    else:
        result['scorer_type'] = 'N/A'
    
    # Add season-specific stats
    for season in seasons:
        season_key = f"backtest_{season.replace('-', '_')}"
        result[season_key] = season_stats.get(season, {
            'profit': 0.0,
            'plays': 0,
            'profitable': False
        })
    
    # Add overall stats
    result['overall'] = {
        'total_profit': round(float(total_profit), 2),
        'total_plays': int(total_plays),
        'win_rate': round(float(win_rate), 1),
        'profitable_seasons': f"{profitable_seasons}/{len(seasons)}"
    }
    
    if notes:
        result['notes'] = notes
    
    return result


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze strategy performance across multiple backtest seasons'
    )
    
    parser.add_argument('--strategy-type', required=True, choices=['2d', '3d'],
                       help='Strategy type: 2d or 3d')
    parser.add_argument('--line-tier', required=True,
                       help='Line tier (e.g., "5-10 (Bench)")')
    parser.add_argument('--spread-bin', required=True,
                       help='Spread bin (e.g., "Pick\'em (-2 to +2)")')
    parser.add_argument('--bet-side', required=True, choices=['UNDER', 'OVER'],
                       help='Bet side: UNDER or OVER')
    parser.add_argument('--strategy-name', default=None,
                       help='Strategy name (optional - auto-generated if not provided)')
    parser.add_argument('--scorer-type', default=None,
                       help='Scorer type for 3D strategies (e.g., "Rim Attacker (≥40.0%)")')
    parser.add_argument('--seasons', nargs='+', default=['2023-24', '2024-25', '2025-26'],
                       help='Seasons to analyze')
    parser.add_argument('--notes', default='',
                       help='Optional notes about the strategy')
    parser.add_argument('--output', default=None,
                       help='Output JSON file path (default: print to stdout)')
    
    args = parser.parse_args()
    
    result = analyze_strategy(
        strategy_type=args.strategy_type,
        line_tier=args.line_tier,
        spread_bin=args.spread_bin,
        bet_side=args.bet_side,
        strategy_name=args.strategy_name,
        seasons=args.seasons,
        scorer_type=args.scorer_type,
        notes=args.notes
    )
    
    print(f"\n{'='*80}")
    print("RESULT JSON")
    print("="*80)
    print(json.dumps(result, indent=2))
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n✅ Saved to: {args.output}")


if __name__ == '__main__':
    main()

