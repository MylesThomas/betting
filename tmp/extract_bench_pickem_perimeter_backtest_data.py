"""
Extract backtest data for bench_pickem_perimeter_under strategy

This script:
1. Loads backtest plays for all 3 seasons (2023-24, 2024-25, 2025-26)
2. Filters to the specific strategy: 5-10 (Bench) | Pick'em (-2 to +2) | UNDER | Perimeter (<40.0%)
3. Calculates per-season and overall performance
4. Outputs JSON with the exact format needed for the v2 config

Strategy definition:
- line_tier: "5-10 (Bench)"
- spread_bin: "Pick'em (-2 to +2)"
- bet_side: "UNDER"
- scorer_type: "Perimeter (<40.0%)"

Usage:
cd betting
python tmp/extract_bench_pickem_perimeter_backtest_data.py
"""

import pandas as pd
import boto3
import json
from io import StringIO

# Setup
s3_client = boto3.client('s3')
bucket = 'nba-betting-mt'

# Strategy criteria
STRATEGY_CRITERIA = {
    'line_tier': '5-10 (Bench)',
    'spread_bin': 'Pick\'em (-2 to +2)',
    'bet_side': 'UNDER',
    'scorer_type': 'Perimeter (<40.0%)'
}

print("="*80)
print("EXTRACTING BACKTEST DATA: bench_pickem_perimeter_under")
print("="*80)
print(f"\nStrategy Criteria:")
for key, value in STRATEGY_CRITERIA.items():
    print(f"  {key}: {value}")

# Load 3D backtest plays for all seasons
all_plays = []
seasons = ['2023-24', '2024-25', '2025-26']

print(f"\n{'='*80}")
print("LOADING BACKTEST PLAYS")
print("="*80)

for season in seasons:
    s3_key = f'data/04_output/backtests/3d/{season}/plays.csv'
    
    try:
        response = s3_client.get_object(Bucket=bucket, Key=s3_key)
        df = pd.read_csv(StringIO(response['Body'].read().decode('utf-8')))
        df['season'] = season
        all_plays.append(df)
        print(f"✅ Loaded {len(df):,} plays from {season}")
    except Exception as e:
        print(f"❌ Failed to load {season}: {e}")

if not all_plays:
    raise ValueError("No backtest plays loaded!")

df_all = pd.concat(all_plays, ignore_index=True)
print(f"\n📊 Total plays loaded: {len(df_all):,}")

# Filter to bench_pickem_perimeter_under strategy
print(f"\n{'='*80}")
print("FILTERING TO STRATEGY")
print("="*80)

strategy_plays = df_all[
    (df_all['line_tier'] == STRATEGY_CRITERIA['line_tier']) &
    (df_all['spread_bin'] == STRATEGY_CRITERIA['spread_bin']) &
    (df_all['bet_side'] == STRATEGY_CRITERIA['bet_side']) &
    (df_all['scorer_type'] == STRATEGY_CRITERIA['scorer_type'])
].copy()

print(f"✅ Filtered to {len(strategy_plays):,} plays for bench_pickem_perimeter_under")

if len(strategy_plays) == 0:
    print("\n⚠️  NO PLAYS FOUND FOR THIS STRATEGY!")
    print("This might be expected if:")
    print("  - The strategy is very specific and rare")
    print("  - The scorer_type filter is too restrictive")
    print("  - The data doesn't have this combination")
    
    # Show what combinations DO exist
    print(f"\n📊 Available combinations for 5-10 (Bench) + Pick'em:")
    bench_pickem = df_all[
        (df_all['line_tier'] == STRATEGY_CRITERIA['line_tier']) &
        (df_all['spread_bin'] == STRATEGY_CRITERIA['spread_bin']) &
        (df_all['bet_side'] == STRATEGY_CRITERIA['bet_side'])
    ]
    
    if len(bench_pickem) > 0:
        scorer_counts = bench_pickem['scorer_type'].value_counts()
        print(scorer_counts)
    
    exit(0)

# Calculate per-season stats
print(f"\n{'='*80}")
print("CALCULATING PER-SEASON PERFORMANCE")
print("="*80)

season_stats = {}
for season in seasons:
    season_plays = strategy_plays[strategy_plays['season'] == season]
    
    if len(season_plays) > 0:
        wins = (season_plays['result'] == 'WIN').sum()
        losses = (season_plays['result'] == 'LOSS').sum()
        pushes = (season_plays['result'] == 'PUSH').sum()
        total_profit = season_plays['profit'].sum()
        win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
        
        season_stats[season] = {
            'profit': round(total_profit, 2),
            'plays': len(season_plays),
            'wins': int(wins),
            'losses': int(losses),
            'pushes': int(pushes),
            'win_rate': round(win_rate, 1),
            'profitable': total_profit > 0
        }
        
        print(f"\n{season}:")
        print(f"  Plays: {len(season_plays)}")
        print(f"  W-L-P: {wins}-{losses}-{pushes}")
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  Profit: ${total_profit:,.2f}")
        print(f"  Status: {'✅ Profitable' if total_profit > 0 else '❌ Losing'}")
    else:
        season_stats[season] = {
            'profit': 0.0,
            'plays': 0,
            'wins': 0,
            'losses': 0,
            'pushes': 0,
            'win_rate': 0.0,
            'profitable': False
        }
        print(f"\n{season}:")
        print(f"  ⚠️  No plays")

# Calculate overall stats
print(f"\n{'='*80}")
print("OVERALL PERFORMANCE")
print("="*80)

total_plays = len(strategy_plays)
total_wins = (strategy_plays['result'] == 'WIN').sum()
total_losses = (strategy_plays['result'] == 'LOSS').sum()
total_pushes = (strategy_plays['result'] == 'PUSH').sum()
total_profit = strategy_plays['profit'].sum()
overall_win_rate = (total_wins / (total_wins + total_losses) * 100) if (total_wins + total_losses) > 0 else 0
profitable_seasons = sum(1 for s in season_stats.values() if s['profitable'])

print(f"\nTotal Plays: {total_plays}")
print(f"W-L-P: {total_wins}-{total_losses}-{total_pushes}")
print(f"Win Rate: {overall_win_rate:.1f}%")
print(f"Total Profit: ${total_profit:,.2f}")
print(f"Profitable Seasons: {profitable_seasons}/3")

# Load training ROI from 2025-26 training strategies
print(f"\n{'='*80}")
print("LOADING TRAINING ROI")
print("="*80)

try:
    response = s3_client.get_object(
        Bucket=bucket, 
        Key='data/03_intermediate/points_by_role_gamespread_6feet_strategies_2025-26_rim40.json'
    )
    training_data = json.loads(response['Body'].read().decode('utf-8'))
    
    if 'strategies' in training_data:
        strategies = training_data['strategies']
    else:
        strategies = training_data
    
    if isinstance(strategies, dict):
        strategies = list(strategies.values())
    
    # Find matching strategy
    training_roi = None
    for strat in strategies:
        if (strat.get('line_tier') == STRATEGY_CRITERIA['line_tier'] and
            strat.get('spread_bin') == STRATEGY_CRITERIA['spread_bin'] and
            strat.get('bet_side') == STRATEGY_CRITERIA['bet_side'] and
            strat.get('scorer_type') == STRATEGY_CRITERIA['scorer_type']):
            training_roi = strat.get('roi', 0.0)
            break
    
    if training_roi is not None:
        print(f"✅ Found training ROI: {training_roi:+.1f}%")
    else:
        print(f"⚠️  Strategy not found in training data, using 0.0%")
        training_roi = 0.0
        
except Exception as e:
    print(f"⚠️  Could not load training data: {e}")
    training_roi = 0.0

# Build final JSON output
output = {
    "strategy_name": "bench_pickem_perimeter_under",
    "strategy_type": "3d",
    "line_tier": STRATEGY_CRITERIA['line_tier'],
    "spread_bin": STRATEGY_CRITERIA['spread_bin'],
    "bet_side": STRATEGY_CRITERIA['bet_side'],
    "scorer_type": STRATEGY_CRITERIA['scorer_type'],
    "training_roi": round(training_roi, 1),
    "backtest_2023_24": {
        "profit": season_stats['2023-24']['profit'],
        "plays": season_stats['2023-24']['plays'],
        "profitable": season_stats['2023-24']['profitable']
    },
    "backtest_2024_25": {
        "profit": season_stats['2024-25']['profit'],
        "plays": season_stats['2024-25']['plays'],
        "profitable": season_stats['2024-25']['profitable']
    },
    "backtest_2025_26": {
        "profit": season_stats['2025-26']['profit'],
        "plays": season_stats['2025-26']['plays'],
        "profitable": season_stats['2025-26']['profitable']
    },
    "overall": {
        "total_profit": round(total_profit, 2),
        "total_plays": total_plays,
        "win_rate": round(overall_win_rate, 1),
        "profitable_seasons": f"{profitable_seasons}/3"
    },
    "notes": "Bench pick'em perimeter players in close games. Hypothesis: Similar to rim attackers, bench perimeter players get reduced usage in close games."
}

print(f"\n{'='*80}")
print("FINAL OUTPUT (JSON)")
print("="*80)
print(json.dumps(output, indent=2))

# Save to file
output_file = '/tmp/bench_pickem_perimeter_under_backtest_data.json'
with open(output_file, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n✅ Saved to: {output_file}")
print("\nYou can now copy this data into the v2 config file!")

