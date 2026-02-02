"""
Quick check of join quality using cached 2025-26 data

Author: Thomas Myles
Date: 2026-01-30
"""

import pandas as pd
from pathlib import Path

SEASON = '2025-26'
cache_path = Path.home() / 'Downloads' / 'tmp' / f'nba_{SEASON}_merged.parquet'

print(f"\n{'='*70}")
print(f"🔍 QUICK JOIN QUALITY CHECK - {SEASON}")
print(f"{'='*70}")

if not cache_path.exists():
    print(f"❌ No cached data found at {cache_path}")
    print("Run the optimizer script first to generate cache.")
    exit(1)

# Load cached merged data
df = pd.read_parquet(cache_path)

print(f"\n✅ Loaded cached data: {len(df)} games")
print(f"   Date range: {df['GAME_DATE'].min()} to {df['GAME_DATE'].max()}")

# Expected games for partial season
days_in_season = (pd.to_datetime(df['GAME_DATE'].max()) - pd.to_datetime(df['GAME_DATE'].min())).days
expected_games = (days_in_season / 180) * 1230  # 1230 games over ~180 days

print(f"\n📊 Coverage:")
print(f"   Days in season: {days_in_season}")
print(f"   Expected games (estimate): ~{expected_games:.0f}")
print(f"   Actual games: {len(df)}")
print(f"   Coverage: {len(df)/expected_games*100:.1f}%")

# Check for duplicates
dup_check = df.groupby(['GAME_DATE', 'AWAY_TEAM', 'HOME_TEAM']).size()
dups = dup_check[dup_check > 1]

if len(dups) > 0:
    print(f"\n⚠️  Found {len(dups)} duplicate games!")
    print(dups.head(10))
else:
    print(f"\n✅ No duplicate games")

# Show sample
print(f"\n📋 Sample games:")
print(df[['GAME_DATE', 'AWAY_TEAM', 'HOME_TEAM', 'AWAY_SCORE', 'HOME_SCORE',
          'away_spread', 'away_ml_odds']].head(10).to_string(index=False))

print(f"\n{'='*70}")
print("For deeper analysis, we need to load raw lines data from S3")
print("(which takes ~5-10 minutes due to pagination)")
print(f"{'='*70}\n")
