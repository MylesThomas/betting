"""
Figure out why we have date/time mismatches between ESPN and Odds API

Goal: Understand why only 115/711 games match

Author: Thomas Myles
Date: 2026-01-30
"""

import pandas as pd
from pathlib import Path
import sys

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.s3_utils import read_df_from_s3, list_s3_files

print("="*70)
print("INVESTIGATING DATE MISMATCHES - 2025-26 SEASON")
print("="*70)

# Load game results
print("\n📊 Loading game results...")
bucket = 'nba-betting-mt'
prefix = 'data/01_input/historical_game_results/'

files = list_s3_files(bucket, prefix)
csv_files = [f for f in files if f.endswith('.csv')]

# Filter to 2025-26 season
season_files = []
for s3_key in csv_files:
    filename = s3_key.split('/')[-1]
    file_date = filename.replace('.csv', '')
    
    if '2025-10-01' <= file_date <= '2026-06-30':
        season_files.append((s3_key, file_date))

print(f"Found {len(season_files)} result files for 2025-26")

# Load all results
all_results = []
for s3_key, file_date in season_files[:10]:  # Just load first 10 for speed
    df = read_df_from_s3(bucket, s3_key)
    df['file_date'] = file_date
    all_results.append(df)

results_df = pd.concat(all_results, ignore_index=True)
results_df['GAME_DATE'] = pd.to_datetime(results_df['GAME_DATE']).dt.date

print(f"Loaded {len(results_df)} game results from first 10 files")
print(f"Date range: {results_df['GAME_DATE'].min()} to {results_df['GAME_DATE'].max()}")

# Count unique games
results_games = results_df.groupby(['GAME_DATE', 'AWAY_TEAM', 'HOME_TEAM']).size()
print(f"Unique games in results: {len(results_games)}")

# Load consensus lines for same period
print("\n📈 Loading game lines...")
lines_bucket = 'the-odds-api-mt'
lines_prefix = 'nba/historical_game_lines/2025-26/'

lines_files = list_s3_files(lines_bucket, lines_prefix)
lines_csv = [f for f in lines_files if f.endswith('.csv')]

print(f"Found {len(lines_csv)} line files")

# Load first 10 line files
all_lines = []
for s3_key in lines_csv[:10]:
    df = read_df_from_s3(lines_bucket, s3_key)
    all_lines.append(df)

lines_df = pd.concat(all_lines, ignore_index=True)
lines_df['game_time'] = pd.to_datetime(lines_df['game_time'])
lines_df['game_date'] = lines_df['game_time'].dt.date

print(f"Loaded {len(lines_df):,} line records from first 10 files")
print(f"Date range: {lines_df['game_date'].min()} to {lines_df['game_date'].max()}")

# Calculate consensus
spreads = lines_df[lines_df['market'] == 'spread'].groupby(
    ['game_date', 'away_team', 'home_team']
).size().reset_index()[['game_date', 'away_team', 'home_team']]

ml = lines_df[lines_df['market'] == 'moneyline'].groupby(
    ['game_date', 'away_team', 'home_team']  
).size().reset_index()[['game_date', 'away_team', 'home_team']]

lines_both = spreads.merge(ml, on=['game_date', 'away_team', 'home_team'])
print(f"Games with both spread & ML: {len(lines_both)}")

# Try to join
matched = results_df.merge(
    lines_both,
    left_on=['GAME_DATE', 'AWAY_TEAM', 'HOME_TEAM'],
    right_on=['game_date', 'away_team', 'home_team'],
    how='inner'
)

print(f"\n{'='*70}")
print(f"MATCH RESULTS")
print(f"{'='*70}")
print(f"Results: {len(results_games)} unique games")
print(f"Lines: {len(lines_both)} games with both spread & ML")
print(f"Matched: {len(matched.groupby(['GAME_DATE', 'AWAY_TEAM', 'HOME_TEAM']).size())} games")
print(f"Match rate: {len(matched)/len(results_games)*100:.1f}%")

# Show unmatched games
print(f"\n🔍 Sample unmatched games from results:")
results_keys = set(zip(results_df['GAME_DATE'], results_df['AWAY_TEAM'], results_df['HOME_TEAM']))
lines_keys = set(zip(lines_both['game_date'], lines_both['away_team'], lines_both['home_team']))

unmatched_results = results_keys - lines_keys
print(f"\nGames in results but NOT in lines ({len(unmatched_results)} total):")
for i, (date, away, home) in enumerate(sorted(unmatched_results)[:5]):
    print(f"  {date}: {away} @ {home}")

unmatched_lines = lines_keys - results_keys  
print(f"\nGames in lines but NOT in results ({len(unmatched_lines)} total):")
for i, (date, away, home) in enumerate(sorted(unmatched_lines)[:5]):
    print(f"  {date}: {away} @ {home}")
