"""
Debug 2023-24 season, +4 spread games to see why P(win | cover) = 100%.

This should NOT be 100% - at +4, you can cover by losing by 1-3 points.
"""

import pandas as pd
from pathlib import Path

# Load cached 2023-24 data
cache_dir = Path.home() / 'Downloads' / 'tmp'
merged_df = pd.read_parquet(cache_dir / 'nba_2023-24_merged.parquet')

print("=" * 80)
print("DEBUGGING 2023-24 SEASON: +4 SPREAD GAMES")
print("=" * 80)

# Find all underdog games with spreads that floor to 4 (i.e., 4.0 to 4.99)
rows = []
for _, row in merged_df.iterrows():
    away_ml = row['away_ml_odds']
    home_ml = row['home_ml_odds']
    
    # Away team is underdog with spread flooring to 4
    if away_ml > 0 and 4.0 <= abs(row['away_spread']) < 5.0:
        spread_raw = abs(row['away_spread'])
        spread_floored = int(spread_raw)
        covered = (row['AWAY_SCORE'] + row['away_spread']) > row['HOME_SCORE']
        won = row['AWAY_WL'] == 'W'
        margin = row['AWAY_SCORE'] - row['HOME_SCORE']
        
        rows.append({
            'underdog': row['AWAY_TEAM'],
            'opponent': row['HOME_TEAM'],
            'location': '@',
            'date': row['GAME_DATE'],
            'spread_raw': spread_raw,
            'spread_floored': spread_floored,
            'away_score': row['AWAY_SCORE'],
            'home_score': row['HOME_SCORE'],
            'margin': margin,
            'covered': covered,
            'won': won,
        })
    
    # Home team is underdog with spread flooring to 4
    if home_ml > 0 and 4.0 <= abs(row['home_spread']) < 5.0:
        spread_raw = abs(row['home_spread'])
        spread_floored = int(spread_raw)
        covered = (row['HOME_SCORE'] + row['home_spread']) > row['AWAY_SCORE']
        won = row['HOME_WL'] == 'W'
        margin = row['HOME_SCORE'] - row['AWAY_SCORE']
        
        rows.append({
            'underdog': row['HOME_TEAM'],
            'opponent': row['AWAY_TEAM'],
            'location': 'vs',
            'date': row['GAME_DATE'],
            'spread_raw': spread_raw,
            'spread_floored': spread_floored,
            'away_score': row['AWAY_SCORE'],
            'home_score': row['HOME_SCORE'],
            'margin': margin,
            'covered': covered,
            'won': won,
        })

df = pd.DataFrame(rows)

print(f"\nTotal games with spread 4.0-4.99: {len(df)}")
print(f"Games where dog covered: {df['covered'].sum()}")
print(f"Games where dog won: {df['won'].sum()}")
print(f"Games where dog covered AND won: {((df['covered']) & (df['won'])).sum()}")
print(f"Games where dog covered but LOST: {((df['covered']) & (~df['won'])).sum()}")

print("\n" + "=" * 80)
print("ALL GAMES (sorted by spread_raw)")
print("=" * 80)
print(df[['underdog', 'location', 'opponent', 'spread_raw', 'spread_floored', 'margin', 
          'away_score', 'home_score', 'covered', 'won']].sort_values('spread_raw').to_string(index=False))

print("\n" + "=" * 80)
print("GAMES WHERE DOG COVERED BUT LOST (should exist!)")
print("=" * 80)
covered_lost = df[(df['covered']) & (~df['won'])]
if len(covered_lost) > 0:
    print(covered_lost[['underdog', 'location', 'opponent', 'spread_raw', 'margin', 
                        'away_score', 'home_score']].to_string(index=False))
else:
    print("❌ NO GAMES FOUND - This is the problem!")
    print("\nLet's check if these are all +4.5 games (which would explain it)...")
    
    # Check the distribution of raw spreads
    print("\nDistribution of raw spreads in this bin:")
    print(df['spread_raw'].value_counts().sort_index())
    
    print("\nIf all games are +4.5, then 100% makes sense because:")
    print("  - At +4.5, to cover you need to lose by ≤4")
    print("  - That means win, lose by 1, lose by 2, lose by 3, or lose by 4")
    print("  - But in this sample, every time they covered, they won outright")
