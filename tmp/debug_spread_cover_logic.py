"""
Debug script to verify spread cover logic for +1.5 spread games.

Check if the 100% win rate when covering is real or a bug.
"""

import pandas as pd
from pathlib import Path

# Load cached 2024-25 data
cache_dir = Path.home() / 'Downloads' / 'tmp'
merged_df = pd.read_parquet(cache_dir / 'nba_2024-25_merged.parquet')

print("=" * 60)
print("DEBUGGING +4.5 SPREAD GAMES (2024-25)")
print("=" * 60)

# Find all underdog games with ~1.5 spread
rows = []
for _, row in merged_df.iterrows():
    away_ml = row['away_ml_odds']
    home_ml = row['home_ml_odds']
    
    # Away team is underdog with ~4.5 spread
    if away_ml > 0 and 4.0 <= abs(row['away_spread']) <= 5.0:
        covered = (row['AWAY_SCORE'] + row['away_spread']) > row['HOME_SCORE']
        won = row['AWAY_WL'] == 'W'
        margin = row['AWAY_SCORE'] - row['HOME_SCORE']  # AWAY team's margin
        adjusted_score = row['AWAY_SCORE'] + row['away_spread']
        
        rows.append({
            'underdog': row['AWAY_TEAM'],
            'opponent': row['HOME_TEAM'],
            'location': '@',
            'date': row['GAME_DATE'],
            'spread': row['away_spread'],
            'dog_score': row['AWAY_SCORE'],
            'opp_score': row['HOME_SCORE'],
            'adjusted_score': adjusted_score,
            'margin': margin,
            'covered': covered,
            'won': won,
        })
    
    # Home team is underdog with ~4.5 spread
    if home_ml > 0 and 4.0 <= abs(row['home_spread']) <= 5.0:
        covered = (row['HOME_SCORE'] + row['home_spread']) > row['AWAY_SCORE']
        won = row['HOME_WL'] == 'W'
        margin = row['HOME_SCORE'] - row['AWAY_SCORE']  # HOME team's margin
        adjusted_score = row['HOME_SCORE'] + row['home_spread']
        
        rows.append({
            'underdog': row['HOME_TEAM'],
            'opponent': row['AWAY_TEAM'],
            'location': 'vs',
            'date': row['GAME_DATE'],
            'spread': row['home_spread'],
            'dog_score': row['HOME_SCORE'],
            'opp_score': row['AWAY_SCORE'],
            'adjusted_score': adjusted_score,
            'margin': margin,
            'covered': covered,
            'won': won,
        })

df = pd.DataFrame(rows)

print(f"\nTotal +1.5 spread games: {len(df)}")
print(f"  Min spread value: {df['spread'].min()}")
print(f"  Max spread value: {df['spread'].max()}")
print(f"Games where dog covered: {df['covered'].sum()}")
print(f"Games where dog won: {df['won'].sum()}")
print(f"Games where dog covered AND won: {((df['covered']) & (df['won'])).sum()}")
print(f"Games where dog covered but LOST: {((df['covered']) & (~df['won'])).sum()}")

print("\n" + "=" * 60)
print("ALL GAMES (showing adjusted score logic)")
print("=" * 60)
print(df[['underdog', 'location', 'opponent', 'spread', 'margin', 'dog_score', 'opp_score', 'adjusted_score', 'covered', 'won']].to_string(index=False))

print("\n" + "=" * 60)
print("SAMPLE: Games where dog COVERED but LOST")
print("=" * 60)
covered_lost = df[(df['covered']) & (~df['won'])]
if len(covered_lost) > 0:
    print(covered_lost[['underdog', 'location', 'opponent', 'spread', 'margin', 'dog_score', 'opp_score', 'adjusted_score']].to_string(index=False))
else:
    print("NO GAMES FOUND - this explains the 100%!")

print("\n" + "=" * 60)
print("SAMPLE: First 10 games where dog covered AND won")
print("=" * 60)
covered_won = df[(df['covered']) & (df['won'])]
print(covered_won[['underdog', 'location', 'opponent', 'spread', 'margin', 'dog_score', 'opp_score', 'adjusted_score']].head(10).to_string(index=False))
