"""
Quick test to check how many unique NBA games we have in line movement data since Christmas.
"""

import boto3
import pandas as pd
from io import BytesIO

# S3 setup
s3 = boto3.client('s3')
bucket = 'betting-line-movement-snapshots'
prefix = 'data/01_input/the-odds-api/nba/line_movement/'

# Load all snapshots
print("Loading line movement snapshots from S3...")
response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
all_data = []

for obj in response.get('Contents', []):
    key = obj['Key']
    if key.endswith('.csv') and 'snapshot_' in key:
        file_obj = s3.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(BytesIO(file_obj['Body'].read()))
        all_data.append(df)
        
print(f"Found {len(all_data)} snapshot files")
df = pd.concat(all_data, ignore_index=True)
df['game_time'] = pd.to_datetime(df['game_time'])
df['game_date'] = df['game_time'].dt.tz_convert('US/Eastern').dt.date

print(f"\n📊 LINE MOVEMENT DATA COVERAGE")
print(f"{'='*60}")
print(f"Total snapshots: {len(df):,}")
print(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")
print(f"Unique games: {df['game_id'].nunique()}")
print(f"Unique bookmakers: {df['bookmaker'].nunique()}")

# Calculate movements
print(f"\n📈 CHECKING MOVEMENTS")
print(f"{'='*60}")
print(f"Columns: {df.columns.tolist()}")

# Check for spread columns
spread_cols = [col for col in df.columns if 'spread' in col.lower()]
print(f"Spread columns: {spread_cols}")

if 'away_spread' in df.columns:
    game_movements = df.groupby(['game_id', 'bookmaker']).agg({
        'away_spread': lambda x: abs(x.max() - x.min())
    }).reset_index()
    game_movements.columns = ['game_id', 'bookmaker', 'movement']
    
    games_with_2plus = game_movements[game_movements['movement'] >= 2.0]['game_id'].nunique()
    games_with_3plus = game_movements[game_movements['movement'] >= 3.0]['game_id'].nunique()
    games_with_4plus = game_movements[game_movements['movement'] >= 4.0]['game_id'].nunique()
    
    print(f"Games with 2+ point movement: {games_with_2plus}")
    print(f"Games with 3+ point movement: {games_with_3plus}")
    print(f"Games with 4+ point movement: {games_with_4plus}")

print(f"\n📅 Games per date:")
print(df.groupby('game_date')['game_id'].nunique().sort_index())

