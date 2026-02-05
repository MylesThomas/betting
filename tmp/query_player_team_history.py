"""
Quick query of player_team_history.parquet using DuckDB.

Downloads from S3 first, then queries locally. Shows players with most team changes.

Usage:
    python tmp/query_player_team_history.py
"""

import duckdb
import pandas as pd
from pathlib import Path
import boto3

# S3 configuration
S3_BUCKET = 'nba-betting-mt'
S3_KEY = 'data/02_cache/player_team_history.parquet'

# Local path
LOCAL_PATH = Path.home() / 'Downloads' / 'tmp' / 'player_team_history.parquet'

# Step 1: Download from S3 to local
print(f"📥 Downloading from S3: s3://{S3_BUCKET}/{S3_KEY}")
LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)

s3_client = boto3.client('s3')
s3_client.download_file(S3_BUCKET, S3_KEY, str(LOCAL_PATH))
print(f"✅ Saved to: {LOCAL_PATH}\n")

# Step 2: Query local file with DuckDB
con = duckdb.connect(':memory:')

query = f"""
WITH player_stint_counts AS (
    SELECT
        player_normalized,
        COUNT(*) as num_stints,
        COUNT(CASE WHEN valid_to IS NULL THEN 1 END) as is_active,
        MIN(valid_from) as first_date,
        MAX(COALESCE(valid_to, CURRENT_DATE)) as last_date,
        LIST(team ORDER BY valid_from) as teams
    FROM '{str(LOCAL_PATH)}'
    GROUP BY player_normalized
)

SELECT
    player_normalized as player,
    num_stints as stints,
    is_active as active,
    teams,
    first_date as first,
    last_date as last,
    DATEDIFF('year', first_date::DATE, last_date::DATE) as years
FROM player_stint_counts
WHERE num_stints >= 2
ORDER BY num_stints DESC, years DESC
LIMIT 30
"""

df = con.execute(query).df()
con.close()

# Format output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', 100)

print("\n" + "="*120)
print("🏀 PLAYERS WITH MOST TEAM CHANGES")
print("="*120)
print(f"\nSource: {LOCAL_PATH}")
print(f"Showing top 30 players with 2+ team stints\n")
print(df.to_string(index=True))
print("\n" + "="*120)
