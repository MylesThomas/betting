"""
Analyze players who have been on the most teams.

Downloads player_team_history.parquet from S3 and queries locally.

Usage:
    python tmp/analyze_player_team_changes.py
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

# Filter config
MIN_TEAM_STINTS = 2  # Only show players with at least this many team stints
TOP_N_PLAYERS = 50   # Show top N players by team changes

# Step 1: Download from S3 to local
print(f"📥 Downloading from S3: s3://{S3_BUCKET}/{S3_KEY}")
LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)

s3_client = boto3.client('s3')
s3_client.download_file(S3_BUCKET, S3_KEY, str(LOCAL_PATH))
print(f"✅ Saved to: {LOCAL_PATH}\n")

# Step 2: Query local file with DuckDB
con = duckdb.connect(':memory:')

query = f"""
WITH player_stints AS (
    SELECT
        player_normalized,
        team,
        valid_from,
        valid_to,
        CASE 
            WHEN valid_to IS NULL THEN 'Current'
            ELSE valid_to::VARCHAR
        END as valid_to_display,
        -- Calculate days with team
        CASE
            WHEN valid_to IS NULL THEN DATEDIFF('day', valid_from::DATE, CURRENT_DATE)
            ELSE DATEDIFF('day', valid_from::DATE, valid_to::DATE)
        END as days_with_team
    FROM '{str(LOCAL_PATH)}'
),

player_summary AS (
    SELECT
        player_normalized,
        COUNT(*) as num_teams,
        COUNT(CASE WHEN valid_to_display = 'Current' THEN 1 END) as currently_active,
        MIN(valid_from) as first_game_date,
        MAX(COALESCE(valid_to, CURRENT_DATE)) as last_game_date,
        SUM(days_with_team) as total_days_tracked,
        LIST(team ORDER BY valid_from) as teams_chronological,
        LIST(valid_from::VARCHAR ORDER BY valid_from) as start_dates
    FROM player_stints
    GROUP BY player_normalized
    HAVING COUNT(*) >= {MIN_TEAM_STINTS}
),

ranked_players AS (
    SELECT
        player_normalized,
        num_teams,
        currently_active,
        first_game_date,
        last_game_date,
        total_days_tracked,
        teams_chronological,
        start_dates,
        DATEDIFF('year', first_game_date::DATE, last_game_date::DATE) as career_years,
        ROUND(num_teams::FLOAT / DATEDIFF('year', first_game_date::DATE, last_game_date::DATE), 2) as teams_per_year
    FROM player_summary
)

SELECT
    player_normalized as player,
    num_teams as teams,
    currently_active as active,
    career_years as yrs,
    teams_per_year as tm_per_yr,
    teams_chronological as team_history,
    first_game_date as first_game,
    last_game_date as last_game
FROM ranked_players
ORDER BY num_teams DESC, career_years DESC
LIMIT {TOP_N_PLAYERS}
"""

print("="*100)
print("🏀 PLAYERS WITH MOST TEAM CHANGES")
print("="*100)
print(f"\nQuerying: {LOCAL_PATH}")
print(f"Showing top {TOP_N_PLAYERS} players with {MIN_TEAM_STINTS}+ team stints")
print()

df = con.execute(query).df()
con.close()

# Display results
if df.empty:
    print("No data found!")
else:
    print(f"Found {len(df)} players with multiple team stints\n")
    
    # Custom display with wrapped team history
    for idx, row in df.iterrows():
        print(f"{idx+1}. {row['player']}")
        print(f"   Teams: {row['teams']} stints | Career: {row['yrs']} years | Rate: {row['tm_per_yr']} teams/year")
        print(f"   Active: {'Yes' if row['active'] > 0 else 'No'}")
        print(f"   History: {' → '.join(row['team_history'])}")
        print(f"   Dates: {row['first_game']} to {row['last_game']}")
        print()

print("="*100)
