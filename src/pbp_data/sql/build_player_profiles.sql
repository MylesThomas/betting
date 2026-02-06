-- Complete pipeline to build player profiles for Monte Carlo simulation
-- Run all steps in sequence
-- Usage: duckdb < src/pbp_data/sql/build_player_profiles.sql

-- Step 1: Game-level stats
.read src/pbp_data/sql/01_game_level_stats.sql

-- Step 2: Quarterly splits
.read src/pbp_data/sql/02_quarter_splits.sql

-- Step 3: Build player profiles
.read src/pbp_data/sql/03_player_profiles.sql

-- Output results
.mode box
.header on

-- Show top 10 players
SELECT 
    player_name,
    num_games,
    ROUND(avg_points_per_game, 1) AS avg_pts,
    ROUND(avg_minutes_per_game, 1) AS avg_mins,
    ROUND(avg_ppm, 3) AS avg_ppm,
    ROUND(std_points, 1) AS std_pts,
    p25_points,
    p50_points,
    p75_points
FROM player_profiles
LIMIT 10;
