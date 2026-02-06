-- Step 1: Get game-level stats per player (total points, total minutes)
-- Input: ~/dev/betting/data/minute_by_minute.parquet
-- Output: Temp table 'game_level_stats'

CREATE OR REPLACE TEMP TABLE game_level_stats AS
SELECT 
    game_id,
    game_date,
    player_id,
    player_name,
    MAX(playing_seconds) / 60.0 AS total_minutes,
    MAX(cumulative_points) AS total_points
FROM '~/dev/betting/data/minute_by_minute.parquet'
GROUP BY game_id, game_date, player_id, player_name;

-- Add PPM
CREATE OR REPLACE TEMP TABLE game_stats_with_ppm AS
SELECT 
    *,
    CASE 
        WHEN total_minutes > 0 THEN total_points / total_minutes 
        ELSE 0 
    END AS points_per_minute
FROM game_level_stats;
