-- Step 3: Build player profile (1 row per player with all distributions)
-- Input: Temp tables from 01 and 02
-- Output: Table 'player_profiles'

CREATE OR REPLACE TABLE player_profiles AS
SELECT 
    g.player_id,
    g.player_name,
    COUNT(*) AS num_games,
    
    -- Summary stats
    AVG(g.total_points) AS avg_points_per_game,
    AVG(g.total_minutes) AS avg_minutes_per_game,
    AVG(g.points_per_minute) AS avg_ppm,
    STDDEV(g.total_points) AS std_points,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY g.total_points) AS p25_points,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY g.total_points) AS p50_points,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY g.total_points) AS p75_points,
    
    -- Full game distributions (ordered by most recent first)
    LIST(g.total_points ORDER BY g.game_date DESC) AS total_points_history,
    LIST(g.total_minutes ORDER BY g.game_date DESC) AS total_minutes_history,
    LIST(g.points_per_minute ORDER BY g.game_date DESC) AS points_per_minute_history,
    
    -- Q1 distributions
    LIST(q.q1_points ORDER BY q.game_date DESC) AS q1_points_history,
    LIST(q.q1_minutes ORDER BY q.game_date DESC) AS q1_minutes_history,
    LIST(q.q1_ppm ORDER BY q.game_date DESC) AS q1_ppm_history,
    
    -- Q2 distributions
    LIST(q.q2_points ORDER BY q.game_date DESC) AS q2_points_history,
    LIST(q.q2_minutes ORDER BY q.game_date DESC) AS q2_minutes_history,
    LIST(q.q2_ppm ORDER BY q.game_date DESC) AS q2_ppm_history,
    
    -- Q3 distributions
    LIST(q.q3_points ORDER BY q.game_date DESC) AS q3_points_history,
    LIST(q.q3_minutes ORDER BY q.game_date DESC) AS q3_minutes_history,
    LIST(q.q3_ppm ORDER BY q.game_date DESC) AS q3_ppm_history,
    
    -- Q4 distributions
    LIST(q.q4_points ORDER BY q.game_date DESC) AS q4_points_history,
    LIST(q.q4_minutes ORDER BY q.game_date DESC) AS q4_minutes_history,
    LIST(q.q4_ppm ORDER BY q.game_date DESC) AS q4_ppm_history
    
FROM game_stats_with_ppm g
LEFT JOIN quarter_splits_with_ppm q 
    ON g.game_id = q.game_id 
    AND g.player_id = q.player_id
GROUP BY g.player_id, g.player_name
HAVING COUNT(*) >= 10  -- Only players with 10+ games
ORDER BY avg_points_per_game DESC;
