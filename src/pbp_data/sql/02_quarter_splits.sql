-- Step 2: Get quarterly splits (Q1, Q2, Q3, Q4)
-- Q1 = minutes 0-11, Q2 = 12-23, Q3 = 24-35, Q4 = 36-47, OT = 48+
-- Input: ~/dev/betting/data/minute_by_minute.parquet
-- Output: Temp table 'quarter_splits_with_ppm'

CREATE OR REPLACE TEMP TABLE quarter_splits AS
SELECT 
    game_id,
    game_date,
    player_id,
    player_name,
    
    -- Q1 (minutes 0-11)
    MAX(CASE WHEN minute <= 11 THEN playing_seconds ELSE 0 END) / 60.0 AS q1_minutes,
    MAX(CASE WHEN minute <= 11 THEN cumulative_points ELSE 0 END) AS q1_points,
    
    -- Q2 (minutes 12-23)
    (MAX(CASE WHEN minute <= 23 THEN playing_seconds ELSE 0 END) - 
     MAX(CASE WHEN minute <= 11 THEN playing_seconds ELSE 0 END)) / 60.0 AS q2_minutes,
    (MAX(CASE WHEN minute <= 23 THEN cumulative_points ELSE 0 END) - 
     MAX(CASE WHEN minute <= 11 THEN cumulative_points ELSE 0 END)) AS q2_points,
    
    -- Q3 (minutes 24-35)
    (MAX(CASE WHEN minute <= 35 THEN playing_seconds ELSE 0 END) - 
     MAX(CASE WHEN minute <= 23 THEN playing_seconds ELSE 0 END)) / 60.0 AS q3_minutes,
    (MAX(CASE WHEN minute <= 35 THEN cumulative_points ELSE 0 END) - 
     MAX(CASE WHEN minute <= 23 THEN cumulative_points ELSE 0 END)) AS q3_points,
    
    -- Q4 (minutes 36-47)
    (MAX(CASE WHEN minute <= 47 THEN playing_seconds ELSE 0 END) - 
     MAX(CASE WHEN minute <= 35 THEN playing_seconds ELSE 0 END)) / 60.0 AS q4_minutes,
    (MAX(CASE WHEN minute <= 47 THEN cumulative_points ELSE 0 END) - 
     MAX(CASE WHEN minute <= 35 THEN cumulative_points ELSE 0 END)) AS q4_points
    
FROM '~/dev/betting/data/minute_by_minute.parquet'
GROUP BY game_id, game_date, player_id, player_name;

-- Add PPM for each quarter
CREATE OR REPLACE TEMP TABLE quarter_splits_with_ppm AS
SELECT 
    *,
    CASE WHEN q1_minutes > 0 THEN q1_points / q1_minutes ELSE 0 END AS q1_ppm,
    CASE WHEN q2_minutes > 0 THEN q2_points / q2_minutes ELSE 0 END AS q2_ppm,
    CASE WHEN q3_minutes > 0 THEN q3_points / q3_minutes ELSE 0 END AS q3_ppm,
    CASE WHEN q4_minutes > 0 THEN q4_points / q4_minutes ELSE 0 END AS q4_ppm
FROM quarter_splits;
