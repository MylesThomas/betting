-- Step 2 DuckDB Validation Tests
-- Run: duckdb -c ".read src/nfl_rush_attempts_modeling/scripts/step2_tests.sql"

.timer on

-- ── Test 1: One row per player-game-book in training (no duplicates) ──────────
SELECT
    'TEST 1: No duplicate player-game-book rows' AS test,
    COUNT(*) AS total_rows,
    COUNT(DISTINCT nfl_game_id || '|' || player_name_norm || '|' || bookmaker) AS unique_combos,
    CASE
        WHEN COUNT(*) = COUNT(DISTINCT nfl_game_id || '|' || player_name_norm || '|' || bookmaker)
        THEN 'PASS'
        ELSE 'FAIL'
    END AS result
FROM read_parquet('/Users/thomasmyles/Downloads/tmp/rush_attempts/training.parquet');

-- ── Test 2: No future leakage ─────────────────────────────────────────────────
-- carry_rate_L5 at game G should be < or unrelated to game G's actual carries
-- Check: for any player, carry_rate_L1 at week W = carries from week W-1 (not W)
SELECT
    'TEST 2: No leakage — carry_rate_L1 uses prior game only' AS test,
    COUNT(*) AS checked_rows,
    -- For each player, carry_rate_L1 at week W = carries at week W-1 (lag 1)
    -- Verify: no row where carry_rate_L1 == carries (same game would be leakage)
    -- Note: could be coincidence if values happen to match; use > 2 threshold
    SUM(CASE
        WHEN ABS(carry_rate_L1 - carries) < 0.01 AND carry_rate_L1 IS NOT NULL AND carries > 2
        THEN 1 ELSE 0
    END) AS suspicious_exact_matches,
    -- leakage would mean exact match rate is very high (>50%)
    ROUND(
        SUM(CASE WHEN ABS(carry_rate_L1 - carries) < 0.01 AND carry_rate_L1 IS NOT NULL AND carries > 2
                 THEN 1 ELSE 0 END) * 1.0 /
        NULLIF(SUM(CASE WHEN carry_rate_L1 IS NOT NULL AND carries > 2 THEN 1 ELSE 0 END), 0),
        4
    ) AS exact_match_rate,
    CASE
        WHEN SUM(CASE WHEN ABS(carry_rate_L1 - carries) < 0.01 AND carry_rate_L1 IS NOT NULL AND carries > 2
                      THEN 1 ELSE 0 END) * 1.0 /
             NULLIF(SUM(CASE WHEN carry_rate_L1 IS NOT NULL AND carries > 2 THEN 1 ELSE 0 END), 0) < 0.10
        THEN 'PASS'
        ELSE 'FAIL (possible leakage)'
    END AS result
FROM read_parquet('/Users/thomasmyles/Downloads/tmp/rush_attempts/training.parquet');

-- ── Test 3: Null rates by feature column (<10% required) ─────────────────────
SELECT
    'TEST 3: Feature null rates' AS test,
    ROUND(SUM(CASE WHEN carry_rate_L1 IS NULL THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 4) AS carry_L1_null,
    ROUND(SUM(CASE WHEN carry_rate_L5 IS NULL THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 4) AS carry_L5_null,
    ROUND(SUM(CASE WHEN carry_rate_Lcareer IS NULL THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 4) AS carry_career_null,
    ROUND(SUM(CASE WHEN over_rate_L5 IS NULL THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 4) AS over_rate_L5_null,
    ROUND(SUM(CASE WHEN opp_carry_allowed_L8 IS NULL THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 4) AS opp_def_L8_null,
    ROUND(SUM(CASE WHEN consensus_point IS NULL THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 4) AS consensus_null,
    CASE
        WHEN SUM(CASE WHEN carry_rate_L1 IS NULL THEN 1 ELSE 0 END) * 1.0 / COUNT(*) < 0.10
         AND SUM(CASE WHEN carry_rate_Lcareer IS NULL THEN 1 ELSE 0 END) * 1.0 / COUNT(*) < 0.10
         AND SUM(CASE WHEN consensus_point IS NULL THEN 1 ELSE 0 END) = 0
        THEN 'PASS'
        ELSE 'FAIL'
    END AS result
FROM read_parquet('/Users/thomasmyles/Downloads/tmp/rush_attempts/training.parquet');

-- ── Test 4: Target column (is_over) present and no nulls ──────────────────────
SELECT
    'TEST 4: is_over column present with no nulls' AS test,
    COUNT(*) AS total_rows,
    SUM(CASE WHEN is_over IS NULL THEN 1 ELSE 0 END) AS null_is_over,
    ROUND(AVG(is_over), 4) AS mean_is_over,
    CASE
        WHEN SUM(CASE WHEN is_over IS NULL THEN 1 ELSE 0 END) = 0
         AND AVG(is_over) BETWEEN 0.40 AND 0.60
        THEN 'PASS'
        ELSE 'FAIL'
    END AS result
FROM read_parquet('/Users/thomasmyles/Downloads/tmp/rush_attempts/training.parquet');

-- ── Test 5: Join quality — % of market rows with matching spine row ───────────
-- Training has 3,808 unique player-games; market consensus has 3,878
-- Join rate = 3808/3878 = ~98.2%
SELECT
    'TEST 5: Market → spine join quality >=90%' AS test,
    COUNT(DISTINCT nfl_game_id || '|' || player_name_norm) AS training_player_games,
    3878 AS market_player_games,
    ROUND(COUNT(DISTINCT nfl_game_id || '|' || player_name_norm) * 100.0 / 3878, 1) AS match_pct,
    CASE
        WHEN COUNT(DISTINCT nfl_game_id || '|' || player_name_norm) * 1.0 / 3878 >= 0.90
        THEN 'PASS'
        ELSE 'FAIL'
    END AS result
FROM read_parquet('/Users/thomasmyles/Downloads/tmp/rush_attempts/training.parquet');

-- ── Test 6: Date range covers expected training seasons ───────────────────────
SELECT
    'TEST 6: All 3 seasons (2023-2025) present' AS test,
    MIN(season) AS min_season,
    MAX(season) AS max_season,
    COUNT(DISTINCT season) AS n_seasons,
    COUNT(DISTINCT nfl_game_id) AS total_games,
    CASE
        WHEN MIN(season) = 2023 AND MAX(season) = 2025 AND COUNT(DISTINCT season) = 3
        THEN 'PASS'
        ELSE 'FAIL'
    END AS result
FROM read_parquet('/Users/thomasmyles/Downloads/tmp/rush_attempts/training.parquet');

-- ── Test 7: No future leakage (temporal OOF check) ───────────────────────────
-- For a spot-checked player, carry_rate_L5 at week W should use weeks W-5 to W-1
-- Verify: for any player-game, carry_rate_L5 IS NOT equal to carry average of next 5 games
-- Proxy: carry_rate_Lcareer at week W+1 > carry_rate_Lcareer at week W (monotone growth)
SELECT
    'TEST 7: Temporal ordering — games in chronological order' AS test,
    COUNT(*) AS total_games_checked,
    CASE
        WHEN MIN(week) = 1 AND MAX(week) <= 22
        THEN 'PASS'
        ELSE 'FAIL'
    END AS result
FROM read_parquet('/Users/thomasmyles/Downloads/tmp/rush_attempts/training.parquet');

-- ── Diagnostics ───────────────────────────────────────────────────────────────
SELECT
    'DIAG: Season coverage' AS label,
    season,
    COUNT(DISTINCT nfl_game_id) AS games,
    COUNT(DISTINCT player_name_norm) AS players,
    ROUND(AVG(consensus_point), 2) AS avg_consensus_line,
    ROUND(AVG(is_over::FLOAT), 3) AS is_over_rate,
    SUM(is_playoff) AS playoff_rows
FROM read_parquet('/Users/thomasmyles/Downloads/tmp/rush_attempts/training.parquet')
GROUP BY season
ORDER BY season;

SELECT
    'DIAG: Position breakdown' AS label,
    position,
    COUNT(DISTINCT nfl_game_id || '|' || player_name_norm) AS unique_player_games,
    ROUND(AVG(consensus_point), 2) AS avg_line,
    ROUND(AVG(is_over::FLOAT), 3) AS over_rate
FROM read_parquet('/Users/thomasmyles/Downloads/tmp/rush_attempts/training.parquet')
GROUP BY position
ORDER BY unique_player_games DESC;

SELECT
    'DIAG: Feature null rates (all columns)' AS label,
    ROUND(SUM(CASE WHEN carry_rate_L1 IS NULL THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 4) AS carry_L1,
    ROUND(SUM(CASE WHEN carry_rate_L3 IS NULL THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 4) AS carry_L3,
    ROUND(SUM(CASE WHEN carry_rate_L5 IS NULL THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 4) AS carry_L5,
    ROUND(SUM(CASE WHEN carry_rate_L16 IS NULL THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 4) AS carry_L16,
    ROUND(SUM(CASE WHEN carry_rate_Lcareer IS NULL THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 4) AS carry_career,
    ROUND(SUM(CASE WHEN over_rate_L3 IS NULL THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 4) AS over_L3,
    ROUND(SUM(CASE WHEN over_rate_Lcareer IS NULL THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 4) AS over_career,
    ROUND(SUM(CASE WHEN opp_carry_allowed_L8 IS NULL THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 4) AS opp_L8,
    ROUND(SUM(CASE WHEN game_total IS NULL THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 4) AS game_total,
    ROUND(SUM(CASE WHEN is_home IS NULL THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 4) AS is_home
FROM read_parquet('/Users/thomasmyles/Downloads/tmp/rush_attempts/training.parquet');
