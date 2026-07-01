-- Step 4 DuckDB Validation Tests
-- Run: duckdb -c ".read src/nfl_rush_attempts_modeling/scripts/step4_tests.sql"

.timer on

-- ── Test 1: All P(over) values between 0 and 1, no nulls ─────────────────────
SELECT
    'TEST 1: p_model in [0,1], no nulls' AS test,
    COUNT(*) AS total_rows,
    SUM(CASE WHEN p_model IS NULL THEN 1 ELSE 0 END) AS null_p_model,
    SUM(CASE WHEN p_model < 0 OR p_model > 1 THEN 1 ELSE 0 END) AS out_of_range,
    ROUND(MIN(p_model), 4) AS min_p_model,
    ROUND(MAX(p_model), 4) AS max_p_model,
    CASE
        WHEN SUM(CASE WHEN p_model IS NULL THEN 1 ELSE 0 END) = 0
         AND SUM(CASE WHEN p_model < 0 OR p_model > 1 THEN 1 ELSE 0 END) = 0
        THEN 'PASS'
        ELSE 'FAIL'
    END AS result
FROM read_parquet('/Users/thomasmyles/Downloads/tmp/rush_attempts/step4_bets.parquet');

-- ── Test 2: All P(market) values between 0 and 1, no nulls ──────────────────
SELECT
    'TEST 2: p_market in [0,1], no nulls' AS test,
    SUM(CASE WHEN p_market IS NULL THEN 1 ELSE 0 END) AS null_p_market,
    SUM(CASE WHEN p_market < 0 OR p_market > 1 THEN 1 ELSE 0 END) AS out_of_range,
    ROUND(MIN(p_market), 4) AS min_p_market,
    ROUND(MAX(p_market), 4) AS max_p_market,
    CASE
        WHEN SUM(CASE WHEN p_market IS NULL THEN 1 ELSE 0 END) = 0
         AND SUM(CASE WHEN p_market < 0 OR p_market > 1 THEN 1 ELSE 0 END) = 0
        THEN 'PASS'
        ELSE 'FAIL'
    END AS result
FROM read_parquet('/Users/thomasmyles/Downloads/tmp/rush_attempts/step4_bets.parquet');

-- ── Test 3: Edge values have a plausible distribution ────────────────────────
-- Mean edge should be small (market is efficient). Flag if all-positive or extreme.
SELECT
    'TEST 3: Edge distribution is plausible' AS test,
    ROUND(MIN(edge), 4) AS min_edge,
    ROUND(MAX(edge), 4) AS max_edge,
    ROUND(AVG(edge), 4) AS mean_edge,
    ROUND(STDDEV(edge), 4) AS std_edge,
    SUM(CASE WHEN edge > 0 THEN 1 ELSE 0 END) AS positive_edge_rows,
    SUM(CASE WHEN edge < 0 THEN 1 ELSE 0 END) AS negative_edge_rows,
    CASE
        WHEN MIN(edge) < 0 AND MAX(edge) > 0
         AND ABS(AVG(edge)) < 0.15
        THEN 'PASS'
        ELSE 'FAIL (suspicious edge distribution — check for leakage)'
    END AS result
FROM read_parquet('/Users/thomasmyles/Downloads/tmp/rush_attempts/step4_bets.parquet');

-- ── Test 4: Calibration — all deciles within 0.15 ────────────────────────────
-- Compute decile, then check max calibration error
WITH deciles AS (
    SELECT
        NTILE(10) OVER (ORDER BY p_model) AS decile,
        p_model,
        is_over
    FROM read_parquet('/Users/thomasmyles/Downloads/tmp/rush_attempts/step4_bets.parquet')
),
cal AS (
    SELECT
        decile,
        AVG(p_model) AS avg_p_model,
        AVG(is_over::FLOAT) AS actual_over_rate,
        ABS(AVG(p_model) - AVG(is_over::FLOAT)) AS calib_error,
        COUNT(*) AS n
    FROM deciles
    GROUP BY decile
)
SELECT
    'TEST 4: All calibration deciles within 0.15' AS test,
    ROUND(MAX(calib_error), 4) AS max_calib_error,
    SUM(CASE WHEN calib_error >= 0.15 THEN 1 ELSE 0 END) AS failing_deciles,
    CASE
        WHEN MAX(calib_error) < 0.15
        THEN 'PASS'
        ELSE 'FAIL (some deciles miscalibrated)'
    END AS result
FROM cal;

-- ── Test 5: Row count matches expected (80% of per-book training) ─────────────
SELECT
    'TEST 5: Row count ~80% of 23,503 per-book rows' AS test,
    COUNT(*) AS step4_rows,
    23503 AS train_rows,
    ROUND(COUNT(*) * 100.0 / 23503, 1) AS pct_of_training,
    CASE
        WHEN COUNT(*) >= 18000 AND COUNT(*) <= 23503
        THEN 'PASS'
        ELSE 'FAIL'
    END AS result
FROM read_parquet('/Users/thomasmyles/Downloads/tmp/rush_attempts/step4_bets.parquet');

-- ── Diagnostics ───────────────────────────────────────────────────────────────
SELECT
    'DIAG: Edge by season' AS label,
    season,
    COUNT(*) AS n_rows,
    ROUND(AVG(p_model), 4) AS avg_p_model,
    ROUND(AVG(p_market), 4) AS avg_p_market,
    ROUND(AVG(edge), 4) AS avg_edge,
    ROUND(AVG(is_over::FLOAT), 4) AS actual_over_rate,
    SUM(CASE WHEN edge > 0.05 THEN 1 ELSE 0 END) AS n_positive_edge_5pct
FROM read_parquet('/Users/thomasmyles/Downloads/tmp/rush_attempts/step4_bets.parquet')
GROUP BY season
ORDER BY season;

SELECT
    'DIAG: Edge by position' AS label,
    position,
    COUNT(*) AS n_rows,
    ROUND(AVG(p_model), 4) AS avg_p_model,
    ROUND(AVG(p_market), 4) AS avg_p_market,
    ROUND(AVG(edge), 4) AS avg_edge,
    ROUND(AVG(is_over::FLOAT), 4) AS actual_over_rate
FROM read_parquet('/Users/thomasmyles/Downloads/tmp/rush_attempts/step4_bets.parquet')
GROUP BY position
ORDER BY n_rows DESC;

SELECT
    'DIAG: Edge by bookmaker' AS label,
    bookmaker,
    COUNT(*) AS n_rows,
    ROUND(AVG(edge), 4) AS avg_edge,
    ROUND(AVG(p_model), 4) AS avg_p_model,
    ROUND(AVG(p_market), 4) AS avg_p_market,
    SUM(CASE WHEN edge > 0.05 THEN 1 ELSE 0 END) AS n_edge_5pct
FROM read_parquet('/Users/thomasmyles/Downloads/tmp/rush_attempts/step4_bets.parquet')
GROUP BY bookmaker
ORDER BY avg_edge DESC;
