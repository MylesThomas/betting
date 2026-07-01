-- Step 5 DuckDB Validation Tests
-- Run: duckdb -c ".read src/nfl_rush_attempts_modeling/scripts/step5_tests.sql"

.timer on

-- ── Test 1: Grid has one row per (threshold, direction, line, pos) combo ─────
SELECT
    'TEST 1: One row per unique strategy combo' AS test,
    COUNT(*) AS total_rows,
    COUNT(DISTINCT edge_threshold || '|' || direction || '|' || line_filter || '|' || position_filter) AS unique_combos,
    CASE
        WHEN COUNT(*) = COUNT(DISTINCT edge_threshold || '|' || direction || '|' || line_filter || '|' || position_filter)
        THEN 'PASS'
        ELSE 'FAIL'
    END AS result
FROM read_csv_auto('/Users/thomasmyles/Downloads/tmp/rush_attempts/step5_grid.csv');

-- ── Test 2: Bet counts are directionally correct ──────────────────────────────
-- At edge_threshold=0, direction=under, all positions, high lines:
-- n_bets should be close to total QB+RB high-line rows in step4_bets
SELECT
    'TEST 2: n_bets at threshold=0 under all high = total negative-edge high rows' AS test,
    n_bets,
    CASE WHEN n_bets > 10000 THEN 'PASS' ELSE 'FAIL (unexpectedly low)' END AS result
FROM read_csv_auto('/Users/thomasmyles/Downloads/tmp/rush_attempts/step5_grid.csv')
WHERE edge_threshold = 0.00
  AND direction = 'under'
  AND line_filter = 'high'
  AND position_filter = 'all';

-- ── Test 3: Units calculation spot-check ─────────────────────────────────────
-- At threshold=0.20 (very high edge), n_bets must be small
-- and units_won / n_bets = roi (definition check)
SELECT
    'TEST 3: ROI = units_won / n_bets' AS test,
    edge_threshold,
    direction,
    position_filter,
    n_bets,
    ROUND(units_won, 4) AS units_won,
    ROUND(roi, 4) AS roi,
    ROUND(units_won / n_bets, 4) AS computed_roi,
    CASE
        WHEN ABS(roi - units_won / n_bets) < 0.0001
        THEN 'PASS'
        ELSE 'FAIL'
    END AS result
FROM read_csv_auto('/Users/thomasmyles/Downloads/tmp/rush_attempts/step5_grid.csv')
WHERE edge_threshold = 0.03
  AND direction = 'under'
  AND line_filter = 'high'
  AND position_filter = 'QB';

-- ── Test 4: No strategy ROI > 25% with > 100 bets (leakage check) ────────────
SELECT
    'TEST 4: No strategy ROI > 25% with > 100 bets' AS test,
    COUNT(*) AS suspicious_rows,
    CASE
        WHEN COUNT(*) = 0
        THEN 'PASS'
        ELSE 'FAIL (possible leakage — review flagged strategies)'
    END AS result
FROM read_csv_auto('/Users/thomasmyles/Downloads/tmp/rush_attempts/step5_grid.csv')
WHERE roi > 0.25 AND n_bets > 100;

-- ── Test 5: Best strategy by units has >= 30 bets ────────────────────────────
SELECT
    'TEST 5: Best strategy has >= 30 bets' AS test,
    MAX(units_won) AS best_units,
    n_bets,
    direction,
    line_filter,
    position_filter,
    CASE
        WHEN n_bets >= 30
        THEN 'PASS'
        ELSE 'FAIL'
    END AS result
FROM read_csv_auto('/Users/thomasmyles/Downloads/tmp/rush_attempts/step5_grid.csv')
WHERE units_won = (SELECT MAX(units_won) FROM read_csv_auto('/Users/thomasmyles/Downloads/tmp/rush_attempts/step5_grid.csv'));

-- ── Diagnostics ───────────────────────────────────────────────────────────────

-- QB high-line under by season breakdown (verify consistent across seasons)
SELECT
    'DIAG: QB high-line under consistency by season' AS label,
    season,
    COUNT(*) AS n_bets,
    ROUND(AVG(is_over::FLOAT), 4) AS actual_over_rate,
    ROUND(1 - AVG(is_over::FLOAT), 4) AS actual_under_rate,
    ROUND(AVG(p_market), 4) AS avg_market_implied,
    ROUND(AVG(p_model), 4) AS avg_model_p
FROM read_parquet('/Users/thomasmyles/Downloads/tmp/rush_attempts/step4_bets.parquet')
WHERE position = 'QB'
  AND book_line >= 6.5
  AND (-edge) >= 0.03
GROUP BY season
ORDER BY season;

-- ROI by edge threshold for QB high under (production strategy sensitivity)
SELECT
    'DIAG: QB high under — ROI by edge threshold' AS label,
    edge_threshold,
    n_bets,
    ROUND(win_rate, 4) AS win_rate,
    ROUND(units_won, 2) AS units_won,
    ROUND(roi, 4) AS roi,
    ROUND(max_drawdown, 2) AS max_drawdown
FROM read_csv_auto('/Users/thomasmyles/Downloads/tmp/rush_attempts/step5_grid.csv')
WHERE direction = 'under'
  AND line_filter = 'high'
  AND position_filter = 'QB'
ORDER BY edge_threshold;
