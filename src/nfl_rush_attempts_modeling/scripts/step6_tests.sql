-- Step 6 DuckDB Validation Tests
-- Run: duckdb -c ".read src/nfl_rush_attempts_modeling/scripts/step6_tests.sql"

.timer on

-- ── Test 1: In-sample ROI is positive for QB high under edge>=0.03 ────────────
-- (IS ROI is lower than OOS here — see diagnostic note — but must be positive)
SELECT
    'TEST 1: IS ROI > 0 for QB high under edge=0.03' AS test,
    ROUND(roi, 4) AS is_roi,
    n_bets,
    CASE
        WHEN roi > 0
        THEN 'PASS'
        ELSE 'FAIL (IS ROI is negative — red flag for OOS strategy)'
    END AS result
FROM read_csv_auto('/Users/thomasmyles/Downloads/tmp/rush_attempts/step6_grid.csv')
WHERE direction = 'under'
  AND line_filter = 'high'
  AND position_filter = 'QB'
  AND edge_threshold = 0.03;

-- ── Test 2: IS/OOS ratio < 5x for key strategy ────────────────────────────────
-- IS ROI can be below OOS ROI for this market (structural QB mispricing, not overfit)
-- but the ratio should not be astronomically inflated in either direction
SELECT
    'TEST 2: IS/OOS ROI ratio for QB high under edge=0.03' AS test,
    ROUND(is_roi, 4) AS is_roi,
    ROUND(oos_roi, 4) AS oos_roi,
    ROUND(ABS(is_roi / NULLIF(oos_roi, 0)), 2) AS is_oos_ratio,
    CASE
        WHEN ABS(is_roi / NULLIF(oos_roi, 0)) < 5.0
        THEN 'PASS'
        ELSE 'FAIL (ratio >= 5x — possible overfit or structural change)'
    END AS result
FROM (
    SELECT
        (SELECT roi FROM read_csv_auto('/Users/thomasmyles/Downloads/tmp/rush_attempts/step6_grid.csv')
         WHERE direction='under' AND line_filter='high' AND position_filter='QB' AND edge_threshold=0.03) AS is_roi,
        (SELECT roi FROM read_csv_auto('/Users/thomasmyles/Downloads/tmp/rush_attempts/step5_grid.csv')
         WHERE direction='under' AND line_filter='high' AND position_filter='QB' AND edge_threshold=0.03) AS oos_roi
);

-- ── Test 3: IS grid has one row per unique strategy combo ─────────────────────
SELECT
    'TEST 3: One row per unique IS strategy combo' AS test,
    COUNT(*) AS total_rows,
    COUNT(DISTINCT edge_threshold || '|' || direction || '|' || line_filter || '|' || position_filter) AS unique_combos,
    CASE
        WHEN COUNT(*) = COUNT(DISTINCT edge_threshold || '|' || direction || '|' || line_filter || '|' || position_filter)
        THEN 'PASS'
        ELSE 'FAIL'
    END AS result
FROM read_csv_auto('/Users/thomasmyles/Downloads/tmp/rush_attempts/step6_grid.csv');

-- ── Test 4: IS n_bets >= OOS n_bets for same strategy ────────────────────────
-- IS uses full dataset (all rows), OOS only uses rows with OOF predictions (~80%)
-- So IS bet counts should be >= OOS bet counts at the same threshold
SELECT
    'TEST 4: IS n_bets >= OOS n_bets for QB high under' AS test,
    is_n,
    oos_n,
    CASE
        WHEN is_n >= oos_n
        THEN 'PASS'
        ELSE 'FAIL (fewer IS bets than OOS — possible data mismatch)'
    END AS result
FROM (
    SELECT
        (SELECT n_bets FROM read_csv_auto('/Users/thomasmyles/Downloads/tmp/rush_attempts/step6_grid.csv')
         WHERE direction='under' AND line_filter='high' AND position_filter='QB' AND edge_threshold=0.03) AS is_n,
        (SELECT n_bets FROM read_csv_auto('/Users/thomasmyles/Downloads/tmp/rush_attempts/step5_grid.csv')
         WHERE direction='under' AND line_filter='high' AND position_filter='QB' AND edge_threshold=0.03) AS oos_n
);

-- ── Test 5: No leakage — IS ROI should not exceed 50% with >100 bets ──────────
-- IS should show inflated ROI vs OOS but not implausibly high — cap at 50%
SELECT
    'TEST 5: No strategy IS ROI > 50% with > 100 bets' AS test,
    COUNT(*) AS suspicious_rows,
    CASE
        WHEN COUNT(*) = 0
        THEN 'PASS'
        ELSE 'FAIL (IS ROI > 50% with >100 bets — suspicious leakage?)'
    END AS result
FROM read_csv_auto('/Users/thomasmyles/Downloads/tmp/rush_attempts/step6_grid.csv')
WHERE roi > 0.50 AND n_bets > 100;

-- ── Diagnostics ───────────────────────────────────────────────────────────────

-- IS vs OOS comparison for QB high under across all edge thresholds
SELECT
    'DIAG: IS vs OOS — QB high under, by edge threshold' AS label,
    is_g.edge_threshold,
    is_g.n_bets AS is_n,
    oos_g.n_bets AS oos_n,
    ROUND(is_g.win_rate, 4) AS is_win_rate,
    ROUND(oos_g.win_rate, 4) AS oos_win_rate,
    ROUND(is_g.roi, 4) AS is_roi,
    ROUND(oos_g.roi, 4) AS oos_roi,
    ROUND(is_g.units_won, 2) AS is_units,
    ROUND(oos_g.units_won, 2) AS oos_units,
    ROUND(ABS(is_g.roi / NULLIF(oos_g.roi, 0)), 2) AS is_oos_ratio
FROM read_csv_auto('/Users/thomasmyles/Downloads/tmp/rush_attempts/step6_grid.csv') is_g
JOIN read_csv_auto('/Users/thomasmyles/Downloads/tmp/rush_attempts/step5_grid.csv') oos_g
  ON is_g.edge_threshold = oos_g.edge_threshold
 AND is_g.direction      = oos_g.direction
 AND is_g.line_filter    = oos_g.line_filter
 AND is_g.position_filter = oos_g.position_filter
WHERE is_g.direction = 'under'
  AND is_g.line_filter = 'high'
  AND is_g.position_filter = 'QB'
ORDER BY is_g.edge_threshold;

-- Top 15 IS strategies by ROI (≥50 bets)
SELECT
    'DIAG: Top IS strategies by ROI (≥50 bets)' AS label,
    edge_threshold, direction, line_filter, position_filter,
    n_bets, ROUND(win_rate, 4) AS win_rate,
    ROUND(units_won, 2) AS units_won, ROUND(roi, 4) AS roi,
    avg_odds
FROM read_csv_auto('/Users/thomasmyles/Downloads/tmp/rush_attempts/step6_grid.csv')
WHERE n_bets >= 50
ORDER BY roi DESC
LIMIT 15;
