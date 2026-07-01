-- Step 3 DuckDB Validation Tests (regression model — target: carries count)
-- Run: duckdb -c ".read src/nfl_rush_attempts_modeling/scripts/step3_tests.sql"

.timer on

-- ── Test 1: OOF predictions cover ≥80% of player-games ──────────────────────
-- First temporal fold (2023 wk 1-14) has no prior data, so ~20% of rows
-- legitimately cannot receive OOF predictions. ≥80% coverage is acceptable.
SELECT
    'TEST 1: OOF predictions cover >=80% of rows' AS test,
    COUNT(*) AS total_rows,
    SUM(CASE WHEN oof_carries IS NOT NULL THEN 1 ELSE 0 END) AS rows_with_predictions,
    ROUND(SUM(CASE WHEN oof_carries IS NOT NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS pct_covered,
    CASE
        WHEN SUM(CASE WHEN oof_carries IS NOT NULL THEN 1 ELSE 0 END) * 1.0 / COUNT(*) >= 0.80
        THEN 'PASS'
        ELSE 'FAIL'
    END AS result
FROM read_parquet('/Users/thomasmyles/Downloads/tmp/rush_attempts/oof_predictions.parquet');

-- ── Test 2: OOF carry predictions are in a plausible range (0–40) ────────────
SELECT
    'TEST 2: oof_carries in plausible range [0,40]' AS test,
    COUNT(*) AS rows_with_preds,
    SUM(CASE WHEN oof_carries < 0 OR oof_carries > 40 THEN 1 ELSE 0 END) AS out_of_range,
    ROUND(MIN(oof_carries), 2) AS min_pred,
    ROUND(MAX(oof_carries), 2) AS max_pred,
    ROUND(AVG(oof_carries), 2) AS mean_pred,
    CASE
        WHEN SUM(CASE WHEN oof_carries < 0 OR oof_carries > 40 THEN 1 ELSE 0 END) = 0
        THEN 'PASS'
        ELSE 'FAIL'
    END AS result
FROM read_parquet('/Users/thomasmyles/Downloads/tmp/rush_attempts/oof_predictions.parquet')
WHERE oof_carries IS NOT NULL;

-- ── Test 3: Directionally sensible — predicted carries correlates with actual ─
-- avg predicted for players who carry a lot (>12 actual) should exceed
-- avg predicted for players who carry less (<6 actual)
SELECT
    'TEST 3: Higher actual carries → higher predicted carries' AS test,
    ROUND(AVG(CASE WHEN carries > 12 THEN oof_carries END), 2) AS avg_pred_heavy_carriers,
    ROUND(AVG(CASE WHEN carries < 6  THEN oof_carries END), 2) AS avg_pred_light_carriers,
    CASE
        WHEN AVG(CASE WHEN carries > 12 THEN oof_carries END) >
             AVG(CASE WHEN carries < 6  THEN oof_carries END)
        THEN 'PASS'
        ELSE 'FAIL'
    END AS result
FROM read_parquet('/Users/thomasmyles/Downloads/tmp/rush_attempts/oof_predictions.parquet')
WHERE oof_carries IS NOT NULL;

-- ── Test 4: 2024+ rows all have OOF predictions (temporal fold boundary) ─────
SELECT
    'TEST 4: 2024+ rows all have oof_carries' AS test,
    COUNT(*) AS rows_2024_plus,
    SUM(CASE WHEN oof_carries IS NULL THEN 1 ELSE 0 END) AS null_preds_in_2024_plus,
    CASE
        WHEN SUM(CASE WHEN oof_carries IS NULL THEN 1 ELSE 0 END) = 0
        THEN 'PASS'
        ELSE 'FAIL (some 2024+ rows missing OOF pred — check fold boundaries)'
    END AS result
FROM read_parquet('/Users/thomasmyles/Downloads/tmp/rush_attempts/oof_predictions.parquet')
WHERE season >= 2024;

-- ── Test 5: Fold 1 has >=200 rows (first fold = null OOF block) ──────────────
SELECT
    'TEST 5: First fold >= 200 rows' AS test,
    SUM(CASE WHEN oof_carries IS NULL THEN 1 ELSE 0 END) AS fold1_rows,
    CASE
        WHEN SUM(CASE WHEN oof_carries IS NULL THEN 1 ELSE 0 END) >= 200
        THEN 'PASS'
        ELSE 'FAIL'
    END AS result
FROM read_parquet('/Users/thomasmyles/Downloads/tmp/rush_attempts/oof_predictions.parquet');

-- ── Test 6: One OOF row per player-game (no duplicates) ──────────────────────
SELECT
    'TEST 6: One OOF row per player-game' AS test,
    COUNT(*) AS total_rows,
    COUNT(DISTINCT nfl_game_id || '|' || player_name_norm) AS unique_player_games,
    CASE
        WHEN COUNT(*) = COUNT(DISTINCT nfl_game_id || '|' || player_name_norm)
        THEN 'PASS'
        ELSE 'FAIL'
    END AS result
FROM read_parquet('/Users/thomasmyles/Downloads/tmp/rush_attempts/oof_predictions.parquet');

-- ── Diagnostics ───────────────────────────────────────────────────────────────
SELECT
    'DIAG: OOF null breakdown by season' AS label,
    season,
    COUNT(*) AS total_rows,
    SUM(CASE WHEN oof_carries IS NULL THEN 1 ELSE 0 END) AS null_preds,
    ROUND(AVG(oof_carries), 2) AS avg_pred_carries,
    ROUND(AVG(carries), 2) AS avg_actual_carries,
    ROUND(SQRT(AVG(POWER(oof_carries - carries, 2))), 4) AS rmse
FROM read_parquet('/Users/thomasmyles/Downloads/tmp/rush_attempts/oof_predictions.parquet')
WHERE oof_carries IS NOT NULL
GROUP BY season
ORDER BY season;

SELECT
    'DIAG: RMSE by carry bucket' AS label,
    CASE
        WHEN carries < 5  THEN '0-4'
        WHEN carries < 10 THEN '5-9'
        WHEN carries < 15 THEN '10-14'
        WHEN carries < 20 THEN '15-19'
        ELSE '20+'
    END AS carry_bucket,
    COUNT(*) AS n_rows,
    ROUND(AVG(carries), 2) AS avg_actual,
    ROUND(AVG(oof_carries), 2) AS avg_predicted,
    ROUND(SQRT(AVG(POWER(oof_carries - carries, 2))), 4) AS rmse
FROM read_parquet('/Users/thomasmyles/Downloads/tmp/rush_attempts/oof_predictions.parquet')
WHERE oof_carries IS NOT NULL
GROUP BY carry_bucket
ORDER BY carry_bucket;
