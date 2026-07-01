-- Step 1 DuckDB Validation Tests
-- Run: duckdb -c ".read src/nfl_rush_attempts_modeling/scripts/step1_tests.sql"

.timer on

-- ── Test 1: Market data row count in expected range ───────────────────────────
SELECT
    'TEST 1: Market raw row count' AS test,
    COUNT(*) AS actual_rows,
    CASE WHEN COUNT(*) BETWEEN 40000 AND 80000 THEN 'PASS' ELSE 'FAIL' END AS result
FROM read_parquet('/Users/thomasmyles/Downloads/tmp/rush_attempts/market_raw.parquet');

-- ── Test 2: No unexpected nulls in key market columns ─────────────────────────
SELECT
    'TEST 2: Null rates in market columns' AS test,
    ROUND(SUM(CASE WHEN player_name IS NULL THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 4) AS player_name_null_rate,
    ROUND(SUM(CASE WHEN point IS NULL THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 4) AS point_null_rate,
    ROUND(SUM(CASE WHEN price IS NULL THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 4) AS price_null_rate,
    ROUND(SUM(CASE WHEN nfl_game_id IS NULL THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 4) AS game_id_null_rate,
    ROUND(SUM(CASE WHEN bookmaker IS NULL THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 4) AS bookmaker_null_rate,
    CASE
        WHEN SUM(CASE WHEN player_name IS NULL THEN 1 ELSE 0 END) = 0
         AND SUM(CASE WHEN point IS NULL THEN 1 ELSE 0 END) = 0
         AND SUM(CASE WHEN nfl_game_id IS NULL THEN 1 ELSE 0 END) = 0
        THEN 'PASS'
        ELSE 'FAIL'
    END AS result
FROM read_parquet('/Users/thomasmyles/Downloads/tmp/rush_attempts/market_raw.parquet');

-- ── Test 3: Line distribution — variable/numeric market (NOT binary) ──────────
-- Expect: variety of .5 values (0.5 through 24.5), NOT concentrated at 0.5
-- Binary skewed test: <5% at a single line value confirms it's variable market
SELECT
    'TEST 3: Variable line market (no single value >50% of lines)' AS test,
    MAX(cnt * 1.0 / total) AS max_single_line_fraction,
    CASE WHEN MAX(cnt * 1.0 / total) < 0.50 THEN 'PASS' ELSE 'FAIL' END AS result
FROM (
    SELECT point, COUNT(*) AS cnt
    FROM read_parquet('/Users/thomasmyles/Downloads/tmp/rush_attempts/market_dedup.parquet')
    GROUP BY point
) t
CROSS JOIN (SELECT COUNT(*) AS total FROM read_parquet('/Users/thomasmyles/Downloads/tmp/rush_attempts/market_dedup.parquet'));

-- ── Test 4: Over + Under + Push rates sum to ~100% ────────────────────────────
SELECT
    'TEST 4: Over+Under+Push hit rates sum to ~100%' AS test,
    ROUND(AVG(CASE WHEN rush_attempts_actual > point THEN 1.0 ELSE 0.0 END), 4) AS over_rate,
    ROUND(AVG(CASE WHEN rush_attempts_actual = point THEN 1.0 ELSE 0.0 END), 4) AS push_rate,
    ROUND(AVG(CASE WHEN rush_attempts_actual < point THEN 1.0 ELSE 0.0 END), 4) AS under_rate,
    ROUND(
        AVG(CASE WHEN rush_attempts_actual > point THEN 1.0 ELSE 0.0 END) +
        AVG(CASE WHEN rush_attempts_actual = point THEN 1.0 ELSE 0.0 END) +
        AVG(CASE WHEN rush_attempts_actual < point THEN 1.0 ELSE 0.0 END),
        4
    ) AS sum_all,
    CASE
        WHEN ABS(
            AVG(CASE WHEN rush_attempts_actual > point THEN 1.0 ELSE 0.0 END) +
            AVG(CASE WHEN rush_attempts_actual = point THEN 1.0 ELSE 0.0 END) +
            AVG(CASE WHEN rush_attempts_actual < point THEN 1.0 ELSE 0.0 END) - 1.0
        ) < 0.01 THEN 'PASS'
        ELSE 'FAIL'
    END AS result
FROM read_parquet('/Users/thomasmyles/Downloads/tmp/rush_attempts/market_joined.parquet')
WHERE rush_attempts_actual IS NOT NULL;

-- ── Test 5: Feature data row count in expected range ──────────────────────────
SELECT
    'TEST 5: PFR row count (all positions, REG season, 3 seasons)' AS test,
    COUNT(*) AS actual_rows,
    CASE WHEN COUNT(*) BETWEEN 5000 AND 12000 THEN 'PASS' ELSE 'FAIL' END AS result
FROM read_parquet('/Users/thomasmyles/Downloads/tmp/rush_attempts/pfr_rushing.parquet');

-- ── Test 6: Key stat column null rate <10% ────────────────────────────────────
SELECT
    'TEST 6: rush_attempts_actual null rate <10%' AS test,
    ROUND(SUM(CASE WHEN rush_attempts_actual IS NULL THEN 1 ELSE 0 END) * 1.0 / COUNT(*), 4) AS null_rate,
    CASE
        WHEN SUM(CASE WHEN rush_attempts_actual IS NULL THEN 1 ELSE 0 END) * 1.0 / COUNT(*) < 0.10
        THEN 'PASS'
        ELSE 'FAIL'
    END AS result
FROM read_parquet('/Users/thomasmyles/Downloads/tmp/rush_attempts/pfr_rushing.parquet');

-- ── Test 7: Date ranges overlap — game weeks align ────────────────────────────
SELECT
    'TEST 7: Market and PFR season/week ranges overlap' AS test,
    m.min_market_season,
    m.max_market_season,
    m.min_market_week,
    m.max_market_week,
    p.min_pfr_season,
    p.max_pfr_season,
    p.min_pfr_week,
    p.max_pfr_week,
    CASE
        WHEN m.min_market_season <= p.max_pfr_season
         AND m.max_market_season >= p.min_pfr_season
         AND m.min_market_week <= p.max_pfr_week
        THEN 'PASS'
        ELSE 'FAIL'
    END AS result
FROM (
    SELECT
        MIN(game_season) AS min_market_season, MAX(game_season) AS max_market_season,
        MIN(game_week) AS min_market_week, MAX(game_week) AS max_market_week
    FROM read_parquet('/Users/thomasmyles/Downloads/tmp/rush_attempts/market_dedup.parquet')
) m
CROSS JOIN (
    SELECT
        MIN(season) AS min_pfr_season, MAX(season) AS max_pfr_season,
        MIN(week) AS min_pfr_week, MAX(week) AS max_pfr_week
    FROM read_parquet('/Users/thomasmyles/Downloads/tmp/rush_attempts/pfr_rushing.parquet')
) p;

-- ── Additional diagnostic: coverage by season + week ─────────────────────────
SELECT
    'DIAGNOSTIC: Games with lines by season' AS label,
    game_season,
    COUNT(DISTINCT nfl_game_id) AS games_with_lines,
    COUNT(DISTINCT player_name) AS unique_players,
    ROUND(AVG(point), 2) AS avg_line
FROM read_parquet('/Users/thomasmyles/Downloads/tmp/rush_attempts/market_dedup.parquet')
GROUP BY game_season
ORDER BY game_season;

-- ── Additional diagnostic: books coverage ────────────────────────────────────
SELECT
    'DIAGNOSTIC: Book coverage' AS label,
    bookmaker,
    COUNT(DISTINCT nfl_game_id) AS games_covered,
    ROUND(COUNT(DISTINCT nfl_game_id) * 100.0 / 828, 1) AS pct_of_games
FROM read_parquet('/Users/thomasmyles/Downloads/tmp/rush_attempts/market_dedup.parquet')
GROUP BY bookmaker
ORDER BY games_covered DESC;

-- ── Additional diagnostic: join quality check ─────────────────────────────────
SELECT
    'DIAGNOSTIC: Join quality (market ↔ PFR)' AS label,
    COUNT(*) AS total_market_rows,
    SUM(CASE WHEN rush_attempts_actual IS NOT NULL THEN 1 ELSE 0 END) AS matched_rows,
    ROUND(SUM(CASE WHEN rush_attempts_actual IS NOT NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS match_pct,
    CASE
        WHEN SUM(CASE WHEN rush_attempts_actual IS NOT NULL THEN 1 ELSE 0 END) * 1.0 / COUNT(*) >= 0.90
        THEN 'PASS (>=90%)'
        ELSE 'WARN (<90%)'
    END AS result
FROM read_parquet('/Users/thomasmyles/Downloads/tmp/rush_attempts/market_joined.parquet');

-- ── Additional diagnostic: unmatched players sample ──────────────────────────
SELECT
    'DIAGNOSTIC: Sample unmatched players (name mismatch or DNP)' AS label,
    player_name,
    game_season,
    COUNT(*) AS unmatched_rows
FROM read_parquet('/Users/thomasmyles/Downloads/tmp/rush_attempts/market_joined.parquet')
WHERE rush_attempts_actual IS NULL
GROUP BY player_name, game_season
ORDER BY unmatched_rows DESC
LIMIT 20;
