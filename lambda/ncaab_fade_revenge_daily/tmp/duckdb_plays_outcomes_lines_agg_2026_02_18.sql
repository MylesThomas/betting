-- Same 3 sources + join as duckdb_plays_outcomes_lines_2026_02_18.sql, then aggregate to one row.
-- Run via: bash lambda/ncaab_fade_revenge_daily/tmp/run_duckdb_plays_outcomes_lines.sh (edit to call this .sql).

WITH
plays AS (
  SELECT *
  FROM read_csv_auto('s3://ncaab-betting-mt/data/04_output/plays/fade-revenge-spot/2026-02-18.csv')
  WHERE bet_team IS NOT NULL
),

outcomes AS (
  SELECT *
  FROM read_csv_auto('s3://ncaab-betting-mt/data/01_input/historical_game_results/2026-02-18.csv')
),

joined AS (
  SELECT
    p.game_date,
    p.home_team,
    p.away_team,
    p.focal_team,
    p.bet_team,
    p.spread_today,
    o.HOME_SCORE,
    o.AWAY_SCORE,
    CASE
      WHEN o.HOME_SCORE IS NULL OR o.AWAY_SCORE IS NULL OR p.spread_today IS NULL THEN NULL
      WHEN TRIM(p.bet_team) = TRIM(o.HOME_TEAM) AND (o.HOME_SCORE - o.AWAY_SCORE) > (-p.spread_today) THEN 1
      WHEN TRIM(p.bet_team) = TRIM(o.HOME_TEAM) THEN 0
      WHEN TRIM(p.bet_team) = TRIM(o.AWAY_TEAM) AND (o.AWAY_SCORE - o.HOME_SCORE) > p.spread_today THEN 1
      WHEN TRIM(p.bet_team) = TRIM(o.AWAY_TEAM) THEN 0
      ELSE NULL
    END AS binary_cover,
    CASE
      WHEN o.HOME_SCORE IS NULL OR o.AWAY_SCORE IS NULL THEN NULL
      WHEN TRIM(p.bet_team) = TRIM(o.HOME_TEAM) AND o.HOME_SCORE > o.AWAY_SCORE THEN 1
      WHEN TRIM(p.bet_team) = TRIM(o.HOME_TEAM) THEN 0
      WHEN TRIM(p.bet_team) = TRIM(o.AWAY_TEAM) AND o.AWAY_SCORE > o.HOME_SCORE THEN 1
      WHEN TRIM(p.bet_team) = TRIM(o.AWAY_TEAM) THEN 0
      ELSE NULL
    END AS binary_win_su
  FROM plays p
  LEFT JOIN outcomes o
    ON CAST(p.game_date AS DATE) = CAST(o.GAME_DATE AS DATE)
   AND TRIM(p.home_team) = TRIM(o.HOME_TEAM)
   AND TRIM(p.away_team) = TRIM(o.AWAY_TEAM)
),

agg AS (
  SELECT
    (SELECT COUNT(*) FROM outcomes) AS total_game_n,
    COUNT(*) AS plays_n,
    COUNT(*) FILTER (WHERE j.spread_today IS NOT NULL) AS plays_with_spread_n,
    SUM(j.binary_cover) AS covers,
    COUNT(*) FILTER (WHERE j.spread_today IS NOT NULL) AS cover_denom_n,
    ROUND(100.0 * SUM(j.binary_cover) / NULLIF(COUNT(*) FILTER (WHERE j.spread_today IS NOT NULL), 0), 1) AS cover_rate_pct,
    SUM(j.binary_win_su) AS wins,
    COUNT(*) FILTER (WHERE j.HOME_SCORE IS NOT NULL) AS win_denom_n,
    ROUND(100.0 * SUM(j.binary_win_su) / NULLIF(COUNT(*) FILTER (WHERE j.HOME_SCORE IS NOT NULL), 0), 1) AS win_rate_pct
  FROM joined j
)

SELECT total_game_n, plays_n, plays_with_spread_n,
       covers, cover_denom_n, cover_rate_pct,
       wins, win_denom_n, win_rate_pct
FROM agg;
