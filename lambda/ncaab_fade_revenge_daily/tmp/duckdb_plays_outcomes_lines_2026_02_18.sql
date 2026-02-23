-- Join 3 data sources: plays (fade-revenge-spot), outcomes (historical_game_results), lines (game_lines).
-- Run via: bash lambda/ncaab_fade_revenge_daily/tmp/run_duckdb_plays_outcomes_lines.sh

WITH
-- 1) Plays: fade-revenge-spot plays for the date (bet_team IS NOT NULL)
plays AS (
  SELECT *
  FROM read_csv_auto('s3://ncaab-betting-mt/data/04_output/plays/fade-revenge-spot/2026-02-18.csv')
  WHERE bet_team IS NOT NULL
),

-- 2) Outcomes: final scores (historical_game_results)
outcomes AS (
  SELECT *
  FROM read_csv_auto('s3://ncaab-betting-mt/data/01_input/historical_game_results/2026-02-18.csv')
),

-- 3) Pregame lines: from game_lines CSV (consensus_spread when fetch_historical_ncaab_season_lines wrote it).
lines AS (
  SELECT
    CAST(date AS DATE) AS line_date,
    home_team AS line_home_team,
    away_team AS line_away_team,
    consensus_spread AS game_line
  FROM read_csv_auto('s3://ncaab-betting-mt/data/01_input/the-odds-api/ncaab/game_lines/2026-02-18.csv')
  GROUP BY date, home_team, away_team, consensus_spread
),

-- 4) Join: plays + outcomes + lines. Cover uses lines_csv_home_spread (game_line).
--    Home spread negative = home favored. Home covers when margin > -spread; away when away_margin > spread.
joined AS (
  SELECT
    p.game_date,
    p.home_team AS home,
    p.away_team AS away,
    p.focal_team,
    p.bet_team,
    p.spread_today AS bet_line_home_spread,
    o.HOME_SCORE,
    o.AWAY_SCORE,
    l.game_line AS lines_csv_home_spread,
    CASE
      WHEN o.HOME_SCORE IS NULL OR o.AWAY_SCORE IS NULL OR l.game_line IS NULL THEN NULL
      WHEN TRIM(p.bet_team) = TRIM(o.HOME_TEAM) AND (o.HOME_SCORE - o.AWAY_SCORE) > (-l.game_line) THEN 1
      WHEN TRIM(p.bet_team) = TRIM(o.HOME_TEAM) THEN 0
      WHEN TRIM(p.bet_team) = TRIM(o.AWAY_TEAM) AND (o.AWAY_SCORE - o.HOME_SCORE) > l.game_line THEN 1
      WHEN TRIM(p.bet_team) = TRIM(o.AWAY_TEAM) THEN 0
      ELSE NULL
    END AS binary_cover,
    -- Binary win straight up: 1 = bet_team won game, 0 = bet_team lost, NULL = no outcome
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
  LEFT JOIN lines l
    ON CAST(o.GAME_DATE AS DATE) = l.line_date
   AND LOWER(TRIM(o.HOME_TEAM)) = LOWER(TRIM(l.line_home_team))
   AND LOWER(TRIM(o.AWAY_TEAM)) = LOWER(TRIM(l.line_away_team))
)

SELECT * FROM joined ORDER BY home, away;
