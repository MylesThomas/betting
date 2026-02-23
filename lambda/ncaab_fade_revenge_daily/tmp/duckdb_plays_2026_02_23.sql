-- Inspect NCAAB Fade Revenge plays file for 2026-02-22.
-- Answers: (1) schema and spreads, (2) were we really betting all rows? (bet_team null = not a play)
--
-- Run from repo root with AWS creds configured. One-shot with credentials:
--
--   duckdb -c "
--   INSTALL httpfs; LOAD httpfs;
--   SET s3_region='us-east-2';
--   SET s3_access_key_id='$(aws configure get aws_access_key_id)';
--   SET s3_secret_access_key='$(aws configure get aws_secret_access_key)';
--   SELECT COUNT(*) AS total_rows,
--     COUNT(CASE WHEN TRIM(COALESCE(bet_team, '')) <> '' THEN 1 END) AS actual_plays,
--     COUNT(*) - COUNT(CASE WHEN TRIM(COALESCE(bet_team, '')) <> '' THEN 1 END) AS not_plays
--   FROM read_csv_auto('s3://ncaab-betting-mt/data/04_output/plays/fade-revenge-spot/2026-02-22.csv');
--   SELECT away_team, home_team, consensus_spread_home, bet_team,
--     CASE WHEN TRIM(COALESCE(bet_team, '')) = '' THEN 'no_play' ELSE 'play' END AS is_play
--   FROM read_csv_auto('s3://ncaab-betting-mt/data/04_output/plays/fade-revenge-spot/2026-02-22.csv')
--   ORDER BY start_time_et, away_team;
--   "
--
-- Or run the blocks below in an interactive duckdb session (INSTALL/LOAD/SET once, then run 2–4).

INSTALL httpfs;
LOAD httpfs;
SET s3_region='us-east-2';

-- 1) Schema of plays CSV
DESCRIBE SELECT * FROM read_csv_auto('s3://ncaab-betting-mt/data/04_output/plays/fade-revenge-spot/2026-02-22.csv');

-- 2) Count: total rows vs actual plays (bet_team not null/empty)
SELECT
  COUNT(*) AS total_rows,
  COUNT(CASE WHEN TRIM(COALESCE(bet_team, '')) <> '' THEN 1 END) AS actual_plays_bet_team_set,
  COUNT(*) - COUNT(CASE WHEN TRIM(COALESCE(bet_team, '')) <> '' THEN 1 END) AS not_plays_bet_team_null_or_empty
FROM read_csv_auto('s3://ncaab-betting-mt/data/04_output/plays/fade-revenge-spot/2026-02-22.csv');

-- 3) All rows: game, spread, bet_team (to confirm which were plays and whether spread is present)
SELECT
  away_team,
  home_team,
  consensus_spread_home,
  bet_team,
  CASE WHEN TRIM(COALESCE(bet_team, '')) = '' THEN 'no_play' ELSE 'play' END AS is_play
FROM read_csv_auto('s3://ncaab-betting-mt/data/04_output/plays/fade-revenge-spot/2026-02-22.csv')
ORDER BY start_time_et, away_team;

-- 4) Only actual plays (what we really bet)
SELECT
  away_team,
  home_team,
  consensus_spread_home,
  bet_team,
  focal_team,
  record
FROM read_csv_auto('s3://ncaab-betting-mt/data/04_output/plays/fade-revenge-spot/2026-02-22.csv')
WHERE TRIM(COALESCE(bet_team, '')) <> ''
ORDER BY start_time_et;

-- 5) Join plays (actual plays only) with outcomes and game lines; compute spread cover W/L
--    consensus_spread (game_lines) = home spread. Home covers if (HOME_SCORE + spread) > AWAY_SCORE;
--    away covers if (AWAY_SCORE - spread) > HOME_SCORE.
WITH
  plays AS (
    SELECT game_date, home_team, away_team, consensus_spread_home AS spread_home_plays, bet_team, start_time_et
    FROM read_csv_auto('s3://ncaab-betting-mt/data/04_output/plays/fade-revenge-spot/2026-02-22.csv')
    WHERE TRIM(COALESCE(bet_team, '')) <> ''
  ),
  outcomes AS (
    SELECT GAME_DATE, HOME_TEAM, AWAY_TEAM, HOME_SCORE, AWAY_SCORE
    FROM read_csv_auto('s3://ncaab-betting-mt/data/01_input/historical_game_results/2026-02-22.csv')
  ),
  lines AS (
    SELECT date, home_team, away_team, consensus_spread
    FROM read_csv_auto('s3://ncaab-betting-mt/data/01_input/the-odds-api/ncaab/game_lines/2026-02-22.csv')
  )
SELECT
  p.away_team,
  o.AWAY_SCORE,
  p.home_team,
  o.HOME_SCORE,
  p.bet_team,
  COALESCE(l.consensus_spread, p.spread_home_plays) AS spread_home,
  o.HOME_SCORE - o.AWAY_SCORE AS point_diff_home,
  (o.HOME_SCORE - o.AWAY_SCORE) + COALESCE(l.consensus_spread, p.spread_home_plays) AS spread_margin_home,
  CASE
    WHEN p.bet_team = p.home_team THEN (o.HOME_SCORE - o.AWAY_SCORE) + COALESCE(l.consensus_spread, p.spread_home_plays)
    WHEN p.bet_team = p.away_team THEN -((o.HOME_SCORE - o.AWAY_SCORE) + COALESCE(l.consensus_spread, p.spread_home_plays))
    ELSE NULL
  END AS spread_margin_bet_team,
  CASE
    WHEN p.bet_team = p.home_team THEN (o.HOME_SCORE + COALESCE(l.consensus_spread, p.spread_home_plays)) > o.AWAY_SCORE
    WHEN p.bet_team = p.away_team THEN (o.AWAY_SCORE - COALESCE(l.consensus_spread, p.spread_home_plays)) > o.HOME_SCORE
    ELSE NULL
  END AS spread_cover,
  CASE
    WHEN p.bet_team = p.home_team AND (o.HOME_SCORE + COALESCE(l.consensus_spread, p.spread_home_plays)) > o.AWAY_SCORE THEN 'W'
    WHEN p.bet_team = p.away_team AND (o.AWAY_SCORE - COALESCE(l.consensus_spread, p.spread_home_plays)) > o.HOME_SCORE THEN 'W'
    WHEN p.bet_team = p.home_team OR p.bet_team = p.away_team THEN 'L'
    ELSE NULL
  END AS result
FROM plays p
JOIN outcomes o
  ON p.home_team = o.HOME_TEAM AND p.away_team = o.AWAY_TEAM
LEFT JOIN lines l
  ON p.home_team = l.home_team AND p.away_team = l.away_team
ORDER BY p.start_time_et, p.away_team;
