-- v1_notebook_rebounds_universe.sql
-- Canonical notebook read query for v1 rebounds universe artifact.

SELECT
  season,
  date,
  game_id,
  player_normalized,
  team_normalized,
  MIN,
  OREB,
  DREB,
  REB,
  bookmaker,
  line,
  odds_over,
  odds_under,
  ROUND(MIN, 1) AS min_rounded,
  (REB - line) AS rebounds_vs_line
FROM read_parquet('~/Downloads/tmp/v1_rebounds_universe.parquet')
WHERE TRUE
  AND MIN > 0
  AND line IS NOT NULL
ORDER BY season, date, player_normalized, game_id;
