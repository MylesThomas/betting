#!/usr/bin/env bash
# Join 2026-02-22 plays (actual plays only) with outcomes and game lines; compute spread W/L.
# Requires: duckdb, aws CLI configured. Run from repo root: bash lambda/ncaab_fade_revenge_daily/tmp/run_duckdb_plays_outcomes_lines.sh

set -e
DATE="${1:-2026-02-22}"

duckdb -c "
INSTALL httpfs;
LOAD httpfs;
SET s3_region='us-east-2';
SET s3_access_key_id='$(aws configure get aws_access_key_id)';
SET s3_secret_access_key='$(aws configure get aws_secret_access_key)';
WITH
  plays AS (
    SELECT game_date, home_team, away_team, consensus_spread_home AS spread_home_plays, bet_team, start_time_et
    FROM read_csv_auto('s3://ncaab-betting-mt/data/04_output/plays/fade-revenge-spot/${DATE}.csv')
    WHERE TRIM(COALESCE(bet_team, '')) <> ''
  ),
  outcomes AS (
    SELECT GAME_DATE, HOME_TEAM, AWAY_TEAM, HOME_SCORE, AWAY_SCORE
    FROM read_csv_auto('s3://ncaab-betting-mt/data/01_input/historical_game_results/${DATE}.csv')
  ),
  lines AS (
    SELECT date, home_team, away_team, consensus_spread
    FROM read_csv_auto('s3://ncaab-betting-mt/data/01_input/the-odds-api/ncaab/game_lines/${DATE}.csv')
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
JOIN outcomes o ON p.home_team = o.HOME_TEAM AND p.away_team = o.AWAY_TEAM
LEFT JOIN lines l ON p.home_team = l.home_team AND p.away_team = l.away_team
ORDER BY p.start_time_et, p.away_team;
"
