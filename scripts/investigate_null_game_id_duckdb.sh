#!/usr/bin/env bash
#
# Investigate why some props have null GAME_ID: (game_date, team_full) from props
# don't match any game in game_results. Uses DuckDB on parquet files.
#
# Run from repo root. Optional: set STRATEGY_PQ and GAME_RESULTS_PQ to override paths.
#
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STRATEGY_PQ="${STRATEGY_PQ:-$HOME/Downloads/tmp/nba_prop_strategies.parquet}"
GAME_RESULTS_PQ="${GAME_RESULTS_PQ:-$REPO_ROOT/data/02_cache/nba_strategy_build/game_results_2023-24_2024-25_2025-26.parquet}"

if [[ ! -f "$STRATEGY_PQ" ]]; then
  echo "Missing strategy parquet: $STRATEGY_PQ"
  echo "Run: python scripts/build_nba_multimarket_strategy_dataset.py --output $STRATEGY_PQ --seasons 2023-24 2024-25 2025-26"
  exit 1
fi

echo "=============================================="
echo "1. Unmatched: distinct (game_date, team_full) with null GAME_ID"
echo "=============================================="
duckdb -c "
  SELECT game_date, team_full, count(*) AS n
  FROM '$STRATEGY_PQ'
  WHERE \"GAME_ID\" IS NULL
  GROUP BY game_date, team_full
  ORDER BY game_date, team_full
  LIMIT 25;
"

echo ""
echo "=============================================="
echo "2. Unmatched: do these game_dates exist in game_results?"
echo "   (Sample 5 unmatched dates)"
echo "=============================================="
if [[ -f "$GAME_RESULTS_PQ" ]]; then
  duckdb -c "
    WITH unmatched_dates AS (
      SELECT DISTINCT game_date FROM '$STRATEGY_PQ' WHERE \"GAME_ID\" IS NULL LIMIT 5
    )
    SELECT u.game_date,
           (SELECT count(*) FROM '$GAME_RESULTS_PQ' g WHERE g.game_date = u.game_date) AS games_that_day
    FROM unmatched_dates u;
  "
else
  echo "Game results parquet not found: $GAME_RESULTS_PQ"
  echo "Run the build script once to populate cache."
fi

echo ""
echo "=============================================="
echo "3. For one unmatched date: which teams in game_results vs props?"
echo "   (Pick first unmatched date)"
echo "=============================================="
if [[ -f "$GAME_RESULTS_PQ" ]]; then
  duckdb -c "
    WITH first_bad_date AS (
      SELECT game_date FROM '$STRATEGY_PQ' WHERE \"GAME_ID\" IS NULL ORDER BY game_date LIMIT 1
    ),
    teams_in_games AS (
      SELECT game_date, home_team AS team FROM '$GAME_RESULTS_PQ' WHERE game_date = (SELECT game_date FROM first_bad_date)
      UNION ALL
      SELECT game_date, away_team AS team FROM '$GAME_RESULTS_PQ' WHERE game_date = (SELECT game_date FROM first_bad_date)
    ),
    teams_in_props AS (
      SELECT DISTINCT game_date, team_full AS team FROM '$STRATEGY_PQ'
      WHERE \"GAME_ID\" IS NULL AND game_date = (SELECT game_date FROM first_bad_date)
    )
    SELECT 'In game_results' AS source, team FROM teams_in_games
    UNION ALL
    SELECT 'In props (unmatched)' AS source, team FROM teams_in_props
    ORDER BY source, team;
  "
else
  echo "Skipped (no game results parquet)."
fi

echo ""
echo "=============================================="
echo "3b. Same date: raw game_results rows (home_team, away_team) vs unmatched props team_full"
echo "   Date 2023-11-21 (many unmatched)"
echo "=============================================="
if [[ -f "$GAME_RESULTS_PQ" ]]; then
  echo "--- Teams in game_results on 2023-11-21 (home_team, away_team) ---"
  duckdb -c "SELECT game_date, home_team, away_team FROM '$GAME_RESULTS_PQ' WHERE game_date = '2023-11-21' ORDER BY home_team;"
    echo "--- Unmatched props (game_date, team_full) for 2023-11-21 ---"
  duckdb -c "SELECT DISTINCT game_date, team_full FROM '$STRATEGY_PQ' WHERE \"GAME_ID\" IS NULL AND game_date = '2023-11-21' ORDER BY team_full;"
fi

echo ""
echo "=============================================="
echo "3c. Season check: game_results vs strategy for 2023-11-21"
echo "   (If season differs, join on game_date+team_full+season fails)"
echo "=============================================="
if [[ -f "$GAME_RESULTS_PQ" ]]; then
  echo "--- game_results: one row for 2023-11-21 (check for season column) ---"
  duckdb -c "SELECT * FROM '$GAME_RESULTS_PQ' WHERE game_date = '2023-11-21' LIMIT 1;" 2>/dev/null || duckdb -c "SELECT * FROM '$GAME_RESULTS_PQ' WHERE \"GAME_DATE\" = '2023-11-21' LIMIT 1;"
  echo "--- strategy: distinct season for unmatched on 2023-11-21 ---"
  duckdb -c "SELECT DISTINCT season FROM '$STRATEGY_PQ' WHERE \"GAME_ID\" IS NULL AND game_date = '2023-11-21';"
  echo "--- strategy: distinct season for matched on 2023-11-21 ---"
  duckdb -c "SELECT DISTINCT season FROM '$STRATEGY_PQ' WHERE \"GAME_ID\" IS NOT NULL AND game_date = '2023-11-21';"
fi

echo ""
echo "=============================================="
echo "4. Unmatched: count by game_date (are dates missing from game_results?)"
echo "=============================================="
if [[ -f "$GAME_RESULTS_PQ" ]]; then
  duckdb -c "
    WITH unmatched AS (
      SELECT game_date, count(*) AS n FROM '$STRATEGY_PQ' WHERE \"GAME_ID\" IS NULL GROUP BY game_date
    ),
    game_dates AS (
      SELECT DISTINCT game_date FROM '$GAME_RESULTS_PQ'
    )
    SELECT u.game_date, u.n AS unmatched_rows,
           CASE WHEN g.game_date IS NOT NULL THEN 'yes' ELSE 'NO' END AS date_in_game_results
    FROM unmatched u
    LEFT JOIN game_dates g ON u.game_date = g.game_date
    ORDER BY u.game_date
    LIMIT 20;
  "
else
  echo "Skipped (no game results parquet)."
fi

echo ""
echo "=============================================="
echo "5. Summary: total unmatched vs matched"
echo "=============================================="
duckdb -c "
  SELECT
    \"GAME_ID\" IS NOT NULL AS matched,
    count(*) AS rows_
  FROM '$STRATEGY_PQ'
  GROUP BY 1;
"

echo ""
echo "=============================================="
echo "6. CONCRETE EXAMPLES: For 5 unmatched (game_date, team_full), show game_results teams on that date"
echo "   (See if team name differs: e.g. LA Clippers vs Los Angeles Clippers)"
echo "=============================================="
if [[ -f "$GAME_RESULTS_PQ" ]]; then
  duckdb -c "
    WITH un AS (
      SELECT game_date, team_full, count(*) AS n
      FROM '$STRATEGY_PQ'
      WHERE \"GAME_ID\" IS NULL
      GROUP BY game_date, team_full
      ORDER BY game_date, team_full
      LIMIT 5
    )
    SELECT un.game_date, un.team_full AS props_team_full, un.n,
           list(g.home_team || ' vs ' || g.away_team) AS games_that_day
    FROM un
    LEFT JOIN (SELECT game_date, home_team, away_team FROM '$GAME_RESULTS_PQ') g ON g.game_date = un.game_date
    GROUP BY un.game_date, un.team_full, un.n
    ORDER BY un.game_date, un.team_full;
  "
else
  echo "Skipped (no game results parquet)."
fi

echo ""
echo "=============================================="
echo "7. Team name mismatch check: does props team_full appear in game_results for that date?"
echo "   (Compare exact strings: game_results uses home_team/away_team)"
echo "=============================================="
if [[ -f "$GAME_RESULTS_PQ" ]]; then
  duckdb -c "
    WITH un AS (
      SELECT DISTINCT game_date, team_full
      FROM '$STRATEGY_PQ'
      WHERE \"GAME_ID\" IS NULL
      ORDER BY game_date, team_full
      LIMIT 10
    ),
    games_teams AS (
      SELECT game_date, home_team AS team FROM '$GAME_RESULTS_PQ'
      UNION ALL
      SELECT game_date, away_team AS team FROM '$GAME_RESULTS_PQ'
    )
    SELECT un.game_date, un.team_full,
           (SELECT count(*) FROM games_teams g WHERE g.game_date = un.game_date AND g.team = un.team_full) AS exact_match_count,
           (SELECT list(g.team) FROM games_teams g WHERE g.game_date = un.game_date) AS all_teams_in_games_that_day
    FROM un;
  "
else
  echo "Skipped (no game results parquet)."
fi

echo ""
echo "=============================================="
echo "8. Sample 3 full rows of unmatched (player, game_date, team_full) for manual inspection"
echo "=============================================="
duckdb -c "
  SELECT player, game_date, team_full, season
  FROM '$STRATEGY_PQ'
  WHERE \"GAME_ID\" IS NULL
  ORDER BY game_date, team_full, player
  LIMIT 3;
"

echo ""
echo "=============================================="
echo "9. WHY exact_match_count = 0? For one example: did that team play that day?"
echo "   Example: 2023-10-27 Minnesota Timberwolves (unmatched) - did MIN play?"
echo "=============================================="
if [[ -f "$GAME_RESULTS_PQ" ]]; then
  duckdb -c "SELECT game_date, home_team, away_team FROM '$GAME_RESULTS_PQ' WHERE game_date = '2023-10-27' ORDER BY home_team;"
  echo ""
  echo "--> If Minnesota is not in the list above, props have WRONG TEAM (e.g. Gary Trent Jr assigned MIN but he was TOR; TOR played that day). Fix: player_team_history / Jr-Sr name normalization."
fi

echo ""
echo "Done. STRATEGY_PQ=$STRATEGY_PQ GAME_RESULTS_PQ=$GAME_RESULTS_PQ"
echo ""
echo "=============================================="
echo "FINDINGS FROM EXAMPLES:"
echo "  - Section 7: exact_match_count=0 for all sampled unmatched (game_date, team_full)."
echo "  - So props' team_full does NOT appear in game_results for that date."
echo "  - Section 9: For 2023-10-27, Minnesota did not play; unmatched props have team_full=Minnesota (e.g. Gary Trent Jr). So WRONG TEAM was assigned (Jr vs Sr or bad history)."
echo "  - Root cause: player_team_history / name normalization assigns wrong team -> (game_date, team_full) not in games. Fix: Jr-Sr exceptions, rebuild history, or normalize ESPN team names if game_results use 'LA Clippers' vs 'Los Angeles Clippers'."
echo "=============================================="
