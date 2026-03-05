"""
Show a compact 5-step report for a backtest run.

Context:
- After running `run_backtest.py` and `validate_runs.py`, we want a single script
  to inspect results quickly.
- Defaults to the most recent `runs/{run_id}` folder.
- Supports `--run-id` to inspect a specific historical run.
- Mirrors the 5 checks used manually in terminal:
  1) artifact presence
  2) summary metrics
  3) validation/comparison
  4) bets + side aggregates
  5) priced predictions/contracts
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd
import yaml


def _resolve_run_dir(runs_dir: Path, run_id: str | None) -> Path:
    if run_id is not None:
        return runs_dir / run_id
    run_dirs = sorted([p for p in runs_dir.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    return run_dirs[0]


def _print_section(title: str, guiding_question: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    print(f"Question: {guiding_question}")


def _run_query(con: duckdb.DuckDBPyConnection, sql: str) -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_colwidth", 200)
    df = con.execute(sql).fetchdf()
    print(df.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Show latest or specified backtest run report")
    parser.add_argument("--run-id", type=str, default=None, help="Optional run_id folder name")
    parser.add_argument("--pred-limit", type=int, default=100, help="Rows to show for predictions table")
    parser.add_argument("--bet-limit", type=int, default=50, help="Rows to show for bets table")
    args = parser.parse_args()

    runs_dir = Path(__file__).resolve().parent / "runs"
    run_dir = _resolve_run_dir(runs_dir=runs_dir, run_id=args.run_id)

    summary_path = run_dir / "summary.json"
    validation_path = run_dir / "validation_summary.json"
    comparison_path = run_dir / "comparison_table.csv"
    predictions_path = run_dir / "predictions.parquet"
    bets_path = run_dir / "bets.parquet"
    config_path = run_dir / "config.yaml"
    manifest_path = run_dir / "manifest.json"
    run_config = yaml.safe_load(config_path.read_text())

    print(f"run_dir: {run_dir}")

    _print_section(
        "1) Artifact Presence",
        "Did this run produce every required output artifact for reproducibility?",
    )
    for path in [config_path, manifest_path, predictions_path, bets_path, summary_path, validation_path, comparison_path]:
        print(path.name)

    con = duckdb.connect()

    _print_section(
        "2) Summary Metrics",
        "How did this run perform at a high level (RMSE, win rate, ROI, signal rate)?",
    )
    _run_query(
        con,
        f"""
        SELECT
          run_id,
          player_name,
          season,
          n_games_played,
          n_games_with_priced_lines,
          n_predictions,
          n_bets,
          ROUND(rmse, 3) AS rmse,
          ROUND(win_rate, 3) AS win_rate,
          ROUND(roi, 3) AS roi,
          ROUND(total_pnl, 3) AS total_pnl,
          ROUND(total_risked, 3) AS total_risked,
          ROUND(signal_rate, 3) AS signal_rate
        FROM read_json_auto('{summary_path.as_posix()}');
        """,
    )

    _print_section(
        "3) Validation + Comparison",
        "How does this run compare versus baseline/prior runs?",
    )
    _run_query(
        con,
        f"""
        SELECT
          current_run,
          baseline_run,
          ROUND(roi_delta_vs_baseline, 3) AS roi_delta_vs_baseline,
          ROUND(rmse_delta_vs_baseline, 3) AS rmse_delta_vs_baseline,
          ROUND(win_rate_delta_vs_baseline, 3) AS win_rate_delta_vs_baseline,
          ROUND(signal_rate_delta_vs_baseline, 3) AS signal_rate_delta_vs_baseline
        FROM read_json_auto('{validation_path.as_posix()}');
        """,
    )
    _run_query(
        con,
        f"""
        SELECT
          run_id,
          ROUND(rmse, 3) AS rmse,
          ROUND(win_rate, 3) AS win_rate,
          ROUND(roi, 3) AS roi,
          n_bets,
          ROUND(signal_rate, 3) AS signal_rate
        FROM read_csv_auto('{comparison_path.as_posix()}', header=true)
        ORDER BY run_id;
        """,
    )

    _print_section(
        "4) Bets + Side Aggregates",
        "What would we have actually bet, and how did those bets perform?",
    )
    print(f"Note: section 4 'odds' uses evaluation_price_view='{run_config['evaluation_price_view']}'")
    _run_query(
        con,
        f"""
        SELECT
          date,
          game_id,
          ROUND(line, 3) AS line,
          side,
          ROUND(odds, 3) AS odds,
          ROUND(
            CASE
              WHEN odds > 0 THEN 100.0 / (odds + 100.0)
              ELSE (-odds) / ((-odds) + 100.0)
            END,
            3
          ) AS p_implied,
          ROUND(stake, 3) AS stake,
          ROUND(p_model, 3) AS p_model,
          ROUND(edge, 3) AS edge,
          result,
          ROUND(pnl, 3) AS pnl
        FROM '{bets_path.as_posix()}'
        ORDER BY date, line
        LIMIT {args.bet_limit};
        """,
    )
    _run_query(
        con,
        f"""
        SELECT
          side,
          COUNT(*) AS n_bets,
          ROUND(AVG(pnl), 3) AS avg_pnl,
          ROUND(SUM(pnl), 3) AS total_pnl,
          ROUND(SUM(stake), 3) AS total_stake,
          ROUND(SUM(pnl) / NULLIF(SUM(stake), 0), 3) AS roi
        FROM '{bets_path.as_posix()}'
        GROUP BY side
        ORDER BY side;
        """,
    )

    _print_section(
        "5) Priced Predictions / Contracts",
        "What did the model think for every available line?",
    )
    print("Data dictionary:")
    print("- date: game date (ET)")
    print("- line: player_threes line value")
    print("- cons: 1 if consensus line, else 0")
    print("- yhat: model expected FG3M")
    print("- fg3m: actual made threes")
    print("- p_o: model probability over")
    print("- p_u: model probability under")
    print("- pio_raw_o: raw implied prob over")
    print("- pio_raw_u: raw implied prob under")
    print("- pio_nv_o: no-vig implied prob over")
    print("- pio_nv_u: no-vig implied prob under")
    print("- e_raw_o: model edge over vs raw")
    print("- e_raw_u: model edge under vs raw")
    print("- e_nv_o: model edge over vs no-vig")
    print("- e_nv_u: model edge under vs no-vig")
    print("- odds_best_o: best available over odds")
    print("- odds_best_u: best available under odds")
    print("- odds_med_o: median over odds")
    print("- odds_med_u: median under odds")
    print("- book_best_o: sportsbook for best over")
    print("- book_best_u: sportsbook for best under")
    print("- best_bet: direction from larger raw edge")
    print("- actual_bet: best_bet after threshold filter")
    _run_query(
        con,
        f"""
        SELECT
          date,
          ROUND(line, 3) AS line,
          is_consensus AS cons,
          ROUND(y_hat, 3) AS yhat,
          ROUND(actual_fg3m, 3) AS fg3m,
          ROUND(p_over, 3) AS p_o,
          ROUND(p_under, 3) AS p_u,
          ROUND(p_implied_over_raw, 3) AS pio_raw_o,
          ROUND(p_implied_under_raw, 3) AS pio_raw_u,
          ROUND(p_implied_over_novig, 3) AS pio_nv_o,
          ROUND(p_implied_under_novig, 3) AS pio_nv_u,
          ROUND(edge_over_raw, 3) AS e_raw_o,
          ROUND(edge_under_raw, 3) AS e_raw_u,
          ROUND(edge_over_novig, 3) AS e_nv_o,
          ROUND(edge_under_novig, 3) AS e_nv_u,
          ROUND(best_over_odds, 3) AS odds_best_o,
          ROUND(best_under_odds, 3) AS odds_best_u,
          ROUND(median_over_odds, 3) AS odds_med_o,
          ROUND(median_under_odds, 3) AS odds_med_u,
          best_over_book AS book_best_o,
          best_under_book AS book_best_u,
          CASE
            WHEN GREATEST(edge_over_raw, edge_under_raw) <= 0 THEN 'na'
            WHEN edge_over_raw >= edge_under_raw THEN 'over'
            ELSE 'under'
          END AS best_bet,
          CASE
            WHEN GREATEST(edge_over_raw, edge_under_raw) <= 0 THEN 'na'
            WHEN GREATEST(edge_over_raw, edge_under_raw) < {run_config['edge_threshold']} THEN 'na'
            WHEN edge_over_raw >= edge_under_raw THEN 'over'
            ELSE 'under'
          END AS actual_bet
        FROM '{predictions_path.as_posix()}'
        ORDER BY date, line
        LIMIT {args.pred_limit};
        """,
    )
    con.close()


if __name__ == "__main__":
    main()

