"""
Review v5 workflow outputs using DuckDB SQL over local CSV/parquet files.

Context:
- The v5 workflow prints phase-level status, but research decisions require
  inspecting ranked model tables, recomposition gains, outliers, robustness,
  and calibration outputs.
- This script provides one command to query all saved artifacts in ~/Downloads/tmp
  (or another directory) and print a concise, deterministic summary.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse CLI args for output review script."""
    parser = argparse.ArgumentParser(description="Review v5 output artifacts with DuckDB SQL.")
    parser.add_argument("--tmp-dir", type=str, default="~/Downloads/tmp")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--show-outliers", type=int, default=10)
    return parser.parse_args()


def required_files(tmp_dir: Path) -> dict[str, Path]:
    """Resolve required artifact paths and fail if any are missing."""
    files = {
        "universe": tmp_dir / "v5_eval_universe.parquet",
        "qc": tmp_dir / "v5_eval_universe_qc.csv",
        "min_models": tmp_dir / "v5_min_models.csv",
        "fga_per_min_models": tmp_dir / "v5_fga_per_min_models.csv",
        "fg3_pct_models": tmp_dir / "v5_fg3_pct_models.csv",
        "fg3_pct_trace": tmp_dir / "v5_fg3_pct_trace.csv",
        "recompose_comparison": tmp_dir / "v5_fg3m_recompose_comparison.csv",
        "recompose_predictions": tmp_dir / "v5_fg3m_recompose_predictions.csv",
        "recompose_outliers": tmp_dir / "v5_fg3m_recompose_outliers.csv",
        "segment_metrics": tmp_dir / "v5_robustness_segment_metrics.csv",
        "stability": tmp_dir / "v5_model_stability_summary.csv",
        "prob_calibration": tmp_dir / "v5_prob_calibration.csv",
        "edge_eval": tmp_dir / "v5_edge_bucket_eval.csv",
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if len(missing) > 0:
        raise FileNotFoundError(
            "Missing required output files. Run the v5 workflow first.\n"
            + "\n".join(missing)
        )
    return files


def print_block(title: str, df: pd.DataFrame) -> None:
    """Print one titled dataframe block."""
    print(f"\n=== {title} ===")
    if df.empty:
        print("(no rows)")
        return
    print(df.to_string(index=False))


def query_df(con: duckdb.DuckDBPyConnection, sql: str) -> pd.DataFrame:
    """Run SQL and return dataframe."""
    return con.execute(sql).fetchdf()


def review_universe(con: duckdb.DuckDBPyConnection, files: dict[str, Path]) -> pd.DataFrame:
    """Summarize universe coverage and counts."""
    return query_df(
        con,
        f"""
        SELECT
          COUNT(*) AS n_rows,
          COUNT(DISTINCT season) AS n_seasons,
          COUNT(DISTINCT player_normalized) AS n_players,
          COUNT(DISTINCT game_id) AS n_games,
          MIN(date) AS min_date,
          MAX(date) AS max_date
        FROM read_parquet('{files["universe"]}')
        """,
    )


def review_qc(con: duckdb.DuckDBPyConnection, files: dict[str, Path]) -> pd.DataFrame:
    """Read QC summary rows."""
    return query_df(
        con,
        f"""
        SELECT *
        FROM read_csv_auto('{files["qc"]}')
        ORDER BY check_type, season, metric_name
        """,
    )


def review_target_models(
    con: duckdb.DuckDBPyConnection,
    path: Path,
    target_name: str,
    top_n: int,
) -> pd.DataFrame:
    """Show top-ranked models for one phase-1 target."""
    return query_df(
        con,
        f"""
        SELECT
          '{target_name}' AS target,
          model,
          fit_type,
          features,
          ROUND(rmse, 4) AS rmse,
          ROUND(mae, 4) AS mae,
          ROUND(r2, 4) AS r2,
          ROUND(rmse_gain_vs_baseline, 4) AS rmse_gain_vs_baseline,
          n_total_rows
        FROM read_csv_auto('{path}')
        ORDER BY rmse ASC, model
        LIMIT {int(top_n)}
        """,
    )


def review_trace_tail(
    con: duckdb.DuckDBPyConnection,
    path: Path,
    top_n: int,
) -> pd.DataFrame:
    """Show latest selection trace rows (for sanity checks)."""
    return query_df(
        con,
        f"""
        SELECT *
        FROM read_csv_auto('{path}')
        ORDER BY mode, step DESC
        LIMIT {int(top_n)}
        """,
    )


def review_recomposition(con: duckdb.DuckDBPyConnection, files: dict[str, Path]) -> pd.DataFrame:
    """Show phase-2 model comparison table."""
    return query_df(
        con,
        f"""
        SELECT
          model,
          ROUND(rmse, 4) AS rmse,
          ROUND(mae, 4) AS mae,
          ROUND(r2, 4) AS r2,
          ROUND(rmse_gain_vs_baseline, 4) AS rmse_gain_vs_baseline,
          ROUND(mae_gain_vs_baseline, 4) AS mae_gain_vs_baseline,
          ROUND(residual_abs_p90, 4) AS residual_abs_p90,
          ROUND(residual_abs_p95, 4) AS residual_abs_p95,
          n_total_rows
        FROM read_csv_auto('{files["recompose_comparison"]}')
        ORDER BY rmse ASC, model
        """,
    )


def review_outliers(
    con: duckdb.DuckDBPyConnection,
    files: dict[str, Path],
    show_outliers: int,
) -> pd.DataFrame:
    """Show top absolute residual outliers from phase 2."""
    return query_df(
        con,
        f"""
        SELECT
          model,
          season,
          date,
          player_normalized,
          game_id,
          matchup,
          ROUND(prediction_fg3m, 3) AS pred_fg3m,
          ROUND(actual_fg3m, 3) AS actual_fg3m,
          ROUND(residual, 3) AS residual,
          ROUND(abs_residual, 3) AS abs_residual
        FROM read_csv_auto('{files["recompose_outliers"]}')
        ORDER BY abs_residual DESC, model
        LIMIT {int(show_outliers)}
        """,
    )


def review_stability(con: duckdb.DuckDBPyConnection, files: dict[str, Path]) -> pd.DataFrame:
    """Summarize instability flags across segments."""
    return query_df(
        con,
        f"""
        SELECT
          segment_type,
          COUNT(*) AS n_rows,
          SUM(unstable_flag) AS n_unstable,
          ROUND(100.0 * SUM(unstable_flag) / COUNT(*), 2) AS unstable_pct
        FROM read_csv_auto('{files["stability"]}')
        GROUP BY segment_type
        ORDER BY unstable_pct DESC, segment_type
        """,
    )


def review_calibration(con: duckdb.DuckDBPyConnection, files: dict[str, Path]) -> pd.DataFrame:
    """Show calibration aggregate with brier metrics."""
    return query_df(
        con,
        f"""
        SELECT
          model,
          ROUND(AVG(model_brier), 5) AS model_brier,
          ROUND(AVG(market_brier), 5) AS market_brier,
          ROUND(AVG(calibration_gap_abs), 5) AS avg_calibration_gap,
          SUM(n) AS n_rows
        FROM read_csv_auto('{files["prob_calibration"]}')
        GROUP BY model
        ORDER BY model
        """,
    )


def review_edge(con: duckdb.DuckDBPyConnection, files: dict[str, Path]) -> pd.DataFrame:
    """Show edge bucket realized rates."""
    return query_df(
        con,
        f"""
        SELECT
          model,
          edge_bucket,
          n,
          ROUND(mean_edge, 4) AS mean_edge,
          ROUND(p_model_mean, 4) AS p_model_mean,
          ROUND(p_market_mean, 4) AS p_market_mean,
          ROUND(realized_over_rate, 4) AS realized_over_rate
        FROM read_csv_auto('{files["edge_eval"]}')
        ORDER BY edge_bucket
        """,
    )


def main() -> None:
    """Run all review queries and print a compact analyst summary."""
    args = parse_args()
    tmp_dir = Path(args.tmp_dir).expanduser()
    files = required_files(tmp_dir=tmp_dir)
    con = duckdb.connect()

    universe_summary = review_universe(con=con, files=files)
    qc_summary = review_qc(con=con, files=files)
    min_top = review_target_models(con, files["min_models"], "MIN", args.top_n)
    fga_pm_top = review_target_models(con, files["fga_per_min_models"], "FG3A_per_min", args.top_n)
    pct_top = review_target_models(con, files["fg3_pct_models"], "FG3_PCT", args.top_n)
    trace_tail = review_trace_tail(con, files["fg3_pct_trace"], args.top_n)
    recompose = review_recomposition(con=con, files=files)
    outliers = review_outliers(con=con, files=files, show_outliers=args.show_outliers)
    stability = review_stability(con=con, files=files)
    calibration = review_calibration(con=con, files=files)
    edge = review_edge(con=con, files=files)
    con.close()

    print(f"tmp_dir={tmp_dir}")
    print_block("Universe Summary", universe_summary)
    print_block("Universe QC", qc_summary)
    print_block("Top MIN Models", min_top)
    print_block("Top FG3A_per_min Models", fga_pm_top)
    print_block("Top FG3_PCT Models", pct_top)
    print_block("FG3_PCT Selection Trace (Tail)", trace_tail)
    print_block("FG3M Recomposition Comparison", recompose)
    print_block("Largest Recomposition Outliers", outliers)
    print_block("Segment Stability Summary", stability)
    print_block("Probability Calibration Summary", calibration)
    print_block("Edge Bucket Evaluation", edge)


if __name__ == "__main__":
    main()

