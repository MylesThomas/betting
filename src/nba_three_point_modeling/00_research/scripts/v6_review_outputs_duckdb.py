"""
Review v6 spread-context outputs using DuckDB SQL over local artifacts.

Context:
- The v6 workflow produces spread universe QC, target/model metrics, per-bin
  effects, and ranked targets.
- This script prints a concise analyst table view that highlights strongest and
  weakest spread-sensitive targets for promotion decisions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse CLI args for v6 output review."""
    parser = argparse.ArgumentParser(description="Review v6 spread workflow outputs.")
    parser.add_argument("--tmp-dir", type=str, default="~/Downloads/tmp")
    parser.add_argument("--top-n", type=int, default=8)
    return parser.parse_args()


def required_files(tmp_dir: Path) -> dict[str, Path]:
    """Resolve required artifact paths and fail fast if missing."""
    files = {
        "universe": tmp_dir / "v6_spread_universe.parquet",
        "qc": tmp_dir / "v6_spread_universe_qc.csv",
        "summary": tmp_dir / "v6_spread_model_summary.csv",
        "bin_effects": tmp_dir / "v6_spread_bin_effects.csv",
        "ranked": tmp_dir / "v6_spread_ranked_targets.csv",
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if len(missing) > 0:
        raise FileNotFoundError(
            "Missing required output files. Run v6 workflow first.\n" + "\n".join(missing)
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
    """Summarize coverage and spread context completeness."""
    return query_df(
        con,
        f"""
        SELECT
          COUNT(*) AS n_rows,
          COUNT(DISTINCT season) AS n_seasons,
          COUNT(DISTINCT player_normalized) AS n_players,
          COUNT(DISTINCT game_id) AS n_games,
          ROUND(100.0 * AVG(CASE WHEN spread_signed IS NOT NULL THEN 1 ELSE 0 END), 2) AS pct_with_spread,
          MIN(date) AS min_date,
          MAX(date) AS max_date
        FROM read_parquet('{files["universe"]}')
        """,
    )


def review_qc(con: duckdb.DuckDBPyConnection, files: dict[str, Path]) -> pd.DataFrame:
    """Show QC metrics."""
    return query_df(
        con,
        f"""
        SELECT *
        FROM read_csv_auto('{files["qc"]}')
        ORDER BY check_type, metric_name
        """,
    )


def review_top_targets(
    con: duckdb.DuckDBPyConnection,
    files: dict[str, Path],
    top_n: int,
) -> pd.DataFrame:
    """Show strongest spread-sensitive targets by ranking table."""
    return query_df(
        con,
        f"""
        SELECT
          target_rank,
          target,
          model,
          n_rows,
          ROUND(rmse_gain_vs_baseline, 5) AS rmse_gain_vs_baseline,
          ROUND(r2_gain_vs_baseline, 5) AS r2_gain_vs_baseline,
          ROUND(mae_gain_vs_baseline, 5) AS mae_gain_vs_baseline
        FROM read_csv_auto('{files["ranked"]}')
        ORDER BY target_rank
        LIMIT {int(top_n)}
        """,
    )


def review_weak_targets(
    con: duckdb.DuckDBPyConnection,
    files: dict[str, Path],
    top_n: int,
) -> pd.DataFrame:
    """Show weakest/no-signal targets from ranking table."""
    return query_df(
        con,
        f"""
        SELECT
          target_rank,
          target,
          model,
          n_rows,
          ROUND(rmse_gain_vs_baseline, 5) AS rmse_gain_vs_baseline,
          ROUND(r2_gain_vs_baseline, 5) AS r2_gain_vs_baseline,
          ROUND(mae_gain_vs_baseline, 5) AS mae_gain_vs_baseline
        FROM read_csv_auto('{files["ranked"]}')
        ORDER BY target_rank DESC
        LIMIT {int(top_n)}
        """,
    )


def review_linear_diagnostics(con: duckdb.DuckDBPyConnection, files: dict[str, Path]) -> pd.DataFrame:
    """Show spread_linear and consensus+spread coefficient diagnostics."""
    return query_df(
        con,
        f"""
        SELECT
          target,
          model,
          n_rows,
          ROUND(rmse, 5) AS rmse,
          ROUND(r2, 5) AS r2,
          ROUND(intercept, 6) AS intercept,
          ROUND(coef_market_consensus_line, 6) AS coef_market_consensus_line,
          ROUND(coef_spread_signed, 6) AS coef_spread_signed,
          ROUND(p_value_spread_signed, 6) AS p_value_spread_signed,
          ROUND(ci_low_spread_signed, 6) AS ci_low_spread_signed,
          ROUND(ci_high_spread_signed, 6) AS ci_high_spread_signed
        FROM read_csv_auto('{files["summary"]}')
        WHERE model IN ('spread_linear', 'consensus_plus_spread')
        ORDER BY target, model
        """,
    )


def review_ranked_equations(
    con: duckdb.DuckDBPyConnection,
    files: dict[str, Path],
    top_n: int,
) -> pd.DataFrame:
    """Show equation strings for ranked best models."""
    return query_df(
        con,
        f"""
        SELECT
          r.target_rank,
          r.target,
          r.model,
          ROUND(s.intercept, 6) AS intercept,
          ROUND(s.coef_market_consensus_line, 6) AS coef_market_consensus_line,
          ROUND(s.coef_spread_signed, 6) AS coef_spread_signed,
          s.equation
        FROM read_csv_auto('{files["ranked"]}') r
        INNER JOIN read_csv_auto('{files["summary"]}') s
          ON r.target = s.target
         AND r.model = s.model
        ORDER BY r.target_rank
        LIMIT {int(top_n)}
        """,
    )


def review_bin_pattern_examples(
    con: duckdb.DuckDBPyConnection,
    files: dict[str, Path],
    top_n: int,
) -> pd.DataFrame:
    """Show bin effect rows for top-ranked targets."""
    return query_df(
        con,
        f"""
        WITH top_targets AS (
          SELECT target
          FROM read_csv_auto('{files["ranked"]}')
          ORDER BY target_rank
          LIMIT {int(top_n)}
        )
        SELECT
          b.target,
          b.spread_bin,
          b.n_rows,
          ROUND(b.mean_outcome, 5) AS mean_outcome,
          ROUND(b.delta_vs_neutral, 5) AS delta_vs_neutral
        FROM read_csv_auto('{files["bin_effects"]}') b
        INNER JOIN top_targets t ON b.target = t.target
        ORDER BY b.target, b.spread_bin
        """,
    )


def main() -> None:
    """Run all v6 review queries and print concise decision tables."""
    args = parse_args()
    tmp_dir = Path(args.tmp_dir).expanduser()
    files = required_files(tmp_dir=tmp_dir)
    con = duckdb.connect()

    universe = review_universe(con=con, files=files)
    qc = review_qc(con=con, files=files)
    strongest = review_top_targets(con=con, files=files, top_n=args.top_n)
    weakest = review_weak_targets(con=con, files=files, top_n=args.top_n)
    linear = review_linear_diagnostics(con=con, files=files)
    equations = review_ranked_equations(con=con, files=files, top_n=args.top_n)
    bins = review_bin_pattern_examples(con=con, files=files, top_n=args.top_n)
    con.close()

    print(f"tmp_dir={tmp_dir}")
    print_block("Universe Summary", universe)
    print_block("Universe QC", qc)
    print_block("Strongest Spread-Sensitive Targets", strongest)
    print_block("Weak / No-Signal Targets", weakest)
    print_block("Linear Spread Diagnostics", linear)
    print_block("Ranked Model Equations", equations)
    print_block("Top-Target Bin Effects", bins)


if __name__ == "__main__":
    main()
