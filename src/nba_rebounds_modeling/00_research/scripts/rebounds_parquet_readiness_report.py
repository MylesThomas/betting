"""
Generate a rebounds-modeling readiness report from universe parquet files.

Context:
- Requested for `src/nba_rebounds_modeling/00_research/scripts`.
- Purpose is to quickly validate whether the cached parquet inputs contain the
  required game-to-game fields for NBA rebounds modeling.
- The report includes: row counts, season coverage, player/game coverage,
  key-duplicate checks, and null rates for required columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse CLI args for rebounds parquet readiness checks."""
    parser = argparse.ArgumentParser(
        description="Build readiness diagnostics for rebounds-modeling parquet inputs."
    )
    parser.add_argument(
        "--input-parquet",
        nargs="*",
        default=[],
        help=(
            "Input parquet paths. If omitted, defaults to known universe files in "
            "~/Downloads/tmp."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="",
        help="Optional output CSV path for long-form report rows.",
    )
    return parser.parse_args()


def default_input_paths() -> list[str]:
    """Return default parquet targets used by current modeling workflow."""
    home_tmp = Path("~/Downloads/tmp").expanduser()
    return [
        str(home_tmp / "v6_spread_universe.parquet"),
        str(home_tmp / "v5_eval_universe.parquet"),
    ]


def load_column_names(con: duckdb.DuckDBPyConnection, parquet_path: str) -> list[str]:
    """Load and return ordered parquet column names."""
    describe_df = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{parquet_path}')"
    ).fetchdf()
    return describe_df["column_name"].astype(str).tolist()


def require_columns(parquet_path: str, available: list[str], required: list[str]) -> None:
    """Fail fast when required columns are missing."""
    available_set = set(available)
    missing = [column for column in required if column not in available_set]
    if missing:
        raise ValueError(f"Missing required columns in {parquet_path}: {missing}")


def build_file_report_rows(
    con: duckdb.DuckDBPyConnection,
    parquet_path: str,
    required_cols: list[str],
    key_cols: list[str],
) -> list[dict]:
    """Build long-form report rows for one parquet file."""
    rows: list[dict] = []
    safe_path = parquet_path.replace("'", "''")

    row_count = int(
        con.execute(f"SELECT COUNT(*) AS row_count FROM read_parquet('{safe_path}')").fetchone()[0]
    )
    rows.append(
        {
            "file_path": parquet_path,
            "check_type": "row_count",
            "season": "*",
            "metric_name": "rows_total",
            "metric_value": float(row_count),
        }
    )

    season_counts = con.execute(
        f"""
        SELECT season, COUNT(*) AS n_rows
        FROM read_parquet('{safe_path}')
        GROUP BY season
        ORDER BY season
        """
    ).fetchdf()
    for _, row in season_counts.iterrows():
        rows.append(
            {
                "file_path": parquet_path,
                "check_type": "season_rows",
                "season": str(row["season"]),
                "metric_name": "rows",
                "metric_value": float(row["n_rows"]),
            }
        )

    player_counts = con.execute(
        f"""
        SELECT season, COUNT(DISTINCT player_normalized) AS n_players
        FROM read_parquet('{safe_path}')
        GROUP BY season
        ORDER BY season
        """
    ).fetchdf()
    for _, row in player_counts.iterrows():
        rows.append(
            {
                "file_path": parquet_path,
                "check_type": "coverage",
                "season": str(row["season"]),
                "metric_name": "distinct_players",
                "metric_value": float(row["n_players"]),
            }
        )

    game_counts = con.execute(
        f"""
        SELECT season, COUNT(DISTINCT game_id) AS n_games
        FROM read_parquet('{safe_path}')
        GROUP BY season
        ORDER BY season
        """
    ).fetchdf()
    for _, row in game_counts.iterrows():
        rows.append(
            {
                "file_path": parquet_path,
                "check_type": "coverage",
                "season": str(row["season"]),
                "metric_name": "distinct_games",
                "metric_value": float(row["n_games"]),
            }
        )

    for column in required_cols:
        null_rate = float(
            con.execute(
                f"""
                SELECT AVG(({column} IS NULL)::DOUBLE) AS null_rate
                FROM read_parquet('{safe_path}')
                """
            ).fetchone()[0]
        )
        rows.append(
            {
                "file_path": parquet_path,
                "check_type": "null_rate",
                "season": "*",
                "metric_name": column,
                "metric_value": null_rate,
            }
        )

    key_expr = ", ".join(key_cols)
    duplicate_key_rows = float(
        con.execute(
            f"""
            WITH dupes AS (
                SELECT {key_expr}, COUNT(*) AS n_rows
                FROM read_parquet('{safe_path}')
                GROUP BY {key_expr}
                HAVING COUNT(*) > 1
            )
            SELECT COALESCE(SUM(n_rows), 0) AS duplicate_key_rows
            FROM dupes
            """
        ).fetchone()[0]
    )
    rows.append(
        {
            "file_path": parquet_path,
            "check_type": "key_quality",
            "season": "*",
            "metric_name": "duplicate_rows_by_player_date_game",
            "metric_value": duplicate_key_rows,
        }
    )

    return rows


def print_file_summary(report_df: pd.DataFrame, parquet_path: str) -> None:
    """Print a concise summary for one parquet file."""
    file_df = report_df[report_df["file_path"] == parquet_path].copy()
    total_rows = int(
        file_df[
            (file_df["check_type"] == "row_count")
            & (file_df["metric_name"] == "rows_total")
        ]["metric_value"].iloc[0]
    )
    dup_rows = int(
        file_df[
            (file_df["check_type"] == "key_quality")
            & (file_df["metric_name"] == "duplicate_rows_by_player_date_game")
        ]["metric_value"].iloc[0]
    )
    seasons = file_df[file_df["check_type"] == "season_rows"][
        ["season", "metric_value"]
    ].sort_values("season")
    null_rates = (
        file_df[file_df["check_type"] == "null_rate"][["metric_name", "metric_value"]]
        .sort_values("metric_value", ascending=False)
        .reset_index(drop=True)
    )

    print("=" * 72)
    print(f"file={parquet_path}")
    print(f"rows_total={total_rows}")
    print("season_rows:")
    print(seasons.to_string(index=False))
    print(f"duplicate_rows_by_player_date_game={dup_rows}")
    print("null_rates_desc:")
    print(null_rates.to_string(index=False))
    print()


def main() -> None:
    """Run rebounds parquet readiness checks and optionally save CSV output."""
    args = parse_args()
    input_paths = args.input_parquet if len(args.input_parquet) > 0 else default_input_paths()
    required_columns = [
        "season",
        "player_normalized",
        "date",
        "game_id",
        "MIN",
        "REB",
    ]
    key_columns = ["player_normalized", "date", "game_id"]

    all_rows: list[dict] = []
    for parquet_path in input_paths:
        con = duckdb.connect()
        available_columns = load_column_names(con=con, parquet_path=parquet_path)
        require_columns(parquet_path=parquet_path, available=available_columns, required=required_columns)
        file_rows = build_file_report_rows(
            con=con,
            parquet_path=parquet_path,
            required_cols=required_columns,
            key_cols=key_columns,
        )
        con.close()
        all_rows.extend(file_rows)

    report_df = pd.DataFrame(all_rows).sort_values(
        ["file_path", "check_type", "season", "metric_name"]
    )

    for parquet_path in input_paths:
        print_file_summary(report_df=report_df, parquet_path=parquet_path)

    if args.output_csv.strip() != "":
        output_path = Path(args.output_csv).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(output_path, index=False)
        print(f"report_csv={output_path}")


if __name__ == "__main__":
    main()
