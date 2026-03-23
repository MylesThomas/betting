"""
Build v1 rebounds universe with canonical bookmaker context.

Context:
- The prior v6 spread universe is 3PT-oriented and does not contain all columns
  needed for rebounds modeling workflow evolution.
- This script builds `v1_rebounds_universe.parquet` directly from canonical S3
  sources used by existing NBA modeling flows.
- Output is one row per player/date/game_id with required identity, rebound
  stats, and a single canonical bookmaker+line+odds snapshot selected from the
  player_rebounds market.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


def ensure_repo_root_on_syspath() -> Path:
    """Find repo root from cwd and add it to sys.path."""
    current = Path.cwd().resolve()
    while True:
        gitignore = current / ".gitignore"
        src_dir = current / "src"
        if gitignore.exists() and src_dir.exists():
            repo_root = current
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))
            return repo_root
        if current.parent == current:
            raise FileNotFoundError("Could not locate repo root with .gitignore and src/")
        current = current.parent


REPO_ROOT = ensure_repo_root_on_syspath()

import duckdb
from src.player_team_history.name_normalization import normalize_from_nba_api
from src.player_team_history.name_normalization import normalize_from_odds_api


def parse_args() -> argparse.Namespace:
    """Parse CLI args for v1 rebounds universe build."""
    parser = argparse.ArgumentParser(description="Build v1 rebounds universe.")
    parser.add_argument("--season", type=str, default="*")
    parser.add_argument("--cache-dir", type=str, default="~/Downloads/tmp")
    parser.add_argument("--use-cache", type=str, default="true")
    parser.add_argument("--force-refresh-cache", type=str, default="false")
    parser.add_argument(
        "--output-universe",
        type=str,
        default="~/Downloads/tmp/v1_rebounds_universe.parquet",
    )
    parser.add_argument("--seed", type=int, default=69)
    return parser.parse_args()


def parse_bool(value: str) -> bool:
    """Parse common string boolean variants."""
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Unsupported boolean value: {value}")


def season_predicate(alias: str, season: str) -> str:
    """Build SQL predicate fragment for season filtering."""
    if season.strip() == "*" or season.strip() == "":
        return "TRUE"
    values = [x.strip() for x in season.split(",") if x.strip() != ""]
    if len(values) == 1:
        return f"{alias}.season = '{values[0]}'"
    quoted = ", ".join([f"'{x}'" for x in values])
    return f"{alias}.season IN ({quoted})"


def connect_duckdb_s3() -> duckdb.DuckDBPyConnection:
    """Create a DuckDB connection configured for S3 access in us-east-2."""
    access_key: str
    secret_key: str
    if "AWS_ACCESS_KEY_ID" in os.environ and "AWS_SECRET_ACCESS_KEY" in os.environ:
        access_key = os.environ["AWS_ACCESS_KEY_ID"]
        secret_key = os.environ["AWS_SECRET_ACCESS_KEY"]
    else:
        access_key = subprocess.check_output(
            ["aws", "configure", "get", "aws_access_key_id"], text=True
        ).strip()
        secret_key = subprocess.check_output(
            ["aws", "configure", "get", "aws_secret_access_key"], text=True
        ).strip()
        if access_key == "" or secret_key == "":
            raise ValueError(
                "Missing AWS credentials. Set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY "
                "or configure via `aws configure`."
            )

    con = duckdb.connect()
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")
    con.execute("SET s3_region='us-east-2'")
    con.execute(f"SET s3_access_key_id='{access_key}'")
    con.execute(f"SET s3_secret_access_key='{secret_key}'")
    if "AWS_SESSION_TOKEN" in os.environ:
        con.execute(f"SET s3_session_token='{os.environ['AWS_SESSION_TOKEN']}'")
    return con


def require_columns(df: pd.DataFrame, required_cols: list[str], frame_name: str) -> None:
    """Fail fast when required columns are missing."""
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{frame_name} missing required columns: {missing}")


def maybe_read_cache(cache_path: Path, enabled: bool, force_refresh: bool) -> pd.DataFrame | None:
    """Read parquet cache when enabled and present."""
    if enabled and (not force_refresh) and cache_path.exists():
        return pd.read_parquet(cache_path)
    return None


def maybe_write_cache(df: pd.DataFrame, cache_path: Path, enabled: bool) -> None:
    """Write parquet cache when enabled."""
    if enabled:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)


def american_to_implied_prob(odds: float) -> float:
    """Convert American odds to implied probability."""
    if np.isnan(odds):
        return float("nan")
    if odds < 0:
        return float((-odds) / ((-odds) + 100.0))
    return float(100.0 / (odds + 100.0))


def remove_vig_two_way(p_over: float, p_under: float) -> tuple[float, float]:
    """Remove vig from a two-way market."""
    total = p_over + p_under
    if total <= 0.0:
        return 0.5, 0.5
    return float(p_over / total), float(p_under / total)


def load_rebounds_logs(
    season: str,
    cache_dir: str,
    use_cache: bool,
    force_refresh_cache: bool,
) -> pd.DataFrame:
    """Load player game logs with rebounds columns from S3 with local cache."""
    cache_path = Path(cache_dir).expanduser() / f"v1_rebounds_logs_{season.replace(',', '_')}.parquet"
    cached = maybe_read_cache(cache_path=cache_path, enabled=use_cache, force_refresh=force_refresh_cache)
    if cached is not None:
        return cached

    con = connect_duckdb_s3()
    query = f"""
    WITH raw AS (
      SELECT
        PLAYER_NAME,
        TEAM_NAME,
        GAME_ID,
        GAME_DATE,
        MIN,
        OREB,
        DREB,
        REB,
        regexp_extract(filename, '/player_game_logs/([^/]+)/', 1) AS season
      FROM read_csv_auto(
        's3://nba-api-mt/player_game_logs/*/*.csv',
        union_by_name=true,
        filename=true
      )
    )
    SELECT *
    FROM raw r
    WHERE {season_predicate('r', season)}
    """
    logs = con.execute(query).fetchdf()
    con.close()

    require_columns(
        logs,
        required_cols=["season", "PLAYER_NAME", "TEAM_NAME", "GAME_ID", "GAME_DATE", "MIN", "OREB", "DREB", "REB"],
        frame_name="raw_logs",
    )

    logs["player_normalized"] = logs["PLAYER_NAME"].apply(normalize_from_nba_api)
    logs["team_normalized"] = logs["TEAM_NAME"].astype(str)
    logs["date"] = pd.to_datetime(logs["GAME_DATE"]).dt.date.astype(str)
    for col in ["MIN", "OREB", "DREB", "REB"]:
        logs[col] = pd.to_numeric(logs[col], errors="coerce")

    out = logs[
        [
            "season",
            "date",
            "GAME_ID",
            "player_normalized",
            "team_normalized",
            "MIN",
            "OREB",
            "DREB",
            "REB",
        ]
    ].rename(columns={"GAME_ID": "game_id"})

    maybe_write_cache(df=out, cache_path=cache_path, enabled=use_cache)
    return out


def load_rebounds_props(
    season: str,
    cache_dir: str,
    use_cache: bool,
    force_refresh_cache: bool,
) -> pd.DataFrame:
    """Load player_rebounds props from S3 with local cache."""
    cache_path = Path(cache_dir).expanduser() / f"v1_rebounds_props_{season.replace(',', '_')}.parquet"
    cached = maybe_read_cache(cache_path=cache_path, enabled=use_cache, force_refresh=force_refresh_cache)
    if cached is not None:
        return cached

    con = connect_duckdb_s3()
    query = f"""
    WITH raw AS (
      SELECT
        player,
        bookmaker,
        game_time,
        market,
        prop_line,
        over_odds,
        under_odds,
        regexp_extract(filename, '/historical_player_props/([^/]+)/', 1) AS season
      FROM read_csv_auto(
        's3://the-odds-api-mt/nba/historical_player_props/*/*.csv',
        union_by_name=true,
        filename=true
      )
    )
    SELECT *
    FROM raw r
    WHERE {season_predicate('r', season)}
      AND market = 'player_rebounds'
      AND prop_line IS NOT NULL
    """
    props = con.execute(query).fetchdf()
    con.close()

    require_columns(
        props,
        required_cols=[
            "season",
            "player",
            "bookmaker",
            "game_time",
            "prop_line",
            "over_odds",
            "under_odds",
        ],
        frame_name="raw_props",
    )

    props["player_normalized"] = props["player"].apply(normalize_from_odds_api)
    game_time_utc = pd.to_datetime(props["game_time"], utc=True)
    props["date"] = game_time_utc.dt.tz_convert("America/New_York").dt.date.astype(str)
    props["line"] = pd.to_numeric(props["prop_line"], errors="coerce")
    props["odds_over"] = pd.to_numeric(props["over_odds"], errors="coerce")
    props["odds_under"] = pd.to_numeric(props["under_odds"], errors="coerce")

    out = props[
        [
            "season",
            "date",
            "player_normalized",
            "bookmaker",
            "line",
            "odds_over",
            "odds_under",
        ]
    ].copy()
    maybe_write_cache(df=out, cache_path=cache_path, enabled=use_cache)
    return out


def select_canonical_book_row(props_df: pd.DataFrame) -> pd.DataFrame:
    """Select one canonical bookmaker row per season/date/player."""
    require_columns(
        props_df,
        required_cols=["season", "date", "player_normalized", "bookmaker", "line", "odds_over", "odds_under"],
        frame_name="props_df",
    )

    book_line = (
        props_df.groupby(
            ["season", "date", "player_normalized", "bookmaker", "line"], as_index=False
        )[["odds_over", "odds_under"]]
        .median()
        .sort_values(["season", "date", "player_normalized", "bookmaker", "line"])
        .reset_index(drop=True)
    )
    book_line["p_over_raw"] = book_line["odds_over"].apply(american_to_implied_prob)
    book_line["p_under_raw"] = book_line["odds_under"].apply(american_to_implied_prob)
    no_vig = book_line.apply(
        lambda row: remove_vig_two_way(float(row["p_over_raw"]), float(row["p_under_raw"])),
        axis=1,
    )
    book_line["p_over_novig"] = [x[0] for x in no_vig]
    book_line["distance_to_5050"] = (book_line["p_over_novig"] - 0.5).abs()
    ranked = book_line.sort_values(
        ["season", "date", "player_normalized", "distance_to_5050", "bookmaker", "line"]
    )
    canonical = (
        ranked.groupby(["season", "date", "player_normalized"], as_index=False)
        .first()[["season", "date", "player_normalized", "bookmaker", "line", "odds_over", "odds_under"]]
        .reset_index(drop=True)
    )
    return canonical


def print_quality_checks(df: pd.DataFrame) -> None:
    """Print required quality checks to stdout."""
    print(f"rows_total={len(df)}")

    season_counts = (
        df.groupby("season", as_index=False)
        .agg(rows=("game_id", "count"))
        .sort_values("season")
        .reset_index(drop=True)
    )
    print("season_coverage:")
    print(season_counts.to_string(index=False))

    for col in ["OREB", "DREB", "REB", "MIN"]:
        null_count = int(df[col].isna().sum())
        print(f"null_count_{col}={null_count}")

    dup_count = int(
        df.duplicated(subset=["player_normalized", "date", "game_id"], keep=False).sum()
    )
    print(f"duplicate_key_count_player_date_game={dup_count}")


def main() -> None:
    """Build v1 rebounds universe parquet."""
    args = parse_args()
    np.random.seed(int(args.seed))
    use_cache = parse_bool(args.use_cache)
    force_refresh_cache = parse_bool(args.force_refresh_cache)

    logs = load_rebounds_logs(
        season=args.season,
        cache_dir=args.cache_dir,
        use_cache=use_cache,
        force_refresh_cache=force_refresh_cache,
    )
    props = load_rebounds_props(
        season=args.season,
        cache_dir=args.cache_dir,
        use_cache=use_cache,
        force_refresh_cache=force_refresh_cache,
    )
    canonical_props = select_canonical_book_row(props_df=props)

    merged = logs.merge(
        canonical_props,
        on=["season", "date", "player_normalized"],
        how="inner",
    )
    required_output_columns = [
        "season",
        "date",
        "game_id",
        "player_normalized",
        "team_normalized",
        "MIN",
        "OREB",
        "DREB",
        "REB",
        "bookmaker",
        "line",
        "odds_over",
        "odds_under",
    ]
    require_columns(
        merged,
        required_cols=required_output_columns,
        frame_name="merged_output",
    )

    duplicate_count = int(
        merged.duplicated(subset=["player_normalized", "date", "game_id"], keep=False).sum()
    )
    if duplicate_count > 0:
        raise ValueError(
            "Duplicate keys found for (player_normalized, date, game_id): "
            f"{duplicate_count} rows"
        )

    out = merged[required_output_columns].sort_values(
        ["season", "date", "player_normalized", "game_id"]
    )

    output_path = Path(args.output_universe).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)

    print_quality_checks(out)
    print(
        "phase=v1_build_rebounds_universe",
        f"rows={len(out)}",
        f"season={args.season}",
        f"output={output_path}",
        sep=" | ",
    )


if __name__ == "__main__":
    main()
