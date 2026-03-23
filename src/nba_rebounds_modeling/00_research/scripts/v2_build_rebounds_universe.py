"""
Build v2 rebounds model feature universe.

Context:
- v1 built a minimal universe with MIN/OREB/DREB/REB + canonical bookmaker line.
- v2 extends that by computing a full leakage-safe rolling feature table ready
  for regression modeling.
- Rolling windows [5, 10, 20, 40, 60, 80] are computed for all stat bases.
- Shot-profile stats (FGA, FG3A, FTA) are sourced from the v6 spread universe
  parquet (already cached locally from the 3PM modeling workflow).
- Market context features (consensus_reb_line, line_range, n_books, etc.) are
  derived from the multi-book props table using the same no-vig / closest-to-50
  canonical line selection as v1.
- All rolling features are leakage-safe via shift(1) before the rolling window.
- Output is one row per player/date/game_id with all features + target (REB).
- Also writes v3_rebounds_props_raw.parquet: one row per bookmaker × posted line
  with over/under odds, no-vig probs, REB outcome, consensus_reb_line (for
  per-book backtests).

Usage:
    python src/nba_rebounds_modeling/00_research/scripts/v2_build_rebounds_universe.py \
        --season "*" \
        --output ~/Downloads/tmp/rebounds_model_features_v2.parquet \
        --output-v3 ~/Downloads/tmp/v3_rebounds_props_raw.parquet
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


# =============================================================================
# REPO ROOT
# =============================================================================

def ensure_repo_root_on_syspath() -> Path:
    """Find repo root from cwd and add it to sys.path."""
    current = Path.cwd().resolve()
    while True:
        if (current / ".gitignore").exists() and (current / "src").exists():
            if str(current) not in sys.path:
                sys.path.insert(0, str(current))
            return current
        if current.parent == current:
            raise FileNotFoundError("Could not locate repo root with .gitignore and src/")
        current = current.parent


REPO_ROOT = ensure_repo_root_on_syspath()

import duckdb
from src.player_team_history.name_normalization import normalize_from_nba_api
from src.player_team_history.name_normalization import normalize_from_odds_api


# =============================================================================
# CONFIG
# =============================================================================

WINDOWS = [5, 10, 20, 40, 60, 80]

MEAN_BASES: list[tuple[str, str]] = [
    ("reb",         "REB"),
    ("oreb",        "OREB"),
    ("dreb",        "DREB"),
    ("reb_per_min", "reb_per_min"),
    ("min",         "MIN"),
    ("fga",         "FGA"),
    ("fg3a",        "FG3A"),
    ("fg3a_share",  "fg3a_share"),
    ("fta",         "FTA"),
]

STD_BASES: list[tuple[str, str]] = [
    ("reb", "REB"),
]

REQUIRED_OUTPUT_COLUMNS = [
    "season",
    "date",
    "game_id",
    "player_normalized",
    "REB",
    "consensus_reb_line",
    "max_line",
    "min_line",
    "line_range",
    "n_books",
    "line_spread",
    "spread_signed",
    "spread_abs",
]


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build v2 rebounds model feature universe.")
    p.add_argument("--season",        type=str, default="*")
    p.add_argument("--cache-dir",     type=str, default="~/Downloads/tmp")
    p.add_argument("--use-cache",     type=str, default="true")
    p.add_argument("--force-refresh-cache", type=str, default="false")
    p.add_argument(
        "--output",
        type=str,
        default="~/Downloads/tmp/rebounds_model_features_v2.parquet",
    )
    p.add_argument(
        "--output-v3",
        type=str,
        default="~/Downloads/tmp/v3_rebounds_props_raw.parquet",
    )
    p.add_argument("--seed", type=int, default=69)
    return p.parse_args()


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Unsupported boolean value: {value!r}")


def season_predicate(alias: str, season: str) -> str:
    if season.strip() in ("*", ""):
        return "TRUE"
    values = [x.strip() for x in season.split(",") if x.strip()]
    if len(values) == 1:
        return f"{alias}.season = '{values[0]}'"
    quoted = ", ".join(f"'{x}'" for x in values)
    return f"{alias}.season IN ({quoted})"


# =============================================================================
# HELPERS
# =============================================================================

def require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def maybe_read_cache(path: Path, enabled: bool, force: bool) -> pd.DataFrame | None:
    if enabled and not force and path.exists():
        return pd.read_parquet(path)
    return None


def maybe_write_cache(df: pd.DataFrame, path: Path, enabled: bool) -> None:
    if enabled:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)


def american_to_implied_prob(odds: float) -> float:
    if np.isnan(odds):
        return float("nan")
    if odds < 0:
        return float((-odds) / ((-odds) + 100.0))
    return float(100.0 / (odds + 100.0))


def remove_vig_two_way(p_over: float, p_under: float) -> tuple[float, float]:
    total = p_over + p_under
    if total <= 0.0:
        return 0.5, 0.5
    return float(p_over / total), float(p_under / total)


# =============================================================================
# S3 CONNECTION
# =============================================================================

def connect_duckdb_s3() -> duckdb.DuckDBPyConnection:
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
        if not access_key or not secret_key:
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


# =============================================================================
# DATA LOADING
# =============================================================================

def load_logs(season: str, cache_dir: str, use_cache: bool, force: bool) -> pd.DataFrame:
    """Load player game logs (MIN, OREB, DREB, REB) from S3 with local cache."""
    cache_path = Path(cache_dir).expanduser() / f"v1_rebounds_logs_{season.replace(',', '_')}.parquet"
    cached = maybe_read_cache(cache_path, use_cache, force)
    if cached is not None:
        print(f"logs: loaded from cache ({len(cached):,} rows)")
        return cached

    con = connect_duckdb_s3()
    query = f"""
    WITH raw AS (
      SELECT
        PLAYER_NAME, TEAM_NAME, GAME_ID, GAME_DATE,
        MIN, OREB, DREB, REB,
        regexp_extract(filename, '/player_game_logs/([^/]+)/', 1) AS season
      FROM read_csv_auto(
        's3://nba-api-mt/player_game_logs/*/*.csv',
        union_by_name=true, filename=true
      )
    )
    SELECT * FROM raw r WHERE {season_predicate('r', season)}
    """
    logs = con.execute(query).fetchdf()
    con.close()

    require_columns(
        logs,
        ["season", "PLAYER_NAME", "TEAM_NAME", "GAME_ID", "GAME_DATE", "MIN", "OREB", "DREB", "REB"],
        "raw_logs",
    )

    logs["player_normalized"] = logs["PLAYER_NAME"].apply(normalize_from_nba_api)
    logs["date"] = pd.to_datetime(logs["GAME_DATE"]).dt.date.astype(str)
    for col in ["MIN", "OREB", "DREB", "REB"]:
        logs[col] = pd.to_numeric(logs[col], errors="coerce")

    out = (
        logs[["season", "date", "GAME_ID", "player_normalized", "MIN", "OREB", "DREB", "REB"]]
        .rename(columns={"GAME_ID": "game_id"})
        .copy()
    )

    maybe_write_cache(out, cache_path, use_cache)
    print(f"logs: fetched from S3 ({len(out):,} rows)")
    return out


def load_props(season: str, cache_dir: str, use_cache: bool, force: bool) -> pd.DataFrame:
    """Load player_rebounds props from S3 with local cache."""
    cache_path = Path(cache_dir).expanduser() / f"v1_rebounds_props_{season.replace(',', '_')}.parquet"
    cached = maybe_read_cache(cache_path, use_cache, force)
    if cached is not None:
        print(f"props: loaded from cache ({len(cached):,} rows)")
        return cached

    con = connect_duckdb_s3()
    query = f"""
    WITH raw AS (
      SELECT
        player, bookmaker, game_time, market, prop_line, over_odds, under_odds,
        regexp_extract(filename, '/historical_player_props/([^/]+)/', 1) AS season
      FROM read_csv_auto(
        's3://the-odds-api-mt/nba/historical_player_props/*/*.csv',
        union_by_name=true, filename=true
      )
    )
    SELECT * FROM raw r
    WHERE {season_predicate('r', season)}
      AND market = 'player_rebounds'
      AND prop_line IS NOT NULL
    """
    props = con.execute(query).fetchdf()
    con.close()

    require_columns(
        props,
        ["season", "player", "bookmaker", "game_time", "prop_line", "over_odds", "under_odds"],
        "raw_props",
    )

    props["player_normalized"] = props["player"].apply(normalize_from_odds_api)
    props["date"] = (
        pd.to_datetime(props["game_time"], utc=True)
        .dt.tz_convert("America/New_York")
        .dt.date.astype(str)
    )
    props["line"] = pd.to_numeric(props["prop_line"], errors="coerce")
    props["odds_over"] = pd.to_numeric(props["over_odds"], errors="coerce")
    props["odds_under"] = pd.to_numeric(props["under_odds"], errors="coerce")

    out = props[["season", "date", "player_normalized", "bookmaker", "line", "odds_over", "odds_under"]].copy()
    maybe_write_cache(out, cache_path, use_cache)
    print(f"props: fetched from S3 ({len(out):,} rows)")
    return out


def load_v6_shot_profile(cache_dir: str) -> pd.DataFrame:
    """Load FGA/FG3A/FTA per player/game from the cached v6 spread universe."""
    v6_path = Path(cache_dir).expanduser() / "v6_spread_universe.parquet"
    if not v6_path.exists():
        raise FileNotFoundError(
            f"v6_spread_universe.parquet not found at {v6_path}. "
            "Run the v6 3PM build first."
        )
    v6 = pd.read_parquet(v6_path, columns=["season", "date", "player_normalized", "game_id", "FGA", "FG3A", "FTA"])
    v6 = v6.drop_duplicates(subset=["season", "date", "player_normalized"])
    v6["date"] = pd.to_datetime(v6["date"]).dt.date.astype(str)
    for col in ["FGA", "FG3A", "FTA"]:
        v6[col] = pd.to_numeric(v6[col], errors="coerce")
    print(f"v6 shot profile: loaded ({len(v6):,} rows)")
    return v6


# =============================================================================
# MARKET PANEL
# =============================================================================

def build_market_panel(props: pd.DataFrame, logs: pd.DataFrame) -> pd.DataFrame:
    """
    Build one-row-per-player/game panel with market context features.
    Canonical line = closest no-vig to 50/50 per bookmaker, then aggregated.
    """
    props = props.dropna(subset=["season", "date", "player_normalized", "bookmaker", "line", "odds_over", "odds_under"]).copy()

    props["p_over_raw"] = props["odds_over"].apply(american_to_implied_prob)
    props["p_under_raw"] = props["odds_under"].apply(american_to_implied_prob)
    novig = props.apply(
        lambda r: remove_vig_two_way(float(r["p_over_raw"]), float(r["p_under_raw"])),
        axis=1,
    )
    props["p_over_novig"] = [x[0] for x in novig]
    props["p_under_novig"] = [x[1] for x in novig]
    props["distance_to_5050"] = (props["p_over_novig"] - 0.5).abs()

    # One row per bookmaker / posted line (before picking canonical line per book)
    book_line = (
        props.groupby(["season", "date", "player_normalized", "bookmaker", "line"], as_index=False)
        .agg(
            odds_over=("odds_over", "median"),
            odds_under=("odds_under", "median"),
            p_over_novig=("p_over_novig", "median"),
            p_under_novig=("p_under_novig", "median"),
            distance_to_5050=("distance_to_5050", "median"),
        )
    )
    main_lines = (
        book_line.sort_values(
            ["season", "date", "player_normalized", "bookmaker", "distance_to_5050", "line"]
        )
        .groupby(["season", "date", "player_normalized", "bookmaker"], as_index=False)
        .first()
    )

    # Join outcome
    logs_key = logs[["season", "date", "player_normalized", "game_id"]].drop_duplicates()
    main_lines = main_lines.merge(logs_key, on=["season", "date", "player_normalized"], how="inner")

    group_keys = ["season", "date", "player_normalized", "game_id"]
    panel = (
        main_lines.groupby(group_keys, as_index=False)
        .agg(
            n_books=("bookmaker", "nunique"),
            min_line=("line", "min"),
            max_line=("line", "max"),
            median_line=("line", "median"),
        )
    )
    panel["consensus_reb_line"] = panel["median_line"]
    panel["line_range"] = panel["max_line"] - panel["min_line"]
    panel["line_spread"] = panel["line_range"]

    print(f"market panel: {len(panel):,} rows, {panel['n_books'].mean():.1f} avg books")
    return panel, book_line


def build_v3_props_raw(
    book_line: pd.DataFrame,
    logs: pd.DataFrame,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """
    Per bookmaker × line rows with odds, no-vig, REB, consensus_reb_line.
    """
    group_keys = ["season", "date", "player_normalized", "game_id"]
    logs_gid = logs[["season", "date", "player_normalized", "game_id", "REB"]].drop_duplicates(
        subset=["season", "date", "player_normalized"]
    )
    cons = panel[group_keys + ["consensus_reb_line"]].drop_duplicates(subset=group_keys)

    raw = book_line.merge(
        logs_gid,
        on=["season", "date", "player_normalized"],
        how="inner",
    )
    raw = raw.merge(cons, on=group_keys, how="inner")

    raw = raw.rename(columns={"odds_over": "over_odds", "odds_under": "under_odds"})
    out_cols = [
        "season",
        "date",
        "player_normalized",
        "game_id",
        "bookmaker",
        "line",
        "over_odds",
        "under_odds",
        "p_over_novig",
        "p_under_novig",
        "REB",
        "consensus_reb_line",
    ]
    require_columns(raw, out_cols, "v3_props_raw")
    return raw[out_cols].sort_values(
        ["season", "date", "player_normalized", "game_id", "bookmaker", "line"]
    ).reset_index(drop=True)


# =============================================================================
# SPREAD CONTEXT
# =============================================================================

def attach_spread(panel: pd.DataFrame, logs: pd.DataFrame, cache_dir: str) -> pd.DataFrame:
    """Attach spread_signed from v6 if available; otherwise fills NaN."""
    try:
        v6_path = Path(cache_dir).expanduser() / "v6_spread_universe.parquet"
        v6_spread = pd.read_parquet(
            v6_path, columns=["season", "date", "player_normalized", "spread_signed"]
        )
        v6_spread["date"] = pd.to_datetime(v6_spread["date"]).dt.date.astype(str)
        v6_spread = v6_spread.drop_duplicates(subset=["season", "date", "player_normalized"])
        panel = panel.merge(v6_spread, on=["season", "date", "player_normalized"], how="left")
    except Exception:
        panel["spread_signed"] = np.nan
    panel["spread_abs"] = panel["spread_signed"].abs()
    return panel


# =============================================================================
# ROLLING FEATURES
# =============================================================================

def build_rolling_features(logs: pd.DataFrame, v6_shots: pd.DataFrame) -> pd.DataFrame:
    """
    Build leakage-safe rolling features for all stat bases and all windows.
    All rolling features use shift(1) before the window to prevent leakage.
    """
    logs_ext = logs[["season", "date", "player_normalized", "game_id", "MIN", "OREB", "DREB", "REB"]].copy()
    logs_ext["date"] = pd.to_datetime(logs_ext["date"]).dt.date.astype(str)

    logs_ext = logs_ext.merge(
        v6_shots[["season", "date", "player_normalized", "FGA", "FG3A", "FTA"]],
        on=["season", "date", "player_normalized"],
        how="left",
    )

    logs_ext["fg3a_share"] = (logs_ext["FG3A"] / logs_ext["FGA"].replace(0, np.nan)).fillna(0.0)
    logs_ext["reb_per_min"] = (logs_ext["REB"] / logs_ext["MIN"].replace(0, np.nan)).fillna(0.0)

    # Sort per player by date for correct rolling order
    logs_ext["_sort_date"] = pd.to_datetime(logs_ext["date"])
    logs_ext = logs_ext.sort_values(["player_normalized", "_sort_date"]).reset_index(drop=True)
    logs_ext = logs_ext.drop(columns=["_sort_date"])

    feat_series = []
    all_feat_cols = []

    for name, col in MEAN_BASES:
        for w in WINDOWS:
            feat_name = f"roll_{name}_mean_{w}"
            s = (
                logs_ext.groupby("player_normalized")[col]
                .transform(lambda x, w=w: x.shift(1).rolling(w, min_periods=1).mean())
            )
            feat_series.append(s.rename(feat_name))
            all_feat_cols.append(feat_name)

    for name, col in STD_BASES:
        for w in WINDOWS:
            feat_name = f"roll_{name}_std_{w}"
            s = (
                logs_ext.groupby("player_normalized")[col]
                .transform(lambda x, w=w: x.shift(1).rolling(w, min_periods=2).std())
            )
            feat_series.append(s.rename(feat_name))
            all_feat_cols.append(feat_name)

    roll_df = pd.concat(feat_series, axis=1)
    out = pd.concat(
        [logs_ext[["season", "date", "player_normalized", "game_id"]], roll_df],
        axis=1,
    )

    print(f"rolling features: {len(all_feat_cols)} columns across {len(WINDOWS)} windows")
    return out


# =============================================================================
# QUALITY CHECKS VIA DUCKDB
# =============================================================================

def run_quality_checks(output_path: Path) -> None:
    """Run DuckDB-based quality checks on the written parquet."""
    p = str(output_path)
    con = duckdb.connect()

    print("\n=== QUALITY CHECKS ===")

    # Row count + seasons
    row_count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{p}')").fetchone()[0]
    print(f"total_rows={row_count:,}")

    seasons = con.execute(
        f"SELECT season, COUNT(*) AS n FROM read_parquet('{p}') GROUP BY season ORDER BY season"
    ).fetchdf()
    print("season_coverage:")
    print(seasons.to_string(index=False))

    # Null counts for critical columns
    null_checks = ["REB", "consensus_reb_line", "roll_reb_mean_10", "roll_reb_mean_80"]
    for col in null_checks:
        nulls = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{p}') WHERE {col} IS NULL"
        ).fetchone()[0]
        print(f"null_count_{col}={nulls:,}")

    # Duplicate key check
    dups = con.execute(f"""
        SELECT COUNT(*) FROM (
            SELECT season, date, player_normalized, game_id, COUNT(*) AS n
            FROM read_parquet('{p}')
            GROUP BY season, date, player_normalized, game_id
            HAVING n > 1
        )
    """).fetchone()[0]
    print(f"duplicate_key_count_player_date_game={dups}")
    if dups > 0:
        raise ValueError(f"Duplicate keys found in output: {dups} rows")

    # Sample latest row
    sample = con.execute(f"""
        SELECT season, date, player_normalized, REB, consensus_reb_line,
               roll_reb_mean_10, roll_reb_mean_80, roll_oreb_mean_10, roll_fga_mean_10
        FROM read_parquet('{p}')
        ORDER BY date DESC
        LIMIT 3
    """).fetchdf()
    print("\nsample (latest 3 rows):")
    print(sample.to_string(index=False))

    con.close()
    print("======================\n")


def run_quality_checks_v3(output_path: Path) -> None:
    """DuckDB checks on v3_rebounds_props_raw.parquet."""
    p = str(output_path)
    con = duckdb.connect()
    print("\n=== V3 PROPS RAW QUALITY CHECKS ===")
    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{p}')").fetchone()[0]
    print(f"v3_total_rows={n:,}")
    null_reb = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{p}') WHERE REB IS NULL"
    ).fetchone()[0]
    print(f"v3_null_REB={null_reb:,}")
    dups = con.execute(f"""
        SELECT COUNT(*) FROM (
            SELECT season, date, player_normalized, game_id, bookmaker, line, COUNT(*) AS c
            FROM read_parquet('{p}')
            GROUP BY 1,2,3,4,5,6 HAVING c > 1
        )
    """).fetchone()[0]
    print(f"v3_duplicate_keys={dups}")
    if dups > 0:
        raise ValueError("v3_rebounds_props_raw has duplicate bookmaker/line keys")
    print("=====================================\n")
    con.close()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    use_cache = parse_bool(args.use_cache)
    force = parse_bool(args.force_refresh_cache)
    cache_dir = args.cache_dir
    output_path = Path(args.output).expanduser()
    output_v3_path = Path(args.output_v3).expanduser()

    print(f"season={args.season}  cache_dir={cache_dir}  output={output_path}  v3={output_v3_path}")

    # 1) Load source data
    logs  = load_logs(args.season, cache_dir, use_cache, force)
    props = load_props(args.season, cache_dir, use_cache, force)
    v6_shots = load_v6_shot_profile(cache_dir)

    # 2) Build market panel + per-book line table
    panel, book_line = build_market_panel(props, logs)
    panel = attach_spread(panel, logs, cache_dir)

    v3_raw = build_v3_props_raw(book_line, logs, panel)
    output_v3_path.parent.mkdir(parents=True, exist_ok=True)
    v3_raw.to_parquet(output_v3_path, index=False)
    print(f"v3_rebounds_props_raw written: {output_v3_path} | rows={len(v3_raw):,}")
    run_quality_checks_v3(output_v3_path)

    # 3) Build rolling features
    rolling = build_rolling_features(logs, v6_shots)

    # 4) Join: panel (market features) + rolling (player form) + REB target
    group_keys = ["season", "date", "player_normalized", "game_id"]

    logs_target = (
        logs[group_keys + ["REB"]]
        .drop_duplicates(subset=group_keys)
    )

    feat_v2 = (
        panel[group_keys + [
            "consensus_reb_line", "max_line", "min_line",
            "line_range", "n_books", "line_spread",
            "spread_signed", "spread_abs",
        ]]
        .merge(logs_target, on=group_keys, how="inner")
        .merge(rolling, on=group_keys, how="left")
    )

    # 5) Fail-fast schema check
    require_columns(feat_v2, REQUIRED_OUTPUT_COLUMNS, "feat_v2")

    dup_count = feat_v2.duplicated(subset=group_keys, keep=False).sum()
    if dup_count > 0:
        raise ValueError(f"Duplicate keys in output before write: {dup_count} rows")

    # 6) Sort + write
    feat_v2 = feat_v2.sort_values(["season", "date", "player_normalized", "game_id"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feat_v2.to_parquet(output_path, index=False)

    # 7) DuckDB quality checks
    run_quality_checks(output_path)

    print(
        "phase=v2_build_rebounds_universe",
        f"rows={len(feat_v2)}",
        f"v3_rows={len(v3_raw)}",
        f"season={args.season}",
        f"output={output_path}",
        f"v3={output_v3_path}",
        sep=" | ",
    )


if __name__ == "__main__":
    main()
