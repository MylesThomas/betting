"""
Build the full (historical) rebounds model feature universe.

Context:
- v1 was a minimal universe (MIN/OREB/DREB/REB + canonical bookmaker line). This
  script is the current full pipeline: a leakage-safe rolling feature table for
  regression, plus per-book v3-style props for backtesting.
- Rolling windows [5, 10, 20, 40, 60, 80] are computed for all stat bases.
- Shot-profile stats (FGA, FG3A, FTA) are sourced from the shot-profile
  universe parquet (cached from the 3PM modeling workflow).
- Market context features (consensus_reb_line, line_range, n_books, etc.) are
  derived from the multi-book props table using the same no-vig / closest-to-50
  canonical line selection as v1.
- All rolling features are leakage-safe via shift(1) before the rolling window.
- Output is one row per player/date/game_id with all features + target (REB).
- Also writes v3_rebounds_props_raw.parquet: one row per bookmaker × posted line
  with over/under odds, no-vig probs, REB outcome, consensus_reb_line (for
  per-book backtests).

Usage:
    python src/nba_rebounds_modeling/00_research/scripts/build_rebounds_full_universe.py \\
        --season "*" \\
        --output ~/Downloads/tmp/rebounds_full_universe.parquet \\
        --output-v3 ~/Downloads/tmp/v3_rebounds_props_raw.parquet
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
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
from src.nba_rebounds_modeling.duckdb_s3_creds import connect_duckdb_s3
from src.nba_rebounds_modeling.rebounds_audit_list_verify import (
    print_audit_sample_to_stdout,
    sample_audit_rows,
    verify_audit_lists_dataframe,
)
from src.nba_rebounds_modeling.rebounds_feature_spec import B_MIN_MAX_AUDIT_LIST_COLS, TEAM_CONTEXT_COLS
from src.player_team_history.name_normalization import normalize_from_nba_api
from src.player_team_history.name_normalization import normalize_from_odds_api
from src.player_team_history.team_normalization import normalize_team_name_from_odds_api


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
    *B_MIN_MAX_AUDIT_LIST_COLS,
    *TEAM_CONTEXT_COLS,
]


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the full rebounds model feature universe.")
    p.add_argument("--season",        type=str, default="*")
    p.add_argument("--cache-dir",     type=str, default="~/Downloads/tmp")
    p.add_argument("--use-cache",     type=str, default="true")
    p.add_argument("--force-refresh-cache", type=str, default="false")
    p.add_argument(
        "--output",
        type=str,
        default="~/Downloads/tmp/rebounds_full_universe.parquet",
    )
    p.add_argument(
        "--output-v3",
        type=str,
        default="~/Downloads/tmp/v3_rebounds_props_raw.parquet",
    )
    p.add_argument("--seed", type=int, default=69)
    p.add_argument(
        "--skip-audit-list",
        action="store_true",
        help="Skip audit-list vs scalar checks (emergency only; not recommended for prod).",
    )
    p.add_argument(
        "--audit-list-full-scan",
        action="store_true",
        help="Verify every row (slow). Same as env REBOUNDS_AUDIT_LIST_FULL_SCAN=1.",
    )
    p.add_argument(
        "--audit-list-max-rows",
        type=int,
        default=500,
        metavar="N",
        help="Random sample size for audit when not full-scan (default 500).",
    )
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
# DATA LOADING
# =============================================================================

def load_logs(season: str, cache_dir: str, use_cache: bool, force: bool) -> pd.DataFrame:
    """Load player game logs (MIN, OREB, DREB, REB) from S3 with local cache."""
    cache_path = Path(cache_dir).expanduser() / f"v1_rebounds_logs_{season.replace(',', '_')}.parquet"
    cached = maybe_read_cache(cache_path, use_cache, force)
    if cached is not None:
        if "team_normalized" not in cached.columns:
            print("logs: cache missing team_normalized; refetching")
            cached = None
        else:
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
    logs["team_normalized"] = logs["TEAM_NAME"].astype(str).map(normalize_team_name_from_odds_api)

    out = (
        logs[
            [
                "season",
                "date",
                "GAME_ID",
                "player_normalized",
                "MIN",
                "OREB",
                "DREB",
                "REB",
                "team_normalized",
            ]
        ]
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
    """Load FGA/FG3A/FTA per player/game from rebounds input universe."""
    input_path = Path(cache_dir).expanduser() / "rebounds_input_universe.parquet"
    if not input_path.exists():
        raise FileNotFoundError(
            f"rebounds_input_universe.parquet not found at {input_path}. "
            "Run build_rebounds_input_universe.py first."
        )
    input_df = pd.read_parquet(
        input_path,
        columns=["season", "date", "player_normalized", "game_id", "FGA", "FG3A", "FTA"],
    )
    input_df = input_df.drop_duplicates(subset=["season", "date", "player_normalized"])
    input_df["date"] = pd.to_datetime(input_df["date"]).dt.date.astype(str)
    for col in ["FGA", "FG3A", "FTA"]:
        input_df[col] = pd.to_numeric(input_df[col], errors="coerce")
    print(f"input universe shot profile: loaded ({len(input_df):,} rows)")
    return input_df


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

    lines_agg = (
        main_lines.groupby(group_keys, as_index=False)
        .agg(
            input_reb_prop_lines=(
                "line",
                lambda s: sorted({float(x) for x in pd.to_numeric(s, errors="coerce").dropna().unique()}),
            )
        )
    )
    panel = panel.merge(lines_agg, on=group_keys, how="left")

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

def load_game_level_spreads(season: str, cache_dir: str, use_cache: bool, force: bool) -> pd.DataFrame:
    """Per game: normalized home/away team names and median spread lines (Odds API historical CSV)."""
    cache_path = Path(cache_dir).expanduser() / f"rebounds_nba_game_spreads_{season.replace(',', '_')}.parquet"
    cached = maybe_read_cache(cache_path, use_cache, force)
    if cached is not None:
        cached["pair_key"] = cached["pair_key"].apply(lambda x: tuple(x) if x is not None else x)
        return cached

    con = connect_duckdb_s3()
    query = f"""
    WITH raw AS (
      SELECT
        home_team,
        away_team,
        market,
        home_line,
        away_line,
        regexp_extract(filename, '/historical_game_lines/([^/]+)/', 1) AS season,
        regexp_extract(filename, 'nba_game_lines_(\\d{{4}}-\\d{{2}}-\\d{{2}})\\.csv', 1) AS date
      FROM read_csv_auto(
        's3://the-odds-api-mt/nba/historical_game_lines/*/nba_game_lines_*.csv',
        union_by_name=true,
        filename=true,
        all_varchar=true
      )
    ),
    spread AS (
      SELECT
        season,
        date,
        home_team,
        away_team,
        median(CAST(home_line AS DOUBLE)) AS home_spread,
        median(CAST(away_line AS DOUBLE)) AS away_spread
      FROM raw r
      WHERE {season_predicate('r', season)}
        AND market = 'spread'
        AND home_line IS NOT NULL
        AND away_line IS NOT NULL
      GROUP BY season, date, home_team, away_team
    )
    SELECT season, date, home_team, away_team, home_spread, away_spread
    FROM spread
    """
    spread_df = con.execute(query).fetchdf()
    con.close()
    spread_df["home_team_norm"] = spread_df["home_team"].astype(str).map(normalize_team_name_from_odds_api)
    spread_df["away_team_norm"] = spread_df["away_team"].astype(str).map(normalize_team_name_from_odds_api)
    spread_df["home_spread"] = pd.to_numeric(spread_df["home_spread"], errors="coerce")
    spread_df["away_spread"] = pd.to_numeric(spread_df["away_spread"], errors="coerce")
    spread_df["pair_key"] = spread_df.apply(
        lambda r: tuple(sorted([r["home_team_norm"], r["away_team_norm"]])),
        axis=1,
    )
    spread_df = spread_df.drop_duplicates(subset=["season", "date", "pair_key"], keep="first")
    maybe_write_cache(spread_df, cache_path, use_cache)
    return spread_df


def _logs_game_pair_keys(logs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (season, date, gid), g in logs.groupby(["season", "date", "game_id"], sort=False):
        u = sorted({x for x in g["team_normalized"].dropna().unique().tolist() if str(x).strip()})
        pk = tuple(u) if len(u) == 2 else None
        rows.append({"season": season, "date": date, "game_id": gid, "pair_key": pk})
    return pd.DataFrame(rows)


def attach_spread(
    panel: pd.DataFrame,
    logs: pd.DataFrame,
    cache_dir: str,
    season: str,
    use_cache: bool,
    force: bool,
) -> pd.DataFrame:
    """
    Attach spread_signed from rebounds_input_universe when present; spread_abs; and
    ``input_spread_by_side`` as ``[home_spread, away_spread]`` from historical game lines
    joined via (season, date, game_id) and team pairing from logs.
    """
    gspread = load_game_level_spreads(season, cache_dir, use_cache, force)
    pair_df = _logs_game_pair_keys(logs)
    gattach = pair_df.merge(
        gspread,
        on=["season", "date", "pair_key"],
        how="left",
    )

    def _spread_pair(row: pd.Series):
        hs = row.get("home_spread")
        aws = row.get("away_spread")
        if pd.isna(hs) or pd.isna(aws):
            return np.nan
        return [float(hs), float(aws)]

    gattach["input_spread_by_side"] = gattach.apply(_spread_pair, axis=1)
    spread_merge_cols = [
        c
        for c in (
            "season",
            "date",
            "game_id",
            "input_spread_by_side",
            "home_team_norm",
            "away_team_norm",
        )
        if c in gattach.columns
    ]
    panel = panel.merge(
        gattach[spread_merge_cols],
        on=["season", "date", "game_id"],
        how="left",
    )

    try:
        input_path = Path(cache_dir).expanduser() / "rebounds_input_universe.parquet"
        input_spread = pd.read_parquet(
            input_path, columns=["season", "date", "player_normalized", "spread_signed"]
        )
        input_spread["date"] = pd.to_datetime(input_spread["date"]).dt.date.astype(str)
        input_spread = input_spread.drop_duplicates(subset=["season", "date", "player_normalized"])
        panel = panel.merge(input_spread, on=["season", "date", "player_normalized"], how="left")
    except Exception:
        panel["spread_signed"] = np.nan
    panel["spread_abs"] = panel["spread_signed"].abs()
    return panel


def build_audit_team_frame(logs: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Per player-game: team + home/away norms for ``verify_audit_lists_dataframe`` spread checks."""
    group_keys = ["season", "date", "player_normalized", "game_id"]
    lt = logs[group_keys + ["team_normalized"]].drop_duplicates(subset=group_keys)
    if "home_team_norm" not in panel.columns or "away_team_norm" not in panel.columns:
        return lt
    game_sides = panel[
        ["season", "date", "game_id", "home_team_norm", "away_team_norm"]
    ].drop_duplicates(subset=["season", "date", "game_id"])
    return lt.merge(game_sides, on=["season", "date", "game_id"], how="left")


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

    # Audit tails: prior games only (indices [max(0,i-w):i]) aligned with shift(1).rolling(w).
    reb_tail_60 = pd.Series(index=logs_ext.index, dtype=object)
    fg3_tail_20 = pd.Series(index=logs_ext.index, dtype=object)
    reb_tail_5 = pd.Series(index=logs_ext.index, dtype=object)
    for _, g in logs_ext.groupby("player_normalized", sort=False):
        reb = g["REB"].to_numpy(dtype=float)
        fg3 = g["FG3A"].to_numpy(dtype=float)
        nloc = len(g)
        r60 = [reb[max(0, i - 60) : i].tolist() for i in range(nloc)]
        f20 = [fg3[max(0, i - 20) : i].tolist() for i in range(nloc)]
        r5 = [reb[max(0, i - 5) : i].tolist() for i in range(nloc)]
        reb_tail_60.loc[g.index] = r60
        fg3_tail_20.loc[g.index] = f20
        reb_tail_5.loc[g.index] = r5

    tail_df = pd.DataFrame(
        {
            "input_reb_tail_60": reb_tail_60,
            "input_fg3a_tail_20": fg3_tail_20,
            "input_reb_tail_5": reb_tail_5,
        }
    )

    out = pd.concat(
        [logs_ext[["season", "date", "player_normalized", "game_id"]], roll_df, tail_df],
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
    panel = attach_spread(panel, logs, cache_dir, args.season, use_cache, force)

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
        logs[group_keys + ["REB", "team_normalized"]]
        .drop_duplicates(subset=group_keys)
    )

    panel_cols = [
        "consensus_reb_line",
        "max_line",
        "min_line",
        "line_range",
        "n_books",
        "line_spread",
        "spread_signed",
        "spread_abs",
        "input_reb_prop_lines",
        "input_spread_by_side",
    ]
    for c in ("home_team_norm", "away_team_norm"):
        if c in panel.columns:
            panel_cols.append(c)

    feature_universe = (
        panel[group_keys + panel_cols]
        .merge(logs_target, on=group_keys, how="inner")
        .merge(rolling, on=group_keys, how="left")
    )

    for c in TEAM_CONTEXT_COLS:
        if c not in feature_universe.columns:
            feature_universe[c] = np.nan

    # 5) Fail-fast schema check
    require_columns(feature_universe, REQUIRED_OUTPUT_COLUMNS, "feature_universe")

    dup_count = feature_universe.duplicated(subset=group_keys, keep=False).sum()
    if dup_count > 0:
        raise ValueError(f"Duplicate keys in output before write: {dup_count} rows")

    strict_env = os.environ.get("REBOUNDS_AUDIT_LIST_STRICT", "1").strip().lower()
    audit_on = strict_env not in ("0", "false", "no") and not bool(
        getattr(args, "skip_audit_list", False)
    )
    if audit_on:
        full_scan = bool(getattr(args, "audit_list_full_scan", False)) or os.environ.get(
            "REBOUNDS_AUDIT_LIST_FULL_SCAN", ""
        ).strip().lower() in ("1", "true", "yes")
        max_audit_rows = None if full_scan else int(args.audit_list_max_rows)
        team_frame = build_audit_team_frame(logs, panel)
        mode = "full" if max_audit_rows is None else "sample"
        n_eff = len(feature_universe) if max_audit_rows is None else min(max_audit_rows, len(feature_universe))
        print(
            "audit_list_verify",
            f"mode={mode}",
            f"n_rows={len(feature_universe):,}",
            f"n_checked~={n_eff:,}",
            sep=" | ",
        )
        audit_sample = sample_audit_rows(feature_universe, max_rows=max_audit_rows)
        verify_audit_lists_dataframe(
            feature_universe,
            team_frame=team_frame,
            max_rows=max_audit_rows,
            sample_df=audit_sample,
        )
        print("audit_list_verify | ok")
        show_n = int(os.environ.get("REBOUNDS_AUDIT_LIST_SHOW_ROWS", "0").strip() or "0")
        if show_n > 0:
            print_audit_sample_to_stdout(feature_universe, team_frame, n_show=show_n, show_by="recent")

    # 6) Sort + write
    feature_universe = feature_universe.sort_values(["season", "date", "player_normalized", "game_id"]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_universe.to_parquet(output_path, index=False)

    # 7) DuckDB quality checks
    run_quality_checks(output_path)

    print(
        "phase=build_rebounds_full_universe",
        f"rows={len(feature_universe)}",
        f"v3_rows={len(v3_raw)}",
        f"season={args.season}",
        f"output={output_path}",
        f"v3={output_v3_path}",
        sep=" | ",
    )


if __name__ == "__main__":
    main()
