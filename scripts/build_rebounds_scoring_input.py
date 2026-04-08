"""
Build rebounds props scoring-input parquet from live rebounds props CSV.

Context:
- Converts live props rows into scorer-ready input for rebounds slate scoring.
- Resolves `game_id` from NBA schedule (team matchup + date).
- Reuses canonical rebounds market transforms from the historical universe
  builder so scoring-input semantics remain single-sourced.

Usage:
    python scripts/build_rebounds_scoring_input.py \
        --live-csv data/live_props/2026-03-24/live_rebounds_props_raw.csv \
        --output data/live_props/2026-03-24/rebounds_props_scoring_input.parquet \
        --date 2026-03-24
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def ensure_repo_root_on_syspath() -> Path:
    current = Path(__file__).resolve().parent
    while True:
        if (current / ".gitignore").exists() and (current / "src").exists():
            if str(current) not in sys.path:
                sys.path.insert(0, str(current))
            return current
        if current.parent == current:
            raise FileNotFoundError("Could not locate repo root")
        current = current.parent


REPO_ROOT = ensure_repo_root_on_syspath()

from src.nba_schedule_utils import get_schedule_for_date, resolve_game_id  # noqa: E402
from src.player_team_history.name_normalization import normalize_from_odds_api  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build rebounds scoring-input parquet.")
    parser.add_argument(
        "--live-csv",
        type=str,
        required=True,
        help="Path or s3:// URI to live rebounds CSV.",
    )
    parser.add_argument("--output", type=str, required=True, help="Output parquet path.")
    parser.add_argument(
        "--date",
        type=str,
        default="",
        help="Target date YYYY-MM-DD (inferred from game_time if omitted).",
    )
    return parser.parse_args()


def load_rebounds_market_transform_functions(repo_root: Path):
    script_path = repo_root / "src/nba_rebounds_modeling/00_research/scripts/v2_build_rebounds_universe.py"
    spec = importlib.util.spec_from_file_location("rebounds_v2_build", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load v2 script module: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_market_panel, module.build_v3_props_raw


def read_csv_any_path(path: str) -> pd.DataFrame:
    if path.startswith("s3://"):
        import boto3
        from io import BytesIO

        bucket, key = path.replace("s3://", "").split("/", 1)
        obj = boto3.client("s3").get_object(Bucket=bucket, Key=key)
        return pd.read_csv(BytesIO(obj["Body"].read()))
    return pd.read_csv(path)


def infer_target_date(df: pd.DataFrame) -> str:
    dt = pd.to_datetime(df["game_time"], errors="coerce", utc=True)
    et_dates = dt.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    return et_dates.mode().iloc[0]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    build_market_panel, build_props_raw = load_rebounds_market_transform_functions(REPO_ROOT)

    logging.info("Reading live props CSV: %s", args.live_csv)
    df = read_csv_any_path(args.live_csv)
    if df.empty:
        raise ValueError("Live props CSV is empty.")

    target_date = args.date if args.date else infer_target_date(df)
    logging.info("Target date: %s", target_date)

    props = df.copy()
    props["player_normalized"] = props["player"].apply(normalize_from_odds_api)
    props = props.loc[props["player_normalized"].notna()].copy()
    props["date"] = target_date
    props["line"] = pd.to_numeric(props["prop_line"], errors="coerce")
    props["odds_over"] = pd.to_numeric(props["over_odds"], errors="coerce")
    props["odds_under"] = pd.to_numeric(props["under_odds"], errors="coerce")

    if "season" not in props.columns:
        raise ValueError("live csv missing required column: season")

    if "home_team" not in props.columns or "away_team" not in props.columns:
        raise ValueError("live csv missing required team columns: home_team and away_team")

    # Keep explicit ID semantics:
    # - odds_event_id: Odds API event identifier (fallback source)
    # - nba_game_id: NBA schedule-resolved GAME_ID
    # - game_id_source: nba_schedule | odds_event_fallback
    # - game_id: canonical ID used by downstream joins
    if "odds_api_event_id" in props.columns:
        props["odds_event_id"] = props["odds_api_event_id"].astype(str)
        props.loc[props["odds_event_id"].isin(["", "nan", "None"]), "odds_event_id"] = np.nan
    else:
        props["odds_event_id"] = np.nan
    props["nba_game_id"] = pd.Series([None] * len(props), dtype="object")
    props["game_id_source"] = "odds_event_fallback"

    schedule_df = get_schedule_for_date(target_date)
    if not schedule_df.empty:
        game_id_cache: dict[str, str | None] = {}

        def map_game_id(row: pd.Series):
            key = f"{row['home_team']}__{row['away_team']}"
            if key not in game_id_cache:
                game_id_cache[key] = resolve_game_id(
                    row["home_team"], row["away_team"], target_date, schedule_df
                )
            return game_id_cache[key]

        resolved_ids = props.apply(map_game_id, axis=1)
        props.loc[resolved_ids.notna(), "nba_game_id"] = resolved_ids[resolved_ids.notna()]
        props.loc[resolved_ids.notna(), "game_id_source"] = "nba_schedule"
    else:
        logging.warning("Schedule unavailable for %s; using odds_api_event_id fallback game_id.", target_date)

    props["game_id"] = props["nba_game_id"].astype("object")
    fallback_mask = props["game_id"].isna() & props["odds_event_id"].notna()
    props.loc[fallback_mask, "game_id"] = props.loc[fallback_mask, "odds_event_id"]

    props = props.loc[props["game_id"].notna()].copy()
    if props.empty:
        raise ValueError("No props rows left after game_id resolution.")

    logs_stub = props[["season", "date", "player_normalized", "game_id"]].drop_duplicates().copy()
    logs_stub["REB"] = np.nan

    panel, book_line = build_market_panel(props, logs_stub)
    v3_raw = build_props_raw(book_line, logs_stub, panel)

    id_cols = ["odds_event_id", "nba_game_id", "game_id_source", "game_id"]
    if "game_id_source" not in v3_raw.columns or "odds_event_id" not in v3_raw.columns:
        join_keys = [
            "season",
            "date",
            "player_normalized",
            "bookmaker",
            "line",
            "over_odds",
            "under_odds",
            "game_id",
        ]
        available_join_keys = [c for c in join_keys if c in v3_raw.columns and c in props.columns]
        id_value_cols = [c for c in id_cols if c not in available_join_keys]
        id_map = props[available_join_keys + id_value_cols].drop_duplicates().copy()
        v3_raw = v3_raw.merge(id_map, on=available_join_keys, how="left")

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    v3_raw.to_parquet(out_path, index=False)
    logging.info("Wrote scoring-input parquet: %s | rows=%s", out_path, f"{len(v3_raw):,}")


if __name__ == "__main__":
    main()
