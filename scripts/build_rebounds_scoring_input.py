"""
Build rebounds props scoring-input parquet from live rebounds props CSV.

Context:
- This replaces ambiguous "live bridge" naming with a clearer purpose:
  convert live props rows into scorer-ready input for
  `prod_score_rebounds_slate.py --props`.
- It resolves `game_id` using schedule data and reuses canonical v2 rebounds
  transforms (`build_market_panel`, `build_v3_props_raw`) so logic remains
  single-sourced.

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


def load_v2_functions(repo_root: Path):
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

    build_market_panel, build_v3_props_raw = load_v2_functions(REPO_ROOT)

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

    # Prefer odds API event ids (fast, stable, and avoids hard dependency on nba_api).
    if "odds_api_event_id" in props.columns:
        props["game_id"] = props["odds_api_event_id"].astype(str)
        props.loc[props["game_id"].isin(["", "nan", "None"]), "game_id"] = np.nan
    else:
        props["game_id"] = np.nan

    unresolved_mask = props["game_id"].isna()
    if unresolved_mask.any():
        schedule_df = get_schedule_for_date(target_date)
        game_id_cache: dict[str, str | None] = {}

        def map_game_id(row: pd.Series):
            key = f"{row['home_team']}__{row['away_team']}"
            if key not in game_id_cache:
                game_id_cache[key] = resolve_game_id(
                    row["home_team"], row["away_team"], target_date, schedule_df
                )
            return game_id_cache[key]

        props.loc[unresolved_mask, "game_id"] = props.loc[unresolved_mask].apply(map_game_id, axis=1)
    props = props.loc[props["game_id"].notna()].copy()
    if props.empty:
        raise ValueError("No props rows left after game_id resolution.")

    logs_stub = props[["season", "date", "player_normalized", "game_id"]].drop_duplicates().copy()
    logs_stub["REB"] = np.nan

    panel, book_line = build_market_panel(props, logs_stub)
    v3_raw = build_v3_props_raw(book_line, logs_stub, panel)

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    v3_raw.to_parquet(out_path, index=False)
    logging.info("Wrote scoring-input parquet: %s | rows=%s", out_path, f"{len(v3_raw):,}")


if __name__ == "__main__":
    main()
