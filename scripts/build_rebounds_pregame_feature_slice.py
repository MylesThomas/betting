"""
Build pregame rebounds feature slice for a slate date.

Context:
- `prod_slice_rebounds_features.py` slices rows where `date == slate_date`.
- For pregame runs, same-day rows may be absent because full feature tables are
  tied to completed game logs.
- This script constructs a scorer-compatible feature slice by combining:
  1) latest available historical player form features before `slate_date`,
  2) same-day market context derived from props scoring-input rows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


ensure_repo_root_on_syspath()

from src.nba_rebounds_modeling.rebounds_feature_spec import B_MIN_MAX_FEATS, GROUP_KEYS  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build pregame rebounds feature slice.")
    p.add_argument("--feat", type=str, required=True, help="Full rebounds feature parquet.")
    p.add_argument("--props", type=str, required=True, help="Scoring-input props parquet.")
    p.add_argument("--slate-date", type=str, required=True, help="Slate date YYYY-MM-DD.")
    p.add_argument("--output", type=str, required=True, help="Output feature slice parquet.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    slate = pd.Timestamp(args.slate_date).normalize()

    feat = pd.read_parquet(Path(args.feat).expanduser())
    props = pd.read_parquet(Path(args.props).expanduser())

    props_d = pd.to_datetime(props["date"]).dt.normalize()
    props_s = props.loc[props_d == slate].copy()
    if len(props_s) == 0:
        raise ValueError(f"No props rows found for slate-date {slate.date()}")

    hist = feat.copy()
    hist["__d"] = pd.to_datetime(hist["date"]).dt.normalize()
    hist = hist.loc[hist["__d"] < slate].copy()
    if len(hist) == 0:
        raise ValueError("No historical feature rows found before slate-date")

    hist = hist.sort_values(["season", "player_normalized", "__d"])
    latest = hist.groupby(["season", "player_normalized"], as_index=False).tail(1).copy()

    market = (
        props_s.groupby(["season", "date", "player_normalized", "game_id"], as_index=False)
        .agg(min_line=("line", "min"), max_line=("line", "max"))
    )

    latest_cols = ["season", "player_normalized", "spread_signed", "roll_reb_mean_60", "roll_fg3a_mean_20", "roll_reb_std_5"]
    latest = latest[latest_cols].copy()

    out = market.merge(latest, on=["season", "player_normalized"], how="left")
    missing = out[B_MIN_MAX_FEATS].isna().any(axis=1)
    if missing.any():
        missing_players = out.loc[missing, ["season", "player_normalized"]].drop_duplicates()
        print(
            "build_rebounds_pregame_feature_slice",
            f"missing_feature_players={len(missing_players):,}",
            "note=rows retained with NaNs; scorer will still run",
            sep=" | ",
        )

    out = out[GROUP_KEYS + B_MIN_MAX_FEATS].copy()
    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(
        "build_rebounds_pregame_feature_slice",
        f"slate={args.slate_date}",
        f"rows={len(out):,}",
        f"output={out_path}",
        sep=" | ",
    )


if __name__ == "__main__":
    main()
