"""
Slice rebounds_model_features_v2.parquet to a single calendar date for prod scoring.

Context:
- build_rebounds_full_universe.py builds the full historical feature table; prod scoring only
  needs player-game rows for the slate date.
- Adds rebounds_feature_schema_version for audit (bump string when v2 output schema changes).

Usage:
    python src/nba_rebounds_modeling/00_research/scripts/prod_slice_rebounds_features.py \\
        --feat ~/Downloads/tmp/rebounds_model_features_v2.parquet \\
        --as-of-date 2025-03-15 \\
        --output ~/Downloads/tmp/rebounds_features_slice_2025-03-15.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def ensure_repo_root_on_syspath() -> Path:
    current = Path.cwd().resolve()
    while True:
        if (current / ".gitignore").exists() and (current / "src").exists():
            if str(current) not in sys.path:
                sys.path.insert(0, str(current))
            return current
        if current.parent == current:
            raise FileNotFoundError("Could not locate repo root")
        current = current.parent


ensure_repo_root_on_syspath()

from src.nba_rebounds_modeling.rebounds_feature_spec import GROUP_KEYS  # noqa: E402

SCHEMA_VERSION = "rebounds_features_v2_b_min_max_team_ctx_1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Slice rebounds features parquet to one date.")
    p.add_argument("--feat", type=str, required=True, help="Full rebounds_model_features_v2.parquet path.")
    p.add_argument("--as-of-date", type=str, required=True, help="Slate date YYYY-MM-DD (UTC-naive match on row date).")
    p.add_argument("--output", type=str, required=True, help="Output parquet path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    feat_path = Path(args.feat).expanduser()
    out_path = Path(args.output).expanduser()
    as_of = pd.Timestamp(args.as_of_date).normalize()

    feat = pd.read_parquet(feat_path)
    for k in GROUP_KEYS:
        if k not in feat.columns:
            raise ValueError(f"feat parquet missing group key column: {k}")

    d = pd.to_datetime(feat["date"]).dt.normalize()
    sl = feat.loc[d == as_of].copy()
    sl["rebounds_feature_schema_version"] = SCHEMA_VERSION

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sl.to_parquet(out_path, index=False)
    print(
        "prod_slice_rebounds_features",
        f"as_of={args.as_of_date}",
        f"rows={len(sl):,}",
        f"output={out_path}",
        sep=" | ",
    )


if __name__ == "__main__":
    main()
