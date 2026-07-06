"""
Build per-book labeled training dataset for NFL WR/TE receiving yards model.

One row per player-book-game. Each row scores that book's specific
line against that book's own de-vigged P(under).

Key columns per row:
  offered_line       — this book's specific line (model feature; book-specific)
  market_under_prob  — de-vigged P(under) from this book only (model feature)
  mkt_consensus_point — median line across all books (context only)
  book               — bookmaker key

Outputs:
  ~/Downloads/tmp/nfl_rec_yards_per_book.parquet   ← training / inference input
  ~/Downloads/tmp/nfl_rec_yards_labeled.parquet     ← wide format, kept for reference

Run:
  python src/nfl_rec_yards_modeling/scripts/build_labeled_dataset.py
"""

from __future__ import annotations

import re
import warnings
from io import BytesIO
from pathlib import Path

import boto3
import numpy as np
import pandas as pd

_SUFFIX_RE  = re.compile(r"\s*,?\s*(Jr\.?|Sr\.?|II{1,2}|IV|V)\.?$", re.IGNORECASE)
_SPECIAL_RE = re.compile(r"['\.\-,]")


def _normalize_name(name: str) -> str:
    s = str(name).strip()
    s = _SUFFIX_RE.sub("", s)
    s = _SPECIAL_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


SPINE_PATH   = Path.home() / "Downloads" / "tmp" / "nfl_rec_yards_historical_spine.parquet"
OUT_PER_BOOK = Path.home() / "Downloads" / "tmp" / "nfl_rec_yards_per_book.parquet"
OUT_WIDE     = Path.home() / "Downloads" / "tmp" / "nfl_rec_yards_labeled.parquet"
MAP_PATH     = Path("data/nfl/rec_yards_name_map.csv")
S3_BUCKET    = "the-odds-api-mt"
S3_PREFIX    = "nfl/props_backfill"
TARGET_MKT   = "player_reception_yds"
SEASONS      = [2023, 2024, 2025]

BEAT_RATE_WINDOWS = [3, 5, 8, 16]

BOOKS = [
    "ballybet",
    "betmgm",
    "betonlineag",
    "betparx",
    "betrivers",
    "bovada",
    "draftkings",
    "espnbet",
    "fanatics",
    "fanduel",
    "fliff",
    "hardrockbet",
    "williamhill_us",
]

BOOK_POINT_COLS = [f"{b}_point"       for b in BOOKS]
BOOK_OVER_COLS  = [f"{b}_over_price"  for b in BOOKS]
BOOK_UNDER_COLS = [f"{b}_under_price" for b in BOOKS]

SUMMARY_COLS = ["n_books", "mkt_consensus_point", "line_min", "line_max", "line_std"]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_s3_odds(s3_client) -> pd.DataFrame:
    print("  Loading S3 odds data...")
    paginator = s3_client.get_paginator("list_objects_v2")
    frames = []
    for season in SEASONS:
        prefix = f"{S3_PREFIX}/{season}/"
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                body = s3_client.get_object(Bucket=S3_BUCKET, Key=obj["Key"])["Body"].read()
                frames.append(pd.read_parquet(BytesIO(body)))

    raw  = pd.concat(frames, ignore_index=True)
    odds = raw[raw["market"] == TARGET_MKT].copy()
    print(f"    Raw rows (all markets): {len(raw):,}  →  {TARGET_MKT}: {len(odds):,} rows")
    print(f"    Player-games with a line: {odds['nfl_game_id'].nunique():,}")
    return odds


def load_name_map() -> dict[str, str]:
    if not MAP_PATH.exists():
        return {}
    df = pd.read_csv(MAP_PATH)
    return {_normalize_name(row["odds_name_raw"]): row["name_norm"] for _, row in df.iterrows()}


# ── Wide pivot ────────────────────────────────────────────────────────────────

def pivot_to_wide(odds: pd.DataFrame, name_map: dict[str, str]) -> pd.DataFrame:
    over_df  = odds[odds["outcome_name"] == "Over"]
    under_df = odds[odds["outcome_name"] == "Under"]

    idx = ["nfl_game_id", "outcome_desc"]

    point_piv = over_df.pivot_table(index=idx, columns="bookmaker", values="point",  aggfunc="first")
    point_piv.columns = [f"{b}_point"       for b in point_piv.columns]
    over_piv  = over_df.pivot_table(index=idx, columns="bookmaker", values="price",  aggfunc="first")
    over_piv.columns  = [f"{b}_over_price"  for b in over_piv.columns]
    under_piv = under_df.pivot_table(index=idx, columns="bookmaker", values="price", aggfunc="first")
    under_piv.columns = [f"{b}_under_price" for b in under_piv.columns]

    wide = point_piv.join(over_piv, how="outer").join(under_piv, how="outer").reset_index()
    wide = wide.rename(columns={"nfl_game_id": "game_id"})

    norm = wide["outcome_desc"].map(_normalize_name)
    wide["player_name_norm"] = norm.map(lambda n: name_map.get(n, n))
    wide = wide.drop(columns=["outcome_desc"])

    book_cols = [c for c in wide.columns if c not in ["game_id", "player_name_norm"]]
    wide = wide.groupby(["game_id", "player_name_norm"], as_index=False)[book_cols].first()

    present_point_cols = [c for c in BOOK_POINT_COLS if c in wide.columns]
    wide["n_books"]         = wide[present_point_cols].notna().sum(axis=1)
    wide["offered_line"]    = wide[present_point_cols].median(axis=1)
    wide["line_min"]        = wide[present_point_cols].min(axis=1)
    wide["line_max"]        = wide[present_point_cols].max(axis=1)
    wide["line_std"]        = wide[present_point_cols].std(axis=1)

    print(f"    Wide odds rows: {len(wide):,}  |  books in data: {len(present_point_cols)}")
    return wide


# ── Beat-rate features ────────────────────────────────────────────────────────

def add_beat_rate_features(labeled: pd.DataFrame) -> pd.DataFrame:
    labeled = labeled.sort_values(["player_name_norm", "season", "week"]).reset_index(drop=True)
    labeled["_over_result"] = (labeled["receiving_yards"] > labeled["offered_line"]).astype(float)

    grp = labeled.groupby("player_name_norm")["_over_result"]
    for w in BEAT_RATE_WINDOWS:
        labeled[f"over_rate_L{w}"] = grp.transform(
            lambda s, _w=w: s.shift(1).rolling(_w, min_periods=1).mean()
        )
    labeled["over_rate_Lcareer"] = grp.transform(lambda s: s.shift(1).expanding().mean())
    labeled = labeled.drop(columns=["_over_result"])
    return labeled


# ── Per-book expansion ────────────────────────────────────────────────────────

def _amer_to_imp(price: float) -> float:
    return -price / (-price + 100) if price < 0 else 100 / (price + 100)


def expand_to_per_book(wide: pd.DataFrame) -> pd.DataFrame:
    wide = wide.rename(columns={"offered_line": "mkt_consensus_point"})

    rows = []
    for _, row in wide.iterrows():
        for book in BOOKS:
            pt = row.get(f"{book}_point")
            op = row.get(f"{book}_over_price")
            up = row.get(f"{book}_under_price")
            if pd.isna(pt) or pd.isna(op) or pd.isna(up):
                continue
            imp_o = _amer_to_imp(float(op))
            imp_u = _amer_to_imp(float(up))
            total = imp_o + imp_u
            if total <= 0:
                continue

            new_row = {c: row[c] for c in wide.columns}
            new_row["book"]                  = book
            new_row["offered_line"]          = float(pt)
            new_row["market_under_prob"]     = imp_u / total
            new_row["market_over_prob"]      = imp_o / total
            new_row["raw_under_prob"]        = imp_u
            new_row["raw_over_prob"]         = imp_o
            new_row["consensus_under_price"] = int(up)
            new_row["consensus_over_price"]  = int(op)
            rows.append(new_row)

    per_book = pd.DataFrame(rows)
    n_player_games = wide[["game_id", "player_name_norm"]].drop_duplicates().shape[0]
    print(f"    Per-book rows: {len(per_book):,}  "
          f"(avg {len(per_book)/n_player_games:.1f} book-lines per player-game)")
    return per_book


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    warnings.filterwarnings("ignore")

    spine = pd.read_parquet(SPINE_PATH)
    print(f"\nSpine: {len(spine):,} rows  |  {spine['player_name'].nunique():,} players  "
          f"|  {spine['player_name_norm'].nunique():,} normalized\n")

    name_map = load_name_map()
    print(f"  Name map: {len(name_map)} entries loaded from {MAP_PATH}")

    s3   = boto3.client("s3")
    odds = load_s3_odds(s3)
    wide = pivot_to_wide(odds, name_map)

    print("\n  Joining spine → odds on player_name_norm...")
    labeled = spine.merge(wide, on=["game_id", "player_name_norm"], how="inner")
    print(f"    Matched rows              : {len(labeled):,}")
    print(f"    Unique players with lines : {labeled['player_name_norm'].nunique():,}")

    unmatched_norm = wide[~wide.set_index(["game_id", "player_name_norm"]).index.isin(
        labeled.set_index(["game_id", "player_name_norm"]).index
    )]["player_name_norm"].nunique()
    total_norm = wide["player_name_norm"].nunique()
    print(f"    Odds names with no spine match : {unmatched_norm:,}")
    print(f"    Unique name match rate         : "
          f"{labeled['player_name_norm'].nunique() / total_norm:.1%}")

    print("\n  Computing player beat-rate features...")
    labeled = add_beat_rate_features(labeled)
    beat_rate_cols = [f"over_rate_L{w}" for w in BEAT_RATE_WINDOWS] + ["over_rate_Lcareer"]
    for col in beat_rate_cols:
        nn = labeled[col].notna().sum()
        print(f"    {col}: {nn:,} non-null ({nn/len(labeled):.1%})")

    # ── Save wide (reference) ──────────────────────────────────────────────────
    spine_cols    = [c for c in spine.columns if c in labeled.columns]
    per_book_cols = []
    for b in BOOKS:
        for col in [f"{b}_point", f"{b}_over_price", f"{b}_under_price"]:
            if col in labeled.columns:
                per_book_cols.append(col)
    wide_col_order = spine_cols + SUMMARY_COLS + beat_rate_cols + per_book_cols
    wide_out = labeled[[c for c in wide_col_order if c in labeled.columns]].copy()
    OUT_WIDE.parent.mkdir(parents=True, exist_ok=True)
    wide_out.to_parquet(OUT_WIDE, index=False)
    print(f"\n  Wide (reference) → {OUT_WIDE}  ({len(wide_out):,} rows)")

    # ── Expand to per-book ─────────────────────────────────────────────────────
    print("\n  Expanding to per-book rows...")
    per_book = expand_to_per_book(labeled)

    id_cols = [
        "book", "offered_line",
        "consensus_under_price", "consensus_over_price",
        "raw_under_prob", "raw_over_prob",
        "market_under_prob", "market_over_prob",
    ]
    context_cols       = ["mkt_consensus_point", "n_books", "line_min", "line_max", "line_std"]
    per_book_wide_cols = [c for c in per_book.columns if any(c.startswith(f"{b}_") for b in BOOKS)]
    spine_in_pb        = [c for c in spine.columns if c in per_book.columns]

    col_order = spine_in_pb + id_cols + context_cols + beat_rate_cols + per_book_wide_cols
    per_book  = per_book[[c for c in col_order if c in per_book.columns]]

    OUT_PER_BOOK.parent.mkdir(parents=True, exist_ok=True)
    per_book.to_parquet(OUT_PER_BOOK, index=False)

    print(f"\n  Position breakdown (per-book rows):")
    pos = (
        per_book.groupby("position", dropna=False)
        .agg(rows=("player_name", "count"), players=("player_name", "nunique"))
        .sort_values("rows", ascending=False)
    )
    print(pos.to_string())

    print(f"\n  Book coverage (rows per book):")
    print(per_book["book"].value_counts().to_string())

    print(f"\n  Line distribution (offered_line = book-specific line):")
    print(per_book["offered_line"].describe().to_string())

    print(f"\n  Players with 2+ distinct lines across books:")
    line_spread = per_book.groupby(["game_id", "player_name_norm"])["offered_line"].nunique()
    multi = (line_spread > 1).sum()
    print(f"    {multi:,} player-games have books disagreeing on the line")

    print(f"\n{'='*60}")
    print(f"  Per-book output : {OUT_PER_BOOK}")
    print(f"  Rows            : {len(per_book):,}")
    print(f"  Columns         : {len(per_book.columns)}")
    print(f"  Seasons         : {per_book['season'].min()}–{per_book['season'].max()}")
    print(f"  Wide (ref)      : {OUT_WIDE}  ({len(wide_out):,} rows)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
