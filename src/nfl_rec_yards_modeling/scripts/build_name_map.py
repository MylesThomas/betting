"""
Build / refresh the odds→spine player name lookup map for NFL rec yards.

Steps:
  1. Normalize both sides: lowercase + strip suffixes + remove special chars
  2. Exact match on normalized names → source=auto_norm
  3. Fuzzy match residuals within same game_id → source=fuzzy_candidate
  4. Write:
     - data/nfl/rec_yards_name_map.csv       confirmed entries
     - data/nfl/rec_yards_name_candidates.csv fuzzy candidates for review

Usage:
  python src/nfl_rec_yards_modeling/scripts/build_name_map.py
  python src/nfl_rec_yards_modeling/scripts/build_name_map.py --new-only
  python src/nfl_rec_yards_modeling/scripts/build_name_map.py --threshold 80
"""

from __future__ import annotations

import argparse
import re
import warnings
from io import BytesIO
from pathlib import Path

import boto3
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz import process as rf_process

SPINE_PATH      = Path.home() / "Downloads" / "tmp" / "nfl_rec_yards_historical_spine.parquet"
MAP_PATH        = Path("data/nfl/rec_yards_name_map.csv")
CANDIDATES_PATH = Path("data/nfl/rec_yards_name_candidates.csv")

S3_BUCKET  = "the-odds-api-mt"
S3_PREFIX  = "nfl/props_backfill"
TARGET_MKT = "player_reception_yds"
SEASONS    = [2023, 2024, 2025]

FUZZY_THRESHOLD = 85

_SUFFIX_RE  = re.compile(r"\s*,?\s*(Jr\.?|Sr\.?|II{1,2}|IV|V)\.?$", re.IGNORECASE)
_SPECIAL_RE = re.compile(r"['\.\-,]")


def normalize(name: str) -> str:
    s = str(name).strip()
    s = _SUFFIX_RE.sub("", s)
    s = _SPECIAL_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def load_odds_names(seasons: list[int]) -> pd.DataFrame:
    print("  Loading odds names from S3...")
    s3        = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    frames    = []
    for season in seasons:
        prefix = f"{S3_PREFIX}/{season}/"
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                body = s3.get_object(Bucket=S3_BUCKET, Key=obj["Key"])["Body"].read()
                df   = pd.read_parquet(BytesIO(body))
                df   = df[(df["market"] == TARGET_MKT) & (df["outcome_name"] == "Over")][
                    ["nfl_game_id", "outcome_desc"]
                ].drop_duplicates()
                frames.append(df)
    non_empty = [f for f in frames if len(f) > 0]
    if not non_empty:
        raise RuntimeError(f"No {TARGET_MKT} data found in S3 for seasons {seasons}")
    odds = pd.concat(non_empty, ignore_index=True).rename(
        columns={"nfl_game_id": "game_id", "outcome_desc": "odds_name"}
    )
    odds = odds.drop_duplicates(subset=["game_id", "odds_name"])
    print(f"    {odds['odds_name'].nunique():,} unique odds names  |  {len(odds):,} player-games")
    return odds


def run_matching(
    odds: pd.DataFrame,
    spine: pd.DataFrame,
    existing_map: pd.DataFrame,
    new_only: bool = False,
    threshold: int = FUZZY_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    spine_norms    = set(spine["player_name_norm"].dropna())
    norm_to_raw    = spine.drop_duplicates("player_name_norm").set_index(
        "player_name_norm"
    )["player_name"].to_dict()

    existing_norms: set[str] = set()
    if not existing_map.empty and "odds_name_raw" in existing_map.columns:
        existing_norms = set(existing_map["odds_name_raw"].map(normalize))

    freq = odds.groupby("odds_name").size().reset_index(name="n_games")
    odds_names = (
        freq.sort_values("n_games", ascending=False)["odds_name"].tolist()
    )

    auto_rows  = []
    fuzzy_rows = []

    for odds_name in odds_names:
        odds_norm = normalize(odds_name)
        n_games   = int(freq[freq["odds_name"] == odds_name]["n_games"].iloc[0])

        if new_only and odds_norm in existing_norms:
            continue

        if odds_norm in spine_norms:
            auto_rows.append({
                "odds_name_raw": odds_name, "spine_name_raw": norm_to_raw.get(odds_norm),
                "name_norm": odds_norm, "source": "auto_norm",
                "n_games": n_games, "fuzzy_score": 100, "notes": "exact norm match",
            })
            continue

        # Fuzzy match against full spine name list
        cand_list = list(spine_norms)
        result    = rf_process.extractOne(odds_norm, cand_list, scorer=fuzz.token_sort_ratio)
        if result is None:
            fuzzy_rows.append({
                "odds_name_raw": odds_name, "spine_name_raw": None,
                "name_norm": odds_norm, "source": "no_candidates",
                "n_games": n_games, "fuzzy_score": 0, "notes": "rapidfuzz returned None",
            })
            continue

        best_norm, score, _ = result
        best_raw = norm_to_raw.get(best_norm, best_norm)
        row = {
            "odds_name_raw": odds_name, "spine_name_raw": best_raw,
            "name_norm": best_norm, "n_games": n_games, "fuzzy_score": score,
            "notes": f"norm: '{odds_norm}' → '{best_norm}'",
        }
        row["source"] = "fuzzy_candidate" if score >= threshold else "no_match"
        if score < threshold:
            row["notes"] = f"best was '{best_raw}' ({score:.1f}) — below threshold"
        fuzzy_rows.append(row)

    confirmed     = pd.DataFrame(auto_rows)
    candidates_df = pd.DataFrame(fuzzy_rows) if fuzzy_rows else pd.DataFrame(
        columns=["odds_name_raw", "spine_name_raw", "name_norm", "source",
                 "n_games", "fuzzy_score", "notes"]
    )
    return confirmed, candidates_df


def main():
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-only",  action="store_true")
    parser.add_argument("--threshold", type=int, default=FUZZY_THRESHOLD)
    args = parser.parse_args()

    print(f"\nNFL Rec Yards — name map builder  (threshold={args.threshold})\n")

    spine = pd.read_parquet(SPINE_PATH, columns=["game_id", "player_name", "player_name_norm"])
    print(f"Spine: {spine['player_name'].nunique():,} unique players  |  {len(spine):,} rows")

    odds = load_odds_names(SEASONS)

    existing_map = pd.DataFrame(
        columns=["odds_name_raw", "spine_name_raw", "name_norm", "source", "n_games", "fuzzy_score"]
    )
    if MAP_PATH.exists():
        existing_map = pd.read_csv(MAP_PATH)
        print(f"Existing map: {len(existing_map)} entries  ({MAP_PATH})")
    else:
        print("No existing map — building from scratch")

    print("\nRunning match pipeline...")
    confirmed, candidates = run_matching(
        odds, spine, existing_map, new_only=args.new_only, threshold=args.threshold
    )

    print(f"\n{'='*65}")
    auto_games = confirmed["n_games"].sum() if not confirmed.empty else 0
    print(f"  Auto-norm resolved  : {len(confirmed)} unique names  ({auto_games} player-games)")
    if not candidates.empty:
        fuzzy_ok = candidates[candidates["source"] == "fuzzy_candidate"]
        no_match = candidates[candidates["source"].isin(["no_match", "no_candidates"])]
        print(f"  Fuzzy candidates    : {len(fuzzy_ok)} names  "
              f"({fuzzy_ok['n_games'].sum() if not fuzzy_ok.empty else 0} player-games)")
        print(f"  No match            : {len(no_match)} names  "
              f"({no_match['n_games'].sum() if not no_match.empty else 0} player-games)")
    print(f"{'='*65}\n")

    if not confirmed.empty:
        if not existing_map.empty:
            existing_norms = set(existing_map["odds_name_raw"].map(normalize))
            new_confirmed  = confirmed[~confirmed["odds_name_raw"].map(normalize).isin(existing_norms)]
        else:
            new_confirmed = confirmed
        updated_map = pd.concat([existing_map, new_confirmed], ignore_index=True)
    else:
        updated_map = existing_map

    MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    updated_map.to_csv(MAP_PATH, index=False)
    print(f"Map written  → {MAP_PATH}  ({len(updated_map)} entries total)")

    if not candidates.empty:
        cand_out = candidates.sort_values("n_games", ascending=False)
        cand_out.to_csv(CANDIDATES_PATH, index=False)
        print(f"Candidates   → {CANDIDATES_PATH}  ({len(cand_out)} rows)")

        fuzzy_ok = candidates[candidates["source"] == "fuzzy_candidate"].sort_values(
            "n_games", ascending=False
        )
        if not fuzzy_ok.empty:
            print(f"\n{'='*65}")
            print(f"  FUZZY CANDIDATES (score ≥ {args.threshold}) — review:")
            print(f"{'='*65}")
            print(fuzzy_ok[
                ["odds_name_raw", "spine_name_raw", "name_norm", "fuzzy_score", "n_games", "notes"]
            ].to_string(index=False))

        no_match = candidates[candidates["source"].isin(["no_match", "no_candidates"])].sort_values(
            "n_games", ascending=False
        )
        if not no_match.empty:
            print(f"\n{'='*65}")
            print(f"  NO MATCH (below threshold):")
            print(f"{'='*65}")
            print(no_match[
                ["odds_name_raw", "spine_name_raw", "name_norm", "fuzzy_score", "n_games", "notes"]
            ].to_string(index=False))

    print(f"\n{'='*65}\n  DONE\n{'='*65}\n")


if __name__ == "__main__":
    main()
