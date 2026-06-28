"""
Build / refresh the odds→spine player name lookup map for NFL tackles.

Steps:
  1. Normalize both sides: lowercase + strip suffixes + remove special chars
  2. Exact match on normalized names → source=auto_norm
  3. Fuzzy match residuals within same game_id → source=fuzzy_candidate
  4. Write two outputs:
     - data/nfl/tackles_name_map.csv       confirmed entries (auto + any prior manual)
     - data/nfl/tackles_name_candidates.csv fuzzy candidates for review

Usage:
  python src/nfl_tackles_modeling/scripts/build_name_map.py
  python src/nfl_tackles_modeling/scripts/build_name_map.py --new-only   # skip names already in map
  python src/nfl_tackles_modeling/scripts/build_name_map.py --threshold 80
"""

from __future__ import annotations

import argparse
import re
import warnings
from io import BytesIO
from pathlib import Path

import boto3
import pandas as pd
from rapidfuzz import process as rf_process
from rapidfuzz import fuzz

SPINE_PATH     = Path.home() / "Downloads" / "tmp" / "nfl_tackles_historical_spine.parquet"
MAP_PATH       = Path("data/nfl/tackles_name_map.csv")
CANDIDATES_PATH = Path("data/nfl/tackles_name_candidates.csv")

S3_BUCKET  = "the-odds-api-mt"
S3_PREFIX  = "nfl/props_backfill"
TARGET_MKT = "player_tackles_assists"
SEASONS    = [2024, 2025]

FUZZY_THRESHOLD = 85   # override with --threshold


# ── Normalization ──────────────────────────────────────────────────────────────

_SUFFIX_RE  = re.compile(r"\s*,?\s*(Jr\.?|Sr\.?|II{1,2}|IV|V)\.?$", re.IGNORECASE)
_SPECIAL_RE = re.compile(r"['\.\-,]")

def normalize(name: str) -> str:
    s = str(name).strip()
    s = _SUFFIX_RE.sub("", s)
    s = _SPECIAL_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


# ── S3 loader ──────────────────────────────────────────────────────────────────

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
    odds = pd.concat(non_empty, ignore_index=True).rename(
        columns={"nfl_game_id": "game_id", "outcome_desc": "odds_name"}
    )
    odds = odds.drop_duplicates(subset=["game_id", "odds_name"])
    print(f"    {odds['odds_name'].nunique():,} unique odds names  |  {len(odds):,} player-games")
    return odds


# ── Match pipeline ─────────────────────────────────────────────────────────────

def run_matching(
    odds: pd.DataFrame,
    spine: pd.DataFrame,
    existing_map: pd.DataFrame,
    new_only: bool,
    threshold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (confirmed_map_rows, fuzzy_candidates).
    Map schema: odds_name_raw, spine_name_raw, name_norm, source, n_games, fuzzy_score
    """
    # Spine provides player_name_norm (pre-built) and player_name (raw display)
    # Build: name_norm → spine_name_raw
    spine_norm_to_raw: dict[str, str] = {}
    for _, row in spine[["player_name", "player_name_norm"]].drop_duplicates().iterrows():
        spine_norm_to_raw.setdefault(row["player_name_norm"], row["player_name"])

    # Per-game spine norms for fuzzy (game_id → list of player_name_norm)
    game_spine_norms: dict[str, list[str]] = (
        spine.dropna(subset=["player_name_norm"])
        .groupby("game_id")["player_name_norm"]
        .apply(list)
        .to_dict()
    )
    # norm → raw lookup for recovering spine_name_raw from fuzzy winner
    norm_to_raw: dict[str, str] = spine_norm_to_raw

    # If --new-only, skip odds names already in the map
    already_mapped_norms: set[str] = set()
    if new_only and not existing_map.empty:
        already_mapped_norms = set(existing_map["odds_name_raw"].map(normalize))
        pre = odds["odds_name"].nunique()
        odds = odds[~odds["odds_name"].map(normalize).isin(already_mapped_norms)]
        print(f"    --new-only: skipping {pre - odds['odds_name'].nunique()} already-mapped names")

    odds_name_games: dict[str, list[str]] = (
        odds.groupby("odds_name")["game_id"].apply(list).to_dict()
    )

    auto_rows: list[dict] = []
    fuzzy_rows: list[dict] = []

    for odds_name, game_ids in odds_name_games.items():
        odds_norm = normalize(odds_name)
        n_games   = len(game_ids)

        # ── Pass 1: exact normalized match ────────────────────────────────────
        if odds_norm in spine_norm_to_raw:
            spine_raw = spine_norm_to_raw[odds_norm]
            if odds_name != spine_raw:  # skip trivial exact matches — no map entry needed
                auto_rows.append({
                    "odds_name_raw": odds_name,
                    "spine_name_raw": spine_raw,
                    "name_norm":     odds_norm,
                    "source":        "auto_norm",
                    "n_games":       n_games,
                    "fuzzy_score":   100,
                })
            continue

        # ── Pass 2: fuzzy within same game_id ─────────────────────────────────
        candidates: set[str] = set()
        for gid in game_ids:
            candidates.update(game_spine_norms.get(gid, []))

        if not candidates:
            fuzzy_rows.append({
                "odds_name_raw": odds_name, "spine_name_raw": None,
                "name_norm": odds_norm, "source": "no_candidates",
                "n_games": n_games, "fuzzy_score": 0,
                "notes": "no spine players in same game",
            })
            continue

        cand_list = list(candidates)
        result = rf_process.extractOne(odds_norm, cand_list, scorer=fuzz.token_sort_ratio)

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

    confirmed = pd.DataFrame(auto_rows)
    candidates_df = pd.DataFrame(fuzzy_rows) if fuzzy_rows else pd.DataFrame(
        columns=["odds_name_raw","spine_name_raw","name_norm","source","n_games","fuzzy_score","notes"]
    )
    return confirmed, candidates_df


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("--new-only",  action="store_true",
                        help="Skip odds names already in the map")
    parser.add_argument("--threshold", type=int, default=FUZZY_THRESHOLD,
                        help=f"Fuzzy match threshold (default {FUZZY_THRESHOLD})")
    args = parser.parse_args()

    print(f"\nNFL Tackles — name map builder  (threshold={args.threshold})\n")

    # Spine now carries player_name_norm (pre-built in build_historical_spine.py)
    spine = pd.read_parquet(SPINE_PATH, columns=["game_id", "player_name", "player_name_norm"])
    print(f"Spine: {spine['player_name'].nunique():,} unique players  |  {len(spine):,} rows")

    odds = load_odds_names(SEASONS)

    existing_map = pd.DataFrame(
        columns=["odds_name_raw","spine_name_raw","name_norm","source","n_games","fuzzy_score"]
    )
    if MAP_PATH.exists():
        existing_map = pd.read_csv(MAP_PATH)
        print(f"Existing map: {len(existing_map)} entries  ({MAP_PATH})")
    else:
        print("No existing map — building from scratch")

    print(f"\nRunning match pipeline...")
    confirmed, candidates = run_matching(
        odds, spine, existing_map, new_only=args.new_only, threshold=args.threshold
    )

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  Auto-norm resolved  : {len(confirmed)} unique names  "
          f"({confirmed['n_games'].sum() if not confirmed.empty else 0} player-games)")

    if not candidates.empty:
        fuzzy_ok = candidates[candidates["source"] == "fuzzy_candidate"]
        no_match = candidates[candidates["source"].isin(["no_match","no_candidates"])]
        print(f"  Fuzzy candidates    : {len(fuzzy_ok)} names  "
              f"({fuzzy_ok['n_games'].sum() if not fuzzy_ok.empty else 0} player-games)")
        print(f"  No match / no cand  : {len(no_match)} names  "
              f"({no_match['n_games'].sum() if not no_match.empty else 0} player-games)")
    print(f"{'='*65}\n")

    # ── Merge confirmed into existing map ─────────────────────────────────────
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

    # ── Write candidates for review ────────────────────────────────────────────
    if not candidates.empty:
        cand_out = candidates.sort_values("n_games", ascending=False)
        cand_out.to_csv(CANDIDATES_PATH, index=False)
        print(f"Candidates   → {CANDIDATES_PATH}  ({len(cand_out)} rows)")
        print(f"\n  Review candidates, accept by changing source to 'fuzzy_accepted',")
        print(f"  then add to {MAP_PATH} (odds_name_raw, spine_name_raw, name_norm, source, n_games).")

        print(f"\n{'='*65}")
        print(f"  FUZZY CANDIDATES (score ≥ {args.threshold}) — review these:")
        print(f"{'='*65}")
        fuzzy_ok = candidates[candidates["source"] == "fuzzy_candidate"].sort_values("n_games", ascending=False)
        if not fuzzy_ok.empty:
            print(fuzzy_ok[["odds_name_raw","spine_name_raw","name_norm","fuzzy_score","n_games","notes"]].to_string(index=False))

        print(f"\n{'='*65}")
        print(f"  NO MATCH (below threshold or no candidates):")
        print(f"{'='*65}")
        no_match = candidates[candidates["source"].isin(["no_match","no_candidates"])].sort_values("n_games", ascending=False)
        if not no_match.empty:
            print(no_match[["odds_name_raw","spine_name_raw","name_norm","fuzzy_score","n_games","notes"]].to_string(index=False))

    print(f"\n{'='*65}\n  DONE\n{'='*65}\n")


if __name__ == "__main__":
    main()
