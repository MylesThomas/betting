"""
Validate PBP-derived tackle counts against PFR official weekly defensive stats.

PFR `def_tackles_combined` = solo + assist (each = 1), same definition as
player_tackles_assists prop market. If our PBP counts diverge from PFR,
our target variable is wrong.

Run:
  python src/nfl_tackles_modeling/scripts/validate_tackle_counts.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import nfl_data_py as nfl
import numpy as np
import pandas as pd

SPINE_PATH = Path.home() / "Downloads" / "tmp" / "nfl_tackles_historical_spine.parquet"
SEASONS    = [2024, 2025]


def section(title: str) -> None:
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def main():
    warnings.filterwarnings("ignore")

    # ── Load spine ─────────────────────────────────────────────────────────────
    spine = pd.read_parquet(SPINE_PATH)
    print(f"\nSpine: {len(spine):,} rows  |  {spine['player_name'].nunique():,} players")

    # ── Load PFR official weekly defensive stats ───────────────────────────────
    print("  Loading PFR weekly defensive stats...")
    pfr_frames = []
    for season in SEASONS:
        pfr_frames.append(nfl.import_weekly_pfr(s_type='def', years=[season]))
    pfr = pd.concat(pfr_frames, ignore_index=True)
    pfr = pfr[pfr["game_type"] == "REG"][
        ["game_id", "season", "week", "team", "pfr_player_name", "pfr_player_id",
         "def_tackles_combined"]
    ].dropna(subset=["def_tackles_combined"]).copy()
    pfr["def_tackles_combined"] = pfr["def_tackles_combined"].astype(int)
    print(f"  PFR: {len(pfr):,} player-game rows  |  {pfr['pfr_player_name'].nunique():,} players")

    # ── Bridge pfr_player_id → player_id (gsis) via players table ─────────────
    players_df = nfl.import_players()[["gsis_id", "pfr_id", "display_name"]].dropna(subset=["pfr_id"])
    pfr_to_gsis = players_df.set_index("pfr_id")["gsis_id"].to_dict()
    pfr["player_id"] = pfr["pfr_player_id"].map(pfr_to_gsis)

    # ── Join PFR → spine on game_id + player_id ────────────────────────────────
    section("JOIN QUALITY")
    merged = spine.merge(
        pfr[["game_id", "player_id", "def_tackles_combined"]],
        on=["game_id", "player_id"],
        how="inner",
    )
    print(f"  Spine rows           : {len(spine):,}")
    print(f"  PFR rows             : {len(pfr):,}")
    print(f"  Matched (game+pid)   : {len(merged):,}  "
          f"({len(merged)/len(spine):.1%} of spine rows)")

    # Fallback: also try name match for rows that didn't join via player_id
    spine_unmatched = spine[~spine["player_id"].isin(merged["player_id"].unique())]
    pfr_norm = pfr.copy()
    pfr_norm["pfr_player_name"] = pfr_norm["pfr_player_name"].str.strip()
    name_matched = spine_unmatched.merge(
        pfr_norm[["game_id", "pfr_player_name", "def_tackles_combined"]]
              .rename(columns={"pfr_player_name": "player_name"}),
        on=["game_id", "player_name"],
        how="inner",
    )
    print(f"  Additional via name  : {len(name_matched):,}")
    merged = pd.concat([merged, name_matched], ignore_index=True)
    print(f"  Total matched        : {len(merged):,}  ({len(merged)/len(spine):.1%} of spine)")

    # ── Compare counts ─────────────────────────────────────────────────────────
    section("TACKLE COUNT COMPARISON — PBP vs PFR")
    merged["diff"] = merged["tackles_assists"] - merged["def_tackles_combined"]

    exact   = (merged["diff"] == 0).mean()
    within1 = (merged["diff"].abs() <= 1).mean()
    mae     = merged["diff"].abs().mean()
    corr    = merged["tackles_assists"].corr(merged["def_tackles_combined"])

    print(f"\n  n compared           : {len(merged):,}")
    print(f"  Exact match          : {exact:.1%}")
    print(f"  Within ±1            : {within1:.1%}")
    print(f"  MAE (PBP vs PFR)     : {mae:.3f}")
    print(f"  Pearson r            : {corr:.4f}")
    print(f"\n  Diff distribution (PBP − PFR):")
    diff_vc = merged["diff"].clip(-5, 5).value_counts().sort_index()
    for val, cnt in diff_vc.items():
        label = f"{val:+d}" if val != -5 and val != 5 else (f"≤{val}" if val == -5 else f"≥{val}")
        bar   = "█" * (cnt // 20)
        print(f"    {label:>4}  {cnt:>5}  {bar}")

    # ── Large discrepancy examples ─────────────────────────────────────────────
    section("LARGE DISCREPANCIES (|diff| ≥ 3)")
    big = merged[merged["diff"].abs() >= 3].sort_values("diff", key=abs, ascending=False)
    if big.empty:
        print("  ✓ No discrepancies ≥ 3 tackles")
    else:
        print(f"  {len(big)} rows with |diff| ≥ 3:\n")
        show = big[["game_id", "season", "week", "player_name", "position",
                    "tackles_assists", "def_tackles_combined", "diff"]].head(30)
        print(show.to_string(index=False))

    # ── Systematic bias check ──────────────────────────────────────────────────
    section("SYSTEMATIC BIAS BY POSITION")
    bias = (
        merged.groupby("position")
        .agg(
            n         = ("diff", "count"),
            mean_diff = ("diff", "mean"),
            mae       = ("diff", lambda x: x.abs().mean()),
        )
        .sort_values("n", ascending=False)
    )
    bias["mean_diff"] = bias["mean_diff"].round(3)
    bias["mae"]       = bias["mae"].round(3)
    print()
    print(bias.to_string())
    print("\n  mean_diff = PBP − PFR  (positive → PBP overcounts vs PFR)")

    # ── Solo vs assist breakdown ───────────────────────────────────────────────
    section("SOLO / ASSIST BREAKDOWN (PBP)")
    print(f"\n  Avg solo tackles     : {spine['solo_tackles'].mean():.2f}")
    print(f"  Avg assist tackles   : {spine['assist_tackles'].mean():.2f}")
    print(f"  Avg combined (spine) : {spine['tackles_assists'].mean():.2f}")
    print(f"\n  % of games with ≥1 assist : {(spine['assist_tackles'] > 0).mean():.1%}")
    print(f"  Avg assists | assist > 0  : {spine[spine['assist_tackles'] > 0]['assist_tackles'].mean():.2f}")
    print(f"\n  Does PBP double-count assists? "
          f"Spine avg = {spine['tackles_assists'].mean():.2f}  |  "
          f"PFR avg (matched) = {merged['def_tackles_combined'].mean():.2f}")

    print(f"\n{'='*65}\n  VALIDATION COMPLETE\n{'='*65}\n")


if __name__ == "__main__":
    main()
