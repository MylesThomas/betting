"""
Audit NFL tackles historical spine for data quality.

Checks:
  1. Duplicate (game_id, player_id) rows
  2. Missing opponents
  3. Week coverage per season
  4. Tackle count outliers
  5. Position breakdown (player-games, avg tackles, avg snap pct)
  6. Zero-tackle rate by position
  7. opp_run_rate_L3 null rate by week
  8. Rolling feature null rates by week
  9. Snap pct vs tackles correlation + decile breakdown
  10. Player name samples (30 random)
  11. Name collision check (display_name → multiple gsis_ids in spine)

Run:
  python src/nfl_tackles_modeling/scripts/audit_historical_spine.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import nfl_data_py as nfl
import pandas as pd

SPINE_PATH = Path.home() / "Downloads" / "tmp" / "nfl_tackles_historical_spine.parquet"

ROLLING_COLS = [
    "tackle_rate_L3", "tackle_rate_L10", "tackle_rate_Lcareer",
    "snap_pct_L3",    "snap_pct_L10",    "snap_pct_Lcareer",
]


def section(n: int, title: str) -> None:
    print(f"\n{'='*65}")
    print(f"  {n}. {title}")
    print(f"{'='*65}")


def main():
    warnings.filterwarnings("ignore")

    df = pd.read_parquet(SPINE_PATH)
    has_pid = "player_id" in df.columns

    print(f"\nLoaded spine: {len(df):,} rows  |  seasons {df['season'].min()}–{df['season'].max()}")
    print(f"  player_name unique : {df['player_name'].nunique():,}")
    if has_pid:
        print(f"  player_id unique   : {df['player_id'].nunique():,}  "
              f"({df['player_id'].isna().sum():,} null)")

    # ── 1. Duplicates ──────────────────────────────────────────────────────────
    section(1, "DUPLICATES — (game_id, player_id or player_name)")
    dup_key = ["game_id", "player_id"] if has_pid else ["game_id", "player_name"]
    dupes = df[df.duplicated(subset=dup_key, keep=False)]
    if dupes.empty:
        print(f"  ✓ No duplicates on {dup_key}")
    else:
        print(f"  ✗ {len(dupes):,} duplicate rows!")
        show_cols = ["game_id", "player_name", "team", "tackles_combined"]
        if has_pid:
            show_cols.insert(2, "player_id")
        print(dupes[show_cols].head(20).to_string(index=False))

    # ── 2. Missing opponents ───────────────────────────────────────────────────
    section(2, "MISSING OPPONENTS")
    null_opp = df["opponent"].isna().sum()
    if null_opp == 0:
        print("  ✓ All rows have an opponent")
    else:
        print(f"  ✗ {null_opp:,} rows missing opponent")
        print(df[df["opponent"].isna()][["game_id", "player_name", "team"]].head(10).to_string(index=False))

    # ── 3. Week coverage ───────────────────────────────────────────────────────
    section(3, "WEEK COVERAGE BY SEASON")
    for season, grp in df.groupby("season"):
        weeks   = sorted(grp["week"].unique())
        missing = [w for w in range(1, 19) if w not in weeks]
        status  = "✓" if not missing else f"✗ missing weeks {missing}"
        print(f"  {season}: weeks {min(weeks)}–{max(weeks)}  ({len(weeks)} weeks)  {status}")

    # ── 4. Outliers ────────────────────────────────────────────────────────────
    section(4, "TACKLE COUNT OUTLIERS — top 20 player-games")
    top_cols = ["game_id", "season", "week", "player_name", "position",
                "team", "tackles_combined"]
    if has_pid:
        top_cols.insert(4, "player_id")
    print(df.nlargest(20, "tackles_combined")[top_cols].to_string(index=False))

    # ── 5. Position breakdown ──────────────────────────────────────────────────
    section(5, "POSITION BREAKDOWN")
    pos = (
        df.groupby("position", dropna=False)
        .agg(
            player_games    = ("player_name", "count"),
            unique_players  = ("player_name", "nunique"),
            avg_tackles     = ("tackles_combined", "mean"),
            avg_snap_pct    = ("defense_pct", "mean"),
        )
        .sort_values("player_games", ascending=False)
    )
    pos["avg_tackles"]  = pos["avg_tackles"].round(2)
    pos["avg_snap_pct"] = pos["avg_snap_pct"].round(3)
    print(pos.to_string())

    # ── 6. Zero-tackle rate by position ───────────────────────────────────────
    section(6, "ZERO-TACKLE RATE BY POSITION")
    zero = (
        df.groupby("position", dropna=False)
        .apply(lambda g: (g["tackles_combined"] == 0).mean(), include_groups=False)
        .rename("zero_pct")
        .reset_index()
        .sort_values("zero_pct", ascending=False)
    )
    zero["zero_pct"] = zero["zero_pct"].map("{:.1%}".format)
    print(zero.to_string(index=False))

    # ── 7. opp_run_rate_L3 null rate by week ──────────────────────────────────
    section(7, "OPP_RUN_RATE_L3 NULL RATE BY WEEK")
    if "opp_run_rate_L3" in df.columns:
        orr = (
            df.groupby("week")["opp_run_rate_L3"]
            .apply(lambda s: s.isna().mean())
            .rename("null_pct")
            .reset_index()
        )
        orr["null_pct"] = orr["null_pct"].map("{:.1%}".format)
        print(orr.to_string(index=False))
    else:
        print("  opp_run_rate_L3 column not present")

    # ── 8. Rolling feature null rates by week ─────────────────────────────────
    section(8, "ROLLING FEATURE NULL RATES BY WEEK")
    present_rolling = [c for c in ROLLING_COLS if c in df.columns]
    null_by_week = (
        df.groupby("week")[present_rolling]
        .apply(lambda g: g.isna().mean())
    )
    # format as percentages
    null_by_week = null_by_week.apply(
        lambda col: col.map("{:.0%}".format)
    )
    print(null_by_week.to_string())

    # ── 9. Snap pct vs tackles correlation ────────────────────────────────────
    section(9, "SNAP PCT vs TACKLES CORRELATION")
    valid = df[["defense_pct", "tackles_combined"]].dropna()
    corr  = valid["defense_pct"].corr(valid["tackles_combined"])
    print(f"  Pearson r = {corr:.3f}  (n={len(valid):,})")

    df["_snap_decile"] = pd.qcut(df["defense_pct"], q=10, labels=False, duplicates="drop")
    deciles = (
        df.groupby("_snap_decile")["tackles_combined"]
        .agg(avg_tackles="mean", n="count")
    )
    deciles["avg_tackles"] = deciles["avg_tackles"].round(2)
    print("\n  Snap decile (0=lowest) → avg tackles:")
    print(deciles.to_string())
    df = df.drop(columns=["_snap_decile"])

    # ── 10. Player name samples ────────────────────────────────────────────────
    section(10, "PLAYER NAME SAMPLES — 30 random")
    sample_names = (
        df["player_name"].dropna().drop_duplicates()
        .sample(min(30, df["player_name"].nunique()), random_state=42)
        .sort_values()
        .tolist()
    )
    for name in sample_names:
        print(f"  {name}")

    # ── 11. Name collision check ───────────────────────────────────────────────
    section(11, "NAME COLLISION CHECK — display_name → multiple gsis_ids")
    players_df = nfl.import_players()[["gsis_id", "display_name"]].dropna(subset=["gsis_id", "display_name"])
    name_counts = (
        players_df.groupby("display_name")["gsis_id"]
        .nunique()
        .rename("n_gsis_ids")
        .reset_index()
    )
    collisions = name_counts[name_counts["n_gsis_ids"] > 1]
    spine_names = set(df["player_name"].dropna().unique())
    in_spine    = collisions[collisions["display_name"].isin(spine_names)]

    if in_spine.empty:
        print(f"  ✓ No name collisions among {len(spine_names):,} player names in the spine")
    else:
        print(f"  ✗ {len(in_spine)} names in the spine map to multiple gsis_ids:")
        print(in_spine.sort_values("n_gsis_ids", ascending=False).to_string(index=False))
        if has_pid:
            print("\n  (rolling features use player_id — these collisions are handled)")
        else:
            print("\n  ⚠  Spine lacks player_id — re-run build_historical_spine.py to fix")

    print(f"\n{'='*65}\n  AUDIT COMPLETE\n{'='*65}\n")


if __name__ == "__main__":
    main()
