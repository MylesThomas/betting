"""
Build multi-season historical spine for NFL sacks modeling.

Loads PBP + snap counts for all available seasons (2013–present), joins them
per player per game, and writes individual cache files:
  ~/Downloads/tmp/nfl_sacks_spine_cache/{season}/{pfr_player_id}.parquet

Then assembles the full spine, computes lagged rolling features on the complete
career history, and saves:
  ~/Downloads/tmp/nfl_sacks_historical_spine.parquet

Rolling windows come from config.yaml. Snap counts only available from 2013.
PBP sack/qb_hit data also loaded from 2013 to match the snap spine anchor.

Run:
    python src/nfl_sacks_modeling/scripts/build_historical_spine.py [--seasons 2013 2014 ...]
    python src/nfl_sacks_modeling/scripts/build_historical_spine.py --force-refetch
"""

import argparse
import warnings
from pathlib import Path

import nfl_data_py as nfl
import numpy as np
import pandas as pd
import yaml

CONFIG_PATH  = Path(__file__).resolve().parents[1] / "config.yaml"
CACHE_DIR    = Path.home() / "Downloads" / "tmp" / "nfl_sacks_spine_cache"
OUT_SPINE    = Path.home() / "Downloads" / "tmp" / "nfl_sacks_historical_spine.parquet"

SNAP_FIRST_SEASON = 2013
CURRENT_SEASON    = 2025


# ── Config ─────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)["nfl_sacks_model"]


# ── PBP: sacks + qb_hits per player per game ──────────────────────────────────

def build_pbp_stats(seasons: list[int]) -> pd.DataFrame:
    print(f"  Loading PBP for seasons {seasons[0]}–{seasons[-1]}...")
    pbp = nfl.import_pbp_data(seasons, columns=[
        "game_id", "week", "season", "season_type", "defteam",
        "sack", "sack_player_name",
        "half_sack_1_player_name", "half_sack_2_player_name",
        "lateral_sack_player_name",
        "qb_hit", "qb_hit_1_player_name", "qb_hit_2_player_name",
    ])
    reg = pbp[pbp["season_type"] == "REG"].copy()

    sack_rows = []
    full_sacks = reg[reg["sack"] == 1]
    for _, r in full_sacks.iterrows():
        if pd.notna(r["half_sack_1_player_name"]):
            sack_rows.append({"game_id": r["game_id"], "week": r["week"],
                               "season": r["season"], "defteam": r["defteam"],
                               "pbp_name": r["half_sack_1_player_name"], "sacks": 0.5})
            if pd.notna(r["half_sack_2_player_name"]):
                sack_rows.append({"game_id": r["game_id"], "week": r["week"],
                                   "season": r["season"], "defteam": r["defteam"],
                                   "pbp_name": r["half_sack_2_player_name"], "sacks": 0.5})
        elif pd.notna(r["sack_player_name"]):
            sack_rows.append({"game_id": r["game_id"], "week": r["week"],
                               "season": r["season"], "defteam": r["defteam"],
                               "pbp_name": r["sack_player_name"], "sacks": 1.0})
            if pd.notna(r["lateral_sack_player_name"]):
                sack_rows.append({"game_id": r["game_id"], "week": r["week"],
                                   "season": r["season"], "defteam": r["defteam"],
                                   "pbp_name": r["lateral_sack_player_name"], "sacks": 1.0})

    sacks_df = (pd.DataFrame(sack_rows)
                .groupby(["game_id", "week", "season", "defteam", "pbp_name"], as_index=False)["sacks"]
                .sum())

    hit_rows = []
    hits = reg[reg["qb_hit"] == 1]
    for _, r in hits.iterrows():
        for col in ["qb_hit_1_player_name", "qb_hit_2_player_name"]:
            if pd.notna(r[col]):
                hit_rows.append({"game_id": r["game_id"], "week": r["week"],
                                  "season": r["season"], "defteam": r["defteam"],
                                  "pbp_name": r[col]})

    hits_df = (pd.DataFrame(hit_rows)
               .groupby(["game_id", "week", "season", "defteam", "pbp_name"], as_index=False)
               .size().rename(columns={"size": "qb_hits"}))

    pbp_stats = sacks_df.merge(hits_df, on=["game_id", "week", "season", "defteam", "pbp_name"], how="outer")
    pbp_stats["sacks"]   = pbp_stats["sacks"].fillna(0)
    pbp_stats["qb_hits"] = pbp_stats["qb_hits"].fillna(0).astype(int)

    total_sacks = pbp_stats["sacks"].sum()
    print(f"    PBP: {len(pbp_stats):,} player-game records  ({total_sacks:.1f} total sacks)")
    return pbp_stats


# ── Snap counts ────────────────────────────────────────────────────────────────

def build_snap_counts(seasons: list[int]) -> pd.DataFrame:
    print(f"  Loading snap counts for seasons {seasons[0]}–{seasons[-1]}...")
    snaps = nfl.import_snap_counts(seasons)
    reg   = snaps[snaps["game_type"] == "REG"].copy()
    def_snaps = reg[reg["defense_snaps"] > 0][
        ["game_id", "week", "season", "player", "pfr_player_id",
         "position", "team", "defense_snaps", "defense_pct"]
    ].copy()

    def full_to_pbp(name: str) -> str:
        parts = name.strip().split()
        return f"{parts[0][0]}.{parts[-1]}" if len(parts) >= 2 else name

    def_snaps["pbp_name"] = def_snaps["player"].apply(full_to_pbp)
    print(f"    Snaps: {len(def_snaps):,} player-game records  ({def_snaps['pfr_player_id'].nunique():,} unique players)")
    return def_snaps


# ── Join PBP + snaps ───────────────────────────────────────────────────────────

def join_season(snaps: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    # Join on team to avoid cross-matching same-abbreviation players from both teams
    # in the same game (e.g. D.Lawrence = DeMarcus Lawrence DAL + Dexter Lawrence NYG)
    joined = snaps.merge(
        pbp[["game_id", "week", "season", "defteam", "pbp_name", "sacks", "qb_hits"]],
        left_on=["game_id", "week", "season", "team", "pbp_name"],
        right_on=["game_id", "week", "season", "defteam", "pbp_name"],
        how="left",
    ).drop(columns=["defteam"], errors="ignore")
    joined["sacks"]   = joined["sacks"].fillna(0)
    joined["qb_hits"] = joined["qb_hits"].fillna(0).astype(int)
    return joined


# ── Per-player-season cache ────────────────────────────────────────────────────

def season_cache_dir(season: int) -> Path:
    return CACHE_DIR / str(season)


def season_is_cached(season: int, player_ids: set[str]) -> bool:
    d = season_cache_dir(season)
    if not d.exists():
        return False
    cached = {f.stem for f in d.glob("*.parquet")}
    return player_ids.issubset(cached)


def write_season_cache(season: int, joined: pd.DataFrame):
    d = season_cache_dir(season)
    d.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for pid, grp in joined.groupby("pfr_player_id"):
        grp.reset_index(drop=True).to_parquet(d / f"{pid}.parquet", index=False)
        n_written += 1
    print(f"    Cached {n_written} player files → {d}")


def read_season_cache(season: int) -> pd.DataFrame:
    d = season_cache_dir(season)
    files = list(d.glob("*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()


# ── Rolling features ───────────────────────────────────────────────────────────

def add_rolling_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    df = df.sort_values(["pfr_player_id", "season", "week", "game_id"]).reset_index(drop=True)
    df["games_played_ytd"] = df.groupby(["pfr_player_id", "season"]).cumcount()

    feature_cols = [
        ("sacks",       "sack_rate"),
        ("qb_hits",     "qbhit_rate"),
        ("defense_pct", "snap_pct"),
    ]
    for src_col, feat_name in feature_cols:
        for w in windows:
            wlabel = "career" if w >= 999 else str(w)
            win    = 10_000   if w >= 999 else w
            df[f"{feat_name}_L{wlabel}"] = (
                df.groupby("pfr_player_id")[src_col]
                .transform(lambda s, _w=win: s.shift(1).rolling(_w, min_periods=1).mean())
            )
    return df


# ── Position group ─────────────────────────────────────────────────────────────

def add_position_group(df: pd.DataFrame, pos_groups: dict, pos_side: dict) -> pd.DataFrame:
    inv_group = {pos.upper(): grp for grp, positions in pos_groups.items() for pos in positions}
    inv_side  = {pos.upper(): side for side, positions in pos_side.items() for pos in positions}
    df["pos_group"] = df["position"].str.upper().map(inv_group).fillna("OTH")
    df["pos_side"]  = df["position"].str.upper().map(inv_side).fillna("other")
    return df


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", type=int,
                        default=list(range(SNAP_FIRST_SEASON, CURRENT_SEASON + 1)))
    parser.add_argument("--force-refetch", action="store_true",
                        help="Ignore cache and re-fetch all seasons from nfl_data_py")
    args = parser.parse_args()

    cfg     = load_config()
    windows = cfg["rolling_windows"]
    seasons = sorted(args.seasons)
    print(f"\nSeasons : {seasons[0]}–{seasons[-1]}  ({len(seasons)} seasons)")
    print(f"Windows : {windows}")
    print(f"Cache   : {CACHE_DIR}")

    # ── Step 1: Fetch + cache any missing seasons ──────────────────────────────
    seasons_to_fetch = []
    for s in seasons:
        d = season_cache_dir(s)
        if args.force_refetch or not d.exists() or not any(d.glob("*.parquet")):
            seasons_to_fetch.append(s)
        else:
            n = len(list(d.glob("*.parquet")))
            print(f"  Season {s}: cache exists ({n} player files) — skipping fetch")

    if seasons_to_fetch:
        print(f"\nFetching {len(seasons_to_fetch)} season(s) from nfl_data_py: {seasons_to_fetch}")
        pbp   = build_pbp_stats(seasons_to_fetch)
        snaps = build_snap_counts(seasons_to_fetch)

        for s in seasons_to_fetch:
            print(f"\n  Processing season {s}...")
            s_snaps = snaps[snaps["season"] == s].copy()
            s_pbp   = pbp[pbp["season"] == s].copy()
            joined  = join_season(s_snaps, s_pbp)
            write_season_cache(s, joined)

    # ── Step 2: Load full spine from cache ─────────────────────────────────────
    print(f"\nLoading all {len(seasons)} seasons from cache...")
    frames = []
    for s in seasons:
        df_s = read_season_cache(s)
        if not df_s.empty:
            frames.append(df_s)
            print(f"  {s}: {len(df_s):,} rows  ({df_s['pfr_player_id'].nunique():,} players)")
    spine = pd.concat(frames, ignore_index=True)
    print(f"\nFull spine: {len(spine):,} rows  ({spine['pfr_player_id'].nunique():,} unique players)")

    # ── Step 3: Position groups ────────────────────────────────────────────────
    print("Adding position groups...")
    spine = add_position_group(spine, cfg["position_groups"], cfg["position_side"])

    # ── Step 4: Rolling features on full career history ────────────────────────
    print(f"Computing rolling features (windows={windows}) on full {len(seasons)}-season history...")
    spine = add_rolling_features(spine, windows)

    # ── Step 5: Save ───────────────────────────────────────────────────────────
    col_order = [
        "game_id", "season", "week", "player", "pfr_player_id",
        "position", "pos_group", "pos_side", "team",
        "defense_snaps", "defense_pct", "sacks", "qb_hits",
        "games_played_ytd",
        *[f"{f}_L{('career' if w >= 999 else w)}"
          for f in ["sack_rate", "qbhit_rate", "snap_pct"]
          for w in windows],
    ]
    spine = spine[[c for c in col_order if c in spine.columns]]
    spine.to_parquet(OUT_SPINE, index=False)

    print(f"\n{'='*55}")
    print(f"  Output  : {OUT_SPINE}")
    print(f"  Rows    : {len(spine):,}")
    print(f"  Players : {spine['pfr_player_id'].nunique():,}")
    print(f"  Seasons : {spine['season'].min()}–{spine['season'].max()}")
    print(f"  Sack rate non-null (L1): {spine['sack_rate_L1'].notna().sum():,} rows")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
