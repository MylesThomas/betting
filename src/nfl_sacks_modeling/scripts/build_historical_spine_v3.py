"""
Build multi-season historical spine for NFL sacks modeling (v3).

v3 vs v1/v2: sack counts come from PFR box score stats (def_sacks) instead of
play-by-play extraction. No name abbreviation or string matching involved — joins
on pfr_player_id directly. PFR data available from 2018 onward only.

PFR source: nflverse-data pfr_advstats (same data as nfl_data_py import_weekly_pfr("def"))
fetched directly so no nfl_data_py version constraint.

Outputs:
  ~/Downloads/tmp/nfl_sacks_historical_spine_v3.parquet
  ~/Downloads/tmp/nfl_sacks_spine_cache_v3/{season}/{pfr_player_id}.parquet

Run:
    python src/nfl_sacks_modeling/scripts/build_historical_spine_v3.py [--seasons 2018 2019 ...]
    python src/nfl_sacks_modeling/scripts/build_historical_spine_v3.py --force-refetch
"""

import argparse
import warnings
from pathlib import Path

import nfl_data_py as nfl
import pandas as pd
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
CACHE_DIR   = Path.home() / "Downloads" / "tmp" / "nfl_sacks_spine_cache_v3"
OUT_SPINE   = Path.home() / "Downloads" / "tmp" / "nfl_sacks_historical_spine_v3.parquet"

PFR_URL           = "https://github.com/nflverse/nflverse-data/releases/download/pfr_advstats/advstats_week_def_{year}.parquet"
SNAP_FIRST_SEASON = 2018   # PFR def data not available before 2018
CURRENT_SEASON    = 2025


# ── Config ─────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)["nfl_sacks_model"]


# ── PFR box score: def_sacks per player per game ───────────────────────────────

def build_pfr_stats(seasons: list[int]) -> pd.DataFrame:
    print(f"  Loading PFR def stats for seasons {seasons[0]}–{seasons[-1]}...")
    frames = []
    for yr in seasons:
        url = PFR_URL.format(year=yr)
        try:
            df = pd.read_parquet(url)
            frames.append(df)
            print(f"    {yr}: {len(df):,} rows")
        except Exception as e:
            print(f"    {yr}: FAILED — {e}")

    pfr = pd.concat(frames, ignore_index=True)
    reg = pfr[pfr["game_type"] == "REG"].copy()

    reg = reg[["game_id", "season", "week", "team", "pfr_player_id", "pfr_player_name",
               "def_sacks", "def_times_hitqb", "def_pressures", "def_tackles_combined"]].copy()
    reg["def_sacks"]         = reg["def_sacks"].fillna(0)
    reg["def_times_hitqb"]   = reg["def_times_hitqb"].fillna(0)
    reg["def_pressures"]     = reg["def_pressures"].fillna(0)
    reg["def_tackles_combined"] = reg["def_tackles_combined"].fillna(0)

    total = reg["def_sacks"].sum()
    print(f"    PFR: {len(reg):,} player-game records  ({total:.1f} total def_sacks)")
    return reg


# ── Snap counts ────────────────────────────────────────────────────────────────

def build_snap_counts(seasons: list[int]) -> pd.DataFrame:
    print(f"  Loading snap counts for seasons {seasons[0]}–{seasons[-1]}...")
    snaps = nfl.import_snap_counts(seasons)
    reg   = snaps[snaps["game_type"] == "REG"].copy()
    def_snaps = reg[reg["defense_snaps"] > 0][
        ["game_id", "week", "season", "player", "pfr_player_id",
         "position", "team", "defense_snaps", "defense_pct"]
    ].copy()
    print(f"    Snaps: {len(def_snaps):,} player-game records  ({def_snaps['pfr_player_id'].nunique():,} unique players)")
    return def_snaps


# ── Join snap counts + PFR ────────────────────────────────────────────────────

def join_season(snaps: pd.DataFrame, pfr: pd.DataFrame) -> pd.DataFrame:
    joined = snaps.merge(
        pfr[["game_id", "season", "week", "pfr_player_id",
             "def_sacks", "def_times_hitqb", "def_pressures"]],
        on=["game_id", "season", "week", "pfr_player_id"],
        how="left",
    )
    joined["sacks"]   = joined["def_sacks"].fillna(0)
    joined["qb_hits"] = joined["def_times_hitqb"].fillna(0).astype(int)
    joined = joined.drop(columns=["def_sacks", "def_times_hitqb", "def_pressures"], errors="ignore")
    return joined


# ── Per-player-season cache ────────────────────────────────────────────────────

def season_cache_dir(season: int) -> Path:
    return CACHE_DIR / str(season)


def write_season_cache(season: int, joined: pd.DataFrame):
    d = season_cache_dir(season)
    d.mkdir(parents=True, exist_ok=True)
    n = 0
    for pid, grp in joined.groupby("pfr_player_id"):
        grp.reset_index(drop=True).to_parquet(d / f"{pid}.parquet", index=False)
        n += 1
    print(f"    Cached {n} player files → {d}")


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
    parser.add_argument("--force-refetch", action="store_true")
    args = parser.parse_args()

    cfg     = load_config()
    windows = cfg["rolling_windows"]
    seasons = sorted(args.seasons)
    print(f"\nSeasons : {seasons[0]}–{seasons[-1]}  ({len(seasons)} seasons)")
    print(f"Windows : {windows}")
    print(f"Cache   : {CACHE_DIR}")
    print(f"Source  : PFR box score (def_sacks) — no PBP name matching\n")

    seasons_to_fetch = []
    for s in seasons:
        d = season_cache_dir(s)
        if args.force_refetch or not d.exists() or not any(d.glob("*.parquet")):
            seasons_to_fetch.append(s)
        else:
            n = len(list(d.glob("*.parquet")))
            print(f"  Season {s}: cache exists ({n} player files) — skipping fetch")

    if seasons_to_fetch:
        print(f"Fetching {len(seasons_to_fetch)} season(s): {seasons_to_fetch}")
        pfr   = build_pfr_stats(seasons_to_fetch)
        snaps = build_snap_counts(seasons_to_fetch)

        for s in seasons_to_fetch:
            print(f"\n  Processing season {s}...")
            s_snaps = snaps[snaps["season"] == s].copy()
            s_pfr   = pfr[pfr["season"] == s].copy()
            joined  = join_season(s_snaps, s_pfr)

            n_with_sacks = (joined["sacks"] > 0).sum()
            n_unmatched  = joined["sacks"].isna().sum()
            print(f"    Joined: {len(joined):,} rows  |  {n_with_sacks} rows with sacks  |  {n_unmatched} unmatched")
            write_season_cache(s, joined)

    print(f"\nLoading all {len(seasons)} seasons from cache...")
    frames = []
    for s in seasons:
        df_s = read_season_cache(s)
        if not df_s.empty:
            frames.append(df_s)
            print(f"  {s}: {len(df_s):,} rows  ({df_s['pfr_player_id'].nunique():,} players)")
    spine = pd.concat(frames, ignore_index=True)
    print(f"\nFull spine: {len(spine):,} rows  ({spine['pfr_player_id'].nunique():,} unique players)")

    print("Adding position groups...")
    spine = add_position_group(spine, cfg["position_groups"], cfg["position_side"])

    print(f"Computing rolling features (windows={windows})...")
    spine = add_rolling_features(spine, windows)

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
    print(f"  Note    : PFR data only available 2018+")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
