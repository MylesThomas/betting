"""
Build historical spine for NFL WR/TE receiving yards modeling.

Receiving actuals sourced from nfl_data_py import_weekly_data (gsis-based).
Snap counts joined for snap% features.

Output:
  ~/Downloads/tmp/nfl_rec_yards_historical_spine.parquet

Run:
  python src/nfl_rec_yards_modeling/scripts/build_historical_spine.py
  python src/nfl_rec_yards_modeling/scripts/build_historical_spine.py --seasons 2025
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import nfl_data_py as nfl
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


DEFAULT_SEASONS = [2023, 2024, 2025]
OUT_SPINE       = Path.home() / "Downloads" / "tmp" / "nfl_rec_yards_historical_spine.parquet"
WINDOWS         = [3, 5, 8, 16]
POSITIONS       = ["WR", "TE"]

WEEKLY_COLS = [
    "player_id", "player_display_name", "position", "recent_team",
    "season", "week", "season_type", "opponent_team",
    "targets", "receptions", "receiving_yards",
    "receiving_air_yards", "receiving_yards_after_catch",
    "receiving_epa", "target_share", "air_yards_share", "wopr",
]

_WEEKLY_OUT_COLS = [
    "player_id", "player_name", "position", "team", "season", "week", "season_type", "opponent",
    "targets", "receptions", "receiving_yards",
    "receiving_air_yards", "receiving_yards_after_catch",
    "receiving_epa", "target_share", "air_yards_share", "wopr",
]


def _build_receiving_from_pbp(season: int) -> pd.DataFrame:
    """PBP fallback when import_weekly_data is unavailable (e.g. current season)."""
    pbp = nfl.import_pbp_data(years=[season], columns=[
        "game_id", "season", "week", "season_type", "posteam", "defteam",
        "play_type", "receiver_player_id",
        "yards_gained", "air_yards", "yards_after_catch",
        "complete_pass", "epa",
    ])
    pbp = pbp[
        (pbp["season_type"] == "REG") &
        (pbp["play_type"] == "pass") &
        pbp["receiver_player_id"].notna()
    ].copy()
    pbp["rec_yds_play"] = np.where(pbp["complete_pass"] == 1, pbp["yards_gained"].fillna(0), 0)
    pbp["yac_play"]     = np.where(pbp["complete_pass"] == 1, pbp["yards_after_catch"].fillna(0), 0)

    player_agg = (
        pbp.groupby(["receiver_player_id", "posteam", "defteam", "season", "week"])
        .agg(
            receiving_yards=("rec_yds_play", "sum"),
            receptions=("complete_pass", "sum"),
            targets=("receiver_player_id", "count"),
            receiving_air_yards=("air_yards", "sum"),
            receiving_yards_after_catch=("yac_play", "sum"),
            receiving_epa=("epa", "sum"),
        )
        .reset_index()
    )
    team_agg = (
        pbp.groupby(["posteam", "season", "week"])
        .agg(team_targets=("receiver_player_id", "count"), team_air_yards=("air_yards", "sum"))
        .reset_index()
    )
    player_agg = player_agg.merge(team_agg, on=["posteam", "season", "week"])
    player_agg["target_share"]   = player_agg["targets"] / player_agg["team_targets"].replace(0, np.nan)
    player_agg["air_yards_share"] = player_agg["receiving_air_yards"] / player_agg["team_air_yards"].replace(0, np.nan)
    player_agg["wopr"]            = 1.5 * player_agg["target_share"] + 0.7 * player_agg["air_yards_share"]

    players_df = nfl.import_players()[["gsis_id", "display_name", "position"]].dropna(subset=["gsis_id"])
    player_agg = player_agg.merge(players_df, left_on="receiver_player_id", right_on="gsis_id", how="left")
    player_agg = player_agg[player_agg["position"].isin(POSITIONS)].copy()

    player_agg = player_agg.rename(columns={
        "receiver_player_id": "player_id",
        "display_name":       "player_name",
        "posteam":            "team",
        "defteam":            "opponent",
    })
    player_agg["season_type"]    = "REG"
    player_agg["receiving_yards"] = player_agg["receiving_yards"].fillna(0.0)
    return player_agg[[c for c in _WEEKLY_OUT_COLS if c in player_agg.columns]].copy()


# ── pfr_id → gsis_id bridge (for snap count join) ────────────────────────────

def build_id_bridge() -> dict[str, str]:
    players = nfl.import_players()[["gsis_id", "pfr_id"]].dropna(subset=["pfr_id", "gsis_id"])
    return dict(zip(players["pfr_id"], players["gsis_id"]))


# ── Step 1: WR/TE weekly receiving stats ─────────────────────────────────────

def build_weekly_receiving(seasons: list[int]) -> pd.DataFrame:
    print(f"  Loading weekly receiving stats for {seasons}...")
    frames = []
    for season in seasons:
        try:
            raw = nfl.import_weekly_data(years=[season])
            df  = raw[
                (raw["season_type"] == "REG") &
                (raw["position"].isin(POSITIONS))
            ][WEEKLY_COLS].copy()
            df = df.rename(columns={
                "player_display_name": "player_name",
                "recent_team": "team",
                "opponent_team": "opponent",
            })
            df["receiving_yards"] = df["receiving_yards"].fillna(0.0)
            frames.append(df)
        except Exception as e:
            print(f"    import_weekly_data failed for {season} ({e}) — using PBP fallback")
            frames.append(_build_receiving_from_pbp(season))

    weekly = pd.concat(frames, ignore_index=True, sort=False)
    print(f"    Weekly: {len(weekly):,} player-game rows  |  "
          f"{weekly['player_id'].nunique():,} unique players  |  "
          f"avg {weekly['receiving_yards'].mean():.1f} rec yards/game")
    return weekly


# ── Step 2: Snap counts ───────────────────────────────────────────────────────

def build_snap_counts(seasons: list[int], id_bridge: dict[str, str]) -> pd.DataFrame:
    print(f"  Loading snap counts for {seasons}...")
    snaps = nfl.import_snap_counts(seasons)
    reg   = snaps[snaps["game_type"] == "REG"].copy()
    off   = reg[reg["offense_snaps"] > 0][[
        "game_id", "week", "season", "pfr_player_id", "player", "position", "team",
        "offense_snaps", "offense_pct",
    ]].rename(columns={"player": "player_name_snap"}).copy()

    off["player_id_snap"] = off["pfr_player_id"].map(id_bridge)
    print(f"    Snaps: {len(off):,} rows  |  {off['player_name_snap'].nunique():,} unique players")
    return off


# ── Step 3: Opponent pass yards allowed ──────────────────────────────────────

def build_opp_pass_defense(seasons: list[int]) -> pd.DataFrame:
    print(f"  Computing opponent pass yards allowed (L3) for {seasons}...")
    pbp = nfl.import_pbp_data(seasons, columns=[
        "game_id", "week", "season", "season_type", "defteam", "passing_yards",
    ])
    pbp = pbp[pbp["season_type"] == "REG"].copy()

    pass_yds = (
        pbp.groupby(["game_id", "week", "season", "defteam"])["passing_yards"]
        .sum().reset_index()
        .rename(columns={"defteam": "team", "passing_yards": "pass_yds_allowed"})
        .sort_values(["season", "team", "week"])
    )
    pass_yds["opp_pass_yds_allowed_L3"] = (
        pass_yds.groupby(["season", "team"])["pass_yds_allowed"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )
    return pass_yds[["game_id", "team", "opp_pass_yds_allowed_L3"]].rename(
        columns={"team": "opponent"}
    )


# ── Step 4: Opponent map ──────────────────────────────────────────────────────

def build_opponent_map(seasons: list[int]) -> pd.DataFrame:
    sched = nfl.import_schedules(seasons)[["game_id", "home_team", "away_team"]]
    home  = sched.rename(columns={"home_team": "team", "away_team": "opponent"})
    away  = sched.rename(columns={"away_team": "team", "home_team": "opponent"})
    return pd.concat([
        home[["game_id", "team", "opponent"]],
        away[["game_id", "team", "opponent"]],
    ], ignore_index=True)


# ── Step 5: Game context ──────────────────────────────────────────────────────

def build_game_context(seasons: list[int]) -> pd.DataFrame:
    """
    Returns one row per (game_id, team) with pre-game betting context.

    proj_own_score: implied points for the team's own offense (drives passing volume).
      Home:  (total + spread) / 2  — home favored when spread > 0
      Away:  (total - spread) / 2
    """
    print(f"  Loading game context (spread/total) for {seasons}...")
    sched = nfl.import_schedules(seasons)[
        ["game_id", "home_team", "away_team", "spread_line", "total_line"]
    ].dropna(subset=["spread_line", "total_line"])

    home = sched.assign(
        team           = sched["home_team"],
        team_spread    = sched["spread_line"],
        game_total     = sched["total_line"],
        proj_own_score = (sched["total_line"] + sched["spread_line"]) / 2,
    )[["game_id", "team", "game_total", "team_spread", "proj_own_score"]]

    away = sched.assign(
        team           = sched["away_team"],
        team_spread    = -sched["spread_line"],
        game_total     = sched["total_line"],
        proj_own_score = (sched["total_line"] - sched["spread_line"]) / 2,
    )[["game_id", "team", "game_total", "team_spread", "proj_own_score"]]

    ctx = pd.concat([home, away], ignore_index=True)
    print(f"    Game context: {len(ctx):,} team-game rows  |  "
          f"avg total={ctx['game_total'].mean():.1f}  "
          f"avg proj_own={ctx['proj_own_score'].mean():.1f}")
    return ctx


# ── Step 6: Build game_id for weekly data ─────────────────────────────────────

def build_game_id_map(seasons: list[int]) -> pd.DataFrame:
    """Maps (season, week, team) → game_id using schedule data."""
    sched = nfl.import_schedules(seasons)[["game_id", "season", "week", "home_team", "away_team"]]
    home  = sched[["game_id", "season", "week", "home_team"]].rename(columns={"home_team": "team"})
    away  = sched[["game_id", "season", "week", "away_team"]].rename(columns={"away_team": "team"})
    return pd.concat([home, away], ignore_index=True)


# ── Step 7: Rolling features ──────────────────────────────────────────────────

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    rolling_pairs = [
        ("receiving_yards",        "receiving_yards"),
        ("target_share",           "target_share"),
        ("air_yards_share",        "air_yards_share"),
        ("wopr",                   "wopr"),
        ("offense_pct",            "snap_pct"),
    ]

    for col, label in rolling_pairs:
        if col not in df.columns:
            continue
        grp = df.groupby("player_id")[col]
        for w in WINDOWS:
            df[f"{label}_L{w}"] = grp.transform(
                lambda s, _w=w: s.shift(1).rolling(_w, min_periods=1).mean()
            )
        df[f"{label}_Lcareer"] = grp.transform(lambda s: s.shift(1).expanding().mean())

    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", type=int, default=DEFAULT_SEASONS)
    args    = parser.parse_args()
    seasons = sorted(args.seasons)

    print(f"\nBuilding rec yards spine: seasons={seasons}")
    print(f"Output: {OUT_SPINE}\n")

    id_bridge = build_id_bridge()
    weekly    = build_weekly_receiving(seasons)
    snaps     = build_snap_counts(seasons, id_bridge)
    opp_pass  = build_opp_pass_defense(seasons)
    game_ctx  = build_game_context(seasons)
    game_ids  = build_game_id_map(seasons)

    # Attach game_id to weekly data (join on season, week, team)
    print("\n  Attaching game_id to weekly data...")
    spine = weekly.merge(game_ids, on=["season", "week", "team"], how="left")
    missing_gid = spine["game_id"].isna().sum()
    if missing_gid > 0:
        print(f"    ⚠  {missing_gid:,} rows with no game_id — dropped")
    spine = spine[spine["game_id"].notna()].copy()

    # Join snap counts (gsis_id-based)
    print("  Joining snap counts...")
    snap_cols = snaps[["game_id", "season", "team", "player_id_snap",
                        "offense_snaps", "offense_pct"]].copy()
    spine = spine.merge(
        snap_cols.rename(columns={"player_id_snap": "player_id"}),
        on=["game_id", "season", "team", "player_id"],
        how="left",
    )
    spine["offense_pct"] = spine["offense_pct"].fillna(0.0)
    snap_matched = spine["offense_snaps"].notna().sum()
    print(f"    Snap info attached: {snap_matched:,} / {len(spine):,} "
          f"({snap_matched/len(spine):.1%})")

    # Attach opponent pass defense + game context
    spine = spine.merge(opp_pass, on=["game_id", "opponent"], how="left")
    spine = spine.merge(game_ctx, on=["game_id", "team"],     how="left")

    # Rolling features
    print("  Computing rolling features...")
    spine = add_rolling_features(spine)

    # Normalized name for cross-source joins
    spine["player_name_norm"] = spine["player_name"].map(_normalize_name)

    col_order = [
        "game_id", "season", "week", "player_id", "player_name", "player_name_norm",
        "position", "team", "opponent",
        "offense_snaps", "offense_pct",
        "targets", "receptions", "receiving_yards",
        "receiving_air_yards", "receiving_yards_after_catch",
        "receiving_epa", "target_share", "air_yards_share", "wopr",
        "game_total", "team_spread", "proj_own_score",
        "opp_pass_yds_allowed_L3",
        "receiving_yards_L3",  "receiving_yards_L5",  "receiving_yards_L8",  "receiving_yards_L16",  "receiving_yards_Lcareer",
        "target_share_L3",     "target_share_L5",     "target_share_L8",     "target_share_L16",     "target_share_Lcareer",
        "air_yards_share_L3",  "air_yards_share_L5",  "air_yards_share_L8",  "air_yards_share_L16",  "air_yards_share_Lcareer",
        "wopr_L3",             "wopr_L5",             "wopr_L8",             "wopr_L16",             "wopr_Lcareer",
        "snap_pct_L3",         "snap_pct_L5",         "snap_pct_L8",         "snap_pct_L16",         "snap_pct_Lcareer",
    ]
    spine = spine[[c for c in col_order if c in spine.columns]]

    OUT_SPINE.parent.mkdir(parents=True, exist_ok=True)
    spine.to_parquet(OUT_SPINE, index=False)

    print(f"\n{'='*55}")
    print(f"  Output   : {OUT_SPINE}")
    print(f"  Rows     : {len(spine):,}")
    print(f"  Players  : {spine['player_id'].nunique():,} IDs  |  "
          f"{spine['player_name'].nunique():,} names")
    print(f"  Seasons  : {spine['season'].min()}–{spine['season'].max()}")
    print(f"  Avg rec yards/game : {spine['receiving_yards'].mean():.1f}")
    for w in WINDOWS:
        col = f"receiving_yards_L{w}"
        nn  = spine[col].notna().sum() if col in spine.columns else 0
        print(f"  {col} non-null : {nn:,} ({nn/len(spine):.1%})")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
