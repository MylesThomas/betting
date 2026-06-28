"""
Build historical spine for NFL tackles modeling.

Tackle actuals sourced from PFR (Pro Football Reference) weekly defensive stats
via nfl_data_py — the official NFL count, same definition used by sportsbooks:
  def_tackles_combined = solo + assist (each = 1)

Snap counts joined for snap% features. PBP not used for tackle extraction.

Output:
  ~/Downloads/tmp/nfl_tackles_historical_spine.parquet

Run:
  python src/nfl_tackles_modeling/scripts/build_historical_spine.py
  python src/nfl_tackles_modeling/scripts/build_historical_spine.py --seasons 2025
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import nfl_data_py as nfl
import pandas as pd

_SUFFIX_RE  = re.compile(r"\s*,?\s*(Jr\.?|Sr\.?|II{1,2}|IV|V)\.?$", re.IGNORECASE)
_SPECIAL_RE = re.compile(r"['\.\-,]")

def _normalize_name(name: str) -> str:
    s = str(name).strip()
    s = _SUFFIX_RE.sub("", s)
    s = _SPECIAL_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

DEFAULT_SEASONS = [2024, 2025]
OUT_SPINE       = Path.home() / "Downloads" / "tmp" / "nfl_tackles_historical_spine.parquet"
WINDOWS         = [3, 5, 8, 16]


# ── Shared: pfr_id → gsis_id bridge ───────────────────────────────────────────

def build_id_bridge() -> dict[str, str]:
    players = nfl.import_players()[["gsis_id", "pfr_id", "display_name"]].dropna(subset=["pfr_id", "gsis_id"])
    return (
        players.set_index("pfr_id")[["gsis_id", "display_name"]].to_dict(orient="index")
    )


# ── Step 1: PFR weekly defense → tackles per player per game ──────────────────

def build_pfr_tackles(seasons: list[int], bridge: dict) -> pd.DataFrame:
    print(f"  Loading PFR weekly defensive stats for {seasons}...")
    frames = []
    for season in seasons:
        df = nfl.import_weekly_pfr(s_type="def", years=[season])
        frames.append(df)
    pfr = pd.concat(frames, ignore_index=True)
    pfr = pfr[pfr["game_type"] == "REG"][[
        "game_id", "season", "week", "team",
        "pfr_player_id", "pfr_player_name", "def_tackles_combined",
    ]].copy()

    pfr["tackles_combined"] = pfr["def_tackles_combined"].fillna(0).astype(int)
    pfr["player_id"]   = pfr["pfr_player_id"].map(lambda x: bridge.get(x, {}).get("gsis_id"))
    pfr["player_name"] = pfr["pfr_player_id"].map(lambda x: bridge.get(x, {}).get("display_name"))
    # Fall back to PFR name where display_name mapping is missing
    pfr["player_name"] = pfr["player_name"].fillna(pfr["pfr_player_name"])

    n_missing_id   = pfr["player_id"].isna().sum()
    n_missing_name = (pfr["player_name"] == pfr["pfr_player_name"]).sum()
    if n_missing_id > 0:
        print(f"    ⚠  {n_missing_id:,} rows ({n_missing_id/len(pfr):.1%}) with no gsis_id — "
              f"rolling will fall back to player_name")

    print(f"    PFR: {len(pfr):,} player-game rows  |  "
          f"{pfr['pfr_player_id'].nunique():,} unique players  |  "
          f"avg {pfr['tackles_combined'].mean():.2f} tackles/game")
    return pfr


# ── Step 2: Snap counts ────────────────────────────────────────────────────────

def build_snap_counts(seasons: list[int], bridge: dict) -> pd.DataFrame:
    print(f"  Loading snap counts for {seasons}...")
    snaps    = nfl.import_snap_counts(seasons)
    reg      = snaps[snaps["game_type"] == "REG"].copy()
    def_snaps = reg[reg["defense_snaps"] > 0][[
        "game_id", "week", "season", "pfr_player_id", "player", "position", "team",
        "defense_snaps", "defense_pct",
    ]].rename(columns={"player": "player_name_snap"}).copy()

    def_snaps["player_id_snap"] = def_snaps["pfr_player_id"].map(
        lambda x: bridge.get(x, {}).get("gsis_id")
    )
    print(f"    Snaps: {len(def_snaps):,} player-game rows  |  "
          f"{def_snaps['player_name_snap'].nunique():,} unique players")
    return def_snaps


# ── Step 3: Opponent run rate ──────────────────────────────────────────────────

def build_opp_features(seasons: list[int]) -> pd.DataFrame:
    print(f"  Computing opponent run rates for {seasons}...")
    pbp = nfl.import_pbp_data(seasons, columns=[
        "game_id", "week", "season", "season_type", "posteam", "play_type",
    ])
    pbp = pbp[pbp["season_type"] == "REG"].copy()

    run_rates = (
        pbp[pbp["play_type"].isin(["run", "pass"])]
        .groupby(["game_id", "week", "season", "posteam"])
        .agg(runs=("play_type", lambda x: (x == "run").sum()), plays=("play_type", "count"))
        .assign(run_rate=lambda d: d["runs"] / d["plays"])
        .reset_index()
        .sort_values(["season", "posteam", "week"])
    )
    run_rates["opp_run_rate_L3"] = (
        run_rates.groupby(["season", "posteam"])["run_rate"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )
    return run_rates[["game_id", "posteam", "opp_run_rate_L3"]].rename(columns={"posteam": "opponent"})


# ── Step 4: Opponent map ───────────────────────────────────────────────────────

def build_opponent_map(seasons: list[int]) -> pd.DataFrame:
    sched = nfl.import_schedules(seasons)[["game_id", "week", "home_team", "away_team"]]
    home  = sched.rename(columns={"home_team": "team", "away_team": "opponent"})
    away  = sched.rename(columns={"away_team": "team", "home_team": "opponent"})
    return pd.concat([home[["game_id", "team", "opponent"]],
                      away[["game_id", "team", "opponent"]]], ignore_index=True)


# ── Step 5: Game context (spread, total, projected scores) ────────────────────

def build_game_context(seasons: list[int]) -> pd.DataFrame:
    """
    Returns one row per (game_id, team) with pre-game betting context.

    spread_line convention in nfl_data_py: positive = home team favored.

    Derived columns (all from the defending team's perspective):
      game_total      — over/under line
      team_spread     — points team is favored by (positive = team is favored)
      proj_opp_score  — opponent's implied points = (total − team_spread) / 2
                        higher → opponent offense runs more plays → more tackles
    """
    print(f"  Loading game context (spread/total) for {seasons}...")
    sched = nfl.import_schedules(seasons)[
        ["game_id", "home_team", "away_team", "spread_line", "total_line"]
    ].dropna(subset=["spread_line", "total_line"])

    # Home team rows: team_spread = spread_line (positive = home favored)
    home = sched.assign(
        team        = sched["home_team"],
        team_spread = sched["spread_line"],
        game_total  = sched["total_line"],
        # home defense faces away offense; away projects less when home is favored
        proj_opp_score = (sched["total_line"] - sched["spread_line"]) / 2,
    )[["game_id", "team", "game_total", "team_spread", "proj_opp_score"]]

    # Away team rows: team_spread = -spread_line (flip: away favored = positive)
    away = sched.assign(
        team        = sched["away_team"],
        team_spread = -sched["spread_line"],
        game_total  = sched["total_line"],
        # away defense faces home offense; home projects more when home is favored
        proj_opp_score = (sched["total_line"] + sched["spread_line"]) / 2,
    )[["game_id", "team", "game_total", "team_spread", "proj_opp_score"]]

    ctx = pd.concat([home, away], ignore_index=True)
    print(f"    Game context: {len(ctx):,} team-game rows  |  "
          f"avg total={ctx['game_total'].mean():.1f}  "
          f"avg proj_opp={ctx['proj_opp_score'].mean():.1f}")
    return ctx


# ── Step 6: Rolling features ───────────────────────────────────────────────────

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df["_roll_key"] = df["player_id"].fillna(df["player_name"])
    df = df.sort_values(["_roll_key", "season", "week"]).reset_index(drop=True)

    for col, label in [("tackles_combined", "tackle_rate"), ("defense_pct", "snap_pct")]:
        grp = df.groupby("_roll_key")[col]
        for w in WINDOWS:
            df[f"{label}_L{w}"] = grp.transform(
                lambda s, _w=w: s.shift(1).rolling(_w, min_periods=1).mean()
            )
        df[f"{label}_Lcareer"] = grp.transform(lambda s: s.shift(1).expanding().mean())

    df = df.drop(columns=["_roll_key"])
    return df


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", type=int, default=DEFAULT_SEASONS)
    args    = parser.parse_args()
    seasons = sorted(args.seasons)

    print(f"\nBuilding tackles spine (PFR source): seasons={seasons}")
    print(f"Output: {OUT_SPINE}\n")

    bridge   = build_id_bridge()
    tackles  = build_pfr_tackles(seasons, bridge)
    snaps    = build_snap_counts(seasons, bridge)
    opp_map  = build_opponent_map(seasons)
    opp_feat = build_opp_features(seasons)
    game_ctx = build_game_context(seasons)

    # PFR is the authoritative universe — left join snap counts for snap% / position
    print("\n  Joining PFR tackles + snap counts...")
    snap_cols = snaps[["game_id", "season", "team", "player_id_snap", "player_name_snap",
                        "position", "defense_snaps", "defense_pct"]].copy()

    # Primary join: match on player_id (gsis_id bridge)
    has_id   = tackles[tackles["player_id"].notna()]
    no_id    = tackles[tackles["player_id"].isna()]

    joined_id = has_id.merge(
        snap_cols.rename(columns={"player_id_snap": "player_id"}),
        on=["game_id", "season", "team", "player_id"],
        how="left",
    )

    # Fallback for the tiny slice with no player_id: match on name
    joined_name = no_id.merge(
        snap_cols.rename(columns={"player_name_snap": "player_name"}),
        on=["game_id", "season", "team", "player_name"],
        how="left",
    )

    spine = pd.concat([joined_id, joined_name], ignore_index=True)
    spine = spine.drop(columns=["player_id_snap", "player_name_snap"], errors="ignore")
    spine["defense_pct"] = spine["defense_pct"].fillna(0.0)

    snap_matched = spine["defense_snaps"].notna().sum()
    print(f"    Snap info attached: {snap_matched:,} / {len(spine):,} rows "
          f"({snap_matched/len(spine):.1%})")

    # Attach opponent + game context
    spine = spine.merge(opp_map,  on=["game_id", "team"],     how="left")
    spine = spine.merge(opp_feat, on=["game_id", "opponent"], how="left")
    spine = spine.merge(game_ctx, on=["game_id", "team"],     how="left")

    # Rolling features
    print("  Computing rolling features...")
    spine = add_rolling_features(spine)

    # Normalized name for cross-source joins (odds API ↔ spine)
    spine["player_name_norm"] = spine["player_name"].map(_normalize_name)

    # Column order
    col_order = [
        "game_id", "season", "week", "player_id", "player_name", "player_name_norm",
        "position", "team", "opponent",
        "defense_snaps", "defense_pct",
        "tackles_combined",
        # game-level betting context (pre-game, no leakage)
        "game_total", "team_spread", "proj_opp_score",
        # opponent tendency
        "opp_run_rate_L3",
        # player rolling features
        "tackle_rate_L3", "tackle_rate_L5", "tackle_rate_L8", "tackle_rate_L16", "tackle_rate_Lcareer",
        "snap_pct_L3",    "snap_pct_L5",    "snap_pct_L8",    "snap_pct_L16",    "snap_pct_Lcareer",
    ]
    spine = spine[[c for c in col_order if c in spine.columns]]

    OUT_SPINE.parent.mkdir(parents=True, exist_ok=True)
    spine.to_parquet(OUT_SPINE, index=False)

    pid_null = spine["player_id"].isna().sum()
    print(f"\n{'='*55}")
    print(f"  Output   : {OUT_SPINE}")
    print(f"  Rows     : {len(spine):,}")
    print(f"  Players  : {spine['player_name'].nunique():,} names  |  "
          f"{spine['player_id'].nunique():,} IDs")
    print(f"  Seasons  : {spine['season'].min()}–{spine['season'].max()}")
    print(f"  Avg tackles/game   : {spine['tackles_combined'].mean():.2f}")
    print(f"  player_id null     : {pid_null:,} ({pid_null/len(spine):.1%})")
    for w in WINDOWS:
        col = f"tackle_rate_L{w}"
        nn  = spine[col].notna().sum() if col in spine.columns else 0
        print(f"  {col} non-null : {nn:,} ({nn/len(spine):.1%})")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
