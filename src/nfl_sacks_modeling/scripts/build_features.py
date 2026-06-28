"""
Build feature matrix for the NFL sacks logistic regression model.

Spine: all defensive player-games with snaps > 0 (for accurate rolling history).
Output filtered to rows where prop_median_price_over is not NaN (at least one book posted Over at 0.5).

Features:
  - Rolling sack rate, qb_hit rate, snap% (windows from config, lagged — no look-ahead)
  - Game total + team spread (pre-kick snapshot, from backfilled game lines)
  - Position group (DL / LB / DB / OTH)
  - games_played_ytd (prior games in season for that player)

Target:
  - 1 = sacks >= 1.0  (win)
  - 0 = sacks == 0.0  (loss)
  - NaN = sacks == 0.5 (push, dropped from training per config)

Output:
  ~/Downloads/tmp/nfl_sacks_features_2025.parquet

Run:
  python src/nfl_sacks_modeling/scripts/build_features.py
"""

import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT   = Path(__file__).resolve().parents[3]
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
JOINED      = Path.home() / "Downloads" / "tmp" / "nfl_sacks_joined_2025_v3.parquet"
GAME_LINES  = Path.home() / "Downloads" / "tmp" / "nfl_game_lines" / "2025"
OUT         = Path.home() / "Downloads" / "tmp" / "nfl_sacks_features_2025.parquet"

TEAM_NAME_MAP = {
    "Arizona Cardinals":     "ARI",
    "Atlanta Falcons":       "ATL",
    "Baltimore Ravens":      "BAL",
    "Buffalo Bills":         "BUF",
    "Carolina Panthers":     "CAR",
    "Chicago Bears":         "CHI",
    "Cincinnati Bengals":    "CIN",
    "Cleveland Browns":      "CLE",
    "Dallas Cowboys":        "DAL",
    "Denver Broncos":        "DEN",
    "Detroit Lions":         "DET",
    "Green Bay Packers":     "GB",
    "Houston Texans":        "HOU",
    "Indianapolis Colts":    "IND",
    "Jacksonville Jaguars":  "JAX",
    "Kansas City Chiefs":    "KC",
    "Las Vegas Raiders":     "LV",
    "Los Angeles Chargers":  "LAC",
    "Los Angeles Rams":      "LA",
    "Miami Dolphins":        "MIA",
    "Minnesota Vikings":     "MIN",
    "New England Patriots":  "NE",
    "New Orleans Saints":    "NO",
    "New York Giants":       "NYG",
    "New York Jets":         "NYJ",
    "Philadelphia Eagles":   "PHI",
    "Pittsburgh Steelers":   "PIT",
    "San Francisco 49ers":   "SF",
    "Seattle Seahawks":      "SEA",
    "Tampa Bay Buccaneers":  "TB",
    "Tennessee Titans":      "TEN",
    "Washington Commanders": "WAS",
}


# ── Config ─────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)["nfl_sacks_model"]


# ── Game lines ─────────────────────────────────────────────────────────────────

def load_game_lines() -> pd.DataFrame:
    """
    Returns one row per (game_id, team) with game_total and team_spread.
    Spread sign convention: negative = team is favored.
    """
    files = glob.glob(str(GAME_LINES / "*.parquet"))
    if not files:
        raise FileNotFoundError(f"No game-line parquets found in {GAME_LINES}")

    raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    rows = []
    for game_id, g in raw.groupby("nfl_game_id"):
        tot = g[g["market"] == "totals"]
        game_total = tot.loc[tot["outcome_name"] == "Over", "point"].median()

        sp = g[g["market"] == "spreads"]
        for outcome_name, sg in sp.groupby("outcome_name"):
            team = TEAM_NAME_MAP.get(outcome_name)
            if team:
                rows.append({
                    "game_id":     game_id,
                    "team":        team,
                    "game_total":  game_total,
                    "team_spread": sg["point"].median(),
                })

    df = pd.DataFrame(rows)
    print(f"  Game lines: {len(df)} team-game rows  ({df['game_id'].nunique()} games)")
    return df


# ── Position group ─────────────────────────────────────────────────────────────

def add_position_group(df: pd.DataFrame, pos_groups: dict, pos_side: dict) -> pd.DataFrame:
    inv_group = {pos.upper(): grp for grp, positions in pos_groups.items() for pos in positions}
    inv_side  = {pos.upper(): side for side, positions in pos_side.items() for pos in positions}
    df["pos_group"] = df["position"].str.upper().map(inv_group).fillna("OTH")
    df["pos_side"]  = df["position"].str.upper().map(inv_side).fillna("other")
    return df


# ── Rolling features ───────────────────────────────────────────────────────────

def add_rolling_features(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """
    Lagged rolling means per player (sorted chronologically).
    window=999 treated as career (all prior games).
    All features use shift(1) to exclude the current game — no look-ahead.
    Grouped by pfr_player_id (not player name) to handle same-name collisions
    e.g. Byron Young (YounBy00/PHI) vs Byron Young (YounBy01/LAR) in 2025.
    """
    df = df.sort_values(["pfr_player_id", "week", "game_id"]).reset_index(drop=True)

    feature_cols = [
        ("sacks",        "sack_rate"),
        ("qb_hits",      "qbhit_rate"),
        ("defense_pct",  "snap_pct"),
    ]

    for src_col, feat_name in feature_cols:
        for w in windows:
            wlabel = "career" if w >= 999 else str(w)
            win    = 10_000 if w >= 999 else w
            df[f"{feat_name}_L{wlabel}"] = (
                df.groupby("pfr_player_id")[src_col]
                .transform(lambda s, _w=win: s.shift(1).rolling(_w, min_periods=1).mean())
            )

    df["games_played_ytd"] = df.groupby("pfr_player_id").cumcount()
    return df


# ── Target ─────────────────────────────────────────────────────────────────────

def add_target(df: pd.DataFrame, drop_pushes: bool) -> pd.DataFrame:
    df["target"] = np.nan
    df.loc[df["sacks"] >= 1.0, "target"] = 1.0
    df.loc[df["sacks"] == 0.0, "target"] = 0.0
    # sacks == 0.5 → target stays NaN (push; dropped from training per config)
    n_push = (df["sacks"] == 0.5).sum()
    if drop_pushes:
        print(f"  Pushes (sacks=0.5): {n_push} rows — target=NaN, excluded from training")
    return df


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    cfg     = load_config()
    windows = cfg["rolling_windows"]

    # ── Load full snap history for accurate rolling features ──────────────────
    print("Loading joined dataset...")
    raw = pd.read_parquet(JOINED)
    played = raw[raw["defense_snaps"] > 0].copy()
    print(f"  {len(played)} player-game rows with snaps > 0  "
          f"({played['player'].nunique()} unique players)")

    # ── Game lines ────────────────────────────────────────────────────────────
    print("Loading game lines...")
    game_lines = load_game_lines()
    played = played.merge(game_lines, on=["game_id", "team"], how="left")
    n_missing_lines = played["game_total"].isna().sum()
    if n_missing_lines:
        print(f"  WARNING: {n_missing_lines} rows missing game lines")

    # ── Position group ────────────────────────────────────────────────────────
    print("Adding position groups...")
    played = add_position_group(played, cfg["position_groups"], cfg["position_side"])
    print(f"  Distribution: {played['pos_group'].value_counts().to_dict()}")

    # ── Rolling features (on full history, then filter to prop rows) ──────────
    print(f"Computing rolling features  (windows={windows})...")
    played = add_rolling_features(played, windows)

    # ── Filter to modelable rows: prop line == 0.5 ────────────────────────────
    print("Filtering to prop rows (line == 0.5)...")
    df = played[
        played["prop_median_price_over"].notna()  # at least one book posted Over at 0.5
    ].copy()
    print(f"  {len(df)} rows")

    # ── Target ────────────────────────────────────────────────────────────────
    df = add_target(df, drop_pushes=cfg["drop_pushes"])

    # ── Column order ──────────────────────────────────────────────────────────
    rolling_cols = [
        f"{feat}_L{('career' if w >= 999 else w)}"
        for feat in ["sack_rate", "qbhit_rate", "snap_pct"]
        for w in windows
    ]
    col_order = [
        "game_id", "week", "player", "pfr_player_id", "position", "pos_group", "pos_side", "team",
        "defense_snaps", "defense_pct",
        "sacks", "qb_hits",
        "game_total", "team_spread",
        # aggregated market (v1/v2)
        "prop_median_line", "prop_median_impl_over", "prop_median_impl_under",
        "prop_mean_impl_over", "prop_mean_impl_under",
        "prop_median_price_over", "prop_median_price_under",
        "prop_min_impl_over", "prop_max_impl_over",
        "prop_min_impl_under", "prop_max_impl_under",
        "prop_best_price_over", "prop_best_price_under",
        "prop_book_spread_over", "prop_book_spread_under",
        "prop_n_books",
        # implied prob bins (v3)
        "prop_median_impl_over_bin", "prop_mean_impl_over_bin",
        "prop_median_impl_under_bin", "prop_mean_impl_under_bin",
        # book-level (v3) — implied only as features; price kept for P&L
        "fanduel_over_0p5_implied",
        "betonline_over_0p5_implied", "betonline_under_0p5_implied",
        "draftkings_over_0p25_implied", "draftkings_under_0p25_implied",
        "fanduel_over_0p5_price",
        "betonline_over_0p5_price", "betonline_under_0p5_price",
        "draftkings_over_0p25_price", "draftkings_under_0p25_price",
        "games_played_ytd",
        *rolling_cols,
        "target",
    ]
    df = df[[c for c in col_order if c in df.columns]]
    df = df.sort_values(["week", "team", "player"]).reset_index(drop=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    df.to_parquet(OUT, index=False)

    n_train = int(df["target"].notna().sum())
    n_push  = int(df["target"].isna().sum())
    n_pos   = int((df["target"] == 1).sum())
    n_neg   = int((df["target"] == 0).sum())

    print(f"\n{'='*55}")
    print(f"  Output : {OUT}")
    print(f"  Rows   : {len(df)}  (all prop rows)")
    print(f"  Train  : {n_train}  (target not NaN)")
    print(f"    Pos  : {n_pos}  (sack, {n_pos/n_train:.1%})")
    print(f"    Neg  : {n_neg}  (no sack, {n_neg/n_train:.1%})")
    print(f"  Pushes : {n_push}  (excluded from training)")
    print(f"  Features ({len(rolling_cols)} rolling): {rolling_cols[:4]} ...")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
