"""
Build feature matrix for NFL sacks props — one script, any season.

Reads from:
  ~/Downloads/tmp/nfl_sacks_historical_spine.parquet  (rolling features, all seasons)
  ~/Downloads/tmp/nfl_defensive_props/{season}/        (props parquets)
  ~/Downloads/tmp/nfl_game_lines/{season}/             (game line parquets)

Outputs:
  ~/Downloads/tmp/nfl_sacks_features_{season}.parquet

Run:
  python src/nfl_sacks_modeling/scripts/build_sacks_features.py --season 2024
  python src/nfl_sacks_modeling/scripts/build_sacks_features.py --season 2025
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
TMP         = Path.home() / "Downloads" / "tmp"
SPINE       = TMP / "nfl_sacks_historical_spine.parquet"

VALIDATION_TARGETS = {
    2024: {
        "Trey Hendrickson": 17.5,
        "Myles Garrett":    14.0,
        "Nik Bonitto":      13.5,
        "Jonathan Greenard": 12.0,
        "Micah Parsons":    12.0,
    },
    2025: {
        "Myles Garrett":    23.0,
        "Brian Burns":      16.5,
        "Danielle Hunter":  15.0,
        "Aidan Hutchinson": 14.5,
        "Nik Bonitto":      14.0,
    },
}

TEAM_NAME_MAP = {
    "Arizona Cardinals":     "ARI", "Atlanta Falcons":       "ATL",
    "Baltimore Ravens":      "BAL", "Buffalo Bills":         "BUF",
    "Carolina Panthers":     "CAR", "Chicago Bears":         "CHI",
    "Cincinnati Bengals":    "CIN", "Cleveland Browns":      "CLE",
    "Dallas Cowboys":        "DAL", "Denver Broncos":        "DEN",
    "Detroit Lions":         "DET", "Green Bay Packers":     "GB",
    "Houston Texans":        "HOU", "Indianapolis Colts":    "IND",
    "Jacksonville Jaguars":  "JAX", "Kansas City Chiefs":    "KC",
    "Las Vegas Raiders":     "LV",  "Los Angeles Chargers":  "LAC",
    "Los Angeles Rams":      "LA",  "Miami Dolphins":        "MIA",
    "Minnesota Vikings":     "MIN", "New England Patriots":  "NE",
    "New Orleans Saints":    "NO",  "New York Giants":       "NYG",
    "New York Jets":         "NYJ", "Philadelphia Eagles":   "PHI",
    "Pittsburgh Steelers":   "PIT", "San Francisco 49ers":   "SF",
    "Seattle Seahawks":      "SEA", "Tampa Bay Buccaneers":  "TB",
    "Tennessee Titans":      "TEN", "Washington Commanders": "WAS",
}

BOOK_CONFIGS = [
    ("fanduel",     "Over",  0.50, "fanduel_over_0p5"),
    ("betonlineag", "Over",  0.50, "betonline_over_0p5"),
    ("betonlineag", "Under", 0.50, "betonline_under_0p5"),
    ("draftkings",  "Over",  0.25, "draftkings_over_0p25"),
    ("draftkings",  "Under", 0.25, "draftkings_under_0p25"),
]

_BIN_EDGES  = list(range(0, 110, 10))
_BIN_LABELS = [f"{i}-{i+10}" for i in range(0, 100, 10)]


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)["nfl_sacks_model"]


def implied_prob(price: float) -> float:
    return abs(price) / (abs(price) + 100.0) if price < 0 else 100.0 / (price + 100.0)


def american_to_decimal(price: float) -> float:
    return 1 + 100 / abs(price) if price < 0 else 1 + price / 100


def decimal_to_american(d: float) -> float:
    return (d - 1) * 100 if d >= 2.0 else -100 / (d - 1)


def _impl_bin(val: float):
    if pd.isna(val):
        return float("nan")
    return pd.cut([val * 100], bins=_BIN_EDGES, labels=_BIN_LABELS,
                  right=True, include_lowest=True).tolist()[0]


# ── Game lines ─────────────────────────────────────────────────────────────────

def load_game_lines(season: int) -> pd.DataFrame:
    files = glob.glob(str(TMP / "nfl_game_lines" / str(season) / "*.parquet"))
    if not files:
        raise FileNotFoundError(f"No game-line parquets found for season {season}")

    raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    rows = []
    for game_id, g in raw.groupby("nfl_game_id"):
        tot        = g[g["market"] == "totals"]
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


# ── Props ──────────────────────────────────────────────────────────────────────

def load_props(season: int) -> pd.DataFrame:
    files = glob.glob(str(TMP / "nfl_defensive_props" / str(season) / "*.parquet"))
    if not files:
        raise FileNotFoundError(f"No props parquets found for season {season}")

    raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    raw["implied_prob"] = raw["price"].apply(implied_prob)

    def agg_player(g):
        over_0_5   = g.loc[(g["outcome_name"] == "Over")  & (g["point"] == 0.5)]
        under_0_5  = g.loc[(g["outcome_name"] == "Under") & (g["point"] == 0.5)]
        over_impl  = over_0_5["implied_prob"]
        under_impl = under_0_5["implied_prob"]
        over_dec   = over_0_5["price"].apply(american_to_decimal)
        under_dec  = under_0_5["price"].apply(american_to_decimal)

        best_price_over  = (over_0_5.loc[over_impl.idxmin(),  "price"]
                            if len(over_impl)  else float("nan"))
        best_price_under = (under_0_5.loc[under_impl.idxmin(), "price"]
                            if len(under_impl) else float("nan"))

        median_over  = over_impl.median()  if len(over_impl)  else float("nan")
        median_under = under_impl.median() if len(under_impl) else float("nan")
        mean_over    = over_impl.mean()    if len(over_impl)  else float("nan")
        mean_under   = under_impl.mean()   if len(under_impl) else float("nan")

        book_data = {}
        for book, side, line, prefix in BOOK_CONFIGS:
            rows = g[(g["bookmaker"] == book) &
                     (g["outcome_name"] == side) &
                     (g["point"] == line)]
            if len(rows):
                row = rows.sort_values("last_update").iloc[-1]
                book_data[f"{prefix}_line"]    = row["point"]
                book_data[f"{prefix}_price"]   = row["price"]
                book_data[f"{prefix}_implied"] = row["implied_prob"]
            else:
                book_data[f"{prefix}_line"]    = float("nan")
                book_data[f"{prefix}_price"]   = float("nan")
                book_data[f"{prefix}_implied"] = float("nan")

        return pd.Series({
            "prop_median_line":           g["point"].median(),
            "prop_median_price_over":     decimal_to_american(over_dec.median())  if len(over_dec)  else float("nan"),
            "prop_median_price_under":    decimal_to_american(under_dec.median()) if len(under_dec) else float("nan"),
            "prop_median_impl_over":      median_over,
            "prop_median_impl_under":     median_under,
            "prop_mean_impl_over":        mean_over,
            "prop_mean_impl_under":       mean_under,
            "prop_min_impl_over":         over_impl.min()  if len(over_impl)  else float("nan"),
            "prop_max_impl_over":         over_impl.max()  if len(over_impl)  else float("nan"),
            "prop_min_impl_under":        under_impl.min() if len(under_impl) else float("nan"),
            "prop_max_impl_under":        under_impl.max() if len(under_impl) else float("nan"),
            "prop_best_price_over":       best_price_over,
            "prop_best_price_under":      best_price_under,
            "prop_book_spread_over":      (over_impl.max() - over_impl.min())  if len(over_impl)  else float("nan"),
            "prop_book_spread_under":     (under_impl.max() - under_impl.min()) if len(under_impl) else float("nan"),
            "prop_n_books":               g["bookmaker"].nunique(),
            "prop_median_impl_over_bin":  _impl_bin(median_over),
            "prop_mean_impl_over_bin":    _impl_bin(mean_over),
            "prop_median_impl_under_bin": _impl_bin(median_under),
            "prop_mean_impl_under_bin":   _impl_bin(mean_under),
            **book_data,
        })

    props = (raw.groupby(["nfl_game_id", "outcome_desc"])
             .apply(agg_player, include_groups=False)
             .reset_index()
             .rename(columns={"outcome_desc": "player_name", "nfl_game_id": "game_id"}))

    print(f"  Props: {len(props)} player-game records  ({props['player_name'].nunique()} unique players)")
    return props


# ── Target ─────────────────────────────────────────────────────────────────────

def add_target(df: pd.DataFrame, drop_pushes: bool) -> pd.DataFrame:
    df["target"] = np.nan
    df.loc[df["sacks"] >= 1.0, "target"] = 1.0
    df.loc[df["sacks"] == 0.0, "target"] = 0.0
    n_push = (df["sacks"] == 0.5).sum()
    if drop_pushes:
        print(f"  Pushes (sacks=0.5): {n_push} rows — target=NaN, excluded from training")
    return df


# ── Validation ─────────────────────────────────────────────────────────────────

def validate(df: pd.DataFrame, season: int):
    targets = VALIDATION_TARGETS.get(season, {})
    if not targets:
        print("  No validation targets defined for this season — skipping")
        return

    totals = df.groupby("player")["sacks"].sum()
    print(f"\n{'='*55}")
    print(f"  VALIDATION — {season} season sack totals")
    print(f"{'='*55}")
    all_passed = True
    for player, expected in targets.items():
        actual = totals.get(player, None)
        if actual is None:
            print(f"  FAIL  {player:<25} not found")
            all_passed = False
        else:
            ok = np.isclose(actual, expected)
            print(f"  {'PASS' if ok else 'FAIL'}  {player:<25} expected={expected}  got={actual}")
            if not ok:
                all_passed = False
    print()
    if not all_passed:
        print("  VALIDATION FAILURES — investigate before trusting the dataset.")
        sys.exit(1)
    print("  All validations passed.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    args   = parser.parse_args()
    season = args.season

    cfg = load_config()
    out = TMP / f"nfl_sacks_features_{season}.parquet"

    print(f"\nBuilding features for {season} season")
    print(f"{'='*55}")

    # ── 1. Spine ──────────────────────────────────────────────────────────────
    print("Loading spine...")
    spine = pd.read_parquet(SPINE)
    df    = spine[spine["season"] == season].copy()
    print(f"  Spine rows for {season}: {len(df):,}  ({df['pfr_player_id'].nunique():,} players)")

    # ── 2. Game lines ─────────────────────────────────────────────────────────
    print("Loading game lines...")
    game_lines = load_game_lines(season)
    df = df.merge(game_lines, on=["game_id", "team"], how="left")
    n_missing = df["game_total"].isna().sum()
    if n_missing:
        print(f"  WARNING: {n_missing} rows missing game lines")

    # ── 3. Props ──────────────────────────────────────────────────────────────
    print("Loading props...")
    props = load_props(season)
    df = df.merge(props, left_on=["game_id", "player"],
                  right_on=["game_id", "player_name"], how="left")
    df = df.drop(columns=["player_name"], errors="ignore")

    # ── 4. Target ─────────────────────────────────────────────────────────────
    df = add_target(df, drop_pushes=cfg["drop_pushes"])

    # ── 5. Validate ───────────────────────────────────────────────────────────
    validate(df, season)

    # ── 6. Filter to prop rows + column order ─────────────────────────────────
    prop_rows = df[df["prop_median_price_over"].notna()].copy()

    windows = cfg["rolling_windows"]
    rolling_cols = [
        f"{feat}_L{('career' if w >= 999 else w)}"
        for feat in ["sack_rate", "qbhit_rate", "snap_pct"]
        for w in windows
    ]
    book_cols = [f"{prefix}_{field}"
                 for _, _, _, prefix in BOOK_CONFIGS
                 for field in ("line", "price", "implied")]

    col_order = [
        "game_id", "week", "player", "pfr_player_id", "position",
        "pos_group", "pos_side", "team",
        "defense_snaps", "defense_pct", "sacks", "qb_hits",
        "game_total", "team_spread",
        "prop_median_line",
        "prop_median_impl_over", "prop_median_impl_under",
        "prop_mean_impl_over",   "prop_mean_impl_under",
        "prop_median_price_over", "prop_median_price_under",
        "prop_min_impl_over", "prop_max_impl_over",
        "prop_min_impl_under", "prop_max_impl_under",
        "prop_best_price_over", "prop_best_price_under",
        "prop_book_spread_over", "prop_book_spread_under",
        "prop_n_books",
        "prop_median_impl_over_bin", "prop_mean_impl_over_bin",
        "prop_median_impl_under_bin", "prop_mean_impl_under_bin",
        *book_cols,
        "games_played_ytd",
        *rolling_cols,
        "target",
    ]
    prop_rows = prop_rows[[c for c in col_order if c in prop_rows.columns]]
    prop_rows = prop_rows.sort_values(["week", "team", "player"]).reset_index(drop=True)

    # ── 7. Save ───────────────────────────────────────────────────────────────
    prop_rows.to_parquet(out, index=False)

    n_train = int(prop_rows["target"].notna().sum())
    n_pos   = int((prop_rows["target"] == 1).sum())
    n_neg   = int((prop_rows["target"] == 0).sum())
    n_push  = int(prop_rows["target"].isna().sum())

    print(f"\n{'='*55}")
    print(f"  Output  : {out}")
    print(f"  Rows    : {len(prop_rows):,}  (all prop rows)")
    print(f"  Train   : {n_train:,}  (target not NaN)")
    print(f"    Pos   : {n_pos:,}  ({n_pos/n_train:.1%})")
    print(f"    Neg   : {n_neg:,}  ({n_neg/n_train:.1%})")
    print(f"  Pushes  : {n_push:,}  (excluded)")
    print(f"  Mkt avg P(Over): {prop_rows['prop_median_impl_over'].mean():.1%}")
    print(f"  Actual Over rate: {(prop_rows['target']==1).mean():.1%}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
