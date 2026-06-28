"""
Build the joined player-sacks dataset for 2025 NFL REG season (v3).

v3 additions over v2:
  Book-level columns (price + implied at each book's canonical line):
    fanduel_over_0p5_price/implied      — FanDuel Over at 0.5 (no Under posted)
    betonline_over_0p5_price/implied    — BetOnline Over at 0.5
    betonline_under_0p5_price/implied   — BetOnline Under at 0.5
    draftkings_over_0p25_price/implied  — DraftKings Over at 0.25 (different line)
    draftkings_under_0p25_price/implied — DraftKings Under at 0.25
  NaN where book did not post that player-game. Multiple snapshots per book
  resolved by taking the most recent (last_update).

  Mean implied prob (across books at 0.5 line):
    prop_mean_impl_over / prop_mean_impl_under

  10%-bucket bin columns for median and mean implied prob (Over and Under):
    prop_median_impl_over_bin / prop_mean_impl_over_bin
    prop_median_impl_under_bin / prop_mean_impl_under_bin
  Bins: "0-10", "10-20", ..., "90-100"

Sources:
  - nfl_data_py PBP       → sacks (full + half), qb_hits per player per game
  - nfl_data_py snap counts → defense_snaps, defense_pct per player per game
  - local sacks props       → median line, implied prob, n_books per player per game

Output:
  ~/Downloads/tmp/nfl_sacks_joined_2025_v3.parquet
  ~/Downloads/tmp/nfl_sacks_joined_2025_v3.csv

Run:
  python src/nfl_sacks_modeling/scripts/build_sacks_dataset_v3.py
"""

import sys
import glob
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

PROPS_DIR   = Path.home() / "Downloads" / "tmp" / "nfl_defensive_props" / "2025"
OUT_DIR     = Path.home() / "Downloads" / "tmp"
OUT_PARQUET = OUT_DIR / "nfl_sacks_joined_2025_v3.parquet"
OUT_CSV     = OUT_DIR / "nfl_sacks_joined_2025_v3.csv"

SEASON = 2025

VALIDATION_TARGETS = {
    "Myles Garrett":     23.0,
    "Brian Burns":       16.5,
    "Danielle Hunter":   15.0,
    "Aidan Hutchinson":  14.5,
    "Nik Bonitto":       14.0,
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def full_to_pbp(name: str) -> str:
    """'Myles Garrett' → 'M.Garrett'  (matches nflfastR abbreviated format)"""
    parts = name.strip().split()
    if len(parts) >= 2:
        return f"{parts[0][0]}.{parts[-1]}"
    return name


def implied_prob(price: float) -> float:
    if price < 0:
        return abs(price) / (abs(price) + 100.0)
    return 100.0 / (price + 100.0)


def american_to_decimal(price: float) -> float:
    return 1 + 100 / abs(price) if price < 0 else 1 + price / 100


def decimal_to_american(d: float) -> float:
    """Convert decimal odds back to American. d must be > 1."""
    if d >= 2.0:
        return (d - 1) * 100
    return -100 / (d - 1)


# ── Step 1: PBP — sacks and qb_hits per player per game ───────────────────────

def build_pbp_stats() -> pd.DataFrame:
    import nfl_data_py as nfl

    print("Loading PBP data...")
    pbp = nfl.import_pbp_data([SEASON], columns=[
        "game_id", "week", "season_type", "defteam",
        "sack", "sack_player_name",
        "half_sack_1_player_name", "half_sack_2_player_name",
        "lateral_sack_player_name",
        "qb_hit", "qb_hit_1_player_name", "qb_hit_2_player_name",
    ])
    reg = pbp[pbp["season_type"] == "REG"].copy()

    sack_rows = []

    # Full sacks
    full_sacks = reg[reg["sack"] == 1][["game_id", "week", "defteam",
                                         "sack_player_name",
                                         "half_sack_1_player_name",
                                         "half_sack_2_player_name",
                                         "lateral_sack_player_name"]]
    for _, r in full_sacks.iterrows():
        if pd.notna(r["half_sack_1_player_name"]):
            sack_rows.append({"game_id": r["game_id"], "week": r["week"],
                               "defteam": r["defteam"],
                               "pbp_name": r["half_sack_1_player_name"], "sacks": 0.5})
            if pd.notna(r["half_sack_2_player_name"]):
                sack_rows.append({"game_id": r["game_id"], "week": r["week"],
                                   "defteam": r["defteam"],
                                   "pbp_name": r["half_sack_2_player_name"], "sacks": 0.5})
        elif pd.notna(r["sack_player_name"]):
            sack_rows.append({"game_id": r["game_id"], "week": r["week"],
                               "defteam": r["defteam"],
                               "pbp_name": r["sack_player_name"], "sacks": 1.0})
            # lateral sack gets credited separately if present
            if pd.notna(r["lateral_sack_player_name"]):
                sack_rows.append({"game_id": r["game_id"], "week": r["week"],
                                   "defteam": r["defteam"],
                                   "pbp_name": r["lateral_sack_player_name"], "sacks": 1.0})

    sacks_df = (pd.DataFrame(sack_rows)
                .groupby(["game_id", "week", "defteam", "pbp_name"], as_index=False)["sacks"]
                .sum())

    # QB hits (player-attributed)
    hit_rows = []
    hits = reg[reg["qb_hit"] == 1][["game_id", "week", "defteam",
                                     "qb_hit_1_player_name", "qb_hit_2_player_name"]]
    for _, r in hits.iterrows():
        for col in ["qb_hit_1_player_name", "qb_hit_2_player_name"]:
            if pd.notna(r[col]):
                hit_rows.append({"game_id": r["game_id"], "week": r["week"],
                                  "defteam": r["defteam"], "pbp_name": r[col]})

    hits_df = (pd.DataFrame(hit_rows)
               .groupby(["game_id", "week", "defteam", "pbp_name"], as_index=False)
               .size()
               .rename(columns={"size": "qb_hits"}))

    pbp_stats = sacks_df.merge(hits_df, on=["game_id", "week", "defteam", "pbp_name"], how="outer")
    pbp_stats["sacks"]   = pbp_stats["sacks"].fillna(0)
    pbp_stats["qb_hits"] = pbp_stats["qb_hits"].fillna(0).astype(int)

    print(f"  PBP: {len(pbp_stats)} player-game records  "
          f"({pbp_stats['sacks'].sum():.1f} total sacks, "
          f"{pbp_stats['qb_hits'].sum()} total qb_hits)")
    return pbp_stats


# ── Step 2: Snap counts ────────────────────────────────────────────────────────

def build_snap_counts() -> pd.DataFrame:
    import nfl_data_py as nfl

    print("Loading snap counts...")
    snaps = nfl.import_snap_counts([SEASON])
    reg   = snaps[snaps["game_type"] == "REG"].copy()

    # Keep defensive players only (defense_snaps > 0)
    def_snaps = reg[reg["defense_snaps"] > 0][
        ["game_id", "week", "player", "pfr_player_id", "position",
         "team", "defense_snaps", "defense_pct"]
    ].copy()

    # Create pbp_name for joining to PBP abbreviated names
    def_snaps["pbp_name"] = def_snaps["player"].apply(full_to_pbp)

    print(f"  Snap counts: {len(def_snaps)} player-game records")
    return def_snaps


# ── Step 3: Props ──────────────────────────────────────────────────────────────

# (bookmaker key in raw data, outcome_name, canonical line, column prefix)
BOOK_CONFIGS = [
    ("fanduel",     "Over",  0.50, "fanduel_over_0p5"),
    ("betonlineag", "Over",  0.50, "betonline_over_0p5"),
    ("betonlineag", "Under", 0.50, "betonline_under_0p5"),
    ("draftkings",  "Over",  0.25, "draftkings_over_0p25"),
    ("draftkings",  "Under", 0.25, "draftkings_under_0p25"),
]

_BIN_EDGES  = list(range(0, 110, 10))
_BIN_LABELS = [f"{i}-{i+10}" for i in range(0, 100, 10)]


def _impl_bin(val: float):
    """Map an implied probability (0–1) to a 10% bucket label e.g. '30-40'."""
    if pd.isna(val):
        return float("nan")
    return pd.cut([val * 100], bins=_BIN_EDGES, labels=_BIN_LABELS,
                  right=True, include_lowest=True).tolist()[0]


def build_props() -> pd.DataFrame:
    print("Loading props parquets...")
    files = glob.glob(str(PROPS_DIR / "*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquets found in {PROPS_DIR}")

    frames = [pd.read_parquet(f) for f in files]
    raw    = pd.concat(frames, ignore_index=True)

    raw["implied_prob"] = raw["price"].apply(implied_prob)

    def agg_player(g):
        # ── base: 0.5-line Over/Under for cross-book aggregates ───────────────
        over_0_5  = g.loc[(g["outcome_name"] == "Over")  & (g["point"] == 0.5)]
        under_0_5 = g.loc[(g["outcome_name"] == "Under") & (g["point"] == 0.5)]

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

        # ── v3: book-level (line / American price / implied) ──────────────────
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
            # ── v1/v2 fields (kept for backward compat) ───────────────────────
            "prop_median_line":             g["point"].median(),
            "prop_min_line":                g["point"].min(),
            "prop_max_line":                g["point"].max(),
            "prop_median_price_over":       decimal_to_american(over_dec.median())  if len(over_dec)  else float("nan"),
            "prop_median_price_under":      decimal_to_american(under_dec.median()) if len(under_dec) else float("nan"),
            "prop_median_impl_over":        median_over,
            "prop_median_impl_under":       median_under,
            "prop_min_impl_over":           over_impl.min()  if len(over_impl)  else float("nan"),
            "prop_max_impl_over":           over_impl.max()  if len(over_impl)  else float("nan"),
            "prop_min_impl_under":          under_impl.min() if len(under_impl) else float("nan"),
            "prop_max_impl_under":          under_impl.max() if len(under_impl) else float("nan"),
            "prop_best_price_over":         best_price_over,
            "prop_best_price_under":        best_price_under,
            "prop_book_spread_over":        (over_impl.max() - over_impl.min())  if len(over_impl)  else float("nan"),
            "prop_book_spread_under":       (under_impl.max() - under_impl.min()) if len(under_impl) else float("nan"),
            "prop_n_books":                 g["bookmaker"].nunique(),
            "prop_books":                   ",".join(sorted(g["bookmaker"].unique())),
            # ── v3 new ────────────────────────────────────────────────────────
            "prop_mean_impl_over":          mean_over,
            "prop_mean_impl_under":         mean_under,
            "prop_median_impl_over_bin":    _impl_bin(median_over),
            "prop_mean_impl_over_bin":      _impl_bin(mean_over),
            "prop_median_impl_under_bin":   _impl_bin(median_under),
            "prop_mean_impl_under_bin":     _impl_bin(mean_under),
            **book_data,
        })

    props = (raw.groupby(["nfl_game_id", "outcome_desc"])
             .apply(agg_player, include_groups=False)
             .reset_index()
             .rename(columns={"outcome_desc": "player_name", "nfl_game_id": "game_id"}))

    print(f"  Props: {len(props)} player-game records  "
          f"({props['player_name'].nunique()} unique players)")
    return props


# ── Step 4: Join ───────────────────────────────────────────────────────────────

def build_joined(pbp_stats, snap_counts, props) -> pd.DataFrame:
    print("Joining...")

    # Snap counts as the spine (all defensive players with snaps)
    # Join PBP on (game_id, pbp_name, team=defteam)
    joined = snap_counts.merge(
        pbp_stats,
        on=["game_id", "week", "pbp_name"],
        how="left",
        suffixes=("", "_pbp"),
    )
    # defteam from PBP should match team from snap counts — drop duplicate
    if "defteam" in joined.columns:
        joined = joined.drop(columns=["defteam"])

    joined["sacks"]   = joined["sacks"].fillna(0)
    joined["qb_hits"] = joined["qb_hits"].fillna(0).astype(int)

    # Join props on (game_id, player full name)
    joined = joined.merge(
        props,
        left_on=["game_id", "player"],
        right_on=["game_id", "player_name"],
        how="left",
    )
    joined = joined.drop(columns=["player_name"], errors="ignore")

    # Outcome: did they go Over their line?
    joined["hit_over"] = joined.apply(
        lambda r: (r["sacks"] > r["prop_median_line"])
                  if pd.notna(r.get("prop_median_line")) else None,
        axis=1,
    )

    # book-level columns derived from BOOK_CONFIGS
    book_cols = [f"{prefix}_{field}"
                 for _, _, _, prefix in BOOK_CONFIGS
                 for field in ("line", "price", "implied")]

    col_order = [
        "game_id", "week", "player", "pfr_player_id", "position", "team",
        "defense_snaps", "defense_pct",
        "sacks", "qb_hits",
        # aggregated (v1/v2)
        "prop_median_line", "prop_min_line", "prop_max_line",
        "prop_median_price_over", "prop_median_price_under",
        "prop_median_impl_over", "prop_median_impl_under",
        "prop_mean_impl_over", "prop_mean_impl_under",
        "prop_min_impl_over", "prop_max_impl_over",
        "prop_min_impl_under", "prop_max_impl_under",
        "prop_best_price_over", "prop_best_price_under",
        "prop_book_spread_over", "prop_book_spread_under",
        "prop_n_books", "prop_books",
        # bins (v3)
        "prop_median_impl_over_bin", "prop_mean_impl_over_bin",
        "prop_median_impl_under_bin", "prop_mean_impl_under_bin",
        # book-level (v3)
        *book_cols,
        "hit_over",
    ]
    joined = joined[[c for c in col_order if c in joined.columns]]
    joined = joined.sort_values(["week", "team", "player"]).reset_index(drop=True)

    print(f"  Joined: {len(joined)} player-game records")
    return joined


# ── Step 5: Validate season totals ────────────────────────────────────────────

def validate(df: pd.DataFrame):
    print(f"\n{'='*60}")
    print("  VALIDATION — season sack totals")
    print(f"{'='*60}")

    season_totals = (df.groupby("player")["sacks"]
                     .sum()
                     .reset_index()
                     .rename(columns={"sacks": "season_sacks"}))

    all_passed = True
    for player, expected in VALIDATION_TARGETS.items():
        row = season_totals[season_totals["player"] == player]
        if row.empty:
            print(f"  FAIL  {player:<25} not found in dataset")
            all_passed = False
        else:
            actual = row["season_sacks"].iloc[0]
            status = "PASS" if actual == expected else "FAIL"
            if status == "FAIL":
                all_passed = False
            print(f"  {status}  {player:<25} expected={expected}  got={actual}")

    print()
    if all_passed:
        print("  All validations passed.")
    else:
        print("  VALIDATION FAILURES — investigate before trusting the dataset.")
        sys.exit(1)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    pbp_stats   = build_pbp_stats()
    snap_counts = build_snap_counts()
    props       = build_props()
    joined      = build_joined(pbp_stats, snap_counts, props)

    validate(joined)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(OUT_PARQUET, index=False)
    joined.to_csv(OUT_CSV, index=False)

    print(f"\n  → {OUT_PARQUET}")
    print(f"  → {OUT_CSV}")

    # Quick coverage summary
    has_prop = joined["prop_median_line"].notna()
    print(f"\n  Player-game rows      : {len(joined)}")
    print(f"  With prop line        : {has_prop.sum()}  ({has_prop.mean()*100:.1f}%)")
    print(f"  Without prop line     : {(~has_prop).sum()}  (no line posted)")


if __name__ == "__main__":
    main()
