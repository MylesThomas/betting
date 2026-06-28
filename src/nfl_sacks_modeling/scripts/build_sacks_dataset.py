"""
Build the joined player-sacks dataset for 2025 NFL REG season.

Sources:
  - nfl_data_py PBP       → sacks (full + half), qb_hits per player per game
  - nfl_data_py snap counts → defense_snaps, defense_pct per player per game
  - local sacks props       → median line, implied prob, n_books per player per game

Output:
  ~/Downloads/tmp/nfl_sacks_joined_2025.parquet
  ~/Downloads/tmp/nfl_sacks_joined_2025.csv

Validation (hard assertions):
  Myles Garrett   23.0 sacks
  Brian Burns     16.5 sacks
  Danielle Hunter 15.0 sacks
  Aidan Hutchinson 14.5 sacks
  Nik Bonitto     14.0 sacks

Run:
  python nfl_sacks_modeling/scripts/build_sacks_dataset.py
"""

import sys
import glob
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

PROPS_DIR   = Path.home() / "Downloads" / "tmp" / "nfl_defensive_props" / "2025"
OUT_DIR     = Path.home() / "Downloads" / "tmp"
OUT_PARQUET = OUT_DIR / "nfl_sacks_joined_2025.parquet"
OUT_CSV     = OUT_DIR / "nfl_sacks_joined_2025.csv"

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

def build_props() -> pd.DataFrame:
    print("Loading props parquets...")
    files = glob.glob(str(PROPS_DIR / "*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquets found in {PROPS_DIR}")

    frames = [pd.read_parquet(f) for f in files]
    raw    = pd.concat(frames, ignore_index=True)

    raw["implied_prob"] = raw["price"].apply(implied_prob)

    props = (raw.groupby(["nfl_game_id", "outcome_desc"])
             .apply(lambda g: pd.Series({
                 "prop_median_line":       g["point"].median(),
                 "prop_min_line":          g["point"].min(),
                 "prop_max_line":          g["point"].max(),
                 # Prices and implied probs filtered to the canonical (0.5) line only
                 # so alt-line rows (1.5, 2.5, etc.) don't pollute the medians
                 # Median in decimal-odds space to avoid near-zero blowup when books straddle American zero
                 "prop_median_price_over": (lambda s: decimal_to_american(s.median()) if len(s) else float("nan"))(
                     g.loc[(g["outcome_name"] == "Over")  & (g["point"] == 0.5), "price"].apply(american_to_decimal)),
                 "prop_median_price_under":(lambda s: decimal_to_american(s.median()) if len(s) else float("nan"))(
                     g.loc[(g["outcome_name"] == "Under") & (g["point"] == 0.5), "price"].apply(american_to_decimal)),
                 "prop_median_impl_over":  g.loc[(g["outcome_name"] == "Over")  & (g["point"] == 0.5), "implied_prob"].median(),
                 "prop_median_impl_under": g.loc[(g["outcome_name"] == "Under") & (g["point"] == 0.5), "implied_prob"].median(),
                 "prop_n_books":           g["bookmaker"].nunique(),
                 "prop_books":             ",".join(sorted(g["bookmaker"].unique())),
             }), include_groups=False)
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

    col_order = [
        "game_id", "week", "player", "pfr_player_id", "position", "team",
        "defense_snaps", "defense_pct",
        "sacks", "qb_hits",
        "prop_median_line", "prop_min_line", "prop_max_line",
        "prop_median_price_over", "prop_median_price_under",
        "prop_median_impl_over", "prop_median_impl_under",
        "prop_n_books", "prop_books",
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
