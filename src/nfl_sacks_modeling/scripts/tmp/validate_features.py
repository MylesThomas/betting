"""
Validation checks on nfl_sacks_features_2025.parquet before modeling.
Each check prints PASS / FAIL with detail.
"""

import sys
import numpy as np
import pandas as pd

FEATURES = "/Users/thomasmyles/Downloads/tmp/nfl_sacks_features_2025.parquet"
JOINED   = "/Users/thomasmyles/Downloads/tmp/nfl_sacks_joined_2025.parquet"

df     = pd.read_parquet(FEATURES)
joined = pd.read_parquet(JOINED)
# Full season history per player (all games with snaps, sorted)
history = (joined[joined["defense_snaps"] > 0]
           .sort_values(["player", "week", "game_id"])
           .reset_index(drop=True))
train = df[df["target"].notna()].copy()

ROLLING_RATE_COLS = [c for c in df.columns if c.startswith("sack_rate_")]
ALL_ROLLING = [c for c in df.columns if any(c.startswith(p) for p in
               ["sack_rate_", "qbhit_rate_", "snap_pct_"])]

failures = []

def check(name: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    line = f"  [{status}]  {name}"
    if detail:
        line += f"  —  {detail}"
    print(line)
    if not passed:
        failures.append(name)


print(f"\n{'='*65}")
print(f"  Validating {FEATURES}")
print(f"  Rows: {len(df)}  |  Train rows: {len(train)}")
print(f"{'='*65}\n")


# ── 1. Row counts ─────────────────────────────────────────────────────────────
print("── Row counts ──")

check("Total rows > 1800", len(df) > 1800, f"n={len(df)}")

max_games = df.groupby("player")["game_id"].nunique().max()
check("No player has > 18 rows", max_games <= 18, f"max={max_games}")

dups = df.duplicated(subset=["player", "game_id"]).sum()
check("No duplicate (player, game_id)", dups == 0, f"dupes={dups}")


# ── 2. Target consistency ─────────────────────────────────────────────────────
print("\n── Target ──")

wrong_1 = ((df["sacks"] >= 1.0) & (df["target"] != 1.0)).sum()
wrong_0 = ((df["sacks"] == 0.0) & (df["target"] != 0.0)).sum()
wrong_nan = ((df["sacks"] == 0.5) & df["target"].notna()).sum()
check("target==1 iff sacks>=1.0", wrong_1 == 0, f"{wrong_1} wrong")
check("target==0 iff sacks==0.0", wrong_0 == 0, f"{wrong_0} wrong")
check("target==NaN iff sacks==0.5 (push)", wrong_nan == 0, f"{wrong_nan} wrong")

n_pos = (train["target"] == 1).sum()
n_neg = (train["target"] == 0).sum()
check("Positive rate 15–35%", 0.15 <= n_pos / len(train) <= 0.35,
      f"{n_pos/len(train):.1%}  ({n_pos} pos / {n_neg} neg)")


# ── 3. No data leakage ────────────────────────────────────────────────────────
print("\n── No look-ahead / leakage ──")

# Rolling features use full season history (including non-prop games).
# So "first prop row" may already have history — check against joined parquet instead.

# games_played_ytd must equal count of prior games in FULL history for that player
ytd_wrong = 0
for pid, grp in df.groupby("pfr_player_id"):
    for _, row in grp.iterrows():
        prior_full = history[
            (history["pfr_player_id"] == pid) &
            (history["week"] < row["week"])
        ]
        if row["games_played_ytd"] != len(prior_full):
            ytd_wrong += 1
check("games_played_ytd matches full-season prior game count", ytd_wrong == 0,
      f"{ytd_wrong} mismatches")

# sack_rate_L1 == sacks from the immediately prior game in full history
l1_wrong = 0
for pid, grp in df.groupby("pfr_player_id"):
    phist = history[history["pfr_player_id"] == pid].sort_values("week").reset_index(drop=True)
    for _, row in grp.iterrows():
        prior = phist[phist["week"] < row["week"]]
        if prior.empty:
            if not pd.isna(row["sack_rate_L1"]):
                l1_wrong += 1
        else:
            expected = prior.iloc[-1]["sacks"]
            if not pd.isna(row["sack_rate_L1"]) and not np.isclose(row["sack_rate_L1"], expected, atol=1e-6):
                l1_wrong += 1
check("sack_rate_L1 == immediately prior game sacks (full history)", l1_wrong == 0,
      f"{l1_wrong} mismatches")

# Players with no prior games in full history must have all-NaN rolling features
no_prior = df[df["games_played_ytd"] == 0]
nan_ok = no_prior[ALL_ROLLING].isna().all(axis=1).all()
check("Rolling features all NaN when games_played_ytd == 0", nan_ok)


# ── 4. Game lines coverage ────────────────────────────────────────────────────
print("\n── Game lines ──")

missing_total  = df["game_total"].isna().sum()
missing_spread = df["team_spread"].isna().sum()
check("game_total non-NaN for all rows", missing_total == 0, f"{missing_total} missing")
check("team_spread non-NaN for all rows", missing_spread == 0, f"{missing_spread} missing")

check("game_total in plausible range (30–60)",
      df["game_total"].dropna().between(30, 60).all(),
      f"min={df['game_total'].min():.1f}  max={df['game_total'].max():.1f}")
check("team_spread in plausible range (-25 to 25)",
      df["team_spread"].dropna().between(-25, 25).all(),
      f"min={df['team_spread'].min():.1f}  max={df['team_spread'].max():.1f}")


# ── 5. Prop price sanity ──────────────────────────────────────────────────────
print("\n── Prop prices ──")

def units_on_win(p):
    if pd.isna(p): return np.nan
    return 100 / abs(p) if p < 0 else p / 100

df["_owu"] = df["prop_median_price_over"].apply(units_on_win)
df["_uwu"] = df["prop_median_price_under"].apply(units_on_win)

bad_over  = (df["_owu"] > 10).sum()
bad_under = (df["_uwu"] > 5).sum()
check("No Over win payout > 10 units", bad_over == 0,
      f"{bad_over} rows (likely bad price)")
check("No Under win payout > 5 units", bad_under == 0,
      f"{bad_under} rows (likely bad price)")

df.drop(columns=["_owu", "_uwu"], inplace=True)


# ── 6. Position group sanity ──────────────────────────────────────────────────
print("\n── Position groups ──")

known = {
    "Myles Garrett":   ("DL", "outside"),
    "Micah Parsons":   ("DL", "outside"),
    "Danielle Hunter": ("DL", "outside"),
    "Brian Burns":     ("LB", "outside"),
    "T.J. Watt":       ("LB", "outside"),
}
for player, (exp_grp, exp_side) in known.items():
    rows = df[df["player"] == player]
    if rows.empty:
        continue
    grp  = rows["pos_group"].iloc[0]
    side = rows["pos_side"].iloc[0]
    check(f"{player} pos_group={exp_grp} pos_side={exp_side}",
          grp == exp_grp and side == exp_side,
          f"got pos_group={grp} pos_side={side}")


# ── 7. Player spot-checks ─────────────────────────────────────────────────────
print("\n── Player spot-checks (Myles Garrett) ──")

mg = df[df["player"] == "Myles Garrett"].sort_values("week").reset_index(drop=True)
check("Garrett has 17 rows", len(mg) == 17, f"got {len(mg)}")
check("Garrett season sacks == 23.0", mg["sacks"].sum() == 23.0,
      f"got {mg['sacks'].sum()}")
check("Garrett wk1 sack_rate_L1 is NaN", pd.isna(mg.loc[0, "sack_rate_L1"]))
check("Garrett wk2 sack_rate_L1 == 2.0",
      np.isclose(mg.loc[1, "sack_rate_L1"], 2.0),
      f"got {mg.loc[1, 'sack_rate_L1']}")
check("Garrett wk3 sack_rate_L3 == mean(2.0, 1.5) == 1.75",
      np.isclose(mg.loc[2, "sack_rate_L3"], 1.75),
      f"got {mg.loc[2, 'sack_rate_L3']}")
check("Garrett all game_total non-NaN", mg["game_total"].notna().all())
check("Garrett all team_spread non-NaN", mg["team_spread"].notna().all())


# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
if failures:
    print(f"  FAILED: {len(failures)} check(s)")
    for f in failures:
        print(f"    - {f}")
    sys.exit(1)
else:
    print(f"  All checks passed. Data is ready for modeling.")
print(f"{'='*65}\n")
