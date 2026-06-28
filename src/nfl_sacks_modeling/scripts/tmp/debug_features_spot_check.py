"""
Adhoc spot-check of nfl_sacks_features_2025.parquet.
Prints all rows for Myles Garrett + Packers sack leader, week by week.
"""

import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: f"{x:.3f}")

FEATURES = "/Users/thomasmyles/Downloads/tmp/nfl_sacks_features_2025.parquet"

df = pd.read_parquet(FEATURES)

ROLLING_COLS = [c for c in df.columns if c.startswith(("sack_rate_", "qbhit_rate_", "snap_pct_"))]

DISPLAY_COLS = [
    "week", "game_id", "pos_group",
    "defense_snaps", "defense_pct",
    "sacks", "qb_hits",
    "game_total", "team_spread",
    "prop_median_line", "prop_median_impl_over", "prop_median_impl_under",
    "prop_n_books",
    "games_played_ytd",
    *ROLLING_COLS,
    "target",
]
DISPLAY_COLS = [c for c in DISPLAY_COLS if c in df.columns]


def show_player(name: str, label: str = None):
    rows = df[df["player"] == name].sort_values("week").reset_index(drop=True)
    header = label or name
    print(f"\n{'='*100}")
    print(f"  {header}  ({name})  —  {len(rows)} rows in features parquet")
    print(f"{'='*100}")
    if rows.empty:
        print("  NOT FOUND")
        return
    print(rows[DISPLAY_COLS].to_string(index=True))
    print(f"\n  Season totals  →  sacks={rows['sacks'].sum():.1f}  qb_hits={rows['qb_hits'].sum():.0f}")
    print(f"  Target distribution  →  1={int((rows['target']==1).sum())}  0={int((rows['target']==0).sum())}  NaN={rows['target'].isna().sum()}")


# ── 1. Myles Garrett ──────────────────────────────────────────────────────────
show_player("Myles Garrett")


# ── 2. Packers sack leader ────────────────────────────────────────────────────
gb_players = df[df["team"] == "GB"]
gb_sacks = gb_players.groupby("player")["sacks"].sum().sort_values(ascending=False)
print(f"\n\nPackers sack leaders (all rows in features parquet):")
print(gb_sacks.head(5).to_string())

gb_leader = gb_sacks.index[0]
show_player(gb_leader, label=f"GB sack leader")
