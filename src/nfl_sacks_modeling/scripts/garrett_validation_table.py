"""
Myles Garrett 2025 season — 17-game feature validation table.

Shows all features that will go into the model:
  - Actual sacks, qb_hits, defense_pct
  - Prop line + implied probs
  - Game total + team spread (CLE perspective, from -30min pre-kick)
  - Rolling sack/qb_hit/snap% rates (windows: 1/3/5/8/16/999, lagged = no look-ahead)
  - Position group
  - Outcome flags (is_over, is_under, is_push)

Output: ~/Downloads/tmp/garrett_validation_2025.html

Run:
  python nfl_sacks_modeling/scripts/garrett_validation_table.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

JOINED      = Path.home() / "Downloads" / "tmp" / "nfl_sacks_joined_2025.parquet"
GAME_LINES  = Path.home() / "Downloads" / "tmp" / "cle_game_lines_2025.parquet"
OUT_HTML    = Path.home() / "Downloads" / "tmp" / "garrett_validation_2025.html"

WINDOWS = [1, 3, 5, 8, 16, 999]


# ── Position group ─────────────────────────────────────────────────────────────

POS_GROUP = {
    "DE": "DL", "DT": "DL", "NT": "DL", "DE/DT": "DL",
    "OLB": "LB", "ILB": "LB", "MLB": "LB", "LB": "LB",
    "CB": "DB", "SS": "DB", "FS": "DB", "DB": "DB", "S": "DB",
}

def pos_group(pos: str) -> str:
    return POS_GROUP.get(str(pos).upper(), "OTH")


# ── Rolling helpers (lagged — excludes current row) ───────────────────────────

def lagged_rolling(series: pd.Series, window: int) -> pd.Series:
    """Mean of previous `window` games. window=999 = all prior games."""
    w = len(series) if window >= 999 else window
    return series.shift(1).rolling(w, min_periods=1).mean()


def lagged_rolling_std(series: pd.Series, window: int) -> pd.Series:
    w = len(series) if window >= 999 else window
    return series.shift(1).rolling(w, min_periods=2).std()


# ── Game lines: aggregate to one row per game ──────────────────────────────────

def load_game_lines() -> pd.DataFrame:
    raw = pd.read_parquet(GAME_LINES)

    rows = []
    for game_id, g in raw.groupby("nfl_game_id"):
        home = g["home_team"].iloc[0]
        away = g["away_team"].iloc[0]
        cle_is_home = home == "CLE"

        # Game total: median Over point across books
        tot = g[g["market"] == "totals"]
        game_total = tot.loc[tot["outcome_name"] == "Over", "point"].median()

        # Spread: CLE perspective (outcome_name is full team name e.g. "Cleveland Browns")
        sp = g[g["market"] == "spreads"]
        cle_spread = sp.loc[sp["outcome_name"].str.contains("Cleveland", case=False, na=False), "point"].median()

        rows.append({
            "nfl_game_id": game_id,
            "game_total":  game_total,
            "cle_spread":  cle_spread,
            "cle_is_home": cle_is_home,
        })

    return pd.DataFrame(rows)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    joined = pd.read_parquet(JOINED)
    game_lines = load_game_lines()

    # Garrett rows only, sorted chronologically
    mg = joined[joined["player"] == "Myles Garrett"].copy()
    mg = mg.sort_values(["week", "game_id"]).reset_index(drop=True)
    assert len(mg) == 17, f"Expected 17 games, got {len(mg)}"

    # Join game lines
    mg = mg.merge(game_lines, left_on="game_id", right_on="nfl_game_id", how="left")

    # Position group
    mg["pos_group"] = mg["position"].apply(pos_group)

    # Outcome flags
    mg["is_over"]  = (mg["sacks"] >= 1.0).astype(int)
    mg["is_under"] = (mg["sacks"] == 0.0).astype(int)
    mg["is_push"]  = (mg["sacks"] == 0.5).astype(int)

    # Rolling features (lagged)
    for w in WINDOWS:
        label = "career" if w >= 999 else str(w)
        mg[f"sack_rate_L{label}"]   = lagged_rolling(mg["sacks"],         w)
        mg[f"qbhit_rate_L{label}"]  = lagged_rolling(mg["qb_hits"],       w)
        mg[f"snap_pct_L{label}"]    = lagged_rolling(mg["defense_pct"],   w)

    mg[f"sack_std_Lcareer"] = lagged_rolling_std(mg["sacks"], 999)
    mg["games_played_ytd"]  = mg.index  # 0-indexed = prior games played

    # ── HTML ──────────────────────────────────────────────────────────────────

    def flt(v, dec=2):
        return "—" if pd.isna(v) else f"{v:.{dec}f}"

    def pct(v):
        return "—" if pd.isna(v) else f"{v:.1%}"

    def odds_fmt(v):
        if pd.isna(v):
            return "—"
        return f"+{int(v)}" if v >= 0 else str(int(v))

    col_groups = [
        ("Game",         ["week", "game_id", "cle_is_home", "pos_group"]),
        ("Outcome",      ["sacks", "qb_hits", "defense_pct", "is_over", "is_under", "is_push"]),
        ("Prop",         ["prop_median_line", "prop_median_impl_over", "prop_median_impl_under",
                          "prop_median_price_over", "prop_median_price_under", "prop_n_books"]),
        ("Game Lines",   ["game_total", "cle_spread"]),
        ("Rolling Sacks",  [f"sack_rate_L{('career' if w>=999 else w)}" for w in WINDOWS]),
        ("Rolling QB Hits",[f"qbhit_rate_L{('career' if w>=999 else w)}" for w in WINDOWS]),
        ("Rolling Snap%",  [f"snap_pct_L{('career' if w>=999 else w)}" for w in WINDOWS]),
        ("Other",        ["games_played_ytd", "sack_std_Lcareer"]),
    ]

    def fmt_val(col, val):
        if col in ("prop_median_price_over", "prop_median_price_under"):
            return odds_fmt(val)
        if col in ("prop_median_impl_over", "prop_median_impl_under", "defense_pct"):
            return pct(val)
        if col.startswith("snap_pct"):
            return pct(val)
        if col in ("is_over", "is_under", "is_push"):
            return str(int(val)) if not pd.isna(val) else "—"
        if col == "cle_is_home":
            return "H" if val else "A"
        if isinstance(val, float):
            return flt(val)
        return str(val) if not pd.isna(val) else "—"

    # Build header with group spans
    header_top = ""
    header_bot = ""
    all_cols = []
    for grp_name, cols in col_groups:
        existing = [c for c in cols if c in mg.columns]
        if not existing:
            continue
        header_top += f'<th colspan="{len(existing)}" style="background:#444;color:white;border-right:2px solid #888;">{grp_name}</th>'
        for c in existing:
            header_bot += f'<th style="white-space:nowrap">{c}</th>'
        all_cols.extend(existing)

    rows_html = ""
    for _, row in mg.iterrows():
        wk = int(row["week"])
        is_over = int(row["is_over"])
        bg = "background:rgba(44,160,44,0.12)" if is_over else "background:rgba(214,39,40,0.08)"
        cells = "".join(f"<td>{fmt_val(c, row[c])}</td>" for c in all_cols)
        rows_html += f'<tr style="{bg}">{cells}</tr>\n'

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Garrett 2025 Validation</title>
  <style>
    body  {{ font-family: sans-serif; max-width: 100%; margin: 30px 40px; }}
    h1    {{ border-bottom: 2px solid #333; padding-bottom: 8px; }}
    table {{ border-collapse: collapse; font-family: monospace; font-size: 12px; white-space: nowrap; }}
    th,td {{ padding: 5px 10px; text-align: right; border-bottom: 1px solid #ddd; }}
    th    {{ background: #222; color: white; }}
    td:first-child, th:first-child {{ text-align: left; }}
    tr:hover {{ filter: brightness(0.95); }}
  </style>
</head>
<body>
  <h1>Myles Garrett — 2025 Season Feature Validation (17 games)</h1>
  <p style="font-family:monospace;font-size:12px;color:#555;">
    Rolling stats are lagged (exclude current game).
    Snap% rolling uses defense_pct (player-level).
    Spread = CLE perspective (negative = CLE favored).
    Green rows = sack (Over hit), red = no sack.
  </p>
  <div style="overflow-x:auto;">
  <table>
    <thead>
      <tr>{header_top}</tr>
      <tr style="background:#333;">{header_bot}</tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>
  </div>
</body>
</html>"""

    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Saved → {OUT_HTML}")
    print(f"\nGarrett 2025 summary:")
    print(f"  Games:    {len(mg)}")
    print(f"  Sacks:    {mg['sacks'].sum():.1f}")
    print(f"  Sack%:    {mg['is_over'].mean():.1%}")
    print(f"  Avg snap%:{mg['defense_pct'].mean():.1%}")
    print(f"  Game total range: {mg['game_total'].min():.1f} – {mg['game_total'].max():.1f}")
    print(f"  CLE spread range: {mg['cle_spread'].min():.1f} – {mg['cle_spread'].max():.1f}")


if __name__ == "__main__":
    main()
