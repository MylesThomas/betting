"""
Step 5 — Grid Search over Betting Strategy

For each combination of (bet_side, min_edge, shrinkage, odds_type, min_books):
  1. Shrink p_model toward 0.5:  p_shrunk = p * (1 - shrink) + 0.5 * shrink
  2. Compute edge:               edge = p_shrunk - novig_prob_over (for over bets)
  3. Filter by edge >= min_edge and any odds constraints
  4. Compute per-bet P/L using BetOnline American odds
  5. Report: n_bets, hit_rate, ROI%, net_pnl (units), MDD, net/MDD

Uses OOF p_model_over_oof (no data leakage).
BetOnline only (bookmaker == 'betonlineag').

Grid:
  bet_side:   over, under, both
  min_edge:   2, 3, 5, 8, 10, 12, 15   (percentage points)
  shrinkage:  0.0, 0.1, 0.25, 0.5, 0.75
  odds_type:  all, plus_only, minus_only
  min_books:  1, 2, 3

Sorts results by net_pnl descending.

Usage:
  uv run python 20260806_step5_grid_search.py
"""

from __future__ import annotations

import warnings
from datetime import datetime
from itertools import product
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO_ROOT  = Path(__file__).resolve().parents[3]
TMP_DIR    = Path.home() / "Downloads" / "tmp" / "pass_yds"
HTML_PATH  = REPO_ROOT / "knowledge-base" / "raw" / "20260806-nfl-qb-pass-yds.html"
S4_PATH    = TMP_DIR / "step4_spine_with_pmodel.parquet"
OUT_PATH   = TMP_DIR / "step5_grid_results.parquet"
ET         = ZoneInfo("America/New_York")


def ts() -> str:
    return datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S ET")


def df_to_html(df: pd.DataFrame, title: str = "") -> str:
    out = ""
    if title:
        out += f'<p><strong>{title}</strong></p>'
    out += '<table border="1" cellpadding="4" cellspacing="0" style="border-collapse:collapse;font-size:13px;font-family:monospace">'
    out += "<thead><tr>" + "".join(
        f"<th style='background:#1a1a2e;color:white;padding:6px 10px'>{c}</th>" for c in df.columns
    ) + "</tr></thead><tbody>"
    for i, row in df.iterrows():
        bg = "#f9f9f9" if i % 2 == 0 else "white"
        out += f"<tr style='background:{bg}'>" + "".join(
            f"<td style='padding:4px 8px'>{v}</td>" for v in row.values
        ) + "</tr>"
    out += "</tbody></table>"
    return out


# ── Load Step 4 spine ─────────────────────────────────────────────────────────

print("[Step 5] Loading Step 4 spine...")
spine = pd.read_parquet(S4_PATH)

# Keep only BetOnline rows with OOF p_model and actuals
bol = spine[
    (spine["bookmaker"] == "betonlineag") &
    (spine["p_model_over_oof"].notna()) &
    (spine["passing_yards"].notna())
].copy().reset_index(drop=True)

print(f"  BetOnline rows with p_model + actuals: {len(bol):,}")
print(f"  Seasons: {sorted(bol['nfl_season'].unique())}")

# American odds → decimal odds → implied prob
def american_to_decimal(amer):
    if amer >= 100:
        return 1 + amer / 100
    else:
        return 1 + 100 / abs(amer)

# Compute novig prob if not already in spine
# Spine should have cons_novig_prob_over; BOL novig needs over + under raw probs
# Use per-book odds columns
# Per-book American odds columns are american_over / american_under
over_col  = "american_over"  if "american_over"  in bol.columns else None
under_col = "american_under" if "american_under" in bol.columns else None

if over_col and under_col:
    bol["dec_over"]  = bol[over_col].apply(lambda x: american_to_decimal(x) if pd.notna(x) else np.nan)
    bol["dec_under"] = bol[under_col].apply(lambda x: american_to_decimal(x) if pd.notna(x) else np.nan)
    bol["raw_over"]  = 1 / bol["dec_over"]
    bol["raw_under"] = 1 / bol["dec_under"]
    bol["novig_over_bol"] = bol["raw_over"] / (bol["raw_over"] + bol["raw_under"])
else:
    print("  Warning: american_over/under not found; using novig_prob_over")
    bol["novig_over_bol"] = bol["novig_prob_over"]

print(f"  Odds columns: over={over_col}, under={under_col}")
print(f"  novig_over mean: {bol['novig_over_bol'].mean():.4f}")


# ── P/L computation ───────────────────────────────────────────────────────────

def compute_pnl(row, bet_side: str):
    """
    bet_side: 'over' or 'under'
    Returns (pnl_units, won) where pnl is in units risked (1 unit per bet).
    """
    if bet_side == "over":
        outcome_win = row["outcome"] == "over"
        col = over_col
    else:
        outcome_win = row["outcome"] == "under"
        col = under_col

    if col is None or pd.isna(row.get(col, np.nan)):
        return np.nan, np.nan

    amer = row[col]
    dec = american_to_decimal(amer)
    if outcome_win:
        return dec - 1, 1
    else:
        return -1.0, 0


# ── Grid search ───────────────────────────────────────────────────────────────

print("\n[Step 5] Running grid search...")

BET_SIDES  = ["over", "under", "both"]
MIN_EDGES  = [2, 3, 5, 8, 10, 12, 15]
SHRINKAGES = [0.0, 0.1, 0.25, 0.5, 0.75]
ODDS_TYPES = ["all", "plus_only", "minus_only"]
MIN_BOOKS  = [1, 2, 3]

grid_rows = []
total_combos = len(BET_SIDES) * len(MIN_EDGES) * len(SHRINKAGES) * len(ODDS_TYPES) * len(MIN_BOOKS)
print(f"  {total_combos:,} combinations")

for shrink, min_edge_pp, bet_side, odds_type, min_bks in product(
    SHRINKAGES, MIN_EDGES, BET_SIDES, ODDS_TYPES, MIN_BOOKS
):
    min_edge = min_edge_pp / 100

    # Apply shrinkage
    p_over_s = bol["p_model_over_oof"] * (1 - shrink) + 0.5 * shrink
    p_under_s = 1 - p_over_s

    # Edge vs BetOnline novig
    edge_over  = p_over_s  - bol["novig_over_bol"]
    edge_under = p_under_s - (1 - bol["novig_over_bol"])

    # Book count filter
    if min_bks > 1:
        bk_mask = bol.get("n_books", pd.Series(np.inf, index=bol.index)) >= min_bks
    else:
        bk_mask = pd.Series(True, index=bol.index)

    # Odds type filter
    if odds_type == "plus_only":
        if over_col:
            odds_mask_over  = bol[over_col]  >= 100
            odds_mask_under = bol[under_col] >= 100 if under_col else pd.Series(False, index=bol.index)
        else:
            odds_mask_over = odds_mask_under = pd.Series(True, index=bol.index)
    elif odds_type == "minus_only":
        if over_col:
            odds_mask_over  = bol[over_col]  < 100
            odds_mask_under = bol[under_col] < 100 if under_col else pd.Series(False, index=bol.index)
        else:
            odds_mask_over = odds_mask_under = pd.Series(True, index=bol.index)
    else:
        odds_mask_over = odds_mask_under = pd.Series(True, index=bol.index)

    if bet_side in ["over", "both"]:
        over_mask = (edge_over >= min_edge) & bk_mask & odds_mask_over
    if bet_side in ["under", "both"]:
        under_mask = (edge_under >= min_edge) & bk_mask & odds_mask_under

    # Collect bets
    pnl_list = []
    won_list = []

    if bet_side in ["over", "both"]:
        for _, row in bol[over_mask].iterrows():
            pnl, won = compute_pnl(row, "over")
            if not np.isnan(pnl):
                pnl_list.append(pnl)
                won_list.append(won)

    if bet_side in ["under", "both"]:
        for _, row in bol[under_mask].iterrows():
            pnl, won = compute_pnl(row, "under")
            if not np.isnan(pnl):
                pnl_list.append(pnl)
                won_list.append(won)

    n = len(pnl_list)
    if n < 5:
        continue

    pnl_arr = np.array(pnl_list)
    won_arr = np.array(won_list)

    net_pnl  = float(pnl_arr.sum())
    hit_rate = float(won_arr.mean())
    roi_pct  = float(net_pnl / n * 100)

    # Max drawdown (running cumulative sum)
    cum = np.cumsum(pnl_arr)
    peak = np.maximum.accumulate(cum)
    dd   = cum - peak
    mdd  = float(dd.min())

    net_mdd = net_pnl / abs(mdd) if mdd < -0.001 else np.nan

    grid_rows.append({
        "shrink":    shrink,
        "min_edge":  min_edge_pp,
        "bet_side":  bet_side,
        "odds_type": odds_type,
        "min_books": min_bks,
        "n_bets":    n,
        "hit_rate":  round(hit_rate, 4),
        "net_pnl":   round(net_pnl, 2),
        "roi_pct":   round(roi_pct, 2),
        "mdd":       round(mdd, 2),
        "net_mdd":   round(net_mdd, 2) if not np.isnan(net_mdd) else None,
    })

grid_df = pd.DataFrame(grid_rows)
grid_df.to_parquet(OUT_PATH, index=False)
print(f"\n  {len(grid_df):,} valid strategy configs (n>=5)")

# Sort by net_pnl desc (primary), then roi_pct
grid_sorted = grid_df[grid_df["n_bets"] >= 30].sort_values("net_pnl", ascending=False).reset_index(drop=True)
print(f"  Configs with n>=30 bets: {len(grid_sorted):,}")


# ── Summary tables ────────────────────────────────────────────────────────────

print("\n=== Top 20 by Net PnL (n>=30) ===")
print(grid_sorted.head(20)[["bet_side","min_edge","shrink","odds_type","min_books","n_bets","hit_rate","net_pnl","roi_pct","mdd","net_mdd"]].to_string(index=False))

print("\n=== Top 20 by ROI% (n>=30) ===")
grid_by_roi = grid_df[grid_df["n_bets"] >= 30].sort_values("roi_pct", ascending=False).reset_index(drop=True)
print(grid_by_roi.head(20)[["bet_side","min_edge","shrink","odds_type","min_books","n_bets","hit_rate","net_pnl","roi_pct","mdd","net_mdd"]].to_string(index=False))

print("\n=== Best Over-Only (n>=30, sorted net_pnl) ===")
over_only = grid_sorted[grid_sorted["bet_side"] == "over"]
print(over_only.head(10)[["min_edge","shrink","odds_type","min_books","n_bets","hit_rate","net_pnl","roi_pct","mdd","net_mdd"]].to_string(index=False))

print("\n=== Best Under-Only (n>=30, sorted net_pnl) ===")
under_only = grid_sorted[grid_sorted["bet_side"] == "under"]
print(under_only.head(10)[["min_edge","shrink","odds_type","min_books","n_bets","hit_rate","net_pnl","roi_pct","mdd","net_mdd"]].to_string(index=False))

best = grid_sorted.iloc[0] if len(grid_sorted) else grid_df.sort_values("net_pnl", ascending=False).iloc[0]
print(f"\nBest overall: {best.to_dict()}")


# ── Per-season breakdown for best strategy ────────────────────────────────────

def apply_strategy(df, bet_side, shrink, min_edge_pp, odds_type, min_bks):
    """Apply strategy to df and return per-row (pnl, won, season) tuples."""
    min_edge = min_edge_pp / 100

    p_over_s  = df["p_model_over_oof"] * (1 - shrink) + 0.5 * shrink
    p_under_s = 1 - p_over_s
    edge_over  = p_over_s  - df["novig_over_bol"]
    edge_under = p_under_s - (1 - df["novig_over_bol"])

    if min_bks > 1:
        bk_mask = df.get("n_books", pd.Series(np.inf, index=df.index)) >= min_bks
    else:
        bk_mask = pd.Series(True, index=df.index)

    if odds_type == "plus_only":
        o_mask = df[over_col] >= 100 if over_col else pd.Series(True, index=df.index)
        u_mask = df[under_col] >= 100 if under_col else pd.Series(False, index=df.index)
    elif odds_type == "minus_only":
        o_mask = df[over_col] < 100 if over_col else pd.Series(True, index=df.index)
        u_mask = df[under_col] < 100 if under_col else pd.Series(False, index=df.index)
    else:
        o_mask = u_mask = pd.Series(True, index=df.index)

    rows = []
    if bet_side in ["over", "both"]:
        mask = (edge_over >= min_edge) & bk_mask & o_mask
        for _, row in df[mask].iterrows():
            pnl, won = compute_pnl(row, "over")
            if not np.isnan(pnl):
                rows.append({"pnl": pnl, "won": won, "season": row["nfl_season"], "side": "over"})
    if bet_side in ["under", "both"]:
        mask = (edge_under >= min_edge) & bk_mask & u_mask
        for _, row in df[mask].iterrows():
            pnl, won = compute_pnl(row, "under")
            if not np.isnan(pnl):
                rows.append({"pnl": pnl, "won": won, "season": row["nfl_season"], "side": "under"})
    return rows

if len(grid_sorted):
    best = grid_sorted.iloc[0]
    bets = apply_strategy(
        bol, best["bet_side"], best["shrink"], int(best["min_edge"]),
        best["odds_type"], int(best["min_books"])
    )
    bets_df = pd.DataFrame(bets)

    seasonal = bets_df.groupby("season").agg(
        n_bets=("pnl", "count"),
        net_pnl=("pnl", "sum"),
        hit_rate=("won", "mean"),
    ).reset_index()
    seasonal["roi_pct"] = (seasonal["net_pnl"] / seasonal["n_bets"] * 100).round(2)
    seasonal["net_pnl"] = seasonal["net_pnl"].round(2)
    seasonal["hit_rate"] = seasonal["hit_rate"].round(4)

    print(f"\n=== Per-Season Breakdown (best strategy: {best['bet_side']} / edge≥{best['min_edge']}pp / shrink={best['shrink']} / {best['odds_type']}) ===")
    print(seasonal.to_string(index=False))
else:
    seasonal = pd.DataFrame()
    print("\nNo strategies with n>=30 found.")


# ── Spot-check traces: Allen, Flacco, Ward, Maye ─────────────────────────────

SPOT_CHECKS = [
    ("josh allen",  "Josh Allen (star starter)"),
    ("joe flacco",  "Joe Flacco (journeyman backup)"),
    ("cam ward",    "Cam Ward (2025 rookie)"),
    ("drake maye",  "Drake Maye (2nd year)"),
]

# Use best strategy params to show which of their games would have qualified
best_shrink   = float(best["shrink"])   if len(grid_sorted) else 0.0
best_edge_pp  = int(best["min_edge"])   if len(grid_sorted) else 5
best_side     = str(best["bet_side"])   if len(grid_sorted) else "both"
best_odds_t   = str(best["odds_type"])  if len(grid_sorted) else "all"

p_over_s  = bol["p_model_over_oof"]  * (1 - best_shrink) + 0.5 * best_shrink
p_under_s = 1 - p_over_s
edge_over  = (p_over_s  - bol["novig_over_bol"]).round(4)
edge_under = (p_under_s - (1 - bol["novig_over_bol"])).round(4)
bol["edge_over"]  = edge_over
bol["edge_under"] = edge_under
bol["qualifies"]  = (
    ((best_side in ["over","both"])  & (edge_over  >= best_edge_pp / 100)) |
    ((best_side in ["under","both"]) & (edge_under >= best_edge_pp / 100))
)

spot_html_parts = []
for name_key, label in SPOT_CHECKS:
    player_df = bol[bol["player_norm"].str.contains(name_key, case=False, na=False)]
    if len(player_df) == 0:
        spot_html_parts.append(f"<p><strong>{label}</strong>: not found in spine.</p>")
        print(f"  {label}: not found")
        continue

    show = (
        player_df
        .drop_duplicates(subset=["nfl_season", "nfl_week"])
        .sort_values(["nfl_season", "nfl_week"])
        [["nfl_season","nfl_week","line","passing_yards","p_model_over_oof",
          "novig_over_bol","edge_over","edge_under","outcome","qualifies"]]
        .rename(columns={
            "passing_yards":      "actual",
            "p_model_over_oof":   "p_over",
            "novig_over_bol":     "mkt_novig",
        })
        .round(4)
        .head(15)
        .reset_index(drop=True)
    )
    n_qualify = int(player_df["qualifies"].sum())
    print(f"  {label}: {len(player_df)} rows, {n_qualify} qualifying bets")
    spot_html_parts.append(
        f"<h4>{label} — {len(player_df)} rows, {n_qualify} qualifying bets under best strategy</h4>"
        + df_to_html(show)
    )


# ── DuckDB tests ──────────────────────────────────────────────────────────────

import duckdb

print("\n[Step 5] Running DuckDB validation tests...")
con = duckdb.connect()
con.register("grid", grid_df)
con.register("bol", bol)

tests = []

def run_test(name, sql, expect_true=True):
    result = con.execute(sql).fetchone()[0]
    passed = bool(result) == expect_true
    status = "PASS" if passed else "FAIL"
    tests.append({"test": name, "status": status, "result": result})
    print(f"  [{status}] {name} → {result}")

# T1: grid has results
run_test("T1: grid has results", "SELECT COUNT(*) > 0 FROM grid")

# T2: no ROI above 50% with n>=100 (sanity ceiling — would be extraordinary)
run_test("T2: no insane ROI (>50%) with n>=100", "SELECT COUNT(*) = 0 FROM grid WHERE roi_pct > 50 AND n_bets >= 100")

# T3: hit rates reasonable for n>=30 configs (small n can swing wildly by luck)
run_test("T3: hit rates in [0.25, 0.80] for n>=30 configs", "SELECT COUNT(*) = 0 FROM grid WHERE n_bets >= 30 AND (hit_rate < 0.25 OR hit_rate > 0.80)")

# T4: novig_over_bol is near 0.5 (market is well-calibrated)
run_test("T4: novig_over mean between 0.45 and 0.55", "SELECT AVG(novig_over_bol) BETWEEN 0.45 AND 0.55 FROM bol")

# T5: net_pnl for best strategy is positive
best_pnl = float(grid_df["net_pnl"].max())
run_test("T5: best strategy net_pnl > 0", f"SELECT MAX(net_pnl) > 0 FROM grid")

# T6: strategies with n>=30 exist
run_test("T6: strategies with n>=30 exist", "SELECT COUNT(*) > 0 FROM grid WHERE n_bets >= 30")

n_pass = sum(1 for t in tests if t["status"] == "PASS")
n_fail = sum(1 for t in tests if t["status"] == "FAIL")
print(f"\n  Tests: {n_pass}/{len(tests)} passed")
tests_df = pd.DataFrame(tests)


# ── HTML section ──────────────────────────────────────────────────────────────

print("\n[Step 5] Writing HTML section...")

top20_net  = grid_sorted.head(20)[["bet_side","min_edge","shrink","odds_type","min_books","n_bets","hit_rate","net_pnl","roi_pct","mdd","net_mdd"]].reset_index(drop=True)
top20_roi  = grid_by_roi.head(20)[["bet_side","min_edge","shrink","odds_type","min_books","n_bets","hit_rate","net_pnl","roi_pct","mdd","net_mdd"]].reset_index(drop=True)
top10_over = over_only.head(10)[["min_edge","shrink","odds_type","min_books","n_bets","hit_rate","net_pnl","roi_pct","mdd","net_mdd"]].reset_index(drop=True)
top10_und  = under_only.head(10)[["min_edge","shrink","odds_type","min_books","n_bets","hit_rate","net_pnl","roi_pct","mdd","net_mdd"]].reset_index(drop=True)

best_display = grid_sorted.iloc[0].to_dict() if len(grid_sorted) else {}
best_str = f"{best_display.get('bet_side','?')} | edge≥{best_display.get('min_edge','?')}pp | shrink={best_display.get('shrink','?')} | {best_display.get('odds_type','?')} | min_books={best_display.get('min_books','?')}"
best_stats = f"n={best_display.get('n_bets','?')} | ROI={best_display.get('roi_pct','?')}% | net={best_display.get('net_pnl','?')}u | MDD={best_display.get('mdd','?')}u | net/MDD={best_display.get('net_mdd','?')}"

html = f"""
<section>
<h2>Step 5 — Grid Search over Betting Strategy</h2>
<p><em>{ts()}</em></p>

<h3>Grid Dimensions</h3>
<ul>
  <li>bet_side: over / under / both</li>
  <li>min_edge: {MIN_EDGES} pp</li>
  <li>shrinkage: {SHRINKAGES}</li>
  <li>odds_type: all / plus_only / minus_only</li>
  <li>min_books: 1 / 2 / 3</li>
  <li>Total combos: {total_combos:,} | Valid (n≥5): {len(grid_df):,} | With n≥30: {len(grid_sorted):,}</li>
</ul>

<h3>Best Strategy</h3>
<p><strong>{best_str}</strong><br>{best_stats}</p>

<h3>Top 20 by Net PnL (n≥30)</h3>
{df_to_html(top20_net)}

<h3>Top 20 by ROI% (n≥30)</h3>
{df_to_html(top20_roi)}

<h3>Best Over-Only Strategies (n≥30)</h3>
{df_to_html(top10_over)}

<h3>Best Under-Only Strategies (n≥30)</h3>
{df_to_html(top10_und)}

{f'<h3>Per-Season Breakdown (best strategy)</h3>{df_to_html(seasonal.reset_index(drop=True))}' if len(seasonal) else ''}

<h3>Spot-Check Players (best strategy: {best_str})</h3>
{''.join(spot_html_parts)}

<h3>DuckDB Test Results</h3>
{df_to_html(tests_df)}

<h3>Key Takeaways</h3>
<ul>
  <li>Best strategy: <strong>{best_str}</strong></li>
  <li>OOS performance: {best_stats}</li>
  <li>Next: Step 6 — In-sample IS eval on full training data + OOS cross-validation review</li>
</ul>
</section>
"""

with open(HTML_PATH, "a", encoding="utf-8") as fh:
    fh.write(html)

print(f"[Step 5] HTML section appended → {HTML_PATH}")
print("\n=== DONE ===")
