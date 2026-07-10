"""
NFL Sacks — Step 5 (OOS) and Step 6 (IS) Grid Search.

New dimensions vs prior sweep (threshold_search.py):
  odds_bucket        all / plus_odds / minus_odds
  shrinkage          0.0 / 0.25 / 0.50 / 0.75
  prediction_method  model / consensus_line

All sweep params loaded from config.yaml grid_search block.

Shrinkage: pulls p_over toward 0.5 (neutral prior) — reduces model confidence.
  shrunk_p = (1 − s) × p_over + s × 0.5
  Then recompute edges against each book's own implied prob.

odds_bucket (direction-aware):
  For UNDER bet: plus_odds = under_implied < 0.5 (under is a dog)
                 minus_odds = under_implied > 0.5 (under is favored — standard for sacks)
  For OVER bet:  plus_odds = over_implied < 0.5 (over is a dog — standard for sacks)
                 minus_odds = over_implied > 0.5 (over is favored — rare)

prediction_method:
  model          — use stored LR p_over
  consensus_line — use historical sack rate (base rate) as p_over for all rows;
                   tests whether edge is from the ML model vs just knowing sack rates differ by book

P&L: win = (1/implied − 1), loss = −1 (actual per-book prices, not flat -110).

Run:
  python src/nfl_sacks_modeling/scripts/20260709_step5_6_grid_search.py
"""

from __future__ import annotations

import itertools
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).parents[3]
CONFIG_PATH = REPO_ROOT / "src" / "nfl_sacks_modeling" / "config.yaml"
TMP         = Path.home() / "Downloads" / "tmp"

OOS_PATH = TMP / "nfl_sacks_scored_2025_norm.parquet"
IS_PATH  = TMP / "nfl_sacks_scored_insample_norm.parquet"

OUT_OOS_CSV = TMP / "sacks_step5_oos_grid_v2.csv"
OUT_IS_CSV  = TMP / "sacks_step6_is_grid_v2.csv"
OUT_HTML    = Path.home() / "Downloads" / "nfl_sacks_sweep_v2.html"

MIN_BETS = 20


# ── Config ────────────────────────────────────────────────────────────────────
def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg["nfl_sacks_model"]["grid_search"]


# ── Odds bucket mask ──────────────────────────────────────────────────────────
def _odds_bucket_mask(df: pd.DataFrame, bucket: str, direction: str) -> pd.Series:
    if bucket == "all":
        return pd.Series(True, index=df.index)
    if direction in ("under", "both"):
        col = "under_implied"
        return df[col] < 0.5 if bucket == "plus_odds" else df[col] > 0.5
    else:  # over
        col = "over_implied"
        return df[col] < 0.5 if bucket == "plus_odds" else df[col] > 0.5


# ── Max drawdown ──────────────────────────────────────────────────────────────
def _max_drawdown(pnl_series: pd.Series) -> float:
    cum  = pnl_series.cumsum()
    peak = cum.cummax()
    return float((peak - cum).max())


# ── Prepare scored frame with shrinkage + prediction_method ──────────────────
def prepare(df: pd.DataFrame, shrinkage: float, prediction_method: str,
            base_rate: float) -> pd.DataFrame:
    """Add shrunk_p, shrunk_edge_over, shrunk_edge_under columns."""
    out = df.copy()

    if prediction_method == "model":
        p = out["p_over"].to_numpy(dtype=float)
    else:  # consensus_line = historical sack rate
        p = np.full(len(out), base_rate)

    if shrinkage > 0 and prediction_method == "model":
        p = (1 - shrinkage) * p + shrinkage * 0.5

    p = np.clip(p, 0.01, 0.99)
    out["_p"]          = p
    out["_edge_over"]  = p - out["over_implied"].to_numpy(dtype=float)
    out["_edge_under"] = (1 - p) - out["under_implied"].to_numpy(dtype=float)
    return out


# ── Evaluate one combo ────────────────────────────────────────────────────────
def eval_combo(
    df: pd.DataFrame,
    total_universe: int,
    direction: str,
    clf_threshold: float,
    min_edge: float,
    odds_bucket: str,
) -> dict:
    bets_frames = []

    if direction in ("under", "both"):
        cands = df[df["under_implied"].notna() & df["target"].notna()].copy()
        cands = cands[cands["_p"] < clf_threshold]
        cands = cands[cands["_edge_under"] >= min_edge]
        cands = cands[_odds_bucket_mask(cands, odds_bucket, "under")]
        if len(cands):
            wins = (cands["target"] == 0).astype(float)
            cands["_pnl"]      = wins * (1 / cands["under_implied"] - 1) - (1 - wins)
            cands["_bet_impl"] = cands["under_implied"]
            cands["_edge"]     = cands["_edge_under"]
            bets_frames.append(cands)

    if direction in ("over", "both"):
        cands = df[df["over_implied"].notna() & df["target"].notna()].copy()
        cands = cands[cands["_p"] > (1 - clf_threshold)]
        cands = cands[cands["_edge_over"] >= min_edge]
        cands = cands[_odds_bucket_mask(cands, odds_bucket, "over")]
        if len(cands):
            wins = (cands["target"] == 1).astype(float)
            cands["_pnl"]      = wins * (1 / cands["over_implied"] - 1) - (1 - wins)
            cands["_bet_impl"] = cands["over_implied"]
            cands["_edge"]     = cands["_edge_over"]
            bets_frames.append(cands)

    if not bets_frames:
        return {
            "n_bets": 0, "pct_of_universe": 0.0,
            "win_rate": np.nan, "push_rate": 0.0,
            "units_won": np.nan, "roi": np.nan,
            "avg_odds": np.nan, "max_drawdown": np.nan,
            "mean_edge_pp": np.nan,
        }

    pool = pd.concat(bets_frames).sort_values("week")
    n         = len(pool)
    wins      = (pool["_pnl"] > 0).sum()
    units_won = float(pool["_pnl"].sum())
    roi       = units_won / n

    # avg_odds: average implied → American
    avg_impl = float(pool["_bet_impl"].mean())
    if avg_impl >= 0.5:
        avg_american = -100 * avg_impl / (1 - avg_impl)
    else:
        avg_american = 100 * (1 - avg_impl) / avg_impl

    return {
        "n_bets":          n,
        "pct_of_universe": n / total_universe if total_universe > 0 else 0.0,
        "win_rate":        round(wins / n, 4),
        "push_rate":       0.0,
        "units_won":       round(units_won, 3),
        "roi":             round(roi, 4),
        "avg_odds":        round(avg_american, 1),
        "max_drawdown":    round(_max_drawdown(pool["_pnl"]), 3),
        "mean_edge_pp":    round(float(pool["_edge"].mean() * 100), 2),
    }


# ── Full sweep ────────────────────────────────────────────────────────────────
def run_sweep(label: str, raw: pd.DataFrame, cfg_gs: dict) -> pd.DataFrame:
    # Base rate = historical sack rate for consensus_line
    base_rate = float(raw["target"].dropna().mean())
    print(f"  Base rate (sack rate): {base_rate:.4f}")

    total_rows = len(raw[raw["target"].notna()])

    clf_thresholds  = cfg_gs["clf_threshold"]
    directions      = cfg_gs["direction"]
    min_edges       = cfg_gs["min_edge"]
    odds_buckets    = cfg_gs["odds_bucket"]
    shrinkages      = cfg_gs["shrinkage"]
    pred_methods    = cfg_gs["prediction_method"]

    # Pre-compute scored frames: (method, shrinkage) pairs
    # consensus_line has no shrinkage sweep (no meaning)
    pm_s_pairs = [
        (pm, s) for pm in pred_methods for s in shrinkages
        if not (pm == "consensus_line" and s > 0)
    ]
    pm_s_pairs = list(dict.fromkeys(pm_s_pairs))

    print(f"  Pre-computing {len(pm_s_pairs)} (method, shrinkage) frames...")
    cache: dict[tuple, pd.DataFrame] = {}
    for pm, s in pm_s_pairs:
        cache[(pm, s)] = prepare(raw, s, pm, base_rate)

    n_combos = len(pm_s_pairs) * len(clf_thresholds) * len(directions) * len(min_edges) * len(odds_buckets)
    print(f"  Running {n_combos:,} combos...")

    rows = []
    for (pm, s), clf_t, direction, min_edge, bucket in itertools.product(
        pm_s_pairs, clf_thresholds, directions, min_edges, odds_buckets
    ):
        stats = eval_combo(
            df             = cache[(pm, s)],
            total_universe = total_rows,
            direction      = direction,
            clf_threshold  = clf_t,
            min_edge       = min_edge,
            odds_bucket    = bucket,
        )
        rows.append({
            "prediction_method": pm,
            "shrinkage":         s,
            "direction":         direction,
            "clf_threshold":     clf_t,
            "min_edge":          min_edge,
            "odds_bucket":       bucket,
            **stats,
        })

    result = pd.DataFrame(rows)
    return result.sort_values("units_won", ascending=False).reset_index(drop=True)


# ── HTML ──────────────────────────────────────────────────────────────────────
def _pct(v) -> str:
    return "—" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.1%}"

def _f2(v) -> str:
    return "—" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.2f}"

def _odds(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"+{v:.0f}" if v > 0 else f"{v:.0f}"

def _heat(val: float, lo: float, hi: float, good_high: bool = True) -> str:
    if np.isnan(val) or hi == lo:
        return "background:#f9fafb"
    t = np.clip((val - lo) / (hi - lo), 0, 1)
    if not good_high:
        t = 1 - t
    r = int(230 - t * 110)
    g = int(190 + t * 65)
    b = int(200 - t * 100)
    return f"background:rgb({r},{g},{b})"


def render_table(df: pd.DataFrame, label: str) -> str:
    valid = df[df["n_bets"] >= MIN_BETS].copy()
    top   = valid.head(120)

    def _col(r, col, fmt_fn, good_high=True):
        v = r[col]
        if isinstance(v, float) and np.isnan(v):
            return '<td style="padding:5px 8px;text-align:center;color:#9ca3af">—</td>'
        lo = valid[col].min()
        hi = valid[col].max()
        bg = _heat(float(v), float(lo), float(hi), good_high)
        return f'<td style="{bg};padding:5px 8px;text-align:center;font-family:monospace">{fmt_fn(v)}</td>'

    rows_html = ""
    for _, r in top.iterrows():
        dd_warn = ""
        if not np.isnan(r["max_drawdown"]) and not np.isnan(r["units_won"]):
            if r["units_won"] > 0 and r["max_drawdown"] > r["units_won"]:
                dd_warn = ' title="⚠ max drawdown exceeds units won"'
        low_n = r["n_bets"] < MIN_BETS
        style = "color:#9ca3af" if low_n else ""
        rows_html += f"""
<tr style="font-size:12px;{style}">
  <td style="padding:5px 8px;text-align:center">{r['prediction_method']}</td>
  <td style="padding:5px 8px;text-align:center">{r['shrinkage']:.2f}</td>
  <td style="padding:5px 8px;text-align:center">{r['direction']}</td>
  <td style="padding:5px 8px;text-align:center">{r['clf_threshold']:.2f}</td>
  <td style="padding:5px 8px;text-align:center">{int(r['min_edge']*100) if r['min_edge'] < 1 else r['min_edge']}pp</td>
  <td style="padding:5px 8px;text-align:center">{r['odds_bucket']}</td>
  {_col(r, 'n_bets', lambda v: f"{int(v):,}")}
  {_col(r, 'pct_of_universe', _pct)}
  {_col(r, 'win_rate', _pct)}
  {_col(r, 'units_won', _f2)}
  {_col(r, 'roi', _pct)}
  <td style="padding:5px 8px;text-align:center;font-family:monospace">{_odds(r['avg_odds'])}</td>
  {_col(r, 'max_drawdown', _f2, good_high=False)}{dd_warn}
  <td style="padding:5px 8px;text-align:center;font-family:monospace">{_f2(r['mean_edge_pp'])}pp</td>
</tr>"""

    return f"""
<h3 style="margin:0 0 8px">{label} — Top 120 by units won (≥{MIN_BETS} bets only)</h3>
<p style="color:#6b7280;font-size:12px;margin:0 0 12px">
  Grey = &lt;{MIN_BETS} bets. ⚠ tooltip = max drawdown &gt; units won.
  P&amp;L uses actual per-book implied prices (not flat -110).
</p>
<div style="overflow-x:auto">
<table style="width:100%;border-collapse:collapse;font-size:12px">
<thead><tr style="background:#1d2d44;color:#fff;font-size:11px">
  <th style="padding:7px 8px">Method</th>
  <th style="padding:7px 8px">Shrink</th>
  <th style="padding:7px 8px">Direction</th>
  <th style="padding:7px 8px">Clf Thresh</th>
  <th style="padding:7px 8px">Edge ≥</th>
  <th style="padding:7px 8px">Odds Bucket</th>
  <th style="padding:7px 8px">N Bets</th>
  <th style="padding:7px 8px">% Universe</th>
  <th style="padding:7px 8px">Win Rate</th>
  <th style="padding:7px 8px">Units Won</th>
  <th style="padding:7px 8px">ROI</th>
  <th style="padding:7px 8px">Avg Odds</th>
  <th style="padding:7px 8px">Max DD</th>
  <th style="padding:7px 8px">Mean Edge</th>
</tr></thead>
<tbody>{rows_html}</tbody>
</table>
</div>"""


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    cfg_gs = load_config()

    print("\n  Loading OOS scored data...")
    oos_raw = pd.read_parquet(OOS_PATH)
    print(f"    {len(oos_raw):,} rows  |  weeks {oos_raw['week'].min()}–{oos_raw['week'].max()}")
    print(f"    Bookmakers: {sorted(oos_raw['bookmaker'].unique())}")

    print("\n  Loading IS scored data...")
    is_raw = pd.read_parquet(IS_PATH)
    print(f"    {len(is_raw):,} rows")

    print("\n" + "="*70)
    print("  STEP 5 — Out-of-Sample Grid Search")
    print("="*70)
    oos_grid = run_sweep("OOS", oos_raw, cfg_gs)
    oos_grid.to_csv(OUT_OOS_CSV, index=False)
    print(f"\n  Saved → {OUT_OOS_CSV}  ({len(oos_grid):,} combos)")

    top5 = oos_grid[oos_grid["n_bets"] >= MIN_BETS].head(15)
    print("\n  Top 15 OOS (≥20 bets) by units won:")
    print(top5[["prediction_method","shrinkage","direction","clf_threshold",
                "min_edge","odds_bucket","n_bets","win_rate","units_won","roi","max_drawdown"]].to_string(index=False))

    print("\n" + "="*70)
    print("  STEP 6 — In-Sample Grid Search")
    print("="*70)
    is_grid = run_sweep("IS", is_raw, cfg_gs)
    is_grid.to_csv(OUT_IS_CSV, index=False)
    print(f"\n  Saved → {OUT_IS_CSV}  ({len(is_grid):,} combos)")

    top6 = is_grid[is_grid["n_bets"] >= MIN_BETS].head(10)
    print("\n  Top 10 IS (≥20 bets) by units won:")
    print(top6[["prediction_method","shrinkage","direction","clf_threshold",
                "min_edge","odds_bucket","n_bets","win_rate","units_won","roi","max_drawdown"]].to_string(index=False))

    # ── HTML ──────────────────────────────────────────────────────────────────
    ts = subprocess.check_output(
        ["bash", "-c", "TZ=America/New_York date '+%Y-%m-%d %H:%M:%S ET'"]
    ).decode().strip()

    n_oos_valid = int((oos_grid["n_bets"] >= MIN_BETS).sum())
    n_is_valid  = int((is_grid["n_bets"] >= MIN_BETS).sum())

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NFL Sacks — Strategy Sweep v2</title>
<style>
  body {{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Arial,sans-serif;
        background:#f4f4f5;margin:0;padding:16px;font-size:13px}}
  .card {{background:#fff;border-radius:8px;border:1px solid #e2e2e4;
          padding:20px;margin-bottom:16px}}
  h2 {{font-size:18px;margin:0 0 4px}}
  p  {{color:#6b7280;font-size:12px;margin:4px 0}}
  .note {{background:#fffbeb;border:1px solid #fcd34d;border-radius:6px;
          padding:10px 14px;margin-bottom:12px;font-size:12px}}
</style>
</head>
<body>
<div class="card">
  <h2>NFL Sacks Props — Strategy Sweep v2</h2>
  <p>Generated: {ts}</p>
  <p>New dimensions: <b>odds_bucket</b> | <b>shrinkage</b> | <b>prediction_method</b></p>
  <p>OOS: {len(oos_grid):,} combos, {n_oos_valid:,} with ≥{MIN_BETS} bets &nbsp;|&nbsp;
     IS: {len(is_grid):,} combos, {n_is_valid:,} with ≥{MIN_BETS} bets</p>
  <div class="note">
    <b>Model:</b> M7 LR (prop_median_impl_over + qbhit_rate_L16 + sack_rate_Lcareer).
    OOS = train 2024, holdout 2025. IS = train+score 2024+2025.<br><br>
    <b>shrinkage:</b> pulls p_over toward 0.5 — (1−s)×p_over + s×0.5 — applied before thresholds.<br>
    <b>consensus_line:</b> replaces ML p_over with the historical sack rate (base rate) for all rows;
    tests whether edge comes from the model or just knowing base rates differ from book odds.<br>
    <b>odds_bucket:</b> for UNDER bets: minus_odds = under_implied &gt; 0.5 (standard for sacks).
  </div>
</div>
<div class="card">
  <h2>Step 5 — Out-of-Sample Results</h2>
  <p>Train 2024 · Holdout 2025 · {len(oos_raw):,} player-game-book rows</p>
  {render_table(oos_grid, "OOS")}
</div>
<div class="card">
  <h2>Step 6 — In-Sample Results</h2>
  <p>Train+Score 2024+2025 (⚠ model trained on this data) · {len(is_raw):,} rows</p>
  {render_table(is_grid, "IS")}
</div>
</body>
</html>"""

    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"\n  Report → {OUT_HTML}")
    print(f"  Open:   open {OUT_HTML}")
    print()


if __name__ == "__main__":
    main()
