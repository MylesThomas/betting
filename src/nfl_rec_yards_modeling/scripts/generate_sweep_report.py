"""
Generate HTML grid-search report for NFL rec yards inference parameters.

Output:
  ~/Downloads/nfl_rec_yards_sweep_report.html

Run:
  python src/nfl_rec_yards_modeling/scripts/generate_sweep_report.py
"""

from __future__ import annotations

import itertools
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import nbinom

warnings.filterwarnings("ignore")

LABELED_PATH = Path.home() / "Downloads" / "tmp" / "nfl_rec_yards_per_book.parquet"
ARTIFACT_DIR = Path.home() / "Downloads" / "tmp" / "nfl_rec_yards_artifacts"
OUT_HTML     = Path.home() / "Downloads" / "nfl_rec_yards_sweep_report.html"

TARGET = "receiving_yards"
HYBRID_NEGBIN_THRESHOLD = 20.5
N_BOOT = 10_000
RNG    = np.random.default_rng(42)
JUICE  = 110
BREAKEVEN = JUICE / (JUICE + 100)

BEST_FEATS = [
    "offered_line", "game_total", "proj_own_score",
    "rec_yards_L8", "target_share_L8", "snap_pct_L8",
    "pos_TE", "market_under_prob",
]
KEEP_POSITIONS = ["WR", "TE"]

DIRECTIONS      = ["UNDER", "OVER", "BOTH"]
EDGE_THRESHOLDS = [0.01, 0.03, 0.05, 0.10, 0.20]
LINE_RANGES     = [(20.5, 99.5), (20.5, 69.5), (30.5, 69.5), (30.5, 99.5)]
MIN_BOOKS_OPTS  = [1, 3, 5]

ANALYST_REC = {
    "direction": "UNDER", "edge": 0.05,
    "line_min": 20.5, "line_max": 99.5, "min_books": 1,
}


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pos_TE"] = (df["position"] == "TE").astype(int)
    return df


def run_inference(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    ols, residuals   = artifacts["ols"], artifacts["residuals"]
    nb_coefs, nb_alpha = artifacts["nb_coefs"], artifacts["nb_alpha"]
    result = df.copy()
    mask   = result[BEST_FEATS].notna().all(axis=1)
    idx    = result.index[mask]
    X      = result.loc[idx, BEST_FEATS].to_numpy(dtype=float)
    line   = result.loc[idx, "offered_line"].to_numpy(dtype=float)

    ols_pred = ols.predict(X)
    X_const  = np.column_stack([np.ones(len(X)), X])
    nb_mu    = np.exp(X_const @ nb_coefs)
    mu_c     = np.clip(nb_mu, 1e-3, None)
    n_nb     = 1.0 / nb_alpha
    p_nb     = nbinom.cdf(np.floor(line).astype(int), n=n_nb, p=n_nb / (n_nb + mu_c))
    samp     = RNG.choice(residuals, size=(len(ols_pred), N_BOOT))
    p_bt     = ((ols_pred[:, None] + samp) <= line[:, None]).mean(axis=1)
    p_hyb    = np.where(line < HYBRID_NEGBIN_THRESHOLD, p_bt, p_nb)
    p_mkt    = result.loc[idx, "market_under_prob"].to_numpy(dtype=float)
    edge     = p_hyb - p_mkt
    rec      = np.select(
        [edge > 0.03, edge < -0.03], ["UNDER", "OVER"], default="PASS",
    )
    result.loc[idx, "ols_pred"]       = np.round(ols_pred, 3)
    result.loc[idx, "p_hybrid"]       = np.round(p_hyb, 4)
    result.loc[idx, "p_market"]       = np.round(p_mkt, 4)
    result.loc[idx, "edge"]           = np.round(edge, 4)
    result.loc[idx, "recommendation"] = rec
    result["actual_under"] = (result[TARGET] <= result["offered_line"]).astype(float)
    result["bet_correct"]  = np.where(
        result["recommendation"] == "UNDER", result["actual_under"],
        np.where(result["recommendation"] == "OVER", 1 - result["actual_under"], np.nan),
    )
    return result


def _max_drawdown(bets: pd.DataFrame) -> float:
    sort_cols = [c for c in ["season", "week"] if c in bets.columns]
    ordered = bets.sort_values(sort_cols) if sort_cols else bets
    pnl     = np.where(ordered["bet_correct"].to_numpy() == 1, 100 / JUICE, -1.0)
    cumsum  = np.cumsum(pnl)
    running_max = np.maximum.accumulate(cumsum)
    return float((running_max - cumsum).max())


def sweep_combo(results: pd.DataFrame, direction: str, edge: float,
                lmin: float, lmax: float, min_books: int) -> dict:
    base_mask = results["offered_line"].between(lmin, lmax) & results["ols_pred"].notna()
    if "n_books" in results.columns:
        base_mask &= results["n_books"] >= min_books
    n_props = int(base_mask.sum())

    mask = base_mask & (results["edge"].abs() >= edge)
    if direction == "OVER":
        mask &= results["recommendation"] == "OVER"
    elif direction == "UNDER":
        mask &= results["recommendation"] == "UNDER"
    else:
        mask &= results["recommendation"].isin(["OVER", "UNDER"])
    bets = results[mask]
    if len(bets) == 0:
        return {"n_props": n_props, "n_bets": 0, "hit_rate": np.nan, "ev_per_unit": np.nan,
                "pct_over": np.nan, "mean_edge_pp": np.nan, "mean_line": np.nan,
                "units_won": np.nan, "max_drawdown": np.nan}
    hr    = bets["bet_correct"].mean()
    n_win = bets["bet_correct"].sum()
    n_los = len(bets) - n_win
    units_won = n_win * (100 / JUICE) - n_los
    return {
        "n_props":      n_props,
        "n_bets":       len(bets),
        "hit_rate":     hr,
        "ev_per_unit":  hr - BREAKEVEN,
        "pct_over":     (bets["recommendation"] == "OVER").mean(),
        "mean_edge_pp": bets["edge"].abs().mean() * 100,
        "mean_line":    bets["offered_line"].mean(),
        "units_won":    units_won,
        "max_drawdown": _max_drawdown(bets),
    }


def build_combos(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for direction, edge, (lmin, lmax), min_books in itertools.product(
        DIRECTIONS, EDGE_THRESHOLDS, LINE_RANGES, MIN_BOOKS_OPTS
    ):
        stats = sweep_combo(results, direction, edge, lmin, lmax, min_books)
        rows.append({
            "direction": direction, "edge": edge,
            "line_min": lmin, "line_max": lmax, "min_books": min_books,
            **stats,
        })
    return pd.DataFrame(rows)


# ── HTML rendering ─────────────────────────────────────────────────────────────

def _gradient_bg(val: float, lo: float, hi: float, good_is_high: bool = True) -> str:
    if np.isnan(val) or hi == lo:
        return "background:#f9fafb"
    t = np.clip((val - lo) / (hi - lo), 0, 1)
    if not good_is_high:
        t = 1 - t
    r = int(220 - t * 120)
    g = int(200 + t * 55)
    b = int(200 - t * 120)
    return f"background:rgb({r},{g},{b})"


def _fmt(val, fmt_str: str = ".2f", suffix: str = "") -> str:
    if np.isnan(val):
        return "—"
    return f"{val:{fmt_str}}{suffix}"


def render_table(combos: pd.DataFrame, analyst_rec: dict) -> str:
    numeric_cols = ["n_bets", "hit_rate", "ev_per_unit", "units_won", "max_drawdown", "mean_line"]
    ranges = {c: (combos[c].min(), combos[c].max()) for c in numeric_cols
              if not combos[c].isna().all()}

    rows_html = ""
    for _, r in combos.iterrows():
        is_analyst = (
            r["direction"] == analyst_rec["direction"] and
            abs(r["edge"] - analyst_rec["edge"]) < 1e-6 and
            abs(r["line_min"] - analyst_rec["line_min"]) < 1e-6 and
            abs(r["line_max"] - analyst_rec["line_max"]) < 1e-6 and
            r["min_books"] == analyst_rec["min_books"]
        )
        row_bg = "background:#fff8dc" if is_analyst else ""

        def _cell(col: str, fmt: str = ".2f", pct: bool = False, lo_bad: bool = False) -> str:
            val = r[col]
            if np.isnan(val):
                return '<td style="padding:6px 8px;text-align:center;color:#9ca3af">—</td>'
            lo, hi = ranges.get(col, (val, val))
            bg = _gradient_bg(val, lo, hi, good_is_high=not lo_bad)
            s = f"{val:.1%}" if pct else f"{val:{fmt}}"
            return f'<td style="{bg};padding:6px 8px;text-align:center;font-family:monospace">{s}</td>'

        rows_html += f"""
<tr style="{row_bg}">
  <td style="padding:6px 8px;text-align:center">{r['direction']}</td>
  <td style="padding:6px 8px;text-align:center">{r['edge']*100:.0f}pp</td>
  <td style="padding:6px 8px;text-align:center">{r['line_min']:.1f}–{r['line_max']:.1f}</td>
  <td style="padding:6px 8px;text-align:center">{int(r['min_books'])}</td>
  {_cell('n_bets', 'd')}
  {_cell('hit_rate', '.1%', pct=True)}
  {_cell('ev_per_unit', '.3f')}
  {_cell('units_won', '.2f')}
  {_cell('max_drawdown', '.2f', lo_bad=True)}
  {_cell('mean_line', '.1f')}
</tr>"""

    return f"""
<table style="width:100%;border-collapse:collapse;font-size:12px">
<thead><tr style="background:#1d2d44;color:#fff">
  <th style="padding:8px">Direction</th>
  <th style="padding:8px">Edge</th>
  <th style="padding:8px">Line Range</th>
  <th style="padding:8px">Min Books</th>
  <th style="padding:8px">N Bets</th>
  <th style="padding:8px">Hit Rate</th>
  <th style="padding:8px">EV/Unit</th>
  <th style="padding:8px">Units Won</th>
  <th style="padding:8px">Max DD</th>
  <th style="padding:8px">Mean Line</th>
</tr></thead>
<tbody>{rows_html}</tbody>
</table>"""


def main():
    print(f"\n  Loading labeled dataset...")
    if not LABELED_PATH.exists():
        sys.exit(f"Not found: {LABELED_PATH}\nRun build_labeled_dataset.py first.")
    df = pd.read_parquet(LABELED_PATH)
    df = df[df["position"].isin(KEEP_POSITIONS)].copy()
    df = add_derived(df)
    print(f"    {len(df):,} rows  |  seasons {sorted(df['season'].unique())}")

    missing = [f for f in ["ols_pipeline.joblib", "residuals.npy",
                            "nb_coefs.npy", "nb_alpha.npy", "meta.json"]
               if not (ARTIFACT_DIR / f).exists()]
    if missing:
        sys.exit(f"Missing artifacts: {missing}\nRun train.py first.")

    artifacts = {
        "ols":       joblib.load(ARTIFACT_DIR / "ols_pipeline.joblib"),
        "residuals": np.load(ARTIFACT_DIR / "residuals.npy"),
        "nb_coefs":  np.load(ARTIFACT_DIR / "nb_coefs.npy"),
        "nb_alpha":  float(np.load(ARTIFACT_DIR / "nb_alpha.npy")[0]),
    }

    print(f"  Running inference (bootstrap {N_BOOT:,} draws)...")
    results = run_inference(df, artifacts)

    print("  Running sweep...")
    combos  = build_combos(results)
    n_valid = (combos["n_bets"] > 0).sum()
    print(f"    {len(combos)} combos  |  {n_valid} with ≥1 bet")

    table_html = render_table(combos, ANALYST_REC)
    meta = json.loads((ARTIFACT_DIR / "meta.json").read_text())

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NFL Rec Yards — Strategy Sweep</title>
<style>
  body {{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Arial,sans-serif;
        background:#f4f4f5;margin:0;padding:16px;font-size:13px}}
  .card {{background:#fff;border-radius:8px;border:1px solid #e2e2e4;
          padding:20px;margin-bottom:16px;max-width:1200px;margin-left:auto;margin-right:auto}}
  h2 {{font-size:18px;margin:0 0 4px}}
  p  {{color:#6b7280;font-size:12px;margin:4px 0}}
</style>
</head>
<body>
<div class="card">
  <h2>NFL Receiving Yards — Strategy Sweep</h2>
  <p>In-sample (⚠ model trained on this data) · Seasons {meta['train_seasons']} · {len(df):,} rows · {n_valid} combos with bets</p>
  <p>Highlighted row = analyst pick: {ANALYST_REC}</p>
</div>
<div class="card">
  {table_html}
</div>
</body>
</html>"""

    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"\n  Report → {OUT_HTML}")
    print(f"  Open with: open {OUT_HTML}")
    print()


if __name__ == "__main__":
    main()
