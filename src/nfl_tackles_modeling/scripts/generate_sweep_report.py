"""
Generate HTML grid-search report for NFL tackles inference parameters.

Runs the full parameter sweep and outputs a self-contained HTML file
with color-coded gradient tables, summary cards, and edge sensitivity.

Output:
  ~/Downloads/nfl_tackles_sweep_report.html

Run:
  python src/nfl_tackles_modeling/scripts/generate_sweep_report.py
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

# ── Paths / constants ─────────────────────────────────────────────────────────
LABELED_PATH = Path.home() / "Downloads" / "tmp" / "nfl_tackles_per_book.parquet"
ARTIFACT_DIR = Path.home() / "Downloads" / "tmp" / "nfl_tackles_artifacts"
OUT_HTML     = Path.home() / "Downloads" / "nfl_tackles_sweep_report.html"

TARGET = "tackles_combined"
HYBRID_NEGBIN_THRESHOLD = 4.5
N_BOOT = 10_000
RNG    = np.random.default_rng(42)
JUICE  = 110
BREAKEVEN = JUICE / (JUICE + 100)   # 0.5238

POS_GROUP_MAP = {
    "LB": "LB", "CB": "CB", "DB": "CB",
    "S": "S", "FS": "S", "SS": "S",
    "DE": "DL", "DT": "DL", "DL": "DL", "NT": "DL",
}
BEST_FEATS = [
    "offered_line", "game_total", "proj_opp_score", "tackle_rate_L16",
    "pos_LB", "pos_CB", "pos_S", "pos_DL", "market_under_prob",
]
DROP_POSITIONS = ["WR", "FB"]

DIRECTIONS      = ["UNDER", "OVER", "BOTH"]
EDGE_THRESHOLDS = [0.01, 0.03, 0.05, 0.10, 0.20]
LINE_RANGES     = [(2.5, 9.5), (4.5, 9.5), (4.5, 8.5)]
MIN_BOOKS_OPTS  = [1, 3, 5]

# ── Analyst pick (hardcoded) ──────────────────────────────────────────────────
# OVER signal is weak; edge≥0.03 does nothing (no bets in 1-3pp range);
# 2.5-9.5 includes uncalibrated low-line tail. This config has clean signal,
# half the max DD, and survives OOS hit-rate degradation.
ANALYST_REC = {"direction": "UNDER", "edge": 0.05, "line_min": 4.5, "line_max": 9.5, "min_books": 1}


# ── Inference logic ───────────────────────────────────────────────────────────

def add_derived(df):
    df = df.copy()
    df["position_group"] = df["position"].map(POS_GROUP_MAP)
    for g in ["LB", "CB", "S", "DL"]:
        df[f"pos_{g}"] = (df["position_group"] == g).astype(int)
    return df


def run_inference(df, artifacts):
    ols, residuals = artifacts["ols"], artifacts["residuals"]
    nb_coefs, nb_alpha = artifacts["nb_coefs"], artifacts["nb_alpha"]
    result = df.copy()
    mask = result[BEST_FEATS].notna().all(axis=1)
    idx  = result.index[mask]
    X    = result.loc[idx, BEST_FEATS].to_numpy(dtype=float)
    line = result.loc[idx, "offered_line"].to_numpy(dtype=float)

    ols_pred = ols.predict(X)
    X_const  = np.column_stack([np.ones(len(X)), X])
    nb_mu    = np.exp(X_const @ nb_coefs)

    mu_c = np.clip(nb_mu, 1e-3, None)
    n_nb = 1.0 / nb_alpha
    p_nb_arr = n_nb / (n_nb + mu_c)
    p_nb = nbinom.cdf(np.floor(line).astype(int), n=n_nb, p=p_nb_arr)
    samp = RNG.choice(residuals, size=(len(ols_pred), N_BOOT))
    sims = ols_pred[:, None] + samp
    p_bt = (sims <= line[:, None]).mean(axis=1)
    p_hyb = np.where(line < HYBRID_NEGBIN_THRESHOLD, p_bt, p_nb)

    p_mkt = result.loc[idx, "market_under_prob"].to_numpy(dtype=float)
    edge  = p_hyb - p_mkt
    rec   = np.select(
        [edge > 0.03, edge < -0.03], ["UNDER", "OVER"], default="PASS"
    )

    result.loc[idx, "ols_pred"]       = np.round(ols_pred, 3)
    result.loc[idx, "p_hybrid"]       = np.round(p_hyb, 4)
    result.loc[idx, "p_market"]       = np.round(p_mkt, 4)
    result.loc[idx, "edge"]           = np.round(edge, 4)
    result.loc[idx, "recommendation"] = rec

    result["actual_under"] = (result[TARGET] <= result["offered_line"]).astype(float)
    result["bet_correct"] = np.where(
        result["recommendation"] == "UNDER", result["actual_under"],
        np.where(result["recommendation"] == "OVER", 1 - result["actual_under"], np.nan),
    )
    return result


def _max_drawdown(bets: pd.DataFrame) -> float:
    """Max peak-to-trough drawdown in units (1 unit per bet, win = +100/110, loss = -1)."""
    sort_cols = [c for c in ["season", "week"] if c in bets.columns]
    ordered = bets.sort_values(sort_cols) if sort_cols else bets
    pnl = np.where(ordered["bet_correct"].to_numpy() == 1, 100 / JUICE, -1.0)
    cumsum = np.cumsum(pnl)
    running_max = np.maximum.accumulate(cumsum)
    return float((running_max - cumsum).max())


def sweep_combo(results, direction, edge, lmin, lmax, min_books):
    # Universe: all scored props in line range with enough books (no edge/direction filter)
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
    hr = bets["bet_correct"].mean()
    n_wins = bets["bet_correct"].sum()
    n_loss = len(bets) - n_wins
    units_won = n_wins * (100 / JUICE) - n_loss
    return {
        "n_props":       n_props,
        "n_bets":        len(bets),
        "hit_rate":      round(hr * 100, 2),
        "ev_per_unit":   round(hr * (100 / JUICE) - (1 - hr), 4),
        "pct_over":      round((bets["recommendation"] == "OVER").mean() * 100, 1),
        "mean_edge_pp":  round(bets["edge"].abs().mean() * 100, 2),
        "mean_line":     round(bets["offered_line"].mean(), 2),
        "units_won":     round(units_won, 2),
        "max_drawdown":  round(_max_drawdown(bets), 2),
    }


# ── Color helpers ─────────────────────────────────────────────────────────────

def _lerp(a, b, t):
    return int(a + (b - a) * max(0.0, min(1.0, t)))


def _interp(t, stops):
    """Multi-stop color interpolation. stops = list of (r,g,b)."""
    t = max(0.0, min(1.0, t))
    n = len(stops) - 1
    i = min(int(t * n), n - 1)
    lo = t * n - i
    r = _lerp(stops[i][0], stops[i+1][0], lo)
    g = _lerp(stops[i][1], stops[i+1][1], lo)
    b = _lerp(stops[i][2], stops[i+1][2], lo)
    return f"rgb({r},{g},{b})"


def ev_color(v):
    if pd.isna(v):
        return "#1c2333", "#8b949e"
    # red → gray → bright green; pivot at 0
    if v < 0:
        t = max(0, min(1, (v + 0.08) / 0.08))   # -0.08 → 0
        bg = _interp(t, [(180, 30, 30), (80, 80, 80)])
    else:
        t = max(0, min(1, v / 0.18))             # 0 → +0.18
        bg = _interp(t, [(60, 90, 60), (30, 200, 80)])
    return bg, "#e6edf3"


def hr_color(v):
    if pd.isna(v):
        return "#1c2333", "#8b949e"
    be = BREAKEVEN * 100   # 52.38
    if v < be:
        t = max(0, min(1, (v - 50) / (be - 50)))
        bg = _interp(t, [(140, 20, 20), (160, 80, 30)])
    else:
        t = max(0, min(1, (v - be) / (66 - be)))
        bg = _interp(t, [(60, 120, 60), (40, 210, 90)])
    return bg, "#e6edf3"


def n_color(v, max_n):
    if pd.isna(v) or max_n == 0:
        return "#1c2333", "#8b949e"
    t = v / max_n
    bg = _interp(t, [(20, 40, 80), (31, 111, 235)])
    return bg, "#e6edf3"


def edge_color(v):
    t = max(0, min(1, (v - 0.01) / (0.20 - 0.01)))
    bg = _interp(t, [(60, 40, 100), (140, 90, 230)])
    return bg, "#e6edf3"


def units_color(v):
    if pd.isna(v):
        return "#1c2333", "#8b949e"
    if v < 0:
        t = max(0, min(1, (-v) / 60))
        bg = _interp(t, [(80, 50, 50), (180, 30, 30)])
    else:
        t = max(0, min(1, v / 150))
        bg = _interp(t, [(50, 80, 50), (30, 200, 80)])
    return bg, "#e6edf3"


def dd_color(v):
    if pd.isna(v):
        return "#1c2333", "#8b949e"
    t = max(0, min(1, v / 30))   # 0 units = green, 30+ = red
    bg = _interp(t, [(40, 120, 40), (180, 40, 40)])
    return bg, "#e6edf3"


DIR_COLORS = {
    "UNDER": ("#1f4e8c", "#79c0ff"),
    "OVER":  ("#5a2d00", "#f0883e"),
    "BOTH":  ("#1a4a3a", "#3fb950"),
}


# ── HTML generation ───────────────────────────────────────────────────────────

def _td(content, bg="#1c2333", fg="#e6edf3", bold=False, align="center"):
    bw = "font-weight:600;" if bold else ""
    return (f'<td style="background:{bg};color:{fg};{bw}'
            f'text-align:{align};padding:7px 10px;border-bottom:1px solid #21262d">'
            f'{content}</td>')


def _matches_cfg(row, cfg):
    if cfg is None:
        return False
    return (row["direction"] == cfg["direction"] and row["edge"] == cfg["edge"] and
            row["line_min"] == cfg["line_min"] and row["line_max"] == cfg["line_max"] and
            row["min_books"] == cfg["min_books"])


def render_table(sweep, max_n, highlight_cfg=None, analyst_cfg=None):
    cols = [
        ("direction", "Direction"), ("edge", "Edge"), ("line_min", "Line Min"),
        ("line_max", "Line Max"), ("min_books", "Min Books"),
        ("n_props", "# Props"), ("n_bets", "# Bets"), ("hit_rate", "Hit Rate %"),
        ("ev_per_unit", "EV/Unit"), ("pct_over", "% Over"),
        ("mean_edge_pp", "Mean Edge pp"), ("mean_line", "Mean Line"),
        ("units_won", "Units Won"), ("max_drawdown", "Max DD"),
    ]

    header_cells = "".join(
        f'<th onclick="sortTable({i})" style="cursor:pointer;padding:10px 12px;'
        f'background:#21262d;color:#8b949e;font-weight:600;font-size:12px;'
        f'text-align:center;border-bottom:2px solid #30363d;white-space:nowrap">'
        f'{label} ↕</th>'
        for i, (_, label) in enumerate(cols)
    )

    rows_html = []
    for _, row in sweep.iterrows():
        is_rec    = _matches_cfg(row, highlight_cfg)
        is_analyst = _matches_cfg(row, analyst_cfg)
        if is_analyst:
            border = "box-shadow:inset 0 0 0 2px #58a6ff;"
        elif is_rec:
            border = "box-shadow:inset 0 0 0 2px #f0883e;"
        else:
            border = ""

        d = row["direction"]
        d_bg, d_fg = DIR_COLORS.get(d, ("#1c2333", "#e6edf3"))
        ev_bg, ev_fg = ev_color(row["ev_per_unit"])
        hr_bg, hr_fg = hr_color(row["hit_rate"])
        nb_bg, nb_fg = n_color(row["n_bets"], max_n)
        ed_bg, ed_fg = edge_color(row["edge"])

        hit_str = f"{row['hit_rate']:.2f}%" if not pd.isna(row["hit_rate"]) else "—"
        ev_str  = f"{row['ev_per_unit']:+.4f}" if not pd.isna(row["ev_per_unit"]) else "—"
        rec_icon = " ♦" if is_analyst else (" ★" if is_rec else "")

        u_bg, u_fg = units_color(row["units_won"])
        d_bg2, d_fg2 = dd_color(row["max_drawdown"])
        units_str = f"{row['units_won']:+.2f}" if not pd.isna(row["units_won"]) else "—"
        dd_str    = f"{row['max_drawdown']:.2f}" if not pd.isna(row["max_drawdown"]) else "—"

        tds = (
            _td(f"<b>{d}{rec_icon}</b>", d_bg, d_fg) +
            _td(f"{row['edge']:.2f}", *ed_bg_fg(row["edge"])) +
            _td(f"{row['line_min']:.1f}") +
            _td(f"{row['line_max']:.1f}") +
            _td(f"{row['min_books']:.0f}") +
            _td(f"{row['n_props']:.0f}") +
            _td(f"{row['n_bets']:.0f}", nb_bg, nb_fg, bold=True) +
            _td(hit_str, hr_bg, hr_fg, bold=True) +
            _td(ev_str, ev_bg, ev_fg, bold=True) +
            _td(f"{row['pct_over']:.1f}%" if not pd.isna(row["pct_over"]) else "—") +
            _td(f"{row['mean_edge_pp']:.2f}" if not pd.isna(row["mean_edge_pp"]) else "—") +
            _td(f"{row['mean_line']:.2f}" if not pd.isna(row["mean_line"]) else "—") +
            _td(units_str, u_bg, u_fg, bold=True) +
            _td(dd_str, d_bg2, d_fg2, bold=True)
        )
        rows_html.append(
            f'<tr class="dr-{d.lower()}" style="{border}">{tds}</tr>'
        )

    return f"""
    <table id="sweep-table" style="width:100%;border-collapse:collapse;font-size:13px">
      <thead><tr>{header_cells}</tr></thead>
      <tbody>{''.join(rows_html)}</tbody>
    </table>"""


def ed_bg_fg(v):
    bg = edge_color(v)
    return bg[0], bg[1]


def render_card(direction, row, is_rec=False):
    if row is None:
        return f'<div class="card"><h3>{direction}</h3><p>No combos with ≥50 bets</p></div>'
    d_bg, d_fg = DIR_COLORS.get(direction, ("#1c2333", "#e6edf3"))
    hr   = f"{row['hit_rate']:.1f}%"
    ev   = f"{row['ev_per_unit']:+.4f}"
    n    = f"{row['n_bets']:.0f}"
    u    = f"{row['units_won']:+.2f}" if not pd.isna(row.get("units_won", np.nan)) else "—"
    ev_b, ev_f = ev_color(row["ev_per_unit"])
    hr_b, hr_f = hr_color(row["hit_rate"])
    u_b,  u_f  = units_color(row.get("units_won", np.nan))
    rec_border = "border:2px solid #f0883e;" if is_rec else ""
    label = "★ Recommended" if is_rec else "Best Config"
    return f"""
    <div class="card" style="{rec_border}">
      <div class="card-header" style="background:{d_bg};color:{d_fg}">
        <span class="card-dir">{direction}</span>
        <span class="card-rank">{label}</span>
      </div>
      <div class="card-body">
        <div class="stat-row">
          <span class="stat-label">Units won</span>
          <span class="stat-val" style="background:{u_b};color:{u_f}">{u}</span>
        </div>
        <div class="stat-row">
          <span class="stat-label">EV / unit</span>
          <span class="stat-val" style="background:{ev_b};color:{ev_f}">{ev}</span>
        </div>
        <div class="stat-row">
          <span class="stat-label">Hit rate</span>
          <span class="stat-val" style="background:{hr_b};color:{hr_f}">{hr}</span>
        </div>
        <div class="stat-row">
          <span class="stat-label">Bets</span>
          <span class="stat-val">{n}</span>
        </div>
        <div class="config-line">
          edge ≥ {row['edge']}  ·  lines {row['line_min']}–{row['line_max']}  ·  min_books {row['min_books']:.0f}
        </div>
      </div>
    </div>"""


def render_sensitivity(sweep):
    sub = sweep[
        (sweep["direction"] == "BOTH") &
        (sweep["line_min"] == 4.5) &
        (sweep["line_max"] == 8.5) &
        (sweep["min_books"] == 3)
    ].sort_values("edge")

    max_n = sub["n_bets"].max()
    rows = []
    for _, r in sub.iterrows():
        ev_b, ev_f = ev_color(r["ev_per_unit"])
        hr_b, hr_f = hr_color(r["hit_rate"])
        nb_b, nb_f = n_color(r["n_bets"], max_n)
        hit_str = f"{r['hit_rate']:.2f}%" if not pd.isna(r["hit_rate"]) else "—"
        ev_str  = f"{r['ev_per_unit']:+.4f}" if not pd.isna(r["ev_per_unit"]) else "—"
        rows.append(f"""
        <tr>
          <td style="text-align:center;padding:8px 14px;font-weight:700;color:#bc8cff">{r['edge']:.2f}</td>
          <td style="text-align:center;padding:8px 14px;background:{nb_b};color:{nb_f};font-weight:600">{r['n_bets']:.0f}</td>
          <td style="text-align:center;padding:8px 14px;background:{hr_b};color:{hr_f};font-weight:600">{hit_str}</td>
          <td style="text-align:center;padding:8px 14px;background:{ev_b};color:{ev_f};font-weight:700">{ev_str}</td>
          <td style="text-align:center;padding:8px 14px;color:#8b949e">{r['mean_edge_pp']:.1f}</td>
        </tr>""")

    return f"""
    <table style="border-collapse:collapse;font-size:13px;width:100%;max-width:620px">
      <thead>
        <tr style="background:#21262d">
          <th style="padding:10px 14px;color:#8b949e;font-size:12px">Edge Threshold</th>
          <th style="padding:10px 14px;color:#8b949e;font-size:12px"># Bets</th>
          <th style="padding:10px 14px;color:#8b949e;font-size:12px">Hit Rate</th>
          <th style="padding:10px 14px;color:#8b949e;font-size:12px">EV / Unit</th>
          <th style="padding:10px 14px;color:#8b949e;font-size:12px">Mean Edge pp</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>"""


def generate_html(sweep, meta, results_n):
    max_n = sweep["n_bets"].max()
    # Best per direction = highest units_won with at least 50 bets
    best_combos = {}
    for d in DIRECTIONS:
        sub = sweep[(sweep["direction"] == d) & (sweep["n_bets"] >= 50)].sort_values(
            "units_won", ascending=False
        )
        best_combos[d] = sub.iloc[0].to_dict() if not sub.empty else None

    # Global recommendation = highest units_won across all directions (≥50 bets)
    candidates = sweep[sweep["n_bets"] >= 50].sort_values("units_won", ascending=False)
    rec_row = candidates.iloc[0].to_dict() if not candidates.empty else None
    rec_cfg = None
    if rec_row is not None:
        rec_cfg = {
            "direction": rec_row["direction"],
            "edge":      rec_row["edge"],
            "line_min":  rec_row["line_min"],
            "line_max":  rec_row["line_max"],
            "min_books": rec_row["min_books"],
        }

    # Look up analyst rec row for callout stats
    analyst_row = None
    ar = ANALYST_REC
    match = sweep[
        (sweep["direction"] == ar["direction"]) & (sweep["edge"] == ar["edge"]) &
        (sweep["line_min"] == ar["line_min"]) & (sweep["line_max"] == ar["line_max"]) &
        (sweep["min_books"] == ar["min_books"])
    ]
    if not match.empty:
        analyst_row = match.iloc[0].to_dict()

    sweep_sorted = sweep[sweep["n_bets"] >= 5].sort_values("units_won", ascending=False)
    table_html = render_table(sweep_sorted, max_n, rec_cfg, ANALYST_REC)

    cards_html = "".join(
        render_card(d, best_combos[d], is_rec=(
            rec_cfg is not None and
            best_combos[d] is not None and
            best_combos[d]["direction"] == rec_cfg["direction"] and
            best_combos[d]["edge"] == rec_cfg["edge"] and
            best_combos[d]["line_min"] == rec_cfg["line_min"] and
            best_combos[d]["line_max"] == rec_cfg["line_max"] and
            best_combos[d]["min_books"] == rec_cfg["min_books"]
        ))
        for d in DIRECTIONS
    )
    sens_html  = render_sensitivity(sweep)

    be_pct = f"{BREAKEVEN*100:.2f}%"
    n_combos = len(DIRECTIONS) * len(EDGE_THRESHOLDS) * len(LINE_RANGES) * len(MIN_BOOKS_OPTS)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NFL Tackles — Param Sweep</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #0d1117; color: #e6edf3;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 14px; line-height: 1.5;
  }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 28px 24px 60px; }}

  /* Header */
  header {{ margin-bottom: 28px; }}
  header h1 {{ font-size: 24px; font-weight: 700; color: #e6edf3; margin-bottom: 8px; }}
  .badge {{
    display: inline-block; padding: 3px 10px; border-radius: 12px;
    font-size: 11px; font-weight: 600; margin-right: 8px;
  }}
  .badge.warning {{ background: #5a2500; color: #f0883e; border: 1px solid #7d4220; }}
  .badge.info    {{ background: #1f3a5f; color: #79c0ff; border: 1px solid #1f6feb; }}
  .meta-row {{ margin-top: 10px; color: #8b949e; font-size: 12px; }}
  .meta-row span {{ margin-right: 20px; }}

  /* Cards */
  .cards-grid {{
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 16px; margin-bottom: 32px;
  }}
  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; overflow: hidden; }}
  .card-header {{
    padding: 12px 16px; display: flex;
    justify-content: space-between; align-items: center;
  }}
  .card-dir {{ font-size: 15px; font-weight: 700; letter-spacing: 0.5px; }}
  .card-rank {{ font-size: 11px; opacity: 0.75; }}
  .card-body {{ padding: 16px; }}
  .stat-row {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
  .stat-label {{ color: #8b949e; font-size: 12px; }}
  .stat-val {{
    font-size: 16px; font-weight: 700; padding: 3px 10px;
    border-radius: 6px; background: #21262d;
  }}
  .config-line {{
    margin-top: 12px; padding-top: 10px; border-top: 1px solid #21262d;
    font-size: 11px; color: #8b949e; font-family: monospace;
  }}

  /* Controls */
  .section-header {{
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 14px;
  }}
  .section-title {{ font-size: 16px; font-weight: 600; color: #e6edf3; }}
  .filter-row {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 14px; }}
  .filter-label {{ font-size: 12px; color: #8b949e; margin-right: 4px; }}
  .filter-btn {{
    padding: 5px 14px; border-radius: 20px; border: 1px solid #30363d;
    background: #21262d; color: #8b949e; font-size: 12px; font-weight: 600;
    cursor: pointer; transition: all .15s;
  }}
  .filter-btn:hover {{ border-color: #8b949e; color: #e6edf3; }}
  .filter-btn.active {{ background: #388bfd; color: #fff; border-color: #388bfd; }}

  /* Table */
  .table-section {{ margin-bottom: 40px; }}
  .table-wrapper {{ overflow-x: auto; border: 1px solid #21262d; border-radius: 8px; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th {{ user-select: none; }}
  th:hover {{ color: #e6edf3 !important; background: #30363d !important; }}
  tr.dr-under, tr.dr-over, tr.dr-both {{ transition: opacity .1s; }}
  tr.hidden {{ display: none; }}

  /* Break-even legend */
  .legend {{
    margin-top: 10px; font-size: 11px; color: #8b949e; display: flex; gap: 16px; flex-wrap: wrap;
  }}
  .legend-item {{ display: flex; align-items: center; gap: 5px; }}
  .legend-swatch {{ width: 12px; height: 12px; border-radius: 3px; display: inline-block; }}

  /* Sensitivity section */
  .sensitivity-section {{ margin-bottom: 40px; }}
  .section-subtitle {{ font-size: 12px; color: #8b949e; margin-top: 4px; margin-bottom: 16px; }}

  /* Rec callout */
  .rec-box {{
    background: #1c2333; border: 1px solid #f0883e; border-radius: 8px;
    padding: 16px 20px; margin-bottom: 28px;
  }}
  .rec-box h3 {{ font-size: 13px; color: #f0883e; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px; }}
  .rec-params {{ font-family: monospace; font-size: 14px; color: #e6edf3; }}
  .rec-params span {{ color: #79c0ff; font-weight: 700; }}
</style>
</head>
<body>
<div class="container">

  <!-- Header -->
  <header>
    <h1>🏈 NFL Tackles · Inference Parameter Sweep</h1>
    <div style="margin-top:8px">
      <span class="badge warning">⚠ In-sample — model trained on this data</span>
      <span class="badge info">Break-even @ -110: {be_pct}</span>
    </div>
    <div class="meta-row" style="margin-top:12px">
      <span>Seasons: {meta['train_seasons']}</span>
      <span>Training rows: {meta['n_rows_train']:,}</span>
      <span>Inference rows: {results_n:,}</span>
      <span>Combos: {n_combos}</span>
      <span>Model: OLS market_L16_game_ctx_pos_overprob</span>
      <span>NegBin α: {meta['nb_alpha']}</span>
    </div>
  </header>

  <!-- Production recommendation -->
  {"" if rec_cfg is None else f'''
  <div class="rec-box">
    <h3>★ Recommended Production Config</h3>
    <div class="rec-params">
      direction=<span>{rec_cfg["direction"]}</span> &nbsp;·&nbsp;
      edge≥<span>{rec_cfg["edge"]}</span> &nbsp;·&nbsp;
      lines <span>{rec_cfg["line_min"]}–{rec_cfg["line_max"]}</span> &nbsp;·&nbsp;
      min_books=<span>{rec_cfg["min_books"]:.0f}</span>
      &nbsp;&nbsp;→&nbsp;&nbsp;
      {int(rec_row["n_bets"])} bets &nbsp;·&nbsp;
      {rec_row["hit_rate"]:.1f}% hit rate &nbsp;·&nbsp;
      EV {rec_row["ev_per_unit"]:+.4f}/unit &nbsp;·&nbsp;
      <b>{rec_row["units_won"]:+.2f} units</b> (in-sample)
    </div>
  </div>'''}

  <!-- Analyst pick -->
  {"" if analyst_row is None else f'''
  <div class="rec-box" style="border-color:#58a6ff;margin-top:10px">
    <h3 style="color:#58a6ff">♦ Analyst Pick (OOS-robust)</h3>
    <div class="rec-params">
      direction=<span style="color:#79c0ff">{ar["direction"]}</span> &nbsp;·&nbsp;
      edge≥<span style="color:#79c0ff">{ar["edge"]}</span> &nbsp;·&nbsp;
      lines <span style="color:#79c0ff">{ar["line_min"]}–{ar["line_max"]}</span> &nbsp;·&nbsp;
      min_books=<span style="color:#79c0ff">{ar["min_books"]:.0f}</span>
      &nbsp;&nbsp;→&nbsp;&nbsp;
      {int(analyst_row["n_bets"])} bets &nbsp;·&nbsp;
      {analyst_row["hit_rate"]:.1f}% hit rate &nbsp;·&nbsp;
      EV {analyst_row["ev_per_unit"]:+.4f}/unit &nbsp;·&nbsp;
      <b>{analyst_row["units_won"]:+.2f} units</b> &nbsp;·&nbsp;
      max DD {analyst_row["max_drawdown"]:.1f} units (in-sample)
    </div>
  </div>'''}

  <!-- Summary cards -->
  <div class="cards-grid">{cards_html}</div>

  <!-- Main sweep table -->
  <div class="table-section">
    <div class="section-header">
      <span class="section-title">All Parameter Combinations</span>
      <span style="font-size:12px;color:#8b949e">Sorted by Units Won ↓ &nbsp;·&nbsp; Click headers to re-sort &nbsp;·&nbsp; ★ = data pick &nbsp;·&nbsp; ♦ = analyst pick</span>
    </div>
    <div class="filter-row">
      <span class="filter-label">Direction:</span>
      <button class="filter-btn active" onclick="filterDir('all', this)">All</button>
      <button class="filter-btn" onclick="filterDir('under', this)">UNDER</button>
      <button class="filter-btn" onclick="filterDir('over', this)">OVER</button>
      <button class="filter-btn" onclick="filterDir('both', this)">BOTH</button>
    </div>
    <div class="table-wrapper">{table_html}</div>
    <div class="legend">
      <span class="legend-item"><span class="legend-swatch" style="background:#b02020"></span> Below break-even ({be_pct})</span>
      <span class="legend-item"><span class="legend-swatch" style="background:#3c7a3c"></span> Above break-even</span>
      <span class="legend-item"><span class="legend-swatch" style="background:#1f6feb"></span> Bet volume</span>
      <span class="legend-item" style="color:#8b949e">All hit rates are in-sample and will be lower OOS</span>
    </div>
  </div>

  <!-- Edge sensitivity -->
  <div class="sensitivity-section">
    <div class="section-title">Edge Threshold Sensitivity</div>
    <div class="section-subtitle">Direction = BOTH &nbsp;·&nbsp; Lines 4.5–8.5 &nbsp;·&nbsp; min_books = 3</div>
    {sens_html}
  </div>

</div>

<script>
// ── Column sort ───────────────────────────────────────────────────────────────
let sortDir = {{}};
function sortTable(col) {{
  const tbl = document.getElementById('sweep-table');
  const rows = Array.from(tbl.tBodies[0].rows);
  sortDir[col] = !sortDir[col];
  rows.sort((a, b) => {{
    let va = a.cells[col].innerText.replace('%','').replace('+','');
    let vb = b.cells[col].innerText.replace('%','').replace('+','');
    va = va === '—' ? (sortDir[col] ? Infinity : -Infinity) : parseFloat(va) || va;
    vb = vb === '—' ? (sortDir[col] ? Infinity : -Infinity) : parseFloat(vb) || vb;
    return sortDir[col] ? (va > vb ? 1 : -1) : (va < vb ? 1 : -1);
  }});
  rows.forEach(r => tbl.tBodies[0].appendChild(r));
}}

// ── Direction filter ──────────────────────────────────────────────────────────
function filterDir(dir, btn) {{
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  const rows = document.querySelectorAll('#sweep-table tbody tr');
  rows.forEach(r => {{
    if (dir === 'all') {{ r.classList.remove('hidden'); return; }}
    r.classList.toggle('hidden', !r.classList.contains('dr-' + dir));
  }});
}}
</script>
</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n  Loading artifacts from {ARTIFACT_DIR}...")
    meta      = json.loads((ARTIFACT_DIR / "meta.json").read_text())
    ols       = joblib.load(ARTIFACT_DIR / "ols_pipeline.joblib")
    residuals = np.load(ARTIFACT_DIR / "residuals.npy")
    nb_coefs  = np.load(ARTIFACT_DIR / "nb_coefs.npy")
    nb_alpha  = float(np.load(ARTIFACT_DIR / "nb_alpha.npy")[0])
    artifacts = {"ols": ols, "residuals": residuals,
                 "nb_coefs": nb_coefs, "nb_alpha": nb_alpha}
    print(f"    Trained seasons: {meta['train_seasons']}  |  NegBin α: {meta['nb_alpha']}")

    print(f"\n  Loading labeled dataset...")
    df = pd.read_parquet(LABELED_PATH)
    df = df[df["position"].notna() & ~df["position"].isin(DROP_POSITIONS)].copy()
    df = add_derived(df)
    print(f"    Rows: {len(df):,}")

    print(f"\n  Running inference [bootstrap: {N_BOOT:,} draws]...")
    results = run_inference(df, artifacts)
    scored  = results["ols_pred"].notna().sum()
    print(f"    Scored: {scored:,}")

    print(f"\n  Sweeping {len(DIRECTIONS)} × {len(EDGE_THRESHOLDS)} × "
          f"{len(LINE_RANGES)} × {len(MIN_BOOKS_OPTS)} = "
          f"{len(DIRECTIONS)*len(EDGE_THRESHOLDS)*len(LINE_RANGES)*len(MIN_BOOKS_OPTS)} combos:")
    print(f"    · directions  : {DIRECTIONS}")
    print(f"    · edge thresh : {EDGE_THRESHOLDS}")
    print(f"    · line ranges : {[(f'{a}–{b}') for a, b in LINE_RANGES]}")
    print(f"    · min books   : {MIN_BOOKS_OPTS}")

    rows = []
    for direction, edge, (lmin, lmax), mb in itertools.product(
        DIRECTIONS, EDGE_THRESHOLDS, LINE_RANGES, MIN_BOOKS_OPTS
    ):
        m = sweep_combo(results, direction, edge, lmin, lmax, mb)
        rows.append({"direction": direction, "edge": edge,
                     "line_min": lmin, "line_max": lmax, "min_books": mb, **m})
    sweep = pd.DataFrame(rows)

    print(f"\n  Generating HTML report...")
    html = generate_html(sweep, meta, int(scored))

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"  Saved → {OUT_HTML}")
    print(f"  Open with:  open '{OUT_HTML}'\n")


if __name__ == "__main__":
    main()
