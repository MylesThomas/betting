"""
NFL WR/TE Receiving Yards — Step 5 (OOS) and Step 6 (IS) Grid Search.

New dimensions vs prior sweep (generate_sweep_report.py):
  odds_bucket        all / plus_odds / minus_odds
  shrinkage          0.0 / 0.25 / 0.50 / 0.75
  prediction_method  model / consensus_line

All sweep params loaded from config/model_config.yaml grid_search block.

Shrinkage notes:
  IS  (Step 6): stat-space shrinkage — pulls raw ols_pred toward the IS mean
                before hybrid prob conversion. This is the canonical form.
  OOS (Step 5): probability-space shrinkage — (1−s)*p_model_oos + s*0.5
                because OOF raw ols_pred was not saved. Equivalent intent:
                reduces model confidence symmetrically toward the 50/50 prior.

odds_bucket derived from market_under_prob (novig consensus):
  plus_odds  OVER bet: market_under_prob > 0.5  (under favored → over is dog)
  minus_odds OVER bet: market_under_prob < 0.5  (over favored)
  Flipped for UNDER bets.

avg_odds: approximate from market_under_prob (novig); labeled accordingly.

Run:
  python src/nfl_rec_yards_modeling/scripts/20260709_step5_6_grid_search.py
"""

from __future__ import annotations

import itertools
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.stats import nbinom

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).parents[3]
CONFIG_PATH  = REPO_ROOT / "src" / "nfl_rec_yards_modeling" / "config" / "model_config.yaml"
TMP          = Path.home() / "Downloads" / "tmp"
ARTIFACT_DIR = TMP / "nfl_rec_yards_artifacts"

OOS_PATH  = TMP / "rec_yards_oos_scored.parquet"
IS_PATH   = TMP / "nfl_rec_yards_per_book.parquet"

OUT_OOS_CSV  = TMP / "rec_yards_step5_oos_grid_v2.csv"
OUT_IS_CSV   = TMP / "rec_yards_step6_is_grid_v2.csv"
OUT_HTML     = Path.home() / "Downloads" / "nfl_rec_yards_sweep_v2.html"

N_BOOT = 10_000
RNG    = np.random.default_rng(42)
JUICE  = 110
BREAKEVEN = JUICE / (JUICE + 100)

KEEP_POSITIONS = ["WR", "TE"]
HYBRID_THRESHOLD = 20.5


# ── Load config ────────────────────────────────────────────────────────────────
def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg["nfl_rec_yards_model"]


# ── Load artifacts ─────────────────────────────────────────────────────────────
def load_artifacts() -> dict:
    meta     = json.loads((ARTIFACT_DIR / "meta.json").read_text())
    best_feats = meta["best_feats"]
    return {
        "ols":        joblib.load(ARTIFACT_DIR / "ols_pipeline.joblib"),
        "residuals":  np.load(ARTIFACT_DIR / "residuals.npy"),
        "nb_coefs":   np.load(ARTIFACT_DIR / "nb_coefs.npy"),
        "nb_alpha":   float(np.load(ARTIFACT_DIR / "nb_alpha.npy")[0]),
        "best_feats": best_feats,
    }


# ── Hybrid prob helpers ────────────────────────────────────────────────────────
def _p_bootstrap(yhat: np.ndarray, line: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    samp = RNG.choice(residuals, size=(len(yhat), N_BOOT))
    return ((yhat[:, None] + samp) <= line[:, None]).mean(axis=1)


def _p_negbin(mu: np.ndarray, line: np.ndarray, alpha: float) -> np.ndarray:
    mu     = np.clip(mu, 1e-3, None)
    n_nb   = 1.0 / alpha
    p_nb   = n_nb / (n_nb + mu)
    return nbinom.cdf(np.floor(line).astype(int), n=n_nb, p=p_nb)


def compute_p_hybrid(
    ols_pred: np.ndarray,
    line: np.ndarray,
    artifacts: dict,
    prediction_method: str,
    X_const: np.ndarray | None = None,
) -> np.ndarray:
    """
    Convert raw yhat → P(under line).

    prediction_method='model':          use ols_pred as yhat for bootstrap;
                                        use NB feature prediction for line >= threshold.
    prediction_method='consensus_line': use line itself as yhat for bootstrap;
                                        use NB with mu=line for line >= threshold.
    """
    residuals = artifacts["residuals"]
    nb_alpha  = artifacts["nb_alpha"]
    nb_coefs  = artifacts["nb_coefs"]

    p_out = np.full(len(ols_pred), np.nan)
    low   = line < HYBRID_THRESHOLD
    high  = ~low

    if low.any():
        if prediction_method == "consensus_line":
            yhat_low = line[low]
        else:
            yhat_low = ols_pred[low]
        p_out[low] = _p_bootstrap(yhat_low, line[low], residuals)

    if high.any():
        if prediction_method == "consensus_line":
            mu_high = line[high]
        else:
            if X_const is None:
                raise ValueError("X_const required for model prediction_method at high lines")
            mu_high = np.exp(X_const[high] @ nb_coefs)
        p_out[high] = _p_negbin(mu_high, line[high], nb_alpha)

    return np.clip(p_out, 0.01, 0.99)


# ── odds_bucket helpers ────────────────────────────────────────────────────────
def _odds_bucket_mask(df: pd.DataFrame, bucket: str, direction: str) -> pd.Series:
    """
    Derive plus/minus/all from market_under_prob (novig consensus).

    For OVER bet:
      plus_odds  = under is favored (market_under_prob > 0.5) → over is a dog
      minus_odds = over is favored (market_under_prob < 0.5)

    For UNDER bet: flip.
    """
    if bucket == "all":
        return pd.Series(True, index=df.index)
    mup = df["market_under_prob"]
    if direction == "OVER":
        return mup > 0.5 if bucket == "plus_odds" else mup < 0.5
    if direction == "UNDER":
        return mup < 0.5 if bucket == "plus_odds" else mup > 0.5
    # BOTH: plus_odds = bet is going against the market's dominant side
    return pd.Series(True, index=df.index)


# ── avg odds helper ────────────────────────────────────────────────────────────
def _avg_american_odds(df: pd.DataFrame, direction: str) -> float:
    """Approximate average American odds from novig market_under_prob."""
    if len(df) == 0:
        return np.nan
    if direction in ("UNDER", "BOTH"):
        p = df["market_under_prob"].mean()
    else:
        p = 1 - df["market_under_prob"].mean()
    if p >= 0.5:
        return round(-100 * p / (1 - p), 1)
    else:
        return round(100 * (1 - p) / p, 1)


# ── max drawdown ───────────────────────────────────────────────────────────────
def _max_drawdown(bets: pd.DataFrame) -> float:
    if len(bets) == 0:
        return np.nan
    sort_keys = [c for c in ["season", "week"] if c in bets.columns]
    ordered   = bets.sort_values(sort_keys) if sort_keys else bets
    wins      = ordered["bet_correct"].to_numpy()
    pnl       = np.where(wins == 1, 100 / JUICE, -1.0)
    cumsum    = np.cumsum(pnl)
    peak      = np.maximum.accumulate(cumsum)
    return float((peak - cumsum).max())


# ── sweep one combo ────────────────────────────────────────────────────────────
def sweep_combo(
    df: pd.DataFrame,
    total_universe: int,
    edge_col: str,
    direction: str,
    edge_thresh: float,
    odds_bucket: str,
    line_min: float,
    line_max: float,
    min_books: int,
) -> dict:
    base = (
        df[edge_col].notna() &
        df["offered_line"].between(line_min, line_max)
    )
    if "n_books" in df.columns:
        base &= df["n_books"] >= min_books

    n_scoreable = int(base.sum())

    if direction == "OVER":
        dir_mask = df[edge_col] < 0     # edge < 0 means p_model_under < p_market_under → model favors OVER
        dir_mask &= base
    elif direction == "UNDER":
        dir_mask = df[edge_col] > 0
        dir_mask &= base
    else:
        dir_mask = df[edge_col].abs() > 0
        dir_mask &= base

    dir_mask &= df[edge_col].abs() >= edge_thresh
    dir_mask &= _odds_bucket_mask(df, odds_bucket, direction)

    bets = df[dir_mask]
    if len(bets) == 0:
        return {
            "n_bets": 0, "pct_of_universe": 0.0,
            "win_rate": np.nan, "push_rate": 0.0,
            "units_won": np.nan, "roi": np.nan,
            "avg_odds": np.nan, "max_drawdown": np.nan,
            "mean_edge_pp": np.nan,
        }

    n = len(bets)
    wins   = bets["bet_correct"].sum()
    losses = n - wins
    units_won = wins * (100 / JUICE) - losses
    roi    = units_won / n

    return {
        "n_bets":          n,
        "pct_of_universe": n / total_universe if total_universe > 0 else 0.0,
        "win_rate":        float(bets["bet_correct"].mean()),
        "push_rate":       0.0,
        "units_won":       round(units_won, 3),
        "roi":             round(roi, 4),
        "avg_odds":        _avg_american_odds(bets, direction),
        "max_drawdown":    round(_max_drawdown(bets), 3),
        "mean_edge_pp":    round(float(bets[edge_col].abs().mean() * 100), 2),
    }


# ── build OOS scored frame with shrinkage + prediction_method applied ──────────
def prepare_oos(
    oos_raw: pd.DataFrame,
    artifacts: dict,
    shrinkage: float,
    prediction_method: str,
) -> pd.DataFrame:
    """
    Returns a copy of oos_raw with columns:
      p_model_adj  — adjusted P(under)
      edge_adj     — p_model_adj − market_under_prob
      bet_correct  — 1 if bet correct, 0 if wrong
    """
    df = oos_raw.copy()

    if prediction_method == "model":
        # OOS raw preds not saved — apply shrinkage in probability space
        p_under = df["p_model_oos"].to_numpy(dtype=float)
        if shrinkage > 0:
            p_under = (1 - shrinkage) * p_under + shrinkage * 0.5
            p_under = np.clip(p_under, 0.01, 0.99)
    else:  # consensus_line — no shrinkage (not meaningful)
        line       = df["offered_line"].to_numpy(dtype=float)
        p_under    = compute_p_hybrid(
            ols_pred          = line,
            line              = line,
            artifacts         = artifacts,
            prediction_method = "consensus_line",
            X_const           = None,
        )

    df["p_model_adj"] = p_under
    df["edge_adj"]    = p_under - df["market_under_prob"].to_numpy(dtype=float)

    # bet_correct: 1 if direction matched actual outcome
    # We compute per-row later in the sweep — store actual for lookup
    return df


def add_bet_correct_oos(df: pd.DataFrame, direction: str, edge_col: str) -> pd.DataFrame:
    df = df.copy()
    if direction == "OVER":
        is_bet = df[edge_col] < 0
        df["bet_correct"] = np.where(is_bet, 1 - df["actual_under"], np.nan)
    elif direction == "UNDER":
        is_bet = df[edge_col] > 0
        df["bet_correct"] = np.where(is_bet, df["actual_under"], np.nan)
    else:
        is_over = df[edge_col] < 0
        is_under = df[edge_col] > 0
        df["bet_correct"] = np.where(
            is_over, 1 - df["actual_under"],
            np.where(is_under, df["actual_under"], np.nan),
        )
    return df


# ── build IS scored frame ──────────────────────────────────────────────────────
def prepare_is(
    is_raw: pd.DataFrame,
    artifacts: dict,
    shrinkage: float,
    prediction_method: str,
) -> pd.DataFrame:
    best_feats = artifacts["best_feats"]
    df         = is_raw.copy()
    df["pos_TE"] = (df["position"] == "TE").astype(int)
    df["actual_under"] = (df["receiving_yards"] <= df["offered_line"]).astype(float)

    mask  = df[best_feats].notna().all(axis=1)
    idx   = df.index[mask]
    X     = df.loc[idx, best_feats].to_numpy(dtype=float)
    line  = df.loc[idx, "offered_line"].to_numpy(dtype=float)

    if prediction_method == "model":
        ols_pred = artifacts["ols"].predict(X)
        mean_pred = float(np.mean(ols_pred))
        if shrinkage > 0:
            ols_pred = (1 - shrinkage) * ols_pred + shrinkage * mean_pred
        X_const = np.column_stack([np.ones(len(X)), X])
        p_under = compute_p_hybrid(
            ols_pred          = ols_pred,
            line              = line,
            artifacts         = artifacts,
            prediction_method = "model",
            X_const           = X_const,
        )
    else:  # consensus_line
        p_under = compute_p_hybrid(
            ols_pred          = line,
            line              = line,
            artifacts         = artifacts,
            prediction_method = "consensus_line",
            X_const           = None,
        )

    df.loc[idx, "p_model_adj"] = p_under
    df.loc[idx, "edge_adj"]    = p_under - df.loc[idx, "market_under_prob"].to_numpy()
    return df


def add_bet_correct_is(df: pd.DataFrame, direction: str, edge_col: str) -> pd.DataFrame:
    df = df.copy()
    if direction == "OVER":
        is_bet = df[edge_col] < 0
        df["bet_correct"] = np.where(is_bet, 1 - df["actual_under"], np.nan)
    elif direction == "UNDER":
        is_bet = df[edge_col] > 0
        df["bet_correct"] = np.where(is_bet, df["actual_under"], np.nan)
    else:
        is_over  = df[edge_col] < 0
        is_under = df[edge_col] > 0
        df["bet_correct"] = np.where(
            is_over, 1 - df["actual_under"],
            np.where(is_under, df["actual_under"], np.nan),
        )
    return df


# ── main sweep ─────────────────────────────────────────────────────────────────
def run_sweep(
    mode: str,
    raw: pd.DataFrame,
    artifacts: dict,
    cfg_gs: dict,
) -> pd.DataFrame:
    """
    mode: 'oos' or 'is'
    Returns DataFrame of sweep results.
    """
    edge_thresholds = cfg_gs["edge_threshold"]
    directions      = cfg_gs["direction"]
    odds_buckets    = cfg_gs["odds_bucket"]
    shrinkages      = cfg_gs["shrinkage"]
    pred_methods    = cfg_gs["prediction_method"]
    min_books_list  = cfg_gs["min_books"]
    line_mins       = cfg_gs["line_min"]
    line_maxs       = cfg_gs["line_max"]

    prepare_fn      = prepare_oos if mode == "oos" else prepare_is
    add_correct_fn  = add_bet_correct_oos if mode == "oos" else add_bet_correct_is

    total_rows = len(raw)
    rows = []
    combos_done = 0

    # Pre-compute scored frames per (prediction_method, shrinkage)
    # so we don't redo the expensive bootstrap for every edge/direction combo
    pm_shrink_pairs = list(itertools.product(pred_methods, shrinkages))
    # consensus_line always uses shrinkage=0 (no meaning otherwise)
    pm_shrink_pairs = [
        (pm, s) for (pm, s) in pm_shrink_pairs
        if not (pm == "consensus_line" and s > 0)
    ]
    pm_shrink_pairs = list(dict.fromkeys(pm_shrink_pairs))  # deduplicate

    scored_cache: dict[tuple, pd.DataFrame] = {}
    print(f"  Pre-computing {len(pm_shrink_pairs)} (method, shrinkage) scored frames...")
    for pm, s in pm_shrink_pairs:
        key = (pm, s)
        scored_cache[key] = prepare_fn(raw, artifacts, s, pm)

    n_total_combos = (
        len(pm_shrink_pairs)
        * len(directions)
        * len(edge_thresholds)
        * len(odds_buckets)
        * len(min_books_list)
        * len(line_mins)
        * len(line_maxs)
    )
    print(f"  Running {n_total_combos:,} combos...")

    for (pm, s), direction, edge_thresh, bucket, min_books, line_min, line_max in itertools.product(
        pm_shrink_pairs, directions, edge_thresholds, odds_buckets, min_books_list, line_mins, line_maxs
    ):
        df_scored = scored_cache[(pm, s)].copy()
        df_scored = add_correct_fn(df_scored, direction, "edge_adj")

        stats = sweep_combo(
            df         = df_scored,
            total_universe = total_rows,
            edge_col   = "edge_adj",
            direction  = direction,
            edge_thresh = edge_thresh,
            odds_bucket = bucket,
            line_min   = line_min,
            line_max   = line_max,
            min_books  = min_books,
        )

        rows.append({
            "prediction_method": pm,
            "shrinkage":         s,
            "direction":         direction,
            "edge_threshold":    edge_thresh,
            "odds_bucket":       bucket,
            "min_books":         min_books,
            "line_min":          line_min,
            "line_max":          line_max,
            "clf_threshold":     None,
            **stats,
        })
        combos_done += 1

    result = pd.DataFrame(rows)
    return result.sort_values("units_won", ascending=False).reset_index(drop=True)


# ── HTML rendering ─────────────────────────────────────────────────────────────
def _pct(v: float) -> str:
    return "—" if np.isnan(v) else f"{v:.1%}"

def _f2(v: float) -> str:
    return "—" if np.isnan(v) else f"{v:.2f}"

def _f3(v: float) -> str:
    return "—" if np.isnan(v) else f"{v:.3f}"

def _n(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return str(int(v))

def _odds(v: float) -> str:
    if np.isnan(v):
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


def render_sweep_table(df: pd.DataFrame, label: str) -> str:
    valid = df[df["n_bets"] > 0].copy()
    top   = valid.head(100)

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
        dd_flag = ""
        if not np.isnan(r["max_drawdown"]) and not np.isnan(r["units_won"]):
            if r["max_drawdown"] > r["units_won"] and r["units_won"] > 0:
                dd_flag = ' title="⚠ max drawdown > units won"'

        low_n = r["n_bets"] < 50
        style = "color:#9ca3af" if low_n else ""

        rows_html += f"""
<tr style="font-size:12px;{style}">
  <td style="padding:5px 8px;text-align:center">{r['prediction_method']}</td>
  <td style="padding:5px 8px;text-align:center">{r['shrinkage']:.2f}</td>
  <td style="padding:5px 8px;text-align:center">{r['direction']}</td>
  <td style="padding:5px 8px;text-align:center">{int(r['edge_threshold']*100) if r['edge_threshold'] < 1 else r['edge_threshold']}pp</td>
  <td style="padding:5px 8px;text-align:center">{r['odds_bucket']}</td>
  <td style="padding:5px 8px;text-align:center">{int(r['min_books'])}</td>
  <td style="padding:5px 8px;text-align:center">{r['line_min']:.0f}–{r['line_max']:.0f}</td>
  {_col(r, 'n_bets', lambda v: f"{int(v):,}")}
  {_col(r, 'pct_of_universe', _pct)}
  {_col(r, 'win_rate', _pct)}
  {_col(r, 'units_won', _f2)}
  {_col(r, 'roi', _pct)}
  <td style="padding:5px 8px;text-align:center;font-family:monospace">{_odds(r['avg_odds'])}</td>
  {_col(r, 'max_drawdown', _f2, good_high=False)}{dd_flag}
  <td style="padding:5px 8px;text-align:center;font-family:monospace">{_f2(r['mean_edge_pp'])}pp</td>
</tr>"""

    return f"""
<h3 style="margin:0 0 8px">{label} — Top 100 rows by units won (rows with 0 bets excluded)</h3>
<p style="color:#6b7280;font-size:12px;margin:0 0 12px">
  Grey rows = &lt;50 bets (not statistically meaningful).
  ⚠ tooltip = max drawdown &gt; units won.
  avg_odds is approximate from novig market_under_prob.
</p>
<div style="overflow-x:auto">
<table style="width:100%;border-collapse:collapse;font-size:12px">
<thead><tr style="background:#1d2d44;color:#fff;font-size:11px">
  <th style="padding:7px 8px">Method</th>
  <th style="padding:7px 8px">Shrink</th>
  <th style="padding:7px 8px">Direction</th>
  <th style="padding:7px 8px">Edge ≥</th>
  <th style="padding:7px 8px">Odds Bucket</th>
  <th style="padding:7px 8px">Min Books</th>
  <th style="padding:7px 8px">Line</th>
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


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    cfg    = load_config()
    cfg_gs = cfg["grid_search"]

    print("\n  Loading artifacts...")
    artifacts = load_artifacts()
    print(f"    Best feats  : {artifacts['best_feats']}")
    print(f"    NB alpha    : {artifacts['nb_alpha']:.6f}")

    print("\n  Loading OOS scored data...")
    oos_raw = pd.read_parquet(OOS_PATH)
    oos_raw = oos_raw[oos_raw["position"].isin(KEEP_POSITIONS)].copy()
    print(f"    {len(oos_raw):,} rows  |  seasons {sorted(oos_raw['season'].unique())}")

    print("\n  Loading IS labeled data...")
    is_raw = pd.read_parquet(IS_PATH)
    is_raw = is_raw[is_raw["position"].isin(KEEP_POSITIONS)].copy()
    print(f"    {len(is_raw):,} rows  |  seasons {sorted(is_raw['season'].unique())}")

    # ── Step 5 — OOS sweep ────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  STEP 5 — Out-of-Sample Grid Search")
    print("="*70)
    oos_grid = run_sweep("oos", oos_raw, artifacts, cfg_gs)
    oos_grid.to_csv(OUT_OOS_CSV, index=False)
    print(f"\n  Saved OOS grid → {OUT_OOS_CSV}  ({len(oos_grid):,} rows)")

    top5 = oos_grid[oos_grid["n_bets"] >= 50].head(10)
    print("\n  Top 10 OOS strategies (≥50 bets) by units won:")
    print(top5[["prediction_method","shrinkage","direction","edge_threshold",
                "odds_bucket","n_bets","win_rate","units_won","roi","max_drawdown"]].to_string(index=False))

    # ── Step 6 — IS sweep ─────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  STEP 6 — In-Sample Grid Search")
    print("="*70)
    is_grid = run_sweep("is", is_raw, artifacts, cfg_gs)
    is_grid.to_csv(OUT_IS_CSV, index=False)
    print(f"\n  Saved IS grid  → {OUT_IS_CSV}  ({len(is_grid):,} rows)")

    top6 = is_grid[is_grid["n_bets"] >= 50].head(10)
    print("\n  Top 10 IS strategies (≥50 bets) by units won:")
    print(top6[["prediction_method","shrinkage","direction","edge_threshold",
                "odds_bucket","n_bets","win_rate","units_won","roi","max_drawdown"]].to_string(index=False))

    # ── HTML report ───────────────────────────────────────────────────────────
    import datetime, subprocess
    ts = subprocess.check_output(
        ["bash","-c","TZ=America/New_York date '+%Y-%m-%d %H:%M:%S ET'"]
    ).decode().strip()

    n_oos_valid = int((oos_grid["n_bets"] >= 50).sum())
    n_is_valid  = int((is_grid["n_bets"] >= 50).sum())

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NFL Rec Yards — Strategy Sweep v2</title>
<style>
  body {{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Arial,sans-serif;
        background:#f4f4f5;margin:0;padding:16px;font-size:13px}}
  .card {{background:#fff;border-radius:8px;border:1px solid #e2e2e4;
          padding:20px;margin-bottom:16px}}
  h2 {{font-size:18px;margin:0 0 4px}}
  h3 {{font-size:15px;margin:12px 0 4px}}
  p  {{color:#6b7280;font-size:12px;margin:4px 0}}
  .note {{background:#fffbeb;border:1px solid #fcd34d;border-radius:6px;
          padding:10px 14px;margin-bottom:12px;font-size:12px}}
</style>
</head>
<body>
<div class="card">
  <h2>NFL Receiving Yards — Strategy Sweep v2</h2>
  <p>Generated: {ts}</p>
  <p>New dimensions: <b>odds_bucket</b> | <b>shrinkage</b> | <b>prediction_method</b></p>
  <p>OOS: {len(oos_grid):,} combos, {n_oos_valid:,} with ≥50 bets &nbsp;|&nbsp;
     IS: {len(is_grid):,} combos, {n_is_valid:,} with ≥50 bets</p>
  <div class="note">
    <b>OOS shrinkage note:</b> OOF raw ols_pred was not saved; OOS shrinkage is applied in
    probability space: <code>(1−s) × p_model_oos + s × 0.5</code>.
    IS shrinkage is applied in stat space (canonical): <code>(1−s) × ols_pred + s × mean(ols_pred)</code>
    before hybrid prob conversion. Shrinkage=0 rows are identical to the prior sweep.<br><br>
    <b>avg_odds note:</b> Approximate from novig market_under_prob. Actual vig-inclusive odds differ.
  </div>
</div>

<div class="card">
  <h2>Step 5 — Out-of-Sample Results</h2>
  <p>Seasons 2023–2025 | OOF predictions | {len(oos_raw):,} scored rows</p>
  {render_sweep_table(oos_grid, "OOS")}
</div>

<div class="card">
  <h2>Step 6 — In-Sample Results</h2>
  <p>Seasons 2023–2025 | IS predictions (⚠ model trained on this data) | {len(is_raw):,} scored rows</p>
  {render_sweep_table(is_grid, "IS")}
</div>

</body>
</html>"""

    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"\n  Report → {OUT_HTML}")
    print(f"  Open:   open {OUT_HTML}")
    print()


if __name__ == "__main__":
    main()
