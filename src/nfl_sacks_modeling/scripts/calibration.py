"""
Calibration analysis for NFL sacks props model — LR and XGB only.

Uncalibrated:    S3 cross-season OOS (train on season A, test on season B)
Platt calibrated: sigmoid fit on training season's 5-fold CV probas → applied to test season
                  (no leakage — calibration layer never sees test-season labels)

Sections:
  1. Reliability diagrams (before/after Platt) + probability distributions
  2. Decision zone table (0.10–0.50, 0.05-wide bins)
  3. Brier scores — model vs market vs always-mean baseline
  4. Strategy impact — optimal Under-only threshold before vs after Platt

Output: ~/Downloads/tmp/nfl_sacks_calibration.html

Run:
    python src/nfl_sacks_modeling/scripts/calibration.py
"""

import base64
import io
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CONFIG_PATH   = Path(__file__).resolve().parents[1] / "config.yaml"
FEATURES_2024 = Path.home() / "Downloads" / "tmp" / "nfl_sacks_features_2024.parquet"
FEATURES_2025 = Path.home() / "Downloads" / "tmp" / "nfl_sacks_features_2025.parquet"
OUT_HTML      = Path.home() / "Downloads" / "tmp" / "nfl_sacks_calibration.html"

THRESHOLDS   = np.round(np.arange(0.05, 0.80, 0.05), 2)
MIN_BETS     = 20
MODEL_LABELS = ["LR", "XGB"]

ACTUAL_RATE  = 0.248   # ~24.8% positive rate across full dataset


# ── Config / features ──────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)["nfl_sacks_model"]


def feature_lists(cfg: dict) -> tuple[list[str], list[str]]:
    windows = cfg["rolling_windows"]
    rolling = [
        f"{feat}_L{('career' if w >= 999 else w)}"
        for feat in ["sack_rate", "qbhit_rate", "snap_pct"]
        for w in windows
    ]
    market_num = [
        "prop_median_impl_over", "prop_median_impl_under",
        "prop_mean_impl_over",   "prop_mean_impl_under",
        "prop_min_impl_over",    "prop_max_impl_over",
        "prop_min_impl_under",   "prop_max_impl_under",
        "prop_book_spread_over", "prop_book_spread_under",
        "prop_n_books",
        "fanduel_over_0p5_implied",
        "betonline_over_0p5_implied", "betonline_under_0p5_implied",
        "draftkings_over_0p25_implied", "draftkings_under_0p25_implied",
    ]
    market_cat = [
        "prop_median_impl_over_bin", "prop_mean_impl_over_bin",
        "prop_median_impl_under_bin", "prop_mean_impl_under_bin",
    ]
    numeric    = rolling + ["game_total", "team_spread", "games_played_ytd"] + market_num
    categorical = ["pos_group", "pos_side"] + market_cat
    return numeric, categorical


# ── Data ───────────────────────────────────────────────────────────────────────

def american_to_implied(price: float) -> float:
    return abs(price) / (abs(price) + 100) if price < 0 else 100 / (price + 100)


def units_on_win(price: float) -> float:
    return 100 / abs(price) if price < 0 else price / 100


def load_season(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    mask = (
        df["prop_median_price_over"].notna() &
        df["prop_median_price_under"].notna() &
        df["target"].notna()
    )
    df = df[mask].copy()
    impl_over  = df["prop_median_price_over"].apply(american_to_implied)
    impl_under = df["prop_median_price_under"].apply(american_to_implied)
    df["market_prob"] = impl_over / (impl_over + impl_under)
    return df.reset_index(drop=True)


# ── Pipelines ──────────────────────────────────────────────────────────────────

def lr_pipeline(num_cols: list, cat_cols: list) -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value=0)),
                          ("sc",  StandardScaler())]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])
    return Pipeline([("pre", pre),
                     ("clf", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"))])


def xgb_pipeline(num_cols: list, cat_cols: list) -> Pipeline:
    pre = ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])
    return Pipeline([("pre", pre),
                     ("clf", xgb.XGBClassifier(n_estimators=300, max_depth=4,
                                               learning_rate=0.05, subsample=0.8,
                                               colsample_bytree=0.8, eval_metric="logloss",
                                               random_state=42, verbosity=0))])


def make_pipeline(label: str, num_cols: list, cat_cols: list) -> Pipeline:
    if label == "LR":  return lr_pipeline(num_cols, cat_cols)
    if label == "XGB": return xgb_pipeline(num_cols, cat_cols)
    raise ValueError(label)


# ── Calibration ────────────────────────────────────────────────────────────────

def fit_platt(oos_proba: np.ndarray, y: np.ndarray) -> LogisticRegression:
    platt = LogisticRegression(C=1e10, max_iter=1000, solver="lbfgs")
    platt.fit(oos_proba.reshape(-1, 1), y)
    return platt


def apply_platt(platt: LogisticRegression, proba: np.ndarray) -> np.ndarray:
    return platt.predict_proba(proba.reshape(-1, 1))[:, 1]


def get_probas(
    label: str, n_cols: list, c_cols: list,
    df24: pd.DataFrame, df25: pd.DataFrame,
    n_inner: int = 5,
) -> dict:
    """
    Returns uncalibrated and Platt-calibrated S3 OOS probabilities.

    For each direction (train A → test B):
      1. Fit base model on season A, predict season B (uncalibrated)
      2. Get season A 5-fold CV proba, fit Platt scaler on those
      3. Apply Platt to season B predictions (calibrated)
    """
    pipe = make_pipeline(label, n_cols, c_cols)
    X24  = df24[n_cols + c_cols]
    y24  = df24["target"].astype(int).values
    X25  = df25[n_cols + c_cols]
    y25  = df25["target"].astype(int).values
    inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=42)

    print(f"  [{label}] Train 2025 → Test 2024 ...")
    p_a = clone(pipe)
    p_a.fit(X25, y25)
    uncal_24  = p_a.predict_proba(X24)[:, 1]
    oos_25    = cross_val_predict(clone(pipe), X25, y25, cv=inner_cv, method="predict_proba")[:, 1]
    platt_a   = fit_platt(oos_25, y25)
    cal_24    = apply_platt(platt_a, uncal_24)

    print(f"  [{label}] Train 2024 → Test 2025 ...")
    p_b = clone(pipe)
    p_b.fit(X24, y24)
    uncal_25  = p_b.predict_proba(X25)[:, 1]
    oos_24    = cross_val_predict(clone(pipe), X24, y24, cv=inner_cv, method="predict_proba")[:, 1]
    platt_b   = fit_platt(oos_24, y24)
    cal_25    = apply_platt(platt_b, uncal_25)

    # S3 combined (2024 rows first, then 2025 — consistent with df_oos ordering)
    return {
        "uncal":  np.concatenate([uncal_24, uncal_25]),
        "cal":    np.concatenate([cal_24,   cal_25]),
        "y":      np.concatenate([y24,      y25]),
        "market": np.concatenate([df24["market_prob"].values, df25["market_prob"].values]),
        "p_over": np.concatenate([df24["prop_median_price_over"].values,
                                  df25["prop_median_price_over"].values]),
        "p_under":np.concatenate([df24["prop_median_price_under"].values,
                                  df25["prop_median_price_under"].values]),
    }


# ── Strategy helpers ───────────────────────────────────────────────────────────

def under_only_result(proba: np.ndarray, y: np.ndarray,
                      price_under: np.ndarray, x: float) -> dict:
    units = [units_on_win(price_under[i]) if y[i] == 0 else -1.0
             for i in range(len(y)) if proba[i] < x]
    n = len(units)
    total = sum(units)
    return {"n": n, "units": total, "roi": total / n if n else np.nan}


def best_x(proba: np.ndarray, y: np.ndarray, price_under: np.ndarray) -> float:
    best, best_u = THRESHOLDS[0], -np.inf
    for x in THRESHOLDS:
        r = under_only_result(proba, y, price_under, x)
        if r["n"] >= MIN_BETS and r["units"] > best_u:
            best_u, best = r["units"], x
    return best


# ── Plotting ───────────────────────────────────────────────────────────────────

COLORS = {
    "uncal":   ("#1f77b4", "-",  "Uncalibrated"),
    "cal":     ("#ff7f0e", "-",  "Platt calibrated"),
    "market":  ("#888888", "--", "Market implied"),
    "perfect": ("black",   ":",  "Perfect calibration"),
}


def make_figure(label: str, data: dict) -> str:
    y      = data["y"]
    uncal  = data["uncal"]
    cal    = data["cal"]
    market = data["market"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{label} — Calibration (S3 cross-season OOS, n={len(y):,})",
                 fontsize=13, fontweight="bold")
    fig.patch.set_facecolor("white")

    # ── Left: reliability diagram ──────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("white")
    lo, hi = 0.10, 0.55
    ax.plot([lo, hi], [lo, hi], color="black", linestyle=":", linewidth=1.2,
            alpha=0.6, label="Perfect calibration")
    ax.axhline(y=ACTUAL_RATE, color="red", linewidth=1, linestyle="--",
               alpha=0.5, label=f"Overall actual rate ({ACTUAL_RATE:.1%})")

    for key, proba in [("uncal", uncal), ("cal", cal), ("market", market)]:
        color, ls, lbl = COLORS[key]
        try:
            frac_pos, mean_pred = calibration_curve(y, proba, n_bins=10, strategy="quantile")
            ax.plot(mean_pred, frac_pos, color=color, linestyle=ls, linewidth=2,
                    marker="o", markersize=5, label=lbl)
        except Exception:
            pass

    ax.set_xlim([lo, hi])
    ax.set_ylim([0.05, 0.50])
    ax.set_xlabel("Mean predicted probability", fontsize=11)
    ax.set_ylabel("Actual positive rate (sack ≥ 1)", fontsize=11)
    ax.set_title("Reliability Diagram", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)

    # ── Right: probability distribution ───────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("white")
    bins = np.arange(0.05, 0.65, 0.025)

    ax2.hist(uncal,  bins=bins, alpha=0.55, color="#1f77b4", density=True, label="Uncalibrated")
    ax2.hist(cal,    bins=bins, alpha=0.55, color="#ff7f0e", density=True, label="Platt calibrated")
    ax2.hist(market, bins=bins, alpha=0.35, color="#888888", density=True, label="Market implied")

    ax2.axvline(x=0.30, color="red",   linewidth=1.8, linestyle="--", label="x=0.30 threshold")
    ax2.axvline(x=ACTUAL_RATE, color="darkred", linewidth=1, linestyle=":",
                label=f"Actual rate ({ACTUAL_RATE:.1%})")

    ax2.set_xlabel("Predicted probability", fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.set_title("Probability Distribution", fontsize=11)
    ax2.set_xlim([0.05, 0.60])
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor="white")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close("all")
    return b64


# ── Decision zone table ────────────────────────────────────────────────────────

def build_decision_zone(label: str, data: dict) -> str:
    y      = data["y"]
    uncal  = data["uncal"]
    cal    = data["cal"]
    market = data["market"]
    bins   = np.arange(0.10, 0.55, 0.05)

    rows_html = ""
    for lo in bins:
        hi = lo + 0.05
        for tag, proba in [("uncal", uncal), ("cal", cal)]:
            mask = (proba >= lo) & (proba < hi)
            n    = mask.sum()
            if n == 0:
                continue
            mean_pred = proba[mask].mean()
            mean_mkt  = market[mask].mean()
            actual    = y[mask].mean()
            gap_model = mean_pred - actual
            gap_mkt   = mean_mkt  - actual

            is_under_zone = hi <= 0.30
            bg = "background:#e8f5e9;" if (is_under_zone and tag == "uncal") else ""

            def _gap_style(g):
                if g > 0.03:  return "color:#c62828;"
                if g < -0.03: return "color:#1565c0;"
                return "color:#2e7d32;"

            tag_label = "Uncal" if tag == "uncal" else "Platt"
            rows_html += (
                f"<tr style='{bg}'>"
                f"<td style='padding:4px 8px;font-weight:bold'>{lo:.2f}–{hi:.2f}</td>"
                f"<td style='padding:4px 8px;color:#666'>{tag_label}</td>"
                f"<td style='padding:4px 8px;text-align:right'>{n}</td>"
                f"<td style='padding:4px 8px;text-align:right'>{mean_pred:.3f}</td>"
                f"<td style='padding:4px 8px;text-align:right'>{mean_mkt:.3f}</td>"
                f"<td style='padding:4px 8px;text-align:right'>{actual:.3f}</td>"
                f"<td style='padding:4px 8px;text-align:right;{_gap_style(gap_model)}'>"
                f"{gap_model:+.3f}</td>"
                f"<td style='padding:4px 8px;text-align:right;{_gap_style(gap_mkt)}'>"
                f"{gap_mkt:+.3f}</td>"
                f"</tr>"
            )

    return (
        f"<div style='margin-top:20px;margin-bottom:8px;'>"
        f"<h3 style='font-family:sans-serif;margin-bottom:6px'>{label} — Decision Zone Table</h3>"
        f"<p style='font-family:monospace;color:#555;font-size:12px;max-width:800px;margin-bottom:8px;'>"
        f"Green highlight = Under-bet rows (proba &lt; 0.30). "
        f"Gap = mean_predicted − actual_rate. Positive gap = model over-estimates P(Over).</p>"
        f"<div style='overflow-x:auto;'>"
        f"<table style='border-collapse:collapse;font-family:monospace;font-size:12px;'>"
        f"<thead><tr style='background:#111;color:white'>"
        f"<th style='padding:5px 8px'>Bin</th><th style='padding:5px 8px'>Version</th>"
        f"<th style='padding:5px 8px'>n</th><th style='padding:5px 8px'>Pred</th>"
        f"<th style='padding:5px 8px'>Market</th><th style='padding:5px 8px'>Actual</th>"
        f"<th style='padding:5px 8px'>Gap (model)</th><th style='padding:5px 8px'>Gap (mkt)</th>"
        f"</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        f"</table></div></div>"
    )


# ── Brier scores ───────────────────────────────────────────────────────────────

def build_brier_table(all_data: dict) -> str:
    rows = ""
    for label in MODEL_LABELS:
        data = all_data[label]
        y    = data["y"]
        for tag, proba in [("Uncalibrated", data["uncal"]), ("Platt calibrated", data["cal"])]:
            bs = brier_score_loss(y, proba)
            rows += (
                f"<tr><td style='padding:4px 10px;font-weight:bold'>{label}</td>"
                f"<td style='padding:4px 10px'>{tag}</td>"
                f"<td style='padding:4px 10px;text-align:right'>{bs:.5f}</td></tr>"
            )

    # Market and always-mean baselines (same y for all, just use first model's)
    y_ref = all_data[MODEL_LABELS[0]]["y"]
    mkt   = all_data[MODEL_LABELS[0]]["market"]
    bs_mkt  = brier_score_loss(y_ref, mkt)
    bs_mean = brier_score_loss(y_ref, np.full_like(y_ref, y_ref.mean(), dtype=float))
    rows += (
        f"<tr style='border-top:2px solid #555'>"
        f"<td style='padding:4px 10px;color:#555'>Market</td>"
        f"<td style='padding:4px 10px;color:#555'>Implied prob</td>"
        f"<td style='padding:4px 10px;text-align:right;color:#555'>{bs_mkt:.5f}</td></tr>"
        f"<tr>"
        f"<td style='padding:4px 10px;color:#555'>Baseline</td>"
        f"<td style='padding:4px 10px;color:#555'>Always predict mean</td>"
        f"<td style='padding:4px 10px;text-align:right;color:#555'>{bs_mean:.5f}</td></tr>"
    )
    return (
        "<div style='margin-bottom:32px;'>"
        "<h2 style='font-family:sans-serif'>Brier Scores</h2>"
        "<p style='font-family:monospace;color:#444;font-size:12px;margin-bottom:8px;'>"
        "Lower = better. Market baseline uses implied probability directly as a prediction.</p>"
        "<table style='border-collapse:collapse;font-family:monospace;font-size:13px;'>"
        "<thead><tr style='background:#333;color:white'>"
        "<th style='padding:6px 10px;text-align:left'>Model</th>"
        "<th style='padding:6px 10px;text-align:left'>Version</th>"
        "<th style='padding:6px 10px'>Brier Score</th>"
        "</tr></thead>"
        f"<tbody>{rows}</tbody>"
        "</table></div>"
    )


# ── Strategy comparison before/after Platt ─────────────────────────────────────

def build_strategy_table(all_data: dict) -> str:
    rows = ""
    for label in MODEL_LABELS:
        data  = all_data[label]
        y     = data["y"]
        pu    = data["p_under"]
        for tag, proba in [("Uncalibrated", data["uncal"]), ("Platt calibrated", data["cal"])]:
            bx   = best_x(proba, y, pu)
            r    = under_only_result(proba, y, pu, bx)
            r30  = under_only_result(proba, y, pu, 0.30)
            med_p = np.median(proba)
            pct_under = (proba < 0.30).mean()

            rows += (
                f"<tr>"
                f"<td style='padding:4px 10px;font-weight:bold'>{label}</td>"
                f"<td style='padding:4px 10px'>{tag}</td>"
                f"<td style='padding:4px 10px;text-align:right'>{med_p:.3f}</td>"
                f"<td style='padding:4px 10px;text-align:right'>{pct_under:.1%}</td>"
                f"<td style='padding:4px 10px;text-align:right;font-weight:bold'>{bx:.2f}</td>"
                f"<td style='padding:4px 10px;text-align:right'>{r['n']}</td>"
                f"<td style='padding:4px 10px;text-align:right;"
                f"color:{'green' if r['units']>0 else 'red'};font-weight:bold'>"
                f"{r['units']:+.1f}u</td>"
                f"<td style='padding:4px 10px;text-align:right'>{r['roi']:+.2%}</td>"
                f"<td style='padding:4px 10px;text-align:right;color:#555'>"
                f"{r30['units']:+.1f}u (n={r30['n']})</td>"
                f"</tr>"
            )

    # Always Under baseline
    y_ref = all_data[MODEL_LABELS[0]]["y"]
    pu    = all_data[MODEL_LABELS[0]]["p_under"]
    always_u = [units_on_win(pu[i]) if y_ref[i] == 0 else -1.0 for i in range(len(y_ref))]
    n_au = len(always_u)
    u_au = sum(always_u)
    rows += (
        f"<tr style='border-top:2px solid #555;color:#555'>"
        f"<td style='padding:4px 10px'>Baseline</td>"
        f"<td style='padding:4px 10px'>Always Under</td>"
        f"<td colspan='2' style='padding:4px 10px'></td>"
        f"<td style='padding:4px 10px;text-align:right'>—</td>"
        f"<td style='padding:4px 10px;text-align:right'>{n_au}</td>"
        f"<td style='padding:4px 10px;text-align:right;color:green;font-weight:bold'>"
        f"{u_au:+.1f}u</td>"
        f"<td style='padding:4px 10px;text-align:right'>{u_au/n_au:+.2%}</td>"
        f"<td style='padding:4px 10px'></td>"
        f"</tr>"
    )

    return (
        "<div style='margin-bottom:48px;'>"
        "<h2 style='font-family:sans-serif'>Strategy Impact — Under-Only Before vs After Platt</h2>"
        "<p style='font-family:monospace;color:#444;font-size:12px;max-width:900px;margin-bottom:8px;line-height:1.6'>"
        "Best x = threshold that maximises total units for Under-only strategy on S3 OOS data. "
        "'At x=0.30' column shows performance at the fixed threshold from oos_eval for reference. "
        "Median prob and %&lt;0.30 show how Platt scaling shifts the probability distribution.</p>"
        "<div style='overflow-x:auto;'>"
        "<table style='border-collapse:collapse;font-family:monospace;font-size:12px;'>"
        "<thead><tr style='background:#111;color:white'>"
        "<th style='padding:5px 10px;text-align:left'>Model</th>"
        "<th style='padding:5px 10px;text-align:left'>Version</th>"
        "<th style='padding:5px 10px'>Median p</th>"
        "<th style='padding:5px 10px'>% &lt; 0.30</th>"
        "<th style='padding:5px 10px'>Best x</th>"
        "<th style='padding:5px 10px'>n bets</th>"
        "<th style='padding:5px 10px'>Units</th>"
        "<th style='padding:5px 10px'>ROI</th>"
        "<th style='padding:5px 10px'>At x=0.30</th>"
        "</tr></thead>"
        f"<tbody>{rows}</tbody>"
        "</table></div></div>"
    )


# ── TLDR ───────────────────────────────────────────────────────────────────────

def build_tldr(all_data: dict) -> str:
    lines = []
    for label in MODEL_LABELS:
        data = all_data[label]
        y, uncal, cal = data["y"], data["uncal"], data["cal"]
        bs_uncal = brier_score_loss(y, uncal)
        bs_cal   = brier_score_loss(y, cal)
        delta    = bs_uncal - bs_cal
        med_shift = np.median(cal) - np.median(uncal)
        lines.append(
            f"<b>{label}</b>: Brier {bs_uncal:.4f} → {bs_cal:.4f} after Platt "
            f"({'better' if delta > 0 else 'worse'} by {abs(delta):.4f}); "
            f"median prob shift {med_shift:+.3f}"
        )
    return (
        "<div style='font-family:monospace;background:#e8f5e9;padding:16px 20px;"
        "border-radius:8px;margin-bottom:32px;font-size:13px;"
        "border-left:4px solid #2ca02c;max-width:950px;line-height:1.9;'>"
        "<b style='font-size:15px'>TLDR</b><br><br>"
        + "<br>".join(lines)
        + "<br><br>"
        "<b>What to look for:</b> Reliability diagram should show model curve closer to the "
        "diagonal after Platt scaling. In the Under-bet zone (proba &lt; 0.30), the key question "
        "is whether actual P(Over) ≈ 22-25% — if so, the Under bets are well-targeted. "
        "A large median prob shift after Platt means the raw scores were biased; a small shift "
        "means the model was already reasonably calibrated."
        "</div>"
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    warnings.filterwarnings("ignore")

    cfg = load_config()
    num_cols, cat_cols = feature_lists(cfg)

    print("Loading data...")
    df24 = load_season(FEATURES_2024)
    df25 = load_season(FEATURES_2025)
    n_cols = [c for c in num_cols if c in df24.columns and c in df25.columns]
    c_cols = [c for c in cat_cols if c in df24.columns and c in df25.columns]
    print(f"  2024: {len(df24)} rows | 2025: {len(df25)} rows")
    print(f"  Features: {len(n_cols)} numeric, {len(c_cols)} categorical")

    all_data = {}
    for label in MODEL_LABELS:
        print(f"\n{label}:")
        all_data[label] = get_probas(label, n_cols, c_cols, df24, df25)

    print("\nBuilding HTML...")

    tldr_html    = build_tldr(all_data)
    brier_html   = build_brier_table(all_data)
    strategy_html = build_strategy_table(all_data)

    model_sections = []
    for label in MODEL_LABELS:
        fig_b64  = make_figure(label, all_data[label])
        dz_html  = build_decision_zone(label, all_data[label])
        img_tag  = f"<img src='data:image/png;base64,{fig_b64}' style='max-width:100%;margin-bottom:8px;'>"
        model_sections.append(
            f"<div style='margin-bottom:48px;'>"
            f"<h2 style='font-family:sans-serif;margin-bottom:8px'>{label} — Reliability Diagram + Distribution</h2>"
            f"{img_tag}{dz_html}</div>"
        )

    html = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>NFL Sacks — Calibration Analysis</title>
  <style>
    body { font-family: monospace; margin: 40px; background: #fafafa; }
    h1,h2,h3 { font-family: sans-serif; }
    table td, table th { border-bottom: 1px solid #e0e0e0; vertical-align: middle; }
    .divider { border-top: 3px solid #333; margin: 44px 0; }
  </style>
</head>
<body>
<h1>NFL Sacks Props — Calibration Analysis</h1>
<p style="font-family:monospace;color:#444;font-size:13px;max-width:900px;line-height:1.6;margin-bottom:24px;">
  <b>Method:</b> S3 cross-season OOS — each row predicted by a model that never saw its season.
  Platt scaling: sigmoid fit on training season's 5-fold CV probabilities → applied to test season predictions.
  No leakage: calibration layer fit entirely on training-season data.
  LR and XGB only (LGBM dropped — near-identical to XGB).
</p>
""" + tldr_html + brier_html + "<div class='divider'></div>" + "\n<div class='divider'></div>\n".join(model_sections) + "<div class='divider'></div>" + strategy_html + """
</body>
</html>"""

    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"\nReport: {OUT_HTML}")

    print(f"\n{'='*55}")
    for label in MODEL_LABELS:
        data = all_data[label]
        y, uncal, cal = data["y"], data["uncal"], data["cal"]
        bx_uncal = best_x(uncal, y, data["p_under"])
        bx_cal   = best_x(cal,   y, data["p_under"])
        r_uncal  = under_only_result(uncal, y, data["p_under"], bx_uncal)
        r_cal    = under_only_result(cal,   y, data["p_under"], bx_cal)
        print(f"  {label} uncal: best x={bx_uncal:.2f}  {r_uncal['units']:+.1f}u  "
              f"median_p={np.median(uncal):.3f}")
        print(f"  {label} cal  : best x={bx_cal:.2f}  {r_cal['units']:+.1f}u  "
              f"median_p={np.median(cal):.3f}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
