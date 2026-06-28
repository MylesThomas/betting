"""
Compare Logistic Regression, XGBoost, and LightGBM on NFL sacks props.

Same feature set and de-vigged market filter as eval_vs_market.py.
5-fold stratified CV for all models. Reports accuracy metrics and P&L vs market.

Tree models (XGB, LGBM) receive NaN rolling features as-is — they handle
missing natively. LR uses constant imputation (0) + StandardScaler.

Input:  ~/Downloads/tmp/nfl_sacks_features_2025.parquet
Output: ~/Downloads/tmp/nfl_sacks_compare_models.html

Run:
    python src/nfl_sacks_modeling/scripts/compare_models.py
"""

import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
FEATURES    = Path.home() / "Downloads" / "tmp" / "nfl_sacks_features_2025.parquet"
OUT_HTML    = Path.home() / "Downloads" / "tmp" / "nfl_sacks_compare_models.html"


# ── Config / feature lists ─────────────────────────────────────────────────────

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
        # book-level implied (v3)
        "fanduel_over_0p5_implied",
        "betonline_over_0p5_implied", "betonline_under_0p5_implied",
        "draftkings_over_0p25_implied", "draftkings_under_0p25_implied",
    ]
    market_cat = [
        "prop_median_impl_over_bin", "prop_mean_impl_over_bin",
        "prop_median_impl_under_bin", "prop_mean_impl_under_bin",
    ]
    numeric = rolling + ["game_total", "team_spread", "games_played_ytd"] + market_num
    categorical = ["pos_group", "pos_side"] + market_cat
    return numeric, categorical


# ── Data ───────────────────────────────────────────────────────────────────────

def american_to_implied(price: float) -> float:
    if price < 0:
        return abs(price) / (abs(price) + 100)
    return 100 / (price + 100)


def units_on_win(price: float) -> float:
    return 100 / abs(price) if price < 0 else price / 100


def load_data(cfg: dict) -> pd.DataFrame:
    df = pd.read_parquet(FEATURES)
    mask = (
        df["prop_median_price_over"].notna() &
        df["prop_median_price_under"].notna() &
        df["target"].notna()
    )
    df = df[mask].copy()

    impl_over  = df["prop_median_price_over"].apply(american_to_implied)
    impl_under = df["prop_median_price_under"].apply(american_to_implied)
    df["market_prob"] = impl_over / (impl_over + impl_under)

    print(f"Rows: {len(df)}  |  Positive rate: {df['target'].mean():.1%}  "
          f"|  Market avg P(Over): {df['market_prob'].mean():.1%}")
    return df


# ── Pipelines ──────────────────────────────────────────────────────────────────

def lr_pipeline(num_cols: list, cat_cols: list) -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imp",   SimpleImputer(strategy="constant", fill_value=0)),
            ("scale", StandardScaler()),
        ]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])
    return Pipeline([("pre", pre),
                     ("clf", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"))])


def xgb_pipeline(num_cols: list, cat_cols: list) -> Pipeline:
    # XGBoost handles NaN natively; categoricals one-hot encoded
    pre = ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])
    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def lgbm_pipeline(num_cols: list, cat_cols: list) -> Pipeline:
    # LightGBM handles NaN natively; categoricals one-hot encoded for consistency
    pre = ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])
    clf = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


# ── CV + metrics ───────────────────────────────────────────────────────────────

def run_cv(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, label: str) -> np.ndarray:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print(f"  CV: {label}...")
    return cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]


def metrics(y: pd.Series, proba: np.ndarray) -> dict:
    return {
        "brier":   brier_score_loss(y, proba),
        "logloss": log_loss(y, proba),
        "auc":     roc_auc_score(y, proba),
    }


def simulate_pnl(df: pd.DataFrame, model_prob: np.ndarray) -> pd.Series:
    units = []
    for i, row in enumerate(df.itertuples(index=False)):
        if model_prob[i] > row.market_prob:
            u = units_on_win(row.prop_median_price_over) if row.target == 1 else -1.0
        else:
            u = units_on_win(row.prop_median_price_under) if row.target == 0 else -1.0
        units.append(u)
    return pd.Series(units)


# ── HTML ───────────────────────────────────────────────────────────────────────

def build_html(
    df: pd.DataFrame,
    results: dict,          # {label: {"proba": np.ndarray, "metrics": dict, "pnl": pd.Series}}
    market_metrics: dict,
) -> str:
    n = len(df)

    # ── Metrics comparison table ──
    models_ordered = ["Logistic Regression", "XGBoost", "LightGBM"]

    def _fmt(val: float, best: float, lower_is_better: bool) -> str:
        is_best = (val == best)
        style   = "color:green;font-weight:bold" if is_best else ""
        return f'<span style="{style}">{val:.5f}</span>'

    metric_rows = ""
    for metric, label, lower_is_better in [
        ("brier",   "Brier ↓",   True),
        ("logloss", "Log-loss ↓", True),
        ("auc",     "AUC ↑",      False),
    ]:
        all_vals = {m: results[m]["metrics"][metric] for m in models_ordered}
        all_vals["Market"] = market_metrics[metric]
        if lower_is_better:
            best = min(all_vals.values())
        else:
            best = max(all_vals.values())

        cells = "".join(
            f"<td>{_fmt(all_vals[m], best, lower_is_better)}</td>"
            for m in [*models_ordered, "Market"]
        )
        metric_rows += f"<tr><td>{label}</td>{cells}</tr>\n"

    metrics_html = f"""
<div style="margin-bottom:48px;">
  <h2 style="font-family:sans-serif;margin-bottom:4px;">Accuracy Metrics — 5-fold CV (OOS)</h2>
  <p style="font-family:monospace;color:#555;font-size:13px;margin-top:0;">
    Market = de-vigged implied probability (proportional de-vig).
    <b>Green = best across all models + market.</b>
  </p>
  <table style="border-collapse:collapse;font-family:monospace;font-size:14px;">
    <thead>
      <tr style="background:#222;color:white;">
        <th>Metric</th>
        <th>Logistic Regression</th><th>XGBoost</th><th>LightGBM</th><th>Market</th>
      </tr>
    </thead>
    <tbody>{metric_rows}</tbody>
  </table>
</div>"""

    # ── P&L comparison table ──
    pnl_rows = ""
    weeks = sorted(df["week"].unique())
    for wk in weeks:
        mask = df["week"] == wk
        row_cells = f"<td>Wk {int(wk)}</td>"
        for m in models_ordered:
            u = results[m]["pnl"][mask.values].sum()
            color = "green" if u >= 0 else "red"
            row_cells += f"<td><span style='color:{color}'>{u:+.2f}</span></td>"
        pnl_rows += f"<tr>{row_cells}</tr>\n"

    # Totals
    totals_row = "<tr style='background:#eee;border-top:2px solid #333'><td><b>TOTAL</b></td>"
    for m in models_ordered:
        total = results[m]["pnl"].sum()
        roi   = total / n
        color = "green" if total >= 0 else "red"
        totals_row += (
            f"<td><b><span style='color:{color}'>"
            f"{total:+.2f}u ({roi:+.3f}/bet)</span></b></td>"
        )
    totals_row += "</tr>\n"

    pnl_html = f"""
<div style="margin-bottom:48px;">
  <h2 style="font-family:sans-serif;margin-bottom:4px;">Simulated P&amp;L by Week — 1 Unit Flat, Bet Every Row</h2>
  <p style="font-family:monospace;color:#555;font-size:13px;margin-top:0;">
    Bet Over when model_prob &gt; market_prob, else Under. No threshold. n={n:,} bets per model.
  </p>
  <table style="border-collapse:collapse;font-family:monospace;font-size:14px;">
    <thead>
      <tr style="background:#222;color:white;">
        <th>Week</th>
        <th>Logistic Regression</th><th>XGBoost</th><th>LightGBM</th>
      </tr>
    </thead>
    <tbody>{pnl_rows}{totals_row}</tbody>
  </table>
</div>"""

    # ── Edge quintile P&L ──
    edge_rows_html = ""
    for m in models_ordered:
        edge = np.abs(results[m]["proba"] - df["market_prob"].values)
        quintiles = pd.qcut(edge, q=5, labels=False, duplicates="drop")
        for q in sorted(np.unique(quintiles)):
            mask  = quintiles == q
            u     = results[m]["pnl"][mask].sum()
            avg_e = edge[mask].mean()
            color = "green" if u >= 0 else "red"
            edge_rows_html += (
                f"<tr><td>{m}</td><td>Q{q+1}</td>"
                f"<td>{avg_e:.1%}</td><td>{mask.sum()}</td>"
                f"<td><span style='color:{color}'>{u:+.2f}</span></td></tr>\n"
            )

    edge_html = f"""
<div style="margin-bottom:48px;">
  <h2 style="font-family:sans-serif;margin-bottom:4px;">P&amp;L by Edge Size (quintiles of |model − market|)</h2>
  <p style="font-family:monospace;color:#555;font-size:13px;margin-top:0;">
    Real edge shows up as P&amp;L increasing in higher quintiles.
  </p>
  <table style="border-collapse:collapse;font-family:monospace;font-size:14px;">
    <thead>
      <tr style="background:#222;color:white;">
        <th>Model</th><th>Quintile</th><th>Avg |edge|</th><th>n</th><th>Units</th>
      </tr>
    </thead>
    <tbody>{edge_rows_html}</tbody>
  </table>
</div>"""

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>NFL Sacks — Model Comparison</title>
  <style>
    body {{ font-family: monospace; margin: 40px; background: #fafafa; }}
    table td, table th {{ padding: 6px 14px; border-bottom: 1px solid #ddd; text-align: left; }}
    h1 {{ font-family: sans-serif; }}
  </style>
</head>
<body>
  <h1>NFL Sacks Props — Model Comparison (LR vs XGBoost vs LightGBM)</h1>
  {metrics_html}
  {pnl_html}
  {edge_html}
</body>
</html>"""


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    cfg = load_config()
    df  = load_data(cfg)

    y   = df["target"].astype(int)
    num_cols, cat_cols = feature_lists(cfg)
    X   = df[num_cols + cat_cols]

    mkt_proba   = df["market_prob"].values
    mkt_metrics = {
        "brier":   brier_score_loss(y, mkt_proba),
        "logloss": log_loss(y, mkt_proba),
        "auc":     roc_auc_score(y, mkt_proba),
    }
    print(f"\nMarket — Brier: {mkt_metrics['brier']:.5f}  "
          f"LogLoss: {mkt_metrics['logloss']:.5f}  AUC: {mkt_metrics['auc']:.5f}")

    models = {
        "Logistic Regression": lr_pipeline(num_cols, cat_cols),
        "XGBoost":             xgb_pipeline(num_cols, cat_cols),
        "LightGBM":            lgbm_pipeline(num_cols, cat_cols),
    }

    results = {}
    print()
    for label, pipe in models.items():
        proba = run_cv(pipe, X, y, label)
        m     = metrics(y, proba)
        pnl   = simulate_pnl(df, proba)
        results[label] = {"proba": proba, "metrics": m, "pnl": pnl}
        print(f"    Brier: {m['brier']:.5f}  LogLoss: {m['logloss']:.5f}  "
              f"AUC: {m['auc']:.5f}  P&L: {pnl.sum():+.2f}u")

    html = build_html(df, results, mkt_metrics)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"\nReport: {OUT_HTML}")


if __name__ == "__main__":
    main()
