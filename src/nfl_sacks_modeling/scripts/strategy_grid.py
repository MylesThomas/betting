"""
Strategy grid search for NFL sacks props.

Five betting strategies, three models, threshold x swept from 0.05 → 0.95.

Strategies (x = model probability threshold):
  1. Both      — bet Over if model_prob >= x, else bet Under  (every row)
  2. Over only — bet Over if model_prob >= x                  (skip rows below x)
  3. Under only— bet Under if model_prob < x                  (skip rows above x)
  4. Edge vs market — bet Over if model_prob > market_prob,   (every row, no x)
                       else bet Under
  5. Always Under   — always bet Under                        (every row, no x)

Models: Logistic Regression, XGBoost, LightGBM (same CV probabilities as compare_models.py).
Filter: both Over and Under prices required (removes FanDuel / one-sided rows).

Output: ~/Downloads/tmp/nfl_sacks_strategy_grid.html

Run:
    python src/nfl_sacks_modeling/scripts/strategy_grid.py
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
FEATURES    = Path.home() / "Downloads" / "tmp" / "nfl_sacks_features_2025.parquet"
OUT_HTML    = Path.home() / "Downloads" / "tmp" / "nfl_sacks_strategy_grid.html"

THRESHOLDS  = np.round(np.arange(0.05, 1.00, 0.05), 2)
MIN_BETS    = 20   # cells with fewer bets are dimmed


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


# ── Models ─────────────────────────────────────────────────────────────────────

def lr_pipeline(num_cols, cat_cols):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value=0)),
                          ("sc",  StandardScaler())]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])
    return Pipeline([("pre", pre),
                     ("clf", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"))])


def xgb_pipeline(num_cols, cat_cols):
    pre = ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])
    return Pipeline([("pre", pre),
                     ("clf", xgb.XGBClassifier(n_estimators=300, max_depth=4,
                                               learning_rate=0.05, subsample=0.8,
                                               colsample_bytree=0.8, eval_metric="logloss",
                                               random_state=42, verbosity=0))])


def lgbm_pipeline(num_cols, cat_cols):
    pre = ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])
    return Pipeline([("pre", pre),
                     ("clf", lgb.LGBMClassifier(n_estimators=300, max_depth=4,
                                                learning_rate=0.05, subsample=0.8,
                                                colsample_bytree=0.8, random_state=42,
                                                verbosity=-1))])


def get_cv_proba(pipe, X, y, label) -> np.ndarray:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print(f"  CV: {label}...")
    return cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]


# ── Strategy evaluation ────────────────────────────────────────────────────────

def eval_threshold_strategy(
    strat: int,
    x: float,
    model_prob: np.ndarray,
    market_prob: np.ndarray,
    y: np.ndarray,
    price_over: np.ndarray,
    price_under: np.ndarray,
) -> dict:
    units = []
    for i in range(len(y)):
        mp = model_prob[i]

        if strat == 1:    # Both — always bet
            bet_over = mp >= x
        elif strat == 2:  # Over only
            if mp < x:
                continue
            bet_over = True
        elif strat == 3:  # Under only
            if mp >= x:
                continue
            bet_over = False
        else:
            raise ValueError(f"strat must be 1-3 for threshold strategies, got {strat}")

        if bet_over:
            u = units_on_win(price_over[i]) if y[i] == 1 else -1.0
        else:
            u = units_on_win(price_under[i]) if y[i] == 0 else -1.0
        units.append(u)

    n      = len(units)
    total  = sum(units)
    return {"n": n, "units": total, "roi": total / n if n else np.nan}


def eval_fixed_strategy(
    strat: int,
    model_prob: np.ndarray,
    market_prob: np.ndarray,
    y: np.ndarray,
    price_over: np.ndarray,
    price_under: np.ndarray,
) -> dict:
    units = []
    for i in range(len(y)):
        if strat == 4:    # Edge vs market
            bet_over = model_prob[i] > market_prob[i]
        elif strat == 5:  # Always Under
            bet_over = False
        else:
            raise ValueError(strat)

        if bet_over:
            u = units_on_win(price_over[i]) if y[i] == 1 else -1.0
        else:
            u = units_on_win(price_under[i]) if y[i] == 0 else -1.0
        units.append(u)

    n     = len(units)
    total = sum(units)
    return {"n": n, "units": total, "roi": total / n if n else np.nan}


# ── HTML ───────────────────────────────────────────────────────────────────────

def _cell(result: dict) -> str:
    n   = result["n"]
    u   = result["units"]
    roi = result["roi"]

    if n == 0:
        return "<td style='color:#ccc;padding:4px 6px;'>—</td>"

    dim = "opacity:0.30;" if n < MIN_BETS else ""
    if np.isnan(roi):
        bg = ""
    elif roi > 0:
        bg = f"background:rgba(44,160,44,{min(roi*5,0.40):.2f});"
    else:
        bg = f"background:rgba(214,39,40,{min(abs(roi)*5,0.40):.2f});"

    color = "green" if u > 0 else "red"
    return (f"<td style='{bg}{dim}padding:4px 6px;white-space:nowrap;'>"
            f"<span style='color:{color};font-weight:bold'>{u:+.1f}u</span>"
            f"<br><span style='color:#555;font-size:10px'>n={n} {roi:+.2%}</span></td>")


def build_html(
    all_results: dict,   # {label: {"auc": float, "thresh": {x: {s: dict}}, "fixed": {s: dict}}}
    df: pd.DataFrame,
) -> str:
    n        = len(df)
    pos_rate = df["target"].mean()
    mkt_avg  = df["market_prob"].mean()

    models  = list(all_results.keys())
    n_cols  = len(models) * 3   # 3 strats per model

    # ── header rows ──
    model_headers = "".join(
        f"<th colspan='3' style='border-left:2px solid #555;text-align:center'>"
        f"{m}<br><span style='font-weight:normal;font-size:10px'>AUC={all_results[m]['auc']:.4f}</span></th>"
        for m in models
    )
    strat_headers = "".join(
        "<th style='border-left:2px solid #555;font-size:11px'>Both</th>"
        "<th style='font-size:11px'>Over↑</th>"
        "<th style='font-size:11px'>Under↓</th>"
        for _ in models
    )
    thead = (
        f"<tr style='background:#111;color:white'><th rowspan='2' style='padding:6px 10px'>x</th>"
        f"{model_headers}</tr>\n"
        f"<tr style='background:#333;color:white'>{strat_headers}</tr>\n"
    )

    # ── threshold rows ──
    tbody = ""
    for x in THRESHOLDS:
        row = f"<td style='padding:4px 8px;font-weight:bold'>{x:.2f}</td>"
        for m in models:
            for s in [1, 2, 3]:
                border = "border-left:2px solid #aaa;" if s == 1 else ""
                cell   = _cell(all_results[m]["thresh"][x][s])
                # inject border into the td
                cell   = cell.replace("<td style='", f"<td style='{border}")
            row += "".join(
                _cell(all_results[m]["thresh"][x][s]).replace("<td style='", f"<td style='{'border-left:2px solid #aaa;' if s==1 else ''}")
                for s in [1, 2, 3]
            )
        tbody += f"<tr>{row}</tr>\n"

    # ── fixed strategy rows ──
    for s_fixed, label in [(4, "S4 Edge vs mkt"), (5, "S5 Always Under")]:
        row = (f"<td style='padding:4px 8px;font-weight:bold;background:#f0f0f0;"
               f"font-size:11px'>{label}</td>")
        for m in models:
            r = all_results[m]["fixed"][s_fixed]
            base_cell = _cell(r).replace("<td style='", "<td style='border-left:2px solid #aaa;font-weight:bold;")
            # fill the 3 columns: result spans 1, blanks for 2+3
            row += base_cell
            row += f"<td colspan='2' style='font-size:10px;color:#888;padding:4px 6px'></td>"
        tbody += f"<tr style='border-top:3px solid #555'>{row}</tr>\n"

    summary = f"""
<div style="font-family:monospace;background:#e8f5e9;padding:12px;border-radius:6px;
            margin-bottom:24px;font-size:12px;border-left:4px solid #2ca02c;">
  <b>Filter:</b> both Over &amp; Under prices required &nbsp;|&nbsp;
  n={n:,} &nbsp;|&nbsp; Actual Over rate: {pos_rate:.1%} &nbsp;|&nbsp;
  Market avg P(Over): {mkt_avg:.1%} &nbsp;(overpriced by ~{mkt_avg-pos_rate:.1%})<br>
  Grid: x=0.05→0.95 step 0.05 &nbsp;|&nbsp; Flat 1u bets &nbsp;|&nbsp;
  5-fold CV (OOS) &nbsp;|&nbsp; Dimmed = n&lt;{MIN_BETS}
</div>"""

    table = f"""
<div style="overflow-x:auto;">
  <table style="border-collapse:collapse;font-family:monospace;font-size:12px;white-space:nowrap;">
    <thead>{thead}</thead>
    <tbody>{tbody}</tbody>
  </table>
</div>"""

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>NFL Sacks — Strategy Grid</title>
  <style>
    body {{ font-family:monospace; margin:30px; background:#fafafa; }}
    h1   {{ font-family:sans-serif; margin-bottom:8px; }}
    table td, table th {{ border-bottom:1px solid #e0e0e0; vertical-align:middle; }}
  </style>
</head>
<body>
  <h1>NFL Sacks Props — Strategy Grid (all models side by side)</h1>
  {summary}
  {table}
</body>
</html>"""


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    warnings.filterwarnings("ignore", message="X does not have valid feature names")

    cfg = load_config()
    df  = load_data(cfg)

    y   = df["target"].astype(int)
    num_cols, cat_cols = feature_lists(cfg)
    # only use cols present in df
    n_cols = [c for c in num_cols if c in df.columns]
    c_cols = [c for c in cat_cols if c in df.columns]
    X = df[n_cols + c_cols]

    model_defs = [
        ("LR",   lr_pipeline(n_cols, c_cols)),
        ("XGB",  xgb_pipeline(n_cols, c_cols)),
        ("LGBM", lgbm_pipeline(n_cols, c_cols)),
    ]

    y_arr       = df["target"].values.astype(int)
    mkt_prob    = df["market_prob"].values
    price_over  = df["prop_median_price_over"].values
    price_under = df["prop_median_price_under"].values

    all_results = {}
    print()
    for label, pipe in model_defs:
        proba = get_cv_proba(pipe, X, y, label)
        auc   = roc_auc_score(y_arr, proba)
        print(f"    {label} — AUC: {auc:.4f}")

        thresh_results = {}
        for x in THRESHOLDS:
            thresh_results[x] = {
                s: eval_threshold_strategy(s, x, proba, mkt_prob, y_arr, price_over, price_under)
                for s in [1, 2, 3]
            }
        fixed_results = {
            s: eval_fixed_strategy(s, proba, mkt_prob, y_arr, price_over, price_under)
            for s in [4, 5]
        }
        all_results[label] = {"auc": auc, "thresh": thresh_results, "fixed": fixed_results}

    html = build_html(all_results, df)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"\nReport: {OUT_HTML}")


if __name__ == "__main__":
    main()
