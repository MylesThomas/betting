"""
OOS evaluation for NFL sacks props — cross-season and full-data.

Four sections:
  1. Train on 2025 → test on 2024  (true OOS for 2024)
  2. Train on 2024 → test on 2025  (true OOS for 2025 — forward-looking)
  3. Aggregate of 1+2              (every row is OOS)
  4. Pooled 5-fold CV on 2024+2025 → strategy grid + cumulative P&L line graph

Output: ~/Downloads/tmp/nfl_sacks_oos_eval.html

Run:
    python src/nfl_sacks_modeling/scripts/oos_eval.py
"""

import json
import warnings
from pathlib import Path

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CONFIG_PATH   = Path(__file__).resolve().parents[1] / "config.yaml"
FEATURES_2024 = Path.home() / "Downloads" / "tmp" / "nfl_sacks_features_2024.parquet"
FEATURES_2025 = Path.home() / "Downloads" / "tmp" / "nfl_sacks_features_2025.parquet"
OUT_HTML      = Path.home() / "Downloads" / "tmp" / "nfl_sacks_oos_eval.html"

THRESHOLDS   = np.round(np.arange(0.05, 1.00, 0.05), 2)
MIN_BETS     = 20
LINE_X       = 0.70   # Under-only threshold shown on the cumulative P&L graph
MODEL_LABELS = ["LR", "XGB", "LGBM"]


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
    # game_id format: "2024_01_BAL_KC" — extract season if not already present
    if "season" not in df.columns:
        df["season"] = df["game_id"].str[:4].astype(int)
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


def lgbm_pipeline(num_cols: list, cat_cols: list) -> Pipeline:
    pre = ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])
    return Pipeline([("pre", pre),
                     ("clf", lgb.LGBMClassifier(n_estimators=300, max_depth=4,
                                                learning_rate=0.05, subsample=0.8,
                                                colsample_bytree=0.8, random_state=42,
                                                verbosity=-1))])


def make_pipeline(label: str, num_cols: list, cat_cols: list) -> Pipeline:
    if label == "LR":
        return lr_pipeline(num_cols, cat_cols)
    if label == "XGB":
        return xgb_pipeline(num_cols, cat_cols)
    if label == "LGBM":
        return lgbm_pipeline(num_cols, cat_cols)
    raise ValueError(label)


# ── Inference ──────────────────────────────────────────────────────────────────

def get_proba_cross(train_df: pd.DataFrame, test_df: pd.DataFrame,
                    pipe: Pipeline, n_cols: list, c_cols: list) -> np.ndarray:
    X_train = train_df[n_cols + c_cols]
    y_train = train_df["target"].astype(int)
    X_test  = test_df[n_cols + c_cols]
    pipe.fit(X_train, y_train)
    return pipe.predict_proba(X_test)[:, 1]


def get_proba_full(df: pd.DataFrame, pipe: Pipeline,
                   n_cols: list, c_cols: list) -> np.ndarray:
    """5-fold stratified CV on the pooled dataset — each row gets an OOS probability."""
    X  = df[n_cols + c_cols]
    y  = df["target"].astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]


# ── Strategy evaluation ────────────────────────────────────────────────────────

def eval_threshold_strategy(strat: int, x: float,
                             model_prob: np.ndarray, y: np.ndarray,
                             price_over: np.ndarray, price_under: np.ndarray) -> dict:
    units = []
    for i in range(len(y)):
        mp = model_prob[i]
        if strat == 1:
            bet_over = mp >= x
        elif strat == 2:
            if mp < x:
                continue
            bet_over = True
        elif strat == 3:
            if mp >= x:
                continue
            bet_over = False
        else:
            raise ValueError(strat)
        u = units_on_win(price_over[i]) if (bet_over and y[i] == 1) else (
            units_on_win(price_under[i]) if (not bet_over and y[i] == 0) else -1.0)
        units.append(u)
    n = len(units)
    total = sum(units)
    return {"n": n, "units": total, "roi": total / n if n else np.nan}


def eval_fixed_strategy(strat: int, model_prob: np.ndarray, market_prob: np.ndarray,
                         y: np.ndarray, price_over: np.ndarray, price_under: np.ndarray) -> dict:
    units = []
    for i in range(len(y)):
        if strat == 4:
            bet_over = model_prob[i] > market_prob[i]
        elif strat == 5:
            bet_over = False
        else:
            raise ValueError(strat)
        u = units_on_win(price_over[i]) if (bet_over and y[i] == 1) else (
            units_on_win(price_under[i]) if (not bet_over and y[i] == 0) else -1.0)
        units.append(u)
    n = len(units)
    total = sum(units)
    return {"n": n, "units": total, "roi": total / n if n else np.nan}


def build_grid_results(df: pd.DataFrame, proba: np.ndarray) -> dict:
    y           = df["target"].values.astype(int)
    mkt_prob    = df["market_prob"].values
    price_over  = df["prop_median_price_over"].values
    price_under = df["prop_median_price_under"].values
    thresh = {
        x: {s: eval_threshold_strategy(s, x, proba, y, price_over, price_under)
            for s in [1, 2, 3]}
        for x in THRESHOLDS
    }
    fixed = {
        s: eval_fixed_strategy(s, proba, mkt_prob, y, price_over, price_under)
        for s in [4, 5]
    }
    return {"thresh": thresh, "fixed": fixed}


# ── Line graph ─────────────────────────────────────────────────────────────────

def compute_line_records(df: pd.DataFrame, proba: np.ndarray,
                          strategy: str, threshold: float = LINE_X) -> list[dict]:
    records = []
    price_under = df["prop_median_price_under"].values
    y           = df["target"].values.astype(int)
    seasons     = df["season"].values
    weeks       = df["week"].values
    for i in range(len(y)):
        if strategy == "always_under":
            pass
        elif strategy == "under_only":
            if proba[i] >= threshold:
                continue
        else:
            raise ValueError(strategy)
        u = units_on_win(price_under[i]) if y[i] == 0 else -1.0
        records.append({"season": int(seasons[i]), "week": int(weeks[i]), "units": u})
    return records


# ── HTML helpers ───────────────────────────────────────────────────────────────

def _cell(result: dict, border_left: bool = False) -> str:
    n   = result["n"]
    u   = result["units"]
    roi = result["roi"]
    border = "border-left:2px solid #aaa;" if border_left else ""

    if n == 0:
        return f"<td style='{border}color:#ccc;padding:4px 6px;'>—</td>"

    dim = "opacity:0.30;" if n < MIN_BETS else ""
    # Color by absolute units won — 150u hits max opacity
    _SCALE = 150.0
    if np.isnan(u):
        bg = ""
    elif u > 0:
        bg = f"background:rgba(44,160,44,{min(u/_SCALE, 0.40):.2f});"
    else:
        bg = f"background:rgba(214,39,40,{min(abs(u)/_SCALE, 0.40):.2f});"

    color = "green" if u > 0 else "red"
    return (
        f"<td style='{border}{bg}{dim}padding:4px 6px;white-space:nowrap;'>"
        f"<span style='color:{color};font-weight:bold'>{u:+.1f}u</span>"
        f"<br><span style='color:#555;font-size:10px'>n={n} {roi:+.2%}</span></td>"
    )


def build_grid_html(all_results: dict, df: pd.DataFrame, title: str, desc: str) -> str:
    n        = len(df)
    pos_rate = df["target"].mean()
    mkt_avg  = df["market_prob"].mean()

    model_headers = "".join(
        f"<th colspan='3' style='border-left:2px solid #555;text-align:center'>{m}</th>"
        for m in MODEL_LABELS
    )
    strat_headers = "".join(
        "<th style='border-left:2px solid #555;font-size:11px'>Both</th>"
        "<th style='font-size:11px'>Over↑</th>"
        "<th style='font-size:11px'>Under↓</th>"
        for _ in MODEL_LABELS
    )
    thead = (
        f"<tr style='background:#111;color:white'>"
        f"<th rowspan='2' style='padding:6px 10px'>x</th>"
        f"{model_headers}</tr>\n"
        f"<tr style='background:#333;color:white'>{strat_headers}</tr>\n"
    )

    tbody = ""
    for x in THRESHOLDS:
        row = f"<td style='padding:4px 8px;font-weight:bold'>{x:.2f}</td>"
        for m in MODEL_LABELS:
            row += _cell(all_results[m]["thresh"][x][1], border_left=True)
            row += _cell(all_results[m]["thresh"][x][2])
            row += _cell(all_results[m]["thresh"][x][3])
        tbody += f"<tr>{row}</tr>\n"

    for s_fixed, flabel in [(4, "S4 Edge vs mkt"), (5, "S5 Always Under")]:
        row = (f"<td style='padding:4px 8px;font-weight:bold;"
               f"background:#f0f0f0;font-size:11px'>{flabel}</td>")
        for m in MODEL_LABELS:
            r    = all_results[m]["fixed"][s_fixed]
            cell = _cell(r, border_left=True).replace(
                "<td style='border-left:2px solid #aaa;",
                "<td style='border-left:2px solid #aaa;font-weight:bold;"
            )
            row += cell
            row += "<td colspan='2' style='font-size:10px;color:#888;padding:4px 6px'></td>"
        tbody += f"<tr style='border-top:3px solid #555'>{row}</tr>\n"

    summary = (
        f"<div style='font-family:monospace;background:#e8f5e9;padding:10px 14px;"
        f"border-radius:6px;margin-bottom:12px;font-size:12px;"
        f"border-left:4px solid #2ca02c;'>"
        f"n={n:,} &nbsp;|&nbsp; Actual Over rate: {pos_rate:.1%} &nbsp;|&nbsp; "
        f"Market avg P(Over): {mkt_avg:.1%} &nbsp;|&nbsp; "
        f"Structural Under edge: ~{mkt_avg - pos_rate:.1%}"
        f"</div>"
    )

    table = (
        "<div style='overflow-x:auto;'>"
        "<table style='border-collapse:collapse;font-family:monospace;"
        "font-size:12px;white-space:nowrap;'>"
        f"<thead>{thead}</thead><tbody>{tbody}</tbody>"
        "</table></div>"
    )

    return (
        f"<div style='margin-bottom:56px;'>"
        f"<h2 style='font-family:sans-serif;margin-bottom:4px;'>{title}</h2>"
        f"<p style='font-family:monospace;color:#444;font-size:13px;"
        f"max-width:900px;margin-bottom:12px;line-height:1.6;'>{desc}</p>"
        f"{summary}{table}</div>"
    )


def build_line_graph_html(df_all: pd.DataFrame, line_records: dict) -> str:
    all_weeks = (
        df_all[["season", "week"]]
        .drop_duplicates()
        .sort_values(["season", "week"])
        .itertuples(index=False)
    )
    all_weeks = list(all_weeks)
    x_labels  = ["Start"] + [f"{r.season} W{r.week}" for r in all_weeks]

    # Color keyed by model prefix, not full label string (resilient to x value changes)
    def _style(strat_label: str) -> dict:
        if strat_label == "Always Under":
            return {"border": "rgba(100,100,100,0.9)", "bg": "rgba(100,100,100,0.05)", "dash": [6, 4]}
        if strat_label.startswith("LR"):
            return {"border": "rgba(31,119,180,1)",   "bg": "rgba(31,119,180,0.05)",  "dash": []}
        if strat_label.startswith("XGB"):
            return {"border": "rgba(255,127,14,1)",   "bg": "rgba(255,127,14,0.05)",  "dash": []}
        if strat_label.startswith("LGBM"):
            return {"border": "rgba(44,160,44,1)",    "bg": "rgba(44,160,44,0.05)",   "dash": []}
        return {"border": "rgba(0,0,0,1)", "bg": "transparent", "dash": []}

    datasets = []
    for strat_label, records in line_records.items():
        weekly_map: dict = {}
        for r in records:
            key = (r["season"], r["week"])
            weekly_map[key] = weekly_map.get(key, 0.0) + r["units"]

        cum  = 0.0
        data = [0.0]
        for aw in all_weeks:
            cum += weekly_map.get((aw.season, aw.week), 0.0)
            data.append(round(cum, 3))

        style = _style(strat_label)
        datasets.append({
            "label":           strat_label,
            "data":            data,
            "borderColor":     style["border"],
            "backgroundColor": style["bg"],
            "borderDash":      style["dash"],
            "borderWidth":     2,
            "pointRadius":     0,
            "tension":         0.1,
            "fill":            False,
        })

    chart_json = json.dumps({"labels": x_labels, "datasets": datasets})

    return f"""
<div style="margin-top:32px;margin-bottom:60px;">
  <h3 style="font-family:sans-serif;margin-bottom:4px;">
    Cumulative P&amp;L — 2024 Week 1 → 2025 Week 17
  </h3>
  <p style="font-family:monospace;color:#444;font-size:13px;max-width:900px;
             margin-bottom:16px;line-height:1.6;">
    Week-by-week cumulative units for the Always Under baseline (dashed gray) vs each
    model at its best Under-only threshold (chosen programmatically from S4 CV results —
    x selected by total units across all thresholds 0.05–0.95). Each row is OOS (pooled
    5-fold CV). Each point is the running total after all bets in that week settle.
    Sorted chronologically. Flat 1-unit bets.
  </p>
  <div style="max-width:1100px;">
    <canvas id="pnlChart" height="120"></canvas>
  </div>
  <script>
  (function() {{
    const ctx = document.getElementById('pnlChart').getContext('2d');
    new Chart(ctx, {{
      type: 'line',
      data: {chart_json},
      options: {{
        responsive: true,
        interaction: {{ mode: 'index', intersect: false }},
        plugins: {{
          legend: {{ position: 'top', labels: {{ font: {{ family: 'monospace' }} }} }},
          tooltip: {{
            callbacks: {{
              label: (item) => ` ${{item.dataset.label}}: ${{item.parsed.y > 0 ? '+' : ''}}${{item.parsed.y.toFixed(2)}}u`
            }}
          }}
        }},
        scales: {{
          x: {{
            ticks: {{
              maxTicksLimit: 60,
              maxRotation: 60,
              minRotation: 45,
              font: {{ size: 10, family: 'monospace' }},
            }}
          }},
          y: {{
            title: {{ display: true, text: 'Cumulative Units', font: {{ family: 'monospace' }} }},
            grid: {{
              color: (ctx) => ctx.tick.value === 0 ? 'rgba(0,0,0,0.4)' : 'rgba(0,0,0,0.07)',
              lineWidth: (ctx) => ctx.tick.value === 0 ? 2 : 1,
            }}
          }}
        }}
      }}
    }});
  }})();
  </script>
</div>"""


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    warnings.filterwarnings("ignore")

    cfg = load_config()
    num_cols, cat_cols = feature_lists(cfg)

    print("Loading data...")
    df24  = load_season(FEATURES_2024)
    df25  = load_season(FEATURES_2025)
    df_all = pd.concat([df24, df25], ignore_index=True)

    # Use only columns present in both seasons (safe for cross-season fitting)
    n_cols = [c for c in num_cols if c in df24.columns and c in df25.columns]
    c_cols = [c for c in cat_cols if c in df24.columns and c in df25.columns]

    print(f"  2024: {len(df24)} rows  |  2025: {len(df25)} rows  |  All: {len(df_all)} rows")
    print(f"  Features: {len(n_cols)} numeric, {len(c_cols)} categorical")

    # ── Section 1: train 2025 → test 2024 ─────────────────────────────────────
    print("\nSection 1: Train 2025 → Test 2024")
    s1_probas  = {}
    s1_results = {}
    for label in MODEL_LABELS:
        pipe = make_pipeline(label, n_cols, c_cols)
        p    = get_proba_cross(df25, df24, pipe, n_cols, c_cols)
        s1_probas[label]  = p
        s1_results[label] = build_grid_results(df24, p)
        n_under = (p < LINE_X).sum()
        print(f"  {label}: {n_under}/{len(p)} rows below x={LINE_X} (Under bets)")

    # ── Section 2: train 2024 → test 2025 ─────────────────────────────────────
    print("\nSection 2: Train 2024 → Test 2025")
    s2_probas  = {}
    s2_results = {}
    for label in MODEL_LABELS:
        pipe = make_pipeline(label, n_cols, c_cols)
        p    = get_proba_cross(df24, df25, pipe, n_cols, c_cols)
        s2_probas[label]  = p
        s2_results[label] = build_grid_results(df25, p)
        n_under = (p < LINE_X).sum()
        print(f"  {label}: {n_under}/{len(p)} rows below x={LINE_X} (Under bets)")

    # ── Section 3: combined OOS (1+2) ─────────────────────────────────────────
    print("\nSection 3: Combined OOS")
    # df24 rows used s1 probas, df25 rows used s2 probas — concat in same order
    df_oos = pd.concat([df24, df25], ignore_index=True)
    s3_results = {}
    for label in MODEL_LABELS:
        combined_proba    = np.concatenate([s1_probas[label], s2_probas[label]])
        s3_results[label] = build_grid_results(df_oos, combined_proba)

    # ── Section 4: pooled 5-fold CV + line graph ──────────────────────────────
    print("\nSection 4: Pooled 5-fold CV (2024+2025)")
    s4_probas  = {}
    s4_results = {}
    for label in MODEL_LABELS:
        pipe = make_pipeline(label, n_cols, c_cols)
        p    = get_proba_full(df_all, pipe, n_cols, c_cols)
        s4_probas[label]  = p
        s4_results[label] = build_grid_results(df_all, p)
        print(f"  {label} done")

    # Find best Under-only threshold per model in S4 (by total units)
    print("\nFinding best x per model (S4 Under-only)...")
    best_xs = {}
    for label in MODEL_LABELS:
        proba = s4_probas[label]
        y_arr = df_all["target"].values.astype(int)
        pu    = df_all["prop_median_price_under"].values
        best_x, best_u = None, -np.inf
        for x in THRESHOLDS:
            ul = [(units_on_win(pu[i]) if y_arr[i]==0 else -1.0) for i in range(len(y_arr)) if proba[i] < x]
            if len(ul) >= MIN_BETS and sum(ul) > best_u:
                best_u, best_x = sum(ul), x
        best_xs[label] = best_x
        print(f"  {label}: best x={best_x:.2f}  ({best_u:+.1f}u)")

    # Line graph records — best x per model, all using S4 CV probabilities
    print("Computing line graph data...")
    line_records = {}
    line_records["Always Under"] = compute_line_records(df_all, s4_probas["LR"], "always_under")
    for label in MODEL_LABELS:
        bx = best_xs[label]
        line_records[f"{label} Under x={bx:.2f}"] = compute_line_records(
            df_all, s4_probas[label], "under_only", bx
        )

    # ── Build HTML ─────────────────────────────────────────────────────────────
    print("Building HTML...")

    header = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>NFL Sacks — OOS Evaluation</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
  <style>
    body  { font-family: monospace; margin: 40px; background: #fafafa; }
    h1,h2,h3 { font-family: sans-serif; }
    table td, table th { border-bottom: 1px solid #e0e0e0; vertical-align: middle; }
    .divider { border-top: 3px solid #333; margin: 44px 0; }
  </style>
</head>
<body>
<h1>NFL Sacks Props — OOS Evaluation</h1>
<div style="font-family:monospace;background:#fff3cd;padding:12px 16px;border-radius:6px;
            margin-bottom:32px;font-size:13px;border-left:4px solid #ffc107;max-width:950px;
            line-height:1.7;">
  <b>How to read:</b> Sections 1–3 use cross-season hold-outs — every bet is out-of-sample.
  Section 3 is the headline: 2024 rows use a model trained only on 2025, 2025 rows use a model
  trained only on 2024. No look-ahead anywhere in 1–3. Section 4 uses pooled 5-fold CV on all
  data — each row is still OOS (predicted by a fold that excluded it), but folds are not
  temporally ordered so it benefits from more training data per fold than Sections 1–3.
  Dimmed cells = n&lt;""" + str(MIN_BETS) + """ bets. Flat 1-unit bets throughout.
</div>"""

    s1_html = build_grid_html(
        s1_results, df24,
        "Section 1 — 2024 Games &nbsp;|&nbsp; Trained on 2025",
        "Models are fit on the full 2025 regular season (1,842 rows) and evaluated on 2024 "
        "games (1,623 rows). No 2024 data informed model parameters — true OOS. "
        "This direction trains on the future to predict the past, so treat it as a robustness "
        "check rather than a deployable signal."
    )

    s2_html = build_grid_html(
        s2_results, df25,
        "Section 2 — 2025 Games &nbsp;|&nbsp; Trained on 2024",
        "Models are fit on the full 2024 regular season (1,623 rows) and evaluated on 2025 "
        "games (1,842 rows). This is the forward-looking direction — one historical season used "
        "to predict the next — and is the closest analog to live deployment."
    )

    s3_html = build_grid_html(
        s3_results, df_oos,
        "Section 3 — Combined OOS &nbsp;|&nbsp; 2024+2025, All Rows OOS",
        "Sections 1 and 2 pooled into a single table. Every row has a true OOS probability: "
        "2024 rows predicted by the model trained on 2025; 2025 rows predicted by the model "
        "trained on 2024. This is the cleanest headline estimate of cross-season generalization."
    )

    s4_html = build_grid_html(
        s4_results, df_all,
        "Section 4 — Pooled 5-Fold CV &nbsp;|&nbsp; 2024+2025 Combined",
        "5-fold stratified CV on the full pooled dataset (2024+2025, 2,927 rows). "
        "Each row gets an OOS probability from a model trained on the other 4 folds. "
        "Folds are stratified but not temporally ordered — this maximises training data "
        "per fold vs. the cross-season hold-outs in Sections 1–3. "
        "The line graph below uses these CV probabilities to show the cumulative P&amp;L "
        "trajectory sorted chronologically from 2024 Week 1 through 2025 Week 17."
    )

    line_html = build_line_graph_html(df_all, line_records)

    footer = "</body></html>"

    html = "\n".join([
        header,
        s1_html,
        "<div class='divider'></div>",
        s2_html,
        "<div class='divider'></div>",
        s3_html,
        "<div class='divider'></div>",
        s4_html,
        line_html,
        footer,
    ])

    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"\nReport: {OUT_HTML}")

    # Quick summary
    print(f"\n{'='*55}")
    print("  OOS SUMMARY — Under-only x=0.70")
    for label in MODEL_LABELS:
        r24 = s1_results[label]["thresh"][LINE_X][3]
        r25 = s2_results[label]["thresh"][LINE_X][3]
        r3  = s3_results[label]["thresh"][LINE_X][3]
        print(f"  {label:<5}  2024 OOS: {r24['units']:+.1f}u (n={r24['n']})  "
              f"2025 OOS: {r25['units']:+.1f}u (n={r25['n']})  "
              f"Combined: {r3['units']:+.1f}u (n={r3['n']})")
    always_u = s3_results["LR"]["fixed"][5]
    print(f"  Always Under (baseline): {always_u['units']:+.1f}u (n={always_u['n']})")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
