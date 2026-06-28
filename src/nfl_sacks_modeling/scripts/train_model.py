"""
Train logistic regression for NFL sacks props (0.5 line, 2025 season).

Target: 1 = sacks >= 1.0 (Over hit), 0 = sacks == 0.0 (Under hit).
Pushes (sacks == 0.5) are excluded from training per config.

NaN rolling features filled with 0 (player's first game of season — no prior history).
Calibration estimated via 5-fold stratified CV on 2025 data (in-sample — add more seasons
for true OOS validation).

Input:  ~/Downloads/tmp/nfl_sacks_features_2025.parquet
Output: ~/Downloads/tmp/nfl_sacks_model_2025.pkl   (trained sklearn Pipeline)
        ~/Downloads/tmp/nfl_sacks_model_2025.html  (report: CV metrics, coefs, calibration)

Run:
    python src/nfl_sacks_modeling/scripts/train_model.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
FEATURES    = Path.home() / "Downloads" / "tmp" / "nfl_sacks_features_2025.parquet"
OUT_PKL     = Path.home() / "Downloads" / "tmp" / "nfl_sacks_model_2025.pkl"
OUT_HTML    = Path.home() / "Downloads" / "tmp" / "nfl_sacks_model_2025.html"


# ── Config ─────────────────────────────────────────────────────────────────────

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
    numeric     = rolling + ["game_total", "team_spread", "games_played_ytd"]
    categorical = ["pos_group", "pos_side"]
    return numeric, categorical


# ── Pipeline ───────────────────────────────────────────────────────────────────

def build_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value=0)),
            ("scale",  StandardScaler()),
        ]), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
    ])
    return Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")),
    ])


# ── Evaluation ─────────────────────────────────────────────────────────────────

def run_cv(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    return {
        "auc":      roc_auc_score(y, proba),
        "logloss":  log_loss(y, proba),
        "brier":    brier_score_loss(y, proba),
        "proba_cv": proba,
    }


def build_calibration_df(y_true: pd.Series, y_pred: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    frac_pos, mean_pred = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy="quantile")
    edges  = np.quantile(y_pred, np.linspace(0, 1, n_bins + 1))
    counts = [((y_pred >= edges[i]) & (y_pred < edges[i + 1])).sum() for i in range(len(edges) - 1)]
    rows = []
    for pred, actual, n in zip(mean_pred, frac_pos, counts):
        rows.append({
            "Predicted prob": f"{pred:.1%}",
            "Actual hit rate": f"{actual:.1%}",
            "Gap": f"{actual - pred:+.1%}",
            "n": int(n),
            "_gap": actual - pred,
        })
    return pd.DataFrame(rows)


def build_coef_df(pipe: Pipeline, numeric_cols: list[str], categorical_cols: list[str]) -> pd.DataFrame:
    clf     = pipe.named_steps["clf"]
    cat_enc = pipe.named_steps["pre"].named_transformers_["cat"]
    cat_names = cat_enc.get_feature_names_out(categorical_cols).tolist()
    all_names = numeric_cols + cat_names

    rows = []
    for name, coef in zip(all_names, clf.coef_[0]):
        rows.append({
            "Feature":     name,
            "Coef":        coef,
            "Odds ratio":  np.exp(coef),
        })
    return pd.DataFrame(rows).sort_values("Coef", key=abs, ascending=False).reset_index(drop=True)


# ── HTML ───────────────────────────────────────────────────────────────────────

def _fmt_coef(val: float) -> str:
    color = "green" if val > 0 else "red"
    return f'<span style="color:{color};font-weight:bold">{val:+.4f}</span>'


def _fmt_gap(val: float) -> str:
    color = "green" if val > 0 else ("red" if val < 0 else "#888")
    return f'<span style="color:{color}">{val:+.1%}</span>'


def build_html(
    seasons: list[int],
    n_train: int,
    pos_rate: float,
    metrics: dict,
    coef_df: pd.DataFrame,
    cal_df: pd.DataFrame,
) -> str:
    # ── summary box ──
    summary = f"""
<div style="font-family:monospace;background:#e8f5e9;padding:14px;border-radius:6px;
            margin-bottom:32px;font-size:13px;border-left:4px solid #2ca02c;">
  <b>Model:</b> Logistic Regression (L2, C=1.0) &nbsp;|&nbsp;
  Season(s): {seasons} &nbsp;|&nbsp;
  Train rows: {n_train:,} &nbsp;|&nbsp;
  Positive rate: {pos_rate:.1%}<br>
  <b>5-fold stratified CV</b> (in-season — true OOS requires additional seasons) &nbsp;|&nbsp;
  AUC: <b>{metrics['auc']:.4f}</b> &nbsp;|&nbsp;
  Log-loss: <b>{metrics['logloss']:.4f}</b> &nbsp;|&nbsp;
  Brier: <b>{metrics['brier']:.4f}</b>
</div>"""

    # ── calibration table ──
    cal_rows = ""
    for _, row in cal_df.iterrows():
        gap_fmt = _fmt_gap(row["_gap"])
        cal_rows += (
            f"<tr><td>{row['Predicted prob']}</td><td>{row['Actual hit rate']}</td>"
            f"<td>{gap_fmt}</td><td>{row['n']}</td></tr>\n"
        )
    cal_table = f"""
<div style="margin-bottom:48px;">
  <h2 style="font-family:sans-serif;margin-bottom:4px;">Calibration (5-fold CV OOS predictions)</h2>
  <p style="font-family:monospace;color:#555;font-size:13px;margin-top:0;">
    Quantile bins — each bin has approx equal number of predictions.
    Gap = Actual − Predicted (green = model under-confident, red = over-confident).
  </p>
  <table style="border-collapse:collapse;font-family:monospace;font-size:14px;">
    <thead>
      <tr style="background:#222;color:white;">
        <th>Predicted prob</th><th>Actual hit rate</th><th>Gap</th><th>n</th>
      </tr>
    </thead>
    <tbody>{cal_rows}</tbody>
  </table>
</div>"""

    # ── coefficient table ──
    coef_rows = ""
    for _, row in coef_df.iterrows():
        coef_rows += (
            f"<tr><td>{row['Feature']}</td>"
            f"<td>{_fmt_coef(row['Coef'])}</td>"
            f"<td>{row['Odds ratio']:.4f}</td></tr>\n"
        )
    coef_table = f"""
<div style="margin-bottom:48px;">
  <h2 style="font-family:sans-serif;margin-bottom:4px;">Feature Coefficients (sorted by |coef|)</h2>
  <p style="font-family:monospace;color:#555;font-size:13px;margin-top:0;">
    Numeric features are standardized (z-score). NaN rolling values imputed as 0.<br>
    Odds ratio = exp(coef); &gt;1 increases sack probability, &lt;1 decreases it.
  </p>
  <table style="border-collapse:collapse;font-family:monospace;font-size:14px;">
    <thead>
      <tr style="background:#222;color:white;">
        <th>Feature</th><th>Coefficient</th><th>Odds ratio</th>
      </tr>
    </thead>
    <tbody>{coef_rows}</tbody>
  </table>
</div>"""

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>NFL Sacks Model 2025</title>
  <style>
    body {{ font-family: monospace; margin: 40px; background: #fafafa; }}
    table td, table th {{ padding: 6px 14px; border-bottom: 1px solid #ddd; text-align: left; }}
    h1 {{ font-family: sans-serif; }}
  </style>
</head>
<body>
  <h1>NFL Player Sacks — Logistic Regression Model Report</h1>
  {summary}
  {cal_table}
  {coef_table}
</body>
</html>"""


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    cfg = load_config()
    df  = pd.read_parquet(FEATURES)

    train = df[df["target"].notna()].copy()
    y     = train["target"].astype(int)

    numeric_cols, categorical_cols = feature_lists(cfg)
    X = train[numeric_cols + categorical_cols]

    n_train  = len(train)
    pos_rate = y.mean()
    print(f"Training rows : {n_train}  |  Positive rate: {pos_rate:.1%}")
    print(f"Numeric features   : {len(numeric_cols)}")
    print(f"Categorical features: {categorical_cols}")

    pipe = build_pipeline(numeric_cols, categorical_cols)

    print("\nRunning 5-fold stratified CV...")
    metrics = run_cv(pipe, X, y)
    print(f"  AUC      : {metrics['auc']:.4f}")
    print(f"  Log-loss : {metrics['logloss']:.4f}")
    print(f"  Brier    : {metrics['brier']:.4f}")

    print("\nFitting final model on all training data...")
    pipe.fit(X, y)

    coef_df = build_coef_df(pipe, numeric_cols, categorical_cols)
    cal_df  = build_calibration_df(y, metrics["proba_cv"])

    print("\nTop 10 features by |coefficient|:")
    print(coef_df[["Feature", "Coef", "Odds ratio"]].head(10).to_string(index=False))

    with open(OUT_PKL, "wb") as f:
        pickle.dump(pipe, f)
    print(f"\nModel saved : {OUT_PKL}")

    html = build_html(cfg["seasons"], n_train, pos_rate, metrics, coef_df, cal_df)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Report      : {OUT_HTML}")


if __name__ == "__main__":
    main()
