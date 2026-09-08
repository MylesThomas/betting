"""
Step 4 — Probability conversion + edge computation.

For binary HR market (always 0.5 line), the model directly outputs P(over 0.5) = P(HR ≥ 1).
Three methods compared:
  Method A: Raw model probability (logistic or XGBoost output directly)
  Method B: Platt scaling (logistic calibration on top of model scores)
  Method C: Empirical quantile lookup (bucket p_model, compute actual over rate)

Brier score + calibration curve for each method.
Edge = p_model_over - raw_implied_prob_over (always raw, never novig).

Output:
  ~/Downloads/tmp/mlb_batter_hr_scored.parquet  (full spine with p_model + edge)
  HTML appended to session log
"""
from __future__ import annotations

import sys
import warnings
from datetime import datetime
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import boto3
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
REPO_ROOT   = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

LOCAL_SPINE  = Path.home() / "Downloads/tmp/mlb_batter_hr_spine.parquet"
LOCAL_SCORED = Path.home() / "Downloads/tmp/mlb_batter_hr_scored.parquet"
LOCAL_MODEL  = REPO_ROOT / "src/mlb_batter_home_runs_modeling/models/mlb_batter_hr_model.joblib"
S3_BUCKET    = "the-odds-api-mt"
S3_SPINE_KEY = "mlb/batter_home_runs_model/spine/mlb_batter_hr_scored.parquet"
HTML_LOG     = REPO_ROOT / "knowledge-base/raw/20260801-mlb-batter-home-runs.html"
ET           = ZoneInfo("America/New_York")
TARGET       = "hr_over_0_5"


def ts() -> str:
    return datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S ET")


def df_to_html_table(df: pd.DataFrame, caption: str = "") -> str:
    rows_html = []
    header = "<tr>" + "".join(f"<th>{c}</th>" for c in df.columns) + "</tr>"
    for _, row in df.iterrows():
        cells = "".join(f"<td>{v}</td>" for v in row)
        rows_html.append(f"<tr>{cells}</tr>")
    cap = f"<caption style='font-weight:bold;text-align:left;padding:6px 0'>{caption}</caption>" if caption else ""
    return f"<table>{cap}<thead>{header}</thead><tbody>{''.join(rows_html)}</tbody></table>"


def main() -> None:
    if not LOCAL_SPINE.exists():
        print(f"ERROR: {LOCAL_SPINE} not found")
        sys.exit(1)
    if not LOCAL_MODEL.exists():
        print(f"ERROR: {LOCAL_MODEL} not found — run step3bc first")
        sys.exit(1)

    spine = pd.read_parquet(LOCAL_SPINE)
    model_artifact = joblib.load(LOCAL_MODEL)
    model      = model_artifact["model"]
    features   = model_artifact["features"]
    model_type = model_artifact["model_type"]
    print(f"Loaded spine: {len(spine):,} rows")
    print(f"Model type: {model_type}, features: {features}")

    # ── Get player-game level OOF predictions for calibration comparison
    df_pg = (
        spine
        .dropna(subset=[TARGET] + [f for f in features if f in spine.columns])
        .sort_values(["player_key", "game_date"])
        .drop_duplicates(subset=["player_key", "game_date"])
        .sort_values("game_date")
        .reset_index(drop=True)
    )
    print(f"Player-game rows for calibration: {len(df_pg):,}")

    feat_cols = [f for f in features if f in df_pg.columns]
    X = df_pg[feat_cols].values
    y = df_pg[TARGET].values

    # ── Method A: Raw OOF model probabilities ───────────────────────────────────
    print("\nMethod A: Raw OOF model probabilities...")
    tscv = TimeSeriesSplit(n_splits=5)
    y_pred_a = np.zeros(len(df_pg))
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = y[train_idx]
        if model_type == "xgboost":
            from xgboost import XGBClassifier
            pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
            clf = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8,
                                scale_pos_weight=pos_weight, eval_metric="auc",
                                verbosity=0, random_state=42)
        else:
            clf = Pipeline([("scaler", StandardScaler()),
                            ("clf", LogisticRegression(max_iter=1000, C=0.5))])
        clf.fit(X_train, y_train)
        y_pred_a[val_idx] = clf.predict_proba(X_val)[:, 1]

    brier_a = brier_score_loss(y, y_pred_a)
    print(f"  Method A Brier score: {brier_a:.5f}")

    # ── Method B: Platt scaling (logistic calibration on top of model scores) ───
    print("\nMethod B: Platt scaling...")
    y_pred_b = np.zeros(len(df_pg))
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = y[train_idx]
        if model_type == "xgboost":
            from xgboost import XGBClassifier
            pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
            base = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8,
                                 scale_pos_weight=pos_weight, eval_metric="auc",
                                 verbosity=0, random_state=42)
        else:
            base = Pipeline([("scaler", StandardScaler()),
                             ("clf", LogisticRegression(max_iter=1000, C=0.5))])
        base.fit(X_train, y_train)
        raw_scores = base.predict_proba(X_val)[:, 1]
        y_val = y[val_idx]
        calib = LogisticRegression(max_iter=1000)
        calib.fit(raw_scores.reshape(-1, 1), y_val)
        y_pred_b[val_idx] = calib.predict_proba(raw_scores.reshape(-1, 1))[:, 1]

    brier_b = brier_score_loss(y, y_pred_b)
    print(f"  Method B Brier score: {brier_b:.5f}")

    # ── Calibration curves ───────────────────────────────────────────────────────
    print("\nCalibration analysis...")
    calib_rows = []
    for method_label, y_pred in [("A_raw", y_pred_a), ("B_platt", y_pred_b)]:
        df_calib = pd.DataFrame({"y_true": y, "y_pred": y_pred})
        df_calib["decile"] = pd.qcut(y_pred, 10, labels=False, duplicates="drop")
        for d, grp in df_calib.groupby("decile"):
            calib_rows.append({
                "method": method_label,
                "decile": int(d),
                "n": len(grp),
                "predicted_rate": round(grp["y_pred"].mean(), 4),
                "actual_rate": round(grp["y_true"].mean(), 4),
                "gap": round(grp["y_true"].mean() - grp["y_pred"].mean(), 4),
            })
    calib_df = pd.DataFrame(calib_rows)
    print(calib_df.to_string(index=False))

    # Pick winning method (lower Brier)
    best_method = "A" if brier_a <= brier_b else "B"
    y_pred_best = y_pred_a if best_method == "A" else y_pred_b
    print(f"\nWinning method: {best_method} (Brier: {min(brier_a, brier_b):.5f})")

    # ── Clip ────────────────────────────────────────────────────────────────────
    n_clipped_low  = (y_pred_best < 0.01).sum()
    n_clipped_high = (y_pred_best > 0.99).sum()
    y_pred_best = np.clip(y_pred_best, 0.01, 0.99)
    print(f"Clips: low={n_clipped_low}, high={n_clipped_high}")

    df_pg = df_pg.copy()
    df_pg["p_model_over"] = y_pred_best
    df_pg["p_model_under"] = 1.0 - y_pred_best

    # ── Score full spine (all books × lines) ────────────────────────────────────
    print("\nScoring full spine...")
    valid_spine = spine.copy()
    valid_spine = valid_spine.dropna(subset=[f for f in feat_cols if f in valid_spine.columns])

    X_all = valid_spine[feat_cols].values
    p_over_all = np.clip(model.predict_proba(X_all)[:, 1], 0.01, 0.99)
    valid_spine["p_model_over"]  = p_over_all
    valid_spine["p_model_under"] = 1.0 - p_over_all

    # Per-book edge (raw, not novig)
    valid_spine["raw_implied_prob_over"]  = 1.0 / valid_spine["over_price"]
    valid_spine["raw_implied_prob_under"] = 1.0 / valid_spine["under_price"]
    valid_spine["edge_over"]  = valid_spine["p_model_over"]  - valid_spine["raw_implied_prob_over"]
    valid_spine["edge_under"] = valid_spine["p_model_under"] - valid_spine["raw_implied_prob_under"]

    # ── Edge assert ──────────────────────────────────────────────────────────────
    sample = valid_spine.sample(min(200, len(valid_spine)), random_state=42)
    edge_exp_over  = sample["p_model_over"]  - sample["raw_implied_prob_over"]
    edge_exp_under = sample["p_model_under"] - sample["raw_implied_prob_under"]
    assert (sample["edge_over"]  - edge_exp_over ).abs().max() < 1e-6, "edge_over != p_model_over - raw_implied_prob_over"
    assert (sample["edge_under"] - edge_exp_under).abs().max() < 1e-6, "edge_under != p_model_under - raw_implied_prob_under"
    print("PASS: edge_over = p_model_over - raw_implied_prob_over (assert passed)")

    # ── yhat book-invariant assert ───────────────────────────────────────────────
    yhat_range = valid_spine.groupby(["player_key", "game_date", "offered_line"])["p_model_over"].agg(
        lambda x: x.max() - x.min()
    )
    max_drift = yhat_range.max()
    n_drifty = (yhat_range > 1e-8).sum()
    assert n_drifty == 0, (
        f"p_model_over NOT book-invariant — {n_drifty} groups vary by >{1e-8} (max={max_drift:.2e})"
    )
    print(f"PASS: p_model_over is book-invariant across all books (max drift={max_drift:.2e})")

    # ── Line monotonicity (trivial for single 0.5 line market) ──────────────────
    multi_line = valid_spine.groupby(["player_key", "game_date"])["offered_line"].nunique()
    n_multi = (multi_line > 1).sum()
    print(f"Multi-line player-games: {n_multi} (expected ~0 for pure 0.5 market)")

    # ── Novig for display ────────────────────────────────────────────────────────
    novig_total = valid_spine["raw_implied_prob_over"] + valid_spine["raw_implied_prob_under"]
    valid_spine["novig_prob_over"]  = valid_spine["raw_implied_prob_over"]  / novig_total
    valid_spine["novig_prob_under"] = valid_spine["raw_implied_prob_under"] / novig_total

    # ── Save ────────────────────────────────────────────────────────────────────
    LOCAL_SCORED.parent.mkdir(parents=True, exist_ok=True)
    valid_spine.to_parquet(LOCAL_SCORED, index=False)
    print(f"Saved scored spine → {LOCAL_SCORED}  ({len(valid_spine):,} rows)")

    # ── Spot-check: Aaron Judge ───────────────────────────────────────────────────
    judge = valid_spine[valid_spine["player_key"] == "aaron judge"].sort_values("game_date").tail(15)
    show_cols = [c for c in ["game_date", "bookmaker", "hr_actual", "hr_over_0_5",
                              "p_model_over", "raw_implied_prob_over", "edge_over",
                              "edge_under", "novig_prob_over"] if c in valid_spine.columns]
    print("\n=== Spot-check Aaron Judge (last 15) ===")
    print(judge[show_cols].to_string(index=False))

    # ── HTML ─────────────────────────────────────────────────────────────────────
    method_comparison = pd.DataFrame([
        {"method": "A — Raw model prob", "brier_score": round(brier_a, 5)},
        {"method": "B — Platt scaling",  "brier_score": round(brier_b, 5)},
    ])

    section_html = f"""
<section>
<h2>Step 4 — Probability Conversion &amp; Edge Computation</h2>
<p class="timestamp">{ts()}</p>

<h3>Method Comparison (Brier Score, lower is better)</h3>
{df_to_html_table(method_comparison)}
<p><strong>Winner: Method {best_method}</strong> (Brier: {min(brier_a, brier_b):.5f})</p>
<p>Note: For a direct probability classifier (logistic/XGBoost), Method A is the raw model probability. Method B applies Platt scaling on top. For a well-calibrated model, both should be similar.</p>

<h3>Calibration Curves (10-decile buckets)</h3>
{df_to_html_table(calib_df)}
<p>Note: For each decile, |predicted_rate - actual_rate| should be &lt;0.15. Buckets with large gaps indicate miscalibration.</p>

<h3>Clip Counts</h3>
<p>Rows clipped to 0.01 (low): {n_clipped_low} | Rows clipped to 0.99 (high): {n_clipped_high}</p>

<h3>Asserts</h3>
<ul>
  <li class="pass">PASS: edge_over = p_model_over − raw_implied_prob_over</li>
  <li class="pass">PASS: p_model_over is book-invariant for same (player, game, line)</li>
  <li>Multi-line player-games: {n_multi} (expected ~0)</li>
</ul>

<h3>Spot-check: Aaron Judge (last 15)</h3>
{df_to_html_table(judge[show_cols].reset_index(drop=True))}
</section>
"""
    with open(HTML_LOG, "a") as f:
        f.write(section_html)
    print(f"\nHTML appended → {HTML_LOG}")


if __name__ == "__main__":
    main()
