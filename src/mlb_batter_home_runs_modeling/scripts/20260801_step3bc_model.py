"""
Steps 3b + 3c — XGBoost feature sweep + final model selection.

3b: XGBoost n=1 sweep (same features as 3a), compare AUC vs logistic.
3c: Combo models (logistic + XGBoost) with top features, OOF CV.
    Save winning model to S3.

Output:
  models/mlb_batter_hr_model.joblib       (local)
  s3://the-odds-api-mt/mlb/batter_home_runs_model/model/mlb_batter_hr_model.joblib
  Appends Step 3b + 3c sections to HTML log.
"""
from __future__ import annotations

import sys
import warnings
import joblib
from datetime import datetime
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import boto3
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed — 3b will be skipped")

warnings.filterwarnings("ignore")
REPO_ROOT   = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

LOCAL_SPINE  = Path.home() / "Downloads/tmp/mlb_batter_hr_spine.parquet"
LOCAL_MODEL  = REPO_ROOT / "src/mlb_batter_home_runs_modeling/models/mlb_batter_hr_model.joblib"
S3_BUCKET    = "the-odds-api-mt"
S3_MODEL_KEY = "mlb/batter_home_runs_model/model/mlb_batter_hr_model.joblib"
HTML_LOG     = REPO_ROOT / "knowledge-base/raw/20260801-mlb-batter-home-runs.html"
ET           = ZoneInfo("America/New_York")
N_SPLITS     = 5
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


# ── Feature sets to try in 3c combos ─────────────────────────────────────────
# Adjusted after seeing 3a results — keep best ~5-8 features per combo.
# These are reasonable starting combos — adapt after reviewing 3a output.
NUMERIC_FEATURES_FULL = [
    "hr_roll_L5",
    "hr_roll_L10",
    "hr_roll_L20",
    "hr_roll_career",
    "ab_roll_career",
    "opp_hr_rate_career",
    "min_raw_implied_prob_under",
    "max_raw_implied_prob_under",
    "is_home",
]

NUMERIC_FEATURES_MARKET_ONLY = [
    "min_raw_implied_prob_under",
    "max_raw_implied_prob_under",
    "min_raw_implied_prob_over",
]

NUMERIC_FEATURES_STATS_ONLY = [
    "hr_roll_L5",
    "hr_roll_L20",
    "hr_roll_career",
    "ab_roll_career",
    "opp_hr_rate_career",
    "is_home",
]


def oof_model(df: pd.DataFrame, numeric_feats: list[str], cat_feats: list[str],
              model_type: str = "logistic") -> tuple[np.ndarray, float, object]:
    """
    OOF training with TimeSeriesSplit.
    Returns (oof_predictions, auc, fitted_model_on_full_data).
    """
    all_feats = [f for f in numeric_feats + cat_feats if f in df.columns]
    valid = df[all_feats + [TARGET]].dropna()
    valid = valid.sort_values("game_date") if "game_date" in valid.columns else valid.reset_index(drop=True)

    # Label-encode categoricals
    le_maps = {}
    for feat in cat_feats:
        if feat in valid.columns:
            le = LabelEncoder()
            valid = valid.copy()
            valid[feat] = le.fit_transform(valid[feat].astype(str))
            le_maps[feat] = le

    X = valid[[f for f in all_feats if f in valid.columns]].values
    y = valid[TARGET].values

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    y_pred_proba = np.zeros(len(valid))

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = y[train_idx]

        if model_type == "logistic":
            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, C=0.5)),
            ])
        elif model_type == "xgboost" and HAS_XGB:
            pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
            clf = XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=pos_weight,
                eval_metric="auc",
                verbosity=0,
                random_state=42,
            )
        else:
            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, C=0.5)),
            ])

        clf.fit(X_train, y_train)
        y_pred_proba[val_idx] = clf.predict_proba(X_val)[:, 1]

    auc = roc_auc_score(y, y_pred_proba) if len(np.unique(y)) > 1 else 0.5

    # Fit on full data for deployment
    if model_type == "logistic":
        full_model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=0.5)),
        ])
    elif model_type == "xgboost" and HAS_XGB:
        pos_weight = (len(y) - y.sum()) / max(y.sum(), 1)
        full_model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            eval_metric="auc",
            verbosity=0,
            random_state=42,
        )
    else:
        full_model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=0.5)),
        ])

    full_model.fit(X, y)

    return y_pred_proba, auc, full_model, valid.index.tolist(), [f for f in all_feats if f in valid.columns]


def main() -> None:
    if not LOCAL_SPINE.exists():
        print(f"ERROR: {LOCAL_SPINE} not found — run 20260801_build_spine.py first")
        sys.exit(1)

    spine = pd.read_parquet(LOCAL_SPINE)
    print(f"Loaded spine: {len(spine):,} rows")

    # Deduplicate to player-game level
    df = (
        spine
        .dropna(subset=[TARGET])
        .sort_values(["player_key", "game_date"])
        .drop_duplicates(subset=["player_key", "game_date"])
        .sort_values("game_date")
        .reset_index(drop=True)
    )
    print(f"Player-game rows: {len(df):,}")

    # ── Step 3b: XGBoost n=1 sweep ──────────────────────────────────────────────
    print("\n=== STEP 3b: XGBoost n=1 sweep ===")
    step3b_results = []
    numeric_candidates = [
        "hr_roll_L1", "hr_roll_L3", "hr_roll_L5", "hr_roll_L10", "hr_roll_L20",
        "hr_roll_season", "hr_roll_career",
        "ab_roll_L5", "ab_roll_career", "games_played_career",
        "opp_hr_rate_career", "opp_hr_rate_L20",
        "is_home",
        "min_line", "max_line",
        "min_raw_implied_prob_over", "max_raw_implied_prob_over",
        "min_raw_implied_prob_under", "max_raw_implied_prob_under",
        "consensus_line",
    ]

    if HAS_XGB:
        for feat in numeric_candidates:
            if feat not in df.columns:
                continue
            null_rate = df[feat].isna().mean()
            if null_rate > 0.5:
                continue
            try:
                _, auc, _, _, _ = oof_model(df[["game_date", feat, TARGET]], [feat], [], model_type="xgboost")
                step3b_results.append({"feature": feat, "model_type": "xgboost", "auc": round(auc, 4)})
                print(f"  {feat:35s}  xgb_auc={auc:.4f}")
            except Exception as e:
                print(f"  {feat}: ERROR — {e}")

    step3b_df = pd.DataFrame(step3b_results).sort_values("auc", ascending=False).reset_index(drop=True)

    # ── Step 3c: Combo models ────────────────────────────────────────────────────
    print("\n=== STEP 3c: Combo models ===")

    combos = [
        {
            "label": "Full numeric combo (logistic)",
            "numeric": NUMERIC_FEATURES_FULL,
            "cat": [],
            "model_type": "logistic",
        },
        {
            "label": "Market features only (logistic)",
            "numeric": NUMERIC_FEATURES_MARKET_ONLY,
            "cat": [],
            "model_type": "logistic",
        },
        {
            "label": "Stats features only (logistic)",
            "numeric": NUMERIC_FEATURES_STATS_ONLY,
            "cat": [],
            "model_type": "logistic",
        },
    ]

    if HAS_XGB:
        combos += [
            {
                "label": "Full numeric combo (XGBoost)",
                "numeric": NUMERIC_FEATURES_FULL,
                "cat": [],
                "model_type": "xgboost",
            },
        ]

    combo_results = []
    best_auc = 0
    best_combo = None
    best_model = None
    best_feats_used = None

    for combo in combos:
        feats_available = [f for f in combo["numeric"] + combo["cat"] if f in df.columns]
        if not feats_available:
            continue
        try:
            df_subset = df[["game_date"] + feats_available + [TARGET]].copy()
            y_pred, auc, full_model, idx, feats_used = oof_model(
                df_subset, combo["numeric"], combo["cat"], combo["model_type"]
            )
            y_true = df.loc[idx, TARGET].values
            y_bin  = (y_pred >= 0.5).astype(int)
            prec  = precision_score(y_true, y_bin, zero_division=0)
            rec   = recall_score(y_true, y_bin, zero_division=0)
            f1    = f1_score(y_true, y_bin, zero_division=0)

            combo_results.append({
                "combo":       combo["label"],
                "model_type":  combo["model_type"],
                "n_features":  len(feats_used),
                "n_samples":   len(idx),
                "auc":         round(auc, 4),
                "precision":   round(prec, 4),
                "recall":      round(rec, 4),
                "f1":          round(f1, 4),
                "features":    ", ".join(feats_used),
            })
            print(f"  {combo['label']:45s}  auc={auc:.4f}  n={len(feats_used)} feats")

            if auc > best_auc:
                best_auc = auc
                best_combo = combo
                best_model = full_model
                best_feats_used = feats_used

        except Exception as e:
            print(f"  {combo['label']}: ERROR — {e}")
            import traceback; traceback.print_exc()

    combo_df = pd.DataFrame(combo_results).sort_values("auc", ascending=False).reset_index(drop=True)
    print("\n=== Combo Rankings ===")
    print(combo_df[["combo", "model_type", "n_features", "n_samples", "auc", "precision", "recall", "f1"]].to_string(index=False))

    # ── yhat book-invariant assert ──────────────────────────────────────────────
    # (Runs on the full spine, not just player-game deduped)
    if best_model is not None:
        print("\n=== yhat book-invariant assert ===")
        # Score the full spine with best model
        valid_spine = spine[[f for f in best_feats_used if f in spine.columns] + ["player_key", "game_date", "offered_line", TARGET]].dropna(subset=[f for f in best_feats_used if f in spine.columns])
        cat_feats_in_model = [f for f in (best_combo.get("cat") or []) if f in valid_spine.columns]
        for feat in cat_feats_in_model:
            le = LabelEncoder()
            valid_spine = valid_spine.copy()
            valid_spine[feat] = le.fit_transform(valid_spine[feat].astype(str))
        X_spine = valid_spine[[f for f in best_feats_used if f in valid_spine.columns]].values
        valid_spine = valid_spine.copy()
        valid_spine["yhat"] = best_model.predict_proba(X_spine)[:, 1]
        yhat_range = valid_spine.groupby(["player_key", "game_date", "offered_line"])["yhat"].agg(
            lambda x: x.max() - x.min()
        )
        max_drift = yhat_range.max()
        n_drifty = (yhat_range > 1e-8).sum()
        assert n_drifty == 0, (
            f"yhat is NOT book-invariant — {n_drifty} (player, game, line) groups vary "
            f"by more than 1e-8 (max_drift={max_drift:.2e}). A per-book feature may be in the model."
        )
        print(f"  PASS: yhat is book-invariant across all books (max drift={max_drift:.2e}).")

    # ── Save winning model ──────────────────────────────────────────────────────
    if best_model is not None:
        LOCAL_MODEL.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": best_model,
            "features": best_feats_used,
            "model_type": best_combo["model_type"],
            "target": TARGET,
            "sklearn_version": sklearn.__version__,
        }, LOCAL_MODEL)
        print(f"  Saved model locally → {LOCAL_MODEL}")

        buf = BytesIO()
        joblib.dump({"model": best_model, "features": best_feats_used,
                     "model_type": best_combo["model_type"], "target": TARGET,
                     "sklearn_version": sklearn.__version__}, buf)
        buf.seek(0)
        boto3.client("s3").put_object(Bucket=S3_BUCKET, Key=S3_MODEL_KEY, Body=buf.read())
        print(f"  Saved model to S3 → s3://{S3_BUCKET}/{S3_MODEL_KEY}")
        print(f"  sklearn version: {sklearn.__version__}")

    # HTML
    section_html = f"""
<section>
<h2>Step 3b — XGBoost n=1 Feature Sweep</h2>
<p class="timestamp">{ts()}</p>
{df_to_html_table(step3b_df, "XGBoost AUC by feature (descending)")}
</section>

<section>
<h2>Step 3c — Combo Models (Logistic + XGBoost, OOF)</h2>
<p class="timestamp">{ts()}</p>

<h3>Combo Rankings</h3>
{df_to_html_table(combo_df, "All combos sorted by AUC")}

<h3>Winning Model</h3>
<div class="config-block">
<p><span class="key">Best combo:</span> {best_combo['label'] if best_combo else 'N/A'}</p>
<p><span class="key">Model type:</span> {best_combo['model_type'] if best_combo else 'N/A'}</p>
<p><span class="key">Features:</span> {', '.join(best_feats_used) if best_feats_used else 'N/A'}</p>
<p><span class="key">OOF AUC:</span> {best_auc:.4f}</p>
<p><span class="key">sklearn version:</span> {sklearn.__version__}</p>
<p><span class="key">S3 model key:</span> s3://{S3_BUCKET}/{S3_MODEL_KEY}</p>
</div>

<h3>book-invariant yhat assert</h3>
<p class="{'pass' if best_model else 'fail'}">{'PASS — yhat identical across all books for same (player, game, line)' if best_model else 'FAIL or not run'}</p>
</section>
"""
    with open(HTML_LOG, "a") as f:
        f.write(section_html)
    print(f"\nHTML appended → {HTML_LOG}")


if __name__ == "__main__":
    main()
