"""
Step 4 — Outcome distribution → P(over) calibration.

Compares three approaches:
  1. Market baseline: novig_prob_over (already knows the line)
  2. LR direct classification: OOF logistic regression P(over_flag=1)
  3. Poisson CDF: Ridge OOF prediction → lambda → P(K > line)

Evaluates Brier score + calibration by decile at each line.
Produces calibration table for HTML log.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

FEATURES_PATH = Path.home() / "Downloads/tmp/mlb_batter_strikeouts_features.parquet"
OUT_PATH      = Path.home() / "Downloads/tmp/mlb_batter_strikeouts_4_calibration.csv"

N_FOLDS = 5

LR_FEATS  = ["novig_prob_over", "offered_line", "k_rate_career", "k_roll_L20"]
REG_FEATS = ["novig_prob_over", "offered_line", "k_rate_career", "k_roll_L20", "opp_k_rate_career"]


def temporal_oof(df: pd.DataFrame, feature_cols: list[str], target_col: str, model_type: str):
    df_s = df.dropna(subset=feature_cols + [target_col]).sort_values("game_date").reset_index(drop=True)
    n = len(df_s)
    fold_size = n // N_FOLDS
    oof_preds = np.full(n, np.nan)

    for fold in range(N_FOLDS):
        train_end = fold_size * (fold + 1)
        val_idx   = df_s.index[train_end:] if fold == N_FOLDS - 1 else df_s.index[train_end: train_end + fold_size]
        train_idx = df_s.index[:train_end]
        if len(train_idx) < 50 or len(val_idx) < 10:
            continue

        X_tr = df_s.loc[train_idx, feature_cols].values
        y_tr = df_s.loc[train_idx, target_col].values
        X_vl = df_s.loc[val_idx,   feature_cols].values

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_vl = scaler.transform(X_vl)

        if model_type == "logistic":
            m = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
            m.fit(X_tr, y_tr)
            oof_preds[val_idx] = m.predict_proba(X_vl)[:, 1]
        else:
            m = Ridge(alpha=1.0)
            m.fit(X_tr, y_tr)
            oof_preds[val_idx] = m.predict(X_vl)

    valid = ~np.isnan(oof_preds)
    return df_s, oof_preds, valid


def poisson_p_over(lam: np.ndarray, line: float) -> np.ndarray:
    """P(X > line) where X ~ Poisson(lam). line is 0.5 or 1.5."""
    k_threshold = int(line + 0.5)  # 0.5 → 1, 1.5 → 2
    # P(X >= k_threshold) = 1 - CDF(k_threshold - 1)
    return 1.0 - poisson.cdf(k_threshold - 1, lam)


def calibration_table(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    fraction_pos, mean_pred = calibration_curve(y_true, y_pred, n_bins=n_bins, strategy="quantile")
    brier = brier_score_loss(y_true, y_pred)
    rows = []
    for mp, fp in zip(mean_pred, fraction_pos):
        rows.append({"pred_bin_center": round(mp, 3), "actual_over_rate": round(fp, 3),
                     "gap": round(fp - mp, 3)})
    return pd.DataFrame(rows), brier


def main():
    print("Loading features...")
    df = pd.read_parquet(FEATURES_PATH)

    results_all = []

    for line_val in [0.5, 1.5]:
        print(f"\n--- Line = {line_val} ---")
        sub = df[df["offered_line"] == line_val].copy()
        sub = sub.dropna(subset=LR_FEATS + REG_FEATS + ["over_flag", "strikeouts", "novig_prob_over"])
        print(f"  {len(sub):,} rows")

        # 1. Market baseline
        y_true = sub["over_flag"].values
        mkt_pred = sub["novig_prob_over"].values
        _, mkt_brier = calibration_table(y_true, np.clip(mkt_pred, 1e-6, 1-1e-6))
        print(f"  Market baseline Brier = {mkt_brier:.4f}")

        # 2. LR direct classification
        df_s_lr, oof_lr, valid_lr = temporal_oof(sub, LR_FEATS, "over_flag", "logistic")
        y_lr = df_s_lr["over_flag"].values[valid_lr]
        p_lr = oof_lr[valid_lr]
        _, lr_brier = calibration_table(y_lr, np.clip(p_lr, 1e-6, 1-1e-6))
        print(f"  LR classification Brier = {lr_brier:.4f}")

        # 3. Poisson CDF from Ridge regression
        df_s_r, oof_r, valid_r = temporal_oof(sub, REG_FEATS, "strikeouts", "ridge")
        lam = np.clip(oof_r[valid_r], 0.05, 10.0)
        p_poisson = poisson_p_over(lam, line_val)
        y_r = df_s_r["over_flag"].values[valid_r]
        _, pois_brier = calibration_table(y_r, np.clip(p_poisson, 1e-6, 1-1e-6))
        print(f"  Poisson CDF Brier = {pois_brier:.4f}")

        # Calibration tables
        for method, y_t, y_p in [
            ("market", y_true, np.clip(mkt_pred, 1e-6, 1-1e-6)),
            ("lr_direct", y_lr, np.clip(p_lr, 1e-6, 1-1e-6)),
            ("poisson_cdf", y_r, np.clip(p_poisson, 1e-6, 1-1e-6)),
        ]:
            cal_df, brier = calibration_table(y_t, y_p)
            cal_df["line"] = line_val
            cal_df["method"] = method
            cal_df["brier_score"] = round(brier, 4)
            results_all.append(cal_df)

        # Summary for this line
        print(f"\n  {'Method':<20} {'Brier':>8}  {'Max |gap|':>10}")
        for method, y_t, y_p, b in [
            ("market",     y_true, np.clip(mkt_pred, 1e-6, 1-1e-6), mkt_brier),
            ("lr_direct",  y_lr,   np.clip(p_lr, 1e-6, 1-1e-6),     lr_brier),
            ("poisson_cdf",y_r,    np.clip(p_poisson, 1e-6, 1-1e-6), pois_brier),
        ]:
            cal_df, _ = calibration_table(y_t, y_p)
            max_gap = cal_df["gap"].abs().max()
            print(f"  {method:<20} {b:>8.4f}  {max_gap:>10.4f}")

    out = pd.concat(results_all, ignore_index=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
