"""
Calibration check for the NFL rec yards model.

For each line bucket, compares:
  - vigged_prob   : raw implied P(under) from under price
  - market_under_prob : de-vigged P(under)
  - model_under_prob  : hybrid P(under)
  - actual_under_rate : fraction of cases where actual < line

Run:
  python src/nfl_rec_yards_modeling/scripts/line_calibration.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import nbinom

warnings.filterwarnings("ignore")

LABELED_PATH = Path.home() / "Downloads" / "tmp" / "nfl_rec_yards_per_book.parquet"
ARTIFACT_DIR = Path.home() / "Downloads" / "tmp" / "nfl_rec_yards_artifacts"

TARGET                  = "receiving_yards"
HYBRID_NEGBIN_THRESHOLD = 20.5
N_BOOT                  = 10_000
RNG                     = np.random.default_rng(42)

BEST_FEATS = [
    "offered_line", "game_total", "proj_own_score",
    "rec_yards_L8", "target_share_L8", "snap_pct_L8",
    "pos_TE", "market_under_prob",
]
KEEP_POSITIONS = ["WR", "TE"]

LINE_BINS        = [0, 24.5, 29.5, 34.5, 39.5, 44.5, 49.5, 54.5, 59.5, 69.5, 99.5, float("inf")]
LINE_LABELS      = ["≤24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-69", "70-99", "100+"]
SNAP_BINS        = [0, 0.25, 0.50, 0.75, 1.01]
SNAP_LABELS      = ["0-25%", "26-50%", "51-75%", "76-100%"]


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pos_TE"] = (df["position"] == "TE").astype(int)
    return df


def compute_model_probs(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    ols, residuals   = artifacts["ols"], artifacts["residuals"]
    nb_coefs, nb_alpha = artifacts["nb_coefs"], artifacts["nb_alpha"]

    result = df.copy()
    mask   = result[BEST_FEATS].notna().all(axis=1)
    idx    = result.index[mask]
    X      = result.loc[idx, BEST_FEATS].to_numpy(dtype=float)
    line   = result.loc[idx, "offered_line"].to_numpy(dtype=float)

    ols_pred = ols.predict(X)
    X_const  = np.column_stack([np.ones(len(X)), X])
    nb_mu    = np.exp(X_const @ nb_coefs)
    mu_c     = np.clip(nb_mu, 1e-3, None)
    n_nb     = 1.0 / nb_alpha
    p_nb     = nbinom.cdf(np.floor(line).astype(int), n=n_nb, p=n_nb / (n_nb + mu_c))
    samp     = RNG.choice(residuals, size=(len(ols_pred), N_BOOT))
    p_bt     = ((ols_pred[:, None] + samp) <= line[:, None]).mean(axis=1)
    p_hyb    = np.where(line < HYBRID_NEGBIN_THRESHOLD, p_bt, p_nb)

    result.loc[idx, "model_under_prob"] = np.round(p_hyb, 4)
    result["actual_under"] = (result[TARGET] <= result["offered_line"]).astype(float)

    def _vigged(row) -> float:
        p = row.get("raw_under_prob", np.nan)
        return float(p) if not pd.isna(p) else np.nan

    result["vigged_prob"] = result.apply(_vigged, axis=1)
    return result


def calibration_table(df: pd.DataFrame, bucket_col: str, labels: list[str]) -> pd.DataFrame:
    rows = []
    grp  = df[df["model_under_prob"].notna()].groupby(bucket_col, observed=True)
    for label, g in grp:
        rows.append({
            "bucket":             label,
            "n":                  len(g),
            "vigged_prob":        g["vigged_prob"].mean(),
            "market_under_prob":  g["market_under_prob"].mean(),
            "model_under_prob":   g["model_under_prob"].mean(),
            "actual_under_rate":  g["actual_under"].mean(),
            "model_vs_actual_pp": (g["model_under_prob"].mean() - g["actual_under"].mean()) * 100,
            "mkt_vs_actual_pp":   (g["market_under_prob"].mean() - g["actual_under"].mean()) * 100,
        })
    return pd.DataFrame(rows)


def main():
    print(f"\nNFL Rec Yards — Line Calibration\n")

    df = pd.read_parquet(LABELED_PATH)
    df = df[df["position"].isin(KEEP_POSITIONS)].copy()
    df = add_derived(df)
    print(f"  Dataset: {len(df):,} rows  |  seasons {sorted(df['season'].unique())}")

    missing = [f for f in ["ols_pipeline.joblib", "residuals.npy",
                            "nb_coefs.npy", "nb_alpha.npy"]
               if not (ARTIFACT_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing: {missing}. Run train.py first.")

    artifacts = {
        "ols":       joblib.load(ARTIFACT_DIR / "ols_pipeline.joblib"),
        "residuals": np.load(ARTIFACT_DIR / "residuals.npy"),
        "nb_coefs":  np.load(ARTIFACT_DIR / "nb_coefs.npy"),
        "nb_alpha":  float(np.load(ARTIFACT_DIR / "nb_alpha.npy")[0]),
    }

    print(f"  Computing model probabilities ({N_BOOT:,} bootstrap draws)...")
    scored = compute_model_probs(df, artifacts)

    scored["line_bucket"] = pd.cut(scored["offered_line"], bins=LINE_BINS, labels=LINE_LABELS)
    scored["snap_bucket"] = pd.cut(scored["snap_pct_L8"],  bins=SNAP_BINS, labels=SNAP_LABELS)

    print(f"\n{'='*75}")
    print("  TABLE 1: By line bucket")
    print(f"{'='*75}")
    t1 = calibration_table(scored, "line_bucket", LINE_LABELS)
    print(t1.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\n{'='*75}")
    print("  TABLE 2: By snap% bucket (model feature quality)")
    print(f"{'='*75}")
    t2 = calibration_table(scored, "snap_bucket", SNAP_LABELS)
    print(t2.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\n{'='*75}")
    print("  TABLE 3: By position")
    print(f"{'='*75}")
    t3 = calibration_table(scored, "position", ["WR", "TE"])
    print(t3.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\n{'='*75}")
    print("  Hybrid threshold check: Bootstrap vs NegBin boundary")
    print(f"  (rows near {HYBRID_NEGBIN_THRESHOLD:.1f})")
    print(f"{'='*75}")
    near = scored[scored["offered_line"].between(HYBRID_NEGBIN_THRESHOLD - 5, HYBRID_NEGBIN_THRESHOLD + 5)]
    print(f"  N rows near threshold: {len(near):,}")
    if len(near):
        print(f"  Mean actual under: {near['actual_under'].mean():.3f}")
        print(f"  Mean model prob:   {near['model_under_prob'].mean():.3f}")
    print()


if __name__ == "__main__":
    main()
