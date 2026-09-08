"""
Step 4 — Poisson CDF: yhat (raw K count) → P(over/under line).

Pipeline:
  yhat_oof (regression) → Poisson CDF → p_model_over / p_model_under
  edge_over  = p_model_over  - raw_implied_prob_over   (raw, vig-inclusive)
  edge_under = p_model_under - raw_implied_prob_under

Required assert: p_model_over is book-invariant — same for every book
  at the same (player, game_date, offered_line).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

OOF_PATH  = Path.home() / "Downloads/tmp/mlb_batter_strikeouts_oof_regression.parquet"
OUT_PATH  = Path.home() / "Downloads/tmp/mlb_batter_strikeouts_step4_scored.parquet"


def poisson_prob_over(mu: np.ndarray, line: np.ndarray) -> np.ndarray:
    """P(X > line) for half-integer lines using floor trick."""
    k = np.floor(line).astype(int)
    return 1.0 - poisson.cdf(k, mu)


def brier_score(y_true: np.ndarray, p_pred: np.ndarray) -> float:
    return float(np.mean((p_pred - y_true) ** 2))


def main():
    print("Loading OOF parquet...")
    df = pd.read_parquet(OOF_PATH)
    print(f"  {len(df):,} rows")

    valid = df["yhat_oof"].notna()
    print(f"  {valid.sum():,} rows with yhat_oof ({(~valid).sum():,} NaN — first OOF fold)")

    df_v = df[valid].copy()

    # ── Poisson CDF ─────────────────────────────────────────────────────────
    mu   = df_v["yhat_oof"].values
    line = df_v["offered_line"].values

    df_v["p_model_over"]  = poisson_prob_over(mu, line)
    df_v["p_model_under"] = 1.0 - df_v["p_model_over"]

    # ── Edge (raw / vig-inclusive) ────────────────────────────────────────
    df_v["edge_over"]  = df_v["p_model_over"]  - df_v["raw_implied_prob_over"]
    df_v["edge_under"] = df_v["p_model_under"] - df_v["raw_implied_prob_under"]

    # ── Required assert: p_model_over is book-invariant ──────────────────
    print("\n=== Required assert: p_model_over is book-invariant ===")
    check = (df_v.groupby(["player_key", "game_date", "offered_line"])["p_model_over"]
                 .apply(lambda x: round(x, 8).nunique()))
    n_varying = (check > 1).sum()
    assert n_varying == 0, (
        f"p_model_over NOT book-invariant — {n_varying} groups vary. "
        f"yhat_oof must be identical across books at the same player-game-line."
    )
    print("  ✅ p_model_over is book-invariant")

    # ── Brier scores ──────────────────────────────────────────────────────
    df_v_clean = df_v.dropna(subset=["over_flag"])
    bs_over    = brier_score(df_v_clean["over_flag"].values, df_v_clean["p_model_over"].values)
    bs_novig   = brier_score(df_v_clean["over_flag"].values, df_v_clean["novig_prob_over"].values)
    print(f"\nBrier score — model:       {bs_over:.5f}")
    print(f"Brier score — novig (mkt): {bs_novig:.5f}")
    print(f"  (lower = better; model vs market as baseline)")

    # ── Calibration by line ───────────────────────────────────────────────
    print("\n=== Calibration by line ===")
    cal = (df_v_clean.groupby("offered_line").agg(
        n          = ("over_flag", "count"),
        over_rate  = ("over_flag", "mean"),
        p_model_over  = ("p_model_over", "mean"),
        novig_prob_over = ("novig_prob_over", "mean"),
        raw_prob_over   = ("raw_implied_prob_over", "mean"),
    ).reset_index())
    cal["model_gap"] = cal["over_rate"] - cal["p_model_over"]
    cal["mkt_gap"]   = cal["over_rate"] - cal["novig_prob_over"]

    print(f"\n{'Line':>6} {'n':>7} {'over_rate':>10} {'p_model':>10} {'novig':>10} {'model_gap':>10} {'mkt_gap':>10}")
    for _, r in cal.iterrows():
        print(f"{r['offered_line']:>6.1f} {r['n']:>7,} {r['over_rate']:>10.3f} "
              f"{r['p_model_over']:>10.3f} {r['novig_prob_over']:>10.3f} "
              f"{r['model_gap']:>10.3f} {r['mkt_gap']:>10.3f}")

    # ── Edge distribution (UNDER focus — the strategy) ────────────────────
    print("\n=== Edge distribution (UNDER strategy) ===")
    for thresh in [0.00, 0.03, 0.05, 0.07, 0.10]:
        q = df_v_clean[df_v_clean["edge_under"] >= thresh]
        if len(q) == 0:
            print(f"  edge_under >= {thresh:.2f}: 0 bets")
            continue
        under_hit = (q["over_flag"] == 0).mean()
        print(f"  edge_under >= {thresh:.2f}: {len(q):,} rows | under_hit={under_hit:.3f}")

    # ── Save ──────────────────────────────────────────────────────────────
    # Merge predictions back to full df (NaN rows keep NaN p_model)
    df = df.merge(
        df_v[["player_key", "game_date", "bookmaker", "offered_line",
              "p_model_over", "p_model_under", "edge_over", "edge_under"]],
        on=["player_key", "game_date", "bookmaker", "offered_line"],
        how="left",
    )
    df.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
