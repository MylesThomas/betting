"""
Step 6 — In-sample grid search for NFL WR/TE receiving yards.

Uses model trained on all data (artifacts in ~/Downloads/tmp/nfl_rec_yards_artifacts/).
Same sweep dimensions as step 5 OOS — used to characterize the strategy ceiling
and verify that the best OOS strategies also show positive IS ROI.

Output:
  ~/Downloads/tmp/nfl_rec_yards_step6_v2.csv
"""

from __future__ import annotations

import itertools
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.stats import nbinom

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
LABELED_PATH = Path.home() / "Downloads" / "tmp" / "nfl_rec_yards_per_book.parquet"
ARTIFACT_DIR = Path.home() / "Downloads" / "tmp" / "nfl_rec_yards_artifacts"
CONFIG_PATH  = Path(__file__).parent.parent / "config" / "model_config.yaml"
OUT_CSV      = Path.home() / "Downloads" / "tmp" / "nfl_rec_yards_step6_v2.csv"

TARGET                  = "receiving_yards"
HYBRID_NEGBIN_THRESHOLD = 20.5
N_BOOT                  = 5_000
RNG                     = np.random.default_rng(42)
KEEP_POSITIONS          = ["WR", "TE"]
BEST_FEATS = [
    "offered_line", "game_total", "proj_own_score",
    "receiving_yards_L8", "target_share_L8", "snap_pct_L8",
    "pos_TE", "market_under_prob",
]


def _compute_p_hybrid(df: pd.DataFrame, artifacts: dict,
                      shrinkage: float, pred_method: str) -> pd.Series:
    ols, residuals = artifacts["ols"], artifacts["residuals"]
    nb_coefs, nb_alpha = artifacts["nb_coefs"], artifacts["nb_alpha"]

    p    = pd.Series(np.nan, index=df.index)
    mask = df[BEST_FEATS].notna().all(axis=1)
    if not mask.any():
        return p

    sub     = df[mask]
    X       = sub[BEST_FEATS].to_numpy(dtype=float)
    X_const = np.column_stack([np.ones(len(X)), X])
    line    = sub["offered_line"].to_numpy(dtype=float)

    ols_pred = ols.predict(X)
    nb_mu    = np.clip(np.exp(X_const @ nb_coefs), 1e-3, None)

    if pred_method == "consensus_line":
        yhat_ols = sub["consensus_line"].to_numpy(dtype=float)
        yhat_nb  = sub["consensus_line"].to_numpy(dtype=float)
    else:
        yhat_ols = ols_pred.copy()
        yhat_nb  = nb_mu.copy()

    if pred_method == "model" and shrinkage > 0:
        mean_ols = float(ols_pred.mean())
        mean_nb  = float(nb_mu.mean())
        yhat_ols = (1 - shrinkage) * yhat_ols + shrinkage * mean_ols
        yhat_nb  = (1 - shrinkage) * yhat_nb  + shrinkage * mean_nb

    bt_mask = line < HYBRID_NEGBIN_THRESHOLD
    nb_mask = ~bt_mask
    p_hyb   = np.empty(len(sub))

    if bt_mask.any():
        pred_bt = yhat_ols[bt_mask]
        line_bt = line[bt_mask]
        samp    = RNG.choice(residuals, size=(bt_mask.sum(), N_BOOT))
        p_hyb[bt_mask] = ((pred_bt[:, None] + samp) <= line_bt[:, None]).mean(axis=1)

    if nb_mask.any():
        n_nb = 1.0 / nb_alpha
        mu   = np.clip(yhat_nb[nb_mask], 1e-3, None)
        p_hyb[nb_mask] = nbinom.cdf(
            np.floor(line[nb_mask]).astype(int),
            n=n_nb, p=n_nb / (n_nb + mu),
        )

    p_hyb = np.clip(p_hyb, 0.01, 0.99)
    p[mask] = p_hyb
    return p


def _max_drawdown(bets: pd.DataFrame) -> float:
    ordered = bets.sort_values(["season", "week"])
    pnl     = np.where(ordered["bet_correct"] == 1, ordered["payout"] - 1, -1.0)
    cs  = np.cumsum(pnl)
    mx  = np.maximum.accumulate(cs)
    return float((mx - cs).max()) if len(cs) else 0.0


def sweep(df: pd.DataFrame, artifacts: dict, cfg: dict) -> pd.DataFrame:
    gs             = cfg["nfl_rec_yards_model"]["grid_search"]
    edges          = gs["edge_threshold"]
    directions     = gs["direction"]
    buckets        = gs["odds_bucket"]
    shrinkages     = gs["shrinkage"]
    methods        = gs["prediction_method"]
    min_books_opts = gs["min_books"]
    line_mins      = gs["line_min"]
    line_maxes     = gs["line_max"]

    total_rows = len(df[df[BEST_FEATS].notna().all(axis=1)])
    rows = []

    # Pre-compute p_hybrid for each (method, shrinkage) combo
    ph_cache: dict[tuple, pd.Series] = {}
    for method, shrinkage in itertools.product(methods, shrinkages):
        if method == "consensus_line" and shrinkage > 0:
            continue
        key = (method, shrinkage)
        if key not in ph_cache:
            print(f"    Computing p_hybrid: method={method}, shrinkage={shrinkage}...")
            ph_cache[key] = _compute_p_hybrid(df, artifacts, shrinkage, method)

    combos = list(itertools.product(
        methods, shrinkages, edges, directions, buckets, min_books_opts, line_mins, line_maxes
    ))
    print(f"  Running {len(combos):,} combos across {total_rows:,} scored rows...")

    for method, shrinkage, edge, direction, bucket, min_books, lmin, lmax in combos:
        if method == "consensus_line" and shrinkage > 0:
            continue

        key    = (method, shrinkage)
        p_hyb  = ph_cache[key]
        p_mkt  = df["market_under_prob"]
        edge_v = p_hyb - p_mkt

        scoreable = (
            df[BEST_FEATS].notna().all(axis=1) &
            df["offered_line"].between(lmin, lmax) &
            (df["n_books"] >= min_books if "n_books" in df.columns else True)
        )
        n_universe = int(scoreable.sum())

        rec = pd.Series("PASS", index=df.index)
        rec[edge_v >  0.001] = "UNDER"
        rec[edge_v < -0.001] = "OVER"

        mop         = 1.0 - p_mkt
        bucket_mask = pd.Series(True, index=df.index)
        if bucket == "plus_odds":
            bucket_mask = mop < 0.50
        elif bucket == "minus_odds":
            bucket_mask = mop > 0.50

        # EV filter: model P(side) must exceed raw (vig-inclusive) book probability
        ev_over  = (1.0 - p_hyb) > df["raw_over_prob"]
        ev_under = p_hyb          > df["raw_under_prob"]

        if direction == "OVER":
            bet_mask = scoreable & bucket_mask & (edge_v.abs() >= edge) & (rec == "OVER") & ev_over
        elif direction == "UNDER":
            bet_mask = scoreable & bucket_mask & (edge_v.abs() >= edge) & (rec == "UNDER") & ev_under
        else:
            bet_mask = scoreable & bucket_mask & (edge_v.abs() >= edge) & (
                ((rec == "OVER") & ev_over) | ((rec == "UNDER") & ev_under)
            )

        bets  = df[bet_mask].copy()
        n_bets = len(bets)

        if n_bets == 0:
            rows.append({
                "prediction_method": method, "shrinkage": shrinkage,
                "edge": edge, "direction": direction, "odds_bucket": bucket,
                "min_books": min_books, "line_min": lmin, "line_max": lmax,
                "n_bets": 0, "pct_of_universe": 0.0,
                "win_rate": np.nan, "push_rate": np.nan, "units_won": np.nan,
                "roi": np.nan, "mean_payout": np.nan, "max_drawdown": np.nan,
                "mean_edge_pp": np.nan, "mean_line": np.nan,
            })
            continue

        bets["_rec"]    = rec[bet_mask].values
        actual_under    = (bets[TARGET] <= bets["offered_line"]).astype(float)
        bet_correct     = np.where(
            bets["_rec"] == "UNDER", actual_under,
            np.where(bets["_rec"] == "OVER", 1 - actual_under, np.nan),
        )
        push = (bets[TARGET] == bets["offered_line"]).astype(float)
        bets["bet_correct"] = bet_correct

        # Payout: use raw (vig-inclusive) implied probs — actual prices you collect
        over_pay  = 1.0 / np.clip(bets["raw_over_prob"].to_numpy(float),  1e-6, 1.0) - 1.0
        under_pay = 1.0 / np.clip(bets["raw_under_prob"].to_numpy(float), 1e-6, 1.0) - 1.0
        side_pay  = np.where(bets["_rec"].to_numpy() == "OVER", over_pay, under_pay)
        bets["payout"] = np.where(bets["bet_correct"] == 1, side_pay, 0.0)

        n_win  = float(np.nansum(bet_correct * (1 - push)))
        n_push = float(push.sum())
        n_loss = n_bets - n_win - n_push

        units_won   = float((bets["bet_correct"] * (1 - push) * side_pay).sum()) - n_loss
        mean_payout = float(side_pay.mean())
        roi         = units_won / n_bets if n_bets else np.nan

        rows.append({
            "prediction_method": method, "shrinkage": shrinkage,
            "edge": edge, "direction": direction, "odds_bucket": bucket,
            "min_books": min_books, "line_min": lmin, "line_max": lmax,
            "n_bets":          n_bets,
            "pct_of_universe": n_bets / n_universe if n_universe else np.nan,
            "win_rate":        float(np.nanmean(bet_correct)) if n_bets else np.nan,
            "push_rate":       n_push / n_bets,
            "units_won":       round(units_won, 2),
            "roi":             round(roi, 4) if not np.isnan(roi) else np.nan,
            "mean_payout":     round(mean_payout, 4),
            "max_drawdown":    round(_max_drawdown(bets), 2),
            "mean_edge_pp":    round(float(edge_v[bet_mask].abs().mean()) * 100, 2),
            "mean_line":       round(float(bets["offered_line"].mean()), 1),
        })

    return pd.DataFrame(rows)


def main():
    print("\nNFL Rec Yards — Step 6: IS Grid Search")
    print("=" * 50)

    cfg = yaml.safe_load(CONFIG_PATH.read_text())

    print("\n  Loading labeled data...")
    df = pd.read_parquet(LABELED_PATH)
    df = df[df["position"].isin(KEEP_POSITIONS)].copy()
    df["pos_TE"] = (df["position"] == "TE").astype(int)
    print(f"    {len(df):,} rows  |  seasons: {sorted(df['season'].unique())}")

    # Compute consensus_line per (player_name_norm, game_id)
    consensus = (
        df.groupby(["player_name_norm", "game_id"])["offered_line"]
        .mean().rename("consensus_line").reset_index()
    )
    df = df.merge(consensus, on=["player_name_norm", "game_id"], how="left")

    print("\n  Loading IS artifacts...")
    artifacts = {
        "ols":       joblib.load(ARTIFACT_DIR / "ols_pipeline.joblib"),
        "residuals": np.load(ARTIFACT_DIR / "residuals.npy"),
        "nb_coefs":  np.load(ARTIFACT_DIR / "nb_coefs.npy"),
        "nb_alpha":  float(np.load(ARTIFACT_DIR / "nb_alpha.npy")[0]),
    }

    print("\n  Running grid search sweep...")
    results = sweep(df, artifacts, cfg)
    results = results[
        ~((results["prediction_method"] == "consensus_line") & (results["shrinkage"] > 0))
    ]
    results = results.sort_values("units_won", ascending=False).reset_index(drop=True)
    results.to_csv(OUT_CSV, index=False)

    print(f"\n  Grid search complete: {len(results):,} combos")
    print(f"  Saved → {OUT_CSV}")

    best = results[results["n_bets"] >= 50].head(10)
    print(f"\n  Top 10 strategies (n_bets ≥ 50):")
    print(best[["prediction_method", "shrinkage", "edge", "direction", "odds_bucket",
                "min_books", "n_bets", "win_rate", "units_won", "roi", "max_drawdown"]].to_string(index=False))
    print()


if __name__ == "__main__":
    main()
