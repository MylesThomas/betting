"""
NFL WR/TE receiving yards inference pipeline.

Bet logic:
  p_market = de-vigged implied P(under) from this book's prices
  p_hybrid = model P(under):
               line < HYBRID_NEGBIN_THRESHOLD → Bootstrap (OLS residual draws)
               line ≥ HYBRID_NEGBIN_THRESHOLD → Negative Binomial NB2
  edge     = p_hybrid − p_market
  UNDER    if  edge >  EDGE_THRESHOLD
  OVER     if  edge < −EDGE_THRESHOLD
  PASS     otherwise

Run:
  python src/nfl_rec_yards_modeling/scripts/infer.py
  python src/nfl_rec_yards_modeling/scripts/infer.py --edge-threshold 0.04
"""

from __future__ import annotations

import argparse
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
OUT_ALL      = Path.home() / "Downloads" / "tmp" / "nfl_rec_yards_inference.parquet"
OUT_BETS     = Path.home() / "Downloads" / "tmp" / "nfl_rec_yards_bets.parquet"

TARGET = "receiving_yards"

# Calibrated; update after running line_calibration.py on OOS walk-forward.
# Below threshold Bootstrap is used because NegBin spreads probability mass
# too broadly at low lines (rec yards props rarely appear below 20.5 anyway).
HYBRID_NEGBIN_THRESHOLD = 20.5

DIRECTION      = "UNDER"
EDGE_THRESHOLD = 0.05
LINE_MIN       = 20.5
LINE_MAX       = 99.5
MIN_BOOKS      = 1

N_BOOT = 10_000
RNG    = np.random.default_rng(42)

BEST_FEATS = [
    "offered_line",
    "game_total",
    "proj_own_score",
    "receiving_yards_L8",
    "target_share_L8",
    "snap_pct_L8",
    "pos_TE",
    "market_under_prob",
]

KEEP_POSITIONS = ["WR", "TE"]


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pos_TE"] = (df["position"] == "TE").astype(int)
    return df


def load_artifacts(artifact_dir: Path) -> dict:
    missing = [f for f in ["ols_pipeline.joblib", "residuals.npy",
                            "nb_coefs.npy", "nb_alpha.npy", "meta.json"]
               if not (artifact_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing artifacts in {artifact_dir}: {missing}\nRun train.py first."
        )
    return {
        "ols":       joblib.load(artifact_dir / "ols_pipeline.joblib"),
        "residuals": np.load(artifact_dir / "residuals.npy"),
        "nb_coefs":  np.load(artifact_dir / "nb_coefs.npy"),
        "nb_alpha":  float(np.load(artifact_dir / "nb_alpha.npy")[0]),
        "meta":      json.loads((artifact_dir / "meta.json").read_text()),
    }


def _p_negbin(mu: np.ndarray, line: np.ndarray, alpha: float) -> np.ndarray:
    mu         = np.clip(mu, 1e-3, None)
    n_nb       = 1.0 / alpha
    p_nb       = n_nb / (n_nb + mu)
    line_floor = np.floor(line).astype(int)
    return nbinom.cdf(line_floor, n=n_nb, p=p_nb)


def _p_bootstrap(pred: np.ndarray, line: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    samples = RNG.choice(residuals, size=(len(pred), N_BOOT))
    sims    = pred[:, None] + samples
    return (sims <= line[:, None]).mean(axis=1)


def run_inference(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    ols       = artifacts["ols"]
    residuals = artifacts["residuals"]
    nb_coefs  = artifacts["nb_coefs"]
    nb_alpha  = artifacts["nb_alpha"]

    result = df.copy()
    mask   = result[BEST_FEATS].notna().all(axis=1)
    idx    = result.index[mask]
    X      = result.loc[idx, BEST_FEATS].to_numpy(dtype=float)
    line   = result.loc[idx, "offered_line"].to_numpy(dtype=float)

    ols_pred = ols.predict(X)
    X_const  = np.column_stack([np.ones(len(X)), X])
    nb_mu    = np.exp(X_const @ nb_coefs)

    p_nb  = _p_negbin(nb_mu, line, nb_alpha)
    p_bt  = _p_bootstrap(ols_pred, line, residuals)
    p_hyb = np.where(line < HYBRID_NEGBIN_THRESHOLD, p_bt, p_nb)

    p_mkt = result.loc[idx, "market_under_prob"].to_numpy(dtype=float)
    edge  = p_hyb - p_mkt
    rec   = np.select(
        [edge > EDGE_THRESHOLD, edge < -EDGE_THRESHOLD],
        ["UNDER", "OVER"],
        default="PASS",
    )

    result.loc[idx, "ols_pred"]       = np.round(ols_pred, 3)
    result.loc[idx, "nb_mu"]          = np.round(nb_mu, 3)
    result.loc[idx, "p_hybrid"]       = np.round(p_hyb, 4)
    result.loc[idx, "p_market"]       = np.round(p_mkt, 4)
    result.loc[idx, "edge"]           = np.round(edge, 4)
    result.loc[idx, "recommendation"] = rec
    return result


def filter_bets(
    results: pd.DataFrame,
    edge_threshold: float = EDGE_THRESHOLD,
    direction: str = DIRECTION,
    line_min: float = LINE_MIN,
    line_max: float = LINE_MAX,
    min_books: int = MIN_BOOKS,
) -> pd.DataFrame:
    mask = (
        (results["recommendation"].notna()) &
        (results["offered_line"] >= line_min) &
        (results["offered_line"] <= line_max) &
        (results["edge"].abs() >= edge_threshold)
    )
    if "n_books" in results.columns:
        mask &= results["n_books"] >= min_books
    if direction == "OVER":
        mask &= results["recommendation"] == "OVER"
    elif direction == "UNDER":
        mask &= results["recommendation"] == "UNDER"
    else:
        mask &= results["recommendation"].isin(["OVER", "UNDER"])
    return results[mask].copy()


def _print_poc_summary(bets: pd.DataFrame, total_scored: int, edge_threshold: float) -> None:
    W = 80
    print(f"\n{'='*W}")
    print("  POC INFERENCE SUMMARY  (⚠  in-sample — model trained on this data)")
    print(f"  Edge threshold: >{edge_threshold*100:.1f}pp  |  "
          f"Lines: {LINE_MIN}–{LINE_MAX}  |  Direction: {DIRECTION}")
    print(f"{'='*W}")
    print(f"  Total rows scored : {total_scored:,}")
    print(f"  Recommended bets  : {len(bets):,}  ({len(bets)/total_scored*100:.1f}% of scored)")

    if len(bets) == 0:
        print("  No bets recommended at this edge threshold.")
        return

    over_bets  = (bets["recommendation"] == "OVER").sum()
    under_bets = (bets["recommendation"] == "UNDER").sum()
    print(f"    OVER  bets      : {over_bets:,}")
    print(f"    UNDER bets      : {under_bets:,}")
    print(f"    Mean |edge|     : {bets['edge'].abs().mean()*100:.2f}pp")
    print(f"    Mean line       : {bets['offered_line'].mean():.1f}")

    if "bet_correct" in bets.columns:
        hit_rate = bets["bet_correct"].mean()
        print(f"\n  In-sample hit rate : {hit_rate*100:.1f}%")

        print(f"\n  By season:")
        for season, grp in bets.groupby("season"):
            hr = grp["bet_correct"].mean()
            print(f"    {season}: {hr*100:.1f}%  ({len(grp):,} bets)")

        print(f"\n  By direction:")
        for rec, grp in bets.groupby("recommendation"):
            hr = grp["bet_correct"].mean()
            print(f"    {rec:5}: {hr*100:.1f}%  ({len(grp):,} bets)")

        print(f"\n  By line bucket:")
        bets = bets.copy()
        bets["bucket"] = pd.cut(
            bets["offered_line"],
            bins=[0, 29.5, 49.5, 69.5, float("inf")],
            labels=["≤29", "30-49", "50-69", "70+"],
        )
        for bucket, grp in bets.groupby("bucket", observed=True):
            hr = grp["bet_correct"].mean()
            print(f"    {bucket}: {hr*100:.1f}%  ({len(grp):,} bets)")

        print(f"\n  By position:")
        if "position" in bets.columns:
            pos_grp = (
                bets.groupby("position", dropna=False)
                .agg(n=("bet_correct", "count"), hit_rate=("bet_correct", "mean"))
                .sort_values("n", ascending=False)
            )
            pos_grp["hit_rate"] = (pos_grp["hit_rate"] * 100).round(1)
            print(pos_grp.to_string())

    print(f"\n  Top 15 highest-edge bets:")
    top_cols = ["player_name", "team", "position", "season", "week", "book",
                "offered_line", "ols_pred", "p_hybrid", "p_market",
                "edge", "recommendation"]
    top_cols = [c for c in top_cols if c in bets.columns]
    top = bets.reindex(columns=top_cols).nlargest(15, "edge")
    for col in ["p_hybrid", "p_market", "edge"]:
        if col in top.columns:
            top[col] = (top[col] * 100).round(2)
    print(top.to_string(index=False))
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir",   type=Path,  default=ARTIFACT_DIR)
    parser.add_argument("--edge-threshold", type=float, default=EDGE_THRESHOLD)
    parser.add_argument("--direction",      type=str,   default=DIRECTION,
                        choices=["UNDER", "OVER", "BOTH"])
    parser.add_argument("--line-min",       type=float, default=LINE_MIN)
    parser.add_argument("--line-max",       type=float, default=LINE_MAX)
    parser.add_argument("--min-books",      type=int,   default=MIN_BOOKS)
    args = parser.parse_args()

    print(f"\n  Loading artifacts from {args.artifact_dir}...")
    artifacts = load_artifacts(args.artifact_dir)
    meta      = artifacts["meta"]
    print(f"    Trained seasons : {meta['train_seasons']}")
    print(f"    Training rows   : {meta['n_rows_train']:,}")
    print(f"    In-sample MAE   : {meta['in_sample_mae']}")
    print(f"    NegBin α        : {meta['nb_alpha']}")

    print(f"\n  Loading labeled dataset (PoC mode)...")
    df = pd.read_parquet(LABELED_PATH)
    df = df[df["position"].isin(KEEP_POSITIONS)].copy()
    df = add_derived(df)
    print(f"    Rows: {len(df):,}  |  Seasons: {sorted(df['season'].unique())}")

    print(f"\n  Running inference [bootstrap: {N_BOOT:,} draws per row]...")
    results = run_inference(df, artifacts)

    # ── Step 3 assert: yhat book-invariant ───────────────────────────────────
    # market_under_prob is in BEST_FEATS and varies by book, so ols_pred will
    # vary across books for the same (player, game, line). This is intentional
    # but means the model uses a per-book feature — flagged here for awareness.
    scored_mask = results["ols_pred"].notna()
    if scored_mask.any():
        yhat_check = (
            results[scored_mask]
            .groupby(["player_name_norm", "game_id", "offered_line"])["ols_pred"]
            .nunique()
        )
        n_viol = int((yhat_check > 1).sum())
        if n_viol:
            print(f"\n  ⚠ ols_pred NOT book-invariant: {n_viol} (player, game, line) groups "
                  f"have varying predictions across books. market_under_prob is a per-book "
                  f"feature in BEST_FEATS — this is the expected cause.")
        else:
            print(f"\n  ✓ ols_pred book-invariant: 0 violations")

    # ── Step 4 assert: p_model clipping [0.01, 0.99] ─────────────────────────
    phyb_scored = results.loc[scored_mask, "p_hybrid"]
    n_low  = int((phyb_scored < 0.01).sum())
    n_high = int((phyb_scored > 0.99).sum())
    results.loc[scored_mask, "p_hybrid"] = phyb_scored.clip(0.01, 0.99)
    total_clip = n_low + n_high
    n_scored   = scored_mask.sum()
    if total_clip:
        print(f"  p_model clip [0.01, 0.99]: {n_low} rows → 0.01, {n_high} rows → 0.99 "
              f"({total_clip / n_scored:.2%} of {n_scored} scored rows) — review these rows")
    else:
        print(f"  p_model clip [0.01, 0.99]: 0 rows hit boundary across {n_scored} scored rows ✓")

    # ── Step 4 assert: line monotonicity ─────────────────────────────────────
    # Higher line → easier to go under → p(under) must be non-decreasing.
    scored_df = results[results["p_hybrid"].notna()].copy()
    pg_counts = scored_df.groupby(["player_name_norm", "game_id"])["offered_line"].nunique()
    multi_pg  = set(pg_counts[pg_counts > 1].index)
    n_inv, inv_examples = 0, []
    for (pn, gid), grp in scored_df.groupby(["player_name_norm", "game_id"]):
        if (pn, gid) not in multi_pg:
            continue
        grp_s = grp.sort_values("offered_line")
        pu = grp_s["p_hybrid"].values
        ls = grp_s["offered_line"].values
        for i in range(len(pu) - 1):
            if pu[i + 1] < pu[i]:
                n_inv += 1
                if len(inv_examples) < 3:
                    inv_examples.append(
                        f"{pn} game={gid} line {ls[i]:.1f}→{ls[i+1]:.1f} "
                        f"p_u {pu[i]:.3f}→{pu[i+1]:.3f}"
                    )
    if multi_pg:
        rate = n_inv / len(multi_pg)
        flag = " ⚠ rate > 2% — investigate line feature" if rate > 0.02 else " ✓ OK"
        print(f"  Line monotonicity: {n_inv} inversions / {len(multi_pg)} multi-line "
              f"player-games ({rate:.1%}){flag}")
        for ex in inv_examples:
            print(f"    example: {ex}")
    else:
        print("  Line monotonicity: no multi-line player-games found")
    print()

    if TARGET in results.columns:
        results["actual_under"] = (results[TARGET] <= results["offered_line"]).astype(float)
        results["bet_correct"]  = np.where(
            results["recommendation"] == "UNDER", results["actual_under"],
            np.where(
                results["recommendation"] == "OVER", 1 - results["actual_under"],
                np.nan,
            ),
        )

    scored = results["ols_pred"].notna().sum()
    bets   = filter_bets(results, args.edge_threshold, args.direction,
                         args.line_min, args.line_max, args.min_books)

    _print_poc_summary(bets, scored, args.edge_threshold)

    OUT_ALL.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(OUT_ALL, index=False)
    bets.to_parquet(OUT_BETS, index=False)
    print(f"  Saved full results → {OUT_ALL}  ({len(results):,} rows)")
    print(f"  Saved bets         → {OUT_BETS}  ({len(bets):,} rows)")
    print()


if __name__ == "__main__":
    main()
