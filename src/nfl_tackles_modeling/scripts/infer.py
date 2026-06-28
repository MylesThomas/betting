"""
NFL tackles inference pipeline.

Loads serialized model artifacts and scores an assembled feature DataFrame,
producing per-player-game probability estimates and bet recommendations.

Bet logic:
  p_market = de-vigged implied P(under) from this book's prices (market_under_prob)
  p_hybrid = model P(under):
               line < HYBRID_NEGBIN_THRESHOLD → Bootstrap (OLS residual draws)
               line ≥ HYBRID_NEGBIN_THRESHOLD → Negative Binomial NB2
  edge     = p_hybrid − p_market   (positive = model leans UNDER vs market)
  UNDER    if  edge >  EDGE_THRESHOLD
  OVER     if  edge < −EDGE_THRESHOLD
  PASS     otherwise

Bet filter (applied on top of direction):
  LINE_MIN ≤ offered_line ≤ LINE_MAX   (lines outside this range have thin
  calibration sample and are excluded from recommendations)

PoC mode (default):
  Runs on the labeled dataset (both training seasons) to validate pipeline
  mechanics. Hit rates will be inflated because the model was trained on this
  data — this is expected and explicitly flagged in the output.

Production mode:
  Pass a DataFrame assembled from the current week's Odds API props +
  rolling spine features. That DataFrame must contain the 9 BEST_FEATS columns
  plus offered_line, player_name, position, team, season, week.

Run:
  python src/nfl_tackles_modeling/scripts/infer.py
  python src/nfl_tackles_modeling/scripts/infer.py --edge-threshold 0.04
  python src/nfl_tackles_modeling/scripts/infer.py --artifact-dir path/to/artifacts
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

# ── Paths ─────────────────────────────────────────────────────────────────────
LABELED_PATH = Path.home() / "Downloads" / "tmp" / "nfl_tackles_per_book.parquet"
ARTIFACT_DIR = Path.home() / "Downloads" / "tmp" / "nfl_tackles_artifacts"
OUT_ALL      = Path.home() / "Downloads" / "tmp" / "nfl_tackles_inference.parquet"
OUT_BETS     = Path.home() / "Downloads" / "tmp" / "nfl_tackles_bets.parquet"

TARGET = "tackles_combined"

# ── Hybrid P(over) sampling config ────────────────────────────────────────────
# Calibrated on 2024→2025 OOS walk-forward; see line_calibration.py.
# line < threshold → Bootstrap: NegBin over-predicts overs at low lines
#                    because the discrete NB2 distribution spreads too much
#                    probability mass above the line near the mean.
# line ≥ threshold → NegBin: well-calibrated in main volume zone (4.5-8.5)
#                    and outperforms Bootstrap in the high tail (9.5+).
HYBRID_NEGBIN_THRESHOLD = 4.5

# ── Production bet filter config ──────────────────────────────────────────────
# Analyst pick: UNDER-only, edge≥0.05, lines 4.5-9.5, min_books=1
# Rationale: OVER signal weak (55% IS hit rate); edge<0.05 adds noise not signal;
# lines <4.5 are in uncalibrated tail; max DD ~20 units vs ~35 for looser configs.
DIRECTION      = "UNDER" # UNDER / OVER / BOTH
EDGE_THRESHOLD = 0.05    # minimum |p_hybrid − p_market| to recommend a bet
LINE_MIN       = 4.5     # below 4.5: NegBin uncalibrated, Bootstrap used instead
LINE_MAX       = 9.5     # above 9.5: very low n, calibration unreliable
MIN_BOOKS      = 1       # minimum books with two-sided market

N_BOOT = 10_000
RNG    = np.random.default_rng(42)

POS_GROUP_MAP = {
    "LB": "LB", "CB": "CB", "DB": "CB",
    "S":  "S",  "FS": "S",  "SS": "S",
    "DE": "DL", "DT": "DL", "DL": "DL", "NT": "DL",
}

BEST_FEATS = [
    "offered_line", "game_total", "proj_opp_score", "tackle_rate_L16",
    "pos_LB", "pos_CB", "pos_S", "pos_DL", "market_under_prob",
]

DROP_POSITIONS = ["WR", "FB"]


# ── Artifact loading ──────────────────────────────────────────────────────────

def load_artifacts(artifact_dir: Path) -> dict:
    missing = [f for f in ["ols_pipeline.joblib", "residuals.npy",
                            "nb_coefs.npy", "nb_alpha.npy", "meta.json"]
               if not (artifact_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing artifacts in {artifact_dir}: {missing}\n"
            f"Run train.py first to generate them."
        )
    return {
        "ols":       joblib.load(artifact_dir / "ols_pipeline.joblib"),
        "residuals": np.load(artifact_dir / "residuals.npy"),
        "nb_coefs":  np.load(artifact_dir / "nb_coefs.npy"),
        "nb_alpha":  float(np.load(artifact_dir / "nb_alpha.npy")[0]),
        "meta":      json.loads((artifact_dir / "meta.json").read_text()),
    }


# ── Feature engineering ───────────────────────────────────────────────────────

def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add position dummies. market_under_prob already present in per-book dataset."""
    df = df.copy()
    df["position_group"] = df["position"].map(POS_GROUP_MAP)
    for g in ["LB", "CB", "S", "DL"]:
        df[f"pos_{g}"] = (df["position_group"] == g).astype(int)
    return df


# ── Probability estimators ────────────────────────────────────────────────────

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


# ── Core inference ────────────────────────────────────────────────────────────

def run_inference(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """
    Score every row in df that has complete features.

    Added columns:
      ols_pred       — OLS point prediction (expected tackles)
      nb_mu          — NegBin predicted mean
      p_hybrid       — hybrid P(under): Bootstrap or NegBin per HYBRID_NEGBIN_THRESHOLD
      p_market       — de-vigged implied P(under) from this book (market_under_prob)
      edge           — p_hybrid − p_market  (+ = model leans UNDER vs market)
      recommendation — UNDER / OVER / PASS
    """
    ols       = artifacts["ols"]
    residuals = artifacts["residuals"]
    nb_coefs  = artifacts["nb_coefs"]
    nb_alpha  = artifacts["nb_alpha"]

    result = df.copy()
    mask   = result[BEST_FEATS].notna().all(axis=1)
    idx    = result.index[mask]
    X      = result.loc[idx, BEST_FEATS].to_numpy(dtype=float)
    line   = result.loc[idx, "offered_line"].to_numpy(dtype=float)

    # Predictions
    ols_pred = ols.predict(X)
    X_const  = np.column_stack([np.ones(len(X)), X])
    nb_mu    = np.exp(X_const @ nb_coefs)

    # Hybrid P(under): Bootstrap at low lines, NegBin elsewhere
    p_nb  = _p_negbin(nb_mu, line, nb_alpha)
    p_bt  = _p_bootstrap(ols_pred, line, residuals)
    p_hyb = np.where(line < HYBRID_NEGBIN_THRESHOLD, p_bt, p_nb)

    # Market P(under) and edge
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
    """Return only actionable rows matching the production config."""
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


# ── PoC summary printing ──────────────────────────────────────────────────────

def _print_poc_summary(bets: pd.DataFrame, total_scored: int, edge_threshold: float) -> None:
    W = 80
    print(f"\n{'='*W}")
    print("  POC INFERENCE SUMMARY  (⚠  in-sample — model trained on this data)")
    print(f"  Edge threshold: >{edge_threshold*100:.1f}pp  |  "
          f"Lines: {LINE_MIN}–{LINE_MAX}  |  "
          f"Hybrid threshold: {HYBRID_NEGBIN_THRESHOLD}")
    print(f"{'='*W}")
    print(f"  Total rows scored     : {total_scored:,}")
    print(f"  Recommended bets      : {len(bets):,}  "
          f"({len(bets)/total_scored*100:.1f}% of scored)")

    if len(bets) == 0:
        print("  No bets recommended at this edge threshold.")
        return

    over_bets  = (bets["recommendation"] == "OVER").sum()
    under_bets = (bets["recommendation"] == "UNDER").sum()
    print(f"    OVER  bets          : {over_bets:,}")
    print(f"    UNDER bets          : {under_bets:,}")
    print(f"    Mean |edge|         : {bets['edge'].abs().mean()*100:.2f}pp")
    print(f"    Mean line           : {bets['offered_line'].mean():.2f}")

    if "bet_correct" in bets.columns:
        hit_rate = bets["bet_correct"].mean()
        print(f"\n  In-sample hit rate    : {hit_rate*100:.1f}%")

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
            bins=[0, 3.5, 6.5, 9.5, float("inf")],
            labels=["0-3", "4-6", "7-9", "10+"],
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


# ── Main ──────────────────────────────────────────────────────────────────────

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
    df = df[df["position"].notna() & ~df["position"].isin(DROP_POSITIONS)].copy()
    df = add_derived(df)
    print(f"    Rows: {len(df):,}  |  Seasons: {sorted(df['season'].unique())}")

    print(f"\n  Running inference [bootstrap: {N_BOOT:,} draws per row]...")
    results = run_inference(df, artifacts)

    # Attach actual outcome for PoC evaluation
    if TARGET in results.columns:
        results["actual_under"] = (
            results[TARGET] <= results["offered_line"]
        ).astype(float)
        results["bet_correct"] = np.where(
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
