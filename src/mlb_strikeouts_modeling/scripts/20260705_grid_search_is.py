"""
MLB Pitcher Strikeouts — Step 6: In-Sample Grid Search
=======================================================
Same grid as Step 5 but uses the full-data trained model (in-sample).
IS ROI is inflated by construction — key check is IS/OOS ratio < 5x.

Outputs:
  ~/Downloads/tmp/mlb_strikeouts/step6_grid_is.csv
  ~/Downloads/tmp/mlb_strikeouts/step6_is_predictions.parquet

Usage:
  python src/mlb_strikeouts_modeling/scripts/v6_grid_search_is.py
"""
from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from io import BytesIO
import boto3

REPO_ROOT  = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR    = Path.home() / "Downloads/tmp/mlb_strikeouts"
MODELS_DIR = REPO_ROOT / "models"
S3_BUCKET  = "the-odds-api-mt"
LABELED_KEY = "mlb/strikeouts_model/labeled/mlb_strikeouts_labeled.parquet"

MIN_EDGES    = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
SHRINKAGES   = [0.0, 0.25, 0.50, 0.75]
DIRECTIONS   = ["under_only", "over_only", "both"]
ODDS_BUCKETS = ["all", "dog_only", "fav_only"]
LINE_BUCKETS = ["all", "low_le4.5", "mid_5.5_6.5", "high_ge7.5"]

N_BOOT = 10_000
RNG    = np.random.default_rng(42)


def bootstrap_p_over_batch(yhat: np.ndarray, line: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    samples = RNG.choice(residuals, size=(len(yhat), N_BOOT), replace=True)
    sims    = yhat[:, None] + samples
    return (sims > line[:, None]).mean(axis=1)


def max_drawdown_units(pnl_series: np.ndarray) -> float:
    if len(pnl_series) == 0:
        return 0.0
    cum = np.cumsum(pnl_series)
    return float((np.maximum.accumulate(cum) - cum).max())


def p_market_to_american(p: float) -> float:
    if p >= 0.5:
        return -(p / (1 - p) * 100)
    return (1 - p) / p * 100


def compute_unit_pnl(is_over: int, side: str, am_odds: float) -> float:
    bet_hits = (is_over == 1) if side == "over" else (is_over == 0)
    if bet_hits:
        return am_odds / 100.0 if am_odds >= 0 else 100.0 / abs(am_odds)
    return -1.0


def run_grid_search(df: pd.DataFrame, residuals: np.ndarray) -> pd.DataFrame:
    df = df.sort_values("game_date").reset_index(drop=True)
    yhat_arr       = df["yhat"].values
    line_arr       = df["line"].values
    p_market_over  = df["p_market_over"].values
    p_market_under = df["p_market_under"].values
    is_over_arr    = df["is_over"].values
    rows = []

    for shrink in SHRINKAGES:
        mean_adj = line_arr + (1.0 - shrink) * (yhat_arr - line_arr)

        p_model_over  = bootstrap_p_over_batch(mean_adj, line_arr, residuals)
        p_model_under = 1.0 - p_model_over

        edge_over  = p_model_over  - p_market_over
        edge_under = p_model_under - p_market_under

        for min_edge, direction, odds_bucket, line_bucket in product(
            MIN_EDGES, DIRECTIONS, ODDS_BUCKETS, LINE_BUCKETS
        ):
            if direction == "under_only":
                bet_mask = edge_under >= min_edge
                sides    = np.where(bet_mask, "under", None)
            elif direction == "over_only":
                bet_mask = edge_over >= min_edge
                sides    = np.where(bet_mask, "over", None)
            else:
                under_q  = edge_under >= min_edge
                over_q   = edge_over  >= min_edge
                bet_mask = under_q | over_q
                sides    = np.where(
                    under_q & (~over_q | (edge_under >= edge_over)), "under",
                    np.where(over_q, "over", None),
                )

            if odds_bucket == "dog_only":
                under_dog = (direction in ("under_only", "both")) & (p_market_under < 0.50)
                over_dog  = (direction in ("over_only",  "both")) & (p_market_over  < 0.50)
                bet_mask  = bet_mask & (under_dog | over_dog)
            elif odds_bucket == "fav_only":
                under_fav = (direction in ("under_only", "both")) & (p_market_under >= 0.50)
                over_fav  = (direction in ("over_only",  "both")) & (p_market_over  >= 0.50)
                bet_mask  = bet_mask & (under_fav | over_fav)

            if line_bucket == "low_le4.5":
                bet_mask = bet_mask & (line_arr <= 4.5)
            elif line_bucket == "mid_5.5_6.5":
                bet_mask = bet_mask & (line_arr >= 5.5) & (line_arr <= 6.5)
            elif line_bucket == "high_ge7.5":
                bet_mask = bet_mask & (line_arr >= 7.5)

            idx    = np.where(bet_mask)[0]
            n_bets = len(idx)
            if n_bets < 30:
                continue

            pnls      = []
            dec_odds  = []
            for i in idx:
                side = sides[i]
                if side is None:
                    continue
                p_mkt = float(p_market_over[i]) if side == "over" else float(p_market_under[i])
                dec_odds.append(1.0 / p_mkt)
                am_odds = p_market_to_american(p_mkt)
                pnls.append(compute_unit_pnl(int(is_over_arr[i]), side, am_odds))

            pnls     = np.array(pnls)
            dec_odds = np.array(dec_odds)
            units    = float(pnls.sum())
            wins     = int((pnls > 0).sum())
            mdd      = max_drawdown_units(pnls)
            n        = len(pnls)

            rows.append({
                "shrinkage":     shrink,
                "min_edge":      min_edge,
                "direction":     direction,
                "odds_bucket":   odds_bucket,
                "line_bucket":   line_bucket,
                "n_bets":        n,
                "win_rate":      round(wins / n, 4),
                "units_won":     round(units, 2),
                "roi":           round(units / n, 4),
                "avg_odds":      round(float(dec_odds.mean()), 4),
                "max_drawdown":  round(mdd, 2),
                "drawdown_flag": mdd > units,
            })

    return pd.DataFrame(rows).sort_values("units_won", ascending=False)


def main():
    print("Loading model artifacts...", flush=True)
    meta      = json.loads((MODELS_DIR / "mlb_strikeouts_meta.json").read_text())
    features  = meta["features"]
    mtype     = meta["model_type"]
    model     = joblib.load(MODELS_DIR / "mlb_strikeouts_model.joblib")
    residuals = np.load(MODELS_DIR / "mlb_strikeouts_residuals.npy")
    sigma     = residuals.std()
    residuals = np.clip(residuals, -5 * sigma, 5 * sigma)
    print(f"  Model: {mtype}  Features: {features}")
    print(f"  Residuals: {len(residuals):,}  σ={sigma:.4f}")

    print("Loading labeled dataset from S3...", flush=True)
    s3   = boto3.client("s3")
    body = s3.get_object(Bucket=S3_BUCKET, Key=LABELED_KEY)["Body"].read()
    df   = pd.read_parquet(BytesIO(body))
    print(f"  Labeled rows: {len(df):,}")

    # v5 features (cl_over_odds_bin_g etc.) live in the labeled dataset, not the spine.
    # Fall back to spine only for truly missing features that aren't in labeled either.
    missing_feats = [f for f in features if f not in df.columns]
    if missing_feats:
        print(f"  Loading spine to get missing features: {missing_feats}")
        spine_body = s3.get_object(
            Bucket=S3_BUCKET, Key="mlb/strikeouts_model/spine/mlb_strikeouts_spine.parquet"
        )["Body"].read()
        spine = pd.read_parquet(BytesIO(spine_body))
        spine["game_date"] = spine["game_date"].astype(str)
        # Only join features that actually exist in the spine
        available = [f for f in missing_feats if f in spine.columns]
        if available:
            spine_join = spine[["player_key", "game_date"] + available].drop_duplicates(
                subset=["player_key", "game_date"]
            )
            df["game_date"] = df["game_date"].astype(str)
            df = df.merge(spine_join, on=["player_key", "game_date"], how="left")
        still_missing = [f for f in missing_feats if f not in df.columns]
        if still_missing:
            print(f"  WARNING: features still missing after spine join: {still_missing}")

    df = df.dropna(subset=features + ["novig_over", "line", "strikeouts"])
    print(f"  Rows after dropna: {len(df):,}")

    # Score with full-data IS model
    df["yhat"]           = model.predict(df[features])
    df["p_market_over"]  = df["novig_over"]
    df["p_market_under"] = 1.0 - df["novig_over"]

    print(f"  yhat range: [{df['yhat'].min():.2f}, {df['yhat'].max():.2f}]")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_DIR / "step6_is_predictions.parquet", index=False)
    print(f"  IS predictions saved → {OUT_DIR}/step6_is_predictions.parquet")

    # Deduplicate to one row per (player_key, game_date, line) — same as OOS
    n_before = len(df)
    df = df.drop_duplicates(subset=["player_key", "game_date", "line"], keep="first")
    print(f"  Deduped: {n_before:,} → {len(df):,} rows")

    print("\nRunning IS grid search...", flush=True)
    results = run_grid_search(df, residuals)
    results.to_csv(OUT_DIR / "step6_grid_is.csv", index=False)
    print(f"Saved: {OUT_DIR}/step6_grid_is.csv  ({len(results):,} valid strategies)")

    print(f"\n── Top 20 IS strategies by units_won ──")
    print(results.head(20)[
        ["shrinkage", "min_edge", "direction", "odds_bucket", "line_bucket",
         "n_bets", "win_rate", "units_won", "roi", "max_drawdown"]
    ].to_string(index=False))

    # IS vs OOS comparison for top OOS strategies
    oos_path = OUT_DIR / "step5_grid_oos.csv"
    if oos_path.exists():
        oos = pd.read_csv(oos_path)
        top_oos = oos.head(5)
        print(f"\n── IS vs OOS comparison for top 5 OOS strategies ──")
        for _, o_row in top_oos.iterrows():
            im = results.copy()
            for k in ["shrinkage", "min_edge", "direction", "odds_bucket", "line_bucket"]:
                im = im[im[k] == o_row[k]]
            if len(im) > 0:
                i_row = im.iloc[0]
                ratio = i_row["roi"] / o_row["roi"] if o_row["roi"] != 0 else float("inf")
                tag = (f'shrink={o_row["shrinkage"]:.2f}|edge={o_row["min_edge"]:.2f}|'
                       f'{o_row["direction"][:5]}|{o_row["odds_bucket"][:3]}|{o_row["line_bucket"][:10]}')
                print(f"\n  {tag}")
                print(f"    OOS: n={o_row['n_bets']:,}  units={o_row['units_won']:.2f}  roi={o_row['roi']:.4f}  mdd={o_row['max_drawdown']:.2f}")
                print(f"    IS:  n={i_row['n_bets']:,}  units={i_row['units_won']:.2f}  roi={i_row['roi']:.4f}  mdd={i_row['max_drawdown']:.2f}")
                print(f"    IS/OOS ratio: {ratio:.2f}x {'✓ PASS' if ratio < 5.0 else '⚠ FLAG - OVERFIT'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
