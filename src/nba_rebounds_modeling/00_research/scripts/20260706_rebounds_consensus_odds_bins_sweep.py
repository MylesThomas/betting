"""
Step 3b — Consensus odds bin features: n=1 individual sweep + odds-direction split.

Adds 4 new candidate features to the rebounds feature universe:
  - consensus_over_odds_bin         (coarse: +, -, even)
  - consensus_over_odds_bin_granular (8 buckets)
  - consensus_under_odds_bin        (coarse: +, -, even)
  - consensus_under_odds_bin_granular (8 buckets)

Consensus = simple average of American odds across all books at the consensus line,
per player-game. Features are player-game level (book-invariant).

Analysis:
  1. n=1 individual OOF sweep — each new feature alone vs production baseline
  2. Odds-direction split on existing production model (OOS leave-one-season-out)

Outputs:
  - /tmp/rebounds_step3b_n1_sweep.csv
  - /tmp/rebounds_step3b_oos_split.csv
  - /tmp/rebounds_step3b_bin_dist.csv

Usage:
    python src/nba_rebounds_modeling/00_research/scripts/20260706_rebounds_consensus_odds_bins_sweep.py \\
        --features ~/Downloads/tmp/rebounds_features.parquet \\
        --props ~/Downloads/tmp/rebounds_props.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.model_selection import KFold


# ── args ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--features", default="/Users/thomasmyles/Downloads/tmp/rebounds_features.parquet")
    p.add_argument("--props", default="/Users/thomasmyles/Downloads/tmp/rebounds_props.parquet")
    p.add_argument("--out-dir", default="/tmp")
    return p.parse_args()


# ── production spec ───────────────────────────────────────────────────────────

PROD_FEATURES = ["min_line", "max_line", "spread_signed", "roll_reb_mean_60", "roll_fg3a_mean_20", "roll_reb_std_5"]
TARGET = "REB"
SEASONS = ["2023-24", "2024-25", "2025-26"]

# Option A prod params
SIGMA_COL = "roll_reb_std_5"
SIGMA_FLOOR = 0.25
SHRINK = 0.0
MIN_EDGE = 0.05


# ── consensus odds bin features ───────────────────────────────────────────────

def american_to_decimal(american: float) -> float:
    if american >= 100:
        return american / 100.0 + 1.0
    return 100.0 / abs(american) + 1.0


GRANULAR_BINS = [
    (-9999, -300, "-500_to_-300"),
    (-300, -200, "-300_to_-200"),
    (-200, -110, "-200_to_-110"),
    (-110, 0,    "-110_to_even"),
    (0, 110,     "even_to_+110"),
    (110, 200,   "+110_to_+200"),
    (200, 300,   "+200_to_+300"),
    (300, 9999,  "+300_plus"),
]


def coarse_bin(american: float) -> str:
    if pd.isna(american):
        return "unknown"
    if american > 0:
        return "+"
    if american < 0:
        return "-"
    return "even"


def granular_bin(american: float) -> str:
    if pd.isna(american):
        return "unknown"
    for lo, hi, label in GRANULAR_BINS:
        if lo < american <= hi:
            return label
    return "+300_plus" if american > 300 else "-500_to_-300"


def build_consensus_bins(props: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per player-game consensus American odds (avg across all books at consensus line).
    Returns player-game level df with 4 bin features.
    """
    # For each player-game, get the consensus line from props
    pg_consensus = (
        props.groupby(["season", "date", "player_normalized", "game_id"], as_index=False)
        ["consensus_reb_line"].first()
    )

    # Filter props rows to consensus line only, then average odds across books
    props_at_consensus = props.merge(
        pg_consensus,
        on=["season", "date", "player_normalized", "game_id", "consensus_reb_line"],
        how="inner",
    )

    consensus_odds = (
        props_at_consensus
        .groupby(["season", "date", "player_normalized", "game_id"], as_index=False)
        .agg(
            consensus_over_odds_mean=("over_odds", "mean"),
            consensus_under_odds_mean=("under_odds", "mean"),
            n_books_at_consensus=("bookmaker", "nunique"),
        )
    )

    consensus_odds["consensus_over_odds_bin"] = consensus_odds["consensus_over_odds_mean"].apply(coarse_bin)
    consensus_odds["consensus_over_odds_bin_granular"] = consensus_odds["consensus_over_odds_mean"].apply(granular_bin)
    consensus_odds["consensus_under_odds_bin"] = consensus_odds["consensus_under_odds_mean"].apply(coarse_bin)
    consensus_odds["consensus_under_odds_bin_granular"] = consensus_odds["consensus_under_odds_mean"].apply(granular_bin)

    return consensus_odds


# ── OOF sweep helpers ─────────────────────────────────────────────────────────

def oof_rmse_r2_ols(df: pd.DataFrame, features: list[str], target: str, n_splits: int = 5) -> dict:
    """Leave-one-season-out OOF: train on 2 seasons, test on 1."""
    results = []
    for test_season in SEASONS:
        train = df.loc[df["season"] != test_season].copy()
        test = df.loc[df["season"] == test_season].copy()
        if len(test) == 0:
            continue

        X_train = train[features].astype(float)
        y_train = train[target].astype(float)
        X_test = test[features].astype(float)
        y_test = test[target].astype(float)

        X_train_c = sm.add_constant(X_train, has_constant="add")
        X_test_c = sm.add_constant(X_test, has_constant="add")
        model = sm.OLS(y_train, X_train_c).fit()
        yhat = model.predict(X_test_c)

        residuals = y_test - yhat
        ss_res = (residuals ** 2).sum()
        ss_tot = ((y_test - y_test.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rmse = float(np.sqrt((residuals ** 2).mean()))
        results.append({"test_season": test_season, "rmse": rmse, "r2": r2, "n_test": len(test)})

    if not results:
        return {"rmse": float("nan"), "r2": float("nan")}
    avg_rmse = float(np.mean([r["rmse"] for r in results]))
    avg_r2 = float(np.mean([r["r2"] for r in results]))
    return {"rmse": avg_rmse, "r2": avg_r2, "by_season": results}


def oof_rmse_r2_ols_categorical(df: pd.DataFrame, cat_feature: str, target: str) -> dict:
    """OOF sweep for a single categorical feature (one-hot encoded)."""
    dummies = pd.get_dummies(df[cat_feature], prefix=cat_feature, drop_first=True)
    feat_cols = list(dummies.columns)
    df2 = pd.concat([df[["season", target]], dummies], axis=1)
    return oof_rmse_r2_ols(df2, feat_cols, target)


# ── odds-direction split on production model ──────────────────────────────────

def american_to_raw_implied(american: np.ndarray) -> np.ndarray:
    odds = american.astype(np.float64, copy=False)
    out = np.empty_like(odds, dtype=np.float64)
    neg = odds < 0
    out[neg] = (-odds[neg]) / ((-odds[neg]) + 100.0)
    out[~neg] = 100.0 / (odds[~neg] + 100.0)
    return out


def oos_score_production(feat_df: pd.DataFrame, props: pd.DataFrame) -> pd.DataFrame:
    """
    Leave-one-season-out OLS with prod features.
    Returns row-level df at (player, date, bookmaker, line) grain with predictions + edge.
    """
    # Join features to props (expand to per-book) — dedupe feat_cols to avoid duplicate columns
    group_keys = ["season", "date", "player_normalized", "game_id"]
    # TARGET (REB) already in props — only bring feature columns from feat_df
    feat_only_cols = list(dict.fromkeys([SIGMA_COL] + PROD_FEATURES))  # dedupe, no TARGET
    feat_slim = feat_df[group_keys + [c for c in feat_only_cols if c in feat_df.columns]].copy()
    joined = props.merge(feat_slim, on=group_keys, how="inner")
    joined = joined.dropna(subset=PROD_FEATURES + [SIGMA_COL, TARGET, "over_odds", "under_odds"]).copy()

    out_parts = []
    for test_season in SEASONS:
        train = feat_df.loc[feat_df["season"] != test_season].copy()
        test_rows = joined.loc[joined["season"] == test_season].copy()
        if len(test_rows) == 0:
            continue

        X_train = train[PROD_FEATURES].astype(float)
        y_train = train[TARGET].astype(float)
        X_test = test_rows[PROD_FEATURES].astype(float)

        X_train_c = sm.add_constant(X_train, has_constant="add")
        X_test_c = sm.add_constant(X_test, has_constant="add")
        model = sm.OLS(y_train, X_train_c).fit()
        yhat = model.predict(X_test_c).to_numpy()

        consensus = test_rows["consensus_reb_line"].astype(float).to_numpy()
        line = test_rows["line"].astype(float).to_numpy()
        sigma_raw = test_rows[SIGMA_COL].astype(float).to_numpy()
        sigma = np.maximum(sigma_raw, SIGMA_FLOOR)
        mean_adj = consensus + (1.0 - SHRINK) * (yhat - consensus)
        z = (line - mean_adj) / sigma
        p_under = norm.cdf(z)

        p_raw_o = american_to_raw_implied(test_rows["over_odds"].to_numpy())
        p_raw_u = american_to_raw_implied(test_rows["under_odds"].to_numpy())
        edge_under = p_under - p_raw_u
        play_under = edge_under > MIN_EDGE

        test_rows = test_rows.copy()
        test_rows["yhat_ols"] = yhat
        test_rows["p_under_ols"] = p_under
        test_rows["edge_under_ols"] = edge_under
        test_rows["play_under"] = play_under
        out_parts.append(test_rows)

    return pd.concat(out_parts, ignore_index=True) if out_parts else pd.DataFrame()


def build_odds_split(scored: pd.DataFrame, bin_col: str) -> pd.DataFrame:
    """Compute hit rate / ROI grouped by odds bin, under-only qualifying plays."""
    plays = scored.loc[scored["play_under"]].copy()
    if len(plays) == 0:
        return pd.DataFrame()

    plays["is_win"] = plays[TARGET] < plays["line"]
    plays["is_loss"] = plays[TARGET] > plays["line"]
    plays["is_push"] = plays[TARGET] == plays["line"]

    def _pnl(row: pd.Series) -> float:
        odds = float(row["under_odds"])
        if row["is_win"]:
            return (odds / 100.0) if odds >= 100 else (100.0 / abs(odds))
        if row["is_loss"]:
            return -1.0
        return 0.0

    plays["pnl"] = plays.apply(_pnl, axis=1)
    n_bets = plays[bin_col].value_counts()

    grouped = (
        plays.groupby(bin_col, as_index=True)
        .agg(
            n_bets=("play_under", "sum"),
            n_win=("is_win", "sum"),
            n_loss=("is_loss", "sum"),
            n_push=("is_push", "sum"),
            pnl=("pnl", "sum"),
        )
        .reset_index()
    )
    grouped["hit_rate"] = grouped["n_win"] / (grouped["n_win"] + grouped["n_loss"]).clip(lower=1)
    grouped["roi"] = grouped["pnl"] / grouped["n_bets"].clip(lower=1)
    return grouped.sort_values("n_bets", ascending=False).reset_index(drop=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)

    print("Loading data...")
    feat = pd.read_parquet(args.features)
    props = pd.read_parquet(args.props)

    print(f"  features: {feat.shape}, props: {props.shape}")

    # ── build bin features ────────────────────────────────────────────────────
    print("\nBuilding consensus odds bin features...")
    bins_df = build_consensus_bins(props)
    print(f"  bin features built: {len(bins_df)} player-games")

    # Join bins to features (player-game level)
    feat_w_bins = feat.merge(
        bins_df[["season", "date", "player_normalized", "game_id",
                 "consensus_over_odds_mean", "consensus_under_odds_mean",
                 "consensus_over_odds_bin", "consensus_over_odds_bin_granular",
                 "consensus_under_odds_bin", "consensus_under_odds_bin_granular",
                 "n_books_at_consensus"]],
        on=["season", "date", "player_normalized", "game_id"],
        how="inner",
    )
    print(f"  joined: {len(feat_w_bins)} rows (dropped {len(feat) - len(feat_w_bins)} unmatched)")

    # Drop rows where bins are unknown (no props data)
    feat_w_bins = feat_w_bins.loc[feat_w_bins["consensus_over_odds_bin"] != "unknown"].copy()
    print(f"  after dropping unknown bins: {len(feat_w_bins)} rows")

    # Drop rows with nulls in any production feature or target
    required_cols = PROD_FEATURES + [TARGET]
    before = len(feat_w_bins)
    feat_w_bins = feat_w_bins.dropna(subset=required_cols).copy()
    print(f"  after dropping nulls in prod features/target: {len(feat_w_bins)} rows (dropped {before - len(feat_w_bins)})")

    # ── bin distribution ──────────────────────────────────────────────────────
    BIN_FEATURES = [
        "consensus_over_odds_bin",
        "consensus_over_odds_bin_granular",
        "consensus_under_odds_bin",
        "consensus_under_odds_bin_granular",
    ]

    dist_rows = []
    for bf in BIN_FEATURES:
        vc = feat_w_bins[bf].value_counts().reset_index()
        vc.columns = ["value", "count"]
        vc["feature"] = bf
        vc["pct"] = (vc["count"] / len(feat_w_bins) * 100).round(1)
        dist_rows.append(vc)
    dist_df = pd.concat(dist_rows, ignore_index=True)
    dist_path = out_dir / "rebounds_step3b_bin_dist.csv"
    dist_df.to_csv(dist_path, index=False)
    print(f"\nBin distributions saved: {dist_path}")
    print(dist_df.to_string(index=False))

    # ── n=1 sweep ─────────────────────────────────────────────────────────────
    print("\n--- n=1 individual feature sweep ---")

    # Baseline: production 6-feature spec
    print("Computing production baseline (6 features)...")
    baseline = oof_rmse_r2_ols(feat_w_bins, PROD_FEATURES, TARGET)
    print(f"  PROD baseline: RMSE={baseline['rmse']:.4f}, R²={baseline['r2']:.4f}")

    sweep_rows = [{"feature": "PROD_BASELINE (6 features)", "type": "numeric",
                   "rmse": baseline["rmse"], "r2": baseline["r2"],
                   "delta_rmse_vs_prod": 0.0, "delta_r2_vs_prod": 0.0}]

    # Each new categorical bin feature alone (OHE)
    for bf in BIN_FEATURES:
        print(f"  Sweeping {bf}...")
        res = oof_rmse_r2_ols_categorical(feat_w_bins, bf, TARGET)
        delta_rmse = res["rmse"] - baseline["rmse"]
        delta_r2 = res["r2"] - baseline["r2"]
        print(f"    RMSE={res['rmse']:.4f} (Δ{delta_rmse:+.4f}), R²={res['r2']:.4f} (Δ{delta_r2:+.4f})")
        sweep_rows.append({
            "feature": bf, "type": "categorical",
            "rmse": res["rmse"], "r2": res["r2"],
            "delta_rmse_vs_prod": delta_rmse, "delta_r2_vs_prod": delta_r2,
        })

    # Each new bin feature ADDED to production spec
    print("\n  Also testing: PROD + each bin feature...")
    for bf in BIN_FEATURES:
        dummies = pd.get_dummies(feat_w_bins[bf], prefix=bf, drop_first=True)
        feat_augmented = pd.concat([feat_w_bins[["season", TARGET] + PROD_FEATURES], dummies], axis=1)
        aug_feats = PROD_FEATURES + list(dummies.columns)
        res = oof_rmse_r2_ols(feat_augmented, aug_feats, TARGET)
        delta_rmse = res["rmse"] - baseline["rmse"]
        delta_r2 = res["r2"] - baseline["r2"]
        label = f"PROD + {bf}"
        print(f"    [{label}] RMSE={res['rmse']:.4f} (Δ{delta_rmse:+.4f}), R²={res['r2']:.4f} (Δ{delta_r2:+.4f})")
        sweep_rows.append({
            "feature": label, "type": "prod+categorical",
            "rmse": res["rmse"], "r2": res["r2"],
            "delta_rmse_vs_prod": delta_rmse, "delta_r2_vs_prod": delta_r2,
        })

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_path = out_dir / "rebounds_step3b_n1_sweep.csv"
    sweep_df.to_csv(sweep_path, index=False)
    print(f"\nSweep results saved: {sweep_path}")
    print(sweep_df.to_string(index=False))

    # ── odds-direction split on production model ───────────────────────────────
    print("\n--- Odds-direction split (production OLS model, OOS) ---")
    props_w_bins = props.merge(
        bins_df[["season", "date", "player_normalized", "game_id",
                 "consensus_over_odds_bin", "consensus_over_odds_bin_granular",
                 "consensus_under_odds_bin", "consensus_under_odds_bin_granular"]],
        on=["season", "date", "player_normalized", "game_id"],
        how="inner",
    )
    print(f"  props with bins: {len(props_w_bins)} rows")

    scored = oos_score_production(feat_w_bins, props_w_bins)
    print(f"  OOS scored rows: {len(scored)}, qualifying plays: {int(scored['play_under'].sum())}")

    split_rows = []
    for bf in ["consensus_over_odds_bin", "consensus_over_odds_bin_granular",
               "consensus_under_odds_bin", "consensus_under_odds_bin_granular"]:
        split = build_odds_split(scored, bf)
        split["split_feature"] = bf
        split_rows.append(split)

    split_df = pd.concat(split_rows, ignore_index=True)
    split_path = out_dir / "rebounds_step3b_oos_split.csv"
    split_df.to_csv(split_path, index=False)
    print(f"\nOdds-direction split saved: {split_path}")
    for bf in ["consensus_over_odds_bin", "consensus_under_odds_bin"]:
        print(f"\n  {bf} (coarse):")
        sub = split_df.loc[split_df["split_feature"] == bf, ["consensus_over_odds_bin", "consensus_under_odds_bin", "n_bets", "n_win", "n_loss", "hit_rate", "roi", "pnl"]]
        sub = split_df.loc[split_df["split_feature"] == bf]
        print(sub.drop(columns=["split_feature"]).to_string(index=False))


if __name__ == "__main__":
    main()
