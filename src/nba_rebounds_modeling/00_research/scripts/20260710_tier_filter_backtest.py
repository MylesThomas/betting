"""
Tier-filter backtest: does restricting bets to a recognition_tier subset improve
on the full-universe OLS result (+986u, +3.61% ROI, 27,311 bets across 3 seasons)?

For each tier (plus "all" as baseline), runs the production params:
  sigma_window=5, shrinkage=0.0, min_edge=0.05, side_policy=under_only

across all 3 OOS test seasons (leave-one-season-out), then aggregates.

Usage:
    python src/nba_rebounds_modeling/00_research/scripts/20260710_tier_filter_backtest.py \
        --feat  ~/Downloads/tmp/rebounds_features.parquet \
        --props ~/Downloads/tmp/rebounds_props.parquet \
        --tiers ~/Downloads/tmp/rebounds_player_tiers.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

B_MIN_MAX_FEATS = ["min_line", "max_line", "spread_signed", "roll_reb_mean_60", "roll_fg3a_mean_20", "roll_reb_std_5"]
TARGET = "REB"
GROUP_KEYS = ["season", "date", "player_normalized", "game_id"]
SIGMA_COL = "roll_reb_std_5"
SIGMA_FLOOR = 0.25
SHRINKAGE = 0.0
MIN_EDGE = 0.05
TIERS_ORDER = ["superstar", "known_starter", "fringe", "unknown"]


def american_to_implied(american: float) -> float:
    if np.isnan(american):
        return float("nan")
    if american < 0:
        return (-american) / (-american + 100.0)
    return 100.0 / (american + 100.0)


def american_profit(american: float) -> float:
    if american >= 100:
        return american / 100.0
    return 100.0 / abs(american)


def run_oos_season(feat: pd.DataFrame, props: pd.DataFrame, test_season: str) -> pd.DataFrame:
    """Train on all seasons except test_season, score test_season, return per-row bet results."""
    cols = B_MIN_MAX_FEATS + [TARGET] + GROUP_KEYS
    train = feat[feat["season"] != test_season].dropna(subset=cols)
    test = feat[feat["season"] == test_season].dropna(subset=cols + [SIGMA_COL])

    X_tr = sm.add_constant(train[B_MIN_MAX_FEATS].astype(float), has_constant="add")
    m = sm.OLS(train[TARGET].astype(float), X_tr).fit()

    X_te = sm.add_constant(test[B_MIN_MAX_FEATS].astype(float), has_constant="add")
    yhat = m.predict(X_te).to_numpy()
    sigma = test[SIGMA_COL].astype(float).clip(lower=SIGMA_FLOOR).to_numpy()
    consensus = test["consensus_reb_line"].astype(float).to_numpy()

    mean_adj = consensus + (1.0 - SHRINKAGE) * (yhat - consensus)

    scored = test[GROUP_KEYS].copy()
    scored["yhat"] = yhat
    scored["mean_adj"] = mean_adj
    scored["sigma"] = sigma

    props_test = props[props["season"] == test_season].copy()
    merged = props_test.merge(scored, on=GROUP_KEYS, how="inner")

    from scipy.stats import norm
    line = merged["line"].astype(float).to_numpy()
    mean_adj_arr = merged["mean_adj"].to_numpy()
    sigma_arr = merged["sigma"].to_numpy()
    under_odds = merged["under_odds"].astype(float).to_numpy()

    p_under_model = norm.cdf((line - mean_adj_arr) / sigma_arr)
    p_under_book = np.array([american_to_implied(x) for x in under_odds])
    edge_under = p_under_model - p_under_book

    merged["edge_under"] = edge_under
    merged["p_under_model"] = p_under_model
    merged["reb_outcome"] = merged["REB"].astype(float)

    return merged


def score_subset(df: pd.DataFrame) -> dict:
    bets = df[df["edge_under"] >= MIN_EDGE].copy()
    n_bets = 0
    n_win = 0
    pnl = 0.0
    for _, row in bets.iterrows():
        line = row["line"]
        reb = row["reb_outcome"]
        if reb == line:
            continue
        odds = row["under_odds"]
        n_bets += 1
        if reb < line:
            pnl += american_profit(odds)
            n_win += 1
        else:
            pnl -= 1.0
    roi = pnl / n_bets if n_bets else float("nan")
    hit = n_win / n_bets if n_bets else float("nan")
    return {"n_bets": n_bets, "n_win": n_win, "pnl_u": pnl, "roi": roi, "hit_rate": hit}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--feat",  default="~/Downloads/tmp/rebounds_features.parquet")
    p.add_argument("--props", default="~/Downloads/tmp/rebounds_props.parquet")
    p.add_argument("--tiers", default="~/Downloads/tmp/rebounds_player_tiers.parquet")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    feat  = pd.read_parquet(Path(args.feat).expanduser())
    props = pd.read_parquet(Path(args.props).expanduser())
    tiers = pd.read_parquet(Path(args.tiers).expanduser())

    print(f"feat: {feat.shape}  props: {props.shape}  tiers: {tiers.shape}")

    # Join tiers onto both feat and props.
    feat  = feat.merge(tiers, on="player_normalized", how="left")
    feat["recognition_tier"] = feat["recognition_tier"].fillna("unknown")
    props = props.merge(tiers, on="player_normalized", how="left")
    props["recognition_tier"] = props["recognition_tier"].fillna("unknown")

    seasons = sorted(feat["season"].unique())
    print(f"Seasons: {seasons}")

    # Tier subsets to evaluate. "all" = no filter (baseline).
    subsets = ["all"] + TIERS_ORDER

    rows = []
    for test_season in seasons:
        print(f"\n--- OOS season: {test_season} ---")
        # Score the full test season once; then filter by tier subset.
        scored = run_oos_season(feat, props, test_season)

        for subset in subsets:
            if subset == "all":
                df_sub = scored
            else:
                df_sub = scored[scored["recognition_tier"] == subset]

            stats = score_subset(df_sub)
            stats["test_season"] = test_season
            stats["tier_filter"] = subset
            rows.append(stats)
            print(f"  {subset:15s}  n={stats['n_bets']:5d}  pnl={stats['pnl_u']:+.1f}u  roi={stats['roi']:+.2%}  hit={stats['hit_rate']:.3f}")

    results = pd.DataFrame(rows)

    # Aggregate across seasons.
    print("\n\n=== Aggregate (all 3 seasons) ===")
    agg = (
        results.groupby("tier_filter")
        .agg(
            n_bets=("n_bets", "sum"),
            pnl_u=("pnl_u", "sum"),
            seasons_positive=("pnl_u", lambda x: (x > 0).sum()),
        )
        .reset_index()
    )
    agg["roi"] = agg["pnl_u"] / agg["n_bets"]
    # Preserve tier order.
    order = ["all"] + TIERS_ORDER
    agg["tier_filter"] = pd.Categorical(agg["tier_filter"], categories=order, ordered=True)
    agg = agg.sort_values("tier_filter")
    print(agg.to_string(index=False))

    out = Path("~/Downloads/tmp/tier_filter_backtest_results.parquet").expanduser()
    results.to_parquet(out, index=False)
    print(f"\nPer-season results → {out}")


if __name__ == "__main__":
    main()
