"""
Full grid search with tier filter dimension.

Sweeps: sigma_window × shrinkage × min_edge × side_policy × tier_filter
tier_filter: all | non_superstar

Trains OLS once per test_season (leave-one-out across all 3 seasons),
then sweeps params over the scored rows — no retraining per combo.

Usage:
    python src/nba_rebounds_modeling/00_research/scripts/20260710_tier_grid_search.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

B_MIN_MAX_FEATS = ["min_line", "max_line", "spread_signed", "roll_reb_mean_60", "roll_fg3a_mean_20", "roll_reb_std_5"]
TARGET = "REB"
GROUP_KEYS = ["season", "date", "player_normalized", "game_id"]

SIGMA_WINDOWS  = [5, 10, 20]
SHRINKAGES     = [0.0, 0.25, 0.50, 0.75]
MIN_EDGES      = [0.01, 0.05, 0.10]
SIDE_POLICIES  = ["both", "over_only", "under_only"]
TIER_FILTERS   = ["all", "non_superstar"]
SIGMA_FLOOR    = 0.25


def american_to_implied(american: float) -> float:
    if np.isnan(american):
        return float("nan")
    return (-american) / (-american + 100.0) if american < 0 else 100.0 / (american + 100.0)


def american_profit(american: float) -> float:
    return american / 100.0 if american >= 100 else 100.0 / abs(american)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--feat",  default="~/Downloads/tmp/rebounds_features.parquet")
    p.add_argument("--props", default="~/Downloads/tmp/rebounds_props.parquet")
    p.add_argument("--tiers", default="~/Downloads/tmp/rebounds_player_tiers.parquet")
    p.add_argument("--out",   default="~/Downloads/tmp/tier_grid_search_results.parquet")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    feat  = pd.read_parquet(Path(args.feat).expanduser())
    props = pd.read_parquet(Path(args.props).expanduser())
    tiers = pd.read_parquet(Path(args.tiers).expanduser())

    feat  = feat.merge(tiers, on="player_normalized", how="left")
    feat["recognition_tier"] = feat["recognition_tier"].fillna("unknown").astype(str)
    props = props.merge(tiers, on="player_normalized", how="left")
    props["recognition_tier"] = props["recognition_tier"].fillna("unknown").astype(str)

    seasons = sorted(feat["season"].unique())
    print(f"Seasons: {seasons}  feat={feat.shape}  props={props.shape}")

    all_rows: list[dict] = []

    for test_season in seasons:
        print(f"\n=== OOS: {test_season} ===")
        cols = B_MIN_MAX_FEATS + [TARGET] + GROUP_KEYS
        sigma_cols = [f"roll_reb_std_{w}" for w in SIGMA_WINDOWS]

        train = feat[feat["season"] != test_season].dropna(subset=cols)
        test  = feat[feat["season"] == test_season].dropna(subset=cols + sigma_cols)

        X_tr = sm.add_constant(train[B_MIN_MAX_FEATS].astype(float), has_constant="add")
        m = sm.OLS(train[TARGET].astype(float), X_tr).fit()

        X_te = sm.add_constant(test[B_MIN_MAX_FEATS].astype(float), has_constant="add")
        yhat = m.predict(X_te).to_numpy()

        scored = test[GROUP_KEYS + sigma_cols].copy()
        scored["yhat"]      = yhat
        scored["consensus"] = test["consensus_reb_line"].astype(float).to_numpy()

        props_test = props[props["season"] == test_season].copy()
        base = props_test.merge(scored, on=GROUP_KEYS, how="inner")

        over_odds_arr  = base["over_odds"].astype(float).to_numpy()
        under_odds_arr = base["under_odds"].astype(float).to_numpy()
        line_arr       = base["line"].astype(float).to_numpy()
        reb_arr        = base["REB"].astype(float).to_numpy()
        consensus_arr  = base["consensus"].to_numpy()
        yhat_arr       = base["yhat"].to_numpy()
        tier_arr       = base["recognition_tier"].to_numpy()

        p_book_o = np.array([american_to_implied(x) for x in over_odds_arr])
        p_book_u = np.array([american_to_implied(x) for x in under_odds_arr])
        is_superstar = tier_arr == "superstar"

        for sig_w in SIGMA_WINDOWS:
            sigma = base[f"roll_reb_std_{sig_w}"].astype(float).clip(lower=SIGMA_FLOOR).to_numpy()

            for shrink in SHRINKAGES:
                mean_adj = consensus_arr + (1.0 - shrink) * (yhat_arr - consensus_arr)
                z        = (line_arr - mean_adj) / sigma
                p_over   = 1.0 - norm.cdf(z)
                p_under  = norm.cdf(z)
                edge_o   = p_over  - p_book_o
                edge_u   = p_under - p_book_u

                for min_edge in MIN_EDGES:
                    for side_policy in SIDE_POLICIES:
                        for tier_filter in TIER_FILTERS:
                            mask = np.ones(len(base), dtype=bool)
                            if tier_filter == "non_superstar":
                                mask &= ~is_superstar

                            if side_policy == "under_only":
                                bet_mask = mask & (edge_u >= min_edge)
                                sides = np.where(bet_mask, "under", None)
                            elif side_policy == "over_only":
                                bet_mask = mask & (edge_o >= min_edge)
                                sides = np.where(bet_mask, "over", None)
                            else:  # both
                                bet_o = mask & (edge_o >= min_edge)
                                bet_u = mask & (edge_u >= min_edge)
                                # where both qualify, pick stronger edge
                                both  = bet_o & bet_u
                                over_stronger = edge_o >= edge_u
                                sides = np.full(len(base), None, dtype=object)
                                sides[bet_o & ~bet_u] = "over"
                                sides[bet_u & ~bet_o] = "under"
                                sides[both &  over_stronger] = "over"
                                sides[both & ~over_stronger] = "under"
                                bet_mask = sides != None  # noqa: E711

                            n_bets = n_win = n_push = 0
                            pnl = 0.0

                            for i in np.where(bet_mask)[0]:
                                reb  = reb_arr[i]
                                line = line_arr[i]
                                if reb == line:
                                    n_push += 1
                                    continue
                                side = sides[i]
                                odds = under_odds_arr[i] if side == "under" else over_odds_arr[i]
                                won  = (reb < line) if side == "under" else (reb > line)
                                n_bets += 1
                                pnl   += american_profit(odds) if won else -1.0
                                n_win += int(won)

                            roi = pnl / n_bets if n_bets else float("nan")
                            all_rows.append({
                                "test_season":  test_season,
                                "sigma_window": sig_w,
                                "shrinkage":    shrink,
                                "min_edge":     min_edge,
                                "side_policy":  side_policy,
                                "tier_filter":  tier_filter,
                                "n_bets":       n_bets,
                                "n_push":       n_push,
                                "n_win":        n_win,
                                "pnl_u":        pnl,
                                "roi":          roi,
                                "hit_rate":     n_win / n_bets if n_bets else float("nan"),
                            })

        print(f"  combos scored: {len(SIGMA_WINDOWS)*len(SHRINKAGES)*len(MIN_EDGES)*len(SIDE_POLICIES)*len(TIER_FILTERS)}")

    results = pd.DataFrame(all_rows)

    # Aggregate across seasons.
    agg = (
        results.groupby(["sigma_window","shrinkage","min_edge","side_policy","tier_filter"])
        .agg(n_bets=("n_bets","sum"), pnl_u=("pnl_u","sum"), seasons_pos=("pnl_u", lambda x: (x>0).sum()))
        .reset_index()
    )
    agg["roi"] = agg["pnl_u"] / agg["n_bets"]

    # Production params: sigma=5, shrink=0, min_edge=0.05, under_only.
    prod = agg[
        (agg["sigma_window"]==5) & (agg["shrinkage"]==0.0) &
        (agg["min_edge"]==0.05)  & (agg["side_policy"]=="under_only")
    ].sort_values("tier_filter")

    print("\n\n=== PRODUCTION PARAMS (σ=5, shrink=0, edge>5pp, under_only) ===")
    print(prod[["tier_filter","n_bets","pnl_u","roi","seasons_pos"]].to_string(index=False))

    # Under-only sweep: compare all vs non_superstar across all param combos.
    under_agg = agg[agg["side_policy"]=="under_only"].copy()
    pivot = under_agg.pivot_table(
        index=["sigma_window","shrinkage","min_edge"],
        columns="tier_filter",
        values=["roi","pnl_u","n_bets"],
        aggfunc="first",
    )
    pivot.columns = ["_".join(c) for c in pivot.columns]
    pivot["roi_delta_pp"] = (pivot["roi_non_superstar"] - pivot["roi_all"]) * 100
    pivot["pnl_delta"]    = pivot["pnl_u_non_superstar"] - pivot["pnl_u_all"]
    pivot = pivot.reset_index()

    print("\n=== UNDER ONLY — ROI delta (non_superstar − all) across param combos ===")
    print(pivot[["sigma_window","shrinkage","min_edge","roi_all","roi_non_superstar","roi_delta_pp","pnl_delta"]].to_string(index=False))

    n_positive = (pivot["roi_delta_pp"] > 0).sum()
    print(f"\nnon_superstar beats all: {n_positive}/{len(pivot)} combos (under_only)")
    print(f"mean roi_delta_pp: {pivot['roi_delta_pp'].mean():+.2f}pp")
    print(f"mean pnl_delta:    {pivot['pnl_delta'].mean():+.1f}u")

    out = Path(args.out).expanduser()
    results.to_parquet(out, index=False)
    print(f"\nFull per-season results → {out}")


if __name__ == "__main__":
    main()
