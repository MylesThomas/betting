"""
Option A edge backtest: Normal(mean_adj, sigma) vs book raw implied prob, per-book rows.

Context:
- v3_rebounds_props_raw.parquet has one row per game/player/bookmaker/posted line
  with over_odds, under_odds, p_over_novig, p_under_novig, REB, consensus_reb_line.
- rebounds_model_features_v2.parquet supplies B_min_max inputs + roll_reb_std_N for sigma.
- Trains OLS on all seasons except --test-season; sweeps shrinkage × sigma × min_edge.
- Each sweep row includes mean vigged implied prob and mean American odds on bets placed.
- side_policy: both (pick stronger edge), over_only, under_only.

Usage:
    python src/nba_rebounds_modeling/00_research/scripts/v3_run_rebounds_edge_backtest.py \\
        --feat ~/Downloads/tmp/rebounds_model_features_v2.parquet \\
        --v3 ~/Downloads/tmp/v3_rebounds_props_raw.parquet \\
        --test-season 2025-26
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.nba_rebounds_modeling.option_a_scoring import option_a_vector_batch, pick_side
from src.nba_rebounds_modeling.rebounds_feature_spec import (
    B_MIN_MAX_FEATS,
    GROUP_KEYS,
    TARGET,
)


def ensure_repo_root_on_syspath() -> Path:
    current = Path.cwd().resolve()
    while True:
        if (current / ".gitignore").exists() and (current / "src").exists():
            if str(current) not in sys.path:
                sys.path.insert(0, str(current))
            return current
        if current.parent == current:
            raise FileNotFoundError("Could not locate repo root")
        current = current.parent


ensure_repo_root_on_syspath()

SIGMA_WINDOWS = [5, 10, 20]
SHRINKAGES = [0.0, 0.25, 0.50, 0.75]
MIN_EDGES = [0.01, 0.05, 0.10]
SIDE_POLICIES = ("both", "over_only", "under_only")
N_BET_WARN = 100


def american_profit_on_win(american: float) -> float:
    if np.isnan(american):
        return float("nan")
    if american >= 100:
        return float(american) / 100.0
    return 100.0 / float(abs(american))


def american_to_implied_prob_vigged(american: float) -> float:
    """Raw implied probability from American odds (includes vig)."""
    if np.isnan(american):
        return float("nan")
    if american < 0:
        return float((-american) / ((-american) + 100.0))
    return float(100.0 / (american + 100.0))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rebounds Option A edge backtest on v3 props.")
    p.add_argument(
        "--feat",
        type=str,
        default="~/Downloads/tmp/rebounds_model_features_v2.parquet",
    )
    p.add_argument(
        "--v3",
        type=str,
        default="~/Downloads/tmp/v3_rebounds_props_raw.parquet",
    )
    p.add_argument("--test-season", type=str, default="")
    p.add_argument("--out-png", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    feat_path = Path(args.feat).expanduser()
    v3_path = Path(args.v3).expanduser()

    feat = pd.read_parquet(feat_path)
    v3 = pd.read_parquet(v3_path)

    for c in B_MIN_MAX_FEATS + [TARGET, "consensus_reb_line"]:
        if c not in feat.columns:
            raise ValueError(f"feat parquet missing column: {c}")

    test_season = args.test_season.strip() or str(feat["season"].max())
    train = feat[feat["season"] != test_season].copy()
    feat_test = feat[feat["season"] == test_season].copy()

    cols_needed = B_MIN_MAX_FEATS + [TARGET]
    train_m = train.dropna(subset=cols_needed)
    X_tr = sm.add_constant(train_m[B_MIN_MAX_FEATS].astype(float), has_constant="add")
    y_tr = train_m[TARGET].astype(float)
    m = sm.OLS(y_tr, X_tr).fit()

    test_m = feat_test.dropna(subset=cols_needed + GROUP_KEYS)
    X_te = sm.add_constant(test_m[B_MIN_MAX_FEATS].astype(float), has_constant="add")
    yhat = m.predict(X_te).to_numpy()
    pred = test_m[GROUP_KEYS].copy()
    pred["yhat"] = yhat

    for w in SIGMA_WINDOWS:
        col = f"roll_reb_std_{w}"
        if col not in feat.columns:
            raise ValueError(f"feat missing {col}")
        pred[col] = test_m[col].to_numpy()

    v3_test = v3[v3["season"] == test_season].copy()
    base = v3_test.merge(pred, on=GROUP_KEYS, how="inner")
    print(
        f"test_season={test_season}  v3_rows={len(v3_test):,}  "
        f"after_merge={len(base):,}  train_n={len(train_m):,}"
    )

    rows_out: list[dict] = []
    best_by_policy: dict[str, tuple[float, int]] = {
        p: (-1e18, SIGMA_WINDOWS[0]) for p in SIDE_POLICIES
    }

    for side_policy in SIDE_POLICIES:
        for sig_w in SIGMA_WINDOWS:
            sig_col = f"roll_reb_std_{sig_w}"
            sigma = base[sig_col].astype(float).clip(lower=0.25).to_numpy()
            consensus = base["consensus_reb_line"].astype(float).to_numpy()
            line = base["line"].astype(float).to_numpy()
            reb = base["REB"].astype(float).to_numpy()
            yhat_arr = base["yhat"].to_numpy()
            over_odds = base["over_odds"].astype(float).to_numpy()
            under_odds = base["under_odds"].astype(float).to_numpy()
            p_book_o = np.array([american_to_implied_prob_vigged(x) for x in over_odds], dtype=np.float64)
            p_book_u = np.array([american_to_implied_prob_vigged(x) for x in under_odds], dtype=np.float64)

            combo_pnl = 0.0
            for shrink in SHRINKAGES:
                mean_adj, z, p_over, p_under, edge_o, edge_u = option_a_vector_batch(
                    consensus,
                    yhat_arr,
                    line,
                    sigma,
                    shrink,
                    p_book_o,
                    p_book_u,
                )

                for min_edge in MIN_EDGES:
                    pnl_total = 0.0
                    n_bets = 0
                    n_win = 0
                    n_push = 0
                    sum_imp_vigged = 0.0
                    sum_american = 0.0

                    for i in range(len(base)):
                        side = pick_side(i, edge_o, edge_u, min_edge, side_policy)
                        if side is None:
                            continue
                        if reb[i] == line[i]:
                            n_push += 1
                            continue
                        if side == "over":
                            odds_am = over_odds[i]
                            won = reb[i] > line[i]
                        else:
                            odds_am = under_odds[i]
                            won = reb[i] < line[i]

                        n_bets += 1
                        sum_imp_vigged += american_to_implied_prob_vigged(odds_am)
                        sum_american += float(odds_am)

                        if won:
                            pnl_total += american_profit_on_win(odds_am)
                            n_win += 1
                        else:
                            pnl_total -= 1.0

                    roi = pnl_total / n_bets if n_bets else float("nan")
                    hit = n_win / n_bets if n_bets else float("nan")
                    mean_imp = sum_imp_vigged / n_bets if n_bets else float("nan")
                    mean_am = sum_american / n_bets if n_bets else float("nan")
                    rows_out.append({
                        "side_policy":   side_policy,
                        "sigma_window":  sig_w,
                        "shrinkage":     shrink,
                        "min_edge":      min_edge,
                        "n_bets":        n_bets,
                        "n_push":        n_push,
                        "n_win":         n_win,
                        "hit_rate":      hit,
                        "mean_implied_prob_vigged": mean_imp,
                        "mean_american_odds":      mean_am,
                        "total_pnl_u":   pnl_total,
                        "roi":           roi,
                    })
                    combo_pnl += pnl_total

            if combo_pnl > best_by_policy[side_policy][0]:
                best_by_policy[side_policy] = (combo_pnl, sig_w)

    out = pd.DataFrame(rows_out)
    print("\n=== SWEEP top 15 per side_policy (by total_pnl_u) ===")
    for pol in SIDE_POLICIES:
        subp = out[out["side_policy"] == pol].sort_values(
            ["total_pnl_u", "n_bets"], ascending=[False, False]
        ).head(15)
        print(f"\n--- {pol} ---")
        print(subp.to_string(index=False))

    low_n = out[out["n_bets"] < N_BET_WARN]
    if len(low_n) > 0:
        print(
            f"\n[WARN] {len(low_n)} rows have n_bets < {N_BET_WARN}"
        )

    for pol in SIDE_POLICIES:
        sig_star = best_by_policy[pol][1]
        sub = out[(out["side_policy"] == pol) & (out["sigma_window"] == sig_star)].copy()
        pivot = sub.pivot_table(
            index="shrinkage", columns="min_edge", values="roi", aggfunc="first"
        )
        print(f"\n=== ROI heatmap  side_policy={pol}  sigma_window={sig_star} ===")
        print(pivot.to_string())

    if args.out_png:
        try:
            import matplotlib.pyplot as plt

            pol = "both"
            sig_star = best_by_policy[pol][1]
            sub = out[(out["side_policy"] == pol) & (out["sigma_window"] == sig_star)]
            pivot = sub.pivot_table(
                index="shrinkage", columns="min_edge", values="roi", aggfunc="first"
            )
            fig, ax = plt.subplots(figsize=(6, 4))
            im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=-0.15, vmax=0.15)
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([str(c) for c in pivot.columns])
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([str(s) for s in pivot.index])
            ax.set_xlabel("min_edge")
            ax.set_ylabel("shrinkage")
            ax.set_title(f"ROI {pol} test={test_season} sigma={sig_star}")
            plt.colorbar(im, ax=ax, label="ROI")
            plt.tight_layout()
            Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(args.out_png, dpi=150)
            print(f"saved {args.out_png}")
            plt.close()
        except Exception as e:
            print(f"[WARN] could not save heatmap: {e}")

    print("\n(OOS test season only; P&L at posted American odds; edge vs raw implied.)")


if __name__ == "__main__":
    main()
