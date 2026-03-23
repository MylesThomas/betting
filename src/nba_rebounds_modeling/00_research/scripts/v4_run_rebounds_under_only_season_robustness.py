"""
OOS robustness: under_only, sigma=roll_reb_std_5, shrink=0, min_edge in {0.05, 0.10}.

Context:
- For each test season in {2023-24, 2024-25, 2025-26}, trains B_min_max OLS on all
  other seasons, predicts yhat on test-season player-games, merges v3 book rows,
  places under bets when Normal P(under) - p_under_novig > min_edge.
- Same Option A Normal / no-vig edge / vigged P&L as v3_run_rebounds_edge_backtest.py.
- Does not sweep shrink, sigma, or side_policy (fixed per research plan).

Usage:
    python src/nba_rebounds_modeling/00_research/scripts/v4_run_rebounds_under_only_season_robustness.py \\
        --feat ~/Downloads/tmp/rebounds_model_features_v2.parquet \\
        --v3 ~/Downloads/tmp/v3_rebounds_props_raw.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm


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

TARGET = "REB"
B_MIN_MAX_FEATS = [
    "min_line",
    "max_line",
    "spread_signed",
    "roll_reb_mean_60",
    "roll_fg3a_mean_20",
    "roll_reb_std_5",
]
GROUP_KEYS = ["season", "date", "player_normalized", "game_id"]
SIGMA_COL = "roll_reb_std_5"
SHRINKAGE = 0.0
MIN_EDGES = [0.05, 0.10]
DEFAULT_TEST_SEASONS = ["2023-24", "2024-25", "2025-26"]
SIGMA_FLOOR = 0.25


def american_profit_on_win(american: float) -> float:
    if np.isnan(american):
        return float("nan")
    if american >= 100:
        return float(american) / 100.0
    return 100.0 / float(abs(american))


def american_to_implied_prob_vigged(american: float) -> float:
    if np.isnan(american):
        return float("nan")
    if american < 0:
        return float((-american) / ((-american) + 100.0))
    return float(100.0 / (american + 100.0))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Under-only fixed-config OOS check across test seasons."
    )
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
    p.add_argument(
        "--test-seasons",
        type=str,
        default=",".join(DEFAULT_TEST_SEASONS),
        help="Comma-separated seasons to use as test holdouts (each run excludes one).",
    )
    p.add_argument(
        "--out-csv",
        type=str,
        default="",
        help="Optional path to write the 6-row summary CSV.",
    )
    return p.parse_args()


def run_one_test_season(
    feat: pd.DataFrame,
    v3: pd.DataFrame,
    test_season: str,
) -> list[dict]:
    train = feat[feat["season"] != test_season].copy()
    feat_test = feat[feat["season"] == test_season].copy()

    cols_needed = B_MIN_MAX_FEATS + [TARGET]
    train_m = train.dropna(subset=cols_needed)
    if len(train_m) < 100:
        raise ValueError(f"train_m too small for test_season={test_season}: n={len(train_m)}")

    X_tr = sm.add_constant(train_m[B_MIN_MAX_FEATS].astype(float), has_constant="add")
    y_tr = train_m[TARGET].astype(float)
    m = sm.OLS(y_tr, X_tr).fit()

    test_m = feat_test.dropna(subset=cols_needed + GROUP_KEYS)
    X_te = sm.add_constant(test_m[B_MIN_MAX_FEATS].astype(float), has_constant="add")
    yhat = m.predict(X_te).to_numpy()

    pred = test_m[GROUP_KEYS].copy()
    pred["yhat"] = yhat
    pred[SIGMA_COL] = test_m[SIGMA_COL].to_numpy()

    v3_test = v3[v3["season"] == test_season].copy()
    base = v3_test.merge(pred, on=GROUP_KEYS, how="inner")

    sigma = base[SIGMA_COL].astype(float).clip(lower=SIGMA_FLOOR).to_numpy()
    consensus = base["consensus_reb_line"].astype(float).to_numpy()
    line = base["line"].astype(float).to_numpy()
    reb = base["REB"].astype(float).to_numpy()
    yhat_arr = base["yhat"].to_numpy()
    p_nov_u = base["p_under_novig"].astype(float).to_numpy()
    under_odds = base["under_odds"].astype(float).to_numpy()

    mean_adj = consensus + (1.0 - SHRINKAGE) * (yhat_arr - consensus)
    z = (line - mean_adj) / sigma
    p_under = norm.cdf(z)
    edge_u = p_under - p_nov_u

    rows = []
    for min_edge in MIN_EDGES:
        pnl_total = 0.0
        n_bets = 0
        n_win = 0
        n_push = 0
        sum_imp = 0.0
        sum_am = 0.0
        for i in range(len(base)):
            if edge_u[i] <= min_edge:
                continue
            if reb[i] == line[i]:
                n_push += 1
                continue
            odds_am = under_odds[i]
            won = reb[i] < line[i]
            n_bets += 1
            sum_imp += american_to_implied_prob_vigged(odds_am)
            sum_am += float(odds_am)
            if won:
                pnl_total += american_profit_on_win(odds_am)
                n_win += 1
            else:
                pnl_total -= 1.0

        roi = pnl_total / n_bets if n_bets else float("nan")
        hit = n_win / n_bets if n_bets else float("nan")
        mean_imp = sum_imp / n_bets if n_bets else float("nan")
        mean_am = sum_am / n_bets if n_bets else float("nan")
        rows.append({
            "test_season":       test_season,
            "train_seasons":     ",".join(sorted(s for s in feat["season"].unique() if s != test_season)),
            "min_edge":          min_edge,
            "n_train_rows":      int(len(train_m)),
            "n_test_feat_rows":  int(len(test_m)),
            "n_v3_test_rows":    int(len(v3_test)),
            "n_merged_rows":     int(len(base)),
            "n_bets":            n_bets,
            "n_push":            n_push,
            "n_win":             n_win,
            "hit_rate":          hit,
            "mean_implied_prob_vigged": mean_imp,
            "mean_american_odds":      mean_am,
            "total_pnl_u":       pnl_total,
            "roi":               roi,
        })
    return rows


def main() -> None:
    args = parse_args()
    feat_path = Path(args.feat).expanduser()
    v3_path = Path(args.v3).expanduser()

    test_seasons = [s.strip() for s in args.test_seasons.split(",") if s.strip()]

    feat = pd.read_parquet(feat_path)
    v3 = pd.read_parquet(v3_path)

    for c in B_MIN_MAX_FEATS + [TARGET, "consensus_reb_line"]:
        if c not in feat.columns:
            raise ValueError(f"feat parquet missing column: {c}")
    if SIGMA_COL not in feat.columns:
        raise ValueError(f"feat missing {SIGMA_COL}")

    all_rows: list[dict] = []
    for ts in test_seasons:
        if ts not in feat["season"].unique():
            raise ValueError(f"test_season {ts!r} not in feat['season']")
        print(f"\n--- test_season={ts} ---")
        part = run_one_test_season(feat, v3, ts)
        for r in part:
            print(
                f"  min_edge={r['min_edge']:.2f}  n_bets={r['n_bets']:,}  "
                f"hit={r['hit_rate']:.4f}  roi={r['roi']:.6f}  pnl_u={r['total_pnl_u']:.2f}"
            )
        all_rows.extend(part)

    out = pd.DataFrame(all_rows)
    out = out.sort_values(["test_season", "min_edge"]).reset_index(drop=True)

    print("\n=== SUMMARY (under_only | sigma=5 | shrink=0) ===")
    print(out.to_string(index=False))

    by_season = out.groupby("test_season")["roi"].apply(lambda s: bool((s > 0).all()))
    n_seasons_both_pos = int(by_season.sum())
    print(
        f"\nseasons where BOTH min_edges have roi>0: {n_seasons_both_pos} / {len(test_seasons)}"
    )

    if args.out_csv:
        outp = Path(args.out_csv).expanduser()
        outp.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(outp, index=False)
        print(f"wrote {outp}")


if __name__ == "__main__":
    main()
