"""
Dedupe comparison: before/after table across all seasons.

For each season (leave-one-out OOS), runs backtest at prod params with and without
--dedupe, then builds a side-by-side delta table.

Prod params: sigma_window=5, shrinkage=0.0, min_edge=0.05
Reports: under_only, over_only, both side policies

Usage:
    python src/nba_rebounds_modeling/00_research/scripts/compare_dedupe_backtest.py \
        --feat ~/Downloads/tmp/rebounds_features.parquet \
        --v3 ~/Downloads/tmp/rebounds_props.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve()
for _ in range(10):
    ROOT = ROOT.parent
    if (ROOT / "src").exists() and (ROOT / ".gitignore").exists():
        break
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.nba_rebounds_modeling.option_a_scoring import (
    PROD_MIN_EDGE,
    PROD_SHRINK,
    option_a_vector_batch,
    pick_side,
)
from src.nba_rebounds_modeling.rebounds_feature_spec import (
    B_MIN_MAX_FEATS,
    GROUP_KEYS,
    TARGET,
)


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


def select_bet_rows(
    n: int,
    dedup_keys: list[tuple],
    edge_o: np.ndarray,
    edge_u: np.ndarray,
    min_edge: float,
    side_policy: str,
    over_odds: np.ndarray,
    under_odds: np.ndarray,
    dedupe: bool,
) -> list[tuple[int, str]]:
    if not dedupe:
        return [
            (i, s)
            for i in range(n)
            if (s := pick_side(i, edge_o, edge_u, min_edge, side_policy)) is not None
        ]

    group_best: dict[tuple, tuple[int, str, float, float]] = {}
    for i in range(n):
        side = pick_side(i, edge_o, edge_u, min_edge, side_policy)
        if side is None:
            continue
        dk = dedup_keys[i]
        odds_am = float(over_odds[i] if side == "over" else under_odds[i])
        best_edge = float(edge_o[i] if side == "over" else edge_u[i])
        if dk not in group_best:
            group_best[dk] = (i, side, best_edge, odds_am)
        else:
            _, _, prev_edge, prev_odds = group_best[dk]
            if best_edge > prev_edge or (best_edge == prev_edge and odds_am > prev_odds):
                group_best[dk] = (i, side, best_edge, odds_am)

    return [(idx, side) for idx, side, _, _ in group_best.values()]

PROD_SIGMA_WINDOW = 5
SIGMA_FLOOR = 0.25
SIDE_POLICIES = ("under_only", "over_only", "both")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--feat", default="~/Downloads/tmp/rebounds_features.parquet")
    p.add_argument("--v3", default="~/Downloads/tmp/rebounds_props.parquet")
    return p.parse_args()


def run_one_season(
    feat: pd.DataFrame,
    v3: pd.DataFrame,
    test_season: str,
    side_policy: str,
    dedupe: bool,
) -> dict:
    train = feat[feat["season"] != test_season]
    feat_test = feat[feat["season"] == test_season]

    cols_needed = B_MIN_MAX_FEATS + [TARGET]
    train_m = train.dropna(subset=cols_needed)
    X_tr = sm.add_constant(train_m[B_MIN_MAX_FEATS].astype(float), has_constant="add")
    y_tr = train_m[TARGET].astype(float)
    model = sm.OLS(y_tr, X_tr).fit()

    test_m = feat_test.dropna(subset=cols_needed + GROUP_KEYS)
    X_te = sm.add_constant(test_m[B_MIN_MAX_FEATS].astype(float), has_constant="add")
    yhat = model.predict(X_te).to_numpy()

    pred = test_m[GROUP_KEYS].copy()
    pred["yhat"] = yhat
    sig_col = f"roll_reb_std_{PROD_SIGMA_WINDOW}"
    pred[sig_col] = test_m[sig_col].to_numpy()

    v3_test = v3[v3["season"] == test_season]
    base = v3_test.merge(pred, on=GROUP_KEYS, how="inner")

    dedup_keys = list(zip(
        base["player_normalized"].tolist(),
        base["game_id"].tolist(),
        base["line"].tolist(),
    ))

    sigma = base[sig_col].astype(float).clip(lower=SIGMA_FLOOR).to_numpy()
    consensus = base["consensus_reb_line"].astype(float).to_numpy()
    line = base["line"].astype(float).to_numpy()
    reb = base["REB"].astype(float).to_numpy()
    yhat_arr = base["yhat"].to_numpy()
    over_odds = base["over_odds"].astype(float).to_numpy()
    under_odds = base["under_odds"].astype(float).to_numpy()
    p_book_o = np.array([american_to_implied_prob_vigged(x) for x in over_odds], dtype=np.float64)
    p_book_u = np.array([american_to_implied_prob_vigged(x) for x in under_odds], dtype=np.float64)

    _, _, _, _, edge_o, edge_u = option_a_vector_batch(
        consensus, yhat_arr, line, sigma, PROD_SHRINK, p_book_o, p_book_u
    )

    bets = select_bet_rows(
        len(base), dedup_keys, edge_o, edge_u,
        PROD_MIN_EDGE, side_policy, over_odds, under_odds, dedupe,
    )

    pnl_total = 0.0
    n_bets = n_win = n_push = 0
    sum_imp_vigged = sum_american = 0.0

    for i, side in bets:
        if reb[i] == line[i]:
            n_push += 1
            continue
        odds_am = over_odds[i] if side == "over" else under_odds[i]
        won = (reb[i] > line[i]) if side == "over" else (reb[i] < line[i])
        n_bets += 1
        sum_imp_vigged += american_to_implied_prob_vigged(odds_am)
        sum_american += float(odds_am)
        if won:
            pnl_total += american_profit_on_win(odds_am)
            n_win += 1
        else:
            pnl_total -= 1.0

    return {
        "season":     test_season,
        "side_policy": side_policy,
        "dedupe":     dedupe,
        "n_bets":     n_bets,
        "n_push":     n_push,
        "n_win":      n_win,
        "hit_rate":   n_win / n_bets if n_bets else float("nan"),
        "total_pnl":  pnl_total,
        "roi":        pnl_total / n_bets if n_bets else float("nan"),
        "mean_am_odds": sum_american / n_bets if n_bets else float("nan"),
        "mean_imp_prob": sum_imp_vigged / n_bets if n_bets else float("nan"),
    }


def build_comparison_table(rows: pd.DataFrame, side_policy: str) -> pd.DataFrame:
    sub = rows[rows["side_policy"] == side_policy].copy()
    before = sub[~sub["dedupe"]].set_index("season")
    after  = sub[ sub["dedupe"]].set_index("season")

    records = []
    for season in sorted(sub["season"].unique()) + ["ALL"]:
        if season == "ALL":
            b_n   = int(before["n_bets"].sum())
            b_nw  = int(before["n_win"].sum())
            b_pnl = before["total_pnl"].sum()
            a_n   = int(after["n_bets"].sum())
            a_nw  = int(after["n_win"].sum())
            a_pnl = after["total_pnl"].sum()
            b_hr  = b_nw / b_n if b_n else float("nan")
            a_hr  = a_nw / a_n if a_n else float("nan")
            b_roi = b_pnl / b_n if b_n else float("nan")
            a_roi = a_pnl / a_n if a_n else float("nan")
            b_am  = before["mean_am_odds"].mean()
            a_am  = after["mean_am_odds"].mean()
        else:
            b = before.loc[season]
            a = after.loc[season]
            b_n, b_nw, b_pnl = int(b["n_bets"]), int(b["n_win"]), b["total_pnl"]
            a_n, a_nw, a_pnl = int(a["n_bets"]), int(a["n_win"]), a["total_pnl"]
            b_hr, a_hr = b["hit_rate"], a["hit_rate"]
            b_roi, a_roi = b["roi"], a["roi"]
            b_am, a_am = b["mean_am_odds"], a["mean_am_odds"]

        records.append({
            "season":           season if season != "ALL" else "ALL (3-season)",
            "n_bets_before":    b_n,
            "n_bets_after":     a_n,
            "n_bets_Δ":         a_n - b_n,
            "hit_rate_before":  b_hr,
            "hit_rate_after":   a_hr,
            "hit_rate_Δ":       a_hr - b_hr,
            "total_pnl_before": b_pnl,
            "total_pnl_after":  a_pnl,
            "total_pnl_Δ":      a_pnl - b_pnl,
            "roi_before":       b_roi,
            "roi_after":        a_roi,
            "roi_Δ":            a_roi - b_roi,
            "avg_odds_before":  b_am,
            "avg_odds_after":   a_am,
            "avg_odds_Δ":       a_am - b_am,
        })

    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    feat = pd.read_parquet(Path(args.feat).expanduser())
    v3   = pd.read_parquet(Path(args.v3).expanduser())

    seasons = sorted(feat["season"].unique())
    print(f"Seasons: {seasons}")
    print(f"Prod params: sigma_window={PROD_SIGMA_WINDOW}, shrinkage={PROD_SHRINK}, min_edge={PROD_MIN_EDGE}\n")

    raw_rows = []
    for season in seasons:
        for side_policy in SIDE_POLICIES:
            for dedupe in (False, True):
                row = run_one_season(feat, v3, season, side_policy, dedupe)
                raw_rows.append(row)
                label = "dedupe" if dedupe else "raw  "
                print(
                    f"  {season}  {side_policy:<10}  {label}  "
                    f"n_bets={row['n_bets']:4d}  "
                    f"pnl={row['total_pnl']:+7.2f}u  "
                    f"roi={row['roi']:+.3f}  "
                    f"hit={row['hit_rate']:.3f}"
                )

    all_rows = pd.DataFrame(raw_rows)

    print("\n" + "=" * 100)
    for side_policy in SIDE_POLICIES:
        tbl = build_comparison_table(all_rows, side_policy)
        print(f"\n{'='*100}")
        print(f"  BEFORE vs AFTER DEDUPE  |  side_policy={side_policy}  |  "
              f"sigma_window={PROD_SIGMA_WINDOW}  shrinkage={PROD_SHRINK}  min_edge={PROD_MIN_EDGE}")
        print(f"{'='*100}")

        pd.set_option("display.float_format", lambda x: f"{x:+.3f}" if abs(x) < 1000 else f"{x:+.1f}")
        pd.set_option("display.max_columns", 20)
        pd.set_option("display.width", 200)

        fmt = {
            "n_bets_before":     "{:.0f}",
            "n_bets_after":      "{:.0f}",
            "n_bets_delta":      "{:+.0f}",
            "hit_rate_before":   "{:.3f}",
            "hit_rate_after":    "{:.3f}",
            "hit_rate_delta":    "{:+.4f}",
            "total_pnl_before":  "{:+.2f}u",
            "total_pnl_after":   "{:+.2f}u",
            "total_pnl_delta":   "{:+.2f}u",
            "roi_before":        "{:+.4f}",
            "roi_after":         "{:+.4f}",
            "roi_delta":         "{:+.4f}",
            "mean_am_odds_before": "{:.1f}",
            "mean_am_odds_after":  "{:.1f}",
            "mean_am_odds_delta":  "{:+.1f}",
        }
        print(tbl.to_string(index=False, formatters={k: (lambda v, f=f: f.format(v)) for k, f in fmt.items()}))


if __name__ == "__main__":
    main()
