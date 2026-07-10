"""
Step 5 — Out-of-sample grid search for NFL WR/TE receiving yards.

OOF approach (temporal walk-forward):
  - 2023 (first season, no prior data): within-season split — odd weeks → even weeks,
    then even weeks → odd weeks; concatenate to cover full season.
  - 2024: train on 2023, evaluate on 2024.
  - 2025: train on 2023+2024, evaluate on 2025.

New dimensions vs prior sweep:
  - odds_bucket  : all / plus_odds (novig market_over_prob < 0.50) / minus_odds (> 0.50)
  - shrinkage    : [0, 0.25, 0.50, 0.75] — shrinks ols_pred / nb_mu toward fold training mean
  - prediction_method: model (ML yhat) vs consensus_line (mean offered_line across books)

All sweep values loaded from config/model_config.yaml grid_search block.

Output:
  ~/Downloads/tmp/nfl_rec_yards_step5_v2.csv
"""

from __future__ import annotations

import itertools
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml
from scipy.stats import nbinom
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.discrete.discrete_model import NegativeBinomial

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
LABELED_PATH = Path.home() / "Downloads" / "tmp" / "nfl_rec_yards_per_book.parquet"
CONFIG_PATH  = Path(__file__).parent.parent / "config" / "model_config.yaml"
OUT_CSV      = Path.home() / "Downloads" / "tmp" / "nfl_rec_yards_step5_v2.csv"

# ── Constants ─────────────────────────────────────────────────────────────────
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

# ── Model fitting ─────────────────────────────────────────────────────────────

def _fit(train_df: pd.DataFrame) -> dict:
    sub = train_df[BEST_FEATS + [TARGET]].dropna()
    X   = sub[BEST_FEATS].to_numpy(dtype=float)
    y   = sub[TARGET].to_numpy(dtype=float)

    ols = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])
    ols.fit(X, y)
    residuals = y - ols.predict(X)

    X_const   = sm.add_constant(X)
    nb_result = NegativeBinomial(y, X_const).fit(disp=False, maxiter=300)
    nb_coefs  = nb_result.params[:-1]
    nb_alpha  = float(np.exp(nb_result.lnalpha))

    mean_ols_pred = float(ols.predict(X).mean())
    mean_nb_mu    = float(np.clip(np.exp(X_const @ nb_coefs), 1e-3, None).mean())

    return {
        "ols":          ols,
        "residuals":    residuals,
        "nb_coefs":     nb_coefs,
        "nb_alpha":     nb_alpha,
        "mean_ols_pred": mean_ols_pred,
        "mean_nb_mu":   mean_nb_mu,
    }


def _predict(test_df: pd.DataFrame, fold: dict) -> pd.DataFrame:
    ols, residuals = fold["ols"], fold["residuals"]
    nb_coefs, nb_alpha = fold["nb_coefs"], fold["nb_alpha"]
    mean_ols, mean_nb = fold["mean_ols_pred"], fold["mean_nb_mu"]

    result = test_df.copy()
    mask   = result[BEST_FEATS].notna().all(axis=1)
    idx    = result.index[mask]
    if idx.empty:
        result["ols_pred"] = np.nan
        result["nb_mu"]    = np.nan
        result["mean_ols_pred"] = mean_ols
        result["mean_nb_mu"]    = mean_nb
        return result

    X      = result.loc[idx, BEST_FEATS].to_numpy(dtype=float)
    X_const = np.column_stack([np.ones(len(X)), X])

    result.loc[idx, "ols_pred"] = ols.predict(X)
    result.loc[idx, "nb_mu"]    = np.clip(np.exp(X_const @ nb_coefs), 1e-3, None)
    result["mean_ols_pred"] = mean_ols
    result["mean_nb_mu"]    = mean_nb
    result["_residuals"]    = [residuals] * len(result)  # store ref
    result["_nb_alpha"]     = nb_alpha
    return result


# ── P(under) computation ──────────────────────────────────────────────────────

def _compute_p_hybrid(df: pd.DataFrame, shrinkage: float,
                      pred_method: str) -> pd.Series:
    """Return p(under) for each row given shrinkage and prediction method."""
    mask = df["ols_pred"].notna()
    p    = pd.Series(np.nan, index=df.index)

    if not mask.any():
        return p

    sub       = df[mask].copy()
    line      = sub["offered_line"].to_numpy(dtype=float)

    if pred_method == "consensus_line":
        # Use mean offered_line across books for this player-game as yhat
        yhat_ols = sub["consensus_line"].to_numpy(dtype=float)
        yhat_nb  = sub["consensus_line"].to_numpy(dtype=float)
    else:
        yhat_ols = sub["ols_pred"].to_numpy(dtype=float)
        yhat_nb  = sub["nb_mu"].to_numpy(dtype=float)

    # Apply shrinkage (no-op when prediction_method=consensus_line per skill spec)
    if pred_method == "model" and shrinkage > 0:
        mean_ols = sub["mean_ols_pred"].to_numpy(dtype=float)
        mean_nb  = sub["mean_nb_mu"].to_numpy(dtype=float)
        yhat_ols = (1 - shrinkage) * yhat_ols + shrinkage * mean_ols
        yhat_nb  = (1 - shrinkage) * yhat_nb  + shrinkage * mean_nb

    # Bootstrap path (line < threshold)
    bt_mask  = line < HYBRID_NEGBIN_THRESHOLD
    nb_mask  = ~bt_mask

    p_hyb = np.empty(len(sub))

    if bt_mask.any():
        residuals = sub.loc[sub.index[bt_mask], "_residuals"].iloc[0]
        pred_bt   = yhat_ols[bt_mask]
        line_bt   = line[bt_mask]
        samp      = RNG.choice(residuals, size=(bt_mask.sum(), N_BOOT))
        p_hyb[bt_mask] = ((pred_bt[:, None] + samp) <= line_bt[:, None]).mean(axis=1)

    if nb_mask.any():
        nb_alpha = float(sub["_nb_alpha"].iloc[0])
        n_nb     = 1.0 / nb_alpha
        mu_nb    = np.clip(yhat_nb[nb_mask], 1e-3, None)
        p_hyb[nb_mask] = nbinom.cdf(
            np.floor(line[nb_mask]).astype(int),
            n=n_nb, p=n_nb / (n_nb + mu_nb),
        )

    p_hyb = np.clip(p_hyb, 0.01, 0.99)
    p.iloc[np.where(mask)[0]] = p_hyb
    return p


# ── OOF generation ────────────────────────────────────────────────────────────

def build_oof_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Generate walk-forward OOF predictions for all rows."""
    seasons = sorted(df["season"].unique())
    chunks  = []

    # Compute consensus_line per (player_name_norm, game_id) across all books
    consensus = (
        df.groupby(["player_name_norm", "game_id"])["offered_line"]
        .mean()
        .rename("consensus_line")
        .reset_index()
    )
    df = df.merge(consensus, on=["player_name_norm", "game_id"], how="left")

    for i, season in enumerate(seasons):
        test_df = df[df["season"] == season].copy()

        if i == 0:
            # No prior season: within-season odd-week / even-week alternating split
            weeks = sorted(test_df["week"].unique())
            odd_weeks  = [w for w in weeks if int(w) % 2 == 1]
            even_weeks = [w for w in weeks if int(w) % 2 == 0]
            parts = []
            for train_weeks, eval_weeks in [
                (odd_weeks, even_weeks),
                (even_weeks, odd_weeks),
            ]:
                if not train_weeks or not eval_weeks:
                    continue
                train = test_df[test_df["week"].isin(train_weeks)]
                evl   = test_df[test_df["week"].isin(eval_weeks)].copy()
                fold  = _fit(train)
                parts.append(_predict(evl, fold))
            if parts:
                chunks.append(pd.concat(parts))
            else:
                chunks.append(_predict(test_df, _fit(test_df)))
        else:
            train_df = df[df["season"].isin(seasons[:i])].copy()
            fold     = _fit(train_df)
            chunks.append(_predict(test_df, fold))

        print(f"    Season {season}: {len(test_df):,} rows → OOF predicted")

    oof = pd.concat(chunks).sort_index()
    return oof


# ── Sweep ─────────────────────────────────────────────────────────────────────

def _max_drawdown(bets: pd.DataFrame) -> float:
    ordered = bets.sort_values(["season", "week"])
    pnl     = np.where(ordered["bet_correct"] == 1,
                       ordered["payout"] - 1, -1.0)
    cs  = np.cumsum(pnl)
    mx  = np.maximum.accumulate(cs)
    return float((mx - cs).max()) if len(cs) else 0.0


def sweep(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    gs        = cfg["nfl_rec_yards_model"]["grid_search"]
    edges     = gs["edge_threshold"]
    directions = gs["direction"]
    buckets   = gs["odds_bucket"]
    shrinkages = gs["shrinkage"]
    methods   = gs["prediction_method"]
    min_books_opts = gs["min_books"]
    line_mins = gs["line_min"]
    line_maxes = gs["line_max"]

    total_rows = len(df[df[BEST_FEATS].notna().all(axis=1)])
    rows = []

    combos = list(itertools.product(
        methods, shrinkages, edges, directions, buckets, min_books_opts, line_mins, line_maxes
    ))
    print(f"  Running {len(combos):,} combos across {total_rows:,} scored rows...")

    # Pre-compute p_hybrid for each (method, shrinkage) combo — expensive part
    ph_cache: dict[tuple, pd.Series] = {}
    for method, shrinkage in itertools.product(methods, shrinkages):
        if method == "consensus_line" and shrinkage > 0:
            continue  # shrinkage has no meaning for consensus_line
        key = (method, shrinkage)
        if key not in ph_cache:
            print(f"    Computing p_hybrid: method={method}, shrinkage={shrinkage}...")
            ph_cache[key] = _compute_p_hybrid(df, shrinkage, method)

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

        # Odds bucket filter (based on novig market_over_prob = 1 - market_under_prob)
        mop = 1.0 - p_mkt  # market_over_prob (novig fair)
        bucket_mask = pd.Series(True, index=df.index)
        if bucket == "plus_odds":
            bucket_mask = mop < 0.50   # OVER is dog (positive odds)
        elif bucket == "minus_odds":
            bucket_mask = mop > 0.50   # OVER is fav (negative odds)

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

        bets = df[bet_mask].copy()
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

        bets["_p_hyb"]  = p_hyb[bet_mask].values
        bets["_edge_v"] = edge_v[bet_mask].values
        bets["_rec"]    = rec[bet_mask].values

        actual_under = (bets[TARGET] <= bets["offered_line"]).astype(float)
        bet_correct  = np.where(
            bets["_rec"] == "UNDER", actual_under,
            np.where(bets["_rec"] == "OVER", 1 - actual_under, np.nan),
        )
        push = (bets[TARGET] == bets["offered_line"]).astype(float)
        bets["bet_correct"] = bet_correct
        bets["push"]        = push
        bets["is_push"]     = push.astype(bool)

        n_win  = float(np.nansum(bet_correct * (1 - push)))
        n_push = float(push.sum())
        n_loss = n_bets - n_win - n_push

        # Payout: use raw (vig-inclusive) implied probs — actual prices you collect
        over_pay  = 1.0 / np.clip(bets["raw_over_prob"].to_numpy(float),  1e-6, 1.0) - 1.0
        under_pay = 1.0 / np.clip(bets["raw_under_prob"].to_numpy(float), 1e-6, 1.0) - 1.0
        side_pay  = np.where(bets["_rec"].to_numpy() == "OVER", over_pay, under_pay)
        bets["payout"] = np.where(bets["bet_correct"] == 1, side_pay, 0.0)

        units_won    = float((bets["bet_correct"] * (1 - bets["push"]) * side_pay).sum()) - n_loss
        mean_payout  = float(side_pay.mean())
        roi          = units_won / n_bets if n_bets else np.nan

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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\nNFL Rec Yards — Step 5: OOS Grid Search")
    print("=" * 50)

    cfg = yaml.safe_load(CONFIG_PATH.read_text())

    print("\n  Loading labeled data...")
    df = pd.read_parquet(LABELED_PATH)
    df = df[df["position"].isin(KEEP_POSITIONS)].copy()
    df["pos_TE"] = (df["position"] == "TE").astype(int)
    print(f"    {len(df):,} rows  |  seasons: {sorted(df['season'].unique())}")

    print("\n  Building OOF predictions (temporal walk-forward)...")
    oof = build_oof_predictions(df)
    n_scored = oof["ols_pred"].notna().sum()
    print(f"  OOF complete: {n_scored:,} rows with predictions out of {len(oof):,}")

    print("\n  Running grid search sweep...")
    results = sweep(oof, cfg)

    # Filter out rows where shrinkage>0 and prediction_method=consensus_line (skip combos)
    results = results[
        ~((results["prediction_method"] == "consensus_line") & (results["shrinkage"] > 0))
    ]

    results = results.sort_values("units_won", ascending=False).reset_index(drop=True)
    OUT_CSV.write_text("")
    results.to_csv(OUT_CSV, index=False)

    print(f"\n  Grid search complete: {len(results):,} combos")
    print(f"  Saved → {OUT_CSV}")

    # Quick summary
    best = results[results["n_bets"] >= 50].head(10)
    print(f"\n  Top 10 strategies (n_bets ≥ 50):")
    print(best[["prediction_method", "shrinkage", "edge", "direction", "odds_bucket",
                "min_books", "n_bets", "win_rate", "units_won", "roi", "max_drawdown"]].to_string(index=False))
    print()


if __name__ == "__main__":
    main()
