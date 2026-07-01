"""
Step 6 — In-Sample Evaluation (Grid Search)

Uses the full-data trained model (best_model.pkl) scored on all training rows.
Residuals = carries - full_model_carries; P(over) via same stratified KDE CDF
approach as Step 4 (but residuals are IS — model has seen every row).

Grid dimensions are identical to Step 5:
  edge_threshold : [0, 0.01, 0.03, 0.05, 0.10, 0.15, 0.20]
  direction      : [over, under, both]
  line_filter    : [all, low (<6.5), high (>=6.5)]
  position_filter: [all, RB, QB]

Key question: does the best OOS strategy (QB high-line under, edge>=0.03)
show positive in-sample ROI? IS/OOS ratio should be <5x — if much higher,
the OOS result is likely noise.

Outputs:
  ~/Downloads/tmp/rush_attempts/step6_is_preds.parquet  (IS predictions + P/edge)
  ~/Downloads/tmp/rush_attempts/step6_grid.csv          (full grid)
"""

from __future__ import annotations

import itertools
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

REPO_ROOT  = Path(__file__).resolve().parents[3]
TRAIN_PATH = Path.home() / "Downloads" / "tmp" / "rush_attempts" / "training.parquet"
MODEL_PATH = REPO_ROOT / "models" / "nfl_rush_attempts" / "best_model.pkl"
OUT_DIR    = Path.home() / "Downloads" / "tmp" / "rush_attempts"

PRED_BINS   = [0, 5, 10, 15, 20, np.inf]
PRED_LABELS = ["lt5", "5to9", "10to14", "15to19", "20plus"]

CARRY_BINS   = [0, 5, 10, 15, 20, np.inf]
CARRY_LABELS = ["lt5", "5to9", "10to14", "15to19", "20plus"]

EDGE_THRESHOLDS = [0.00, 0.01, 0.03, 0.05, 0.10, 0.15, 0.20]
DIRECTIONS      = ["over", "under", "both"]
LINE_FILTERS    = ["all", "low", "high"]
POS_FILTERS     = ["all", "RB", "QB"]


# ── Feature engineering (mirror step3_train.py exactly) ──────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    carry_tier = pd.cut(df["carry_rate_L8"], bins=CARRY_BINS,
                        labels=CARRY_LABELS, right=False)
    for label in CARRY_LABELS[1:]:
        df[f"carry_bucket_L8_{label}"] = (carry_tier == label).astype(int)
    line_tier = pd.cut(df["consensus_point"], bins=CARRY_BINS,
                       labels=CARRY_LABELS, right=False)
    for label in CARRY_LABELS[1:]:
        df[f"line_bucket_{label}"] = (line_tier == label).astype(int)
    df["line_deviation"]  = df["consensus_point"] - df["carry_rate_L8"]
    df["carry_trend"]     = df["carry_rate_L3"] - df["carry_rate_L5"]
    df["bell_cow_flag"]   = (df["carry_rate_Lcareer"] >= 12).astype(int)
    df["line_x_pos_RB"]   = df["consensus_point"] * df["pos_RB"]
    df["line_x_bell_cow"] = df["consensus_point"] * df["bell_cow_flag"]
    return df


# ── Residual CDF (same as step4_calibration.py) ───────────────────────────────

def build_residual_cdfs(residuals: np.ndarray, pred_bucket: pd.Series) -> dict:
    cdfs = {}
    print("\nIn-sample residual distribution by predicted carry bucket:")
    print(f"  {'bucket':<10} {'n':>6}  {'mean_resid':>11}  {'std_resid':>10}  {'p10':>6}  {'p90':>6}")

    global_r = residuals[~np.isnan(residuals)]

    for label in PRED_LABELS:
        mask = pred_bucket == label
        r = residuals[mask]
        r = r[~np.isnan(r)]
        if len(r) < 20:
            r = global_r

        mean_r, std_r = r.mean(), r.std()
        p10, p90 = np.percentile(r, 10), np.percentile(r, 90)
        print(f"  {label:<10} {len(r):>6}  {mean_r:>+11.3f}  {std_r:>10.3f}  {p10:>6.1f}  {p90:>6.1f}")

        kde = stats.gaussian_kde(r, bw_method="scott")
        x_min = r.min() - 3 * std_r
        x_max = r.max() + 3 * std_r
        xs = np.linspace(x_min, x_max, 500)
        pdf_vals = kde(xs)
        cdf_vals = np.cumsum(pdf_vals) * (xs[1] - xs[0])
        cdf_vals = np.clip(cdf_vals / cdf_vals[-1], 0, 1)
        cdf_fn = interp1d(xs, cdf_vals, kind="linear",
                          bounds_error=False, fill_value=(0.0, 1.0))
        cdfs[label] = cdf_fn
    return cdfs


# ── Payout / strategy helpers (mirror step5_grid_search.py) ──────────────────

def american_to_decimal(american: float) -> float:
    if american > 0:
        return 1 + american / 100
    else:
        return 1 + 100 / abs(american)


def max_drawdown(pnl_series: np.ndarray) -> float:
    cum  = np.cumsum(pnl_series)
    peak = cum[0]
    max_dd = 0.0
    for v in cum:
        if v > peak:
            peak = v
        dd = peak - v
        if dd > max_dd:
            max_dd = dd
    return round(max_dd, 4)


def evaluate_strategy(df: pd.DataFrame, direction: str,
                      edge_threshold: float) -> dict | None:
    edge      = df["edge"].values
    is_over   = df["is_over"].values
    over_price  = df["book_over_price"].values
    under_price = df["book_under_price"].values

    if direction == "over":
        mask    = edge >= edge_threshold
        correct = is_over[mask] == 1
        prices  = over_price[mask]
    elif direction == "under":
        mask    = (-edge) >= edge_threshold
        correct = is_over[mask] == 0
        prices  = under_price[mask]
    else:
        over_mask  = edge >= edge_threshold
        under_mask = (-edge) >= edge_threshold
        mask       = over_mask | under_mask
        bet_over   = np.where(over_mask, True, False)
        correct = np.where(bet_over, is_over == 1, is_over == 0)[mask]
        prices  = np.where(bet_over, over_price, under_price)[mask]

    n_bets = int(mask.sum())
    if n_bets == 0:
        return None

    decimals = np.vectorize(american_to_decimal)(prices)
    profit   = np.where(correct, decimals - 1, -1.0)

    units_won = round(float(profit.sum()), 4)
    win_rate  = round(float(correct.mean()), 4)
    roi       = round(units_won / n_bets, 4)
    max_dd    = max_drawdown(profit)

    avg_dec = float(decimals.mean())
    if avg_dec >= 2.0:
        avg_odds_american = round((avg_dec - 1) * 100, 1)
    else:
        avg_odds_american = round(-100 / (avg_dec - 1), 1)

    return {
        "n_bets":       n_bets,
        "win_rate":     win_rate,
        "push_rate":    0.0,
        "units_won":    units_won,
        "roi":          roi,
        "avg_odds":     avg_odds_american,
        "max_drawdown": max_dd,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    # ── Load data ────────────────────────────────────────────────────────────
    train = pd.read_parquet(TRAIN_PATH)
    print(f"Training rows (per-book): {len(train):,}")

    # Fill NaNs (mirror step3_train.py)
    rolling_carry  = [c for c in train.columns if c.startswith("carry_rate_L")]
    rolling_yards  = [c for c in train.columns if c.startswith("rush_yards_L")]
    opp_def        = [c for c in train.columns if c.startswith("opp_carry_allowed_")]
    over_rate      = [c for c in train.columns if c.startswith("over_rate_L")]
    for col in rolling_carry + rolling_yards + opp_def:
        train[col] = train[col].fillna(0)
    for col in over_rate:
        train[col] = train[col].fillna(0.5)

    # Dedup to one row per player-game (same logic as step3_train.py)
    df_pg = (
        train.sort_values("n_books", ascending=False)
             .drop_duplicates(subset=["nfl_game_id", "player_name_norm"])
             .reset_index(drop=True)
    )
    df_pg = engineer_features(df_pg)
    print(f"Player-game rows: {len(df_pg):,}")

    # ── Load model + score all rows ───────────────────────────────────────────
    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)

    model      = artifact["model"]
    scaler     = artifact["scaler"]
    features   = artifact["features"]
    model_type = artifact["model_type"]
    print(f"\nLoaded model: {model_type}  features: {features}")

    X = df_pg[features].values
    if scaler is not None:
        X = scaler.transform(X)
    df_pg["is_carries"] = model.predict(X)  # in-sample (IS) predictions
    print(f"IS predictions — mean: {df_pg['is_carries'].mean():.2f}  "
          f"std: {df_pg['is_carries'].std():.2f}  "
          f"range: [{df_pg['is_carries'].min():.2f}, {df_pg['is_carries'].max():.2f}]")

    # ── Residuals + stratified CDF ────────────────────────────────────────────
    df_pg["residual"]   = df_pg["carries"] - df_pg["is_carries"]
    df_pg["pred_bucket"] = pd.cut(df_pg["is_carries"], bins=PRED_BINS,
                                   labels=PRED_LABELS, right=False).astype(str)

    cdfs = build_residual_cdfs(df_pg["residual"].values, df_pg["pred_bucket"])

    # ── Join IS predictions to per-book rows ──────────────────────────────────
    slim = df_pg[["nfl_game_id", "player_name_norm", "is_carries", "pred_bucket"]].copy()
    df = train.merge(slim, on=["nfl_game_id", "player_name_norm"], how="inner")
    df = df.reset_index(drop=True)
    print(f"\nPer-book rows after join: {len(df):,}")

    # ── P(over) via IS residual CDF ───────────────────────────────────────────
    df["shortfall"] = df["book_line"] - df["is_carries"]
    p_model_vals    = np.empty(len(df))
    for bucket in PRED_LABELS:
        mask = df["pred_bucket"] == bucket
        if mask.sum() == 0:
            continue
        shortfalls = df.loc[mask, "shortfall"].values
        p_model_vals[mask] = np.array([
            float(1.0 - cdfs[bucket](s)) for s in shortfalls
        ])
    df["p_model"]  = p_model_vals
    df["p_market"] = df["book_over_prob"]
    df["edge"]     = df["p_model"] - df["p_market"]

    print(f"\nIS edge stats — mean: {df['edge'].mean():.4f}  "
          f"std: {df['edge'].std():.4f}  "
          f"range: [{df['edge'].min():.4f}, {df['edge'].max():.4f}]")

    # ── Save IS predictions ───────────────────────────────────────────────────
    out_cols = [
        "nfl_game_id", "player_name_norm", "player_display_name",
        "bookmaker", "season", "week", "is_playoff",
        "position", "carries", "is_over",
        "book_line", "book_over_price", "book_under_price",
        "is_carries", "shortfall", "pred_bucket",
        "p_model", "p_market", "edge",
    ]
    out = df[out_cols].copy()
    out.to_parquet(OUT_DIR / "step6_is_preds.parquet", index=False)
    print(f"Saved {len(out):,} IS prediction rows")

    # ── Grid search (identical dimensions to Step 5) ──────────────────────────
    df = df.sort_values(["season", "week"]).reset_index(drop=True)
    results = []

    for edge_thresh, direction, line_filter, pos_filter in itertools.product(
        EDGE_THRESHOLDS, DIRECTIONS, LINE_FILTERS, POS_FILTERS
    ):
        sub = df.copy()

        if pos_filter == "RB":
            sub = sub[sub["position"] == "RB"]
        elif pos_filter == "QB":
            sub = sub[sub["position"] == "QB"]

        if line_filter == "low":
            sub = sub[sub["book_line"] < 6.5]
        elif line_filter == "high":
            sub = sub[sub["book_line"] >= 6.5]

        if len(sub) == 0:
            continue

        metrics = evaluate_strategy(sub, direction, edge_thresh)
        if metrics is None:
            continue

        row = {
            "edge_threshold":  edge_thresh,
            "direction":       direction,
            "line_filter":     line_filter,
            "position_filter": pos_filter,
            **metrics,
        }
        results.append(row)

    grid = pd.DataFrame(results)
    grid = grid.sort_values(["units_won", "n_bets"], ascending=[False, False])
    grid.to_csv(OUT_DIR / "step6_grid.csv", index=False)
    print(f"\nGrid search: {len(grid):,} combos evaluated")

    # ── Load OOS grid for IS/OOS comparison ───────────────────────────────────
    oos_path = OUT_DIR / "step5_grid.csv"
    oos_grid = pd.read_csv(oos_path)

    # Key strategy: QB high-line under, edge >= 0.03
    key_filter = dict(direction="under", line_filter="high",
                      position_filter="QB", edge_threshold=0.03)

    def get_row(g, **kw):
        q = g.copy()
        for k, v in kw.items():
            q = q[q[k] == v]
        return q.iloc[0] if len(q) > 0 else None

    is_key  = get_row(grid, **key_filter)
    oos_key = get_row(oos_grid, **key_filter)

    print("\n=== KEY STRATEGY: QB high-line under, edge>=0.03 ===")
    print(f"  OOS — n_bets={oos_key['n_bets']}  ROI={oos_key['roi']:.4f}  "
          f"units_won={oos_key['units_won']:.2f}")
    print(f"  IS  — n_bets={is_key['n_bets']}  ROI={is_key['roi']:.4f}  "
          f"units_won={is_key['units_won']:.2f}")
    if oos_key["roi"] != 0:
        ratio = abs(is_key["roi"] / oos_key["roi"])
        print(f"  IS/OOS ROI ratio: {ratio:.2f}x  "
              f"({'OK' if ratio < 5 else 'WARNING: >5x — possible overfit'})")

    # ── Top 20 IS ─────────────────────────────────────────────────────────────
    print("\n=== TOP 20 IS by units won (≥50 bets) ===")
    top = grid[grid["n_bets"] >= 50].head(20)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print(top[["edge_threshold","direction","line_filter","position_filter",
               "n_bets","win_rate","units_won","roi","avg_odds","max_drawdown"]].to_string(index=False))

    # ── IS vs OOS comparison table for QB high-line under ────────────────────
    print("\n=== IS vs OOS — QB high under, by edge threshold ===")
    rows = []
    for thresh in EDGE_THRESHOLDS:
        is_r  = get_row(grid,     direction="under", line_filter="high",
                        position_filter="QB", edge_threshold=thresh)
        oos_r = get_row(oos_grid, direction="under", line_filter="high",
                        position_filter="QB", edge_threshold=thresh)
        if is_r is not None and oos_r is not None:
            rows.append({
                "edge_threshold": thresh,
                "IS_n_bets":   is_r["n_bets"],   "OOS_n_bets":  oos_r["n_bets"],
                "IS_ROI":      round(is_r["roi"], 4), "OOS_ROI": round(oos_r["roi"], 4),
                "IS_units":    round(is_r["units_won"], 2),
                "OOS_units":   round(oos_r["units_won"], 2),
            })
    cmp = pd.DataFrame(rows)
    print(cmp.to_string(index=False))

    print(f"\nSaved IS grid to {OUT_DIR / 'step6_grid.csv'}")
    print("=== Step 6 complete ===")


if __name__ == "__main__":
    run()
