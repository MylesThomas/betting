"""
Step 7 — Mock Email Output for MLB Game Totals Pipeline.

Simulates the daily email for a given date:
  1. Trains Ridge regression + Method C calibration on all data BEFORE the target date
  2. Predicts y_hat for games on the target date
  3. Qualifies bets: line=9.5, direction=under, edge>0
  4. Prints the email in the format used by all other pipelines

Usage:
  python src/mlb_game_totals_modeling/scripts/20260711_step7_mock_email.py [YYYY-MM-DD]
  Default date: 2026-06-24 (busiest 9.5-line day in 2026 OOS window)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

LOCAL_DIR  = Path.home() / "Downloads/tmp/mlb_game_totals"
SPINE_PATH = LOCAL_DIR / "game_totals_spine.parquet"

FEATURE_COLS = [
    "consensus_line",
    "park_factor",
    "combined_ra_L10",
    "home_ra_L20",
    "combined_ra_L5",
    "combined_rs_L10",
    "home_rs_L10",
    "away_rs_L3",
    "combined_ra_career",
    "away_ra_L5",
    "home_ra_L10",
    "away_ra_L10",
]

MIN_GAMES_CAL = 30
EDGE_MIN      = 0.0
TARGET_LINE   = 9.5

BOOK_DISPLAY = {
    "draftkings":      "DraftKings",
    "fanduel":         "FanDuel",
    "betmgm":          "BetMGM",
    "pointsbetus":     "PointsBet",
    "caesars":         "Caesars",
    "betonlineag":     "BetOnline",
    "bovada":          "Bovada",
    "mybookieag":      "MyBookie",
    "betus":           "BetUS",
    "lowvig":          "LowVig",
    "windcreek":       "Wind Creek",
    "williamhill_us":  "William Hill",
    "superbook":       "SuperBook",
    "betrivers":       "BetRivers",
    "unibet_us":       "Unibet",
}


def load_spine() -> pd.DataFrame:
    df = pd.read_parquet(SPINE_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["combined_ra_L5"]     = df["home_ra_L5"]     + df["away_ra_L5"]
    df["combined_ra_L10"]    = df["home_ra_L10"]    + df["away_ra_L10"]
    df["combined_ra_career"] = df["home_ra_career"] + df["away_ra_career"]
    df["combined_rs_L10"]    = df["home_rs_L10"]    + df["away_rs_L10"]
    return df


def get_game_level(df: pd.DataFrame, yhat_col: str | None = None) -> pd.DataFrame:
    cols = ["game_pk", "game_date", "season", "total_runs"] + FEATURE_COLS
    if yhat_col:
        cols.append(yhat_col)
    cols = list(dict.fromkeys(cols))
    return df[cols].drop_duplicates("game_pk").dropna(subset=FEATURE_COLS)


def fit_ridge(games_train: pd.DataFrame, games_pred: pd.DataFrame) -> np.ndarray:
    X_tr = games_train[FEATURE_COLS].values.astype(float)
    y_tr = games_train["total_runs"].values.astype(float)
    X_te = games_pred[FEATURE_COLS].values.astype(float)

    sc   = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)

    mdl = Ridge(alpha=50)
    mdl.fit(X_tr, y_tr)
    return mdl.predict(X_te)


def fit_calibration_predict(games_train: pd.DataFrame, spine_pred: pd.DataFrame) -> pd.DataFrame:
    spine_pred = spine_pred.copy()
    spine_pred["p_model_over"]  = np.nan
    spine_pred["p_model_under"] = np.nan

    tr_valid = games_train.dropna(subset=["y_hat_train", "hit_over"])

    sc_g = StandardScaler()
    X_g  = sc_g.fit_transform(tr_valid[["y_hat_train"]].values.astype(float))
    clf_g = LogisticRegression(max_iter=500, C=0.5)
    clf_g.fit(X_g, tr_valid["hit_over"].values.astype(int))

    for line_val in spine_pred["line"].unique():
        te_mask = spine_pred["line"] == line_val
        tr_mask = games_train["line"] == line_val

        tr_sub = games_train[tr_mask].dropna(subset=["y_hat_train", "hit_over"])
        te_rows = spine_pred[te_mask]

        if len(tr_sub) < MIN_GAMES_CAL:
            X_te = sc_g.transform(te_rows[["y_hat_pred"]].values.astype(float))
            preds = clf_g.predict_proba(X_te)[:, 1]
        else:
            sc  = StandardScaler()
            X_tr = sc.fit_transform(tr_sub[["y_hat_train"]].values.astype(float))
            X_te = sc.transform(te_rows[["y_hat_pred"]].values.astype(float))
            clf  = LogisticRegression(max_iter=500, C=0.5)
            clf.fit(X_tr, tr_sub["hit_over"].values.astype(int))
            preds = clf.predict_proba(X_te)[:, 1]

        spine_pred.loc[te_mask, "p_model_over"]  = preds
        spine_pred.loc[te_mask, "p_model_under"] = 1.0 - preds

    spine_pred["edge_under"] = spine_pred["p_model_under"] - spine_pred["raw_prob_under"]
    spine_pred["edge_over"]  = spine_pred["p_model_over"]  - spine_pred["raw_prob_over"]
    return spine_pred


def american_to_decimal(american: float) -> float:
    if american > 0:
        return 1 + american / 100
    else:
        return 1 + 100 / abs(american)


def format_american(american: float) -> str:
    return f"+{int(american)}" if american > 0 else f"{int(american)}"


def print_email(bets: pd.DataFrame, target_date: str) -> None:
    """Print the daily email in pipeline format."""
    n_plays = len(bets)
    print(f"\n{'='*70}")
    print(f"  MLB GAME TOTALS — 9.5 UNDER BETS — {target_date}")
    print(f"  {n_plays} qualifying bet{'s' if n_plays != 1 else ''} (edge > 0 vs raw)")
    print(f"{'='*70}")

    if len(bets) == 0:
        print("  No qualifying bets today.")
        return

    # Group by game
    for (home, away), game_bets in bets.groupby(["home_team", "away_team"], sort=False):
        n_game = len(game_bets)
        print(f"\n  {away} @ {home}  ·  {n_game} book{'s' if n_game != 1 else ''}")
        print(f"  {'─'*60}")

        row0 = game_bets.iloc[0]
        print(f"  {'Model y_hat:':<22} {row0['y_hat_pred']:.2f} projected total runs")
        print(f"  {'Model P(under 9.5):':<22} {row0['p_model_under']:.1%}")

        # Show model inputs
        feat_vals = {
            "consensus_line":    row0.get("consensus_line", float("nan")),
            "combined_ra_L10":   row0.get("combined_ra_L10", float("nan")),
            "park_factor":       row0.get("park_factor", float("nan")),
            "home_ra_L20":       row0.get("home_ra_L20", float("nan")),
        }
        feat_str = "  |  ".join(f"{k}={v:.2f}" for k, v in feat_vals.items() if not np.isnan(v))
        print(f"  {'Inputs:':<22} {feat_str}")
        print()

        # Per-book rows
        header = f"  {'Book':<18} {'Line':>5}  {'Over':>6}  {'Under':>6}  {'Raw Ov':>7}  {'Raw Un':>7}  {'Fair Ov':>8}  {'Fair Un':>8}  {'Mdl Un':>8}  {'Edge':>7}"
        print(header)
        print(f"  {'─'*110}")

        for _, r in game_bets.sort_values("edge_under", ascending=False).iterrows():
            book_name = BOOK_DISPLAY.get(r["bookmaker"], r["bookmaker"])
            over_str  = format_american(r["over_price"])
            under_str = format_american(r["under_price"])
            raw_over  = r["raw_prob_over"]
            raw_under = r["raw_prob_under"]
            novig_over  = r["novig_prob_over"]
            novig_under = r["novig_prob_under"]
            p_model_under = r["p_model_under"]
            edge = r["edge_under"]
            marker = " ◄" if edge >= 0.02 else ""

            print(f"  {book_name:<18} {r['line']:>5.1f}  {over_str:>6}  {under_str:>6}  "
                  f"{raw_over:>7.1%}  {raw_under:>7.1%}  {novig_over:>8.1%}  {novig_under:>8.1%}  "
                  f"{p_model_under:>8.1%}  {edge:>+7.1%}{marker}")

        # Settlement (if known)
        if row0.get("total_runs") and not pd.isna(row0.get("total_runs")):
            actual = row0["total_runs"]
            result = "UNDER ✓" if actual < 9.5 else ("OVER ✗" if actual > 9.5 else "PUSH")
            print(f"\n  Actual: {actual:.0f} runs → {result}")

    print(f"\n{'='*70}\n")


def main() -> None:
    target_date = sys.argv[1] if len(sys.argv) > 1 else "2026-06-24"
    target_dt   = pd.Timestamp(target_date)

    print(f"Loading spine...")
    spine = load_spine()
    print(f"  {len(spine)} rows, {spine['game_pk'].nunique()} games")

    # Train on all games strictly before target_date
    train_df = spine[spine["game_date"] < target_dt].copy()
    pred_df  = spine[spine["game_date"] == target_dt].copy()

    print(f"  Train rows: {len(train_df)} ({train_df['game_date'].min().date()} – {train_df['game_date'].max().date()})")
    print(f"  Pred rows:  {len(pred_df)} games on {target_date}")

    if len(pred_df) == 0:
        print("No data for target date — check date or spine coverage.")
        return

    # Game-level for regression
    games_train = get_game_level(train_df).dropna(subset=["total_runs"])
    games_pred  = get_game_level(pred_df)

    print(f"  Train games: {len(games_train)}  |  Pred games: {len(games_pred)}")

    # Fit Ridge on training, predict on target date
    y_hat_pred = fit_ridge(games_train, games_pred)
    games_pred = games_pred.copy()
    games_pred["y_hat_pred"] = y_hat_pred

    # Fit Ridge on training games (in-sample) for calibration reference
    y_hat_train = fit_ridge(games_train, games_train)
    games_train = games_train.copy()
    games_train["y_hat_train"] = y_hat_train

    # Add hit_over and line to training games for calibration fitting
    hit_over_map = train_df.drop_duplicates("game_pk").set_index("game_pk")["hit_over"]
    games_train["hit_over"] = games_train["game_pk"].map(hit_over_map)
    games_train["line"]     = games_train["consensus_line"].round(1)

    # Broadcast y_hat to pred spine rows
    yhat_map = games_pred.set_index("game_pk")["y_hat_pred"]
    pred_df["y_hat_pred"] = pred_df["game_pk"].map(yhat_map)
    pred_df = pred_df.dropna(subset=["y_hat_pred"])

    # Add game-level features to pred_df for display (only FEATURE_COLS not already present)
    game_feat_cols = ["game_pk"] + FEATURE_COLS
    game_feats = games_pred[game_feat_cols]
    # Drop any FEATURE_COLS already in pred_df to avoid collision on merge
    cols_to_drop = [c for c in FEATURE_COLS if c in pred_df.columns]
    pred_df = pred_df.drop(columns=cols_to_drop, errors="ignore")
    pred_df = pred_df.merge(game_feats, on="game_pk", how="left")

    # Calibrate
    pred_df = fit_calibration_predict(games_train, pred_df)

    print(f"  Calibrated rows: {pred_df['p_model_under'].notna().sum()}")

    # Qualify: line=9.5, under, edge>0
    bets = pred_df[
        (pred_df["line"] == TARGET_LINE) &
        (pred_df["edge_under"] > EDGE_MIN)
    ].copy()

    print(f"\n  9.5 UNDER qualifying bets (edge > 0): {len(bets)}")

    if len(bets) > 0:
        print(f"  Edge range: [{bets['edge_under'].min():.1%}, {bets['edge_under'].max():.1%}]")
        print(f"  Books represented: {sorted(bets['bookmaker'].unique())}")
        print(f"  Games represented: {bets[['home_team','away_team']].drop_duplicates().shape[0]}")

    print_email(bets, target_date)

    # Also show all 9.5-line bets for context (even those with edge <= 0)
    all_95 = pred_df[pred_df["line"] == TARGET_LINE].copy()
    print(f"  (Total 9.5-line rows on {target_date}: {len(all_95)} across {all_95['game_pk'].nunique()} games, {all_95['bookmaker'].nunique()} books)")
    print(f"\n  Qualifying fraction: {len(bets)}/{len(all_95)} = {len(bets)/len(all_95):.0%}")

    # Edge distribution for all 9.5 rows
    print(f"\n  Edge distribution (all 9.5 unders):")
    print(f"    mean edge: {all_95['edge_under'].mean():.3f}")
    print(f"    edge > 0:  {(all_95['edge_under'] > 0).sum()} / {len(all_95)}")
    print(f"    edge > 2pp: {(all_95['edge_under'] > 0.02).sum()} / {len(all_95)}")


if __name__ == "__main__":
    main()
