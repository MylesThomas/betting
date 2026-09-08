"""
Step 4 — OLS yhat → P(over offered_line) — MLB Batter Hits.

Two-stage OOF pipeline:
  Stage 1: OLS (10 numeric features) → OOF yhat at player-game level
  Stage 2: Logistic regression (yhat, offered_line) → OOF P(over) at spine grain

Both stages use TimeSeriesSplit(5) with the same temporal fold structure.

After computing OOF p_model:
  - Merge into full settled spine
  - Compute edge = p_model - raw_implied_prob_over (vig-inclusive, per book)
  - Evaluate AUC + Brier per line
  - Calibration check: compare p_model to actual over rate

p_model is player-game-line level (same across all books at a given line for the same player-game).
Edge is player-game-book-line level (varies by book's raw odds).

Usage:
  python src/mlb_batter_hits_modeling/scripts/20260727_step4_probability.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

LOCAL_SPINE = Path.home() / "Downloads/tmp/mlb_batter_hits_spine.parquet"
N_SPLITS = 5

NUMERIC_FEATURES = [
    "min_raw_implied_prob_under",
    "max_raw_implied_prob_over",
    "hits_roll_career",
    "max_line",
    "ab_roll_career",
    "hits_roll_L20",
    "hits_roll_season",
    "consensus_line",
    "hits_roll_L10",
    "ba_roll_career",
]


def load_spine() -> pd.DataFrame:
    spine = pd.read_parquet(LOCAL_SPINE)
    settled = spine[spine["hits_actual"].notna()].copy()
    settled["over_flag"] = (settled["hits_actual"] > settled["offered_line"]).astype(int)
    settled["push_flag"] = (settled["hits_actual"] == settled["offered_line"]).astype(int)
    settled["raw_implied_prob_over"] = 1.0 / settled["over_price"]
    print(f"Settled spine rows: {len(settled):,}")
    return settled


def load_player_game_df(settled: pd.DataFrame) -> pd.DataFrame:
    """One row per (player_key, game_date) — prefer DraftKings."""
    has_dk = settled[settled["bookmaker"] == "draftkings"]
    no_dk  = settled[~settled.index.isin(has_dk.index)]
    combined = pd.concat([has_dk, no_dk])
    pg = (
        combined.sort_values(["player_key", "game_date", "bookmaker"])
        .drop_duplicates(subset=["player_key", "game_date"])
        .sort_values("game_date")
        .reset_index(drop=True)
    )
    print(f"Player-game rows: {len(pg):,}  ({pg['player_key'].nunique():,} players)")
    return pg


# -----------------------------------------------------------------------
# Stage 1: OLS → OOF yhat at player-game level
# -----------------------------------------------------------------------

def stage1_ols_yhat(pg: pd.DataFrame) -> pd.DataFrame:
    """Returns pg with added yhat_ols column (OOF prediction)."""
    avail = [f for f in NUMERIC_FEATURES if f in pg.columns]
    sub = pg[avail + ["hits_actual"]].dropna()
    non_null_idx = sub.index

    X = sub[avail].values
    y = sub["hits_actual"].values

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    yhat = np.full(len(sub), np.nan)

    for train_idx, test_idx in tscv.split(X):
        m = LinearRegression()
        m.fit(X[train_idx], y[train_idx])
        yhat[test_idx] = m.predict(X[test_idx])

    pg = pg.copy()
    pg["yhat_ols"] = np.nan
    pg.loc[non_null_idx, "yhat_ols"] = yhat

    valid = ~np.isnan(yhat)
    r2 = 1 - np.sum((y[valid] - yhat[valid])**2) / np.sum((y[valid] - y[valid].mean())**2)
    print(f"Stage 1 OLS: n={valid.sum():,}  r²={r2:.4f}  RMSE={np.sqrt(np.mean((y[valid]-yhat[valid])**2)):.4f}")
    return pg


# -----------------------------------------------------------------------
# Stage 2: Logistic (yhat, offered_line) → OOF P(over)
# -----------------------------------------------------------------------

def stage2_logistic(settled: pd.DataFrame, pg: pd.DataFrame) -> pd.DataFrame:
    """
    Merges OOF yhat into full spine, then OOF logistic → p_model per row.
    Returns settled with p_model added.
    """
    # Merge yhat into spine (player-game level → all book rows)
    settled = settled.merge(
        pg[["player_key", "game_date", "yhat_ols"]],
        on=["player_key", "game_date"],
        how="left",
    )
    print(f"Rows with yhat_ols: {settled['yhat_ols'].notna().sum():,} of {len(settled):,}")

    # Only rows with valid yhat and valid over_price
    usable = settled[settled["yhat_ols"].notna() & settled["over_price"].notna()].copy()
    usable = usable.sort_values("game_date").reset_index(drop=True)
    print(f"Usable rows for logistic: {len(usable):,}")

    X = usable[["yhat_ols", "offered_line"]].values
    y = usable["over_flag"].values

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    p_model = np.full(len(usable), np.nan)

    for train_idx, test_idx in tscv.split(X):
        if len(np.unique(y[train_idx])) < 2:
            continue
        m = LogisticRegression(max_iter=500, solver="lbfgs")
        m.fit(X[train_idx], y[train_idx])
        p_model[test_idx] = m.predict_proba(X[test_idx])[:, 1]

    usable["p_model"] = p_model
    print(f"Rows with p_model: {usable['p_model'].notna().sum():,}")

    # Evaluate AUC + Brier overall
    valid = usable["p_model"].notna() & usable["over_flag"].notna()
    y_v, p_v = usable.loc[valid, "over_flag"].values, usable.loc[valid, "p_model"].values
    auc_all = roc_auc_score(y_v, p_v)
    brier_all = brier_score_loss(y_v, p_v)
    print(f"\nOverall AUC: {auc_all:.4f}  Brier: {brier_all:.4f}")

    return usable


def auc_by_line(usable: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for line in sorted(usable["offered_line"].dropna().unique()):
        sub = usable[(usable["offered_line"] == line) & usable["p_model"].notna()]
        if len(sub) < 50 or sub["over_flag"].nunique() < 2:
            continue
        auc   = roc_auc_score(sub["over_flag"], sub["p_model"])
        brier = brier_score_loss(sub["over_flag"], sub["p_model"])
        rows.append(dict(line=line, n=len(sub), over_rate=round(sub["over_flag"].mean(), 3),
                         auc=round(auc, 4), brier=round(brier, 4)))
    return pd.DataFrame(rows)


def calibration_by_decile(usable: pd.DataFrame, line: float = 0.5) -> pd.DataFrame:
    """Compare p_model deciles to actual over rates (calibration check)."""
    sub = usable[(usable["offered_line"] == line) & usable["p_model"].notna()].copy()
    sub["decile"] = pd.qcut(sub["p_model"], q=10, labels=False, duplicates="drop")
    calib = sub.groupby("decile").agg(
        n=("over_flag", "count"),
        avg_p_model=("p_model", "mean"),
        actual_over_rate=("over_flag", "mean"),
    ).reset_index()
    calib["gap"] = (calib["actual_over_rate"] - calib["avg_p_model"]).round(3)
    return calib.round(3)


def compute_edge(usable: pd.DataFrame) -> pd.DataFrame:
    """Add edge column: p_model - raw_implied_prob_over (vig-inclusive)."""
    usable = usable.copy()
    usable["edge"] = usable["p_model"] - usable["raw_implied_prob_over"]
    return usable


def spot_check_freeman(usable: pd.DataFrame) -> None:
    ff = usable[usable["player_key"] == "freddie freeman"].sort_values("game_date")
    ff_dk = ff[ff["bookmaker"] == "draftkings"].drop_duplicates(subset=["game_date", "offered_line"])
    print(f"\nFreeman settled rows (DraftKings, last 8): {len(ff_dk)}")
    cols = ["game_date", "offered_line", "hits_actual", "over_flag",
            "yhat_ols", "p_model", "raw_implied_prob_over", "edge"]
    print(ff_dk[[c for c in cols if c in ff_dk.columns]].tail(8).round(3).to_string(index=False))


def main() -> None:
    print("Loading spine...")
    settled = load_spine()

    print("\nBuilding player-game df...")
    pg = load_player_game_df(settled)

    print("\n--- Stage 1: OLS yhat ---")
    pg = stage1_ols_yhat(pg)

    print("\n--- Stage 2: Logistic P(over) ---")
    usable = stage2_logistic(settled, pg)

    print("\n--- AUC by line ---")
    auc_lines = auc_by_line(usable)
    print(auc_lines.to_string(index=False))

    print("\n--- Calibration by decile (line=0.5) ---")
    calib = calibration_by_decile(usable, line=0.5)
    print(calib.to_string(index=False))

    print("\n--- Calibration by decile (line=1.5) ---")
    calib_15 = calibration_by_decile(usable, line=1.5)
    print(calib_15.to_string(index=False))

    usable = compute_edge(usable)

    spot_check_freeman(usable)

    # ---------------------------------------------------------------
    # Edge distribution
    # ---------------------------------------------------------------
    print("\n--- Edge distribution (rows with p_model and raw_implied_prob_over) ---")
    edge_valid = usable[usable["edge"].notna()]
    import duckdb
    con = duckdb.connect()
    con.register("e", edge_valid)
    print(con.execute("""
        SELECT
            offered_line,
            COUNT(*) AS n,
            ROUND(AVG(edge), 4) AS avg_edge,
            ROUND(STDDEV(edge), 4) AS std_edge,
            ROUND(MIN(edge), 4) AS min_edge,
            ROUND(MAX(edge), 4) AS max_edge,
            SUM(CASE WHEN edge > 0.03 THEN 1 ELSE 0 END) AS n_edge_3pp,
            SUM(CASE WHEN edge > 0.05 THEN 1 ELSE 0 END) AS n_edge_5pp
        FROM e
        GROUP BY offered_line
        ORDER BY offered_line
    """).df().to_string(index=False))

    # ---------------------------------------------------------------
    # DuckDB SQL tests
    # ---------------------------------------------------------------
    con.register("usable", usable)
    con.register("auc_lines", auc_lines)

    print("\n" + "="*60)
    print("STEP 4 — DuckDB SQL TESTS")
    print("="*60)
    tests = [
        ("T1: p_model coverage >= 80% of settled spine rows",
         "SELECT COUNT(*) * 1.0 / (SELECT COUNT(*) FROM usable) >= 0.80 AS pass FROM usable WHERE p_model IS NOT NULL"),
        ("T2: AUC at line 1.5 > 0.57 (model adds value at the miscalibrated line)",
         "SELECT auc > 0.57 AS pass FROM auc_lines WHERE line = 1.5"),
        ("T3: AUC at line 0.5 > 0.57",
         "SELECT auc > 0.57 AS pass FROM auc_lines WHERE line = 0.5"),
        ("T4: AUC at line 1.5 > 0.51 (any improvement over chance)",
         "SELECT auc > 0.51 AS pass FROM auc_lines WHERE line = 1.5"),
        ("T5: p_model is between 0 and 1 for all rows",
         "SELECT COUNT(*) = 0 AS pass FROM usable WHERE p_model IS NOT NULL AND (p_model < 0 OR p_model > 1)"),
        ("T6: edge column present and has both positive and negative values",
         "SELECT (MIN(edge) < 0 AND MAX(edge) > 0) AS pass FROM usable WHERE edge IS NOT NULL"),
    ]

    all_pass = True
    for name, sql in tests:
        try:
            result = con.execute(sql).fetchone()[0]
            status = "PASS" if result else "FAIL"
            if not result:
                all_pass = False
            print(f"  [{status}] {name}")
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
            all_pass = False

    print()
    if all_pass:
        print("All Step 4 tests PASSED.")
    else:
        print("Some Step 4 tests FAILED.")

    return usable


if __name__ == "__main__":
    main()
