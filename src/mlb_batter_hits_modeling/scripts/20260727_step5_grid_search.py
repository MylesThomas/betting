"""
Step 5 — Grid Search over direction, line, and edge threshold — MLB Batter Hits.

Computes p_model (same 2-stage OOF pipeline as Step 4), then sweeps:
  direction: OVER, UNDER
  line:      0.5, 1.5  (2.5+ have no positive edge from Step 4)
  edge_min:  0, 1, 2, 3, 5, 7, 10 pp

For each cell: n_bets, hit_rate, net_units, ROI%, max_drawdown, net/MDD.

Edge definitions:
  over_edge  = p_model - raw_implied_prob_over   (1/over_price, vig-inclusive)
  under_edge = (1-p_model) - raw_implied_prob_under  (1/under_price, vig-inclusive)

Bets require valid price in the direction being bet.
P&L: win → +(decimal_price - 1), lose → -1.

Usage:
  python src/mlb_batter_hits_modeling/scripts/20260727_step5_grid_search.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

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
    settled["over_flag"]  = (settled["hits_actual"] > settled["offered_line"]).astype(int)
    settled["under_flag"] = (settled["hits_actual"] < settled["offered_line"]).astype(int)
    settled["raw_implied_prob_over"]  = 1.0 / settled["over_price"]
    settled["raw_implied_prob_under"] = 1.0 / settled["under_price"]
    return settled


def build_p_model(settled: pd.DataFrame) -> pd.DataFrame:
    """2-stage OOF: OLS → yhat, then Logistic(yhat, line) → p_model."""
    # Stage 1: player-game OOF yhat
    has_dk = settled[settled["bookmaker"] == "draftkings"]
    no_dk  = settled[~settled.index.isin(has_dk.index)]
    pg = (
        pd.concat([has_dk, no_dk])
        .sort_values(["player_key", "game_date", "bookmaker"])
        .drop_duplicates(subset=["player_key", "game_date"])
        .sort_values("game_date")
        .reset_index(drop=True)
    )

    avail = [f for f in NUMERIC_FEATURES if f in pg.columns]
    sub = pg[avail + ["hits_actual"]].dropna()
    X1 = sub[avail].values
    y1 = sub["hits_actual"].values
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    yhat = np.full(len(sub), np.nan)
    for tr, te in tscv.split(X1):
        m = LinearRegression()
        m.fit(X1[tr], y1[tr])
        yhat[te] = m.predict(X1[te])
    pg["yhat_ols"] = np.nan
    pg.loc[sub.index, "yhat_ols"] = yhat

    # Stage 2: merge yhat into full spine, OOF logistic
    settled = settled.merge(
        pg[["player_key", "game_date", "yhat_ols"]],
        on=["player_key", "game_date"],
        how="left",
    )
    usable = settled[settled["yhat_ols"].notna() & settled["over_price"].notna()].copy()
    usable = usable.sort_values("game_date").reset_index(drop=True)

    X2 = usable[["yhat_ols", "offered_line"]].values
    y2 = usable["over_flag"].values
    tscv2 = TimeSeriesSplit(n_splits=N_SPLITS)
    p_model = np.full(len(usable), np.nan)
    for tr, te in tscv2.split(X2):
        if len(np.unique(y2[tr])) < 2:
            continue
        m = LogisticRegression(max_iter=500, solver="lbfgs")
        m.fit(X2[tr], y2[tr])
        p_model[te] = m.predict_proba(X2[te])[:, 1]

    usable["p_model"] = p_model
    usable["over_edge"]  = usable["p_model"] - usable["raw_implied_prob_over"]
    usable["under_edge"] = (1 - usable["p_model"]) - usable["raw_implied_prob_under"]
    print(f"Usable rows with p_model: {usable['p_model'].notna().sum():,}")
    return usable


def compute_pnl(df: pd.DataFrame, direction: str) -> pd.Series:
    """Vectorised P&L: +( price - 1 ) on win, -1 on loss."""
    if direction == "over":
        win   = df["over_flag"].astype(float)
        price = df["over_price"]
    else:
        win   = df["under_flag"].astype(float)
        price = df["under_price"]
    return np.where(win == 1, price - 1, -1.0)


def max_drawdown(cumulative_pnl: np.ndarray) -> float:
    """Peak-to-trough drawdown on cumulative P&L series."""
    peak = np.maximum.accumulate(cumulative_pnl)
    dd   = peak - cumulative_pnl
    return float(dd.max())


def grid_search(usable: pd.DataFrame) -> pd.DataFrame:
    lines     = [0.5, 1.5]
    directions = ["over", "under"]
    thresholds = [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10]

    rows = []
    for line in lines:
        sub = usable[usable["offered_line"] == line].copy()
        for direction in directions:
            edge_col  = f"{direction}_edge"
            price_col = f"{direction}_price"
            flag_col  = f"{direction}_flag"

            sub_dir = sub[sub[price_col].notna() & sub[edge_col].notna()].copy()
            sub_dir["pnl"] = compute_pnl(sub_dir, direction)

            for thresh in thresholds:
                bets = sub_dir[sub_dir[edge_col] >= thresh].sort_values("game_date")
                if len(bets) == 0:
                    continue
                n_bets   = len(bets)
                hit_rate = bets[flag_col].mean()
                net      = bets["pnl"].sum()
                roi      = net / n_bets
                cum_pnl  = bets["pnl"].cumsum().values
                mdd      = max_drawdown(cum_pnl)
                net_mdd  = net / mdd if mdd > 0 else np.inf
                rows.append(dict(
                    line=line,
                    direction=direction,
                    edge_min=thresh,
                    n_bets=n_bets,
                    hit_rate=round(hit_rate, 4),
                    net_units=round(net, 2),
                    roi_pct=round(roi * 100, 2),
                    max_dd=round(mdd, 2),
                    net_mdd=round(net_mdd, 2),
                ))

    return pd.DataFrame(rows).sort_values("net_units", ascending=False)


def main() -> None:
    print("Loading spine...")
    settled = load_spine()
    print(f"Settled rows: {len(settled):,}")

    print("\nBuilding p_model (2-stage OOF)...")
    usable = build_p_model(settled)

    print("\n--- Edge summary by line/direction ---")
    con = duckdb.connect()
    con.register("u", usable)
    print(con.execute("""
        SELECT
            offered_line,
            'over'  AS direction,
            COUNT(*) AS n,
            ROUND(AVG(over_edge), 4) AS avg_edge,
            SUM(CASE WHEN over_edge >= 0.03 THEN 1 ELSE 0 END) AS n_3pp
        FROM u WHERE over_price IS NOT NULL AND over_edge IS NOT NULL
        GROUP BY offered_line
        UNION ALL
        SELECT
            offered_line,
            'under' AS direction,
            COUNT(*) AS n,
            ROUND(AVG(under_edge), 4) AS avg_edge,
            SUM(CASE WHEN under_edge >= 0.03 THEN 1 ELSE 0 END) AS n_3pp
        FROM u WHERE under_price IS NOT NULL AND under_edge IS NOT NULL
        GROUP BY offered_line
        ORDER BY offered_line, direction
    """).df().to_string(index=False))

    print("\n--- Running grid search ---")
    results = grid_search(usable)

    print("\n--- Grid search results (sorted by net_units) ---")
    print(results.round(3).to_string(index=False))

    print("\n--- Top 10 by net_units ---")
    print(results.head(10).round(3).to_string(index=False))

    print("\n--- Best ROI% (min 50 bets) ---")
    print(results[results["n_bets"] >= 50].sort_values("roi_pct", ascending=False).head(10).round(3).to_string(index=False))

    # ---------------------------------------------------------------
    # Per-season performance for best strategy
    # ---------------------------------------------------------------
    best = results[results["n_bets"] >= 50].iloc[0]
    print(f"\n--- Per-season: best strategy (line={best['line']}, dir={best['direction']}, edge>={best['edge_min']*100:.0f}pp) ---")
    sub = usable[
        (usable["offered_line"] == best["line"]) &
        (usable[f"{best['direction']}_edge"] >= best["edge_min"]) &
        (usable[f"{best['direction']}_price"].notna())
    ].copy()
    sub["pnl"] = compute_pnl(sub, best["direction"])
    sub["season"] = pd.to_datetime(sub["game_date"]).dt.year
    by_season = sub.groupby("season").agg(
        n_bets=("pnl", "count"),
        hit_rate=(f"{best['direction']}_flag", "mean"),
        net_units=("pnl", "sum"),
    ).reset_index()
    by_season["roi_pct"] = (by_season["net_units"] / by_season["n_bets"] * 100).round(2)
    by_season["hit_rate"] = by_season["hit_rate"].round(3)
    by_season["net_units"] = by_season["net_units"].round(2)
    print(by_season.to_string(index=False))

    # ---------------------------------------------------------------
    # DuckDB SQL tests
    # ---------------------------------------------------------------
    con.register("results", results)
    con.register("best_sub", sub)

    print("\n" + "="*60)
    print("STEP 5 — DuckDB SQL TESTS")
    print("="*60)
    tests = [
        ("T1: At least one strategy with net_units > 0 and n_bets >= 100",
         "SELECT COUNT(*) > 0 AS pass FROM results WHERE net_units > 0 AND n_bets >= 100"),
        ("T2: Best strategy ROI > 2%",
         "SELECT roi_pct > 2 AS pass FROM results WHERE n_bets >= 50 ORDER BY net_units DESC LIMIT 1"),
        ("T3: At least one under strategy at line 1.5 with positive net_units (market overprices overs)",
         "SELECT COUNT(*) > 0 AS pass FROM results WHERE line = 1.5 AND direction = 'under' AND net_units > 0 AND n_bets >= 50"),
        ("T4: Best strategy n_bets >= 50",
         "SELECT n_bets >= 50 AS pass FROM results ORDER BY net_units DESC LIMIT 1"),
        ("T5: Best strategy net/MDD >= 1.0 (profitable relative to drawdown)",
         "SELECT net_mdd >= 1.0 AS pass FROM results WHERE n_bets >= 50 ORDER BY net_units DESC LIMIT 1"),
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
        print("All Step 5 tests PASSED.")
    else:
        print("Some Step 5 tests FAILED.")


if __name__ == "__main__":
    main()
