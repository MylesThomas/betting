"""
Step 6 / 7 — IS grid + Mock Email — MLB Batter Hits.

Step 6: Train production OLS + logistic on ALL historical settled data.
        Evaluate IS metrics (upper-bound check vs OOS from Step 5).

Step 7: Build features for today's games (2026-07-27) using:
        - Rolling player features from latest settled spine row
        - Today's market features (min/max implied probs, lines)
        Apply production models → p_model → edge → format email.

Strategy: 0.5 UNDER, edge ≥ 2pp  (primary from Step 5)

Usage:
  python src/mlb_batter_hits_modeling/scripts/20260727_step6_mock_email.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

LOCAL_SPINE  = Path.home() / "Downloads/tmp/mlb_batter_hits_spine.parquet"
LOCAL_MARKET = Path.home() / "Downloads/tmp/mlb_batter_hits_market_raw.parquet"

TODAY = "2026-07-27"
STRATEGY_LINE      = 0.5
STRATEGY_DIRECTION = "under"
STRATEGY_EDGE_MIN  = 0.02   # 2pp

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


# -----------------------------------------------------------------------
# Step 6: IS (in-sample) grid — production model trained on full data
# -----------------------------------------------------------------------

def train_production_models(settled: pd.DataFrame):
    """
    Train OLS and logistic on ALL settled data.
    Returns (ols_model, logistic_model, feature_names).
    """
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

    ols = LinearRegression()
    ols.fit(X1, y1)

    # IS prediction for logistic training data
    settled = settled.copy()
    pg["yhat_ols_full"] = np.nan
    pg.loc[sub.index, "yhat_ols_full"] = ols.predict(X1)

    settled = settled.merge(
        pg[["player_key", "game_date", "yhat_ols_full"]],
        on=["player_key", "game_date"], how="left",
    )
    settled["over_flag"] = (settled["hits_actual"] > settled["offered_line"]).astype(int)

    logit_sub = settled[settled["yhat_ols_full"].notna() & settled["over_price"].notna()].copy()
    X2 = logit_sub[["yhat_ols_full", "offered_line"]].values
    y2 = logit_sub["over_flag"].values

    logit = LogisticRegression(max_iter=500, solver="lbfgs")
    logit.fit(X2, y2)

    # IS metrics
    p_is = logit.predict_proba(X2)[:, 1]
    auc_is = roc_auc_score(y2, p_is)
    print(f"IS AUC (all lines): {auc_is:.4f}")

    # IS metrics by line
    for line in [0.5, 1.5]:
        mask = logit_sub["offered_line"] == line
        if mask.sum() > 100 and logit_sub.loc[mask, "over_flag"].nunique() == 2:
            a = roc_auc_score(y2[mask], p_is[mask])
            print(f"  IS AUC line={line}: {a:.4f}")

    return ols, logit, avail, settled, logit_sub, p_is


def is_grid(settled: pd.DataFrame, logit_sub: pd.DataFrame, p_is: np.ndarray) -> pd.DataFrame:
    """Quick IS grid for over/under at 0.5 and 1.5."""
    logit_sub = logit_sub.copy()
    logit_sub["p_model_is"] = p_is
    logit_sub["over_edge_is"]  = p_is - (1.0 / logit_sub["over_price"])
    logit_sub["under_edge_is"] = (1 - p_is) - (1.0 / logit_sub["under_price"].where(logit_sub["under_price"].notna()))
    logit_sub["under_flag"] = (logit_sub["hits_actual"] < logit_sub["offered_line"]).astype(int)

    rows = []
    for line in [0.5, 1.5]:
        for direction in ["over", "under"]:
            for thresh in [0.01, 0.02, 0.03, 0.05]:
                edge_col  = f"{direction}_edge_is"
                flag_col  = f"{direction}_flag"
                price_col = f"{direction}_price"
                sub = logit_sub[
                    (logit_sub["offered_line"] == line) &
                    logit_sub[price_col].notna() &
                    logit_sub[edge_col].notna() &
                    (logit_sub[edge_col] >= thresh)
                ]
                if len(sub) < 10:
                    continue
                n = len(sub)
                hit = sub[flag_col].mean()
                pnl = np.where(sub[flag_col] == 1, sub[price_col] - 1, -1.0)
                net = pnl.sum()
                roi = net / n * 100
                rows.append(dict(line=line, direction=direction, edge_min=thresh,
                                 n_bets=n, hit_rate=round(hit, 3),
                                 net_units=round(net, 2), roi_pct=round(roi, 2)))
    return pd.DataFrame(rows).sort_values("net_units", ascending=False)


# -----------------------------------------------------------------------
# Step 7: Build today's features + produce email
# -----------------------------------------------------------------------

def normalize_name(name: str) -> str:
    import unicodedata, re
    if not isinstance(name, str): return ""
    name = name.lower().strip()
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = re.sub(r"\s*\(\d{4}\)", "", name)
    name = re.sub(r"[''`]", "", name)
    name = re.sub(r"[-]", " ", name)
    name = re.sub(r"\.", "", name)
    name = re.sub(r"\s+(jr|sr|ii|iii|iv)$", "", name)
    name = re.sub(r"\b([a-z]) ([a-z])\b", r"\1\2", name)
    name = re.sub(r"\b([a-z]) ([a-z])\b", r"\1\2", name)
    name = re.sub(r"(?<=\s)[a-z](?=\s)", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    NAME_MAP = {
        "daniel vogelbach":   "dan vogelbach",
        "donnie walton":      "donovan walton",
        "eddy alvarez":       "francisco alvarez",
        "josh kuroda grauer": "joshua kuroda grauer",
    }
    return NAME_MAP.get(name, name)


def build_today_features(settled: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """
    For each player in today's market:
      - Get latest rolling features from spine (most recent settled game)
      - Get today's market features (min/max implied probs, lines, etc.)
    Returns one row per (player_key, bookmaker, offered_line) for today.
    """
    today_market = market[market["game_date"] == TODAY].copy()
    today_market["player_key"] = today_market["player_name"].apply(normalize_name)
    today_market["raw_implied_prob_over"]  = 1.0 / today_market["over_price"]
    today_market["raw_implied_prob_under"] = 1.0 / today_market["under_price"]

    # Min/max features at player-game level
    pg_feats = today_market.groupby("player_key").agg(
        min_line=("line", "min"),
        max_line=("line", "max"),
        min_raw_implied_prob_over=("raw_implied_prob_over", "min"),
        max_raw_implied_prob_over=("raw_implied_prob_over", "max"),
        min_raw_implied_prob_under=("raw_implied_prob_under", "min"),
        max_raw_implied_prob_under=("raw_implied_prob_under", "max"),
        consensus_line=("line", "mean"),
    ).reset_index()

    # Get latest rolling features per player from spine
    rolling_cols = [
        "player_key", "game_date",
        "hits_roll_career", "hits_roll_L1", "hits_roll_L5", "hits_roll_L10",
        "hits_roll_L20", "hits_roll_season", "ba_roll_career", "ba_roll_L5",
        "ab_roll_career", "stand", "team",
    ]
    avail_rolling = [c for c in rolling_cols if c in settled.columns]
    latest = (
        settled[avail_rolling].dropna(subset=["hits_roll_career"])
        .sort_values("game_date")
        .drop_duplicates(subset=["player_key"], keep="last")
    )

    # Merge rolling features into today's market
    today_market = today_market.merge(latest, on="player_key", how="left", suffixes=("", "_spine"))
    today_market = today_market.merge(pg_feats, on="player_key", how="left", suffixes=("", "_pg"))

    # Override market min/max cols with pg-level ones
    for col in ["min_line", "max_line", "min_raw_implied_prob_over", "max_raw_implied_prob_over",
                "min_raw_implied_prob_under", "max_raw_implied_prob_under", "consensus_line"]:
        if col + "_pg" in today_market.columns:
            today_market[col] = today_market[col + "_pg"].fillna(today_market.get(col, np.nan))

    return today_market


def apply_production_model(today_df: pd.DataFrame, ols, logit, feat_names: list) -> pd.DataFrame:
    """Predict yhat then p_model for today's rows."""
    today_df = today_df.copy()
    avail = [f for f in feat_names if f in today_df.columns]
    X = today_df[avail].values
    has_all = ~np.any(np.isnan(X), axis=1)

    today_df["yhat_ols"] = np.nan
    if has_all.any():
        today_df.loc[has_all, "yhat_ols"] = ols.predict(X[has_all])

    X2 = today_df[["yhat_ols", "line"]].rename(columns={"line": "offered_line"})
    has_yhat = today_df["yhat_ols"].notna()
    today_df["p_model"] = np.nan
    if has_yhat.any():
        today_df.loc[has_yhat, "p_model"] = logit.predict_proba(X2[has_yhat])[: , 1]

    today_df["over_edge"]  = today_df["p_model"] - today_df["raw_implied_prob_over"]
    today_df["under_edge"] = (1 - today_df["p_model"]) - today_df["raw_implied_prob_under"]
    return today_df


def format_email(today_df: pd.DataFrame) -> None:
    """Print email rows for the primary strategy: 0.5 UNDER, edge >= 2pp."""
    print(f"\n{'='*80}")
    print(f"MLB BATTER HITS — DAILY EMAIL — {TODAY}")
    print(f"Strategy: line={STRATEGY_LINE} {STRATEGY_DIRECTION.upper()}, edge >= {STRATEGY_EDGE_MIN*100:.0f}pp")
    print(f"{'='*80}")

    sub = today_df[
        (today_df["line"] == STRATEGY_LINE) &
        today_df["under_edge"].notna() &
        (today_df["under_edge"] >= STRATEGY_EDGE_MIN) &
        today_df["under_price"].notna()
    ].copy()

    if len(sub) == 0:
        print("  No qualifying bets today.")
        return

    # Compute American odds from decimal
    def dec_to_american(d):
        if pd.isna(d): return "N/A"
        if d >= 2.0:
            return f"+{int(round((d-1)*100))}"
        else:
            return f"-{int(round(100/(d-1)))}"

    sub["over_american"]  = sub["over_price"].apply(dec_to_american)
    sub["under_american"] = sub["under_price"].apply(dec_to_american)
    sub["under_edge_pp"]  = (sub["under_edge"] * 100).round(1)
    sub["p_model_pct"]    = (sub["p_model"] * 100).round(1)
    sub["raw_under_pct"]  = (sub["raw_implied_prob_under"] * 100).round(1)

    # Group by player (each player may appear at multiple books)
    players = sub.groupby("player_key").agg(
        player_name=("player_name", "first"),
        n_books=("bookmaker", "nunique"),
        books=("bookmaker", lambda x: ", ".join(sorted(x.unique()))),
        best_under_price=("under_price", "max"),
        avg_under_edge=("under_edge", "mean"),
        p_model=("p_model", "first"),
        yhat=("yhat_ols", "first"),
        hits_roll_career=("hits_roll_career", "first"),
        hits_roll_L5=("hits_roll_L5", "first"),
    ).reset_index().sort_values("avg_under_edge", ascending=False)

    print(f"\n{'Player':<30} {'Books':>3} {'Line':>5} {'Under $':>8} {'p_model':>8} {'Under Edge':>10} {'yhat':>6} {'H/G(L5)':>8}")
    print("-"*80)
    for _, row in players.iterrows():
        amer = dec_to_american(row["best_under_price"])
        print(
            f"{row['player_name']:<30} "
            f"{row['n_books']:>3} "
            f"{STRATEGY_LINE:>5.1f} "
            f"{amer:>8} "
            f"{row['p_model']*100:>7.1f}% "
            f"{row['avg_under_edge']*100:>+9.1f}pp "
            f"{row['yhat']:>6.3f} "
            f"{row['hits_roll_L5']:>8.2f}"
        )

    print(f"\nTotal qualifying bets: {len(sub):,} ({players['player_key'].nunique()} players, "
          f"{sub['bookmaker'].nunique()} books)")

    # Full detail for top 5 players
    print(f"\n--- Full detail (top 5 players by edge) ---")
    top5 = players.head(5)["player_key"].tolist()
    detail = sub[sub["player_key"].isin(top5)].sort_values(["player_key", "under_edge"], ascending=[True, False])
    cols = ["player_name", "bookmaker", "line", "over_american", "under_american",
            "p_model_pct", "raw_under_pct", "under_edge_pp", "yhat_ols", "hits_roll_career", "hits_roll_L5"]
    print(detail[[c for c in cols if c in detail.columns]].to_string(index=False))


def main() -> None:
    print("Loading spine...")
    settled_raw = pd.read_parquet(LOCAL_SPINE)
    settled = settled_raw[settled_raw["hits_actual"].notna()].copy()
    settled["raw_implied_prob_over"]  = 1.0 / settled["over_price"]
    settled["raw_implied_prob_under"] = 1.0 / settled["under_price"]
    print(f"Settled rows: {len(settled):,}")

    print("\n--- Step 6: Train production models (IS) ---")
    ols, logit, feat_names, settled_with_yhat, logit_sub, p_is = train_production_models(settled)

    print("\n--- Step 6: IS grid ---")
    is_results = is_grid(settled, logit_sub, p_is)
    print(is_results.sort_values("net_units", ascending=False).head(12).to_string(index=False))

    print("\n--- Step 7: Build today's features ---")
    market = pd.read_parquet(LOCAL_MARKET)
    today_df = build_today_features(settled, market)
    print(f"Today market rows: {len(today_df):,}")
    match_rate = today_df["hits_roll_career"].notna().mean()
    print(f"Feature match rate (has hits_roll_career): {match_rate:.1%}")

    print("\n--- Step 7: Apply production model ---")
    today_df = apply_production_model(today_df, ols, logit, feat_names)
    pred_rate = today_df["p_model"].notna().mean()
    print(f"Rows with p_model: {pred_rate:.1%}")

    format_email(today_df)

    # Sample freeman today
    ff = today_df[today_df["player_key"] == "freddie freeman"]
    if len(ff) > 0:
        print(f"\n--- Freeman today ---")
        cols = ["player_name", "bookmaker", "line", "over_price", "under_price",
                "yhat_ols", "p_model", "under_edge", "hits_roll_career", "hits_roll_L5"]
        print(ff[[c for c in cols if c in ff.columns]].sort_values("line").to_string(index=False))

    # ---------------------------------------------------------------
    # DuckDB SQL tests
    # ---------------------------------------------------------------
    import duckdb
    con = duckdb.connect()
    con.register("today", today_df)
    con.register("is_res", is_results)

    print("\n" + "="*60)
    print("STEP 6/7 — DuckDB SQL TESTS")
    print("="*60)
    tests = [
        ("T1: Today has >= 100 players with p_model",
         "SELECT COUNT(DISTINCT player_key) >= 100 AS pass FROM today WHERE p_model IS NOT NULL"),
        ("T2: IS grid 0.5 UNDER edge>=2pp has positive ROI (confirms OOS direction)",
         "SELECT roi_pct > 0 AS pass FROM is_res WHERE line = 0.5 AND direction = 'under' AND edge_min = 0.02"),
        ("T3: Today has at least 1 qualifying under bet at line 0.5 edge>=2pp",
         "SELECT COUNT(*) >= 1 AS pass FROM today WHERE line = 0.5 AND under_edge >= 0.02 AND under_price IS NOT NULL AND p_model IS NOT NULL"),
        ("T4: All today's p_model values between 0 and 1",
         "SELECT COUNT(*) = 0 AS pass FROM today WHERE p_model IS NOT NULL AND (p_model < 0 OR p_model > 1)"),
        ("T5: Today's market has >= 10 games",
         f"SELECT COUNT(DISTINCT event_id) >= 10 AS pass FROM today"),
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
        print("All Step 6/7 tests PASSED.")
    else:
        print("Some Step 6/7 tests FAILED.")


if __name__ == "__main__":
    main()
