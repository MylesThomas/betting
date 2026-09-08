"""Avg line and avg odds breakdown for both strategies."""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

SPINE_PATH = Path.home() / "Downloads/tmp/mlb_game_totals/game_totals_spine.parquet"
FEATURE_COLS = [
    "consensus_line","park_factor","combined_ra_L10","home_ra_L20",
    "combined_ra_L5","combined_rs_L10","home_rs_L10","away_rs_L3",
    "combined_ra_career","away_ra_L5","home_ra_L10","away_ra_L10",
]

def load_spine():
    df = pd.read_parquet(SPINE_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["combined_ra_L5"]     = df["home_ra_L5"]     + df["away_ra_L5"]
    df["combined_ra_L10"]    = df["home_ra_L10"]    + df["away_ra_L10"]
    df["combined_ra_career"] = df["home_ra_career"] + df["away_ra_career"]
    df["combined_rs_L10"]    = df["home_rs_L10"]    + df["away_rs_L10"]
    return df[df["line"] <= 13.0]

def fit_pipeline(train_df, test_df, spine_train):
    games_tr = train_df[list(dict.fromkeys(["game_pk","game_date","season","total_runs"]+FEATURE_COLS))].drop_duplicates("game_pk").dropna(subset=["total_runs"]+FEATURE_COLS)
    games_te = test_df[list(dict.fromkeys(["game_pk","game_date","season","total_runs"]+FEATURE_COLS))].drop_duplicates("game_pk").dropna(subset=["total_runs"]+FEATURE_COLS)
    sc = StandardScaler()
    X_tr = sc.fit_transform(games_tr[FEATURE_COLS].values.astype(float))
    X_te = sc.transform(games_te[FEATURE_COLS].values.astype(float))
    mdl = Ridge(alpha=50); mdl.fit(X_tr, games_tr["total_runs"].values.astype(float))
    y_tr = mdl.predict(X_tr); y_te = mdl.predict(X_te)
    games_tr = games_tr.copy(); games_tr["y_hat"] = y_tr
    hit_map = spine_train.drop_duplicates("game_pk").set_index("game_pk")["hit_over"]
    games_tr["hit_over"] = games_tr["game_pk"].map(hit_map)
    games_tr["line_bucket"] = games_tr["consensus_line"].round(1)
    g = games_tr.dropna(subset=["y_hat","hit_over"])
    sc_g = StandardScaler(); clf_g = LogisticRegression(max_iter=500, C=0.5)
    clf_g.fit(sc_g.fit_transform(g[["y_hat"]].values.astype(float)), g["hit_over"].values.astype(int))
    calib = {None: (sc_g, clf_g)}
    for lv in sorted(g["line_bucket"].unique()):
        sub = g[g["line_bucket"]==lv]
        if len(sub) < 30: continue
        s2 = StandardScaler(); c2 = LogisticRegression(max_iter=500, C=0.5)
        c2.fit(s2.fit_transform(sub[["y_hat"]].values.astype(float)), sub["hit_over"].values.astype(int))
        calib[lv] = (s2, c2)
    ymap = dict(zip(games_te["game_pk"], y_te))
    ts = test_df.copy(); ts["y_hat_test"] = ts["game_pk"].map(ymap)
    ts = ts.dropna(subset=["y_hat_test"])
    po = []
    for _, row in ts.iterrows():
        bkt = round(float(row["line"]), 1); s2, c2 = calib.get(bkt, calib[None])
        po.append(float(c2.predict_proba(s2.transform(np.array([[row["y_hat_test"]]])))[0, 1]))
    ts["p_model_over"] = po; ts["p_model_under"] = 1 - ts["p_model_over"]
    return ts

def american_to_implied(odds):
    o = float(odds)
    return 100 / (o + 100) if o > 0 else abs(o) / (abs(o) + 100)

def implied_to_american(p):
    p = float(p)
    if p >= 0.5:
        return f"-{round(p / (1 - p) * 100)}"
    else:
        return f"+{round((1 - p) / p * 100)}"

def fmt_odds(prices: pd.Series) -> str:
    avg_implied = prices.apply(american_to_implied).mean()
    return f"{implied_to_american(avg_implied)} ({avg_implied:.1%} implied)"

def summarise(bets, label):
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  n bets      : {len(bets):,}")
    print(f"  avg line    : {bets['line'].mean():.3f}")
    print(f"  line dist   : { {k: int(v) for k, v in bets['line'].value_counts().sort_index().items()} }")

    under_bets = bets[bets["side"] == "under"]
    over_bets  = bets[bets["side"] == "over"]

    print(f"\n  Side split  : {len(under_bets):,} under ({len(under_bets)/len(bets):.1%})  |  {len(over_bets):,} over ({len(over_bets)/len(bets):.1%})")

    if len(under_bets) > 0:
        hit_u = under_bets["hit"].mean()
        print(f"  Under bets  : avg odds {fmt_odds(under_bets['price'])}  |  hit rate {hit_u:.1%}")
        print(f"    line dist : { {k: int(v) for k, v in under_bets['line'].value_counts().sort_index().items()} }")

    if len(over_bets) > 0:
        hit_o = over_bets["hit"].mean()
        print(f"  Over bets   : avg odds {fmt_odds(over_bets['price'])}  |  hit rate {hit_o:.1%}")
        print(f"    line dist : { {k: int(v) for k, v in over_bets['line'].value_counts().sort_index().items()} }")

    print(f"\n  Overall hit : {bets['hit'].mean():.1%}")

def main():
    spine = load_spine()
    all_scored = []
    for train_seasons, test_season in [([2024], 2025), ([2024, 2025], 2026)]:
        tr = spine[spine["season"].isin(train_seasons)]
        te = spine[spine["season"] == test_season]
        scored = fit_pipeline(tr, te, tr)
        scored["test_season"] = test_season
        all_scored.append(scored)
    ts = pd.concat(all_scored, ignore_index=True)

    # ── Strategy A: blind 9.5 under ───────────────────────────────────────────
    bm = ts[ts["line"] == 9.5].copy()
    bm["side"]  = "under"
    bm["price"] = bm["under_price"]
    bm["hit"]   = bm["hit_under"]
    bm = bm.dropna(subset=["hit", "price"])
    summarise(bm, "STRATEGY A — Blind 9.5 Under")

    # ── Strategy B: model both, edge>=0.02, [8.5,9.5], shrink=0.50 ───────────
    sub = ts[ts["line"].isin([8.5, 9.5])].copy()
    shrink = 0.50
    p_u = (1 - shrink) * sub["p_model_under"] + shrink * sub["novig_prob_under"]
    p_o = (1 - shrink) * sub["p_model_over"]  + shrink * sub["novig_prob_over"]
    sub["edge_u"] = p_u - sub["raw_prob_under"]
    sub["edge_o"] = p_o - sub["raw_prob_over"]
    sub["edge_eff"] = sub[["edge_u", "edge_o"]].max(axis=1)
    sub["bet_under"] = sub["edge_u"] >= sub["edge_o"]
    sub["side"]  = sub["bet_under"].map({True: "under", False: "over"})
    sub["price"] = sub.apply(lambda r: r["under_price"] if r["bet_under"] else r["over_price"], axis=1)
    sub["hit"]   = sub.apply(lambda r: r["hit_under"]   if r["bet_under"] else r["hit_over"],  axis=1)
    sub = sub[sub["edge_eff"] >= 0.02].dropna(subset=["hit", "price"])
    summarise(sub, "STRATEGY B — Model Both / [8.5,9.5] / edge>=0.02 / shrink=0.50")

if __name__ == "__main__":
    main()
