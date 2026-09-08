"""
Step 6d — Deep dive into top "both" configs on [8.5, 9.5].

Focus configs (from step6c, sorted by net_pnl):
  A: model, both, edge>=0.02, [8.5,9.5], all,        shrink=0.50 — +285u, n=1623
  B: model, both, edge>=0.02, [8.5,9.5], minus_odds,  shrink=0.50 — +234u, n=1254
  C: model, both, edge>=0.07, [8.5,9.5], all,         shrink=0.00 — +247u, n=1327

Questions:
  1. Per-season split (2025 vs 2026) — is signal consistent?
  2. Per-line split (8.5 vs 9.5) — where is the alpha?
  3. Over vs under breakdown within "both" — how many each, hit rate each?
  4. Monthly breakdown — is it spread across the season or concentrated?
  5. Per-book breakdown — is it driven by specific books?
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
    "consensus_line", "park_factor", "combined_ra_L10", "home_ra_L20",
    "combined_ra_L5", "combined_rs_L10", "home_rs_L10", "away_rs_L3",
    "combined_ra_career", "away_ra_L5", "home_ra_L10", "away_ra_L10",
]
MIN_GAMES_CAL = 30
MAX_LINE      = 13.0

CONFIGS = [
    dict(label="A: edge>=0.02, [8.5,9.5], all,   shrink=0.50", edge=0.02, lines=[8.5,9.5], odds="all",        shrink=0.50),
    dict(label="B: edge>=0.02, [8.5,9.5], minus,  shrink=0.50", edge=0.02, lines=[8.5,9.5], odds="minus_odds", shrink=0.50),
    dict(label="C: edge>=0.07, [8.5,9.5], all,   shrink=0.00", edge=0.07, lines=[8.5,9.5], odds="all",        shrink=0.00),
]


# ── Pipeline (same as step6c) ──────────────────────────────────────────────────

def load_spine() -> pd.DataFrame:
    df = pd.read_parquet(SPINE_PATH)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["combined_ra_L5"]     = df["home_ra_L5"]     + df["away_ra_L5"]
    df["combined_ra_L10"]    = df["home_ra_L10"]    + df["away_ra_L10"]
    df["combined_ra_career"] = df["home_ra_career"] + df["away_ra_career"]
    df["combined_rs_L10"]    = df["home_rs_L10"]    + df["away_rs_L10"]
    return df[df["line"] <= MAX_LINE]


def get_game_level(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["game_pk","game_date","season","total_runs"] + FEATURE_COLS
    return df[list(dict.fromkeys(cols))].drop_duplicates("game_pk").dropna(subset=["total_runs"]+FEATURE_COLS)


def fit_pipeline(train_df, test_df, spine_train) -> pd.DataFrame:
    games_tr = get_game_level(train_df)
    games_te = get_game_level(test_df)

    sc = StandardScaler()
    X_tr = sc.fit_transform(games_tr[FEATURE_COLS].values.astype(float))
    X_te = sc.transform(games_te[FEATURE_COLS].values.astype(float))
    mdl  = Ridge(alpha=50)
    mdl.fit(X_tr, games_tr["total_runs"].values.astype(float))
    y_hat_tr = mdl.predict(X_tr)
    y_hat_te = mdl.predict(X_te)

    games_tr = games_tr.copy()
    games_tr["y_hat"] = y_hat_tr
    hit_map  = spine_train.drop_duplicates("game_pk").set_index("game_pk")["hit_over"]
    games_tr["hit_over"]    = games_tr["game_pk"].map(hit_map)
    games_tr["line_bucket"] = games_tr["consensus_line"].round(1)

    g_valid = games_tr.dropna(subset=["y_hat","hit_over"])
    sc_g = StandardScaler(); X_g = sc_g.fit_transform(g_valid[["y_hat"]].values.astype(float))
    clf_g = LogisticRegression(max_iter=500, C=0.5)
    clf_g.fit(X_g, g_valid["hit_over"].values.astype(int))
    calib = {None: (sc_g, clf_g)}
    for lv in sorted(g_valid["line_bucket"].unique()):
        sub = g_valid[g_valid["line_bucket"]==lv]
        if len(sub) < MIN_GAMES_CAL: continue
        s2 = StandardScaler(); X_s = s2.fit_transform(sub[["y_hat"]].values.astype(float))
        c2 = LogisticRegression(max_iter=500, C=0.5); c2.fit(X_s, sub["hit_over"].values.astype(int))
        calib[lv] = (s2, c2)

    yhat_map = dict(zip(games_te["game_pk"], y_hat_te))
    ts = test_df.copy()
    ts["y_hat_test"] = ts["game_pk"].map(yhat_map)
    ts = ts.dropna(subset=["y_hat_test"])

    p_over = []
    for _, row in ts.iterrows():
        bkt = round(float(row["line"]), 1)
        s2, c2 = calib.get(bkt, calib[None])
        X = s2.transform(np.array([[row["y_hat_test"]]]))
        p_over.append(float(c2.predict_proba(X)[0,1]))
    ts["p_model_over"]  = p_over
    ts["p_model_under"] = 1.0 - ts["p_model_over"]
    return ts


def pnl_bet(price, hit):
    p = float(price)
    return (p/100 if p > 0 else 100/abs(p)) if int(hit)==1 else -1.0


def apply_config(ts: pd.DataFrame, edge: float, lines: list, odds: str, shrink: float) -> pd.DataFrame:
    sub = ts[ts["line"].isin(lines)].copy()
    p_u = (1-shrink)*sub["p_model_under"] + shrink*sub["novig_prob_under"]
    p_o = (1-shrink)*sub["p_model_over"]  + shrink*sub["novig_prob_over"]
    sub["edge_u"] = p_u - sub["raw_prob_under"]
    sub["edge_o"] = p_o - sub["raw_prob_over"]
    sub["edge_eff"] = sub[["edge_u","edge_o"]].max(axis=1)
    sub["bet_under"] = sub["edge_u"] >= sub["edge_o"]
    sub["hit"]   = sub.apply(lambda r: r["hit_under"]   if r["bet_under"] else r["hit_over"],  axis=1)
    sub["price"] = sub.apply(lambda r: r["under_price"] if r["bet_under"] else r["over_price"], axis=1)
    sub["side"]  = sub["bet_under"].map({True:"under", False:"over"})
    sub = sub[sub["edge_eff"] >= edge]
    if odds == "plus_odds":  sub = sub[sub["price"] > 0]
    if odds == "minus_odds": sub = sub[sub["price"] < 0]
    return sub.dropna(subset=["hit","price"]).sort_values("game_date").reset_index(drop=True)


def stats(sub: pd.DataFrame, label: str = "") -> dict:
    if len(sub) == 0:
        return {}
    sub = sub.copy()
    sub["pnl"] = sub.apply(lambda r: pnl_bet(r["price"], r["hit"]), axis=1)
    n    = len(sub)
    nw   = int(sub["hit"].sum())
    net  = float(sub["pnl"].sum())
    roi  = net/n*100
    hr   = nw/n
    cum  = sub["pnl"].cumsum()
    mdd  = float((cum - cum.cummax()).min())
    npm  = round(net/abs(mdd),2) if mdd != 0 else float("inf")
    return dict(label=label, n=n, hit_rate=round(hr,3), net=round(net,2), roi=round(roi,2), mdd=round(mdd,2), net_per_mdd=npm)


def print_table(rows: list[dict], cols: list[str]) -> None:
    if not rows: return
    widths = {c: max(len(c), max(len(str(r.get(c,""))) for r in rows)) for c in cols}
    hdr = "  ".join(f"{c:<{widths[c]}}" for c in cols)
    print("  " + hdr)
    print("  " + "  ".join("-"*widths[c] for c in cols))
    for r in rows:
        print("  " + "  ".join(f"{str(r.get(c,'—')):<{widths[c]}}" for c in cols))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading spine...")
    spine = load_spine()

    all_scored = []
    for train_seasons, test_season in [([2024], 2025), ([2024,2025], 2026)]:
        tr = spine[spine["season"].isin(train_seasons)]
        te = spine[spine["season"]==test_season]
        scored = fit_pipeline(tr, te, tr)
        scored["test_season"] = test_season
        all_scored.append(scored)
    combined = pd.concat(all_scored, ignore_index=True)
    print(f"Combined OOS: {len(combined):,} rows\n")

    cols = ["label","n","hit_rate","net","roi","mdd","net_per_mdd"]

    for cfg in CONFIGS:
        bets = apply_config(combined, cfg["edge"], cfg["lines"], cfg["odds"], cfg["shrink"])
        print(f"\n{'='*70}")
        print(f"CONFIG {cfg['label']}")
        print(f"{'='*70}")

        # Overall
        row = stats(bets, "OVERALL")
        print_table([row], cols)

        # 1. Per season
        print("\n-- Per season --")
        season_rows = [stats(bets[bets["test_season"]==yr], str(yr)) for yr in [2025,2026]]
        print_table(season_rows, cols)

        # 2. Per line
        print("\n-- Per line --")
        line_rows = [stats(bets[bets["line"]==l], f"line={l}") for l in sorted(bets["line"].unique())]
        print_table(line_rows, cols)

        # 3. Over vs under breakdown within "both"
        print("\n-- Side breakdown (over vs under bets within 'both') --")
        side_rows = [stats(bets[bets["side"]==s], s) for s in ["under","over"]]
        print_table(side_rows, cols)
        n_u = (bets["side"]=="under").sum()
        n_o = (bets["side"]=="over").sum()
        print(f"  → {n_u} under bets ({n_u/len(bets):.1%}),  {n_o} over bets ({n_o/len(bets):.1%})")

        # 4. Monthly breakdown
        print("\n-- Monthly breakdown (net units, sorted by month) --")
        bets2 = bets.copy()
        bets2["pnl"] = bets2.apply(lambda r: pnl_bet(r["price"], r["hit"]), axis=1)
        bets2["month"] = bets2["game_date"].dt.to_period("M").astype(str)
        monthly = (bets2.groupby("month")
                   .agg(n=("pnl","count"), net=("pnl","sum"), hit_rate=("hit","mean"))
                   .reset_index()
                   .sort_values("month"))
        monthly["net"] = monthly["net"].round(2)
        monthly["hit_rate"] = monthly["hit_rate"].round(3)
        print_table(monthly.to_dict("records"), ["month","n","hit_rate","net"])

        # 5. Per book (top 10 by net units)
        print("\n-- Per book (sorted by net units) --")
        bets2["book"] = bets2["bookmaker"]
        book_rows = []
        for bk, grp in bets2.groupby("book"):
            s = stats(grp, bk)
            if s: book_rows.append(s)
        book_rows = sorted(book_rows, key=lambda r: r["net"], reverse=True)
        print_table(book_rows, ["label","n","hit_rate","net","roi","mdd","net_per_mdd"])


if __name__ == "__main__":
    main()
