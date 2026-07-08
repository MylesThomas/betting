"""
Step 6 — In-Sample Grid Search.
Method A only (consensus_line) since it won OOS.
Train residuals on ALL data → score ALL data → same grid.
Compare IS vs OOS for the candidate strategies.
"""
import numpy as np
import pandas as pd
import yaml, itertools

CONFIG_PATH = "src/mlb_pitcher_outs_modeling/config/config.yaml"
SPINE_PATH  = "/Users/thomasmyles/Downloads/tmp/mlb_pitcher_outs_spine.parquet"
IS_OOF_PATH = "/Users/thomasmyles/Downloads/tmp/mlb_pitcher_outs_is_scores.parquet"
OUT_PATH    = "/Users/thomasmyles/Downloads/tmp/mlb_pitcher_outs_grid_search_is.csv"

RNG_SEED    = 42
N_BOOTSTRAP = 10_000
rng = np.random.default_rng(RNG_SEED)

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)
gs = cfg["grid_search"]
EDGE_THRESHOLDS = gs["edge_threshold"]
DIRECTIONS      = gs["direction"]
ODDS_BUCKETS    = gs["odds_bucket"]
SHRINKAGES      = gs["shrinkage"]

# ── Load full spine ───────────────────────────────────────────────────────────
spine = pd.read_parquet(SPINE_PATH)
spine["season"]   = spine["game_date"].str[:4].astype(int)
spine["is_under"] = (spine["outs_recorded"] < spine["line"]).astype(float)

pg = spine.drop_duplicates(subset=["player_key","game_date"])
residuals_all = (pg["outs_recorded"] - pg["consensus_line"]).dropna().values
print(f"IS residuals: n={len(residuals_all):,}  mean={residuals_all.mean():.3f}  std={residuals_all.std():.3f}")

# Bootstrap P(under) — same per-player-game approach as Step 4
print("Computing IS bootstrap probabilities...")
pg_unique = spine.drop_duplicates(subset=["player_key","game_date"]).reset_index(drop=True)
yhat_vals = pg_unique["consensus_line"].values
samples   = rng.choice(residuals_all, size=(len(pg_unique), N_BOOTSTRAP), replace=True)
simulated = yhat_vals[:, None] + samples  # (n_pg, N_BOOT)

pg_index = {(pk,gd): i for i,(pk,gd) in enumerate(zip(pg_unique["player_key"], pg_unique["game_date"]))}
pg_lines = spine.drop_duplicates(subset=["player_key","game_date","line"])[
    ["player_key","game_date","line"]].copy()
pg_lines["p_under"] = [
    float((simulated[pg_index[(pk,gd)]] < line).mean())
    for pk,gd,line in zip(pg_lines["player_key"],pg_lines["game_date"],pg_lines["line"])
]

is_df = spine.merge(pg_lines[["player_key","game_date","line","p_under"]],
                    on=["player_key","game_date","line"], how="left")
is_df["yhat"] = is_df["consensus_line"]
print(f"IS scored rows: {len(is_df):,}")

is_df.to_parquet(IS_OOF_PATH, index=False)

# IS Brier
brier_is = ((is_df["p_under"] - is_df["is_under"]) ** 2).mean()
print(f"IS Brier score: {brier_is:.5f}  (OOS was 0.24921)")

# ── Grid search ───────────────────────────────────────────────────────────────
total_rows = len(is_df)
results = []

for shrinkage in SHRINKAGES:
    is_df["p_under_s"] = is_df["p_under"] * (1-shrinkage) + 0.5*shrinkage
    is_df["p_over_s"]  = 1 - is_df["p_under_s"]
    is_df["edge_under"] = is_df["p_under_s"] - is_df["novig_prob_under"]
    is_df["edge_over"]  = is_df["p_over_s"]  - is_df["novig_prob_over"]

    for edge_thresh, direction, odds_bucket in itertools.product(EDGE_THRESHOLDS, DIRECTIONS, ODDS_BUCKETS):
        if direction == "under":
            cands = is_df[is_df["edge_under"] >= edge_thresh].copy()
            cands["bet_direction"] = "under"
            cands["edge"] = cands["edge_under"]
            cands["odds"] = cands["under_price"]
        elif direction == "over":
            cands = is_df[is_df["edge_over"] >= edge_thresh].copy()
            cands["bet_direction"] = "over"
            cands["edge"] = cands["edge_over"]
            cands["odds"] = cands["over_price"]
        else:
            u = is_df[is_df["edge_under"] >= edge_thresh].copy()
            u["bet_direction"]="under"; u["edge"]=u["edge_under"]; u["odds"]=u["under_price"]
            o = is_df[is_df["edge_over"]  >= edge_thresh].copy()
            o["bet_direction"]="over";  o["edge"]=o["edge_over"];  o["odds"]=o["over_price"]
            cands = pd.concat([u,o], ignore_index=True)

        if odds_bucket == "plus_odds":  cands = cands[cands["odds"] > 2.0]
        elif odds_bucket == "minus_odds": cands = cands[cands["odds"] <= 2.0]
        if len(cands) == 0: continue

        cands = cands.reset_index(drop=True)
        actual = cands["outs_recorded"].values
        line   = cands["line"].values
        bdir   = cands["bet_direction"].values
        outcomes = []
        for i in range(len(cands)):
            if bdir[i] == "under":
                outcomes.append("win" if actual[i]<line[i] else ("push" if actual[i]==line[i] else "loss"))
            else:
                outcomes.append("win" if actual[i]>line[i] else ("push" if actual[i]==line[i] else "loss"))
        cands["outcome"] = outcomes
        cands["pnl"] = np.where(cands["outcome"]=="win", cands["odds"]-1,
                        np.where(cands["outcome"]=="push", 0.0, -1.0))

        n_bets   = len(cands)
        units    = cands["pnl"].sum()
        roi      = units / n_bets
        win_rate = (cands["outcome"]=="win").mean()
        push_rate= (cands["outcome"]=="push").mean()
        avg_odds = cands["odds"].mean()
        cumulative = cands.sort_values("game_date")["pnl"].cumsum().values
        peak = np.maximum.accumulate(cumulative)
        max_dd = (peak - cumulative).max()

        results.append({"shrinkage":shrinkage,"edge_threshold":edge_thresh,
                        "direction":direction,"odds_bucket":odds_bucket,
                        "n_bets":n_bets,"pct_of_universe":round(n_bets/total_rows,4),
                        "win_rate":round(win_rate,4),"push_rate":round(push_rate,4),
                        "units_won":round(units,3),"roi":round(roi,4),
                        "avg_odds":round(avg_odds,4),"max_drawdown":round(max_dd,3)})

is_gs = pd.DataFrame(results).sort_values("units_won", ascending=False)
is_gs.to_csv(OUT_PATH, index=False)
print(f"Saved {len(is_gs)} IS combos → {OUT_PATH}")

# ── Compare IS vs OOS for top candidates ─────────────────────────────────────
oos_a = pd.read_csv("/Users/thomasmyles/Downloads/tmp/mlb_pitcher_outs_grid_search_method_a.csv")
CANDIDATES = [
    {"shrinkage":0.00, "edge_threshold":0.15, "direction":"under", "odds_bucket":"minus_odds"},
    {"shrinkage":0.25, "edge_threshold":0.10, "direction":"under", "odds_bucket":"minus_odds"},
    {"shrinkage":0.00, "edge_threshold":0.10, "direction":"under", "odds_bucket":"minus_odds"},
    {"shrinkage":0.50, "edge_threshold":0.05, "direction":"under", "odds_bucket":"minus_odds"},
]

print(f"\n{'Strategy':<50}  {'IS_n':>6}  {'IS_roi':>7}  {'OOS_n':>6}  {'OOS_roi':>8}  {'IS/OOS':>7}")
print("-"*100)
for c in CANDIDATES:
    is_row  = is_gs[(is_gs["shrinkage"]==c["shrinkage"]) & (is_gs["edge_threshold"]==c["edge_threshold"]) &
                    (is_gs["direction"]==c["direction"]) & (is_gs["odds_bucket"]==c["odds_bucket"])]
    oos_row = oos_a[(oos_a["shrinkage"]==c["shrinkage"]) & (oos_a["edge_threshold"]==c["edge_threshold"]) &
                    (oos_a["direction"]==c["direction"]) & (oos_a["odds_bucket"]==c["odds_bucket"])]
    if len(is_row)==0 or len(oos_row)==0: continue
    ir, or_ = is_row.iloc[0], oos_row.iloc[0]
    ratio = ir["roi"]/or_["roi"] if or_["roi"]!=0 else float("nan")
    label = f"UNDER, minus_odds, edge≥{c['edge_threshold']*100:.0f}pp, shrink={c['shrinkage']}"
    print(f"{label:<50}  {ir['n_bets']:>6,}  {ir['roi']:>+7.2%}  {or_['n_bets']:>6,}  {or_['roi']:>+8.2%}  {ratio:>7.2f}x")
print("-"*100)
print("IS/OOS ratio > 5x = overfitting warning")
