"""
Step 5 — OOS Grid Search.
Runs for both Method A (yhat=consensus_line) and Method B (yhat=OLS).
Grid dimensions loaded from config.yaml.
Outputs: CSV with all strategy combos + financial metrics per method.
"""
import numpy as np
import pandas as pd
import yaml, itertools
from pathlib import Path

CONFIG_PATH = "src/mlb_pitcher_outs_modeling/config/config.yaml"
OOF_A_PATH  = "/Users/thomasmyles/Downloads/tmp/mlb_pitcher_outs_oof_method_a.parquet"
OOF_B_PATH  = "/Users/thomasmyles/Downloads/tmp/mlb_pitcher_outs_oof_method_b.parquet"
OUT_A_PATH  = "/Users/thomasmyles/Downloads/tmp/mlb_pitcher_outs_grid_search_method_a.csv"
OUT_B_PATH  = "/Users/thomasmyles/Downloads/tmp/mlb_pitcher_outs_grid_search_method_b.csv"

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

gs = cfg["grid_search"]
EDGE_THRESHOLDS = gs["edge_threshold"]       # [0, 0.01, 0.03, 0.05, 0.10, 0.15, 0.20]
DIRECTIONS      = gs["direction"]            # ["over", "under", "both"]
ODDS_BUCKETS    = gs["odds_bucket"]          # ["all", "plus_odds", "minus_odds"]
SHRINKAGES      = gs["shrinkage"]            # [0, 0.25, 0.50, 0.75]

def compute_novig(row):
    return row["novig_prob_under"]

def run_grid_search(oof, method_name):
    # Add edge columns at shrinkage=0 (we'll apply shrinkage inline)
    oof = oof.copy()
    oof["under_price_decimal"] = oof["under_price"]  # already decimal in spine
    oof["over_price_decimal"]  = oof["over_price"]

    # p_under_raw = model output before shrinkage
    oof["p_under_raw"] = oof["p_under"]
    oof["p_over_raw"]  = 1 - oof["p_under"]

    total_rows = len(oof)
    results = []

    for shrinkage in SHRINKAGES:
        # Apply shrinkage: pull p_model toward 0.5
        oof["p_under_s"] = oof["p_under_raw"] * (1 - shrinkage) + 0.5 * shrinkage
        oof["p_over_s"]  = 1 - oof["p_under_s"]

        # Per-book edge (uses that book's own novig)
        oof["edge_under"] = oof["p_under_s"] - oof["novig_prob_under"]
        oof["edge_over"]  = oof["p_over_s"]  - oof["novig_prob_over"]

        for edge_thresh, direction, odds_bucket in itertools.product(
            EDGE_THRESHOLDS, DIRECTIONS, ODDS_BUCKETS
        ):
            # Filter direction
            if direction == "under":
                cands = oof[oof["edge_under"] >= edge_thresh].copy()
                cands["bet_direction"] = "under"
                cands["edge"]         = cands["edge_under"]
                cands["odds"]         = cands["under_price_decimal"]
            elif direction == "over":
                cands = oof[oof["edge_over"] >= edge_thresh].copy()
                cands["bet_direction"] = "over"
                cands["edge"]         = cands["edge_over"]
                cands["odds"]         = cands["over_price_decimal"]
            else:  # both
                under_c = oof[oof["edge_under"] >= edge_thresh].copy()
                under_c["bet_direction"] = "under"
                under_c["edge"]         = under_c["edge_under"]
                under_c["odds"]         = under_c["under_price_decimal"]
                over_c  = oof[oof["edge_over"]  >= edge_thresh].copy()
                over_c["bet_direction"] = "over"
                over_c["edge"]         = over_c["edge_over"]
                over_c["odds"]         = over_c["over_price_decimal"]
                cands = pd.concat([under_c, over_c], ignore_index=True)

            # Filter odds bucket
            if odds_bucket == "plus_odds":
                cands = cands[cands["odds"] > 2.0]   # decimal > 2.0 = American +odds
            elif odds_bucket == "minus_odds":
                cands = cands[cands["odds"] <= 2.0]

            if len(cands) == 0:
                continue

            # Compute outcome (win / loss / push)
            actual = cands["outs_recorded"]
            line   = cands["line"]
            def outcome(row):
                if row["bet_direction"] == "under":
                    if actual[row.name] < line[row.name]:  return "win"
                    elif actual[row.name] == line[row.name]: return "push"
                    else: return "loss"
                else:
                    if actual[row.name] > line[row.name]:  return "win"
                    elif actual[row.name] == line[row.name]: return "push"
                    else: return "loss"

            cands = cands.reset_index(drop=True)
            actual = cands["outs_recorded"]
            line   = cands["line"]
            bdir   = cands["bet_direction"]
            outcomes = []
            for i in range(len(cands)):
                if bdir.iloc[i] == "under":
                    if actual.iloc[i] < line.iloc[i]:  outcomes.append("win")
                    elif actual.iloc[i] == line.iloc[i]: outcomes.append("push")
                    else: outcomes.append("loss")
                else:
                    if actual.iloc[i] > line.iloc[i]:  outcomes.append("win")
                    elif actual.iloc[i] == line.iloc[i]: outcomes.append("push")
                    else: outcomes.append("loss")
            cands["outcome"] = outcomes

            # P&L: flat 1 unit per bet
            cands["pnl"] = np.where(
                cands["outcome"] == "win",  cands["odds"] - 1,
                np.where(cands["outcome"] == "push", 0.0, -1.0)
            )

            n_bets   = len(cands)
            win_rate = (cands["outcome"] == "win").mean()
            push_rate= (cands["outcome"] == "push").mean()
            units    = cands["pnl"].sum()
            roi      = units / n_bets
            avg_odds = cands["odds"].mean()

            # Max drawdown
            cumulative = cands.sort_values("game_date")["pnl"].cumsum().values
            peak = np.maximum.accumulate(cumulative)
            drawdown = peak - cumulative
            max_dd   = drawdown.max()

            results.append({
                "method":          method_name,
                "shrinkage":       shrinkage,
                "edge_threshold":  edge_thresh,
                "direction":       direction,
                "odds_bucket":     odds_bucket,
                "n_bets":          n_bets,
                "pct_of_universe": round(n_bets / total_rows, 4),
                "win_rate":        round(win_rate, 4),
                "push_rate":       round(push_rate, 4),
                "units_won":       round(units, 3),
                "roi":             round(roi, 4),
                "avg_odds":        round(avg_odds, 4),
                "max_drawdown":    round(max_dd, 3),
            })

    results_df = pd.DataFrame(results).sort_values("units_won", ascending=False)
    return results_df

print("Running grid search — Method A (yhat=consensus_line)...")
oof_a = pd.read_parquet(OOF_A_PATH)
gs_a  = run_grid_search(oof_a, "A_consensus_line")
gs_a.to_csv(OUT_A_PATH, index=False)
print(f"  {len(gs_a):,} combos → {OUT_A_PATH}")

print("Running grid search — Method B (yhat=OLS model)...")
oof_b = pd.read_parquet(OOF_B_PATH)
gs_b  = run_grid_search(oof_b, "B_ols_model")
gs_b.to_csv(OUT_B_PATH, index=False)
print(f"  {len(gs_b):,} combos → {OUT_B_PATH}")

# ── Summary: top 20 strategies across both methods ───────────────────────────
combined = pd.concat([gs_a, gs_b], ignore_index=True)
top = combined[combined["n_bets"] >= 50].sort_values("units_won", ascending=False).head(30)

print("\n── Top 30 strategies (n_bets ≥ 50) by units_won ──")
print(top[["method","shrinkage","edge_threshold","direction","odds_bucket",
           "n_bets","win_rate","units_won","roi","avg_odds","max_drawdown"]].to_string(index=False))

# ── Best UNDER strategies specifically ───────────────────────────────────────
print("\n── Best UNDER strategies (n_bets ≥ 30) sorted by ROI ──")
under_top = combined[(combined["direction"]=="under") & (combined["n_bets"]>=30)].sort_values("roi", ascending=False).head(20)
print(under_top[["method","shrinkage","edge_threshold","direction","odds_bucket",
                 "n_bets","win_rate","units_won","roi","avg_odds","max_drawdown"]].to_string(index=False))

# ── Flag: ROI > 25% with > 100 bets (suspicious) ─────────────────────────────
suspicious = combined[(combined["roi"] > 0.25) & (combined["n_bets"] > 100)]
if len(suspicious):
    print(f"\n⚠️  {len(suspicious)} strategies with ROI>25% and n>100 — CHECK FOR LEAKAGE")
    print(suspicious[["method","edge_threshold","direction","odds_bucket","n_bets","roi"]].to_string(index=False))

# ── Flag: win_rate > 0.50 ─────────────────────────────────────────────────────
high_wr = combined[(combined["win_rate"] > 0.50) & (combined["n_bets"] >= 50)]
print(f"\nStrategies with win_rate > 0.50 and n≥50: {len(high_wr)}")
if len(high_wr):
    print(high_wr[["method","edge_threshold","direction","odds_bucket","n_bets","win_rate","roi"]].head(10).to_string(index=False))
