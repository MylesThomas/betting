"""
Step 6 — Full 4-table IS characterization for C2b (production strategy).
C2b: edge≥10pp, shrink=0.25, UNDER, minus_odds, line≤17.5
OOS: 446 bets, +16.63% ROI, max_dd=24.9 (no DD > units)
"""
import numpy as np
import pandas as pd

IS_PATH  = "/Users/thomasmyles/Downloads/tmp/mlb_pitcher_outs_is_scores.parquet"
OOF_A    = "/Users/thomasmyles/Downloads/tmp/mlb_pitcher_outs_oof_method_a.parquet"

is_df = pd.read_parquet(IS_PATH)
oof_a = pd.read_parquet(OOF_A)

EDGE_THRESH = 0.10
SHRINKAGE   = 0.25
LINE_MAX    = 17.5

def apply_strategy(df, edge_thresh=EDGE_THRESH, shrinkage=SHRINKAGE, line_max=LINE_MAX):
    d = df.copy()
    d["p_under_s"] = d["p_under"] * (1-shrinkage) + 0.5*shrinkage
    d["edge_under"] = d["p_under_s"] - d["novig_prob_under"]
    cands = d[(d["edge_under"] >= edge_thresh) & (d["under_price"] <= 2.0)].copy()
    if line_max is not None:
        cands = cands[cands["line"] <= line_max]
    cands = cands.reset_index(drop=True)
    a = cands["outs_recorded"].values
    l = cands["line"].values
    cands["outcome"] = ["win" if a[i]<l[i] else ("push" if a[i]==l[i] else "loss") for i in range(len(cands))]
    cands["pnl"] = np.where(cands["outcome"]=="win", cands["under_price"]-1,
                   np.where(cands["outcome"]=="push", 0.0, -1.0))
    return cands

def summary_row(cands, universe_n):
    n = len(cands)
    if n == 0: return {}
    units = cands["pnl"].sum()
    cum = cands.sort_values("game_date")["pnl"].cumsum().values
    max_dd = (np.maximum.accumulate(cum) - cum).max()
    return {
        "n_bets": n, "pct_of_universe": round(n/universe_n, 4),
        "win_rate": round((cands["outcome"]=="win").mean(), 4),
        "push_rate": round((cands["outcome"]=="push").mean(), 4),
        "units_won": round(units, 3),
        "roi": round(units/n, 4),
        "max_drawdown": round(max_dd, 3),
        "avg_implied_prob": round((1/cands["under_price"]).mean(), 4),
        "avg_odds": round(cands["under_price"].mean(), 4),
    }

print("=" * 70)
print("C2b: edge≥10pp, shrink=0.25, UNDER, minus_odds, line≤17.5")
print("=" * 70)

# ── IS characterization ────────────────────────────────────────────────────
is_cands = apply_strategy(is_df)
sr = summary_row(is_cands, len(is_df))

print("\n[Table 1 — Summary (IS)]")
print(f"  n_bets={sr['n_bets']:,}  pct={sr['pct_of_universe']:.2%}  "
      f"win_rate={sr['win_rate']:.3f}  push_rate={sr['push_rate']:.3f}  "
      f"units={sr['units_won']:+.1f}  roi={sr['roi']:+.3f}  "
      f"max_dd={sr['max_drawdown']:.1f}  avg_odds={sr['avg_odds']:.4f}  "
      f"avg_imp={sr['avg_implied_prob']:.4f}")
print(f"  breakeven_wr={sr['avg_implied_prob']:.3f}  actual_wr={sr['win_rate']:.3f}  "
      f"edge_vs_breakeven={sr['win_rate']-sr['avg_implied_prob']:+.3f}")

print("\n[Table 2 — Odds sub-bucket breakdown (IS)]")
is_cands["odds_sub"] = pd.cut(is_cands["under_price"],
    bins=[1.0, 1.40, 1.60, 1.80, 2.0],
    labels=["≤1.40 (≥-250)", "1.40-1.60 (-150 to -250)", "1.60-1.80 (-125 to -150)", "1.80-2.0 (-100 to -125)"])
for bucket, g in is_cands.groupby("odds_sub", observed=True):
    if len(g) == 0: continue
    n = len(g); u = g["pnl"].sum()
    wr = (g["outcome"]=="win").mean()
    bev = (1/g["under_price"]).mean()
    print(f"  {str(bucket):<30s}: n={n:4,}  win_rate={wr:.3f}  breakeven={bev:.3f}  "
          f"units={u:+.1f}  roi={u/n:+.3f}  avg_odds={g['under_price'].mean():.4f}")

print("\n[Table 3 — By bookmaker (IS, sorted by ROI)]")
bk = is_cands.groupby("bookmaker").apply(lambda g: pd.Series({
    "n": len(g), "win_rate": (g["outcome"]=="win").mean(),
    "units": g["pnl"].sum(), "roi": g["pnl"].sum()/len(g),
    "avg_odds": g["under_price"].mean(),
    "breakeven_wr": (1/g["under_price"]).mean(),
}), include_groups=False).sort_values("roi", ascending=False)
print(f"  {'Book':<25s}  {'n':>5}  {'wr':>6}  {'bev':>6}  {'units':>8}  {'roi':>8}  {'avg_odds':>9}")
print(f"  {'-'*25}  {'-----':>5}  {'------':>6}  {'------':>6}  {'--------':>8}  {'--------':>8}  {'---------':>9}")
for book, row in bk.iterrows():
    flag = " ⚠️" if row["roi"] < 0 else ""
    print(f"  {book:<25s}  {row['n']:5.0f}  {row['win_rate']:6.3f}  {row['breakeven_wr']:6.3f}  "
          f"{row['units']:+8.1f}  {row['roi']:+8.3f}  {row['avg_odds']:9.4f}{flag}")

print("\n[Table 4 — Line bucket breakdown (IS)]")
is_cands["line_bucket"] = pd.cut(is_cands["line"], bins=[0,15.5,16.5,17.5,99],
                                  labels=["≤15.5","15.5-16.5","16.5-17.5","≥17.5"])
print(f"  {'Bucket':<15s}  {'n':>5}  {'wr':>6}  {'push':>6}  {'units':>8}  {'roi':>8}  {'avg_odds':>9}")
print(f"  {'-'*15}  {'-----':>5}  {'------':>6}  {'------':>6}  {'--------':>8}  {'--------':>8}  {'---------':>9}")
for lb, g in is_cands.groupby("line_bucket", observed=False):
    if len(g) == 0: continue
    wr = (g["outcome"]=="win").mean()
    pr = (g["outcome"]=="push").mean()
    u  = g["pnl"].sum()
    print(f"  {str(lb):<15s}  {len(g):5,}  {wr:6.3f}  {pr:6.3f}  {u:+8.1f}  {u/len(g):+8.3f}  "
          f"{g['under_price'].mean():9.4f}")

print("\n[Table 5 — By season (IS)]")
is_cands["season"] = is_cands["game_date"].str[:4]
print(f"  {'Season':<8}  {'n':>5}  {'wr':>6}  {'units':>8}  {'roi':>8}")
print(f"  {'--------':<8}  {'-----':>5}  {'------':>6}  {'--------':>8}  {'--------':>8}")
for season, g in is_cands.groupby("season"):
    u = g["pnl"].sum()
    print(f"  {season:<8}  {len(g):5,}  {(g['outcome']=='win').mean():6.3f}  {u:+8.1f}  {u/len(g):+8.3f}")

# ── OOS validation ─────────────────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("OOS VALIDATION (C2b)")
print("=" * 70)

oos_cands = apply_strategy(oof_a)
oos_sr = summary_row(oos_cands, len(oof_a))
print(f"\n  n_bets={oos_sr['n_bets']:,}  win_rate={oos_sr['win_rate']:.3f}  "
      f"units={oos_sr['units_won']:+.1f}  roi={oos_sr['roi']:+.3f}  max_dd={oos_sr['max_drawdown']:.1f}")

print("\nOOS by season:")
oos_cands["season"] = oos_cands["game_date"].str[:4]
for season, g in oos_cands.groupby("season"):
    u = g["pnl"].sum()
    print(f"  {season}: n={len(g):3,}  wr={(g['outcome']=='win').mean():.3f}  roi={u/len(g):+.3f}  units={u:+.1f}")

print("\nOOS by line bucket:")
oos_cands["line_bucket"] = pd.cut(oos_cands["line"], bins=[0,15.5,16.5,17.5,99],
                                   labels=["≤15.5","15.5-16.5","16.5-17.5","≥17.5"])
for lb, g in oos_cands.groupby("line_bucket", observed=False):
    if len(g)==0: continue
    wr=(g["outcome"]=="win").mean(); u=g["pnl"].sum()
    print(f"  {lb}: n={len(g):3,}  wr={wr:.3f}  roi={u/len(g):+.3f}  avg_odds={g['under_price'].mean():.4f}")

print("\nOOS by bookmaker:")
bk_oos = oos_cands.groupby("bookmaker").apply(lambda g: pd.Series({
    "n": len(g), "win_rate": (g["outcome"]=="win").mean(),
    "units": g["pnl"].sum(), "roi": g["pnl"].sum()/len(g),
}), include_groups=False).sort_values("roi", ascending=False)
for book, row in bk_oos.iterrows():
    flag = " ⚠️" if row["roi"] < 0 else ""
    print(f"  {book:<25s}: n={row['n']:3.0f}  wr={row['win_rate']:.3f}  "
          f"units={row['units']:+.1f}  roi={row['roi']:+.3f}{flag}")
