"""
Step 4 — Probability conversion comparison.
Method A: yhat = consensus_line (no model), residuals centered on the market line.
Method B: yhat from LinearRegression model.
Both: 10k bootstrap samples → P(under line).
Compare Brier scores and calibration. Pick the better one.
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings("ignore")

RNG_SEED    = 42
N_BOOTSTRAP = 10_000
SPINE_PATH  = "/Users/thomasmyles/Downloads/tmp/mlb_pitcher_outs_spine.parquet"
OOF_A_PATH  = "/Users/thomasmyles/Downloads/tmp/mlb_pitcher_outs_oof_method_a.parquet"
OOF_B_PATH  = "/Users/thomasmyles/Downloads/tmp/mlb_pitcher_outs_oof_method_b.parquet"

rng = np.random.default_rng(RNG_SEED)

# ── Load spine ────────────────────────────────────────────────────────────────
df_full = pd.read_parquet(SPINE_PATH)
df_full["season"] = df_full["game_date"].str[:4].astype(int)
df_full["is_under"] = (df_full["outs_recorded"] < df_full["line"]).astype(float)

FOLDS = [
    {"train_seasons": [2024],       "test_season": 2025},
    {"train_seasons": [2024, 2025], "test_season": 2026},
]

# Deduplicated player-game view for residual fitting
df_pg = df_full.drop_duplicates(subset=["player_key", "game_date"]).copy()

def bootstrap_p_under_grain(df_grain, residuals, yhat_col, n_boot=N_BOOTSTRAP, rng=rng):
    """Compute P(under line) at (player_key, game_date, line) grain.
    Draws ONE set of bootstrap samples per player-game, then evaluates ALL of that
    game's line values against the same samples. This guarantees monotonicity by
    construction: P(under L1) <= P(under L2) whenever L1 <= L2, because the same
    simulated outcomes are used for all lines. Also ensures all book rows at the same
    player-game-line get identical p_under.
    """
    # One row per player-game: get the yhat
    pg = df_grain[["player_key", "game_date", yhat_col]].drop_duplicates(
        subset=["player_key", "game_date"]
    ).reset_index(drop=True)
    yhat_vals = pg[yhat_col].values  # (n_pg,)

    # Draw n_boot samples per player-game
    samples   = rng.choice(residuals, size=(len(pg), n_boot), replace=True)  # (n_pg, n_boot)
    simulated = yhat_vals[:, None] + samples  # (n_pg, n_boot)

    # Build a lookup: (player_key, game_date) → row index in pg
    pg_index = {(pk, gd): i for i, (pk, gd) in enumerate(zip(pg["player_key"], pg["game_date"]))}

    # Unique (player_key, game_date, line) combos
    pg_lines = df_grain.drop_duplicates(subset=["player_key", "game_date", "line"])[
        ["player_key", "game_date", "line"]
    ].copy()
    pg_lines["p_under"] = [
        float((simulated[pg_index[(pk, gd)]] < line).mean())
        for pk, gd, line in zip(pg_lines["player_key"], pg_lines["game_date"], pg_lines["line"])
    ]

    result = df_grain.merge(
        pg_lines[["player_key", "game_date", "line", "p_under"]],
        on=["player_key", "game_date", "line"], how="left"
    )
    return result["p_under"].values

# ── Method A: yhat = consensus_line ──────────────────────────────────────────
print("=" * 60)
print("METHOD A: yhat = consensus_line (no model)")
print("=" * 60)

oof_a_rows = []
for fold in FOLDS:
    train_pg = df_pg[df_pg["season"].isin(fold["train_seasons"])]
    test_full = df_full[df_full["season"] == fold["test_season"]].copy()

    # Residuals from training set: outs_recorded - consensus_line
    residuals_a = (train_pg["outs_recorded"] - train_pg["consensus_line"]).dropna().values
    print(f"  Fold test={fold['test_season']}: n_train_residuals={len(residuals_a):,}  "
          f"residual mean={residuals_a.mean():.3f}  std={residuals_a.std():.3f}")

    test_full["yhat"] = test_full["consensus_line"]
    p_under_a = bootstrap_p_under_grain(test_full, residuals_a, yhat_col="yhat")
    test_full["p_under"]    = p_under_a
    test_full["method"]     = "A_consensus_line"
    oof_a_rows.append(test_full)

oof_a = pd.concat(oof_a_rows, ignore_index=True)
brier_a = ((oof_a["p_under"] - oof_a["is_under"]) ** 2).mean()
print(f"  Method A Brier score: {brier_a:.5f}")

# ── Method B: yhat from LinearRegression model ───────────────────────────────
print()
print("=" * 60)
print("METHOD B: yhat from LinearRegression (consensus_line feature)")
print("=" * 60)

oof_b_rows = []
for fold in FOLDS:
    train_pg = df_pg[df_pg["season"].isin(fold["train_seasons"])].dropna(subset=["consensus_line","outs_recorded"])
    test_full = df_full[df_full["season"] == fold["test_season"]].copy()

    # Train OOF model
    model = LinearRegression()
    model.fit(train_pg[["consensus_line"]].values, train_pg["outs_recorded"].values)

    # Residuals from training set: outs_recorded - yhat_train
    yhat_train = model.predict(train_pg[["consensus_line"]].values)
    residuals_b = train_pg["outs_recorded"].values - yhat_train
    print(f"  Fold test={fold['test_season']}: n_train_residuals={len(residuals_b):,}  "
          f"residual mean={residuals_b.mean():.3f}  std={residuals_b.std():.3f}  "
          f"coef={model.coef_[0]:.4f}  intercept={model.intercept_:.4f}")

    test_full["yhat"] = model.predict(test_full[["consensus_line"]].values)
    p_under_b = bootstrap_p_under_grain(test_full, residuals_b, yhat_col="yhat")
    test_full["p_under"] = p_under_b
    test_full["method"]  = "B_ols_model"
    oof_b_rows.append(test_full)

oof_b = pd.concat(oof_b_rows, ignore_index=True)
brier_b = ((oof_b["p_under"] - oof_b["is_under"]) ** 2).mean()
print(f"  Method B Brier score: {brier_b:.5f}")

# ── Calibration comparison ────────────────────────────────────────────────────
print("\n── Calibration (10 decile buckets, |predicted - actual| < 0.15 = OK) ──")
print(f"{'Bucket':>8}  {'A_pred':>8}  {'A_actual':>8}  {'A_gap':>8}  {'B_pred':>8}  {'B_actual':>8}  {'B_gap':>8}")

oof_a["decile"] = pd.qcut(oof_a["p_under"], q=10, duplicates="drop", labels=False)
oof_b["decile"] = pd.qcut(oof_b["p_under"], q=10, duplicates="drop", labels=False)
for d in range(10):
    a_grp = oof_a[oof_a["decile"] == d]
    b_grp = oof_b[oof_b["decile"] == d]
    a_pred   = a_grp["p_under"].mean() if len(a_grp) else float("nan")
    a_actual = a_grp["is_under"].mean() if len(a_grp) else float("nan")
    b_pred   = b_grp["p_under"].mean() if len(b_grp) else float("nan")
    b_actual = b_grp["is_under"].mean() if len(b_grp) else float("nan")
    print(f"{d+1:>8}  {a_pred:>8.3f}  {a_actual:>8.3f}  {(a_actual-a_pred):>+8.3f}  "
          f"{b_pred:>8.3f}  {b_actual:>8.3f}  {(b_actual-b_pred):>+8.3f}")

# ── Clipping check ────────────────────────────────────────────────────────────
print("\n── Clipping to [0.01, 0.99] ──")
for oof, name in [(oof_a, "A"), (oof_b, "B")]:
    n_lo = (oof["p_under"] < 0.01).sum()
    n_hi = (oof["p_under"] > 0.99).sum()
    print(f"  Method {name}: clipped_to_0.01={n_lo} ({n_lo/len(oof)*100:.2f}%)  "
          f"clipped_to_0.99={n_hi} ({n_hi/len(oof)*100:.2f}%)")
    oof["p_under"] = oof["p_under"].clip(0.01, 0.99)

# ── Monotonicity check: p_under increases as line increases (same player-game) ─
print("\n── Line monotonicity (p_under should increase as line increases) ──")
for oof, name in [(oof_a, "A"), (oof_b, "B")]:
    multi = oof.groupby(["player_key", "game_date"]).filter(lambda x: x["line"].nunique() > 1)
    inversions = 0
    total_pairs = 0
    for (pk, gd), grp in multi.groupby(["player_key", "game_date"]):
        grp = grp.sort_values("line")
        lines = grp["line"].values
        pus   = grp["p_under"].values
        for i in range(len(lines) - 1):
            total_pairs += 1
            if pus[i+1] < pus[i] - 1e-6:
                inversions += 1
    rate = inversions / total_pairs if total_pairs > 0 else 0
    print(f"  Method {name}: inversions={inversions}/{total_pairs} ({rate*100:.2f}%)")

# ── Save OOF predictions ──────────────────────────────────────────────────────
oof_a.to_parquet(OOF_A_PATH, index=False)
oof_b.to_parquet(OOF_B_PATH, index=False)
print(f"\nSaved Method A OOF → {OOF_A_PATH}  ({len(oof_a):,} rows)")
print(f"Saved Method B OOF → {OOF_B_PATH}  ({len(oof_b):,} rows)")

# ── Freddy Peralta spot-check ─────────────────────────────────────────────────
print("\n── Freddy Peralta spot-check (latest 8 starts, Method A vs B) ──")
fp_a = oof_a[oof_a["player_key"].str.contains("peralta", case=False)].drop_duplicates("game_date").sort_values("game_date").tail(8)
fp_b = oof_b[oof_b["player_key"].str.contains("peralta", case=False)].drop_duplicates("game_date").sort_values("game_date").tail(8)
merged = fp_a[["game_date","line","consensus_line","outs_recorded","is_under","p_under"]].copy()
merged.columns = ["game_date","line","consensus_line","actual","is_under","p_under_A"]
merged["p_under_B"] = fp_b["p_under"].values
merged["novig_under"] = fp_a["novig_prob_under"].values
merged["edge_A"] = (merged["p_under_A"] - merged["novig_under"]).round(4)
merged["edge_B"] = (merged["p_under_B"] - merged["novig_under"]).round(4)
print(merged.to_string(index=False))

print(f"\nBrier score summary:")
print(f"  Method A (yhat=consensus_line): {brier_a:.5f}")
print(f"  Method B (yhat=OLS model):      {brier_b:.5f}")
print(f"  Winner: {'A' if brier_a <= brier_b else 'B'} (lower is better)")
