"""
Step 3a — Individual feature sweep (LinearRegression only).
Target: outs_recorded (continuous).
OOF folds: train 2024 → test 2025, train 2024+2025 → test 2026.
No classifier. Outputs RMSE / MAE / R² / coefficient per feature.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

SPINE_PATH = "/Users/thomasmyles/Downloads/tmp/mlb_pitcher_outs_spine.parquet"
OUT_PATH   = "/Users/thomasmyles/Downloads/tmp/mlb_pitcher_outs_step3a_results.csv"

# ── Load spine, deduplicate to player-game level ─────────────────────────────
# Model predicts outs_recorded per player-game; book rows are identical on all
# features except novig_prob_over/under (excluded from model).
df_full = pd.read_parquet(SPINE_PATH)
df = (
    df_full
    .drop_duplicates(subset=["player_key", "game_date"])
    .copy()
)
df["season"] = df["game_date"].str[:4].astype(int)
print(f"Player-game rows: {len(df):,}  (from {len(df_full):,} spine rows)")
print(f"Season counts:\n{df['season'].value_counts().sort_index()}")

TARGET = "outs_recorded"

# ── OOF folds (temporal) ─────────────────────────────────────────────────────
FOLDS = [
    {"train_seasons": [2024],       "test_season": 2025},
    {"train_seasons": [2024, 2025], "test_season": 2026},
]

# ── Feature candidates ────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    # Rolling outs — cross-season
    "outs_roll_career", "outs_roll_c20", "outs_roll_c10", "outs_roll_c5",
    "outs_roll_c3", "outs_roll_c1",
    # Rolling outs — within-season
    "outs_roll_season", "outs_roll_s20", "outs_roll_s10", "outs_roll_s5",
    "outs_roll_s3", "outs_roll_s1",
    # Rolling K — cross-season
    "k_roll_career", "k_roll_c20", "k_roll_c10", "k_roll_c5",
    "k_roll_c3", "k_roll_c1",
    # Rolling K — within-season
    "k_roll_season",
    # Opponent
    "opp_k_against_season",
    # Market (book-invariant)
    "consensus_line",
    "min_line", "max_line",
    "min_raw_prob_over", "max_raw_prob_over",
    "min_raw_prob_under", "max_raw_prob_under",
    # Team context
    "team_run_line_point", "team_moneyline_odds",
    # Game context
    "is_home", "days_rest", "game_month", "start_num_season",
    # IP rolling
    "ip_roll_season",
    # Book count
    "n_books",
]

CATEGORICAL_FEATURES = [
    "over_price_bucket_coarse",
    "over_price_bucket_fine",
    "under_price_bucket_coarse",
    "under_price_bucket_fine",
]

def eval_feature(df, feature, is_cat=False):
    """Train LinearRegression on one feature using OOF temporal folds."""
    sub = df[["player_key", "game_date", "season", feature, TARGET]].dropna().copy()

    if is_cat:
        le = LabelEncoder()
        sub[feature] = le.fit_transform(sub[feature].astype(str))

    oof_preds, oof_actuals = [], []
    for fold in FOLDS:
        train = sub[sub["season"].isin(fold["train_seasons"])]
        test  = sub[sub["season"] == fold["test_season"]]
        if len(train) < 50 or len(test) < 10:
            continue
        X_tr = train[[feature]].values
        y_tr = train[TARGET].values
        X_te = test[[feature]].values
        y_te = test[TARGET].values
        model = LinearRegression()
        model.fit(X_tr, y_tr)
        oof_preds.append(model.predict(X_te))
        oof_actuals.append(y_te)

    if not oof_preds:
        return None

    y_pred  = np.concatenate(oof_preds)
    y_true  = np.concatenate(oof_actuals)
    n       = len(y_true)
    rmse    = np.sqrt(mean_squared_error(y_true, y_pred))
    mae     = mean_absolute_error(y_true, y_pred)
    r2      = r2_score(y_true, y_pred)

    # Fit on all data to get coefficient (for directional check)
    all_sub = sub.copy()
    model_all = LinearRegression()
    model_all.fit(all_sub[[feature]].values, all_sub[TARGET].values)
    coef = model_all.coef_[0]

    return {
        "feature":  feature,
        "type":     "categorical" if is_cat else "numeric",
        "n_oof":    n,
        "rmse":     round(rmse, 4),
        "mae":      round(mae, 4),
        "r2":       round(r2, 5),
        "coef":     round(coef, 5),
    }

# ── Run sweep ─────────────────────────────────────────────────────────────────
results = []
for feat in NUMERIC_FEATURES:
    if feat not in df.columns:
        print(f"  SKIP (not in spine): {feat}")
        continue
    r = eval_feature(df, feat, is_cat=False)
    if r:
        results.append(r)
        print(f"  {feat:45s}  RMSE={r['rmse']:.4f}  MAE={r['mae']:.4f}  R²={r['r2']:.5f}  coef={r['coef']:+.4f}")

for feat in CATEGORICAL_FEATURES:
    if feat not in df.columns:
        print(f"  SKIP (not in spine): {feat}")
        continue
    r = eval_feature(df, feat, is_cat=True)
    if r:
        results.append(r)
        print(f"  {feat:45s}  RMSE={r['rmse']:.4f}  MAE={r['mae']:.4f}  R²={r['r2']:.5f}  coef={r['coef']:+.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(results).sort_values("rmse")
results_df.to_csv(OUT_PATH, index=False)
print(f"\nSaved {len(results_df)} features → {OUT_PATH}")
print("\nTop 15 by RMSE (lowest = best):")
print(results_df.head(15).to_string(index=False))
