"""
Step 3b — Individual feature sweep (XGBRegressor only).
Same features/folds as 3a. Outputs RMSE / MAE / R² per feature.
Stacks with 3a CSV for side-by-side comparison.
"""
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

SPINE_PATH  = "/Users/thomasmyles/Downloads/tmp/mlb_pitcher_outs_spine.parquet"
STEP3A_PATH = "/Users/thomasmyles/Downloads/tmp/mlb_pitcher_outs_step3a_results.csv"
OUT_PATH    = "/Users/thomasmyles/Downloads/tmp/mlb_pitcher_outs_step3b_results.csv"

df_full = pd.read_parquet(SPINE_PATH)
df = df_full.drop_duplicates(subset=["player_key", "game_date"]).copy()
df["season"] = df["game_date"].str[:4].astype(int)
print(f"Player-game rows: {len(df):,}")

TARGET = "outs_recorded"
FOLDS  = [
    {"train_seasons": [2024],       "test_season": 2025},
    {"train_seasons": [2024, 2025], "test_season": 2026},
]

NUMERIC_FEATURES = [
    "outs_roll_career", "outs_roll_c20", "outs_roll_c10", "outs_roll_c5",
    "outs_roll_c3", "outs_roll_c1",
    "outs_roll_season", "outs_roll_s20", "outs_roll_s10", "outs_roll_s5",
    "outs_roll_s3", "outs_roll_s1",
    "k_roll_career", "k_roll_c20", "k_roll_c10", "k_roll_c5",
    "k_roll_c3", "k_roll_c1", "k_roll_season",
    "opp_k_against_season",
    "consensus_line", "min_line", "max_line",
    "min_raw_prob_over", "max_raw_prob_over",
    "min_raw_prob_under", "max_raw_prob_under",
    "team_run_line_point", "team_moneyline_odds",
    "is_home", "days_rest", "game_month", "start_num_season",
    "ip_roll_season", "n_books",
]
CATEGORICAL_FEATURES = [
    "over_price_bucket_coarse", "over_price_bucket_fine",
    "under_price_bucket_coarse", "under_price_bucket_fine",
]

def eval_feature(df, feature, is_cat=False):
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
        model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                             random_state=42, verbosity=0)
        model.fit(train[[feature]].values, train[TARGET].values)
        oof_preds.append(model.predict(test[[feature]].values))
        oof_actuals.append(test[TARGET].values)
    if not oof_preds:
        return None
    y_pred = np.concatenate(oof_preds)
    y_true = np.concatenate(oof_actuals)
    return {
        "feature": feature,
        "type":    "categorical" if is_cat else "numeric",
        "n_oof":   len(y_true),
        "rmse":    round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "mae":     round(mean_absolute_error(y_true, y_pred), 4),
        "r2":      round(r2_score(y_true, y_pred), 5),
        "coef":    None,
    }

results = []
for feat in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
    if feat not in df.columns:
        continue
    r = eval_feature(df, feat, is_cat=(feat in CATEGORICAL_FEATURES))
    if r:
        results.append(r)
        print(f"  {feat:45s}  RMSE={r['rmse']:.4f}  MAE={r['mae']:.4f}  R²={r['r2']:.5f}")

xgb_df = pd.DataFrame(results).sort_values("rmse")
xgb_df["model_type"] = "xgboost"

# Stack with Step 3a
step3a = pd.read_csv(STEP3A_PATH)
step3a["model_type"] = "linear"
combined = pd.concat([step3a.assign(model_type="linear"), xgb_df], ignore_index=True)
combined = combined.sort_values(["feature", "model_type"])
combined.to_csv(OUT_PATH, index=False)
print(f"\nSaved {len(combined)} rows → {OUT_PATH}")

print("\nXGBoost top 15 by RMSE:")
print(xgb_df.head(15).to_string(index=False))

print("\nSide-by-side for top 15 linear features:")
top_linear = step3a.nsmallest(15, "rmse")["feature"].tolist()
cmp = combined[combined["feature"].isin(top_linear)].pivot_table(
    index="feature", columns="model_type", values=["rmse","r2"]
).sort_values(("rmse","linear"))
print(cmp.to_string())
