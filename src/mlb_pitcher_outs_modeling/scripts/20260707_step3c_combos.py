"""
Step 3c — Combo model evaluation (LinearRegression + XGBRegressor).
OOF folds: train 2024 → test 2025, train 2024+2025 → test 2026.
Target: outs_recorded (continuous).
Saves winning model artifact + OOF predictions.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import joblib, sklearn, warnings
warnings.filterwarnings("ignore")

SPINE_PATH  = "/Users/thomasmyles/Downloads/tmp/mlb_pitcher_outs_spine.parquet"
OOF_PATH    = "/Users/thomasmyles/Downloads/tmp/mlb_pitcher_outs_oof_preds.parquet"
MODEL_PATH  = "/Users/thomasmyles/Downloads/tmp/mlb_pitcher_outs_model/model.joblib"
OUT_PATH    = "/Users/thomasmyles/Downloads/tmp/mlb_pitcher_outs_step3c_results.csv"

df_full = pd.read_parquet(SPINE_PATH)
df = df_full.drop_duplicates(subset=["player_key", "game_date"]).copy()
df["season"] = df["game_date"].str[:4].astype(int)
print(f"Player-game rows: {len(df):,}")

TARGET = "outs_recorded"
FOLDS  = [
    {"train_seasons": [2024],       "test_season": 2025},
    {"train_seasons": [2024, 2025], "test_season": 2026},
]

# Encode categorical odds bin features
CAT_COLS = ["over_price_bucket_coarse", "over_price_bucket_fine",
            "under_price_bucket_coarse", "under_price_bucket_fine"]
ENCODERS = {}
for col in CAT_COLS:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col].astype(str))
    ENCODERS[col] = le

COMBOS = [
    # 1. Line only — baseline
    {
        "name": "line_only",
        "features": ["consensus_line"],
        "rationale": "Market line is the best single predictor (RMSE 3.52, R² 0.102). Baseline.",
    },
    # 2. Line + best outs rolling
    {
        "name": "line_outs_c10",
        "features": ["consensus_line", "outs_roll_c10"],
        "rationale": "outs_roll_c10 is the strongest rolling feature (RMSE 3.624). Adds recent form on top of the market.",
    },
    # 3. Line + outs rolling (short + career)
    {
        "name": "line_outs_c10_career",
        "features": ["consensus_line", "outs_roll_c10", "outs_roll_career"],
        "rationale": "Combines recent form (c10) with long-run baseline (career). Career avg as the floor, c10 as the current-form signal.",
    },
    # 4. Line + outs rolling (short + season)
    {
        "name": "line_outs_c10_season",
        "features": ["consensus_line", "outs_roll_c10", "outs_roll_season"],
        "rationale": "Season rolling vs career rolling — test whether within-season context is more relevant.",
    },
    # 5. Line + outs (short + career) + K rolling
    {
        "name": "line_outs_career_k5",
        "features": ["consensus_line", "outs_roll_c10", "outs_roll_career", "k_roll_c5"],
        "rationale": "K rolling showed moderate single-feature signal (RMSE 3.68). Test if K rate adds to outs prediction.",
    },
    # 6. Line + outs + team context
    {
        "name": "line_outs_c10_career_team",
        "features": ["consensus_line", "outs_roll_c10", "outs_roll_career",
                     "team_run_line_point", "team_moneyline_odds"],
        "rationale": "Team context: favored pitchers may go deeper (less need for bullpen). moneyline added moderate signal in 3b.",
    },
    # 7. Broader combo — top 8 features
    {
        "name": "broad_8",
        "features": ["consensus_line", "outs_roll_c10", "outs_roll_career",
                     "k_roll_c5", "k_roll_career",
                     "team_run_line_point", "team_moneyline_odds", "is_home"],
        "rationale": "Broader feature set. Tests whether combining many weak signals improves over the tight combos.",
    },
    # 8. Line + outs + opponent K rate
    {
        "name": "line_outs_c10_career_opp",
        "features": ["consensus_line", "outs_roll_c10", "outs_roll_career", "opp_k_against_season"],
        "rationale": "Opponent K rate tests matchup signal. Weak individually but may add in combos.",
    },
    # 9. Line + outs + odds bins
    {
        "name": "line_outs_c10_career_bins",
        "features": ["consensus_line", "outs_roll_c10", "outs_roll_career",
                     "over_price_bucket_coarse_enc", "under_price_bucket_coarse_enc"],
        "rationale": "Odds bins (over/under favorite) encode market structural signal beyond the raw line.",
    },
    # 10. Best combo + all context
    {
        "name": "full_combo",
        "features": ["consensus_line", "outs_roll_c10", "outs_roll_career",
                     "k_roll_c5", "k_roll_career",
                     "team_run_line_point", "team_moneyline_odds",
                     "opp_k_against_season", "is_home", "days_rest"],
        "rationale": "Kitchen sink. If this doesn't beat broad_8, extra features add noise.",
    },
]

def eval_combo(df, features, model_cls, model_name):
    sub = df[["player_key", "game_date", "season"] + features + [TARGET]].dropna().copy()
    oof_rows = []
    for fold in FOLDS:
        train = sub[sub["season"].isin(fold["train_seasons"])]
        test  = sub[sub["season"] == fold["test_season"]]
        if len(train) < 50 or len(test) < 10:
            continue
        model = model_cls()
        model.fit(train[features].values, train[TARGET].values)
        preds = model.predict(test[features].values)
        fold_df = test[["player_key", "game_date", "season", TARGET]].copy()
        fold_df["yhat"] = preds
        oof_rows.append(fold_df)
    if not oof_rows:
        return None, None
    oof = pd.concat(oof_rows)
    y_pred = oof["yhat"].values
    y_true = oof[TARGET].values
    return {
        "n_oof": len(y_true),
        "rmse":  round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "mae":   round(mean_absolute_error(y_true, y_pred), 4),
        "r2":    round(r2_score(y_true, y_pred), 5),
    }, oof

MODELS = [
    ("linear", lambda: LinearRegression()),
    ("xgboost", lambda: XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.8,
                                     random_state=42, verbosity=0)),
]

results = []
best_rmse, best_oof, best_combo_name, best_model_type = 9999, None, None, None

for combo in COMBOS:
    for model_name, model_cls in MODELS:
        feats_in_df = [f for f in combo["features"] if f in df.columns]
        if len(feats_in_df) < len(combo["features"]):
            missing = set(combo["features"]) - set(df.columns)
            print(f"  SKIP {combo['name']} ({model_name}) — missing cols: {missing}")
            continue
        metrics, oof = eval_combo(df, feats_in_df, model_cls, model_name)
        if metrics is None:
            continue
        row = {
            "combo_name":    combo["name"],
            "features":      ", ".join(feats_in_df),
            "n_features":    len(feats_in_df),
            "model_type":    model_name,
            "n_oof":         metrics["n_oof"],
            "rmse":          metrics["rmse"],
            "mae":           metrics["mae"],
            "r2":            metrics["r2"],
            "rationale":     combo["rationale"],
        }
        results.append(row)
        print(f"  {combo['name']:35s} {model_name:8s}  RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  R²={metrics['r2']:.5f}")
        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_oof  = oof
            best_combo_name = combo["name"]
            best_model_cls  = model_cls
            best_model_type = model_name
            best_features   = feats_in_df

results_df = pd.DataFrame(results).sort_values("rmse")
results_df.to_csv(OUT_PATH, index=False)
print(f"\nSaved {len(results_df)} combos → {OUT_PATH}")
print(f"\nBest: {best_combo_name} ({best_model_type})  RMSE={best_rmse:.4f}")

# ── Yhat book-invariant check ─────────────────────────────────────────────────
# Broadcast OOF predictions back to full spine
full_check = df_full.copy()
full_check["season"] = full_check["game_date"].str[:4].astype(int)
for col in CAT_COLS:
    if col + "_enc" not in best_oof.columns:
        full_check[col + "_enc"] = ENCODERS[col].transform(full_check[col].astype(str))

oof_pg = best_oof[["player_key", "game_date", "yhat"]].copy()
full_check = full_check.merge(oof_pg, on=["player_key", "game_date"], how="left")

# Check that yhat is the same across all books for the same player-game-line
check = full_check.dropna(subset=["yhat"]).groupby(["player_key", "game_date", "line"])["yhat"].nunique()
assert (check == 1).all(), (
    f"yhat is NOT book-invariant — {(check > 1).sum()} (player, game, line) groups vary across books. "
    f"Check that no per-book feature is inside the model."
)
print("✅ yhat is book-invariant across all books for the same player-game-line.")

# ── Save OOF predictions ──────────────────────────────────────────────────────
best_oof.to_parquet(OOF_PATH, index=False)
print(f"OOF predictions saved → {OOF_PATH}  ({len(best_oof):,} rows)")

# ── Retrain on ALL data and save model ───────────────────────────────────────
print(f"\nRetraining {best_model_type} on all data with features: {best_features}")
sub_all = df[best_features + [TARGET]].dropna()
final_model = best_model_cls()
final_model.fit(sub_all[best_features].values, sub_all[TARGET].values)
joblib.dump(final_model, MODEL_PATH)
print(f"Model saved → {MODEL_PATH}")
print(f"sklearn version: {sklearn.__version__}")
print(f"Features: {best_features}")

# ── Spot-check: Freddy Peralta ────────────────────────────────────────────────
print("\n── Freddy Peralta spot-check ──")
fp_oof = best_oof[best_oof["player_key"].str.contains("peralta", case=False)]
print(fp_oof[["player_key","game_date","season","outs_recorded","yhat"]].tail(10).to_string(index=False))
