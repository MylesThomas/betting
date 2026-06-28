"""
EDA for NFL tackles labeled dataset.

Sections:
  1. Unmatched names audit (odds names with no spine match)
  2. Target distribution (tackles_combined)
  3. Feature correlations with target
  4. 1-predictor model performance (5-fold CV R², MAE, Pearson r)
  5. Position-level breakdown

Note: defense_pct and defense_snaps are post-game actuals — excluded from
feature analysis. Only pre-game rolling features and market line are valid predictors.

Run:
  python src/nfl_tackles_modeling/scripts/eda_labeled_dataset.py
"""

from __future__ import annotations

import warnings
from io import BytesIO
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

LABELED_PATH = Path.home() / "Downloads" / "tmp" / "nfl_tackles_labeled.parquet"
SPINE_PATH   = Path.home() / "Downloads" / "tmp" / "nfl_tackles_historical_spine.parquet"
S3_BUCKET    = "the-odds-api-mt"
S3_PREFIX    = "nfl/props_backfill"
TARGET_MKT   = "player_tackles_assists"
SEASONS      = [2024, 2025]

# Pre-game features only — no leakage
FEATURES = [
    "tackle_rate_L3",
    "tackle_rate_L10",
    "tackle_rate_Lcareer",
    "snap_pct_L3",
    "snap_pct_L10",
    "snap_pct_Lcareer",
    "opp_run_rate_L3",
    "offered_line",
]

TARGET = "tackles_combined"


def section(n: int, title: str) -> None:
    print(f"\n{'='*65}")
    print(f"  {n}. {title}")
    print(f"{'='*65}")


# ── 1. Unmatched names audit ───────────────────────────────────────────────────

def audit_unmatched(labeled: pd.DataFrame) -> None:
    section(1, "UNMATCHED NAMES AUDIT")
    print("  Loading odds names from S3...")

    s3        = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    frames    = []
    for season in SEASONS:
        prefix = f"{S3_PREFIX}/{season}/"
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                body = s3.get_object(Bucket=S3_BUCKET, Key=obj["Key"])["Body"].read()
                df   = pd.read_parquet(BytesIO(body))
                df   = df[df["market"] == TARGET_MKT][["nfl_game_id", "outcome_desc", "outcome_name"]].drop_duplicates()
                frames.append(df)

    non_empty = [f for f in frames if len(f) > 0]
    all_odds  = pd.concat(non_empty, ignore_index=True)
    odds_names = (
        all_odds[all_odds["outcome_name"] == "Over"]
        .drop_duplicates(subset=["nfl_game_id", "outcome_desc"])
    )

    matched_keys   = set(zip(labeled["game_id"], labeled["player_name"]))
    odds_names["matched"] = odds_names.apply(
        lambda r: (r["nfl_game_id"], r["outcome_desc"]) in matched_keys, axis=1
    )

    unmatched = (
        odds_names[~odds_names["matched"]]
        .groupby("outcome_desc")["nfl_game_id"]
        .count()
        .rename("n_games")
        .sort_values(ascending=False)
        .reset_index()
    )

    print(f"\n  {len(unmatched)} odds names with no spine match  "
          f"({len(odds_names):,} total odds player-games  →  "
          f"{odds_names['matched'].mean():.1%} matched)\n")
    print(unmatched.rename(columns={"outcome_desc": "odds_name"}).to_string(index=False))

    # Pattern check — does the name appear anywhere in the spine?
    spine     = pd.read_parquet(SPINE_PATH, columns=["player_name"])
    spine_set = set(spine["player_name"].dropna().unique())
    unmatched["in_spine_partial"] = unmatched["outcome_desc"].apply(
        lambda n: any(n.split()[-1] in sn for sn in spine_set)
    )
    partial = unmatched[unmatched["in_spine_partial"]]
    if not partial.empty:
        print(f"\n  Names not matched but last name exists in spine (format mismatch candidates):")
        print(partial[["outcome_desc", "n_games"]].to_string(index=False))


# ── 2. Target distribution ─────────────────────────────────────────────────────

def target_distribution(df: pd.DataFrame) -> None:
    section(2, "TARGET DISTRIBUTION — tackles_combined")
    stats = df[TARGET].describe(percentiles=[.1, .25, .5, .75, .9])
    print(f"\n  {stats.to_string()}")

    print("\n  Value counts (0–15+):")
    vc = df[TARGET].clip(upper=15).value_counts().sort_index()
    for val, cnt in vc.items():
        bar  = "█" * (cnt // 30)
        label = f"{val}+" if val == 15 else str(val)
        print(f"    {label:>3}  {cnt:>5}  {bar}")


# ── 3. Feature correlations ────────────────────────────────────────────────────

def feature_correlations(df: pd.DataFrame) -> None:
    section(3, "FEATURE CORRELATIONS WITH tackles_combined")
    corrs = (
        df[FEATURES + [TARGET]]
        .corr()[TARGET]
        .drop(TARGET)
        .sort_values(key=abs, ascending=False)
    )
    print()
    for feat, r in corrs.items():
        bar = "█" * int(abs(r) * 40)
        print(f"  {feat:<25}  r={r:+.3f}  {bar}")


# ── 4. 1-predictor models ──────────────────────────────────────────────────────

def _eval_model(model, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    r2_scores  = cross_val_score(model, X, y, cv=5, scoring="r2")
    mae_scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")
    return r2_scores.mean(), -mae_scores.mean()


def single_predictor_models(df: pd.DataFrame) -> None:
    section(4, "1-PREDICTOR MODELS — LR vs XGBoost, 5-fold CV")

    lr_model  = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])
    xgb_model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1,
                              verbosity=0, random_state=42)

    rows = []
    for feat in FEATURES:
        sub = df[[feat, TARGET]].dropna()
        if len(sub) < 100:
            continue
        X = sub[[feat]].values
        y = sub[TARGET].values

        for model_name, model in [("LR", lr_model), ("XGBoost", xgb_model)]:
            r2, mae = _eval_model(model, X, y)
            rows.append({"model": model_name, "predictors": feat, "n": len(sub),
                         "cv_r2": r2, "cv_mae": mae})

    results = pd.DataFrame(rows)

    # Delta vs market consensus (offered_line LR row = reference)
    ref_mae = results.loc[
        (results["model"] == "LR") & (results["predictors"] == "offered_line"), "cv_mae"
    ].iloc[0]
    results["delta_vs_market"] = results["cv_mae"] - ref_mae  # negative = better than market

    results = results.sort_values(["cv_r2"], ascending=False)

    hdr = f"  {'Model':<9} {'Predictors':<25} {'n':>5} {'CV R²':>7} {'CV MAE':>8} {'Δ vs mkt':>10}"
    print(f"\n{hdr}")
    print(f"  {'-'*8} {'-'*24} {'-'*5} {'-'*7} {'-'*8} {'-'*10}")
    for _, row in results.iterrows():
        delta_str = f"{row['delta_vs_market']:+.3f}" if row["predictors"] != "offered_line" or row["model"] != "LR" else "  0.000"
        print(f"  {row['model']:<9} {row['predictors']:<25} {row['n']:>5} "
              f"{row['cv_r2']:>7.3f} {row['cv_mae']:>8.3f} {delta_str:>10}")

    print(f"\n  Baseline MAE (predict mean): {df[TARGET].std():.3f}")


# ── 5. Market calibration curve ───────────────────────────────────────────────

def market_calibration(df: pd.DataFrame) -> None:
    section(5, "MARKET CALIBRATION — actual vs line by line bucket")
    sub = df[["offered_line", TARGET]].dropna()
    sub["line_bucket"] = pd.cut(
        sub["offered_line"],
        bins=[0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 99],
        labels=["≤3.0", "3.5", "4.5", "5.5", "6.5", "7.5", "8.5+"],
    )
    cal = (
        sub.groupby("line_bucket", observed=True)
        .agg(
            n          = (TARGET, "count"),
            avg_line   = ("offered_line", "mean"),
            avg_actual = (TARGET, "mean"),
            over_rate  = (TARGET, lambda x: (x.values > sub.loc[x.index, "offered_line"].values).mean()),
        )
    )
    cal["bias"]      = (cal["avg_actual"] - cal["avg_line"]).round(2)
    cal["avg_line"]  = cal["avg_line"].round(2)
    cal["avg_actual"]= cal["avg_actual"].round(2)
    cal["over_rate"] = cal["over_rate"].map("{:.1%}".format)
    print()
    print(cal.to_string())
    print("\n  bias = avg_actual − avg_line  (negative → market overprices overs)")


# ── 6. Season split ────────────────────────────────────────────────────────────

def season_split(df: pd.DataFrame) -> None:
    section(6, "SEASON SPLIT — does the model generalize 2024 → 2025?")
    lr_model = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])

    print(f"\n  {'Feature':<25}  {'2024 MAE':>9}  {'2025 MAE':>9}  {'Δ (25-24)':>10}")
    print(f"  {'-'*24}  {'-'*9}  {'-'*9}  {'-'*10}")

    for feat in ["tackle_rate_L10", "tackle_rate_Lcareer", "offered_line", "tackle_rate_L3"]:
        maes = {}
        for season in [2024, 2025]:
            sub = df[df["season"] == season][[feat, TARGET]].dropna()
            if len(sub) < 50:
                maes[season] = float("nan")
                continue
            X = sub[[feat]].values
            y = sub[TARGET].values
            scores = cross_val_score(lr_model, X, y, cv=5, scoring="neg_mean_absolute_error")
            maes[season] = -scores.mean()

        delta = maes[2025] - maes[2024] if not any(np.isnan(v) for v in maes.values()) else float("nan")
        print(f"  {feat:<25}  {maes[2024]:>9.3f}  {maes[2025]:>9.3f}  {delta:>+10.3f}")


# ── 7. Position-level breakdown ────────────────────────────────────────────────

def position_breakdown(df: pd.DataFrame) -> None:
    section(7, "POSITION-LEVEL BREAKDOWN")
    pos = (
        df.groupby("position")
        .agg(
            n             = (TARGET, "count"),
            avg_tackles   = (TARGET, "mean"),
            std_tackles   = (TARGET, "std"),
            median_line   = ("offered_line", "median"),
            beat_line_pct = (TARGET, lambda x: (
                x.values > df.loc[x.index, "offered_line"].values
            ).mean()),
        )
        .sort_values("n", ascending=False)
    )
    pos["avg_tackles"]   = pos["avg_tackles"].round(2)
    pos["std_tackles"]   = pos["std_tackles"].round(2)
    pos["median_line"]   = pos["median_line"].round(1)
    pos["beat_line_pct"] = pos["beat_line_pct"].map("{:.1%}".format)
    print()
    print(pos.to_string())


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    warnings.filterwarnings("ignore")

    df = pd.read_parquet(LABELED_PATH)
    print(f"\nLoaded labeled dataset: {len(df):,} rows  |  {df['player_name'].nunique():,} players")

    audit_unmatched(df)
    target_distribution(df)
    feature_correlations(df)
    single_predictor_models(df)
    market_calibration(df)
    season_split(df)
    position_breakdown(df)

    print(f"\n{'='*65}\n  EDA COMPLETE\n{'='*65}\n")


if __name__ == "__main__":
    main()
