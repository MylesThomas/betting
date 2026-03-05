"""
NBA multimarket strategy: feature importance from game-results-only predictors.

Loads the unified strategy parquet (game results + props + game lines + actuals, one row
per player-game). For each of 9 markets, runs:
1) Regression models (XGBoost + regression tree) to predict market actuals.
2) Classification models (XGBoost + decision tree) to predict whether a prop covers.

Features intentionally exclude betting-line and odds columns; we only use numeric game
result/context fields (for example minutes, points, rebounds, assists, team scores).

Context: docs/exec-plans/active/nba-multimarket-strategy-analysis.md
Data: Built by scripts/build_nba_multimarket_strategy_dataset.py
Requires: pandas, numpy, sklearn, xgboost (pip install xgboost if missing)

Usage:
    python analysis/analyze_nba_multimarket_strategy_ml.py
    python analysis/analyze_nba_multimarket_strategy_ml.py --parquet ~/Downloads/tmp/nba_prop_strategies.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Repo root for imports
def _repo_root() -> Path:
    p = Path(__file__).resolve().parent.parent
    if (p / ".gitignore").exists():
        return p
    raise RuntimeError("Repo root not found (no .gitignore)")

REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Same 9 markets and target columns as build script
PROP_MARKET_ORDER = [
    "player_assists",
    "player_blocks",
    "player_double_double",
    "player_points",
    "player_points_rebounds_assists",
    "player_rebounds",
    "player_steals",
    "player_threes",
    "player_triple_double",
]

MARKET_TO_ACTUAL = {
    "player_points": "actual_points",
    "player_rebounds": "actual_rebounds",
    "player_assists": "actual_assists",
    "player_threes": "actual_threes",
    "player_steals": "actual_steals",
    "player_blocks": "actual_blocks",
    "player_points_rebounds_assists": "actual_points_rebounds_assists",
    "player_double_double": "actual_double_double",
    "player_triple_double": "actual_triple_double",
}

IDENTITY_COLUMNS = {
    "player",
    "player_normalized",
    "game_time",
    "fetch_date",
    "game_date",
    "GAME_ID",
    "team_abbr",
    "team_full",
    "home_team",
    "away_team",
    "log_player_id",
    "log_player_name",
    "log_team_id",
    "log_team_name",
    "log_game_id",
    "log_game_date",
    "log_matchup",
    "log_wl",
    "log_season",
    "log_team_abbreviation",
    "season",
    "bookmakers",
}

BETTING_COLUMN_PATTERNS = (
    "market_median_value_",
    "spread",
    "moneyline",
    "odds",
)

MARKET_TO_DIRECT_ACTUAL_ALIASES = {
    "player_points": {"actual_points", "actual_pts"},
    "player_rebounds": {"actual_rebounds", "actual_reb"},
    "player_assists": {"actual_assists", "actual_ast"},
    "player_threes": {"actual_threes"},
    "player_steals": {"actual_steals", "actual_stl"},
    "player_blocks": {"actual_blocks", "actual_blk"},
    "player_points_rebounds_assists": {"actual_points_rebounds_assists"},
    "player_double_double": {"actual_double_double"},
    "player_triple_double": {"actual_triple_double"},
}


def load_parquet(path: Path) -> pd.DataFrame:
    """Load strategy parquet and ensure required columns exist."""
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Parquet not found: {path}")
    df = pd.read_parquet(path)
    for col in ["team_full", "home_team", "away_team", "home_spread", "away_spread"]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    return df


def get_game_result_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric feature columns that exclude betting/line and identity columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = []
    for col in numeric_cols:
        if col in IDENTITY_COLUMNS:
            continue
        if any(pattern in col for pattern in BETTING_COLUMN_PATTERNS):
            continue
        feature_cols.append(col)
    return feature_cols


def build_market_frames(
    df: pd.DataFrame, market: str, feature_cols: list[str]
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Build feature matrix and targets for one market.
    - y_reg: market actual value
    - y_cls: 1 if market actual > market median line else 0
    """
    line_col = f"market_median_value_{market}"
    target_col = MARKET_TO_ACTUAL[market]
    if line_col not in df.columns:
        return (
            pd.DataFrame(),
            pd.Series(dtype=float),
            pd.Series(dtype=int),
        )
    if target_col not in df.columns:
        return (
            pd.DataFrame(),
            pd.Series(dtype=float),
            pd.Series(dtype=int),
        )

    y_reg = df[target_col]
    line_vals = df[line_col]
    market_feature_cols = [
        col
        for col in feature_cols
        if col not in MARKET_TO_DIRECT_ACTUAL_ALIASES[market]
    ]
    X = df[market_feature_cols].copy()
    valid = X.notna().all(axis=1) & y_reg.notna() & line_vals.notna()
    X = X.loc[valid].copy()
    y_reg = y_reg.loc[valid].astype(float)
    y_cls = (y_reg > line_vals.loc[valid]).astype(int)
    return X, y_reg, y_cls


def run_xgboost(X: pd.DataFrame, y: pd.Series):
    """Fit XGBoost regressor and return feature importances (same order as X.columns)."""
    try:
        import xgboost as xgb
    except ImportError:
        return None
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        verbosity=0,
    )
    model.fit(X, y)
    imp = model.feature_importances_
    return dict(zip(X.columns, imp))


def run_regression_tree(X: pd.DataFrame, y: pd.Series):
    """Fit a single regression tree and return feature importances."""
    from sklearn.tree import DecisionTreeRegressor
    model = DecisionTreeRegressor(max_depth=6, random_state=42)
    model.fit(X, y)
    imp = model.feature_importances_
    return dict(zip(X.columns, imp))


def run_xgboost_classifier(X: pd.DataFrame, y: pd.Series):
    """Fit XGBoost classifier and return feature importances."""
    try:
        import xgboost as xgb
    except ImportError:
        return None
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        verbosity=0,
    )
    model.fit(X, y)
    imp = model.feature_importances_
    return dict(zip(X.columns, imp))


def run_classification_tree(X: pd.DataFrame, y: pd.Series):
    """Fit a single classification tree and return feature importances."""
    from sklearn.tree import DecisionTreeClassifier

    model = DecisionTreeClassifier(max_depth=6, random_state=42)
    model.fit(X, y)
    imp = model.feature_importances_
    return dict(zip(X.columns, imp))


def run_univariate_regression_sweep(
    df: pd.DataFrame, market: str, min_rows: int = 200
) -> pd.DataFrame:
    """
    For one market, run one-variable linear regression for each numeric predictor.

    This is a separate diagnostic from tree/XGBoost importance:
    - outcome y: market actual column
    - predictor X: one numeric column at a time
    """
    target_col = MARKET_TO_ACTUAL[market]
    line_col = f"market_median_value_{market}"
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    excluded_predictors = set(MARKET_TO_DIRECT_ACTUAL_ALIASES[market])
    excluded_predictors.add(target_col)
    excluded_predictors.add(line_col)

    rows = []
    y_full = df[target_col].astype(float)
    for predictor in numeric_cols:
        if predictor in excluded_predictors:
            continue
        x_full = df[predictor].astype(float)
        valid = x_full.notna() & y_full.notna()
        n = int(valid.sum())
        if n < min_rows:
            continue
        x = x_full.loc[valid]
        if x.nunique() < 2:
            continue
        y = y_full.loc[valid]
        model = LinearRegression()
        model.fit(x.to_numpy().reshape(-1, 1), y.to_numpy())
        r2 = float(model.score(x.to_numpy().reshape(-1, 1), y.to_numpy()))
        corr = float(x.corr(y))
        rows.append(
            {
                "market": market,
                "predictor": predictor,
                "n": n,
                "coef": float(model.coef_[0]),
                "intercept": float(model.intercept_),
                "r2": r2,
                "abs_corr": abs(corr),
                "corr": corr,
            }
        )
    if rows:
        out = pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)
    else:
        out = pd.DataFrame(
            columns=["market", "predictor", "n", "coef", "intercept", "r2", "abs_corr", "corr"]
        )
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Per-market game-results-only feature importance (regression + classification)"
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path.home() / "Downloads" / "tmp" / "nba_prop_strategies.parquet",
        help="Path to strategy parquet",
    )
    args = parser.parse_args()

    print("Loading parquet...")
    df = load_parquet(args.parquet)
    print(f"   Rows: {len(df):,}")

    has_xgb = False
    try:
        import xgboost  # noqa: F401
        has_xgb = True
    except ImportError:
        print("   xgboost not installed; only regression tree will run (pip install xgboost)")

    feature_cols = get_game_result_feature_columns(df)
    if not feature_cols:
        raise ValueError("No game-result numeric features found after filtering betting columns.")

    print(f"   Using {len(feature_cols)} game-result features (no line/odds columns)")

    results = []
    for market in PROP_MARKET_ORDER:
        X, y_reg, y_cls = build_market_frames(df, market, feature_cols)
        if X.empty or len(y_reg) < 100:
            print(f"   {market}: skip (insufficient data: {len(y_reg)} rows)")
            continue
        if y_cls.nunique() < 2:
            print(
                f"   {market}: skip classification "
                f"(single class after filtering, n={len(y_cls):,})"
            )
            continue

        tree_reg_imp = run_regression_tree(X, y_reg)
        xgb_reg_imp = run_xgboost(X, y_reg) if has_xgb else None
        tree_cls_imp = run_classification_tree(X, y_cls)
        xgb_cls_imp = run_xgboost_classifier(X, y_cls) if has_xgb else None

        for feat in X.columns:
            row_reg = {
                "market": market,
                "task": "regression_actual",
                "feature": feat,
                "n": len(y_reg),
                "tree_importance": tree_reg_imp[feat],
                "xgb_importance": xgb_reg_imp[feat] if xgb_reg_imp else np.nan,
            }
            results.append(row_reg)
            row_cls = {
                "market": market,
                "task": "classification_cover",
                "feature": feat,
                "n": len(y_cls),
                "tree_importance": tree_cls_imp[feat],
                "xgb_importance": xgb_cls_imp[feat] if xgb_cls_imp else np.nan,
            }
            results.append(row_cls)

        reg_tree_top = max(tree_reg_imp, key=tree_reg_imp.__getitem__)
        cls_tree_top = max(tree_cls_imp, key=tree_cls_imp.__getitem__)
        if xgb_reg_imp:
            reg_xgb_top = max(xgb_reg_imp, key=xgb_reg_imp.__getitem__)
        else:
            reg_xgb_top = "-"
        if xgb_cls_imp:
            cls_xgb_top = max(xgb_cls_imp, key=xgb_cls_imp.__getitem__)
        else:
            cls_xgb_top = "-"
        print(
            f"   {market}: n={len(y_reg):,} "
            f"reg_tree_top={reg_tree_top} reg_xgb_top={reg_xgb_top} "
            f"cls_tree_top={cls_tree_top} cls_xgb_top={cls_xgb_top}"
        )

    out = pd.DataFrame(results)
    out_path = Path(args.parquet).parent / "nba_multimarket_ml_importance.parquet"
    out.to_parquet(out_path, index=False)
    print(f"\nWrote importance table to {out_path}")

    # One-variable-at-a-time linear regression by market
    univariate_results = []
    for market in PROP_MARKET_ORDER:
        uni = run_univariate_regression_sweep(df, market)
        if uni.empty:
            continue
        univariate_results.append(uni)

    if univariate_results:
        univariate_out = pd.concat(univariate_results, ignore_index=True)
        univariate_path = Path(args.parquet).parent / "nba_multimarket_univariate_regression.parquet"
        univariate_out.to_parquet(univariate_path, index=False)
        print(f"Wrote univariate regression table to {univariate_path}")
        print("\n--- Top one-variable regressions by market (by R^2) ---")
        for market in PROP_MARKET_ORDER:
            m = univariate_out[univariate_out["market"] == market]
            if m.empty:
                continue
            top = m.sort_values("r2", ascending=False).head(5)
            print(f"\n{market}")
            print(top[["predictor", "n", "coef", "r2", "corr"]].to_string(index=False))

    # Pretty print summary
    for task in ["regression_actual", "classification_cover"]:
        task_df = out[out["task"] == task].copy()
        print(f"\n--- Top tree features ({task}) ---")
        tree_top = (
            task_df.sort_values("tree_importance", ascending=False)
            .groupby("market", as_index=False)
            .head(5)
        )
        print(tree_top[["market", "feature", "tree_importance"]].to_string(index=False))
        if has_xgb and task_df["xgb_importance"].notna().any():
            print(f"\n--- Top XGBoost features ({task}) ---")
            xgb_top = (
                task_df.sort_values("xgb_importance", ascending=False)
                .groupby("market", as_index=False)
                .head(5)
            )
            print(xgb_top[["market", "feature", "xgb_importance"]].to_string(index=False))


if __name__ == "__main__":
    main()
