"""
NBA multimarket strategy: XGBoost + regression tree variable importance per market.

Loads the unified strategy parquet (game results + props + game lines + actuals, one row
per player-game). For each of 9 markets, trains XGBoost and a regression tree to predict
that market's actual outcome; reports feature importance to see what correlates with
higher actuals (line, team spread, home/away).

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

FEATURE_NAMES = ["line", "team_spread", "is_home"]


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


def build_features(df: pd.DataFrame, market: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build feature matrix X and target y for one market.
    Drops rows where line or target is null.
    """
    line_col = f"market_median_value_{market}"
    target_col = MARKET_TO_ACTUAL[market]
    if line_col not in df.columns or target_col not in df.columns:
        return pd.DataFrame(), pd.Series(dtype=float)

    # Team spread from player's team perspective: home_spread when team is home, else away_spread
    team_spread = np.where(df["team_full"] == df["home_team"], df["home_spread"], df["away_spread"])
    is_home = (df["team_full"] == df["home_team"]).astype(int)

    out = pd.DataFrame({
        "line": df[line_col],
        "team_spread": team_spread,
        "is_home": is_home,
    })
    out["target"] = df[target_col]

    valid = out.notna().all(axis=1)
    out = out.loc[valid].copy()
    y = out.pop("target")
    return out, y


def run_xgboost(X: pd.DataFrame, y: pd.Series):
    """Fit XGBoost regressor and return feature importances (same order as X.columns)."""
    try:
        import xgboost as xgb
    except ImportError:
        return None
    model = xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42, verbosity=0)
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


def main():
    parser = argparse.ArgumentParser(description="Per-market XGBoost + regression tree variable importance")
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

    results = []
    for market in PROP_MARKET_ORDER:
        X, y = build_features(df, market)
        if X.empty or len(y) < 100:
            print(f"   {market}: skip (insufficient data: {len(y)} rows)")
            continue

        tree_imp = run_regression_tree(X, y)
        xgb_imp = run_xgboost(X, y) if has_xgb else None

        for feat in FEATURE_NAMES:
            row = {"market": market, "feature": feat, "n": len(y)}
            row["tree_importance"] = tree_imp.get(feat, 0.0)
            row["xgb_importance"] = xgb_imp.get(feat, 0.0) if xgb_imp else None
            results.append(row)

        print(f"   {market}: n={len(y):,}  tree_top={max(tree_imp, key=tree_imp.get)}  xgb_top={max(xgb_imp, key=xgb_imp.get) if xgb_imp else '-'}")

    out = pd.DataFrame(results)
    out_path = Path(args.parquet).parent / "nba_multimarket_ml_importance.parquet"
    out.to_parquet(out_path, index=False)
    print(f"\nWrote importance table to {out_path}")

    # Pretty print summary
    print("\n--- Feature importance (regression tree) ---")
    pivot = out.pivot_table(index="market", columns="feature", values="tree_importance")
    print(pivot.round(4).to_string())
    if has_xgb and out["xgb_importance"].notna().any():
        print("\n--- Feature importance (XGBoost) ---")
        pivot_xgb = out.pivot_table(index="market", columns="feature", values="xgb_importance")
        print(pivot_xgb.round(4).to_string())


if __name__ == "__main__":
    main()
