"""
Run an overfit-first in-sample sweep to beat consensus rebounds line.

Context:
- User requested a tmp script that can be mirrored in a notebook workflow.
- Objective is strictly in-sample feature combination discovery for REB.
- This script rebuilds props/logs/panel from cached parquet artifacts so no
  notebook state export (panel/main) is required.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge


def parse_args() -> argparse.Namespace:
    """Parse CLI args for the overfit sweep."""
    parser = argparse.ArgumentParser(description="Overfit-first rebounds model sweep.")
    parser.add_argument("--tmp-dir", type=str, default="~/Downloads/tmp")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="~/Downloads/tmp/rebounds_overfit_outputs",
    )
    parser.add_argument("--run-bruteforce", action="store_true")
    parser.add_argument("--seed", type=int, default=69)
    return parser.parse_args()


def american_to_implied_prob(odds: float) -> float:
    """Convert American odds to implied probability."""
    if np.isnan(odds):
        return float("nan")
    if odds < 0:
        return float((-odds) / ((-odds) + 100.0))
    return float(100.0 / (odds + 100.0))


def remove_vig_two_way(p_over: float, p_under: float) -> tuple[float, float]:
    """Remove vig from two-way implied probabilities."""
    total = p_over + p_under
    if total <= 0.0:
        return 0.5, 0.5
    return float(p_over / total), float(p_under / total)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute RMSE/MAE/R2."""
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def load_cached_artifacts(tmp_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load cached props/logs/v6 artifacts from tmp directory."""
    props_files = sorted(tmp_dir.glob("v1_rebounds_props_*.parquet"))
    logs_files = sorted(tmp_dir.glob("v1_rebounds_logs_*.parquet"))
    v6_path = tmp_dir / "v6_spread_universe.parquet"

    if len(props_files) == 0:
        raise ValueError("No v1_rebounds_props_*.parquet files found in tmp dir")
    if len(logs_files) == 0:
        raise ValueError("No v1_rebounds_logs_*.parquet files found in tmp dir")
    if not v6_path.exists():
        raise ValueError("Missing v6_spread_universe.parquet in tmp dir")

    props = pd.concat([pd.read_parquet(p) for p in props_files], ignore_index=True)
    logs = pd.concat([pd.read_parquet(p) for p in logs_files], ignore_index=True)
    v6 = pd.read_parquet(v6_path)
    return props, logs, v6


def build_main(props: pd.DataFrame, logs: pd.DataFrame) -> pd.DataFrame:
    """Build canonical one-line-per-book rows and join realized REB outcomes."""
    required_props = ["season", "date", "player_normalized", "bookmaker", "line", "odds_over", "odds_under"]
    required_logs = ["season", "date", "player_normalized", "game_id", "MIN", "OREB", "DREB", "REB"]
    missing_props = [c for c in required_props if c not in props.columns]
    missing_logs = [c for c in required_logs if c not in logs.columns]
    if missing_props:
        raise ValueError(f"props missing required columns: {missing_props}")
    if missing_logs:
        raise ValueError(f"logs missing required columns: {missing_logs}")

    props = props.copy()
    logs = logs.copy()
    props["date"] = pd.to_datetime(props["date"])
    logs["date"] = pd.to_datetime(logs["date"])
    for c in ["line", "odds_over", "odds_under"]:
        props[c] = pd.to_numeric(props[c], errors="coerce")
    for c in ["MIN", "OREB", "DREB", "REB"]:
        logs[c] = pd.to_numeric(logs[c], errors="coerce")

    props = props.dropna(subset=required_props).copy()
    logs = logs.dropna(subset=required_logs).copy()

    logs_dup = logs.duplicated(subset=["season", "date", "player_normalized"], keep=False).sum()
    if logs_dup > 0:
        raise ValueError(f"logs has duplicate season/date/player keys: {logs_dup}")

    props["p_over_raw"] = props["odds_over"].apply(american_to_implied_prob)
    props["p_under_raw"] = props["odds_under"].apply(american_to_implied_prob)
    no_vig = props.apply(
        lambda r: remove_vig_two_way(float(r["p_over_raw"]), float(r["p_under_raw"])),
        axis=1,
    )
    props["p_over_novig"] = [x[0] for x in no_vig]
    props["p_under_novig"] = [x[1] for x in no_vig]
    props["distance_to_5050"] = (props["p_over_novig"] - 0.5).abs()

    book_line = (
        props.groupby(["season", "date", "player_normalized", "bookmaker", "line"], as_index=False)
        .agg(
            odds_over=("odds_over", "median"),
            odds_under=("odds_under", "median"),
            p_over_raw=("p_over_raw", "median"),
            p_under_raw=("p_under_raw", "median"),
            p_over_novig=("p_over_novig", "median"),
            p_under_novig=("p_under_novig", "median"),
            distance_to_5050=("distance_to_5050", "median"),
        )
    )
    main = (
        book_line.sort_values(
            ["season", "date", "player_normalized", "bookmaker", "distance_to_5050", "line"]
        )
        .groupby(["season", "date", "player_normalized", "bookmaker"], as_index=False)
        .first()
    )
    main = main.merge(
        logs[["season", "date", "player_normalized", "game_id", "MIN", "OREB", "DREB", "REB"]],
        on=["season", "date", "player_normalized"],
        how="inner",
    )
    return main


def build_panel(main: pd.DataFrame, v6: pd.DataFrame) -> pd.DataFrame:
    """Build one-row-per-game panel with disagreement context and outcomes."""
    group_keys = ["season", "date", "player_normalized", "game_id"]

    panel = (
        main.groupby(group_keys, as_index=False)
        .agg(
            n_books=("bookmaker", "nunique"),
            min_line=("line", "min"),
            max_line=("line", "max"),
            median_line=("line", "median"),
            std_line=("line", "std"),
            MIN=("MIN", "first"),
            OREB=("OREB", "first"),
            DREB=("DREB", "first"),
            REB=("REB", "first"),
        )
    )
    panel["std_line"] = panel["std_line"].fillna(0.0)
    panel["line_range"] = panel["max_line"] - panel["min_line"]
    panel["line_spread"] = panel["line_range"]
    panel["consensus_reb_line"] = panel["median_line"]

    v6_cols = ["season", "date", "player_normalized", "game_id", "FGA", "FG3A", "FTA", "spread_signed"]
    missing_v6 = [c for c in v6_cols if c not in v6.columns]
    if missing_v6:
        raise ValueError(f"v6 missing required columns: {missing_v6}")
    usage = v6[v6_cols].copy()
    usage["date"] = pd.to_datetime(usage["date"])
    usage = usage.drop_duplicates(subset=["season", "date", "player_normalized", "game_id"])

    panel = panel.merge(usage, on=["season", "date", "player_normalized", "game_id"], how="left")
    panel["spread_abs"] = panel["spread_signed"].abs()
    return panel


def build_feature_table(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build leakage-safe rolling feature table with windows through 80."""
    base = panel.copy().sort_values(["player_normalized", "date", "game_id"]).reset_index(drop=True)
    base["fg3a_share"] = np.where(base["FGA"] > 0, base["FG3A"] / base["FGA"], np.nan)
    base["reb_per_min_actual"] = np.where(base["MIN"] > 0, base["REB"] / base["MIN"], np.nan)

    g = base.groupby("player_normalized", group_keys=False)
    windows = [5, 10, 20, 40, 60, 80]
    for w in windows:
        base[f"roll_reb_mean_{w}"] = g["REB"].transform(lambda s: s.rolling(w, min_periods=1).mean().shift(1))
        base[f"roll_reb_std_{w}"] = g["REB"].transform(lambda s: s.rolling(w, min_periods=2).std().shift(1))
        base[f"roll_oreb_mean_{w}"] = g["OREB"].transform(lambda s: s.rolling(w, min_periods=1).mean().shift(1))
        base[f"roll_dreb_mean_{w}"] = g["DREB"].transform(lambda s: s.rolling(w, min_periods=1).mean().shift(1))
        base[f"roll_reb_per_min_{w}"] = g["reb_per_min_actual"].transform(
            lambda s: s.rolling(w, min_periods=1).mean().shift(1)
        )
        base[f"roll_min_mean_{w}"] = g["MIN"].transform(lambda s: s.rolling(w, min_periods=1).mean().shift(1))
        base[f"roll_fga_mean_{w}"] = g["FGA"].transform(lambda s: s.rolling(w, min_periods=1).mean().shift(1))
        base[f"roll_fg3a_mean_{w}"] = g["FG3A"].transform(lambda s: s.rolling(w, min_periods=1).mean().shift(1))
        base[f"roll_fg3a_share_mean_{w}"] = g["fg3a_share"].transform(
            lambda s: s.rolling(w, min_periods=1).mean().shift(1)
        )
        base[f"roll_fta_mean_{w}"] = g["FTA"].transform(lambda s: s.rolling(w, min_periods=1).mean().shift(1))

    feature_cols = [
        "consensus_reb_line",
        "max_line",
        "min_line",
        "line_range",
        "n_books",
        "line_spread",
        "spread_signed",
        "spread_abs",
    ]
    for prefix in [
        "roll_reb_mean_",
        "roll_reb_std_",
        "roll_oreb_mean_",
        "roll_dreb_mean_",
        "roll_reb_per_min_",
        "roll_min_mean_",
        "roll_fga_mean_",
        "roll_fg3a_mean_",
        "roll_fg3a_share_mean_",
        "roll_fta_mean_",
    ]:
        feature_cols.extend([c for c in base.columns if c.startswith(prefix)])

    # curated interactions
    interaction_specs = [
        ("consensus_reb_line", "roll_reb_mean_10"),
        ("consensus_reb_line", "n_books"),
        ("roll_reb_mean_10", "line_range"),
        ("roll_reb_per_min_10", "roll_min_mean_10"),
        ("consensus_reb_line", "roll_reb_mean_80"),
        ("roll_reb_per_min_80", "roll_min_mean_80"),
    ]
    for a, b in interaction_specs:
        if a in base.columns and b in base.columns:
            col = f"int_{a}__{b}"
            base[col] = base[a] * base[b]
            feature_cols.append(col)

    # preserve order and uniqueness
    feature_cols = list(dict.fromkeys(feature_cols))
    return base, feature_cols


def prepare_model_matrix(feature_df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Prepare model frame, fill feature NAs with median, and return X/y."""
    required = ["season", "date", "player_normalized", "game_id", "REB", "consensus_reb_line"]
    missing = [c for c in required if c not in feature_df.columns]
    if missing:
        raise ValueError(f"feature_df missing required columns: {missing}")

    ordered_cols = ["season", "date", "player_normalized", "game_id", "REB", "consensus_reb_line"] + feature_cols
    ordered_cols = list(dict.fromkeys(ordered_cols))
    model_df = feature_df[ordered_cols].copy()
    model_df = model_df.dropna(subset=["REB", "consensus_reb_line"]).copy()

    X = model_df[feature_cols].copy()
    med = X.median(numeric_only=True)
    X = X.fillna(med)
    X = X.astype(float)
    y = model_df["REB"].to_numpy(dtype=float)
    return model_df, X, y


def run_univariate_rank(X: pd.DataFrame, y: np.ndarray, feature_cols: list[str]) -> pd.DataFrame:
    """Score one-feature OLS models and return ranking."""
    rows = []
    for f in feature_cols:
        lr = LinearRegression()
        lr.fit(X[[f]], y)
        y_hat = lr.predict(X[[f]])
        m = metrics(y, y_hat)
        rows.append(
            {
                "model": "OLS_UNIVARIATE",
                "features": f,
                "n_features": 1,
                "rmse": m["rmse"],
                "mae": m["mae"],
                "r2": m["r2"],
                "intercept": float(lr.intercept_),
                "coef_summary": f"{f}:{float(lr.coef_[0]):.6f}",
            }
        )
    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def run_forward_selection(X: pd.DataFrame, y: np.ndarray, seed_feature: str, threshold: float) -> list[str]:
    """Greedy forward selection by RMSE improvement."""
    selected = [seed_feature]
    remaining = [f for f in X.columns.tolist() if f != seed_feature]
    lr = LinearRegression()
    lr.fit(X[selected], y)
    best_rmse = metrics(y, lr.predict(X[selected]))["rmse"]

    while True:
        best_feat = None
        best_next_rmse = best_rmse
        for f in remaining:
            trial = selected + [f]
            lr.fit(X[trial], y)
            rmse = metrics(y, lr.predict(X[trial]))["rmse"]
            if rmse < best_next_rmse:
                best_next_rmse = rmse
                best_feat = f
        if best_feat is None:
            break
        gain = best_rmse - best_next_rmse
        if gain < threshold:
            break
        selected.append(best_feat)
        remaining.remove(best_feat)
        best_rmse = best_next_rmse
    return selected


def run_backward_prune(X: pd.DataFrame, y: np.ndarray, features: list[str]) -> list[str]:
    """Backward prune: remove features that do not worsen RMSE."""
    kept = list(features)
    lr = LinearRegression()
    changed = True
    while changed and len(kept) > 1:
        changed = False
        lr.fit(X[kept], y)
        current_rmse = metrics(y, lr.predict(X[kept]))["rmse"]
        for f in list(kept):
            trial = [x for x in kept if x != f]
            lr.fit(X[trial], y)
            trial_rmse = metrics(y, lr.predict(X[trial]))["rmse"]
            if trial_rmse <= current_rmse:
                kept = trial
                changed = True
                break
    return kept


def add_config_result(results: list[dict], model_name: str, feature_names: list[str], y: np.ndarray, y_hat: np.ndarray) -> None:
    """Append model metrics row to results."""
    m = metrics(y, y_hat)
    results.append(
        {
            "model": model_name,
            "n_features": int(len(feature_names)),
            "features": ",".join(feature_names),
            "rmse": m["rmse"],
            "mae": m["mae"],
            "r2": m["r2"],
        }
    )


def run_model_sweep(X: pd.DataFrame, y: np.ndarray, feature_cols: list[str], run_bruteforce: bool, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run baseline + model sweep and return leaderboard and univariate rank."""
    np.random.seed(seed)
    results: list[dict] = []

    # baseline consensus-only
    baseline_features = ["consensus_reb_line"]
    baseline_pred = X["consensus_reb_line"].to_numpy(dtype=float)
    add_config_result(results, "BASELINE_CONSENSUS", baseline_features, y, baseline_pred)
    baseline_rmse = metrics(y, baseline_pred)["rmse"]

    univariate = run_univariate_rank(X=X, y=y, feature_cols=feature_cols)

    # full OLS
    lr = LinearRegression()
    lr.fit(X[feature_cols], y)
    add_config_result(results, "OLS_FULL", feature_cols, y, lr.predict(X[feature_cols]))

    # forward + prune
    selected = run_forward_selection(X=X, y=y, seed_feature="consensus_reb_line", threshold=0.005)
    lr.fit(X[selected], y)
    add_config_result(results, "OLS_FORWARD", selected, y, lr.predict(X[selected]))
    pruned = run_backward_prune(X=X, y=y, features=selected)
    lr.fit(X[pruned], y)
    add_config_result(results, "OLS_PRUNED", pruned, y, lr.predict(X[pruned]))

    # ridge grid
    for alpha in [0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]:
        ridge = Ridge(alpha=alpha, random_state=seed)
        ridge.fit(X[feature_cols], y)
        add_config_result(results, f"RIDGE_a{alpha}", feature_cols, y, ridge.predict(X[feature_cols]))

    # elastic net grid
    for alpha in [0.001, 0.01, 0.1, 1.0]:
        for l1_ratio in [0.2, 0.5, 0.8]:
            en = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=seed, max_iter=10000)
            en.fit(X[feature_cols], y)
            add_config_result(
                results,
                f"ELASTICNET_a{alpha}_l1{l1_ratio}",
                feature_cols,
                y,
                en.predict(X[feature_cols]),
            )

    # xgboost (if available)
    try:
        from xgboost import XGBRegressor

        xgb = XGBRegressor(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=4,
        )
        xgb.fit(X[feature_cols], y)
        add_config_result(results, "XGBOOST_MD4", feature_cols, y, xgb.predict(X[feature_cols]))
    except Exception:
        pass

    # lightgbm (if available)
    try:
        from lightgbm import LGBMRegressor

        lgb = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
        )
        lgb.fit(X[feature_cols], y)
        add_config_result(results, "LIGHTGBM", feature_cols, y, lgb.predict(X[feature_cols]))
    except Exception:
        pass

    # optional brute force OLS combos
    if run_bruteforce:
        top_features = univariate.head(15)["features"].tolist()
        for k in [2, 3, 4]:
            for combo in combinations(top_features, k):
                cols = list(combo)
                lr.fit(X[cols], y)
                add_config_result(results, f"OLS_COMBO_{k}", cols, y, lr.predict(X[cols]))

    leaderboard = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
    leaderboard["rmse_delta_vs_baseline"] = baseline_rmse - leaderboard["rmse"]
    leaderboard["rmse_delta_pct_vs_baseline"] = 100.0 * leaderboard["rmse_delta_vs_baseline"] / baseline_rmse
    return leaderboard, univariate


def write_outputs(
    output_dir: Path,
    leaderboard: pd.DataFrame,
    univariate: pd.DataFrame,
    model_df: pd.DataFrame,
    X: pd.DataFrame,
    y: np.ndarray,
) -> None:
    """Write output artifacts and print concise summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    top20 = leaderboard.head(20).copy()
    champion = top20.iloc[0]

    leaderboard.to_csv(output_dir / "leaderboard_all.csv", index=False)
    top20.to_csv(output_dir / "leaderboard_top20.csv", index=False)
    univariate.to_csv(output_dir / "univariate_rank.csv", index=False)

    # champion predictions
    champion_features = champion["features"].split(",")
    if champion["model"] == "BASELINE_CONSENSUS":
        y_hat = X["consensus_reb_line"].to_numpy(dtype=float)
    else:
        lr = LinearRegression()
        lr.fit(X[champion_features], y)
        y_hat = lr.predict(X[champion_features])
    pred = model_df[["season", "date", "player_normalized", "game_id", "REB"]].copy()
    pred["y_hat"] = y_hat
    pred["resid"] = pred["REB"] - pred["y_hat"]
    pred.to_parquet(output_dir / "champion_predictions.parquet", index=False)

    print(f"rows={len(model_df)}")
    print(f"top20_saved={output_dir / 'leaderboard_top20.csv'}")
    print(f"univariate_saved={output_dir / 'univariate_rank.csv'}")
    print(f"champion_predictions_saved={output_dir / 'champion_predictions.parquet'}")
    print("champion")
    print(champion.to_string())


def main() -> None:
    """Run full overfit-first in-sample sweep."""
    args = parse_args()
    tmp_dir = Path(args.tmp_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    props, logs, v6 = load_cached_artifacts(tmp_dir=tmp_dir)
    main_df = build_main(props=props, logs=logs)
    panel = build_panel(main=main_df, v6=v6)
    feature_df, feature_cols = build_feature_table(panel=panel)
    model_df, X, y = prepare_model_matrix(feature_df=feature_df, feature_cols=feature_cols)
    leaderboard, univariate = run_model_sweep(
        X=X,
        y=y,
        feature_cols=feature_cols,
        run_bruteforce=args.run_bruteforce,
        seed=args.seed,
    )
    write_outputs(
        output_dir=output_dir,
        leaderboard=leaderboard,
        univariate=univariate,
        model_df=model_df,
        X=X,
        y=y,
    )


if __name__ == "__main__":
    main()
