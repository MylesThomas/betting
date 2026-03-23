"""
OOS regression sweep: alternate feature specs without min_line+max_line together (except R0).

Context:
- Builds line_mid, line_width (delta), width_eq_0, line_width_sq from min_line/max_line in-memory.
- For each test season, trains on other seasons; reports RMSE, MAE, R2 for OLS and XGB.
- Optional: --v3 + under_only PnL (same Option A as v4/v5: roll_reb_std_5 σ, shrink=0, min_edge list).
- Optional: bootstrap (Monte Carlo) CIs on total PnL and mean ROI over placed bets (resample bets w/ replacement).

Usage:
    python src/nba_rebounds_modeling/00_research/scripts/v7_spec_sweep_rebounds_oos.py \\
        --feat ~/Downloads/tmp/rebounds_model_features_v2.parquet \\
        --v3 ~/Downloads/tmp/v3_rebounds_props_raw.parquet \\
        --out-csv ~/Downloads/tmp/rebounds_spec_sweep_oos.csv \\
        --out-pnl-csv ~/Downloads/tmp/rebounds_spec_sweep_pnl_mc.csv \\
        --mc-samples 2000
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm


def ensure_repo_root_on_syspath() -> Path:
    current = Path.cwd().resolve()
    while True:
        if (current / ".gitignore").exists() and (current / "src").exists():
            if str(current) not in sys.path:
                sys.path.insert(0, str(current))
            return current
        if current.parent == current:
            raise FileNotFoundError("Could not locate repo root")
        current = current.parent


ensure_repo_root_on_syspath()

TARGET = "REB"
GROUP_KEYS = ["season", "date", "player_normalized", "game_id"]
ROLL_FEATS = ["spread_signed", "roll_reb_mean_60", "roll_fg3a_mean_20", "roll_reb_std_5"]
DEFAULT_TEST_SEASONS = ["2023-24", "2024-25", "2025-26"]

DERIVED = ["line_mid", "line_width", "width_eq_0", "line_width_sq"]


def augment_line_features(feat: pd.DataFrame) -> pd.DataFrame:
    out = feat.copy()
    mn = out["min_line"].astype(float)
    mx = out["max_line"].astype(float)
    delta = mx - mn
    if (delta < -1e-9).any():
        bad = int((delta < -1e-9).sum())
        raise ValueError(f"max_line < min_line for {bad} rows; fix upstream data")
    out["line_width"] = delta
    out["line_mid"] = (mn + mx) / 2.0
    out["width_eq_0"] = (delta <= 1e-9).astype(np.float64)
    out["line_width_sq"] = out["line_width"].astype(np.float64) ** 2
    return out


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    err = y_true.astype(float) - y_pred.astype(float)
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    yt = y_true.astype(float)
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return rmse, mae, r2


def fit_predict_ols(train_m: pd.DataFrame, test_m: pd.DataFrame, cols: list[str]) -> np.ndarray:
    X_tr = sm.add_constant(train_m[cols].astype(float), has_constant="add")
    y_tr = train_m[TARGET].astype(float)
    m = sm.OLS(y_tr, X_tr).fit()
    X_te = sm.add_constant(test_m[cols].astype(float), has_constant="add")
    return m.predict(X_te).to_numpy()


def fit_predict_xgb(train_m: pd.DataFrame, test_m: pd.DataFrame, cols: list[str]) -> np.ndarray:
    import xgboost as xgb

    X_tr = train_m[cols].astype(float)
    y_tr = train_m[TARGET].astype(float)
    X_te = test_m[cols].astype(float)
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        reg_alpha=0.5,
        random_state=69,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_tr, y_tr)
    return model.predict(X_te)


def build_spec_table() -> dict[str, list[str]]:
    r = ROLL_FEATS
    return {
        "R0": ["min_line", "max_line", *r],
        "P1": ["line_mid", "line_width", *r],
        "P2": ["min_line", "line_width", *r],
        "P3": ["max_line", "line_width", *r],
        "A1": ["line_mid", *r],
        "A2": ["min_line", *r],
        "A3": ["max_line", *r],
        "E1": ["line_mid", "line_width", "width_eq_0", *r],
        "E2": ["line_mid", "line_width", "line_width_sq", *r],
    }


def load_v5_module():
    here = Path(__file__).resolve().parent
    path = here / "v5_compare_rebounds_models_oos.py"
    spec = importlib.util.spec_from_file_location("v5_compare_rebounds_models_oos", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def collect_under_only_bet_pnls(
    base: pd.DataFrame,
    min_edge: float,
    sigma_col: str,
    sigma_floor: float,
    shrinkage: float,
    american_profit_on_win,
) -> np.ndarray:
    """Unit PnL per placed under bet (same rules as v5 under_only_pnl_rows)."""
    sigma = base[sigma_col].astype(float).clip(lower=sigma_floor).to_numpy()
    consensus = base["consensus_reb_line"].astype(float).to_numpy()
    line = base["line"].astype(float).to_numpy()
    reb = base["REB"].astype(float).to_numpy()
    yhat_arr = base["yhat"].astype(float).to_numpy()
    p_nov_u = base["p_under_novig"].astype(float).to_numpy()
    under_odds = base["under_odds"].astype(float).to_numpy()

    mean_adj = consensus + (1.0 - shrinkage) * (yhat_arr - consensus)
    z = (line - mean_adj) / sigma
    p_under = norm.cdf(z)
    edge_u = p_under - p_nov_u

    pnls: list[float] = []
    for i in range(len(base)):
        if edge_u[i] <= min_edge:
            continue
        if reb[i] == line[i]:
            continue
        odds_am = float(under_odds[i])
        won = reb[i] < line[i]
        if won:
            pnls.append(float(american_profit_on_win(odds_am)))
        else:
            pnls.append(-1.0)
    return np.array(pnls, dtype=np.float64)


def bootstrap_pnl_roi_quantiles(
    pnls: np.ndarray,
    n_mc: int,
    seed: int,
) -> dict[str, float]:
    """Resample placed bets with replacement; percentiles for sum(PnL) and mean(PnL)=ROI."""
    out = {
        "mc_sum_p025": float("nan"),
        "mc_sum_p500": float("nan"),
        "mc_sum_p975": float("nan"),
        "mc_roi_p025": float("nan"),
        "mc_roi_p500": float("nan"),
        "mc_roi_p975": float("nan"),
    }
    if n_mc <= 0 or len(pnls) == 0:
        return out
    rng = np.random.default_rng(seed)
    n = len(pnls)
    idx = rng.integers(0, n, size=(n_mc, n))
    sums = pnls[idx].sum(axis=1).astype(np.float64)
    rois = pnls[idx].mean(axis=1).astype(np.float64)
    out["mc_sum_p025"] = float(np.percentile(sums, 2.5))
    out["mc_sum_p500"] = float(np.percentile(sums, 50.0))
    out["mc_sum_p975"] = float(np.percentile(sums, 97.5))
    out["mc_roi_p025"] = float(np.percentile(rois, 2.5))
    out["mc_roi_p500"] = float(np.percentile(rois, 50.0))
    out["mc_roi_p975"] = float(np.percentile(rois, 97.5))
    return out


def validate_specs(specs: dict[str, list[str]]) -> None:
    for name, cols in specs.items():
        has_min = "min_line" in cols
        has_max = "max_line" in cols
        if has_min and has_max:
            if name != "R0":
                raise ValueError(f"spec {name} must not include both min_line and max_line (except R0)")
        for c in cols:
            if c not in DERIVED and c not in ("min_line", "max_line") and c not in ROLL_FEATS:
                raise ValueError(f"spec {name}: unknown column {c!r}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OOS spec sweep: rebounds REB ~ feature sets.")
    p.add_argument("--feat", type=str, default="~/Downloads/tmp/rebounds_model_features_v2.parquet")
    p.add_argument(
        "--out-csv",
        type=str,
        default="",
        help="Write long results CSV (all specs × models × seasons)",
    )
    p.add_argument(
        "--specs",
        type=str,
        default="R0,P1,P2,P3,A1,A2,A3,E1,E2",
        help="Comma-separated spec ids",
    )
    p.add_argument(
        "--models",
        type=str,
        default="ols,xgb",
        help="Comma-separated: ols, xgb",
    )
    p.add_argument(
        "--test-seasons",
        type=str,
        default=",".join(DEFAULT_TEST_SEASONS),
    )
    p.add_argument(
        "--v3",
        type=str,
        default="",
        help="v3 props parquet; if set, compute under_only PnL per spec/model/season",
    )
    p.add_argument(
        "--min-edges",
        type=str,
        default="0.05,0.10",
        help="Comma-separated min_edge thresholds for under_only",
    )
    p.add_argument(
        "--mc-samples",
        type=int,
        default=2000,
        help="Bootstrap draws over placed bets (0 to skip MC quantiles)",
    )
    p.add_argument(
        "--mc-seed",
        type=int,
        default=69,
        help="RNG seed for bootstrap",
    )
    p.add_argument(
        "--out-pnl-csv",
        type=str,
        default="",
        help="Write PnL + MC quantiles CSV (defaults next to --out-csv if --v3 set)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    feat_path = Path(args.feat).expanduser()
    test_seasons = [s.strip() for s in args.test_seasons.split(",") if s.strip()]
    spec_ids = [s.strip().upper() for s in args.specs.split(",") if s.strip()]
    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    min_edges = [float(x.strip()) for x in args.min_edges.split(",") if x.strip()]

    for m in models:
        if m not in ("ols", "xgb"):
            raise ValueError(f"Unknown model {m!r}; use ols or xgb")

    all_specs = build_spec_table()
    validate_specs(all_specs)

    for sid in spec_ids:
        if sid not in all_specs:
            raise ValueError(f"Unknown spec {sid!r}; choose from {list(all_specs)}")

    v5 = None
    v3 = None
    v3_path: Path | None = None
    if args.v3:
        v3_path = Path(args.v3).expanduser()
        v5 = load_v5_module()
        v3 = pd.read_parquet(v3_path)
        for c in [
            "season",
            "date",
            "player_normalized",
            "game_id",
            "line",
            "REB",
            "consensus_reb_line",
            "p_under_novig",
            "under_odds",
        ]:
            if c not in v3.columns:
                raise ValueError(f"v3 parquet missing column: {c}")

    base_cols = ["min_line", "max_line", *ROLL_FEATS, TARGET, "consensus_reb_line", *GROUP_KEYS]
    feat = pd.read_parquet(feat_path)
    for c in base_cols:
        if c not in feat.columns:
            raise ValueError(f"feat parquet missing column: {c}")

    feat = augment_line_features(feat)

    rows: list[dict] = []
    pnl_rows: list[dict] = []
    for ts in test_seasons:
        if ts not in feat["season"].unique():
            raise ValueError(f"test_season {ts!r} not in feat['season']")

    for ts in test_seasons:
        train = feat[feat["season"] != ts].copy()
        feat_test = feat[feat["season"] == ts].copy()
        train_seasons_str = ",".join(sorted(s for s in feat["season"].unique() if s != ts))

        for sid in spec_ids:
            cols = all_specs[sid]
            cols_needed = cols + [TARGET]
            train_m = train.dropna(subset=cols_needed)
            test_m = feat_test.dropna(subset=cols_needed + GROUP_KEYS)

            if len(train_m) < 100:
                raise ValueError(f"train_m too small: test_season={ts} spec={sid}")

            y_true = test_m[TARGET].to_numpy(dtype=float)

            for model_name in models:
                if model_name == "ols":
                    yhat = fit_predict_ols(train_m, test_m, cols)
                else:
                    yhat = fit_predict_xgb(train_m, test_m, cols)

                rmse, mae, r2 = regression_metrics(y_true, yhat)
                rows.append({
                    "spec": sid,
                    "model": model_name,
                    "test_season": ts,
                    "n_train": int(len(train_m)),
                    "n_test": int(len(test_m)),
                    "n_features": len(cols),
                    "feature_list": ",".join(cols),
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2,
                })

                if v3 is not None and v5 is not None:
                    pred = test_m[v5.GROUP_KEYS].copy()
                    pred["yhat"] = yhat
                    pred[v5.SIGMA_COL] = test_m[v5.SIGMA_COL].to_numpy()
                    v3_test = v3[v3["season"] == ts].copy()
                    base = v3_test.merge(pred, on=v5.GROUP_KEYS, how="inner")
                    tag = f"{sid}_{model_name}"
                    for me in min_edges:
                        pnls = collect_under_only_bet_pnls(
                            base,
                            me,
                            v5.SIGMA_COL,
                            v5.SIGMA_FLOOR,
                            v5.SHRINKAGE,
                            v5.american_profit_on_win,
                        )
                        n_bets = int(len(pnls))
                        total_pnl = float(pnls.sum()) if n_bets else float("nan")
                        roi = float(pnls.mean()) if n_bets else float("nan")
                        mc_off = zlib.crc32(f"{sid}|{model_name}|{ts}|{me}".encode()) % (2**31)
                        mc = bootstrap_pnl_roi_quantiles(
                            pnls,
                            args.mc_samples,
                            int(args.mc_seed) + mc_off,
                        )
                        pnl_rows.append({
                            "spec": sid,
                            "model": model_name,
                            "policy_tag": tag,
                            "test_season": ts,
                            "train_seasons": train_seasons_str,
                            "min_edge": me,
                            "n_train_rows": int(len(train_m)),
                            "n_test_feat_rows": int(len(test_m)),
                            "n_v3_test_rows": int(len(v3_test)),
                            "n_merged_rows": int(len(base)),
                            "n_bets": n_bets,
                            "total_pnl_u": total_pnl,
                            "roi": roi,
                            **mc,
                        })

    out = pd.DataFrame(rows)

    # ΔRMSE vs R0 + OLS for same test_season
    ref = out[(out["spec"] == "R0") & (out["model"] == "ols")][["test_season", "rmse"]].rename(
        columns={"rmse": "rmse_r0_ols"}
    )
    out = out.merge(ref, on="test_season", how="left")
    out["delta_rmse_vs_r0_ols"] = out["rmse"] - out["rmse_r0_ols"]

    out = out.sort_values(["test_season", "spec", "model"]).reset_index(drop=True)

    print(out.to_string(index=False))

    if args.out_csv:
        outp = Path(args.out_csv).expanduser()
        outp.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(outp, index=False)
        print(f"\nwrote {outp}")

    if v3_path is not None and pnl_rows:
        pnl_df = pd.DataFrame(pnl_rows).sort_values(
            ["test_season", "min_edge", "spec", "model"]
        ).reset_index(drop=True)
        print("\n=== under_only PnL + bootstrap (placed bets) ===")
        print(pnl_df.to_string(index=False))
        pnl_out = Path(args.out_pnl_csv).expanduser() if args.out_pnl_csv else None
        if pnl_out is None and args.out_csv:
            p = Path(args.out_csv).expanduser()
            pnl_out = p.parent / f"{p.stem}_pnl_mc.csv"
        if pnl_out is not None:
            pnl_out.parent.mkdir(parents=True, exist_ok=True)
            pnl_df.to_csv(pnl_out, index=False)
            print(f"\nwrote {pnl_out}")


if __name__ == "__main__":
    main()
