"""
OOS diagnostics: how XGB vs OLS (optional Ridge) differ on yhat, p_under, and min_edge passes.

Context:
- Same parquets and split logic as v5 (train = all seasons except test_season).
- Same under_only math: shrink=0, sigma = roll_reb_std_5 clipped at SIGMA_FLOOR,
  p_under = Phi((line - yhat) / sigma), edge = p_under - p_under_novig.
- Writes CSVs + matplotlib figures to --out-dir for notebook replication.

Usage:
    python src/nba_rebounds_modeling/00_research/scripts/v6_diagnose_xgb_vs_linear_rebounds_oos.py \\
        --feat ~/Downloads/tmp/rebounds_model_features_v2.parquet \\
        --v3 ~/Downloads/tmp/v3_rebounds_props_raw.parquet \\
        --out-dir ~/Downloads/tmp/rebounds_xgb_ols_diag \\
        --min-edge 0.05
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
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


def load_v5_module():
    here = Path(__file__).resolve().parent
    path = here / "v5_compare_rebounds_models_oos.py"
    spec = importlib.util.spec_from_file_location("v5_compare_rebounds_models_oos", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fit_ols_result(train_m: pd.DataFrame, v5):
    X_tr = sm.add_constant(train_m[v5.B_MIN_MAX_FEATS].astype(float), has_constant="add")
    y_tr = train_m[v5.TARGET].astype(float)
    return sm.OLS(y_tr, X_tr).fit()


def fit_xgb_model(train_m: pd.DataFrame, v5):
    import xgboost as xgb

    X_tr = train_m[v5.B_MIN_MAX_FEATS].astype(float)
    y_tr = train_m[v5.TARGET].astype(float)
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
        importance_type="gain",
    )
    model.fit(X_tr, y_tr)
    return model


def p_under_edges(
    consensus: np.ndarray,
    line: np.ndarray,
    sigma: np.ndarray,
    yhat: np.ndarray,
    p_nov_u: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean_adj = consensus + (1.0 - 0.0) * (yhat - consensus)
    z = (line - mean_adj) / sigma
    p_under = norm.cdf(z)
    edge = p_under - p_nov_u
    return p_under, edge


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose XGB vs linear OOS rebounds models.")
    p.add_argument("--feat", type=str, default="~/Downloads/tmp/rebounds_model_features_v2.parquet")
    p.add_argument("--v3", type=str, default="~/Downloads/tmp/v3_rebounds_props_raw.parquet")
    p.add_argument(
        "--test-seasons",
        type=str,
        default=",".join(["2023-24", "2024-25", "2025-26"]),
    )
    p.add_argument("--min-edge", type=float, default=0.05)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--top-dy", type=int, default=25, help="Rows to write per season for largest |yhat_xgb - yhat_ols|")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    v5 = load_v5_module()
    feat_path = Path(args.feat).expanduser()
    v3_path = Path(args.v3).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    test_seasons = [s.strip() for s in args.test_seasons.split(",") if s.strip()]
    min_edge = float(args.min_edge)

    feat = pd.read_parquet(feat_path)
    v3 = pd.read_parquet(v3_path)

    for c in v5.B_MIN_MAX_FEATS + [v5.TARGET, "consensus_reb_line"]:
        if c not in feat.columns:
            raise ValueError(f"feat missing: {c}")

    coef_rows: list[dict] = []
    imp_rows: list[dict] = []
    disag_rows: list[dict] = []
    top_dy_chunks: list[pd.DataFrame] = []
    bin_rows: list[dict] = []
    plot_frames: list[tuple[str, pd.DataFrame]] = []

    for ts in test_seasons:
        if ts not in feat["season"].unique():
            raise ValueError(f"test_season {ts!r} not in feat")

        train = feat[feat["season"] != ts].copy()
        feat_test = feat[feat["season"] == ts].copy()
        cols_needed = v5.B_MIN_MAX_FEATS + [v5.TARGET]
        train_m = train.dropna(subset=cols_needed)
        test_m = feat_test.dropna(subset=cols_needed + v5.GROUP_KEYS)
        if len(train_m) < 100:
            raise ValueError(f"train_m too small: {ts}")

        ols_res = fit_ols_result(train_m, v5)
        coef_row = {"test_season": ts}
        for name in ols_res.params.index:
            coef_row[f"ols_coef__{name}"] = float(ols_res.params[name])
        coef_rows.append(coef_row)

        xgb_model = fit_xgb_model(train_m, v5)
        for j, col in enumerate(v5.B_MIN_MAX_FEATS):
            imp_rows.append({
                "test_season": ts,
                "feature": col,
                "xgb_importance_gain": float(xgb_model.feature_importances_[j]),
            })

        yhat_ols = v5.fit_predict_ols(train_m, test_m)
        yhat_xgb = v5.fit_predict_xgb(train_m, test_m)

        y_true = test_m[v5.TARGET].to_numpy().astype(float)
        dy = yhat_xgb.astype(float) - yhat_ols.astype(float)
        corr = float(np.corrcoef(yhat_ols, yhat_xgb)[0, 1])

        rmse_ols = float(np.sqrt(np.mean((y_true - yhat_ols) ** 2)))
        rmse_xgb = float(np.sqrt(np.mean((y_true - yhat_xgb) ** 2)))

        pred = test_m[v5.GROUP_KEYS].copy()
        pred["yhat_ols"] = yhat_ols
        pred["yhat_xgb"] = yhat_xgb
        pred[v5.SIGMA_COL] = test_m[v5.SIGMA_COL].to_numpy()

        v3_test = v3[v3["season"] == ts].copy()
        base = v3_test.merge(pred, on=v5.GROUP_KEYS, how="inner")

        sigma = base[v5.SIGMA_COL].astype(float).clip(lower=v5.SIGMA_FLOOR).to_numpy()
        consensus = base["consensus_reb_line"].astype(float).to_numpy()
        line = base["line"].astype(float).to_numpy()
        reb = base["REB"].astype(float).to_numpy()
        p_nov_u = base["p_under_novig"].astype(float).to_numpy()

        p_u_ols, edge_ols = p_under_edges(consensus, line, sigma, base["yhat_ols"].to_numpy(), p_nov_u)
        p_u_xgb, edge_xgb = p_under_edges(consensus, line, sigma, base["yhat_xgb"].to_numpy(), p_nov_u)
        dp = p_u_xgb - p_u_ols

        plot_frames.append((ts, base[["yhat_ols", "yhat_xgb"]].assign(dp_under=dp)))

        pass_ols = edge_ols > min_edge
        pass_xgb = edge_xgb > min_edge
        not_push = reb != line

        n_merged = len(base)
        disag_rows.append({
            "test_season": ts,
            "n_test_regression": int(len(test_m)),
            "n_merged": n_merged,
            "corr_yhat_ols_xgb": corr,
            "mean_dy": float(np.mean(dy)),
            "std_dy": float(np.std(dy)),
            "p95_abs_dy": float(np.percentile(np.abs(dy), 95)),
            "rmse_ols": rmse_ols,
            "rmse_xgb": rmse_xgb,
            "mean_dp_under": float(np.mean(dp)),
            "std_dp_under": float(np.std(dp)),
            "edge_pass_ols": int(pass_ols.sum()),
            "edge_pass_xgb": int(pass_xgb.sum()),
            "xgb_only_edge": int((pass_xgb & ~pass_ols).sum()),
            "ols_only_edge": int((pass_ols & ~pass_xgb).sum()),
            "both_edge": int((pass_ols & pass_xgb).sum()),
            "neither_edge": int((~pass_ols & ~pass_xgb).sum()),
            "bet_ols": int((pass_ols & not_push).sum()),
            "bet_xgb": int((pass_xgb & not_push).sum()),
            "bet_xgb_not_ols": int((pass_xgb & ~pass_ols & not_push).sum()),
            "bet_ols_not_xgb": int((pass_ols & ~pass_xgb & not_push).sum()),
        })

        # Top |dy| on regression rows (test_m), merge context from test_m
        tm = test_m.copy()
        tm["yhat_ols"] = yhat_ols
        tm["yhat_xgb"] = yhat_xgb
        tm["dy"] = dy
        tm["abs_dy"] = np.abs(dy)
        top = tm.nlargest(args.top_dy, "abs_dy")[
            v5.GROUP_KEYS
            + v5.B_MIN_MAX_FEATS
            + [v5.TARGET, "consensus_reb_line", "yhat_ols", "yhat_xgb", "dy"]
        ].copy()
        top.insert(0, "test_season", ts)
        top_dy_chunks.append(top)

        # Binned RMSE by posted line (merged base has line)
        line_series = base["line"].astype(float)
        try:
            base["_line_bin"] = pd.qcut(line_series, q=5, duplicates="drop")
        except ValueError:
            base["_line_bin"] = "all"
        for bin_label, sub in base.groupby("_line_bin", observed=True):
            idx = sub.index
            yo = sub["yhat_ols"].to_numpy()
            yx = sub["yhat_xgb"].to_numpy()
            yt = sub["REB"].astype(float).to_numpy()
            bin_rows.append({
                "test_season": ts,
                "line_bin": str(bin_label),
                "n": int(len(sub)),
                "rmse_ols": float(np.sqrt(np.mean((yt - yo) ** 2))),
                "rmse_xgb": float(np.sqrt(np.mean((yt - yx) ** 2))),
            })

    coef_df = pd.DataFrame(coef_rows).sort_values("test_season")
    imp_df = pd.DataFrame(imp_rows).sort_values(["test_season", "xgb_importance_gain"], ascending=[True, False])
    disag_df = pd.DataFrame(disag_rows).sort_values("test_season")
    top_dy_df = pd.concat(top_dy_chunks, ignore_index=True)
    bin_df = pd.DataFrame(bin_rows).sort_values(["test_season", "line_bin"])

    coef_path = out_dir / "ols_coef_by_test_season.csv"
    imp_path = out_dir / "xgb_feature_importance_by_test_season.csv"
    disag_path = out_dir / "disagreement_edge_summary.csv"
    top_dy_path = out_dir / "top_abs_dyhat_rows.csv"
    bin_path = out_dir / "rmse_by_line_quantile_bin.csv"

    coef_df.to_csv(coef_path, index=False)
    imp_df.to_csv(imp_path, index=False)
    disag_df.to_csv(disag_path, index=False)
    top_dy_df.to_csv(top_dy_path, index=False)
    bin_df.to_csv(bin_path, index=False)

    print(f"wrote {coef_path}")
    print(f"wrote {imp_path}")
    print(f"wrote {disag_path}")
    print(f"wrote {top_dy_path}")
    print(f"wrote {bin_path}")

    # --- plots ---
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skip figures", file=sys.stderr)
        plt = None

    if plt is not None:
        for ts, pf in plot_frames:
            yo = pf["yhat_ols"].to_numpy()
            yx = pf["yhat_xgb"].to_numpy()
            dy_m = yx.astype(float) - yo.astype(float)
            dp = pf["dp_under"].to_numpy()

            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            ax0, ax1, ax2 = axes
            ax0.scatter(yo, yx, s=3, alpha=0.25)
            lims = [min(yo.min(), yx.min()), max(yo.max(), yx.max())]
            ax0.plot(lims, lims, "k--", lw=1)
            ax0.set_xlabel("yhat OLS")
            ax0.set_ylabel("yhat XGB")
            ax0.set_title(f"{ts}  yhat scatter (merged v3 rows)")

            ax1.hist(dy_m, bins=60, color="steelblue", edgecolor="white", alpha=0.85)
            ax1.set_title(f"{ts}  dy = yhat_xgb - yhat_ols")
            ax1.set_xlabel("dy")

            ax2.hist(dp, bins=60, color="darkseagreen", edgecolor="white", alpha=0.85)
            ax2.set_title(f"{ts}  dp_under = p_u_xgb - p_u_ols")
            ax2.set_xlabel("dp_under")

            fig.tight_layout()
            fig_path = out_dir / f"fig_{ts.replace('-', '_')}_yhat_dy_punder.png"
            fig.savefig(fig_path, dpi=150)
            plt.close(fig)
            print(f"wrote {fig_path}")

    print("\n=== disagreement + edge flips (min_edge=%.3f) ===" % min_edge)
    print(disag_df.to_string(index=False))

    print("\nDone. Read CSVs/plots from:", out_dir)


if __name__ == "__main__":
    main()
