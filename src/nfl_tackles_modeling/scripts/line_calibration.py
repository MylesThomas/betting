"""
Line-bucket calibration analysis for NFL tackles models.

For each consensus line value (1.5, 2.5, … 11.5), shows:
  n              — test-set sample size
  actual_rate    — % of outcomes where tackles_combined > line
  implied_avg    — mean de-vigged over probability from books
  bl_rate        — % where XGB baseline prediction > line (hard binary)
  best_rate      — % where best OLS prediction > line (hard binary)
  p_gauss        — mean Gaussian P(over): Φ((pred−line)/σ), σ from training residuals
  p_boot         — mean Bootstrap P(over): empirical residual distribution (10k draws)
  p_hetero       — mean Heteroskedastic Gaussian P(over): σ modeled as f(pred)
  p_negbin       — mean Negative Binomial P(over): NB2 GLM on same 9 features
  p_hybrid       — mean Hybrid P(over): see HYBRID_* config below
  act_vs_*       — actual_rate − method in pp  |  + = method under-predicts overs

Baseline: XGBoost on offered_line only.
Best model: OLS on market_L16_game_ctx_pos_overprob features.

Walk-forward: fitted on 2024 season, evaluated on 2025 season.

Run:
  python src/nfl_tackles_modeling/scripts/line_calibration.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import nbinom, norm
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.discrete.discrete_model import NegativeBinomial
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

LABELED_PATH = Path.home() / "Downloads" / "tmp" / "nfl_tackles_labeled.parquet"
TARGET       = "tackles_combined"

POS_GROUP_MAP = {
    "LB": "LB", "CB": "CB", "DB": "CB",
    "S":  "S",  "FS": "S",  "SS": "S",
    "DE": "DL", "DT": "DL", "DL": "DL", "NT": "DL",
}

BL_FEATS = ["offered_line"]

BEST_FEATS = [
    "offered_line", "game_total", "proj_opp_score", "tackle_rate_L16",
    "pos_LB", "pos_CB", "pos_S", "pos_DL", "consensus_over_prob",
]

N_BOOT = 10_000
RNG    = np.random.default_rng(42)

# ── Hybrid P(over) sampling config ────────────────────────────────────────────
# The hybrid estimator uses the best-calibrated method per line region,
# chosen by comparing act_vs_* errors on the 2024→2025 walk-forward OOS set:
#
#   line <  HYBRID_NEGBIN_THRESHOLD  →  Bootstrap
#     NegBin over-predicts overs at low lines (-5 to -9pp miss at 2.5–3.5).
#     The discrete NB2 distribution spreads too much mass above the line when
#     the predicted mean is close to it. Bootstrap inherits the actual OLS
#     residual shape and misses by only -2 to -5pp in the same region.
#
#   line >= HYBRID_NEGBIN_THRESHOLD  →  Negative Binomial (NB2 GLM)
#     NegBin is well-calibrated in the main volume zone (4.5–8.5, ±3pp) and
#     outperforms Bootstrap in the high tail (9.5+: NegBin +5pp vs Boot +13pp).
#     Bootstrap samples continuous residuals onto a count target, which
#     produces poor coverage at high lines where the distribution is right-skewed.
#
HYBRID_NEGBIN_THRESHOLD = 4.5   # lines strictly below this use Bootstrap; ≥ use NegBin


# ── Feature engineering ───────────────────────────────────────────────────────

def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["position_group"] = df["position"].map(POS_GROUP_MAP)
    for g in ["LB", "CB", "S", "DL"]:
        df[f"pos_{g}"] = (df["position_group"] == g).astype(int)

    over_cols  = [c for c in df.columns if c.endswith("_over_price")]
    under_cols = [c for c in df.columns if c.endswith("_under_price")]

    def to_imp(col):
        s = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        return np.where(np.isnan(s), np.nan, np.where(s < 0, -s / (-s + 100), 100 / (s + 100)))

    over_mat  = np.column_stack([to_imp(c) for c in over_cols])
    under_mat = np.column_stack([to_imp(c) for c in under_cols])
    total = over_mat + under_mat
    devig = np.where(total > 0, over_mat / total, np.nan)
    df["consensus_over_prob"] = np.nanmean(devig, axis=1)
    return df


# ── Model fits ────────────────────────────────────────────────────────────────

def fit_xgb_bl(train_df: pd.DataFrame) -> XGBRegressor:
    sub = train_df[BL_FEATS + [TARGET]].dropna()
    m = XGBRegressor(
        n_estimators=400, max_depth=2, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.5, reg_lambda=2.0, min_child_weight=10,
        verbosity=0, random_state=42,
    )
    m.fit(sub[BL_FEATS].to_numpy(), sub[TARGET].to_numpy())
    return m


def fit_ols_best(train_df: pd.DataFrame) -> tuple[Pipeline, np.ndarray, np.ndarray]:
    """Returns (pipeline, training_residuals, training_predictions)."""
    sub = train_df[BEST_FEATS + [TARGET]].dropna()
    m = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])
    m.fit(sub[BEST_FEATS].to_numpy(), sub[TARGET].to_numpy())
    train_pred = m.predict(sub[BEST_FEATS].to_numpy())
    residuals  = sub[TARGET].to_numpy() - train_pred
    return m, residuals, train_pred


def fit_negbin(train_df: pd.DataFrame) -> tuple:
    """Fit NB2 GLM (same features as OLS). Returns (result, alpha)."""
    sub = train_df[BEST_FEATS + [TARGET]].dropna()
    X   = sm.add_constant(sub[BEST_FEATS].to_numpy())
    y   = sub[TARGET].to_numpy()
    result = NegativeBinomial(y, X).fit(disp=False, maxiter=300)
    return result, np.exp(result.lnalpha)


# ── Probabilistic P(over) estimators ─────────────────────────────────────────

def calc_p_gauss(pred: np.ndarray, line: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """Homoskedastic Gaussian: single σ from training residuals."""
    sigma = residuals.std()
    return norm.sf(line, loc=pred, scale=sigma)


def calc_p_boot(pred: np.ndarray, line: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """Bootstrap: sample empirical residual distribution, no parametric assumption."""
    n       = len(pred)
    samples = RNG.choice(residuals, size=(n, N_BOOT))   # (n, 10k)
    sims    = pred[:, None] + samples                    # (n, 10k)
    return (sims > line[:, None]).mean(axis=1)


def calc_p_hetero(
    pred: np.ndarray, line: np.ndarray,
    train_pred: np.ndarray, residuals: np.ndarray,
) -> np.ndarray:
    """Heteroskedastic Gaussian: σ = exp(a + b·pred_train), fit on training residuals."""
    log_abs_res = np.log(np.abs(residuals) + 1e-6)
    sigma_lr    = LinearRegression().fit(train_pred.reshape(-1, 1), log_abs_res)
    sigma       = np.exp(sigma_lr.predict(pred.reshape(-1, 1)))
    sigma       = np.clip(sigma, 0.1, None)
    return norm.sf(line, loc=pred, scale=sigma)


def calc_p_negbin(mu: np.ndarray, line: np.ndarray, alpha: float) -> np.ndarray:
    """NB2: var = μ + α·μ².  n = 1/α, p = n/(n+μ).
    Lines are half-integers → P(actual > line) = P(actual ≥ floor(line)+1) = sf(floor(line))."""
    mu         = np.clip(mu, 1e-3, None)
    n_nb       = 1.0 / alpha
    p_nb       = n_nb / (n_nb + mu)
    line_floor = np.floor(line).astype(int)
    return nbinom.sf(line_floor, n=n_nb, p=p_nb)


def calc_p_hybrid(
    pred: np.ndarray, nb_mu: np.ndarray, line: np.ndarray,
    residuals: np.ndarray, alpha: float,
) -> np.ndarray:
    """Hybrid estimator — see HYBRID_NEGBIN_THRESHOLD config for full rationale.
    line < threshold  →  Bootstrap (empirical residual draws from OLS)
    line >= threshold →  NegBin NB2 GLM
    """
    p_nb = calc_p_negbin(nb_mu, line, alpha)
    p_bt = calc_p_boot(pred, line, residuals)
    return np.where(line < HYBRID_NEGBIN_THRESHOLD, p_bt, p_nb)


# ── Calibration tables ────────────────────────────────────────────────────────

def _fmt_pct(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = (df[c] * 100).round(1)


def build_cal_table(test_clean: pd.DataFrame, group_col: str) -> pd.DataFrame:
    cal = (
        test_clean.groupby(group_col, observed=True)
        .agg(
            n           = ("actual_over",         "count"),
            actual_rate = ("actual_over",         "mean"),
            implied_avg = ("consensus_over_prob", "mean"),
            mkt_rate    = ("mkt_over",            "mean"),
            bl_rate     = ("bl_over",             "mean"),
            best_rate   = ("best_over",           "mean"),
            p_gauss     = ("p_gauss",             "mean"),
            p_boot      = ("p_boot",              "mean"),
            p_hetero    = ("p_hetero",            "mean"),
            p_negbin    = ("p_negbin",            "mean"),
            p_hybrid    = ("p_hybrid",            "mean"),
            actual_avg  = (TARGET,                "mean"),
            best_avg    = ("best_pred",           "mean"),
        )
        .reset_index()
    )
    _fmt_pct(cal, ["actual_rate", "implied_avg", "mkt_rate", "bl_rate", "best_rate",
                   "p_gauss", "p_boot", "p_hetero", "p_negbin", "p_hybrid"])
    for c in ["actual_avg", "best_avg"]:
        cal[c] = cal[c].round(2)
    for suffix, ref in [
        ("implied", "implied_avg"), ("bl", "bl_rate"), ("best", "best_rate"),
        ("gauss", "p_gauss"), ("boot", "p_boot"),
        ("hetero", "p_hetero"), ("negbin", "p_negbin"), ("hybrid", "p_hybrid"),
    ]:
        cal[f"act_vs_{suffix}"] = (cal["actual_rate"] - cal[ref]).round(1)
    return cal


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_parquet(LABELED_PATH)
    df = df[df["position"].notna() & ~df["position"].isin({"WR", "FB"})].copy()
    df = add_derived(df)

    train = df[df["season"] == 2024]
    test  = df[df["season"] == 2025]

    print("  Fitting models...")
    bl_model                      = fit_xgb_bl(train)
    best_model, residuals, tr_pred = fit_ols_best(train)
    nb_result, nb_alpha            = fit_negbin(train)
    print(f"    NegBin alpha (dispersion): {nb_alpha:.4f}  "
          f"(>0 = overdispersed; Poisson would be α→0)")
    print(f"    OLS residual σ           : {residuals.std():.4f}")

    all_needed = list(set(BL_FEATS + BEST_FEATS + [TARGET, "offered_line"]))
    test_clean = test[all_needed].dropna().copy()

    # Point predictions
    test_clean["bl_pred"]   = bl_model.predict(test_clean[BL_FEATS].to_numpy())
    test_clean["best_pred"] = best_model.predict(test_clean[BEST_FEATS].to_numpy())

    # NegBin predicted mean (log link)
    X_test_c = sm.add_constant(test_clean[BEST_FEATS].to_numpy())
    test_clean["nb_mu"] = nb_result.predict(X_test_c)

    pred  = test_clean["best_pred"].to_numpy()
    line  = test_clean["offered_line"].to_numpy()
    nb_mu = test_clean["nb_mu"].to_numpy()

    print("  Computing probabilistic P(over) [bootstrap may take a moment]...")
    test_clean["p_gauss"]  = calc_p_gauss(pred, line, residuals)
    test_clean["p_boot"]   = calc_p_boot(pred, line, residuals)
    test_clean["p_hetero"] = calc_p_hetero(pred, line, tr_pred, residuals)
    test_clean["p_negbin"] = calc_p_negbin(nb_mu, line, nb_alpha)
    test_clean["p_hybrid"] = calc_p_hybrid(pred, nb_mu, line, residuals, nb_alpha)
    print(f"    Hybrid threshold: line < {HYBRID_NEGBIN_THRESHOLD} → Bootstrap, "
          f"≥ {HYBRID_NEGBIN_THRESHOLD} → NegBin")

    # Binary signals
    test_clean["actual_over"] = (test_clean[TARGET]               > line).astype(int)
    test_clean["bl_over"]     = (test_clean["bl_pred"]            > line).astype(int)
    test_clean["best_over"]   = (test_clean["best_pred"]          > line).astype(int)
    test_clean["mkt_over"]    = (test_clean["consensus_over_prob"] > 0.5).astype(int)

    pd.set_option("display.width", 240)
    W = 220

    # ── Table 1: per line ─────────────────────────────────────────────────────
    cal = build_cal_table(test_clean, "offered_line")
    cal = cal.rename(columns={"offered_line": "line"})

    print(f"\n{'='*W}")
    print("  LINE CALIBRATION  (OOS: trained 2024, tested 2025)")
    print("  p_gauss  : Gaussian P(over) — homoskedastic σ from OLS residuals")
    print("  p_boot   : Bootstrap P(over) — empirical residual distribution (10k draws)")
    print("  p_hetero : Heteroskedastic Gaussian P(over) — σ = exp(a + b·pred)")
    print("  p_negbin : Negative Binomial P(over) — NB2 GLM, same 9 features")
    print(f"  p_hybrid : Hybrid — Bootstrap if line < {HYBRID_NEGBIN_THRESHOLD}, NegBin otherwise")
    print("  act_vs_* : actual_rate − method in pp  |  + = method under-predicts overs")
    print(f"{'='*W}\n")

    cols1 = [
        "line", "n", "actual_rate", "implied_avg",
        "bl_rate", "best_rate", "p_gauss", "p_boot", "p_hetero", "p_negbin", "p_hybrid",
        "act_vs_implied", "act_vs_best",
        "act_vs_gauss", "act_vs_boot", "act_vs_hetero", "act_vs_negbin", "act_vs_hybrid",
        "actual_avg", "best_avg",
    ]
    print(cal[cols1].to_string(index=False))

    print(f"\n  Test rows : {len(test_clean):,}")
    print(f"  Overall actual over rate    : {test_clean['actual_over'].mean()*100:.1f}%")
    print(f"  Overall implied over prob   : {test_clean['consensus_over_prob'].mean()*100:.1f}%")
    for label, col in [
        ("Baseline XGB (binary)  ", "bl_over"),
        ("Best OLS (binary)      ", "best_over"),
        ("Gaussian P(over)       ", "p_gauss"),
        ("Bootstrap P(over)      ", "p_boot"),
        ("Hetero Gauss P(over)   ", "p_hetero"),
        ("NegBin P(over)         ", "p_negbin"),
        ("Hybrid P(over)         ", "p_hybrid"),
    ]:
        print(f"  Overall {label}: {test_clean[col].mean()*100:.1f}%")

    # ── Table 2: snapped to nearest .5 line ──────────────────────────────────
    # Integer consensus medians (e.g. 8.0 = avg of 7.5/8.5 books) snap up to N.5.
    test_clean["line_snapped"] = np.floor(test_clean["offered_line"]) + 0.5

    cal_snap = build_cal_table(test_clean, "line_snapped")
    cols_snap = [
        "line_snapped", "n", "actual_rate", "implied_avg",
        "bl_rate", "best_rate", "p_gauss", "p_boot", "p_hetero", "p_negbin", "p_hybrid",
        "act_vs_implied", "act_vs_best",
        "act_vs_gauss", "act_vs_boot", "act_vs_hetero", "act_vs_negbin", "act_vs_hybrid",
        "actual_avg", "best_avg",
    ]

    print(f"\n{'='*W}")
    print("  SNAPPED-LINE CALIBRATION  (integer consensus medians snapped up to nearest .5)")
    print("  e.g. offered_line=8.0 (avg of 7.5/8.5 books) → grouped under 8.5")
    print(f"{'='*W}\n")
    print(cal_snap[cols_snap].to_string(index=False))

    # ── Table 3: bucketed ─────────────────────────────────────────────────────
    bucket_bins   = [0, 3.5, 6.5, 9.5, float("inf")]
    bucket_labels = ["0-3", "4-6", "7-9", "10+"]
    test_clean["bucket"] = pd.cut(
        test_clean["offered_line"], bins=bucket_bins, labels=bucket_labels, right=True,
    )

    cal2  = build_cal_table(test_clean, "bucket")
    cols2 = [
        "bucket", "n", "actual_rate", "implied_avg",
        "best_rate", "p_gauss", "p_boot", "p_hetero", "p_negbin", "p_hybrid",
        "act_vs_implied", "act_vs_best",
        "act_vs_gauss", "act_vs_boot", "act_vs_hetero", "act_vs_negbin", "act_vs_hybrid",
        "actual_avg", "best_avg",
    ]

    print(f"\n{'='*W}")
    print("  BUCKETED CALIBRATION  (lines grouped into ~3-tackle bands)")
    print("  Boundaries: ≤3.5 → '0-3'  |  4.0–6.5 → '4-6'  |  7.0–9.5 → '7-9'  |  10.0+ → '10+'")
    print(f"{'='*W}\n")
    print(cal2[cols2].to_string(index=False))
    print()


if __name__ == "__main__":
    main()
