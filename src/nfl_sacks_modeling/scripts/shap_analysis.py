"""
SHAP feature importance for NFL sacks props model.

Fits LR, XGB, LGBM on full 2024+2025 pooled data (2,927 rows), computes SHAP
values, and generates:
  - Beeswarm plot per model (top 20 features: direction + magnitude distribution)
  - Bar chart per model (mean |SHAP|, top 20)
  - Cross-model ranking table (which features rank where across models)
  - Grouped importance breakdown (market vs. player history vs. game context)

Key question: does prop_median_impl_over carry all the signal, or do rolling
sack rates / snap pct add genuine incremental value on top of the market?

Output: ~/Downloads/tmp/nfl_sacks_shap.html

Run:
    python src/nfl_sacks_modeling/scripts/shap_analysis.py
"""

import base64
import io
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CONFIG_PATH   = Path(__file__).resolve().parents[1] / "config.yaml"
FEATURES_2024 = Path.home() / "Downloads" / "tmp" / "nfl_sacks_features_2024.parquet"
FEATURES_2025 = Path.home() / "Downloads" / "tmp" / "nfl_sacks_features_2025.parquet"
OUT_HTML      = Path.home() / "Downloads" / "tmp" / "nfl_sacks_shap.html"

TOP_N         = 20   # features per beeswarm/bar
MODEL_LABELS  = ["LR", "XGB", "LGBM"]

# ── Feature categories for grouped importance ──────────────────────────────────
FEATURE_CATEGORIES = [
    ("Market (impl prob)",  lambda n: ("impl" in n or "bin" in n) and n.startswith("prop_")),
    ("Market (line/price)", lambda n: ("price" in n or "_line" in n or "book_spread" in n or "n_books" in n) and n.startswith("prop_")),
    ("Per-book odds",       lambda n: any(n.startswith(b) for b in ["fanduel", "betonline", "draftkings"])),
    ("Sack rate (rolling)", lambda n: n.startswith("sack_rate")),
    ("QB hit rate",         lambda n: n.startswith("qbhit_rate")),
    ("Snap pct (rolling)",  lambda n: n.startswith("snap_pct")),
    ("Games played YTD",    lambda n: n == "games_played_ytd"),
    ("Game context",        lambda n: n in ("game_total", "team_spread")),
    ("Position / other",    lambda n: True),   # catch-all
]


# ── Config / features ──────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)["nfl_sacks_model"]


def feature_lists(cfg: dict) -> tuple[list[str], list[str]]:
    windows = cfg["rolling_windows"]
    rolling = [
        f"{feat}_L{('career' if w >= 999 else w)}"
        for feat in ["sack_rate", "qbhit_rate", "snap_pct"]
        for w in windows
    ]
    market_num = [
        "prop_median_impl_over", "prop_median_impl_under",
        "prop_mean_impl_over",   "prop_mean_impl_under",
        "prop_min_impl_over",    "prop_max_impl_over",
        "prop_min_impl_under",   "prop_max_impl_under",
        "prop_book_spread_over", "prop_book_spread_under",
        "prop_n_books",
        "fanduel_over_0p5_implied",
        "betonline_over_0p5_implied", "betonline_under_0p5_implied",
        "draftkings_over_0p25_implied", "draftkings_under_0p25_implied",
    ]
    market_cat = [
        "prop_median_impl_over_bin", "prop_mean_impl_over_bin",
        "prop_median_impl_under_bin", "prop_mean_impl_under_bin",
    ]
    numeric    = rolling + ["game_total", "team_spread", "games_played_ytd"] + market_num
    categorical = ["pos_group", "pos_side"] + market_cat
    return numeric, categorical


# ── Data ───────────────────────────────────────────────────────────────────────

def american_to_implied(price: float) -> float:
    return abs(price) / (abs(price) + 100) if price < 0 else 100 / (price + 100)


def load_data(cfg: dict) -> pd.DataFrame:
    df24 = pd.read_parquet(FEATURES_2024)
    df25 = pd.read_parquet(FEATURES_2025)
    df   = pd.concat([df24, df25], ignore_index=True)
    mask = (
        df["prop_median_price_over"].notna() &
        df["prop_median_price_under"].notna() &
        df["target"].notna()
    )
    df = df[mask].copy()
    impl_over  = df["prop_median_price_over"].apply(american_to_implied)
    impl_under = df["prop_median_price_under"].apply(american_to_implied)
    df["market_prob"] = impl_over / (impl_over + impl_under)
    print(f"  Rows: {len(df):,}  |  Pos rate: {df['target'].mean():.1%}  "
          f"|  Market avg P(Over): {df['market_prob'].mean():.1%}")
    return df.reset_index(drop=True)


# ── Pipelines ──────────────────────────────────────────────────────────────────

def lr_pipeline(num_cols: list, cat_cols: list) -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value=0)),
                          ("sc",  StandardScaler())]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])
    return Pipeline([("pre", pre),
                     ("clf", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs"))])


def xgb_pipeline(num_cols: list, cat_cols: list) -> Pipeline:
    pre = ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])
    return Pipeline([("pre", pre),
                     ("clf", xgb.XGBClassifier(n_estimators=300, max_depth=4,
                                               learning_rate=0.05, subsample=0.8,
                                               colsample_bytree=0.8, eval_metric="logloss",
                                               random_state=42, verbosity=0))])


def lgbm_pipeline(num_cols: list, cat_cols: list) -> Pipeline:
    pre = ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])
    return Pipeline([("pre", pre),
                     ("clf", lgb.LGBMClassifier(n_estimators=300, max_depth=4,
                                                learning_rate=0.05, subsample=0.8,
                                                colsample_bytree=0.8, random_state=42,
                                                verbosity=-1))])


def make_pipeline(label: str, num_cols: list, cat_cols: list) -> Pipeline:
    if label == "LR":   return lr_pipeline(num_cols, cat_cols)
    if label == "XGB":  return xgb_pipeline(num_cols, cat_cols)
    if label == "LGBM": return lgbm_pipeline(num_cols, cat_cols)
    raise ValueError(label)


# ── SHAP ───────────────────────────────────────────────────────────────────────

def _clean_name(raw: str) -> str:
    for prefix in ("num__", "cat__"):
        if raw.startswith(prefix):
            return raw[len(prefix):]
    return raw


def fit_and_explain(
    label: str,
    pipe: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> shap.Explanation:
    print(f"  Fitting {label}...")
    pipe.fit(X, y)

    X_transformed = pipe["pre"].transform(X)
    feature_names = [_clean_name(n) for n in pipe["pre"].get_feature_names_out()]

    print(f"  Computing SHAP for {label}...")
    clf = pipe["clf"]

    if label == "LR":
        masker   = shap.maskers.Independent(X_transformed, max_samples=500)
        explainer = shap.LinearExplainer(clf, masker)
        sv       = explainer(X_transformed)
    else:
        explainer = shap.TreeExplainer(clf)
        sv       = explainer(X_transformed, check_additivity=False)

    # Normalise to 2D if tree model returns 3D (n_samples, n_features, n_classes)
    if sv.values.ndim == 3:
        sv_values = sv.values[:, :, 1]
        sv_base   = sv.base_values[:, 1] if sv.base_values.ndim == 2 else sv.base_values
        sv = shap.Explanation(
            values       = sv_values,
            base_values  = sv_base,
            data         = sv.data,
            feature_names = feature_names,
        )
    else:
        sv = shap.Explanation(
            values       = sv.values,
            base_values  = sv.base_values,
            data         = sv.data,
            feature_names = feature_names,
        )

    print(f"  {label} SHAP done — {sv.values.shape[1]} features")
    return sv


# ── Plotting ───────────────────────────────────────────────────────────────────

def _fig_to_b64() -> str:
    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor="white")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close("all")
    return b64


def make_beeswarm_b64(sv: shap.Explanation, title: str) -> str:
    plt.rcParams.update({"font.size": 10, "figure.figsize": [12, 8], "figure.facecolor": "white"})
    shap.plots.beeswarm(sv, max_display=TOP_N, show=False)
    plt.title(title, fontsize=12, pad=12)
    plt.tight_layout()
    return _fig_to_b64()


def make_bar_b64(sv: shap.Explanation, title: str) -> str:
    plt.rcParams.update({"font.size": 10, "figure.figsize": [10, 8], "figure.facecolor": "white"})
    shap.plots.bar(sv, max_display=TOP_N, show=False)
    plt.title(title, fontsize=12, pad=12)
    plt.tight_layout()
    return _fig_to_b64()


# ── Per-feature mean |SHAP| ────────────────────────────────────────────────────

def mean_abs_shap(sv: shap.Explanation) -> pd.Series:
    return pd.Series(
        np.abs(sv.values).mean(axis=0),
        index=sv.feature_names,
    ).sort_values(ascending=False)


def categorise(feature_name: str) -> str:
    for cat_label, test in FEATURE_CATEGORIES:
        if test(feature_name):
            return cat_label
    return "Other"


def grouped_importance(mas: pd.Series) -> pd.DataFrame:
    rows = []
    for cat_label, _ in FEATURE_CATEGORIES:
        feats_in_cat = [f for f in mas.index if categorise(f) == cat_label]
        if feats_in_cat:
            total = mas[feats_in_cat].sum()
            rows.append({
                "Category":        cat_label,
                "n_features":      len(feats_in_cat),
                "sum_mean_abs_SHAP": total,
                "top_feature":     feats_in_cat[0] if feats_in_cat else "",
            })
    df = pd.DataFrame(rows).sort_values("sum_mean_abs_SHAP", ascending=False).reset_index(drop=True)
    df["share_pct"] = df["sum_mean_abs_SHAP"] / df["sum_mean_abs_SHAP"].sum() * 100
    return df


# ── HTML ───────────────────────────────────────────────────────────────────────

def img_tag(b64: str) -> str:
    return f"<img src='data:image/png;base64,{b64}' style='max-width:100%;margin-bottom:8px;'>"


def build_model_section(label: str, sv: shap.Explanation) -> str:
    mas     = mean_abs_shap(sv)
    grp     = grouped_importance(mas)
    bee_b64 = make_beeswarm_b64(sv, f"{label} — Beeswarm (top {TOP_N} features)")
    bar_b64 = make_bar_b64(sv, f"{label} — Mean |SHAP| (top {TOP_N} features)")

    # Grouped importance table
    grp_rows = ""
    max_val  = grp["sum_mean_abs_SHAP"].max()
    for _, row in grp.iterrows():
        bar_w  = int(row["sum_mean_abs_SHAP"] / max_val * 120)
        grp_rows += (
            f"<tr>"
            f"<td style='padding:4px 10px;font-weight:bold'>{row['Category']}</td>"
            f"<td style='padding:4px 8px;text-align:right'>{row['n_features']}</td>"
            f"<td style='padding:4px 8px;text-align:right'>{row['sum_mean_abs_SHAP']:.4f}</td>"
            f"<td style='padding:4px 8px;text-align:right'>{row['share_pct']:.1f}%</td>"
            f"<td style='padding:4px 8px'>"
            f"<div style='width:{bar_w}px;height:12px;background:#2196f3;border-radius:2px;display:inline-block'></div>"
            f"</td>"
            f"<td style='padding:4px 8px;color:#666;font-size:11px'>{row['top_feature']}</td>"
            f"</tr>"
        )
    grp_table = (
        "<table style='border-collapse:collapse;font-family:monospace;font-size:12px;margin-bottom:20px;'>"
        "<thead><tr style='background:#eee'>"
        "<th style='padding:5px 10px;text-align:left'>Category</th>"
        "<th style='padding:5px 8px'>n feats</th>"
        "<th style='padding:5px 8px'>Σ mean|SHAP|</th>"
        "<th style='padding:5px 8px'>Share</th>"
        "<th style='padding:5px 8px;min-width:140px'>Bar</th>"
        "<th style='padding:5px 8px;text-align:left'>Top feature</th>"
        "</tr></thead>"
        f"<tbody>{grp_rows}</tbody>"
        "</table>"
    )

    return (
        f"<div style='margin-bottom:64px;'>"
        f"<h2 style='font-family:sans-serif'>{label} — Feature Importance</h2>"
        f"<h3 style='font-family:sans-serif;color:#444;margin-top:0'>Grouped Importance (Σ mean |SHAP|)</h3>"
        f"{grp_table}"
        f"<div style='display:flex;gap:24px;flex-wrap:wrap;'>"
        f"<div>{img_tag(bee_b64)}</div>"
        f"<div>{img_tag(bar_b64)}</div>"
        f"</div>"
        f"</div>"
    )


def build_cross_model_table(all_mas: dict[str, pd.Series]) -> str:
    all_features = set()
    for mas in all_mas.values():
        all_features.update(mas.index)

    # Build dataframe: rows = features, cols = models
    rows = []
    for feat in all_features:
        row = {"feature": feat, "category": categorise(feat)}
        for label, mas in all_mas.items():
            row[f"{label}_mas"] = mas.get(feat, 0.0)
        row["avg_mas"] = np.mean([mas.get(feat, 0.0) for mas in all_mas.values()])
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("avg_mas", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    # Assign ranks per model
    for label in MODEL_LABELS:
        col = f"{label}_mas"
        df[f"{label}_rank"] = df[col].rank(ascending=False, method="min").astype(int)

    top = df.head(30)

    # Color scale: lighter = smaller
    max_avg = top["avg_mas"].max()

    def _bg(v: float) -> str:
        alpha = min(v / max_avg * 0.5, 0.5)
        return f"background:rgba(33,150,243,{alpha:.2f});"

    thead = (
        "<tr style='background:#111;color:white'>"
        "<th style='padding:5px 8px'>#</th>"
        "<th style='padding:5px 8px;text-align:left'>Feature</th>"
        "<th style='padding:5px 8px;text-align:left'>Category</th>"
        + "".join(f"<th style='padding:5px 8px'>{m}</th><th style='padding:5px 4px'>rank</th>"
                  for m in MODEL_LABELS)
        + "<th style='padding:5px 8px'>Avg</th>"
        "</tr>"
    )

    tbody = ""
    for _, row in top.iterrows():
        avg = row["avg_mas"]
        tbody += (
            f"<tr>"
            f"<td style='padding:4px 8px;text-align:right;color:#888'>{int(row['rank'])}</td>"
            f"<td style='padding:4px 8px;font-weight:bold;white-space:nowrap'>{row['feature']}</td>"
            f"<td style='padding:4px 8px;color:#555;font-size:11px'>{row['category']}</td>"
        )
        for label in MODEL_LABELS:
            mas_val  = row[f"{label}_mas"]
            rank_val = int(row[f"{label}_rank"])
            tbody += (
                f"<td style='padding:4px 8px;text-align:right;{_bg(mas_val)}'>{mas_val:.4f}</td>"
                f"<td style='padding:4px 6px;text-align:right;color:#888;font-size:11px'>#{rank_val}</td>"
            )
        tbody += (
            f"<td style='padding:4px 8px;text-align:right;font-weight:bold;{_bg(avg)}'>{avg:.4f}</td>"
            f"</tr>"
        )

    return (
        "<div style='margin-bottom:64px;'>"
        "<h2 style='font-family:sans-serif'>Cross-Model Feature Ranking (Top 30)</h2>"
        "<p style='font-family:monospace;color:#444;font-size:13px;max-width:900px;line-height:1.6;margin-bottom:12px'>"
        "Mean |SHAP| per feature across all three models. Blue intensity = average importance. "
        "Model-specific ranks shown in grey. Sorted by average importance across models."
        "</p>"
        "<div style='overflow-x:auto;'>"
        "<table style='border-collapse:collapse;font-family:monospace;font-size:12px;'>"
        f"<thead>{thead}</thead><tbody>{tbody}</tbody>"
        "</table></div></div>"
    )


def build_tldr(all_mas: dict[str, pd.Series]) -> str:
    # Gather grouped shares per model
    lines = []
    for label, mas in all_mas.items():
        grp = grouped_importance(mas)
        market_impl = grp[grp["Category"] == "Market (impl prob)"]["share_pct"].values
        rolling_all = grp[grp["Category"].isin(["Sack rate (rolling)", "QB hit rate", "Snap pct (rolling)"])]["share_pct"].sum()
        market_pct  = float(market_impl[0]) if len(market_impl) else 0.0
        lines.append(f"  <b>{label}</b>: market-implied-prob = {market_pct:.0f}% of signal; rolling player features = {rolling_all:.0f}%")

    bullets = "<br>".join(lines)
    return (
        "<div style='font-family:monospace;background:#e8f5e9;padding:16px 20px;"
        "border-radius:8px;margin-bottom:40px;font-size:13px;"
        "border-left:4px solid #2ca02c;max-width:950px;line-height:1.8;'>"
        "<b style='font-size:15px'>TLDR — What Is The Model Actually Learning?</b><br><br>"
        f"{bullets}<br><br>"
        "<b>Key finding:</b> LR and the tree models tell different stories. LR is the most "
        "interpretable — it leans on the market implied probability (~43%) to capture the structural "
        "Under edge, with rolling player features as secondary signal (~31%). The tree models (XGB/LGBM) "
        "flip this: rolling player history dominates (~57-59%), particularly <code>qbhit_rate_Lcareer</code> "
        "and <code>snap_pct_L1</code>, with market features playing a smaller role (~11-12%). "
        "This suggests the tree models have found non-linear player-specific patterns — career QB hit "
        "rate as a proxy for pass-rush role/skill, and last-week snap % as a recency signal — that are "
        "largely independent of how the market is pricing the bet. "
        "The consistent Under-only x=0.30 edge likely reflects <i>both</i>: a structural market "
        "mispricing (market overprices the Over at ~31% vs ~25% actual) AND the model correctly "
        "identifying lower-upside players from their play history."
        "</div>"
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    warnings.filterwarnings("ignore")

    cfg = load_config()
    num_cols, cat_cols = feature_lists(cfg)

    print("Loading data...")
    df = load_data(cfg)

    n_cols = [c for c in num_cols if c in df.columns]
    c_cols = [c for c in cat_cols if c in df.columns]
    X = df[n_cols + c_cols]
    y = df["target"].astype(int)

    print(f"  Features: {len(n_cols)} numeric, {len(c_cols)} categorical")

    # Fit all models + compute SHAP
    all_sv  = {}
    all_mas = {}
    for label in MODEL_LABELS:
        pipe     = make_pipeline(label, n_cols, c_cols)
        sv       = fit_and_explain(label, pipe, X, y)
        all_sv[label]  = sv
        all_mas[label] = mean_abs_shap(sv)

    # Build HTML sections
    print("\nBuilding HTML...")
    plt.rcParams.update({"font.family": "monospace"})

    header = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>NFL Sacks — SHAP Feature Importance</title>
  <style>
    body  { font-family: monospace; margin: 40px; background: #fafafa; }
    h1,h2,h3 { font-family: sans-serif; }
    table td, table th { border-bottom: 1px solid #e0e0e0; vertical-align: middle; }
    .divider { border-top: 3px solid #333; margin: 44px 0; }
  </style>
</head>
<body>
<h1>NFL Sacks Props — SHAP Feature Importance</h1>
<p style="font-family:monospace;color:#444;font-size:13px;max-width:900px;line-height:1.6;margin-bottom:24px;">
  Models fit on full 2024+2025 pooled data (2,927 rows). SHAP values computed on all rows.
  LR uses LinearExplainer; XGB/LGBM use TreeExplainer. Beeswarm = distribution + direction per feature;
  Bar = mean |SHAP| (overall importance). Features sorted by mean absolute SHAP value.
</p>"""

    tldr_html     = build_tldr(all_mas)
    cross_html    = build_cross_model_table(all_mas)
    model_sections = [build_model_section(label, all_sv[label]) for label in MODEL_LABELS]

    footer = "</body></html>"

    html = "\n".join([
        header,
        tldr_html,
        cross_html,
        "<div class='divider'></div>",
        *[s + "\n<div class='divider'></div>" for s in model_sections[:-1]],
        model_sections[-1],
        footer,
    ])

    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"\nReport: {OUT_HTML}")

    # Console summary
    print(f"\n{'='*55}")
    print("  SHAP SUMMARY — Market vs. Player Features")
    for label, mas in all_mas.items():
        grp         = grouped_importance(mas)
        mkt_impl    = grp[grp["Category"] == "Market (impl prob)"]["share_pct"].values
        rolling_sum = grp[grp["Category"].isin(
            ["Sack rate (rolling)", "QB hit rate", "Snap pct (rolling)"])]["share_pct"].sum()
        mkt_pct     = float(mkt_impl[0]) if len(mkt_impl) else 0.0
        top3 = mas.head(3).index.tolist()
        print(f"  {label}: market-impl={mkt_pct:.0f}%  rolling={rolling_sum:.0f}%  top3={top3}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
