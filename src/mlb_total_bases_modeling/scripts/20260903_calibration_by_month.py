"""
MLB Total Bases — Calibration Analysis by Month (2026-09-03)
=============================================================
Diagnoses the backtest (63.3% win) vs prod (56.1% win) gap by auditing
model calibration month-by-month across the full spine (2024–2026).

Steps:
  1. Score full spine with production model
  2. Calibration by month — all bets (no edge filter)
  3. Calibration by month — prod strategy (edge >= 5pp)
  4. Reliability diagram bins: IS vs OOS
  5. Seasonal pattern pooled across years
  6. Summary table + HTML report

Output:
  knowledge-base/raw/20260903-mlb-tb-calibration-by-month.html
"""
from __future__ import annotations

import re
import sys
import unicodedata
from io import BytesIO
from pathlib import Path

import boto3
import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

S3_BUCKET       = "the-odds-api-mt"
SPINE_KEY       = "mlb/total_bases_model/spine/mlb_total_bases_spine.parquet"
MARKET_RAW_KEY  = "mlb/total_bases_model/market_raw/mlb_total_bases_market_raw.parquet"
MODEL_KEY       = "mlb/total_bases_model/model/mlb_tb_regression_v2.joblib"
HTML_OUT        = REPO_ROOT / "knowledge-base/raw/20260903-mlb-tb-calibration-by-month.html"

IS_SEASONS  = [2024, 2025]
OOS_SEASON  = 2026
EDGE_THRESH = 0.05
LINE        = 1.5

MANUAL_MAP = {
    "daniel vogelbach":    "Dan Vogelbach",
    "michael a taylor":    "Michael Taylor",
    "max muncy (2002)":    "Max Muncy",
    "diego a castillo":    "Diego Castillo",
    "james jarvis":        "Jim Jarvis",
    "donnie walton":       "Donovan Walton",
    "josh kuroda-grauer":  "Joshua Kuroda-Grauer",
}


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = name.lower()
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = re.sub(r"[.,'\-]", "", name)
    name = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", name)
    name = re.sub(r"\s+", "", name)
    return name.strip()


# ── Load + score ──────────────────────────────────────────────────────────────

def load_and_score() -> pd.DataFrame:
    s3 = boto3.client("s3")
    manual_norm = {normalize_name(k): normalize_name(v) for k, v in MANUAL_MAP.items()}

    # ── Step 1: load consensus spine (player-game-line, has actuals + rolling features)
    print("Loading spine (consensus, player-game-line)...")
    spine = pd.read_parquet(BytesIO(
        s3.get_object(Bucket=S3_BUCKET, Key=SPINE_KEY)["Body"].read()
    ))
    spine["game_date"] = spine["game_date"].astype(str)
    print(f"  {len(spine):,} rows | {spine['game_date'].min()} → {spine['game_date'].max()}")

    # ── Step 2: score at player-game-line level (features same across all bookmakers)
    print("Loading model bundle...")
    bundle = joblib.load(BytesIO(
        s3.get_object(Bucket=S3_BUCKET, Key=MODEL_KEY)["Body"].read()
    ))
    model    = bundle["model"]
    scaler   = bundle["scaler"]
    features = bundle["features_numeric"]
    calibs   = bundle.get("calib_models", {})

    upg = (
        spine[["name_norm", "game_date", "line"] + features]
        .drop_duplicates(subset=["name_norm", "game_date", "line"])
        .dropna(subset=features)
        .copy()
    )
    upg["y_hat"] = model.predict(scaler.transform(upg[features].values.astype(float)))

    scored_rows = []
    for ln, calib in calibs.items():
        sub = upg[upg["line"] == ln].copy()
        if sub.empty:
            continue
        sub["p_model_under"] = 1.0 - np.clip(
            calib.predict_proba(sub["y_hat"].values.reshape(-1, 1))[:, 1], 0.01, 0.99
        )
        scored_rows.append(sub)

    p_df = pd.concat(scored_rows, ignore_index=True)

    # Attach p_model_under + actuals to spine, filter to target line
    spine_scored = spine.merge(
        p_df[["name_norm", "game_date", "line", "y_hat", "p_model_under"]],
        on=["name_norm", "game_date", "line"],
        how="inner",
    )
    spine_scored = spine_scored[spine_scored["line"] == LINE].copy()
    print(f"  Scored {spine_scored['name_norm'].nunique():,} unique players at line={LINE}")

    # ── Step 3: load market raw (per-bookmaker) and join — this matches prod granularity
    print("Loading market raw (per-bookmaker)...")
    market = pd.read_parquet(BytesIO(
        s3.get_object(Bucket=S3_BUCKET, Key=MARKET_RAW_KEY)["Body"].read()
    ))
    market["name_norm"] = (
        market["player_name"].map(normalize_name).map(lambda n: manual_norm.get(n, n))
    )
    market["game_date"] = market["game_date"].astype(str)
    market = market[
        (market["market_key"] == "batter_total_bases")
        & (market["line"] == LINE)
        & market["under_price"].notna()
        & market["over_price"].notna()
        & (market["under_price"] > 1.0)
        & (market["over_price"] > 1.0)
    ].copy()
    print(f"  {len(market):,} bookmaker rows at line={LINE}")

    # ── Step 4: join per-bookmaker market to scored spine
    spine_cols = [
        "name_norm", "game_date", "line",
        "total_bases",
        "p_model_under", "y_hat",
        "min_line", "max_line",
    ]
    df = market.merge(
        spine_scored[spine_cols],
        on=["name_norm", "game_date", "line"],
        how="inner",
    )

    df["raw_prob_under"] = 1.0 / df["under_price"]
    df["edge_under"]     = df["p_model_under"] - df["raw_prob_under"]
    df["actual_under"]   = (df["total_bases"] <= 1).astype(int)
    df["season"]         = pd.to_datetime(df["game_date"]).dt.year
    df["month"]          = pd.to_datetime(df["game_date"]).dt.to_period("M").astype(str)
    df["cal_month"]      = pd.to_datetime(df["game_date"]).dt.month
    df["split"]          = df["season"].apply(lambda y: "IS" if y in IS_SEASONS else "OOS")

    print(f"  Final: {len(df):,} player-game-bookmaker rows at line={LINE}")
    return df


def net_units(grp: pd.DataFrame) -> float:
    wins = grp["actual_under"] == 1
    return float((grp.loc[wins, "under_price"] - 1).sum() - (~wins).sum())
    return float(payout[wins].sum() - (~wins).sum())


# ── Step 2/3: calibration by month ───────────────────────────────────────────

def calib_by_month(df: pd.DataFrame, edge_min: float | None = None) -> pd.DataFrame:
    sub = df[df["edge_under"] >= edge_min].copy() if edge_min is not None else df.copy()
    agg = (
        sub.groupby("month")
        .apply(lambda g: pd.Series({
            "n":            len(g),
            "actual_rate":  g["actual_under"].mean(),
            "avg_p_model":  g["p_model_under"].mean(),
            "avg_raw_prob": g["raw_prob_under"].mean(),
            "net_units":    net_units(g),
            "split":        g["split"].iloc[0],
        }), include_groups=False)
        .reset_index()
    )
    agg["roi"]             = agg["net_units"] / agg["n"]
    agg["calib_err"]       = agg["actual_rate"] - agg["avg_p_model"]
    agg["break_even"]      = agg["avg_raw_prob"]
    return agg.sort_values("month")


# ── Step 4: reliability diagram bins ─────────────────────────────────────────

def reliability_bins(df: pd.DataFrame) -> pd.DataFrame:
    bins = np.arange(0.50, 0.96, 0.05)
    df = df.copy()
    df["bin"] = pd.cut(df["p_model_under"], bins=bins, right=False)
    rows = []
    for split in ["IS", "OOS"]:
        sub = df[df["split"] == split]
        for b, grp in sub.groupby("bin", observed=True):
            rows.append({
                "split":          split,
                "bin_lo":         float(b.left),
                "bin_hi":         float(b.right),
                "bin_label":      f"{b.left:.2f}–{b.right:.2f}",
                "n":              len(grp),
                "avg_p_model":    grp["p_model_under"].mean(),
                "actual_rate":    grp["actual_under"].mean(),
                "calib_err":      grp["actual_under"].mean() - grp["p_model_under"].mean(),
            })
    return pd.DataFrame(rows)


# ── Step 5: seasonal pattern pooled across years ──────────────────────────────

def seasonal_pattern(df: pd.DataFrame) -> pd.DataFrame:
    MONTH_NAMES = {4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep"}
    agg = (
        df.groupby("cal_month")
        .apply(lambda g: pd.Series({
            "n":           len(g),
            "actual_rate": g["actual_under"].mean(),
            "avg_p_model": g["p_model_under"].mean(),
            "avg_raw_prob":g["raw_prob_under"].mean(),
            "net_units":   net_units(g),
        }), include_groups=False)
        .reset_index()
    )
    agg["roi"]       = agg["net_units"] / agg["n"]
    agg["calib_err"] = agg["actual_rate"] - agg["avg_p_model"]
    agg["month_name"]= agg["cal_month"].map(MONTH_NAMES)
    return agg.sort_values("cal_month")


# ── HTML builder ──────────────────────────────────────────────────────────────

_SS = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Arial,sans-serif"

def _pct(v, decimals=1):
    try:
        return f"{float(v)*100:.{decimals}f}%"
    except Exception:
        return "—"

def _sgn(v, decimals=1):
    try:
        return f"{float(v)*100:+.{decimals}f}%"
    except Exception:
        return "—"

def _u(v):
    try:
        return f"{float(v):+.1f}u"
    except Exception:
        return "—"

def _color_calib(err):
    try:
        e = float(err)
        if e < -0.04:
            return "#c0392b"   # model badly over-optimistic
        if e < -0.02:
            return "#e67e22"
        if e > 0.02:
            return "#27ae60"
        return "#222"
    except Exception:
        return "#222"

def _color_roi(roi):
    try:
        r = float(roi)
        return "#276221" if r >= 0 else "#c0392b"
    except Exception:
        return "#222"

def _split_badge(split):
    if split == "IS":
        return "<span style='background:#dce8f5;color:#1a5276;padding:1px 5px;border-radius:3px;font-size:10px'>IS</span>"
    return "<span style='background:#fdebd0;color:#784212;padding:1px 5px;border-radius:3px;font-size:10px'>OOS</span>"


def _table(rows_html: str, headers: list[str]) -> str:
    ths = "".join(f"<th>{h}</th>" for h in headers)
    return f"""
<table>
  <tr>{ths}</tr>
  {rows_html}
</table>"""


def build_html(
    all_monthly: pd.DataFrame,
    prod_monthly: pd.DataFrame,
    rel_bins: pd.DataFrame,
    seasonal: pd.DataFrame,
    prod_edge_min: float,
) -> str:

    # ── Section 1: all bets by month
    rows1 = ""
    for _, r in all_monthly.iterrows():
        ce_color = _color_calib(r["calib_err"])
        rows1 += (
            f"<tr>"
            f"<td>{r['month']}&nbsp;{_split_badge(r['split'])}</td>"
            f"<td style='text-align:right'>{int(r['n']):,}</td>"
            f"<td style='text-align:right'>{_pct(r['actual_rate'])}</td>"
            f"<td style='text-align:right'>{_pct(r['avg_p_model'])}</td>"
            f"<td style='text-align:right'>{_pct(r['avg_raw_prob'])}</td>"
            f"<td style='text-align:right;color:{ce_color};font-weight:bold'>{_sgn(r['calib_err'])}</td>"
            f"</tr>\n"
        )
    sec1 = _table(rows1, ["Month", "n", "Actual UNDER%", "Model UNDER%", "Market implied%", "Calib error (actual−model)"])

    # ── Section 2: prod strategy by month
    rows2 = ""
    for _, r in prod_monthly.iterrows():
        ce_color  = _color_calib(r["calib_err"])
        roi_color = _color_roi(r["roi"])
        rows2 += (
            f"<tr>"
            f"<td>{r['month']}&nbsp;{_split_badge(r['split'])}</td>"
            f"<td style='text-align:right'>{int(r['n']):,}</td>"
            f"<td style='text-align:right'>{_pct(r['actual_rate'])}</td>"
            f"<td style='text-align:right'>{_pct(r['avg_p_model'])}</td>"
            f"<td style='text-align:right'>{_pct(r['avg_raw_prob'])}</td>"
            f"<td style='text-align:right;color:{ce_color};font-weight:bold'>{_sgn(r['calib_err'])}</td>"
            f"<td style='text-align:right'>{_u(r['net_units'])}</td>"
            f"<td style='text-align:right;color:{roi_color};font-weight:bold'>{_sgn(r['roi'])}</td>"
            f"</tr>\n"
        )
    sec2 = _table(rows2, ["Month", "n", "Actual UNDER%", "Model UNDER%", "Market implied%", "Calib error", "Net units", "ROI"])

    # ── Section 3: reliability bins
    rows3 = ""
    for split in ["IS", "OOS"]:
        sub = rel_bins[rel_bins["split"] == split]
        for _, r in sub.iterrows():
            ce_color = _color_calib(r["calib_err"])
            rows3 += (
                f"<tr>"
                f"<td>{_split_badge(split)}</td>"
                f"<td style='text-align:right'>{r['bin_label']}</td>"
                f"<td style='text-align:right'>{int(r['n']):,}</td>"
                f"<td style='text-align:right'>{_pct(r['avg_p_model'])}</td>"
                f"<td style='text-align:right'>{_pct(r['actual_rate'])}</td>"
                f"<td style='text-align:right;color:{ce_color};font-weight:bold'>{_sgn(r['calib_err'])}</td>"
                f"</tr>\n"
            )
    sec3 = _table(rows3, ["Split", "Model prob bin", "n", "Avg model%", "Actual%", "Calib error"])

    # ── Section 4: seasonal pattern
    rows4 = ""
    for _, r in seasonal.iterrows():
        ce_color  = _color_calib(r["calib_err"])
        roi_color = _color_roi(r["roi"])
        rows4 += (
            f"<tr>"
            f"<td>{r['month_name']}</td>"
            f"<td style='text-align:right'>{int(r['n']):,}</td>"
            f"<td style='text-align:right'>{_pct(r['actual_rate'])}</td>"
            f"<td style='text-align:right'>{_pct(r['avg_p_model'])}</td>"
            f"<td style='text-align:right'>{_pct(r['avg_raw_prob'])}</td>"
            f"<td style='text-align:right;color:{ce_color};font-weight:bold'>{_sgn(r['calib_err'])}</td>"
            f"<td style='text-align:right;color:{roi_color};font-weight:bold'>{_sgn(r['roi'])}</td>"
            f"</tr>\n"
        )
    sec4 = _table(rows4, ["Month", "n (all years)", "Actual UNDER%", "Model UNDER%", "Market implied%", "Calib error", "ROI (prod filter)"])

    return f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'>
<title>MLB Total Bases — Calibration by Month</title>
<style>
  body {{font-family:{_SS};color:#222;max-width:960px;margin:auto;padding:24px}}
  h1 {{color:#2c3e50;font-size:20px}}
  h2 {{color:#2c3e50;font-size:15px;margin-top:28px;border-bottom:1px solid #ddd;padding-bottom:4px}}
  p  {{font-size:13px;color:#444;line-height:1.5}}
  table {{border-collapse:collapse;width:100%;margin-top:8px;font-size:12px}}
  th {{background:#2c3e50;color:#fff;padding:7px 10px;text-align:left;white-space:nowrap}}
  td {{padding:5px 10px;border-bottom:1px solid #e8e8e8}}
  tr:nth-child(even) td {{background:#f9f9f9}}
  .note {{background:#fdf6e3;border-left:3px solid #f0ad4e;padding:10px 14px;font-size:12px;margin:12px 0}}
</style>
</head><body>
<h1>MLB Total Bases — Calibration by Month</h1>
<p>Model: XGBoost regression + per-line logistic calibration (IS = 2024–2025, OOS = 2026).
Strategy: UNDER 1.5, edge ≥ {prod_edge_min:.0%}, all odds. Prod live 2026-07-05.</p>

<div class='note'>
<strong>Context:</strong> Backtest OOS showed 63.3% win rate (+5.11% ROI).
Prod Jul–Sep shows 56.1% win rate (−7.7% ROI). Average prod odds = −154 (60.7% break-even).
Win rate trend: Jul 58.2% → Aug 54.5% → Sep 51.7%.
</div>

<h2>1. Calibration by month — ALL bets (no edge filter)</h2>
<p>Calib error = actual rate minus model predicted rate.
Negative = model too optimistic (overestimates UNDER probability).</p>
{sec1}

<h2>2. Calibration by month — Production strategy (edge ≥ {prod_edge_min:.0%})</h2>
<p>Only rows we actually bet on. Net units and ROI included.</p>
{sec2}

<h2>3. Reliability diagram — IS vs OOS</h2>
<p>Model probability binned in 5pp increments. Well-calibrated = actual% ≈ model%.
Systematic below-diagonal = model overestimates UNDER probability.</p>
{sec3}

<h2>4. Seasonal pattern (all years pooled, prod filter applied)</h2>
<p>Pooling 2024–2025–2026 by calendar month isolates the seasonal signal.</p>
{sec4}

</body></html>"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df = load_and_score()

    print("\nStep 2: calibration by month (all bets)...")
    all_monthly = calib_by_month(df)

    print("Step 3: calibration by month (prod strategy)...")
    prod_monthly = calib_by_month(df, edge_min=EDGE_THRESH)

    print("Step 4: reliability bins IS vs OOS...")
    rel = reliability_bins(df)

    print("Step 5: seasonal pattern pooled...")
    seasonal_all = seasonal_pattern(df)
    # Prod-filtered version for ROI column
    seasonal_prod = seasonal_pattern(df[df["edge_under"] >= EDGE_THRESH].copy())
    seasonal_all = seasonal_all.merge(
        seasonal_prod[["cal_month", "roi"]].rename(columns={"roi": "roi"}),
        on="cal_month", how="left", suffixes=("_all", ""),
    )

    # ── Print to terminal
    print("\n=== CALIBRATION BY MONTH (all bets, line=1.5) ===")
    print(f"{'Month':<10} {'split':<4} {'n':>6} {'actual%':>8} {'model%':>8} {'market%':>8} {'calib_err':>10}")
    print("-" * 60)
    for _, r in all_monthly.iterrows():
        print(f"{r['month']:<10} {r['split']:<4} {int(r['n']):>6,} "
              f"{r['actual_rate']:>8.1%} {r['avg_p_model']:>8.1%} "
              f"{r['avg_raw_prob']:>8.1%} {r['calib_err']:>+10.1%}")

    print(f"\n=== PROD STRATEGY (edge>={EDGE_THRESH:.0%}, line=1.5) ===")
    print(f"{'Month':<10} {'split':<4} {'n':>6} {'actual%':>8} {'model%':>8} {'market%':>8} {'calib_err':>10} {'net_u':>8} {'roi':>8}")
    print("-" * 80)
    for _, r in prod_monthly.iterrows():
        print(f"{r['month']:<10} {r['split']:<4} {int(r['n']):>6,} "
              f"{r['actual_rate']:>8.1%} {r['avg_p_model']:>8.1%} "
              f"{r['avg_raw_prob']:>8.1%} {r['calib_err']:>+10.1%} "
              f"{r['net_units']:>+8.1f} {r['roi']:>+8.1%}")

    print("\n=== RELIABILITY BINS ===")
    for split in ["IS", "OOS"]:
        print(f"\n  {split}:")
        sub = rel[rel["split"] == split]
        print(f"  {'bin':<12} {'n':>6} {'model%':>8} {'actual%':>8} {'err':>8}")
        for _, r in sub.iterrows():
            print(f"  {r['bin_label']:<12} {int(r['n']):>6,} {r['avg_p_model']:>8.1%} {r['actual_rate']:>8.1%} {r['calib_err']:>+8.1%}")

    print("\n=== SEASONAL PATTERN (all years pooled, prod filter) ===")
    print(f"{'Month':<6} {'n':>7} {'actual%':>8} {'model%':>8} {'market%':>8} {'calib_err':>10} {'roi':>8}")
    print("-" * 60)
    for _, r in seasonal_all.iterrows():
        print(f"{r['month_name']:<6} {int(r['n']):>7,} "
              f"{r['actual_rate']:>8.1%} {r['avg_p_model']:>8.1%} "
              f"{r['avg_raw_prob']:>8.1%} {r['calib_err']:>+10.1%} "
              f"{r['roi']:>+8.1%}")

    # ── Write HTML
    html = build_html(all_monthly, prod_monthly, rel, seasonal_all, EDGE_THRESH)
    HTML_OUT.parent.mkdir(parents=True, exist_ok=True)
    HTML_OUT.write_text(html, encoding="utf-8")
    print(f"\nHTML → {HTML_OUT}")


if __name__ == "__main__":
    main()
