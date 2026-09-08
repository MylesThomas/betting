"""
MLB Total Bases — BetOnline-only grid search (2026-08-05)
=========================================================
Same load_and_score pipeline as 20260801_grid_7bin.py,
filtered to bookmaker = betonlineag.

Grid: edge_min × line_bucket × odds_bin
IS = 2024+2025  OOS = 2026

HTML → knowledge-base/raw/20260805-mlb-tb-betonline-grid.html

Usage:
  python src/mlb_total_bases_modeling/scripts/20260805_betonline_grid.py
"""
from __future__ import annotations

import sys
import warnings
from io import BytesIO
from pathlib import Path

import boto3
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

S3_BUCKET = "the-odds-api-mt"
SPINE_KEY = "mlb/total_bases_model/regression/mlb_tb_reg_spine.parquet"
MODEL_KEY = "mlb/total_bases_model/model/mlb_tb_regression_v2.joblib"
BOOK      = "betonlineag"

IS_SEASONS  = [2024, 2025]
OOS_SEASON  = 2026
TARGET      = "total_bases"

EDGE_THRESHOLDS = [0.00, 0.01, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
LINE_BUCKETS    = ["all_lines", "0.5_only", "1.5_only"]

BINS = [
    ("Plus (better than −110)", "plus",  0.000, 0.524),
    ("Even (−110 to −120)",     "even",  0.524, 0.545),
    ("−120 to −150",            "l120",  0.545, 0.600),
    ("−150 to −175",            "m150",  0.600, 0.636),
    ("−175 to −200",            "h175",  0.636, 0.667),
    ("−200 to −250",            "vh200", 0.667, 0.714),
    ("−250+",                   "ex250", 0.714, 1.001),
]
BIN_KEYS   = [b[1] for b in BINS]
BIN_LABELS = [b[0] for b in BINS]

HTML_PATH = REPO_ROOT / "knowledge-base/raw/20260805-mlb-tb-betonline-grid.html"
_SS = "-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif"


# ── Load + score ───────────────────────────────────────────────────────────────

def load_and_score() -> pd.DataFrame:
    s3 = boto3.client("s3")
    print("Loading spine from S3...")
    spine = pd.read_parquet(BytesIO(
        s3.get_object(Bucket=S3_BUCKET, Key=SPINE_KEY)["Body"].read()
    ))
    print(f"  {len(spine):,} rows · {spine['game_date'].min()} → {spine['game_date'].max()}")

    # Filter to BetOnline only
    bol = spine[spine["bookmaker"] == BOOK].copy()
    print(f"  BetOnline rows: {len(bol):,}")

    print("Loading model bundle...")
    bundle = joblib.load(BytesIO(
        s3.get_object(Bucket=S3_BUCKET, Key=MODEL_KEY)["Body"].read()
    ))

    bol["season"]         = pd.to_datetime(bol["game_date"]).dt.year
    bol["month"]          = pd.to_datetime(bol["game_date"]).dt.to_period("M").astype(str)
    bol["raw_prob_under"] = 1.0 / bol["under_price"]

    lr = (bol.groupby(["name_norm", "game_date"])["line"]
          .agg(min_line="min", max_line="max").reset_index())
    bol = bol.merge(lr, on=["name_norm", "game_date"], how="left")

    model    = bundle["model"]
    scaler   = bundle["scaler"]
    features = bundle["features_numeric"]
    calibs   = bundle.get("calib_models", {})

    upg = (bol[["name_norm", "game_date", "line"] + features]
           .drop_duplicates(subset=["name_norm", "game_date", "line"])
           .dropna(subset=features).copy())
    upg["y_hat"] = model.predict(scaler.transform(upg[features].values.astype(float)))

    rows = []
    for line, calib in calibs.items():
        sub = upg[upg["line"] == line].copy()
        if sub.empty:
            continue
        sub["p_model_under"] = 1.0 - np.clip(
            calib.predict_proba(sub["y_hat"].values.reshape(-1, 1))[:, 1], 0.01, 0.99
        )
        rows.append(sub)

    p_df = pd.concat(rows, ignore_index=True)
    df = bol.merge(p_df[["name_norm", "game_date", "line", "y_hat", "p_model_under"]],
                   on=["name_norm", "game_date", "line"], how="inner")
    df["edge_under"] = df["p_model_under"] - df["raw_prob_under"]

    df["bin"] = "other"
    for _, key, lo, hi in BINS:
        df.loc[(df["raw_prob_under"] >= lo) & (df["raw_prob_under"] < hi), "bin"] = key

    print(f"  Scored {len(df):,} BetOnline rows")
    return df


# ── Helpers ────────────────────────────────────────────────────────────────────

def _max_dd(pnl: np.ndarray) -> float:
    if len(pnl) == 0:
        return 0.0
    cum  = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum)
    return float(np.max(peak - cum))


def _raw_to_am(p: float) -> str:
    d = 1.0 / max(p, 1e-9)
    return f"+{int(round((d-1)*100))}" if d >= 2.0 else f"−{int(round(100/(d-1)))}"


# ── Grids ──────────────────────────────────────────────────────────────────────

def run_grid(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for et in EDGE_THRESHOLDS:
        for lb in LINE_BUCKETS:
            sub = (df[df["line"] == 0.5] if lb == "0.5_only" else
                   df[df["line"] == 1.5] if lb == "1.5_only" else df).copy()
            sub = sub[sub["edge_under"] >= et]
            if len(sub) < 10:
                continue
            won    = (sub[TARGET] < sub["line"]).values.astype(float)
            payout = sub["under_price"].values - 1.0
            pnl    = np.where(won, payout, -1.0)
            records.append({
                "edge_min":    et,
                "line_bucket": lb,
                "n_bets":      len(pnl),
                "win_pct":     round(float(won.mean()), 4),
                "net_units":   round(float(pnl.sum()), 2),
                "roi_pct":     round(float(pnl.mean()) * 100, 2),
                "avg_odds":    _raw_to_am(float(sub["raw_prob_under"].mean())),
                "max_dd":      round(_max_dd(pnl), 2),
            })
    return pd.DataFrame(records).sort_values("net_units", ascending=False).reset_index(drop=True)


def run_odds_bin_grid(df: pd.DataFrame) -> pd.DataFrame:
    """At production params (edge≥5pp, line=1.5): performance per odds bin."""
    sub = df[(df["line"] == 1.5) & (df["edge_under"] >= 0.05)].copy()
    records = []
    for label, key, lo, hi in BINS:
        b = sub[(sub["raw_prob_under"] >= lo) & (sub["raw_prob_under"] < hi)]
        if len(b) < 5:
            continue
        won    = (b[TARGET] < b["line"]).values.astype(float)
        payout = b["under_price"].values - 1.0
        pnl    = np.where(won, payout, -1.0)
        records.append({
            "odds_bin":  label,
            "n_bets":    len(pnl),
            "win_pct":   round(float(won.mean()), 4),
            "net_units": round(float(pnl.sum()), 2),
            "roi_pct":   round(float(pnl.mean()) * 100, 2),
            "avg_odds":  _raw_to_am(float(b["raw_prob_under"].mean())),
        })
    return pd.DataFrame(records).sort_values("net_units", ascending=False).reset_index(drop=True)


def run_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Monthly at production params (edge≥5pp, line=1.5)."""
    sub = df[(df["line"] == 1.5) & (df["edge_under"] >= 0.05)].copy()
    records = []
    for month, grp in sub.groupby("month"):
        won    = (grp[TARGET] < grp["line"]).values.astype(float)
        payout = grp["under_price"].values - 1.0
        pnl    = np.where(won, payout, -1.0)
        records.append({
            "month":     month,
            "season":    int(str(month)[:4]),
            "n_bets":    len(pnl),
            "win_pct":   round(float(won.mean()), 4),
            "net_units": round(float(pnl.sum()), 2),
            "roi_pct":   round(float(pnl.mean()) * 100, 2),
        })
    return pd.DataFrame(records)


def run_edge_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Edge distribution: what edge values does BetOnline offer at line=1.5?"""
    sub = df[df["line"] == 1.5].copy()
    cuts = [-1.0, -0.10, -0.05, 0.00, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 1.0]
    labels = ["<−10pp","−10 to −5pp","−5 to 0pp","0 to 3pp","3 to 5pp",
              "5 to 7pp","7 to 10pp","10 to 15pp","15 to 20pp","20pp+"]
    sub["edge_bin"] = pd.cut(sub["edge_under"], bins=cuts, labels=labels, right=False)
    return (
        sub.groupby("edge_bin", observed=True)
        .size()
        .rename("n_bets")
        .reset_index()
        .assign(pct=lambda x: (x["n_bets"] / x["n_bets"].sum() * 100).round(1))
    )


# ── SQL tests ──────────────────────────────────────────────────────────────────

def run_tests(df: pd.DataFrame, is_g: pd.DataFrame, oos_g: pd.DataFrame) -> bool:
    import duckdb
    con = duckdb.connect()
    con.register("df", df)
    con.register("isg", is_g)
    con.register("oosg", oos_g)
    tests = [
        ("T1: only betonlineag rows",
         f"SELECT COUNT(DISTINCT bookmaker) = 1 AS pass FROM df"),
        ("T2: production row present in OOS",
         "SELECT COUNT(*) >= 1 AS pass FROM oosg WHERE edge_min=0.05 AND line_bucket='1.5_only'"),
        ("T3: IS n_bets > OOS n_bets at prod params",
         "SELECT (SELECT n_bets FROM isg WHERE edge_min=0.05 AND line_bucket='1.5_only') "
         "> (SELECT n_bets FROM oosg WHERE edge_min=0.05 AND line_bucket='1.5_only') AS pass"),
        ("T4: no null net_units",
         "SELECT COUNT(*) = 0 AS pass FROM oosg WHERE net_units IS NULL"),
        ("T5: OOS has >=3 edge_min values",
         "SELECT COUNT(DISTINCT edge_min) >= 3 AS pass FROM oosg"),
    ]
    all_pass = True
    print("\nSQL Tests:")
    for name, sql in tests:
        try:
            val = con.execute(sql).fetchone()
            ok  = bool(val[0]) if val else False
            if not ok:
                all_pass = False
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        except Exception as e:
            all_pass = False
            print(f"  [ERROR] {name}: {e}")
    return all_pass


# ── HTML ──────────────────────────────────────────────────────────────────────

def _td_u(v: float) -> str:
    c = "#1a7f37" if v > 0 else "#d32f2f"
    return f'<td style="text-align:right;color:{c};font-weight:bold">{v:+.1f}u</td>'


def _td_roi(v: float) -> str:
    c = "#1a7f37" if v > 0 else "#d32f2f"
    return f'<td style="text-align:right;color:{c}">{v:+.2f}%</td>'


def _th(cols):
    return "".join(f'<th style="padding:4px 8px;white-space:nowrap">{c}</th>' for c in cols)


def _thead(cols):
    return f'<thead><tr style="background:#2c3e50;color:#fff">{_th(cols)}</tr></thead>'


def _grid_html(df: pd.DataFrame, prod_edge=0.05, prod_line="1.5_only") -> str:
    rows = ""
    for _, r in df.iterrows():
        is_prod = (r["edge_min"] == prod_edge and r["line_bucket"] == prod_line)
        bg = ' style="background:#e8f5e9"' if is_prod else ""
        rows += (
            f'<tr{bg}>'
            f'<td style="text-align:right;padding:4px 8px">{r["edge_min"]:.0%}</td>'
            f'<td style="padding:4px 8px">{r["line_bucket"]}</td>'
            f'<td style="text-align:right;padding:4px 8px">{r["n_bets"]:,}</td>'
            f'<td style="text-align:right;padding:4px 8px">{r["win_pct"]:.1%}</td>'
            + _td_u(r["net_units"])
            + _td_roi(r["roi_pct"])
            + f'<td style="text-align:right;padding:4px 8px">{r["avg_odds"]}</td>'
            f'<td style="text-align:right;padding:4px 8px">{r["max_dd"]:.1f}u</td>'
            f'</tr>'
        )
    return (
        f'<table style="font-size:12px;border-collapse:collapse;border:1px solid #ddd;background:#fff">'
        f'{_thead(["Edge≥","Line","N","Win%","Net Units","ROI%","Avg Odds","Max DD"])}'
        f'<tbody>{rows}</tbody></table>'
    )


def _odds_bin_html(df: pd.DataFrame) -> str:
    rows = ""
    for _, r in df.iterrows():
        rows += (
            f'<tr>'
            f'<td style="padding:4px 8px">{r["odds_bin"]}</td>'
            f'<td style="text-align:right;padding:4px 8px">{r["n_bets"]:,}</td>'
            f'<td style="text-align:right;padding:4px 8px">{r["win_pct"]:.1%}</td>'
            + _td_u(r["net_units"])
            + _td_roi(r["roi_pct"])
            + f'<td style="text-align:right;padding:4px 8px">{r["avg_odds"]}</td>'
            f'</tr>'
        )
    return (
        f'<table style="font-size:12px;border-collapse:collapse;border:1px solid #ddd;background:#fff">'
        f'{_thead(["Odds Bin","N","Win%","Net Units","ROI%","Avg Odds"])}'
        f'<tbody>{rows}</tbody></table>'
    )


def _monthly_html(df: pd.DataFrame) -> str:
    rows = ""
    cum = 0.0
    for _, r in df.iterrows():
        cum += r["net_units"]
        is_oos = r["season"] == OOS_SEASON
        bg = ' style="background:#fffde7"' if is_oos else ""
        rows += (
            f'<tr{bg}>'
            f'<td style="padding:4px 8px"><b>{r["month"]}</b></td>'
            f'<td style="text-align:right;padding:4px 8px">{r["n_bets"]:,}</td>'
            f'<td style="text-align:right;padding:4px 8px">{r["win_pct"]:.1%}</td>'
            + _td_u(r["net_units"])
            + _td_roi(r["roi_pct"])
            + f'<td style="text-align:right;padding:4px 8px">{cum:+.1f}u</td>'
            f'</tr>'
        )
    return (
        f'<table style="font-size:12px;border-collapse:collapse;border:1px solid #ddd;background:#fff">'
        f'{_thead(["Month","N","Win%","Net Units","ROI%","Cumulative"])}'
        f'<tbody>{rows}</tbody></table>'
    )


def _edge_dist_html(df: pd.DataFrame) -> str:
    rows = ""
    for _, r in df.iterrows():
        rows += (
            f'<tr>'
            f'<td style="padding:4px 8px">{r["edge_bin"]}</td>'
            f'<td style="text-align:right;padding:4px 8px">{r["n_bets"]:,}</td>'
            f'<td style="text-align:right;padding:4px 8px">{r["pct"]:.1f}%</td>'
            f'</tr>'
        )
    return (
        f'<table style="font-size:12px;border-collapse:collapse;border:1px solid #ddd;background:#fff">'
        f'{_thead(["Edge Bucket","N","% of bets"])}'
        f'<tbody>{rows}</tbody></table>'
    )


def build_html(is_g, oos_g, odds_is, odds_oos, monthly, edge_dist,
               is_prod_row, oos_prod_row, test_pass) -> str:
    def _prod_stat(r):
        if r is None or r.empty:
            return "N/A"
        r = r.iloc[0]
        return f"n={r['n_bets']:,} · {r['net_units']:+.1f}u · {r['roi_pct']:+.2f}% ROI · {r['win_pct']:.1%} win"

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>MLB Total Bases — BetOnline Grid (2026-08-05)</title>
<style>
  body {{ font-family:{_SS}; font-size:13px; color:#222; margin:24px; background:#f5f5f5; }}
  h1   {{ color:#1a1a2e; font-size:20px; margin-bottom:4px; }}
  h2   {{ color:#1a1a2e; font-size:15px; margin:20px 0 6px; border-bottom:2px solid #2c3e50; padding-bottom:4px; }}
  h3   {{ color:#333; font-size:13px; margin:14px 0 4px; }}
  .note {{ color:#555; font-size:11px; margin-bottom:8px; }}
  .pass {{ color:#1a7f37; font-weight:bold; }}
  .fail {{ color:#d32f2f; font-weight:bold; }}
  .stat {{ background:#fff; border:1px solid #ddd; border-radius:6px; padding:10px 16px; display:inline-block; margin:4px; font-size:13px; }}
</style>
</head>
<body>

<h1>MLB Total Bases — BetOnline (betonlineag) Grid Search</h1>
<p class="note">Generated 2026-08-05 · Strategy: UNDER · IS=2024+2025, OOS=2026 · Yellow rows = OOS months</p>

<div>
  <div class="stat"><b>IS prod params:</b> {_prod_stat(is_prod_row)}</div>
  <div class="stat"><b>OOS prod params:</b> {_prod_stat(oos_prod_row)}</div>
</div>

<h2>1. IS Grid (2024+2025) — sorted by net units</h2>
<p class="note">Green = production params (edge≥5pp, line=1.5).</p>
{_grid_html(is_g)}

<h2>2. OOS Grid (2026) — sorted by net units</h2>
{_grid_html(oos_g)}

<h2>3. Monthly Breakdown — production params (edge≥5pp, line=1.5)</h2>
<p class="note">Yellow = OOS (2026). Cumulative = running total across all months.</p>
{_monthly_html(monthly)}

<h2>4. Odds Bin Breakdown — IS (2024+2025), prod params</h2>
<p class="note">Which odds ranges are generating the profit at BetOnline?</p>
{_odds_bin_html(odds_is)}

<h2>5. Odds Bin Breakdown — OOS (2026), prod params</h2>
{_odds_bin_html(odds_oos)}

<h2>6. Edge Distribution — BetOnline, line=1.5 (all bets)</h2>
<p class="note">How many BetOnline rows qualify at each edge threshold?</p>
{_edge_dist_html(edge_dist)}

<h2>7. Test Results</h2>
<p class="{'pass' if test_pass else 'fail'}">{"PASS ✓" if test_pass else "FAIL ✗"}</p>

</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    df = load_and_score()

    is_df  = df[df["season"].isin(IS_SEASONS)].copy()
    oos_df = df[df["season"] == OOS_SEASON].copy()
    print(f"IS: {len(is_df):,}  OOS: {len(oos_df):,}")

    print("Running grids...")
    is_g  = run_grid(is_df)
    oos_g = run_grid(oos_df)

    print("Running odds bin breakdown...")
    odds_is  = run_odds_bin_grid(is_df)
    odds_oos = run_odds_bin_grid(oos_df)

    print("Running monthly breakdown...")
    monthly   = run_monthly(df)
    edge_dist = run_edge_distribution(df)

    # Production row readout
    is_prod  = is_g[(is_g["edge_min"]==0.05) & (is_g["line_bucket"]=="1.5_only")]
    oos_prod = oos_g[(oos_g["edge_min"]==0.05) & (oos_g["line_bucket"]=="1.5_only")]

    print("\n=== IS production params (edge≥5pp, line=1.5) ===")
    print(is_prod.to_string(index=False))
    print("\n=== OOS production params (edge≥5pp, line=1.5) ===")
    print(oos_prod.to_string(index=False))

    print("\n=== IS grid (top 15) ===")
    print(is_g.head(15).to_string(index=False))
    print("\n=== OOS grid (top 15) ===")
    print(oos_g.head(15).to_string(index=False))

    print("\n=== IS odds bins ===")
    print(odds_is.to_string(index=False))
    print("\n=== OOS odds bins ===")
    print(odds_oos.to_string(index=False))

    print("\n=== Monthly ===")
    print(monthly.to_string(index=False))

    test_pass = run_tests(df, is_g, oos_g)
    print("\nOVERALL:", "PASS ✓" if test_pass else "FAIL ✗")

    html = build_html(is_g, oos_g, odds_is, odds_oos, monthly, edge_dist,
                      is_prod, oos_prod, test_pass)
    HTML_PATH.write_text(html)
    print(f"\nHTML → {HTML_PATH}")


if __name__ == "__main__":
    main()
