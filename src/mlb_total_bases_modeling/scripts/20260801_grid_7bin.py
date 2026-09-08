"""
MLB Total Bases — IS/OOS Grid with 7-Bin Odds Breakdown + 2026 Monthly (2026-08-01)
=====================================================================================
Grid rows: edge_min × line_bucket (same as before).
14 breakdown columns per row — 7 odds bins × (pct_bets, net_units):

  plus       better than −110  raw_prob < 0.524
  even       −110 to −120      raw_prob 0.524–0.545
  l120       −120 to −150      raw_prob 0.545–0.600
  m150       −150 to −175      raw_prob 0.600–0.636
  h175       −175 to −200      raw_prob 0.636–0.667
  vh200      −200 to −250      raw_prob 0.667–0.714
  ex250      −250+             raw_prob 0.714+

Second output: 2026 monthly breakdown (rows = month, same 14 cols).
Appends a new <section> to knowledge-base/raw/20260801-mlb-tb-odds-breakdown-grid.html.

Usage:
  python src/mlb_total_bases_modeling/scripts/20260801_grid_7bin.py
"""
from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import boto3
import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

S3_BUCKET = "the-odds-api-mt"
SPINE_KEY = "mlb/total_bases_model/regression/mlb_tb_reg_spine.parquet"
MODEL_KEY = "mlb/total_bases_model/model/mlb_tb_regression_v2.joblib"
OUT_DIR   = Path.home() / "Downloads/tmp/mlb_total_bases"
HTML_PATH = REPO_ROOT / "knowledge-base/raw/20260801-mlb-tb-odds-breakdown-grid.html"
TARGET    = "total_bases"

IS_SEASONS = [2024, 2025]
OOS_SEASON = 2026

EDGE_THRESHOLDS = [0.00, 0.01, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
LINE_BUCKETS    = ["all_lines", "0.5_only", "1.5_only"]

# 7 odds bins: (label, short_key, lo_raw_prob_under, hi_raw_prob_under)
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


# ── Data ──────────────────────────────────────────────────────────────────────

def load_and_score() -> pd.DataFrame:
    s3 = boto3.client("s3")
    print("Loading spine...")
    spine = pd.read_parquet(BytesIO(
        s3.get_object(Bucket=S3_BUCKET, Key=SPINE_KEY)["Body"].read()
    ))
    print(f"  {len(spine):,} rows · {spine['game_date'].min()} → {spine['game_date'].max()}")
    print("Loading model bundle...")
    bundle = joblib.load(BytesIO(
        s3.get_object(Bucket=S3_BUCKET, Key=MODEL_KEY)["Body"].read()
    ))

    df = spine.copy()
    df["season"]         = pd.to_datetime(df["game_date"]).dt.year
    df["month"]          = pd.to_datetime(df["game_date"]).dt.to_period("M").astype(str)
    df["raw_prob_under"] = 1.0 / df["under_price"]

    lr = (df.groupby(["name_norm", "game_date"])["line"]
          .agg(min_line="min", max_line="max").reset_index())
    df = df.merge(lr, on=["name_norm", "game_date"], how="left")

    model    = bundle["model"]
    scaler   = bundle["scaler"]
    features = bundle["features_numeric"]
    calibs   = bundle.get("calib_models", {})

    upg = (df[["name_norm", "game_date", "line"] + features]
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
    df = df.merge(p_df[["name_norm", "game_date", "line", "y_hat", "p_model_under"]],
                  on=["name_norm", "game_date", "line"], how="inner")
    df["edge_under"] = df["p_model_under"] - df["raw_prob_under"]

    # assign bin key
    df["bin"] = "other"
    for label, key, lo, hi in BINS:
        df.loc[(df["raw_prob_under"] >= lo) & (df["raw_prob_under"] < hi), "bin"] = key

    print(f"  Scored {len(df):,} rows")
    return df


# ── Grid ──────────────────────────────────────────────────────────────────────

def _breakdown(sub: pd.DataFrame) -> dict:
    n = len(sub)
    if n == 0:
        return {f"pct_{k}": 0.0 for k in BIN_KEYS} | {f"u_{k}": 0.0 for k in BIN_KEYS}
    won    = (sub[TARGET] < sub["line"]).values.astype(float)
    payout = sub["under_price"].values - 1.0
    pnl    = np.where(won, payout, -1.0)
    result = {}
    for _, key, lo, hi in BINS:
        mask = (sub["raw_prob_under"].values >= lo) & (sub["raw_prob_under"].values < hi)
        result[f"pct_{key}"]  = round(mask.sum() / n * 100, 1)
        result[f"u_{key}"]    = round(float(pnl[mask].sum()), 2)
    return result


def _max_dd(pnl: np.ndarray) -> float:
    cum  = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum)
    return float(np.max(peak - cum)) if len(pnl) else 0.0


def _raw_to_am(p: float) -> str:
    d = 1.0 / max(p, 1e-9)
    return f"+{int(round((d-1)*100))}" if d >= 2.0 else f"−{int(round(100/(d-1)))}"


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
            rec = {
                "edge_min":   et,
                "line_bucket": lb,
                "n_bets":     len(pnl),
                "win_pct":    round(float(won.mean()), 4),
                "net_units":  round(float(pnl.sum()), 2),
                "roi_pct":    round(float(pnl.mean()) * 100, 2),
                "avg_odds":   _raw_to_am(float(sub["raw_prob_under"].mean())),
                "max_dd":     round(_max_dd(pnl), 2),
            }
            rec.update(_breakdown(sub))
            records.append(rec)
    return pd.DataFrame(records).sort_values("net_units", ascending=False).reset_index(drop=True)


def run_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """2026 monthly breakdown at production params (edge≥5pp, line=1.5)."""
    sub = df[(df["season"] == OOS_SEASON) &
             (df["line"] == 1.5) &
             (df["edge_under"] >= 0.05)].copy()
    records = []
    for month, grp in sub.groupby("month"):
        won    = (grp[TARGET] < grp["line"]).values.astype(float)
        payout = grp["under_price"].values - 1.0
        pnl    = np.where(won, payout, -1.0)
        rec = {
            "month":     month,
            "n_bets":    len(pnl),
            "win_pct":   round(float(won.mean()), 4),
            "net_units": round(float(pnl.sum()), 2),
            "roi_pct":   round(float(pnl.mean()) * 100, 2),
            "avg_odds":  _raw_to_am(float(grp["raw_prob_under"].mean())),
        }
        rec.update(_breakdown(grp))
        records.append(rec)
    return pd.DataFrame(records)


# ── HTML ──────────────────────────────────────────────────────────────────────

_BIN_COLORS = {
    "plus":  "#1a7f37",
    "even":  "#888888",
    "l120":  "#e65100",
    "m150":  "#d32f2f",
    "h175":  "#b71c1c",
    "vh200": "#880e4f",
    "ex250": "#4a0000",
}


def _units_td(v: float) -> str:
    c = "#1a7f37" if v > 0 else "#d32f2f"
    return f'<td style="text-align:right;color:{c};font-weight:bold">{v:+.1f}u</td>'


def _grid_table(df: pd.DataFrame, prod_edge: float = 0.05,
                prod_line: str = "1.5_only") -> str:
    base_hdrs  = ["Edge≥", "Line", "N", "Win%", "Net Units", "ROI%", "Avg Odds", "Max DD"]
    bin_hdrs   = []
    for label, key, *_ in BINS:
        bin_hdrs += [f"% {label}", f"u {label}"]

    all_hdrs = base_hdrs + bin_hdrs
    ths = "".join(f'<th style="white-space:nowrap">{h}</th>' for h in all_hdrs)

    rows_html = ""
    for _, r in df.iterrows():
        is_prod = (r["edge_min"] == prod_edge and r["line_bucket"] == prod_line)
        bg = ' style="background:#e8f5e9"' if is_prod else ""
        roi_c = "#1a7f37" if r["roi_pct"] > 0 else "#d32f2f"
        row = (
            f'<tr{bg}>'
            f'<td>{r["edge_min"]:.0%}</td>'
            f'<td>{r["line_bucket"]}</td>'
            f'<td style="text-align:right">{r["n_bets"]:,}</td>'
            f'<td style="text-align:right">{r["win_pct"]:.1%}</td>'
            + _units_td(r["net_units"])
            + f'<td style="text-align:right;color:{roi_c}">{r["roi_pct"]:+.2f}%</td>'
            f'<td style="text-align:right">{r["avg_odds"]}</td>'
            f'<td style="text-align:right">{r["max_dd"]:.1f}u</td>'
        )
        for _, key, *_ in BINS:
            c = _BIN_COLORS[key]
            row += f'<td style="text-align:right;color:{c}">{r[f"pct_{key}"]:.1f}%</td>'
            row += _units_td(r[f"u_{key}"])
        row += "</tr>"
        rows_html += row

    return (
        f'<table style="font-size:10px;border-collapse:collapse;width:100%;background:#fff">'
        f'<thead><tr style="background:#2c3e50;color:#fff">{ths}</tr></thead>'
        f'<tbody>{rows_html}</tbody></table>'
    )


def _monthly_table(df: pd.DataFrame) -> str:
    base_hdrs = ["Month", "N", "Win%", "Net Units", "ROI%", "Avg Odds"]
    bin_hdrs  = []
    for label, key, *_ in BINS:
        bin_hdrs += [f"% {label}", f"u {label}"]

    all_hdrs = base_hdrs + bin_hdrs
    ths = "".join(f'<th style="white-space:nowrap">{h}</th>' for h in all_hdrs)

    rows_html = ""
    cum_units = 0.0
    for _, r in df.iterrows():
        cum_units += r["net_units"]
        roi_c = "#1a7f37" if r["roi_pct"] > 0 else "#d32f2f"
        row = (
            f'<tr>'
            f'<td><b>{r["month"]}</b></td>'
            f'<td style="text-align:right">{r["n_bets"]:,}</td>'
            f'<td style="text-align:right">{r["win_pct"]:.1%}</td>'
            + _units_td(r["net_units"])
            + f'<td style="text-align:right;color:{roi_c}">{r["roi_pct"]:+.2f}%</td>'
            f'<td style="text-align:right">{r["avg_odds"]}</td>'
        )
        for _, key, *_ in BINS:
            c = _BIN_COLORS[key]
            row += f'<td style="text-align:right;color:{c}">{r[f"pct_{key}"]:.1f}%</td>'
            row += _units_td(r[f"u_{key}"])
        row += f'</tr>'
        rows_html += row

    return (
        f'<table style="font-size:10px;border-collapse:collapse;width:100%;background:#fff">'
        f'<thead><tr style="background:#2c3e50;color:#fff">{ths}</tr></thead>'
        f'<tbody>{rows_html}</tbody></table>'
    )


def _legend_html() -> str:
    items = "".join(
        f'<span style="color:{_BIN_COLORS[key]};margin-right:14px">■ {label}</span>'
        for label, key, *_ in BINS
    )
    return f'<p style="font-size:11px;margin:8px 0 12px">{items}</p>'


def build_section(is_grid: pd.DataFrame, oos_grid: pd.DataFrame,
                  monthly: pd.DataFrame) -> str:
    ss = "-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif"
    return f"""
<hr style="border:none;border-top:3px solid #2c3e50;margin:32px 0">
<div style="font-family:{ss};font-size:13px;color:#222">

<h2 style="color:#1a1a2e;font-size:18px;margin-bottom:4px">
  7-Bin Odds Breakdown Grid — 2026-08-01
</h2>
<p style="color:#555;margin:0 0 12px;font-size:12px">
  14 new columns per row (7 bins × %bets + net units).
  Green row = production strategy (edge≥5pp, line=1.5).
</p>
{_legend_html()}

<h3 style="font-size:14px;margin:20px 0 6px;color:#1a1a2e">IS — 2024+2025</h3>
{_grid_table(is_grid)}

<h3 style="font-size:14px;margin:20px 0 6px;color:#1a1a2e">OOS — 2026</h3>
{_grid_table(oos_grid)}

<h3 style="font-size:14px;margin:20px 0 6px;color:#1a1a2e">
  2026 Monthly Breakdown — production params (edge≥5pp, line=1.5)
</h3>
<p style="font-size:11px;color:#666;margin-bottom:8px">
  Rows = calendar month. Use this to check if the model is in a rough patch
  or if performance is consistent across 2026.
</p>
{_monthly_table(monthly)}

</div>
"""


# ── Tests ─────────────────────────────────────────────────────────────────────

def run_tests(df: pd.DataFrame, is_g: pd.DataFrame, oos_g: pd.DataFrame,
              monthly: pd.DataFrame) -> bool:
    import duckdb
    con = duckdb.connect()
    con.register("spine", df)
    con.register("isg",   is_g)
    con.register("oosg",  oos_g)
    con.register("mon",   monthly)

    pct_cols = " + ".join(f"pct_{k}" for k in BIN_KEYS)
    u_cols   = " + ".join(f"u_{k}"   for k in BIN_KEYS)

    tests = [
        ("T1: bin pcts sum to ~100 in IS",
         f"SELECT COUNT(*) = 0 AS pass FROM isg WHERE ABS({pct_cols} - 100) > 0.3"),
        ("T2: bin pcts sum to ~100 in OOS",
         f"SELECT COUNT(*) = 0 AS pass FROM oosg WHERE ABS({pct_cols} - 100) > 0.3"),
        ("T3: bin units sum to net_units in IS (within 0.1u)",
         f"SELECT COUNT(*) = 0 AS pass FROM isg WHERE ABS({u_cols} - net_units) > 0.1"),
        ("T4: bin units sum to net_units in OOS (within 0.1u)",
         f"SELECT COUNT(*) = 0 AS pass FROM oosg WHERE ABS({u_cols} - net_units) > 0.1"),
        ("T5: production row present in OOS",
         "SELECT COUNT(*) >= 1 AS pass FROM oosg WHERE edge_min=0.05 AND line_bucket='1.5_only'"),
        ("T6: production row > 90% minus-odds bins (l120+m150+h175+vh200+ex250)",
         "SELECT (pct_l120+pct_m150+pct_h175+pct_vh200+pct_ex250) > 90 AS pass "
         "FROM oosg WHERE edge_min=0.05 AND line_bucket='1.5_only' LIMIT 1"),
        ("T7: monthly table has ≥ 3 months of 2026 data",
         "SELECT COUNT(*) >= 3 AS pass FROM mon"),
        ("T8: no null net_units in OOS grid",
         "SELECT COUNT(*) = 0 AS pass FROM oosg WHERE net_units IS NULL"),
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import warnings
    warnings.filterwarnings("ignore")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_and_score()

    is_df  = df[df["season"].isin(IS_SEASONS)]
    oos_df = df[df["season"] == OOS_SEASON]
    print(f"IS: {len(is_df):,}  OOS: {len(oos_df):,}")

    print("Running IS grid...")
    is_grid = run_grid(is_df)
    print("Running OOS grid...")
    oos_grid = run_grid(oos_df)
    print("Running 2026 monthly breakdown...")
    monthly = run_monthly(df)

    is_grid.to_csv(OUT_DIR  / "mlb_tb_is_7bin_grid.csv",  index=False)
    oos_grid.to_csv(OUT_DIR / "mlb_tb_oos_7bin_grid.csv", index=False)
    monthly.to_csv(OUT_DIR  / "mlb_tb_2026_monthly.csv",  index=False)

    # Print key results to terminal
    print("\n── OOS production row (edge≥5pp, line=1.5) ──")
    prod = oos_grid[(oos_grid["edge_min"] == 0.05) & (oos_grid["line_bucket"] == "1.5_only")]
    if not prod.empty:
        r = prod.iloc[0]
        print(f"  n={r['n_bets']:,}  win={r['win_pct']:.1%}  units={r['net_units']:+.1f}  ROI={r['roi_pct']:+.2f}%")
        for label, key, *_ in BINS:
            print(f"  {label:30s}  {r[f'pct_{key}']:5.1f}%  {r[f'u_{key}']:+.1f}u")

    print("\n── 2026 monthly (edge≥5pp, line=1.5) ──")
    print(monthly[["month","n_bets","win_pct","net_units","roi_pct",
                   "pct_l120","u_l120","pct_m150","u_m150","pct_h175","u_h175"]].to_string(index=False))

    all_pass = run_tests(df, is_grid, oos_grid, monthly)
    print("\nOVERALL:", "PASS ✓" if all_pass else "FAIL ✗")

    # Append new section to existing HTML
    if HTML_PATH.exists():
        existing = HTML_PATH.read_text()
        section  = build_section(is_grid, oos_grid, monthly)
        new_html = existing.replace("</body>", section + "\n</body>")
        HTML_PATH.write_text(new_html)
        print(f"Appended new section → {HTML_PATH}")
    else:
        HTML_PATH.write_text(
            f"<!DOCTYPE html><html><head><meta charset='utf-8'></head><body>"
            f"{build_section(is_grid, oos_grid, monthly)}</body></html>"
        )
        print(f"Created → {HTML_PATH}")


if __name__ == "__main__":
    main()
