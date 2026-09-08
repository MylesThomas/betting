"""
MLB Total Bases — IS/OOS Grid Search with Odds-Type Breakdown (2026-08-01)
==========================================================================
Runs the same edge × line × direction grid as 20260710_grid_search_raw_edge.py
but adds 6 columns to every row showing the +/even/- breakdown within that cell:

  pct_plus   % of bets at plus odds  (raw_prob_under < 0.488, under_price > ~2.05)
  units_plus net units from those bets
  pct_even   % of bets near even     (0.488 ≤ raw_prob_under < 0.512)
  units_even net units
  pct_minus  % of bets at minus odds (raw_prob_under ≥ 0.512)
  units_minus net units

IS  = 2024 + 2025
OOS = 2026

Usage:
  python src/mlb_total_bases_modeling/scripts/20260801_grid_with_odds_breakdown.py
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
HTML_OUT  = REPO_ROOT / "knowledge-base/raw/20260801-mlb-tb-odds-breakdown-grid.html"
TARGET    = "total_bases"

IS_SEASONS = [2024, 2025]
OOS_SEASON = 2026

EDGE_THRESHOLDS = [0.00, 0.01, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]
LINE_BUCKETS    = ["all_lines", "0.5_only", "1.5_only"]

# Odds-type boundaries (on raw_prob_under = 1/under_price)
PLUS_HI  = 0.488   # raw_prob < 0.488 → under_price > ~2.05 → plus odds
MINUS_LO = 0.512   # raw_prob ≥ 0.512 → under_price < ~1.95 → minus odds
# between 0.488 and 0.512 = "even" (~-105 to +105)


def load_and_score() -> pd.DataFrame:
    s3 = boto3.client("s3")
    print("Loading regression spine from S3...")
    spine = pd.read_parquet(BytesIO(
        s3.get_object(Bucket=S3_BUCKET, Key=SPINE_KEY)["Body"].read()
    ))
    print(f"  {len(spine):,} rows · {spine['game_date'].min()} → {spine['game_date'].max()}")

    print("Loading model bundle from S3...")
    bundle = joblib.load(BytesIO(
        s3.get_object(Bucket=S3_BUCKET, Key=MODEL_KEY)["Body"].read()
    ))

    df = spine.copy()
    df["season"]         = pd.to_datetime(df["game_date"]).dt.year
    df["raw_prob_under"] = 1.0 / df["under_price"]
    df["raw_prob_over"]  = 1.0 / df["over_price"]

    lr = (
        df.groupby(["name_norm", "game_date"])["line"]
        .agg(min_line="min", max_line="max")
        .reset_index()
    )
    df = df.merge(lr, on=["name_norm", "game_date"], how="left")

    model    = bundle["model"]
    scaler   = bundle["scaler"]
    features = bundle["features_numeric"]
    calibs   = bundle.get("calib_models", {})

    upg = df[["name_norm", "game_date", "line"] + features].drop_duplicates(
        subset=["name_norm", "game_date", "line"]
    ).dropna(subset=features).copy()

    upg["y_hat"] = model.predict(
        scaler.transform(upg[features].values.astype(float))
    )

    rows = []
    for line, calib in calibs.items():
        sub = upg[upg["line"] == line].copy()
        if sub.empty:
            continue
        sub["p_model_over"]  = np.clip(
            calib.predict_proba(sub["y_hat"].values.reshape(-1, 1))[:, 1], 0.01, 0.99
        )
        sub["p_model_under"] = 1.0 - sub["p_model_over"]
        rows.append(sub)

    p_df = pd.concat(rows, ignore_index=True)
    df = df.merge(
        p_df[["name_norm", "game_date", "line", "y_hat", "p_model_under"]],
        on=["name_norm", "game_date", "line"], how="inner"
    )
    df["edge_under"] = df["p_model_under"] - df["raw_prob_under"]
    print(f"  Scored {len(df):,} rows")
    return df


def _odds_breakdown(sub: pd.DataFrame) -> dict:
    """Compute +/even/- split for a filtered subset."""
    if sub.empty:
        return dict(pct_plus=0, units_plus=0, pct_even=0, units_even=0,
                    pct_minus=0, units_minus=0)

    won    = (sub[TARGET] < sub["line"]).values.astype(float)
    payout = sub["under_price"].values - 1.0
    pnl    = np.where(won, payout, -1.0)

    n = len(sub)

    def _slice(mask):
        if mask.sum() == 0:
            return 0.0, 0.0
        return round(mask.sum() / n * 100, 1), round(float(pnl[mask].sum()), 2)

    plus_mask  = sub["raw_prob_under"].values < PLUS_HI
    even_mask  = (sub["raw_prob_under"].values >= PLUS_HI) & (sub["raw_prob_under"].values < MINUS_LO)
    minus_mask = sub["raw_prob_under"].values >= MINUS_LO

    pp, up   = _slice(plus_mask)
    pe, ue   = _slice(even_mask)
    pm, um   = _slice(minus_mask)

    return dict(pct_plus=pp,  units_plus=up,
                pct_even=pe,  units_even=ue,
                pct_minus=pm, units_minus=um)


def max_dd(pnl: np.ndarray) -> float:
    cum  = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum)
    return float(np.max(peak - cum)) if len(pnl) else 0.0


def run_grid(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for edge_thresh in EDGE_THRESHOLDS:
        for line_bucket in LINE_BUCKETS:
            sub = df[df["line"] == 0.5].copy() if line_bucket == "0.5_only" else \
                  df[df["line"] == 1.5].copy() if line_bucket == "1.5_only" else \
                  df.copy()
            sub = sub[sub["edge_under"] >= edge_thresh]
            if len(sub) < 10:
                continue

            won    = (sub[TARGET] < sub["line"]).values.astype(float)
            payout = sub["under_price"].values - 1.0
            pnl    = np.where(won, payout, -1.0)
            n      = len(pnl)

            rec = {
                "edge_min":    edge_thresh,
                "line_bucket": line_bucket,
                "n_bets":      n,
                "win_pct":     round(float(won.mean()), 4),
                "net_units":   round(float(pnl.sum()), 2),
                "roi_pct":     round(float(pnl.mean()) * 100, 2),
                "max_dd":      round(max_dd(pnl), 2),
                "avg_odds_am": _raw_to_am(float(sub["raw_prob_under"].mean())),
            }
            rec.update(_odds_breakdown(sub))
            records.append(rec)

    return pd.DataFrame(records).sort_values("net_units", ascending=False).reset_index(drop=True)


def _raw_to_am(p: float) -> str:
    d = 1.0 / p
    if d >= 2.0:
        return f"+{int(round((d - 1) * 100))}"
    return f"−{int(round(100 / (d - 1)))}"


# ── HTML helpers ──────────────────────────────────────────────────────────────

def _cell_units(v: float) -> str:
    cls = "color:#1a7f37;font-weight:bold" if v > 0 else "color:#d32f2f"
    return f'<td style="text-align:right;{cls}">{v:+.1f}u</td>'


def _tbl(df: pd.DataFrame) -> str:
    cols = ["edge_min", "line_bucket", "n_bets", "win_pct", "net_units", "roi_pct",
            "avg_odds_am", "max_dd",
            "pct_plus", "units_plus", "pct_even", "units_even", "pct_minus", "units_minus"]
    hdrs = ["Edge≥", "Line", "N", "Win%", "Net Units", "ROI%", "Avg Odds", "Max DD",
            "% Plus", "+Units", "% Even", "EvenUnits", "% Minus", "−Units"]

    ths = "".join(f'<th>{h}</th>' for h in hdrs)
    rows_html = ""
    for _, r in df.iterrows():
        is_prod = (r["edge_min"] == 0.05 and r["line_bucket"] == "1.5_only")
        style   = ' style="background:#e8f5e9"' if is_prod else ""
        roi_cls = "color:#1a7f37;font-weight:bold" if r["roi_pct"] > 0 else "color:#d32f2f"
        rows_html += (
            f'<tr{style}>'
            f'<td>{r["edge_min"]:.0%}</td>'
            f'<td>{r["line_bucket"]}</td>'
            f'<td style="text-align:right">{r["n_bets"]:,}</td>'
            f'<td style="text-align:right">{r["win_pct"]:.1%}</td>'
            + _cell_units(r["net_units"]) +
            f'<td style="text-align:right;{roi_cls}">{r["roi_pct"]:+.2f}%</td>'
            f'<td style="text-align:right">{r["avg_odds_am"]}</td>'
            f'<td style="text-align:right">{r["max_dd"]:.1f}u</td>'
            f'<td style="text-align:right;color:#1a7f37">{r["pct_plus"]:.1f}%</td>'
            + _cell_units(r["units_plus"]) +
            f'<td style="text-align:right;color:#888">{r["pct_even"]:.1f}%</td>'
            + _cell_units(r["units_even"]) +
            f'<td style="text-align:right;color:#d32f2f">{r["pct_minus"]:.1f}%</td>'
            + _cell_units(r["units_minus"]) +
            '</tr>'
        )
    return f'<thead><tr>{ths}</tr></thead><tbody>{rows_html}</tbody>'


def build_html(is_grid: pd.DataFrame, oos_grid: pd.DataFrame) -> str:
    ss = "-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif"
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>MLB TB Odds Breakdown Grid 2026-08-01</title>
<style>
body{{font-family:{ss};font-size:13px;color:#222;background:#f5f5f5;padding:20px}}
h1{{color:#1a1a2e;font-size:20px}} h2{{color:#1a1a2e;font-size:15px;margin-top:24px}}
p{{margin:6px 0;line-height:1.5}}
table{{border-collapse:collapse;width:100%;font-size:11px;margin-bottom:20px;background:#fff}}
th{{background:#2c3e50;color:#fff;padding:5px 7px;text-align:left;white-space:nowrap}}
td{{padding:3px 7px;border-bottom:1px solid #eee;white-space:nowrap}}
tr:hover td{{background:#f0f4ff}}
.card{{background:#fff;border-radius:8px;padding:16px 20px;margin-bottom:20px;box-shadow:0 1px 4px rgba(0,0,0,.1)}}
.legend{{font-size:11px;color:#555;margin-bottom:10px}}
</style></head><body>
<div class="card">
<h1>MLB Total Bases — IS/OOS Grid with Odds-Type Breakdown</h1>
<p>Date: 2026-08-01 · Spine: 2024–2026-07-04 · Model: XGBoost v2 + Method C calibration</p>
<p><b>New columns:</b> Each grid row now shows what % of its bets were at plus / even / minus odds
and how many units came from each bucket. Green row = current production strategy.</p>
<p class="legend">
  <span style="color:#1a7f37;font-weight:bold">Green %/units</span> = plus odds (&gt;+100) &nbsp;|&nbsp;
  <span style="color:#888">Grey</span> = even (approx −105 to +105) &nbsp;|&nbsp;
  <span style="color:#d32f2f">Red %</span> = minus odds (&lt;−100)
</p>
</div>

<div class="card">
<h2>IS — 2024+2025</h2>
<table>{_tbl(is_grid)}</table>
</div>

<div class="card">
<h2>OOS — 2026</h2>
<table>{_tbl(oos_grid)}</table>
</div>
</body></html>"""


def run_tests(df: pd.DataFrame, is_grid: pd.DataFrame, oos_grid: pd.DataFrame) -> bool:
    import duckdb
    con = duckdb.connect()
    con.register("spine", df)
    con.register("isg", is_grid)
    con.register("oosg", oos_grid)

    tests = [
        ("T1: pct_plus + pct_even + pct_minus ≈ 100 for all rows (within 0.2pp)",
         "SELECT COUNT(*) = 0 AS pass FROM isg WHERE ABS(pct_plus + pct_even + pct_minus - 100) > 0.2"),
        ("T2: same check OOS",
         "SELECT COUNT(*) = 0 AS pass FROM oosg WHERE ABS(pct_plus + pct_even + pct_minus - 100) > 0.2"),
        ("T3: units_plus + units_even + units_minus ≈ net_units (within 0.1u) IS",
         "SELECT COUNT(*) = 0 AS pass FROM isg WHERE ABS(units_plus + units_even + units_minus - net_units) > 0.1"),
        ("T4: same check OOS",
         "SELECT COUNT(*) = 0 AS pass FROM oosg WHERE ABS(units_plus + units_even + units_minus - net_units) > 0.1"),
        ("T5: production strategy row exists in OOS (edge=5%, line=1.5)",
         "SELECT COUNT(*) >= 1 AS pass FROM oosg WHERE edge_min = 0.05 AND line_bucket = '1.5_only'"),
        ("T6: pct_minus > 90 for production row (confirms most bets are minus odds)",
         "SELECT pct_minus > 90 AS pass FROM oosg WHERE edge_min = 0.05 AND line_bucket = '1.5_only' LIMIT 1"),
        ("T7: minus-odds units dominate production row (units_minus > units_plus)",
         "SELECT units_minus > units_plus AS pass FROM oosg WHERE edge_min = 0.05 AND line_bucket = '1.5_only' LIMIT 1"),
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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_and_score()

    is_df  = df[df["season"].isin(IS_SEASONS)]
    oos_df = df[df["season"] == OOS_SEASON]
    print(f"\nIS: {len(is_df):,} rows · OOS: {len(oos_df):,} rows")

    print("Running IS grid...")
    is_grid = run_grid(is_df)
    print("Running OOS grid...")
    oos_grid = run_grid(oos_df)

    is_grid.to_csv(OUT_DIR  / "mlb_tb_is_odds_breakdown_grid.csv",  index=False)
    oos_grid.to_csv(OUT_DIR / "mlb_tb_oos_odds_breakdown_grid.csv", index=False)

    # Print production strategy rows
    print("\n── Production strategy (edge≥5pp, line=1.5) ──")
    for label, g in [("IS", is_grid), ("OOS", oos_grid)]:
        row = g[(g["edge_min"] == 0.05) & (g["line_bucket"] == "1.5_only")]
        if not row.empty:
            r = row.iloc[0]
            print(f"\n  {label}: n={r['n_bets']:,}  win={r['win_pct']:.1%}  units={r['net_units']:+.1f}  ROI={r['roi_pct']:+.2f}%  avg_odds={r['avg_odds_am']}")
            print(f"       Plus odds:  {r['pct_plus']:.1f}% of bets → {r['units_plus']:+.1f}u")
            print(f"       Even odds:  {r['pct_even']:.1f}% of bets → {r['units_even']:+.1f}u")
            print(f"       Minus odds: {r['pct_minus']:.1f}% of bets → {r['units_minus']:+.1f}u")

    print("\n── Full OOS grid (top 15 by units) ──")
    print(oos_grid.head(15)[
        ["edge_min","line_bucket","n_bets","win_pct","net_units","roi_pct",
         "pct_plus","units_plus","pct_even","units_even","pct_minus","units_minus"]
    ].to_string(index=False))

    all_pass = run_tests(df, is_grid, oos_grid)
    html = build_html(is_grid, oos_grid)
    HTML_OUT.write_text(html)
    print(f"\nHTML → {HTML_OUT}")
    print("OVERALL:", "PASS ✓" if all_pass else "FAIL ✗")


if __name__ == "__main__":
    main()
