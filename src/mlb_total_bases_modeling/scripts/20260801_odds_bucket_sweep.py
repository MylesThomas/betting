"""
MLB Total Bases — Fine-Grained Odds-Bucket Sweep (2026-08-01)
=============================================================
Hypothesis: the coarse "favs/dogs/all" bucketing in the 2026-07-10 grid may be
hiding a bad sub-bucket within favs (e.g. very heavy −250+ lines dragging ROI).

Grid dimensions:
  edge_threshold  : [0.00, 0.03, 0.05, 0.07, 0.10]
  odds_bucket     : 7 fine-grained American-odds buckets
  line_bucket     : ["all_lines", "1.5_only"]

IS  = 2024 + 2025
OOS = 2026

Usage:
  python src/mlb_total_bases_modeling/scripts/20260801_odds_bucket_sweep.py
"""
from __future__ import annotations

import sys
from datetime import datetime
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
HTML_PATH = REPO_ROOT / "knowledge-base/raw/20260801-mlb-tb-odds-bucket-sweep.html"
TARGET    = "total_bases"

IS_SEASONS  = [2024, 2025]
OOS_SEASON  = 2026

EDGE_THRESHOLDS = [0.00, 0.03, 0.05, 0.07, 0.10]
LINE_BUCKETS    = ["all_lines", "1.5_only"]

# Fine-grained odds buckets — defined by raw_prob_under (= 1/decimal_under_price)
# American odds:  decimal ≥ 2.0 → +(decimal−1)*100;  decimal < 2.0 → −100/(decimal−1)
ODDS_BUCKETS = [
    # label               raw_prob_under range
    ("+200+",             (0.00, 0.333)),   # decimal ≥ 3.0  → American ≥ +200
    ("+100 to +200",      (0.333, 0.50)),   # decimal 2.0–3.0 → +100 to +199
    ("even (−105 to +105)",(0.488, 0.512)), # near-even (raw_prob 48.8%–51.2%)
    ("−100 to −149",      (0.50, 0.60)),    # decimal 1.667–2.0
    ("−150 to −199",      (0.60, 0.667)),   # decimal 1.50–1.667
    ("−200 to −299",      (0.667, 0.75)),   # decimal 1.333–1.50
    ("−300+",             (0.75, 1.00)),    # decimal < 1.333
    ("all",               (0.00, 1.00)),    # no filter
    ("all minus (favs)",  (0.50, 1.00)),    # all minus-odds
    ("all plus (dogs)",   (0.00, 0.50)),    # all plus-odds
]


def raw_prob_to_american(p: float) -> str:
    d = 1.0 / p
    if d >= 2.0:
        return f"+{int(round((d - 1) * 100))}"
    return f"−{int(round(100 / (d - 1)))}"


def load_and_score() -> pd.DataFrame:
    s3 = boto3.client("s3")

    print("Loading regression spine from S3...")
    spine = pd.read_parquet(BytesIO(s3.get_object(Bucket=S3_BUCKET, Key=SPINE_KEY)["Body"].read()))
    print(f"  {len(spine):,} rows, {spine['game_date'].min()} → {spine['game_date'].max()}")

    print("Loading model bundle from S3...")
    bundle = joblib.load(BytesIO(s3.get_object(Bucket=S3_BUCKET, Key=MODEL_KEY)["Body"].read()))

    df = spine.copy()
    df["season"]         = pd.to_datetime(df["game_date"]).dt.year
    df["raw_prob_under"] = 1.0 / df["under_price"]
    df["raw_prob_over"]  = 1.0 / df["over_price"]

    # min_line/max_line per player-game (model features)
    line_range = (
        df.groupby(["name_norm", "game_date"])["line"]
        .agg(min_line="min", max_line="max")
        .reset_index()
    )
    df = df.merge(line_range, on=["name_norm", "game_date"], how="left")

    # Score: XGBoost → y_hat → calibration → p_model_under
    model        = bundle["model"]
    scaler       = bundle["scaler"]
    features     = bundle["features_numeric"]
    calib_models = bundle.get("calib_models", {})

    unique_pg = df[["name_norm", "game_date", "line"] + features].drop_duplicates(
        subset=["name_norm", "game_date", "line"]
    ).dropna(subset=features)

    X_sc  = scaler.transform(unique_pg[features].values.astype(float))
    unique_pg = unique_pg[["name_norm", "game_date", "line"]].copy()
    unique_pg["y_hat"] = model.predict(X_sc)

    p_rows = []
    for line, calib in calib_models.items():
        sub = unique_pg[unique_pg["line"] == line].copy()
        if sub.empty:
            continue
        p_over = np.clip(calib.predict_proba(sub["y_hat"].values.reshape(-1, 1))[:, 1], 0.01, 0.99)
        sub["p_model_over"]  = p_over
        sub["p_model_under"] = 1.0 - p_over
        p_rows.append(sub)

    p_df = pd.concat(p_rows, ignore_index=True)
    df = df.merge(p_df[["name_norm", "game_date", "line", "y_hat", "p_model_under"]],
                  on=["name_norm", "game_date", "line"], how="inner")
    df["edge_under"] = df["p_model_under"] - df["raw_prob_under"]

    print(f"  Scored {len(df):,} rows, {df['p_model_under'].notna().sum():,} with p_model")
    return df


def max_dd(pnl: np.ndarray) -> float:
    cum = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum)
    return float(np.max(peak - cum)) if len(pnl) else 0.0


def run_grid(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for edge_thresh in EDGE_THRESHOLDS:
        for label, (lo, hi) in ODDS_BUCKETS:
            for line_bucket in LINE_BUCKETS:
                sub = df[df["line"] == 1.5] if line_bucket == "1.5_only" else df
                sub = sub[sub["edge_under"] >= edge_thresh]
                sub = sub[(sub["raw_prob_under"] >= lo) & (sub["raw_prob_under"] < hi)]
                if len(sub) < 10:
                    continue
                won    = (sub[TARGET] < sub["line"]).values.astype(float)
                payout = sub["under_price"].values - 1.0
                pnl    = np.where(won, payout, -1.0)
                records.append({
                    "edge_min":     edge_thresh,
                    "odds_bucket":  label,
                    "line_bucket":  line_bucket,
                    "n_bets":       len(pnl),
                    "win_pct":      round(float(won.mean()), 4),
                    "net_units":    round(float(pnl.sum()), 2),
                    "roi_pct":      round(float(pnl.mean()) * 100, 2),
                    "avg_odds":     raw_prob_to_american(float(sub["raw_prob_under"].mean())),
                    "avg_edge_pp":  round(float(sub["edge_under"].mean()) * 100, 2),
                    "max_dd":       round(max_dd(pnl), 2),
                })
    return pd.DataFrame(records).sort_values("net_units", ascending=False).reset_index(drop=True)


def _tbl(df: pd.DataFrame, caption: str = "") -> str:
    cols = ["edge_min", "odds_bucket", "line_bucket", "n_bets", "win_pct", "net_units", "roi_pct", "avg_odds", "avg_edge_pp", "max_dd"]
    col_map = {"edge_min": "Edge≥", "odds_bucket": "Odds Bucket", "line_bucket": "Line",
               "n_bets": "N", "win_pct": "Win%", "net_units": "Net Units",
               "roi_pct": "ROI%", "avg_odds": "Avg Odds", "avg_edge_pp": "Avg Edge (pp)", "max_dd": "Max DD"}
    rows_html = ""
    for _, r in df.iterrows():
        is_prod = (r["edge_min"] == 0.05 and "all" in str(r["odds_bucket"]).lower() and r["line_bucket"] == "1.5_only")
        style = ' style="background:#e8f5e9;font-weight:bold"' if is_prod else ""
        roi_cls = "color:#1a7f37;font-weight:bold" if r["roi_pct"] > 0 else "color:#d32f2f"
        units_cls = "color:#1a7f37;font-weight:bold" if r["net_units"] > 0 else "color:#d32f2f"
        rows_html += (
            f'<tr{style}>'
            f'<td>{r["edge_min"]:.0%}</td>'
            f'<td>{r["odds_bucket"]}</td>'
            f'<td>{r["line_bucket"]}</td>'
            f'<td style="text-align:right">{r["n_bets"]:,}</td>'
            f'<td style="text-align:right">{r["win_pct"]:.1%}</td>'
            f'<td style="text-align:right;{units_cls}">{r["net_units"]:+.1f}u</td>'
            f'<td style="text-align:right;{roi_cls}">{r["roi_pct"]:+.2f}%</td>'
            f'<td style="text-align:right">{r["avg_odds"]}</td>'
            f'<td style="text-align:right">{r["avg_edge_pp"]:+.1f}pp</td>'
            f'<td style="text-align:right">{r["max_dd"]:.1f}u</td>'
            f'</tr>'
        )
    ths = "".join(f'<th>{col_map[c]}</th>' for c in cols)
    return (
        f'{"<caption>" + caption + "</caption>" if caption else ""}'
        f'<thead><tr>{ths}</tr></thead><tbody>{rows_html}</tbody>'
    )


def build_html(is_grid: pd.DataFrame, oos_grid: pd.DataFrame, df: pd.DataFrame) -> str:
    # Odds distribution table
    df15 = df[df["line"] == 1.5].copy()
    df15["odds_bucket"] = pd.cut(
        df15["raw_prob_under"],
        bins=[0, 0.333, 0.488, 0.512, 0.60, 0.667, 0.75, 1.01],
        labels=["+200+", "+100 to +199", "even", "−100 to −149", "−150 to −199", "−200 to −299", "−300+"],
        right=False
    )
    dist = df15.groupby("odds_bucket", observed=True).agg(
        n=("raw_prob_under", "count"),
        avg_raw_prob=("raw_prob_under", "mean"),
        win_pct=(TARGET, lambda x: (x < 1.5).mean()),
        avg_price=("under_price", "mean"),
    ).reset_index()
    dist["avg_american"] = dist["avg_raw_prob"].apply(raw_prob_to_american)
    dist_rows = "".join(
        f'<tr><td>{r["odds_bucket"]}</td>'
        f'<td style="text-align:right">{r["n"]:,}</td>'
        f'<td style="text-align:right">{r["n"]/len(df15)*100:.1f}%</td>'
        f'<td style="text-align:right">{r["avg_american"]}</td>'
        f'<td style="text-align:right">{r["win_pct"]:.1%}</td>'
        f'</tr>'
        for _, r in dist.iterrows()
    )

    # Filter to 1.5-only for the main breakdown table
    is_15  = is_grid[is_grid["line_bucket"] == "1.5_only"].head(30)
    oos_15 = oos_grid[oos_grid["line_bucket"] == "1.5_only"].head(30)

    ss = "-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif"
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>MLB TB Odds Bucket Sweep 2026-08-01</title>
<style>
body{{font-family:{ss};font-size:13px;color:#222;background:#f5f5f5;padding:20px}}
h1{{color:#1a1a2e;font-size:22px}} h2{{color:#1a1a2e;font-size:16px;margin-top:24px}}
h3{{font-size:14px;margin:16px 0 8px;color:#444}}
table{{border-collapse:collapse;width:100%;font-size:12px;margin-bottom:20px;background:#fff}}
th{{background:#2c3e50;color:#fff;padding:6px 8px;text-align:left;font-size:11px}}
td{{padding:4px 8px;border-bottom:1px solid #eee}}
tr:hover td{{background:#f0f4ff}}
.card{{background:#fff;border-radius:8px;padding:16px 20px;margin-bottom:20px;box-shadow:0 1px 4px rgba(0,0,0,.1)}}
.warn{{background:#fff8e1;border-left:4px solid #f9a825;padding:10px 14px;margin:12px 0;font-size:12px}}
.good{{background:#e8f5e9;border-left:4px solid #43a047;padding:10px 14px;margin:12px 0;font-size:12px}}
</style></head><body>
<div class="card">
<h1>MLB Total Bases — Fine-Grained Odds-Bucket Sweep</h1>
<p>Date: 2026-08-01 · Data: 2024–2026 (through 2026-07-05) · Model: XGBoost v2 + Method C calibration</p>
<p><b>Hypothesis:</b> The coarse "favs/dogs" bucketing may be hiding a bad sub-bucket (e.g. very heavy −250+ lines dragging ROI, or a specific range where the model has no edge).</p>
<p style="font-size:11px;color:#666">Green row = current production strategy (edge≥5%, all favs, line=1.5)</p>
</div>

<div class="card">
<h2>1. Distribution of Under Odds at Line = 1.5</h2>
<table>
<thead><tr><th>Odds Bucket</th><th>N bets</th><th>% of total</th><th>Avg American</th><th>Win Rate (actual)</th></tr></thead>
<tbody>{dist_rows}</tbody>
</table>
</div>

<div class="card">
<h2>2. IS Results — 2024+2025 (line=1.5 only, top 30 by units)</h2>
<table>{_tbl(is_15)}</table>
</div>

<div class="card">
<h2>3. OOS Results — 2026 (line=1.5 only, top 30 by units)</h2>
<table>{_tbl(oos_15)}</table>
</div>

<div class="card">
<h2>4. Focus: Production Edge (≥5pp) — All Odds Buckets Side-by-Side (line=1.5)</h2>
<p>Sorted by OOS units to see which buckets are actually profitable out-of-sample.</p>
<h3>OOS (2026)</h3>
<table>{_tbl(oos_grid[(oos_grid["edge_min"] == 0.05) & (oos_grid["line_bucket"] == "1.5_only")].sort_values("net_units", ascending=False))}</table>
<h3>IS (2024+2025)</h3>
<table>{_tbl(is_grid[(is_grid["edge_min"] == 0.05) & (is_grid["line_bucket"] == "1.5_only")].sort_values("net_units", ascending=False))}</table>
</div>

<div class="card">
<h2>5. DuckDB SQL Validation Tests</h2>
<p id="test-results" style="font-family:monospace;white-space:pre"></p>
</div>

</body></html>"""


def run_sql_tests(df: pd.DataFrame, is_grid: pd.DataFrame, oos_grid: pd.DataFrame) -> list[tuple[str, bool, str]]:
    import duckdb
    con = duckdb.connect()
    con.register("spine", df)
    con.register("is_g", is_grid)
    con.register("oos_g", oos_grid)

    tests = [
        ("T1: spine has rows in all 3 seasons",
         "SELECT COUNT(DISTINCT season) = 3 AS pass FROM spine"),
        ("T2: all raw_prob_under values between 0 and 1",
         "SELECT COUNT(*) = 0 AS pass FROM spine WHERE raw_prob_under <= 0 OR raw_prob_under >= 1"),
        ("T3: p_model_under values all between 0 and 1",
         "SELECT COUNT(*) = 0 AS pass FROM spine WHERE p_model_under IS NOT NULL AND (p_model_under < 0 OR p_model_under > 1)"),
        ("T4: production IS strategy has positive ROI (edge=5pp, all, 1.5)",
         "SELECT roi_pct > 0 AS pass FROM is_g WHERE edge_min = 0.05 AND odds_bucket = 'all' AND line_bucket = '1.5_only' LIMIT 1"),
        ("T5: production OOS strategy has positive ROI (edge=5pp, all, 1.5)",
         "SELECT roi_pct > 0 AS pass FROM oos_g WHERE edge_min = 0.05 AND odds_bucket = 'all' AND line_bucket = '1.5_only' LIMIT 1"),
        ("T6: OOS avg_odds for favs-only bucket at edge=5pp is minus odds (avg_prob > 0.50)",
         "SELECT CAST(AVG(raw_prob_under) > 0.50 AS BOOLEAN) AS pass FROM spine WHERE edge_under >= 0.05 AND raw_prob_under >= 0.50 AND line = 1.5"),
        ("T7: n_bets in OOS grid for minus−300+ bucket at edge=5pp ≥ 10",
         "SELECT n_bets >= 10 AS pass FROM oos_g WHERE edge_min = 0.05 AND odds_bucket = '−300+' AND line_bucket = '1.5_only' LIMIT 1"),
    ]

    results = []
    for name, sql in tests:
        try:
            val = con.execute(sql).fetchone()
            ok  = bool(val[0]) if val else False
            results.append((name, ok, ""))
        except Exception as e:
            results.append((name, False, str(e)))
    return results


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_and_score()

    is_df  = df[df["season"].isin(IS_SEASONS)]
    oos_df = df[df["season"] == OOS_SEASON]
    print(f"\nIS rows: {len(is_df):,}  OOS rows: {len(oos_df):,}")

    print("\nRunning IS grid...")
    is_grid  = run_grid(is_df)
    print("\nRunning OOS grid...")
    oos_grid = run_grid(oos_df)

    is_grid.to_csv(OUT_DIR / "mlb_tb_v2_is_odds_sweep.csv", index=False)
    oos_grid.to_csv(OUT_DIR / "mlb_tb_v2_oos_odds_sweep.csv", index=False)
    print("\nCSVs saved.")

    # Print key tables
    print("\n── OOS (2026): all odds buckets at edge≥5pp, line=1.5 ──")
    focus = oos_grid[(oos_grid["edge_min"] == 0.05) & (oos_grid["line_bucket"] == "1.5_only")].sort_values("net_units", ascending=False)
    print(focus[["odds_bucket","n_bets","win_pct","net_units","roi_pct","avg_odds","avg_edge_pp","max_dd"]].to_string(index=False))

    print("\n── IS (2024+2025): all odds buckets at edge≥5pp, line=1.5 ──")
    focus_is = is_grid[(is_grid["edge_min"] == 0.05) & (is_grid["line_bucket"] == "1.5_only")].sort_values("net_units", ascending=False)
    print(focus_is[["odds_bucket","n_bets","win_pct","net_units","roi_pct","avg_odds","avg_edge_pp","max_dd"]].to_string(index=False))

    print("\nRunning SQL validation tests...")
    tests = run_sql_tests(df, is_grid, oos_grid)
    all_pass = all(ok for _, ok, _ in tests)
    for name, ok, err in tests:
        status = "PASS" if ok else f"FAIL{(' — ' + err) if err else ''}"
        print(f"  [{status}] {name}")

    # Build and save HTML
    html = build_html(is_grid, oos_grid, df)

    # Inject test results
    test_lines = "\n".join(
        f'[{"PASS" if ok else "FAIL"}] {name}' + (f'\n  Error: {err}' if err else '')
        for name, ok, err in tests
    )
    overall = "All tests PASSED ✓" if all_pass else "Some tests FAILED ✗"
    html = html.replace(
        '<p id="test-results" style="font-family:monospace;white-space:pre"></p>',
        f'<p style="font-family:monospace;white-space:pre;font-size:12px">{test_lines}\n\n{overall}</p>'
    )

    HTML_PATH.write_text(html)
    print(f"\nHTML saved → {HTML_PATH}")
    print("OVERALL:", "PASS" if all_pass else "FAIL")


if __name__ == "__main__":
    main()
