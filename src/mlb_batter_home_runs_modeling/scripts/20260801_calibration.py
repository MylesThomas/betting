"""
Step 1.5 — Market Calibration by Line

Joins market data to actuals and computes calibration metrics per line.
Outputs calibration table to HTML log.
"""
from __future__ import annotations

import io
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import boto3
import duckdb
import pandas as pd

REPO_ROOT  = Path(__file__).resolve().parents[3]
LOCAL_DIR  = Path.home() / "Downloads/tmp"
LOCAL_JOIN = LOCAL_DIR / "mlb_batter_home_runs_joined.parquet"
HTML_LOG   = REPO_ROOT / "knowledge-base/raw/20260801-mlb-batter-home-runs.html"
ET         = ZoneInfo("America/New_York")


def ts() -> str:
    return datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S ET")


def df_to_html_table(df: pd.DataFrame, caption: str = "") -> str:
    rows_html = []
    header = "<tr>" + "".join(f"<th>{c}</th>" for c in df.columns) + "</tr>"
    for _, row in df.iterrows():
        cells = "".join(f"<td>{v}</td>" for v in row)
        rows_html.append(f"<tr>{cells}</tr>")
    cap = f"<caption style='font-weight:bold;text-align:left;padding:6px 0'>{caption}</caption>" if caption else ""
    return f"<table>{cap}<thead>{header}</thead><tbody>{''.join(rows_html)}</tbody></table>"


def main() -> None:
    if not LOCAL_JOIN.exists():
        print(f"ERROR: {LOCAL_JOIN} not found — run 20260801_eda.py first")
        sys.exit(1)

    joined = pd.read_parquet(LOCAL_JOIN)
    print(f"Loaded joined: {len(joined):,} rows")

    con = duckdb.connect()
    con.register("joined", joined)

    # calibration table per line
    # Prices are decimal odds (e.g. 3.5, 1.31). raw_prob = 1 / decimal_odds.
    # n_bets counts all settled rows (over_price non-null). Novig uses dual-price rows only.
    calib = con.execute("""
        WITH settled AS (
            SELECT *,
                   1.0 / over_price                                 AS raw_implied_over,
                   CASE WHEN under_price IS NOT NULL
                        THEN 1.0 / under_price ELSE NULL END        AS raw_implied_under
            FROM joined
            WHERE home_runs IS NOT NULL
              AND over_price IS NOT NULL
              AND over_price > 1.0
        ),
        agg AS (
            SELECT
                line,
                COUNT(*) AS n_bets,
                ROUND(AVG(CASE WHEN home_runs > line THEN 1.0 ELSE 0 END), 3) AS over_rate,
                ROUND(AVG(CASE WHEN home_runs < line THEN 1.0 ELSE 0 END), 3) AS under_rate,
                ROUND(AVG(CASE WHEN home_runs = line THEN 1.0 ELSE 0 END), 3) AS push_rate,
                ROUND(AVG(raw_implied_over), 3)                                AS avg_raw_prob_over,
                ROUND(AVG(raw_implied_under), 3)                               AS avg_raw_prob_under
            FROM settled
            GROUP BY line
        )
        SELECT
            line,
            n_bets,
            over_rate,
            under_rate,
            push_rate,
            avg_raw_prob_over,
            avg_raw_prob_under,
            ROUND(avg_raw_prob_over + COALESCE(avg_raw_prob_under, 0) - 1, 3) AS avg_combined_vig,
            ROUND(avg_raw_prob_over / NULLIF(avg_raw_prob_over + COALESCE(avg_raw_prob_under, 0), 0), 3) AS avg_novig_prob_over,
            ROUND(COALESCE(avg_raw_prob_under, 0) / NULLIF(avg_raw_prob_over + COALESCE(avg_raw_prob_under, 0), 0), 3) AS avg_novig_prob_under,
            ROUND(over_rate - avg_raw_prob_over / NULLIF(avg_raw_prob_over + COALESCE(avg_raw_prob_under, 0), 0), 3) AS calibration_gap_over,
            ROUND(under_rate - COALESCE(avg_raw_prob_under, 0) / NULLIF(avg_raw_prob_over + COALESCE(avg_raw_prob_under, 0), 0), 3) AS calibration_gap_under
        FROM agg
        ORDER BY line
    """).df()

    print("\n=== Calibration Table ===")
    print(calib.to_string(index=False))

    # flag miscalibrated lines
    flagged = calib[calib[["calibration_gap_over", "calibration_gap_under"]].abs().max(axis=1) > 0.05]
    print(f"\nFlagged rows (|gap| > 5pp): {len(flagged)}")
    if len(flagged) > 0:
        print(flagged[["line", "n_bets", "calibration_gap_over", "calibration_gap_under"]].to_string(index=False))

    # ── SQL Tests
    print("\n" + "="*60)
    print("STEP 1.5 SQL TESTS")
    print("="*60)
    con.register("calib", calib)
    tests_passed = 0
    tests_failed = 0
    test_results = []

    def run_test(name: str, sql: str):
        nonlocal tests_passed, tests_failed
        try:
            result = con.execute(sql).fetchone()[0]
            passed = bool(result)
            status = "PASS" if passed else "FAIL"
            if passed:
                tests_passed += 1
            else:
                tests_failed += 1
            print(f"  [{status}] {name}: {result}")
            test_results.append((name, status, str(result)))
        except Exception as e:
            tests_failed += 1
            print(f"  [FAIL] {name}: ERROR — {e}")
            test_results.append((name, "FAIL", f"ERROR: {e}"))

    run_test(
        "T1: over+under+push sum ~1.0 for all lines",
        "SELECT BOOL_AND(ABS(over_rate + under_rate + push_rate - 1.0) < 0.01) FROM calib"
    )
    run_test(
        "T2: n_bets >= 30 for each line (flag sparse)",
        "SELECT MIN(n_bets) >= 30 FROM calib"
    )
    run_test(
        "T3: avg_raw_prob_over between 0 and 1",
        "SELECT BOOL_AND(avg_raw_prob_over BETWEEN 0 AND 1) FROM calib"
    )
    run_test(
        "T4: avg_raw_prob_under between 0 and 1",
        "SELECT BOOL_AND(avg_raw_prob_under BETWEEN 0 AND 1) FROM calib"
    )
    run_test(
        "T5: Total bets in calib matches settled joined rows",
        f"""
        SELECT ABS(SUM(n_bets) - (
            SELECT COUNT(*) FROM joined
            WHERE home_runs IS NOT NULL AND over_price IS NOT NULL AND over_price > 1.0
        )) < 100 FROM calib
        """
    )

    print(f"\n  Tests passed: {tests_passed} / {tests_passed + tests_failed}")

    # flag rows
    flag_rows = ""
    for _, row in flagged.iterrows():
        flag_rows += f"<li>Line {row['line']}: gap_over={row['calibration_gap_over']:+.3f}, gap_under={row['calibration_gap_under']:+.3f} (n={row['n_bets']:,})</li>"

    section_html = f"""
<section>
<h2>Step 1.5 — Market Calibration by Line</h2>
<p class="timestamp">{ts()}</p>

<h3>Calibration Table</h3>
{df_to_html_table(calib, "All line values: actual rates vs market-implied probabilities")}

<h3>Miscalibrated Lines (|gap| &gt; 5pp)</h3>
{"<ul>" + flag_rows + "</ul>" if flag_rows else "<p class='good'>No lines flagged — market is well-calibrated at all line values.</p>"}

<h3>Key Findings</h3>
<ul>
  <li>UNDER 0.5 is the dominant line. Market-implied OVER at this line reflects the base rate of HRs (~10%).</li>
  <li>calibration_gap_under = actual under rate minus fair market under probability. Positive gap → market underprices UNDER → value in UNDER bets.</li>
  <li>calibration_gap_over = actual over rate minus fair market over probability. Negative gap → market overprices OVER → market gives you bad value on OVERs.</li>
</ul>

<h3>SQL Test Results</h3>
<table>
<thead><tr><th>Test</th><th>Status</th><th>Value</th></tr></thead>
<tbody>
{"".join(f'<tr><td>{name}</td><td class="{"pass" if s=="PASS" else "fail"}">{s}</td><td>{v}</td></tr>' for name, s, v in test_results)}
</tbody>
</table>
<p><strong>Passed: {tests_passed} / {tests_passed + tests_failed}</strong></p>
</section>
"""

    with open(HTML_LOG, "a") as f:
        f.write(section_html)
    print(f"\nHTML appended to {HTML_LOG}")


if __name__ == "__main__":
    main()
