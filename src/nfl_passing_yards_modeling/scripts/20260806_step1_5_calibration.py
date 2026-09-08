"""
Step 1.5 — Market Calibration by Line

Join BetOnline pass_yds lines to actual passing yards (nfl_data_py),
compute over/under/push outcomes, and build calibration table by line bucket.

Appends a new section to knowledge-base/raw/20260806-nfl-qb-pass-yds.html.

Usage:
  uv run python 20260806_step1_5_calibration.py
"""

from __future__ import annotations

import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

ET = ZoneInfo("America/New_York")

TMP_DIR   = Path.home() / "Downloads" / "tmp" / "pass_yds"
HTML_PATH = REPO_ROOT / "knowledge-base" / "raw" / "20260806-nfl-qb-pass-yds.html"
BOL_PATH  = TMP_DIR / "step1_lines_betonline.parquet"
FEAT_PATH = TMP_DIR / "weekly_player_data.parquet"
OUT_PATH  = TMP_DIR / "step1_5_calibration.parquet"


def ts() -> str:
    return datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S ET")


def df_to_html(df: pd.DataFrame, title: str = "") -> str:
    out = ""
    if title:
        out += f'<p><strong>{title}</strong></p>'
    out += '<table border="1" cellpadding="4" cellspacing="0" style="border-collapse:collapse;font-size:13px;font-family:monospace">'
    out += "<thead><tr>" + "".join(
        f"<th style='background:#1a1a2e;color:white;padding:6px 10px'>{c}</th>" for c in df.columns
    ) + "</tr></thead><tbody>"
    for i, row in df.iterrows():
        bg = "#f9f9f9" if i % 2 == 0 else "white"
        out += f"<tr style='background:{bg}'>" + "".join(
            f"<td style='padding:4px 8px'>{v}</td>" for v in row.values
        ) + "</tr>"
    out += "</tbody></table>"
    return out


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    n = unicodedata.normalize("NFKD", name)
    n = "".join(c for c in n if not unicodedata.combining(c))
    n = n.lower()
    n = re.sub(r"[''`]", "", n)
    n = re.sub(r"[^a-z0-9 ]", " ", n)
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


# ── Load data ─────────────────────────────────────────────────────────────────

print("[Step 1.5] Loading BetOnline lines...")
bol = pd.read_parquet(BOL_PATH)
# Reg season only (week 1–18)
bol = bol[bol["nfl_week"] <= 18].copy()
print(f"  BetOnline reg-season rows: {len(bol):,}")

print("[Step 1.5] Loading feature data (actuals)...")
feat = pd.read_parquet(FEAT_PATH)
qbs  = feat[(feat["position"] == "QB") & (feat["attempts"] > 0)].copy()
print(f"  QB game rows: {len(qbs):,}")

# ── Normalize names for join ──────────────────────────────────────────────────

bol["player_norm"]  = bol["player_name"].map(normalize_name)
qbs["player_norm"]  = qbs["player_display_name"].map(normalize_name)

# ── Join ──────────────────────────────────────────────────────────────────────

print("[Step 1.5] Joining lines to actuals on (player_norm, season, week)...")

joined = bol.merge(
    qbs[["player_norm", "season", "week", "passing_yards", "player_display_name", "recent_team"]],
    left_on  =["player_norm", "nfl_season", "nfl_week"],
    right_on =["player_norm", "season",     "week"],
    how="left",
)

n_total   = len(joined)
n_matched = joined["passing_yards"].notna().sum()
match_pct = n_matched / n_total * 100
print(f"  Join: {n_matched:,} / {n_total:,} rows matched actuals ({match_pct:.1f}%)")

# Flag unmatched names for review
unmatched_names = (
    joined[joined["passing_yards"].isna()]["player_name"]
    .value_counts()
    .head(20)
    .reset_index()
)
unmatched_names.columns = ["player_name", "n_unmatched"]
print(f"\nTop unmatched player names:")
print(unmatched_names.head(10).to_string(index=False))

# ── Compute outcome ───────────────────────────────────────────────────────────

df = joined[joined["passing_yards"].notna()].copy()

df["outcome"] = "push"
df.loc[df["passing_yards"] > df["line"], "outcome"] = "over"
df.loc[df["passing_yards"] < df["line"], "outcome"] = "under"

over_rate  = (df["outcome"] == "over").mean()
under_rate = (df["outcome"] == "under").mean()
push_rate  = (df["outcome"] == "push").mean()
print(f"\nOverall outcomes: over={over_rate:.3f}  under={under_rate:.3f}  push={push_rate:.4f}")

# ── Calibration by line ───────────────────────────────────────────────────────
# Round line to nearest 5 for bucketing (lots of distinct lines)
df["line_bucket"] = (df["line"] / 10).round() * 10

calib = (
    df.groupby("line")
    .agg(
        n_bets              =("outcome",         "count"),
        over_rate           =("outcome",         lambda x: (x == "over").mean()),
        under_rate          =("outcome",         lambda x: (x == "under").mean()),
        push_rate           =("outcome",         lambda x: (x == "push").mean()),
        avg_raw_prob_over   =("raw_prob_over",   "mean"),
        avg_raw_prob_under  =("raw_prob_under",  "mean"),
        avg_novig_prob_over =("novig_prob_over",  "mean"),
        avg_novig_prob_under=("novig_prob_under", "mean"),
    )
    .reset_index()
    .sort_values("line")
)

calib["avg_combined_vig"]    = calib["avg_raw_prob_over"] + calib["avg_raw_prob_under"] - 1
calib["calibration_gap_over"]  = calib["over_rate"]  - calib["avg_novig_prob_over"]
calib["calibration_gap_under"] = calib["under_rate"] - calib["avg_novig_prob_under"]

# Round display
for col in ["over_rate","under_rate","push_rate","avg_raw_prob_over","avg_raw_prob_under",
            "avg_combined_vig","avg_novig_prob_over","avg_novig_prob_under",
            "calibration_gap_over","calibration_gap_under"]:
    calib[col] = calib[col].round(3)

# 10-yard bucket summary (more stable with low n at each exact line)
calib_bucketed = (
    df.groupby("line_bucket")
    .agg(
        n_bets              =("outcome",         "count"),
        over_rate           =("outcome",         lambda x: (x == "over").mean()),
        under_rate          =("outcome",         lambda x: (x == "under").mean()),
        push_rate           =("outcome",         lambda x: (x == "push").mean()),
        avg_raw_prob_over   =("raw_prob_over",   "mean"),
        avg_raw_prob_under  =("raw_prob_under",  "mean"),
        avg_novig_prob_over =("novig_prob_over",  "mean"),
        avg_novig_prob_under=("novig_prob_under", "mean"),
    )
    .reset_index()
    .sort_values("line_bucket")
)
calib_bucketed["avg_combined_vig"]     = calib_bucketed["avg_raw_prob_over"] + calib_bucketed["avg_raw_prob_under"] - 1
calib_bucketed["calibration_gap_over"]  = calib_bucketed["over_rate"]  - calib_bucketed["avg_novig_prob_over"]
calib_bucketed["calibration_gap_under"] = calib_bucketed["under_rate"] - calib_bucketed["avg_novig_prob_under"]

for col in ["over_rate","under_rate","push_rate","avg_raw_prob_over","avg_raw_prob_under",
            "avg_combined_vig","avg_novig_prob_over","avg_novig_prob_under",
            "calibration_gap_over","calibration_gap_under"]:
    calib_bucketed[col] = calib_bucketed[col].round(3)

print("\nCalibration by 10-yd bucket:")
print(calib_bucketed.to_string(index=False))

# Flag buckets with large miscalibration
flagged = calib_bucketed[
    (calib_bucketed["calibration_gap_over"].abs() > 0.05) |
    (calib_bucketed["calibration_gap_under"].abs() > 0.05)
]
print(f"\nMiscalibrated buckets (|gap| > 5pp): {len(flagged)}")
if len(flagged):
    print(flagged[["line_bucket","n_bets","over_rate","avg_novig_prob_over","calibration_gap_over","calibration_gap_under"]].to_string(index=False))

# Spot-check Josh Allen
allen = df[df["player_name"].str.contains("Josh Allen", na=False, case=False)]
print(f"\nJosh Allen calibration rows: {len(allen)}")
print(allen[["nfl_season","nfl_week","line","passing_yards","outcome","novig_prob_over"]].head(10).to_string(index=False))

df.to_parquet(OUT_PATH, index=False)
print(f"\nSaved joined calibration data to: {OUT_PATH}")


# ── DuckDB tests ──────────────────────────────────────────────────────────────

print("\n[Step 1.5] Running DuckDB tests...")

con = duckdb.connect()
con.execute(f"CREATE TABLE cal AS SELECT * FROM read_parquet('{OUT_PATH}')")

tests = []

def run_test(name, sql, expect_zero=True):
    result = con.execute(sql).fetchone()[0]
    passed = (result == 0) if expect_zero else (result > 0)
    status = "PASS" if passed else "FAIL"
    tests.append({"Test": name, "Result": result, "Status": status})
    icon = "✓" if passed else "✗"
    print(f"  {icon} [{status}] {name}: {result}")
    return passed

# T1: each outcome row sums to ~1.0 — check via SQL on bucketed agg
run_test(
    "outcome rates sum to ~1.0 (within 1%)",
    """SELECT COUNT(*) FROM (
        SELECT line,
               SUM(CASE WHEN outcome='over' THEN 1 ELSE 0 END)*1.0/COUNT(*) AS o,
               SUM(CASE WHEN outcome='under' THEN 1 ELSE 0 END)*1.0/COUNT(*) AS u,
               SUM(CASE WHEN outcome='push' THEN 1 ELSE 0 END)*1.0/COUNT(*) AS p
        FROM cal GROUP BY line
    ) WHERE ABS(o+u+p - 1.0) > 0.01""",
    expect_zero=True,
)

# T2: flag line values with < 30 obs
low_n = con.execute(
    "SELECT COUNT(*) FROM (SELECT line, COUNT(*) n FROM cal GROUP BY line HAVING n < 30)"
).fetchone()[0]
tests.append({"Test": "Lines with <30 observations (flag only)", "Result": low_n, "Status": "FLAG"})
print(f"  ~ [FLAG] Lines with <30 observations: {low_n}")

# T3: raw_prob_over and raw_prob_under in (0,1)
run_test("raw_prob_over in (0,1)", "SELECT COUNT(*) FROM cal WHERE raw_prob_over <= 0 OR raw_prob_over >= 1", expect_zero=True)
run_test("raw_prob_under in (0,1)", "SELECT COUNT(*) FROM cal WHERE raw_prob_under <= 0 OR raw_prob_under >= 1", expect_zero=True)

# T4: total bets matches expected matched row count
actual_rows = con.execute("SELECT COUNT(*) FROM cal").fetchone()[0]
run_test(
    f"Total bets in cal = {n_matched} (matched join count)",
    f"SELECT CASE WHEN COUNT(*) = {n_matched} THEN 0 ELSE 1 END FROM cal",
    expect_zero=True,
)

test_results = pd.DataFrame(tests)
n_pass = (test_results["Status"] == "PASS").sum()
n_fail = (test_results["Status"] == "FAIL").sum()
print(f"\n  Tests: {n_pass} passed, {n_fail} failed")


# ── HTML section ──────────────────────────────────────────────────────────────

print("\n[Step 1.5] Writing HTML section...")

# Miscalibrated buckets for flags section
misc_rows = calib_bucketed[
    (calib_bucketed["calibration_gap_over"].abs() > 0.05) |
    (calib_bucketed["calibration_gap_under"].abs() > 0.05)
]

html = f"""
<section>
<h2>Step 1.5 — Market Calibration by Line</h2>
<p><em>{ts()}</em></p>

<h3>Join Quality</h3>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>BetOnline reg-season rows</td><td>{n_total:,}</td></tr>
<tr><td>Rows matched to actuals</td><td>{n_matched:,}</td></tr>
<tr><td>Match rate</td><td><strong>{match_pct:.1f}%</strong></td></tr>
<tr><td>Overall over rate</td><td>{over_rate:.3f}</td></tr>
<tr><td>Overall under rate</td><td>{under_rate:.3f}</td></tr>
<tr><td>Overall push rate</td><td>{push_rate:.4f}</td></tr>
</table>
"""

if len(unmatched_names):
    html += df_to_html(unmatched_names, "Top unmatched player names (check these)")

html += f"""
<h3>Calibration by 10-Yard Bucket</h3>
<p>Each row is a 10-yard bucket of lines (e.g. 220 = lines 215–224.5).
<strong>calibration_gap_over > 0</strong> means actuals beat the fair price → market under-priced overs.
<strong>calibration_gap_over &lt; 0</strong> means market over-priced overs.
Buckets flagged where |gap| &gt; 5pp.</p>
"""

# Highlight flagged rows
calib_bucketed_display = calib_bucketed.copy()
html += '<table border="1" cellpadding="4" cellspacing="0" style="border-collapse:collapse;font-size:13px;font-family:monospace">'
html += "<thead><tr>" + "".join(
    f"<th style='background:#1a1a2e;color:white;padding:6px 10px'>{c}</th>" for c in calib_bucketed.columns
) + "</tr></thead><tbody>"
for i, row in calib_bucketed.iterrows():
    is_flagged = abs(row["calibration_gap_over"]) > 0.05 or abs(row["calibration_gap_under"]) > 0.05
    bg = "#fff3cd" if is_flagged else ("#f9f9f9" if i % 2 == 0 else "white")
    html += f"<tr style='background:{bg}'>" + "".join(
        f"<td style='padding:4px 8px'>{v}</td>" for v in row.values
    ) + "</tr>"
html += "</tbody></table>"

html += f"""
<h3>Exact-Line Calibration (top 30 by n_bets)</h3>
"""
calib_top = calib.sort_values("n_bets", ascending=False).head(30).sort_values("line")
html += df_to_html(calib_top, "Most-observed exact line values — calibration detail")

html += f"""
<h3>Spot-Check — Josh Allen</h3>
"""
allen_display = allen[["nfl_season","nfl_week","line","passing_yards","outcome","novig_prob_over","novig_prob_under"]].copy().round(3)
html += df_to_html(allen_display, f"Josh Allen ({len(allen)} rows) — lines vs actuals")

html += f"""
<h3>Step 1.5 Test Results</h3>
<table>
<tr><th>Test</th><th>Result</th><th>Status</th></tr>
"""
for _, row in test_results.iterrows():
    color = "#d4edda" if row["Status"] == "PASS" else ("#fff3cd" if row["Status"] == "FLAG" else "#f8d7da")
    html += f"<tr style='background:{color}'><td>{row['Test']}</td><td>{row['Result']}</td><td><strong>{row['Status']}</strong></td></tr>"

html += f"""
</table>
<p><strong>{n_pass} passed, {n_fail} failed</strong></p>

<h3>Flagged Items</h3>
<ul>
"""

flags = []

if match_pct < 90:
    flags.append(f"CRITICAL: join match rate {match_pct:.1f}% is below 90% — check name normalization")
elif match_pct < 95:
    flags.append(f"Join match rate {match_pct:.1f}% — below 95%, investigate top unmatched names")

if len(misc_rows) > 0:
    flags.append(f"{len(misc_rows)} line buckets with |calibration_gap| > 5pp — see table above (highlighted yellow)")
    for _, r in misc_rows.iterrows():
        flags.append(
            f"  Bucket {r['line_bucket']:.0f}: n={r['n_bets']}, over_rate={r['over_rate']:.3f}, "
            f"gap_over={r['calibration_gap_over']:+.3f}, gap_under={r['calibration_gap_under']:+.3f}"
        )

if over_rate > 0.52:
    flags.append(f"Overall over rate {over_rate:.3f} > 52% — overs hit more than expected; check if systematic")
elif under_rate > 0.52:
    flags.append(f"Overall under rate {under_rate:.3f} > 52% — potential UNDER edge")

if not flags:
    flags.append("No major flags — calibration looks reasonable")

for f in flags:
    html += f"<li>{f}</li>"

html += "</ul></section>"

with open(HTML_PATH, "a", encoding="utf-8") as f:
    f.write(html)

print(f"[Step 1.5] HTML section appended: {HTML_PATH}")
print("\n=== DONE ===")
