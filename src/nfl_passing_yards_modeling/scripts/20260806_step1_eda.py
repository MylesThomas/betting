"""
Step 0 + Step 1 — Scope confirmation, HTML log creation, and EDA.

NFL QB Passing Yards pipeline research session.
HTML log: knowledge-base/raw/20260806-nfl-qb-pass-yds.html
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

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

ET = ZoneInfo("America/New_York")

S3_BUCKET   = "the-odds-api-mt"
S3_PREFIX   = "nfl/pass_yds_model"
SEASONS     = [2023, 2024, 2025]
MARKET      = "player_pass_yds"
FOCUS_BOOK  = "betonlineag"
PLAYER_DATA = Path.home() / "Downloads" / "tmp" / "pass_yds" / "weekly_player_data.parquet"
HTML_PATH   = REPO_ROOT / "knowledge-base" / "raw" / "20260806-nfl-qb-pass-yds.html"
TMP_DIR     = Path.home() / "Downloads" / "tmp" / "pass_yds"
TMP_DIR.mkdir(parents=True, exist_ok=True)

SPOT_CHECK_PLAYER = "Josh Allen"

s3 = boto3.client("s3")


def ts() -> str:
    return datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S ET")


def df_to_html(df: pd.DataFrame, title: str = "") -> str:
    rows = ""
    if title:
        rows += f'<p><strong>{title}</strong></p>'
    rows += '<table border="1" cellpadding="4" cellspacing="0" style="border-collapse:collapse;font-size:13px;font-family:monospace">'
    rows += "<thead><tr>" + "".join(f"<th style='background:#1a1a2e;color:white;padding:6px 10px'>{c}</th>" for c in df.columns) + "</tr></thead>"
    rows += "<tbody>"
    for i, row in df.iterrows():
        bg = "#f9f9f9" if i % 2 == 0 else "white"
        rows += f"<tr style='background:{bg}'>" + "".join(f"<td style='padding:4px 8px'>{v}</td>" for v in row.values) + "</tr>"
    rows += "</tbody></table>"
    return rows


# ── Step 0 — scaffold HTML ────────────────────────────────────────────────────

STEP0_HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>NFL QB Passing Yards — Research Session</title>
<style>
  body {{ font-family: system-ui, Arial, sans-serif; font-size: 14px; max-width: 1400px; margin: 0 auto; padding: 20px; background: #fafafa; }}
  h1 {{ background: #1a1a2e; color: white; padding: 16px 20px; border-radius: 6px; }}
  h2 {{ background: #2d2d44; color: white; padding: 10px 16px; border-radius: 4px; margin-top: 32px; }}
  h3 {{ color: #1a1a2e; border-bottom: 2px solid #1a1a2e; padding-bottom: 4px; }}
  section {{ background: white; border: 1px solid #ddd; border-radius: 6px; padding: 20px; margin: 16px 0; }}
  table {{ border-collapse: collapse; font-size: 13px; font-family: monospace; margin: 8px 0; }}
  th {{ background: #1a1a2e; color: white; padding: 6px 10px; }}
  td {{ padding: 4px 8px; border: 1px solid #ccc; }}
  tr:nth-child(even) {{ background: #f9f9f9; }}
  .pass {{ background: #fce8e6 !important; }}
  .flag {{ background: #fff3cd !important; }}
  .good {{ color: #2d7a2d; font-weight: bold; }}
  .warn {{ color: #c8760a; font-weight: bold; }}
  .bad  {{ color: #c0392b; font-weight: bold; }}
  pre {{ background: #f4f4f4; padding: 10px; border-radius: 4px; font-size: 12px; overflow-x: auto; }}
</style>
</head>
<body>

<h1>NFL QB Passing Yards — Research Session Log</h1>

<section>
<h2>Step 0 — Configuration</h2>
<p><em>{ts()}</em></p>

<h3>Market Config</h3>
<table>
<tr><th>Field</th><th>Value</th></tr>
<tr><td>Market</td><td>NFL QB Passing Yards</td></tr>
<tr><td>Odds API market key</td><td><code>player_pass_yds</code></td></tr>
<tr><td>Market type</td><td>Variable / numeric line (e.g. 225.5, 237.5, 249.5)</td></tr>
<tr><td>Bet direction hypothesis</td><td>UNDER likely has edge — books shade lines toward overs; public loves rooting for big passing games</td></tr>
<tr><td>Modeling book</td><td><strong>betonlineag</strong> only (all books pulled for EDA context in Step 1, then filtered)</td></tr>
<tr><td>Odds seasons</td><td>2023, 2024, 2025</td></tr>
<tr><td>Feature data seasons</td><td>1999–2025 (nfl_data_py weekly)</td></tr>
<tr><td>S3 odds path</td><td><code>s3://the-odds-api-mt/nfl/pass_yds_model/{{season}}/{{nfl_game_id}}.parquet</code></td></tr>
<tr><td>Feature data path</td><td><code>~/Downloads/tmp/pass_yds/weekly_player_data.parquet</code></td></tr>
<tr><td>Project dir</td><td><code>src/nfl_passing_yards_modeling/</code></td></tr>
</table>

<h3>Spot-Check Player</h3>
<table>
<tr><th>Field</th><th>Value</th></tr>
<tr><td>Player</td><td><strong>Josh Allen</strong></td></tr>
<tr><td>Team</td><td>Buffalo Bills</td></tr>
<tr><td>Why chosen</td><td>Active starter, full career data 2018–2025, high-volume passer, well-known — easy to sanity-check projections</td></tr>
</table>
</section>
"""

HTML_PATH.write_text(STEP0_HTML, encoding="utf-8")
print(f"[Step 0] HTML log created: {HTML_PATH}")


# ── Step 1 — Load market data ─────────────────────────────────────────────────

print("\n[Step 1] Loading market data from S3...")

frames = []
for season in SEASONS:
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=f"{S3_PREFIX}/{season}/"):
        keys += [o["Key"] for o in page.get("Contents", [])]
    print(f"  Season {season}: {len(keys)} game files")
    for key in keys:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
        df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
        df = df[df["market"] == MARKET].copy()
        if len(df):
            frames.append(df)

raw = pd.concat(frames, ignore_index=True)
print(f"  Total raw rows: {len(raw):,}")

# Normalize: pivot over/under into one row per (player, game, book, line)
raw["outcome_name"] = raw["outcome_name"].str.strip()
raw["outcome_desc"] = raw["outcome_desc"].str.strip()

over  = raw[raw["outcome_name"] == "Over" ].rename(columns={"price": "price_over"})
under = raw[raw["outcome_name"] == "Under"].rename(columns={"price": "price_under"})

join_cols = ["nfl_game_id", "season", "bookmaker", "outcome_desc", "point"]
lines = over.merge(
    under[join_cols + ["price_under"]],
    on=join_cols,
    how="inner"
).rename(columns={
    "outcome_desc": "player_name",
    "point":        "line",
    "price_over":   "american_over",
    "price_under":  "american_under",
})

# Parse game_date from nfl_game_id (format: 2023_03_NYG_SF → need schedule lookup)
# nfl_game_id encodes season_week_away_home — extract week and season
lines["nfl_season"]  = lines["nfl_game_id"].str.split("_").str[0].astype(int)
lines["nfl_week"]    = lines["nfl_game_id"].str.split("_").str[1].astype(int)
lines["home_team"]   = lines["nfl_game_id"].str.split("_").str[3]
lines["away_team"]   = lines["nfl_game_id"].str.split("_").str[2]

# Convert American odds to decimal
def american_to_decimal(x):
    x = pd.to_numeric(x, errors="coerce")
    return x.apply(lambda v: (v / 100 + 1) if v > 0 else (100 / abs(v) + 1) if v < 0 else None)

lines["decimal_over"]  = american_to_decimal(lines["american_over"])
lines["decimal_under"] = american_to_decimal(lines["american_under"])
lines["raw_prob_over"]  = 1 / lines["decimal_over"]
lines["raw_prob_under"] = 1 / lines["decimal_under"]
lines["vig"] = lines["raw_prob_over"] + lines["raw_prob_under"] - 1
lines["novig_prob_over"]  = lines["raw_prob_over"]  / (lines["raw_prob_over"] + lines["raw_prob_under"])
lines["novig_prob_under"] = lines["raw_prob_under"] / (lines["raw_prob_over"] + lines["raw_prob_under"])

lines_path = TMP_DIR / "step1_lines_all_books.parquet"
lines.to_parquet(lines_path, index=False)
print(f"  Pivoted rows (all books): {len(lines):,}  →  saved to {lines_path}")

# BetOnline subset
bol = lines[lines["bookmaker"] == FOCUS_BOOK].copy()
bol_path = TMP_DIR / "step1_lines_betonline.parquet"
bol.to_parquet(bol_path, index=False)
print(f"  BetOnline rows: {len(bol):,}")


# ── Step 1 — EDA ──────────────────────────────────────────────────────────────

print("\n[Step 1] Running EDA...")

# 1a. Coverage by season + book
coverage_by_book = (
    lines.groupby(["nfl_season", "bookmaker"])
    .agg(n_rows=("line", "count"), n_games=("nfl_game_id", "nunique"), n_players=("player_name", "nunique"))
    .reset_index()
    .sort_values(["nfl_season", "n_rows"], ascending=[True, False])
)
print("\nCoverage by season + book:")
print(coverage_by_book.to_string(index=False))

# 1b. BetOnline coverage by season + week
bol_weekly = (
    bol.groupby(["nfl_season", "nfl_week"])
    .agg(n_games=("nfl_game_id", "nunique"), n_players=("player_name", "nunique"), n_rows=("line", "count"))
    .reset_index()
    .sort_values(["nfl_season", "nfl_week"])
)
print("\nBetOnline coverage by season/week (sample):")
print(bol_weekly.head(20).to_string(index=False))

# 1c. Line distribution (BetOnline)
line_dist = (
    bol.groupby("line")
    .agg(n=("line", "count"), n_players=("player_name", "nunique"))
    .reset_index()
    .sort_values("line")
)
print("\nBetOnline line distribution:")
print(line_dist.to_string(index=False))

# 1d. Book coverage summary
book_summary = (
    lines.groupby("bookmaker")
    .agg(n_rows=("line", "count"), n_games=("nfl_game_id", "nunique"), n_players=("player_name", "nunique"))
    .reset_index()
    .sort_values("n_rows", ascending=False)
)
print("\nAll-book coverage summary:")
print(book_summary.to_string(index=False))

# 1e. BetOnline games per season with no pass_yds lines
total_games_per_season = {s: 285 for s in SEASONS}  # ~285 games per season event_id_map
bol_games = bol.groupby("nfl_season")["nfl_game_id"].nunique()
print("\nBetOnline: games with pass_yds lines vs total games:")
for s in SEASONS:
    covered = bol_games.get(s, 0)
    print(f"  {s}: {covered} / ~{total_games_per_season[s]} games ({covered/total_games_per_season[s]*100:.1f}%)")

# 1f. Players per game (BetOnline)
players_per_game = (
    bol.groupby(["nfl_season", "nfl_game_id"])["player_name"]
    .nunique()
    .reset_index()
    .rename(columns={"player_name": "n_players"})
)
ppg_dist = players_per_game["n_players"].value_counts().sort_index()
print("\nPlayers per game distribution (BetOnline):")
print(ppg_dist.to_string())

# 1g. Spot-check Josh Allen (BetOnline)
allen_bol = bol[bol["player_name"].str.contains("Josh Allen|J.Allen|J. Allen", na=False, case=False)]
print(f"\nJosh Allen BetOnline rows: {len(allen_bol)}")
if len(allen_bol):
    print(allen_bol[["nfl_season", "nfl_week", "nfl_game_id", "line", "american_over", "american_under", "novig_prob_over"]].head(10).to_string(index=False))


# ── Step 1 — Load feature data ────────────────────────────────────────────────

print("\n[Step 1] Loading feature data (nfl_data_py weekly)...")
features = pd.read_parquet(PLAYER_DATA)
qbs = features[(features["position"] == "QB") & (features["attempts"] > 0)].copy()

print(f"  QB game rows: {len(qbs):,}")
print(f"  Seasons: {qbs['season'].min()}–{qbs['season'].max()}")
print(f"  Unique QBs: {qbs['player_id'].nunique():,}")

# Passing yards distribution
pyd_stats = qbs["passing_yards"].describe().round(1)
print(f"\nPassing yards distribution:\n{pyd_stats.to_string()}")

# Allen feature spot-check
allen_feat = qbs[qbs["player_display_name"].str.contains("Josh Allen", na=False)]
print(f"\nJosh Allen feature rows: {len(allen_feat)}")
print(allen_feat[["season", "week", "attempts", "passing_yards", "rushing_yards"]].tail(8).to_string(index=False))


# ── Step 1 — DuckDB tests ─────────────────────────────────────────────────────

print("\n[Step 1] Running DuckDB tests...")

con = duckdb.connect()
con.execute(f"CREATE TABLE lines AS SELECT * FROM read_parquet('{lines_path}')")
con.execute(f"CREATE TABLE bol AS SELECT * FROM read_parquet('{bol_path}')")

tests = []

def run_test(name, sql, expect_zero=True):
    result = con.execute(sql).fetchone()[0]
    passed = (result == 0) if expect_zero else (result > 0)
    status = "PASS" if passed else "FAIL"
    tests.append({"Test": name, "Result": result, "Status": status})
    icon = "✓" if passed else "✗"
    print(f"  {icon} [{status}] {name}: {result}")
    return passed

# T1: market row count in expected range
run_test("All-books row count > 5,000", "SELECT CASE WHEN COUNT(*) > 5000 THEN 0 ELSE 1 END FROM lines", expect_zero=True)

# T2: BetOnline row count > 1,000
run_test("BetOnline row count > 1,000", "SELECT CASE WHEN COUNT(*) > 1000 THEN 0 ELSE 1 END FROM bol", expect_zero=True)

# T3: no nulls in key columns (BetOnline)
run_test("No nulls in player_name (bol)", "SELECT COUNT(*) FROM bol WHERE player_name IS NULL", expect_zero=True)
run_test("No nulls in line (bol)", "SELECT COUNT(*) FROM bol WHERE line IS NULL", expect_zero=True)
run_test("No nulls in american_over (bol)", "SELECT COUNT(*) FROM bol WHERE american_over IS NULL", expect_zero=True)
run_test("No nulls in american_under (bol)", "SELECT COUNT(*) FROM bol WHERE american_under IS NULL", expect_zero=True)

# T4: line values are numeric and in plausible range (100–500 yards)
run_test("All lines 100–500 (bol)", "SELECT COUNT(*) FROM bol WHERE line < 100 OR line > 500", expect_zero=True)

# T5: raw probs between 0 and 1
run_test("raw_prob_over in (0,1)", "SELECT COUNT(*) FROM bol WHERE raw_prob_over <= 0 OR raw_prob_over >= 1", expect_zero=True)
run_test("raw_prob_under in (0,1)", "SELECT COUNT(*) FROM bol WHERE raw_prob_under <= 0 OR raw_prob_under >= 1", expect_zero=True)

# T6: vig is positive (sum of raw probs > 1)
run_test("Avg vig > 0", "SELECT CASE WHEN AVG(vig) > 0 THEN 0 ELSE 1 END FROM bol", expect_zero=True)

# T7: seasons covered
run_test("All 3 seasons present (bol)", "SELECT CASE WHEN COUNT(DISTINCT nfl_season) = 3 THEN 0 ELSE 1 END FROM bol", expect_zero=True)

# T8: Josh Allen has rows in BetOnline
run_test("Josh Allen rows > 0 (bol)", "SELECT CASE WHEN COUNT(*) > 0 THEN 0 ELSE 1 END FROM bol WHERE player_name ILIKE '%allen%'", expect_zero=True)

test_results = pd.DataFrame(tests)
n_pass = (test_results["Status"] == "PASS").sum()
n_fail = (test_results["Status"] == "FAIL").sum()
print(f"\n  Tests: {n_pass} passed, {n_fail} failed")


# ── Step 1 — Write HTML section ───────────────────────────────────────────────

print("\n[Step 1] Writing HTML section...")

# Vig avg for display
avg_vig = bol["vig"].mean()
avg_line = bol["line"].mean()

step1_html = f"""
<section>
<h2>Step 1 — Data Pull + EDA</h2>
<p><em>{ts()}</em></p>

<h3>1a. All-Book Coverage by Season</h3>
{df_to_html(coverage_by_book, "Rows / games / players per season per book (sorted by row count)")}

<h3>1b. BetOnline Coverage by Season</h3>
"""

# Season summary for BetOnline
bol_season_summary = (
    bol.groupby("nfl_season")
    .agg(
        n_rows=("line", "count"),
        n_games=("nfl_game_id", "nunique"),
        n_players=("player_name", "nunique"),
        n_weeks=("nfl_week", "nunique"),
        avg_line=("line", "mean"),
        avg_vig_pct=("vig", lambda x: round(x.mean() * 100, 2)),
    )
    .reset_index()
    .round(2)
)
step1_html += df_to_html(bol_season_summary, "BetOnline summary by season")

step1_html += f"""
<h3>1c. BetOnline Line Distribution</h3>
<p>Average line across all BetOnline rows: <strong>{avg_line:.1f}</strong> | Average vig: <strong>{avg_vig*100:.2f}pp</strong></p>
{df_to_html(line_dist, "Distinct line values posted by BetOnline")}

<h3>1d. All-Book Coverage Summary</h3>
{df_to_html(book_summary, "All books — total rows, games, players across 2023–2025")}

<h3>1e. Players Per Game (BetOnline)</h3>
"""

ppg_df = ppg_dist.reset_index()
ppg_df.columns = ["n_players_per_game", "n_games"]
step1_html += df_to_html(ppg_df, "How many QBs get a pass_yds line per game (BetOnline)")

step1_html += f"""
<h3>1f. Feature Data Summary</h3>
<table>
<tr><th>Field</th><th>Value</th></tr>
<tr><td>Source</td><td>nfl_data_py weekly (nflverse)</td></tr>
<tr><td>Seasons</td><td>{qbs['season'].min()}–{qbs['season'].max()}</td></tr>
<tr><td>QB game rows (attempts &gt; 0)</td><td>{len(qbs):,}</td></tr>
<tr><td>Unique QB player_ids</td><td>{qbs['player_id'].nunique():,}</td></tr>
<tr><td>Passing yards mean</td><td>{qbs['passing_yards'].mean():.1f}</td></tr>
<tr><td>Passing yards std</td><td>{qbs['passing_yards'].std():.1f}</td></tr>
<tr><td>Passing yards min</td><td>{qbs['passing_yards'].min():.0f}</td></tr>
<tr><td>Passing yards max</td><td>{qbs['passing_yards'].max():.0f}</td></tr>
</table>

<h3>1g. Spot-Check — Josh Allen (BetOnline)</h3>
"""

if len(allen_bol):
    allen_display = allen_bol[["nfl_season", "nfl_week", "nfl_game_id", "line", "american_over", "american_under", "novig_prob_over", "novig_prob_under"]].head(15).round(3)
    step1_html += df_to_html(allen_display, f"First 15 Josh Allen BetOnline rows ({len(allen_bol)} total)")
else:
    step1_html += "<p class='bad'>Josh Allen NOT FOUND in BetOnline data — name matching issue to investigate</p>"

allen_feat_display = allen_feat[["season", "week", "attempts", "passing_yards", "rushing_yards", "passing_tds", "interceptions"]].tail(10)
step1_html += df_to_html(allen_feat_display, "Josh Allen last 10 feature rows (nfl_data_py)")

step1_html += f"""
<h3>1h. DuckDB Test Results</h3>
<table>
<tr><th>Test</th><th>Result</th><th>Status</th></tr>
"""
for _, row in test_results.iterrows():
    color = "#d4edda" if row["Status"] == "PASS" else "#f8d7da"
    step1_html += f"<tr style='background:{color}'><td>{row['Test']}</td><td>{row['Result']}</td><td><strong>{row['Status']}</strong></td></tr>"

step1_html += f"""
</table>
<p><strong>{n_pass} passed, {n_fail} failed</strong></p>

<h3>Flagged Items</h3>
<ul>
"""

# auto-flag items
flags = []

# Check weeks with no BetOnline coverage
early_weeks = bol_weekly[bol_weekly["nfl_week"] <= 2]["nfl_season"].unique()
if len(early_weeks) == 0:
    flags.append("No BetOnline lines for Weeks 1–2 in any season — BetOnline may not post pass_yds props early in the season")
elif len(bol_weekly[bol_weekly["nfl_week"] == 1]) == 0:
    flags.append("Week 1 has no BetOnline pass_yds lines — coverage starts Week 2 or later")

# Check if line range is suspiciously narrow
line_range = line_dist["line"].max() - line_dist["line"].min()
if line_range < 50:
    flags.append(f"Line range is narrow ({line_dist['line'].min()}–{line_dist['line'].max()}) — may be fewer line variants than other markets")

# Check players per game
if ppg_dist.get(1, 0) > ppg_dist.get(2, 0):
    flags.append("Most games have only 1 QB with a pass_yds line — books typically only price the starting QB, not both")

# Allen check
if len(allen_bol) == 0:
    flags.append("CRITICAL: Josh Allen not found in BetOnline data — name normalization needed before Step 2")

for f in flags:
    step1_html += f"<li class='warn'>{f}</li>"

if not flags:
    step1_html += "<li class='good'>No flags — data looks clean</li>"

step1_html += "</ul></section>"

with open(HTML_PATH, "a", encoding="utf-8") as f:
    f.write(step1_html)

print(f"[Step 1] HTML section appended: {HTML_PATH}")
print("\n=== DONE ===")
print(f"Open: open '{HTML_PATH}'")
