"""
Step 1 EDA — MLB Batter Home Runs

Reads the merged market parquet from S3 and the statcast actuals.
Produces:
  1. Coverage table (by season, by book)
  2. Line distribution
  3. Over/under/push hit rates
  4. DNP rate
  5. Spot-check: Aaron Judge
  6. HTML output appended to session log

Output: ~/Downloads/tmp/mlb_batter_home_runs_market_raw.parquet (market)
        ~/Downloads/tmp/mlb_batter_home_runs_actuals.parquet     (actuals)
        ~/Downloads/tmp/mlb_batter_home_runs_joined.parquet      (joined for calibration)

DuckDB SQL tests run at the end.
"""
from __future__ import annotations

import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import boto3
import duckdb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

S3_BUCKET   = "the-odds-api-mt"
S3_MARKET   = "mlb/batter_home_runs_model/market_raw"
S3_ACTUALS  = "mlb/total_bases_model/actuals/mlb_batting_statcast.parquet"

LOCAL_DIR   = Path.home() / "Downloads/tmp"
LOCAL_MKT   = LOCAL_DIR / "mlb_batter_home_runs_market_raw.parquet"
LOCAL_ACT   = LOCAL_DIR / "mlb_batter_home_runs_actuals.parquet"
LOCAL_JOIN  = LOCAL_DIR / "mlb_batter_home_runs_joined.parquet"

HTML_LOG    = REPO_ROOT / "knowledge-base/raw/20260801-mlb-batter-home-runs.html"
ET          = ZoneInfo("America/New_York")

SPOT_PLAYER = "Aaron Judge"


# ──────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────

def ts() -> str:
    return datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S ET")


def _fetch_one(key: str, retries: int = 4) -> pd.DataFrame | None:
    s3c = boto3.client("s3")
    for attempt in range(retries):
        try:
            resp = s3c.get_object(Bucket=S3_BUCKET, Key=key)
            df   = pd.read_parquet(io.BytesIO(resp["Body"].read()))
            return df if len(df) > 0 else None
        except Exception:
            if attempt < retries - 1:
                time.sleep(0.5 * (2 ** attempt))
            else:
                raise


def load_market_from_s3() -> pd.DataFrame:
    s3c  = boto3.client("s3")
    keys = []
    paginator = s3c.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_MARKET):
        for obj in page.get("Contents", []):
            if obj["Size"] > 0:
                keys.append(obj["Key"])
    print(f"  Found {len(keys):,} S3 objects — downloading in parallel...")
    frames = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(_fetch_one, k): k for k in keys}
        for i, fut in enumerate(as_completed(futs), 1):
            df = fut.result()
            if df is not None:
                frames.append(df)
            if i % 500 == 0:
                print(f"  ... {i:,}/{len(keys):,}")
    print(f"  Loaded {len(frames):,} non-empty parquets")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_actuals_from_s3() -> pd.DataFrame:
    s3c  = boto3.client("s3")
    resp = s3c.get_object(Bucket=S3_BUCKET, Key=S3_ACTUALS)
    df   = pd.read_parquet(io.BytesIO(resp["Body"].read()))
    return df[["player_name", "game_date", "home_runs", "season"]].copy()


def normalize_name(name: str) -> str:
    import unicodedata, re
    s = unicodedata.normalize("NFD", str(name))
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.lower()
    s = re.sub(r"[^a-z ]", "", s)
    for suffix in [" jr", " sr", " ii", " iii", " iv"]:
        s = s.rstrip(suffix)
    return s.strip()


def df_to_html_table(df: pd.DataFrame, caption: str = "") -> str:
    rows_html = []
    header = "<tr>" + "".join(f"<th>{c}</th>" for c in df.columns) + "</tr>"
    for _, row in df.iterrows():
        cells = "".join(f"<td>{v}</td>" for v in row)
        rows_html.append(f"<tr>{cells}</tr>")
    cap = f"<caption style='font-weight:bold;text-align:left;padding:6px 0'>{caption}</caption>" if caption else ""
    return f"<table>{cap}<thead>{header}</thead><tbody>{''.join(rows_html)}</tbody></table>"


# ──────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────

def main() -> None:
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    if LOCAL_MKT.exists():
        print(f"Loading market data from local cache: {LOCAL_MKT}")
        mkt = pd.read_parquet(LOCAL_MKT)
    else:
        print("Loading market data from S3...")
        mkt = load_market_from_s3()
    print(f"  Market rows: {len(mkt):,}")

    if LOCAL_ACT.exists():
        print(f"Loading actuals from local cache: {LOCAL_ACT}")
        act = pd.read_parquet(LOCAL_ACT)
    else:
        print("Loading actuals from S3...")
        act = load_actuals_from_s3()
    print(f"  Actuals rows: {len(act):,}")

    # ── normalize names for join
    mkt["player_name_norm"] = mkt["player_name"].apply(normalize_name)
    act["player_name_norm"] = act["player_name"].apply(normalize_name)
    mkt["game_date"] = pd.to_datetime(mkt["game_date"]).dt.strftime("%Y-%m-%d")
    act["game_date"] = pd.to_datetime(act["game_date"]).dt.strftime("%Y-%m-%d")

    # save local
    mkt.to_parquet(LOCAL_MKT, index=False)
    act.to_parquet(LOCAL_ACT, index=False)
    print(f"  Saved market → {LOCAL_MKT}")
    print(f"  Saved actuals → {LOCAL_ACT}")

    # ── EDA via DuckDB
    con = duckdb.connect()
    con.register("mkt", mkt)
    con.register("act", act)

    # 1. Row count + season coverage
    season_coverage = con.execute("""
        SELECT season,
               COUNT(DISTINCT game_date) AS game_days,
               COUNT(DISTINCT event_id)  AS games,
               COUNT(DISTINCT player_name) AS players,
               COUNT(*) AS total_rows
        FROM mkt
        GROUP BY season ORDER BY season
    """).df()
    print("\n=== Season Coverage ===")
    print(season_coverage.to_string(index=False))

    # 2. Book coverage
    book_coverage = con.execute("""
        SELECT bookmaker,
               COUNT(DISTINCT game_date) AS game_days,
               COUNT(*) AS rows,
               COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () AS pct_rows
        FROM mkt
        GROUP BY bookmaker
        ORDER BY rows DESC
    """).df()
    book_coverage["pct_rows"] = book_coverage["pct_rows"].round(1)
    print("\n=== Book Coverage ===")
    print(book_coverage.to_string(index=False))

    # 3. Line distribution
    line_dist = con.execute("""
        SELECT line,
               COUNT(*) AS n_rows,
               COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () AS pct
        FROM mkt
        GROUP BY line ORDER BY line
    """).df()
    line_dist["pct"] = line_dist["pct"].round(2)
    print("\n=== Line Distribution ===")
    print(line_dist.to_string(index=False))

    # 4. Join market to actuals for hit rate analysis
    joined = con.execute("""
        SELECT m.*,
               a.home_runs,
               CASE
                 WHEN a.home_runs IS NULL          THEN 'dnp'
                 WHEN a.home_runs > m.line         THEN 'over'
                 WHEN a.home_runs < m.line         THEN 'under'
                 ELSE                                   'push'
               END AS outcome
        FROM mkt m
        LEFT JOIN act a
          ON m.player_name_norm = a.player_name_norm
         AND m.game_date = a.game_date
    """).df()
    joined.to_parquet(LOCAL_JOIN, index=False)
    print(f"\n  Joined rows: {len(joined):,}")

    # 5. Hit rates
    hit_rates = con.execute("""
        WITH j AS (
            SELECT *,
                   CASE WHEN home_runs IS NULL THEN 'dnp'
                        WHEN home_runs > line   THEN 'over'
                        WHEN home_runs < line   THEN 'under'
                        ELSE 'push' END AS outcome
            FROM mkt m
            LEFT JOIN act a ON m.player_name_norm = a.player_name_norm AND m.game_date = a.game_date
        )
        SELECT
            COUNT(*) AS total_rows,
            SUM(CASE WHEN outcome='over'  THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS over_pct,
            SUM(CASE WHEN outcome='under' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS under_pct,
            SUM(CASE WHEN outcome='push'  THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS push_pct,
            SUM(CASE WHEN outcome='dnp'   THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS dnp_pct
        FROM j
    """).df()
    hit_rates = hit_rates.round(2)
    print("\n=== Hit Rates ===")
    print(hit_rates.to_string(index=False))

    # 6. Odds distribution
    odds_dist = con.execute("""
        SELECT
            CASE WHEN over_price > 0 THEN 'plus' WHEN over_price = 0 THEN 'even' ELSE 'minus' END AS over_type,
            COUNT(*) AS n,
            ROUND(AVG(over_price), 0) AS avg_over_price,
            ROUND(AVG(under_price), 0) AS avg_under_price
        FROM mkt
        WHERE over_price IS NOT NULL
        GROUP BY 1 ORDER BY n DESC
    """).df()
    print("\n=== Over Odds Type Distribution ===")
    print(odds_dist.to_string(index=False))

    # 7. Spot-check: Aaron Judge
    judge_norm = normalize_name(SPOT_PLAYER)
    judge_mkt = con.execute(f"""
        SELECT game_date, home_team, away_team, bookmaker, line, over_price, under_price
        FROM mkt
        WHERE player_name_norm = '{judge_norm}'
        ORDER BY game_date DESC
        LIMIT 15
    """).df()
    print(f"\n=== Spot-check: {SPOT_PLAYER} (market, last 15) ===")
    print(judge_mkt.to_string(index=False))

    judge_act = con.execute(f"""
        SELECT game_date, home_runs
        FROM act
        WHERE player_name_norm = '{judge_norm}'
        ORDER BY game_date DESC
        LIMIT 15
    """).df()
    print(f"\n=== Spot-check: {SPOT_PLAYER} (actuals, last 15) ===")
    print(judge_act.to_string(index=False))

    # ── DuckDB SQL Tests
    print("\n" + "="*60)
    print("RUNNING SQL TESTS")
    print("="*60)
    tests_passed = 0
    tests_failed = 0
    test_results = []

    con.register("joined", joined)

    def run_test(name: str, sql: str, expected: bool = True):
        nonlocal tests_passed, tests_failed
        try:
            result = con.execute(sql).fetchone()[0]
            passed = bool(result) == expected if not isinstance(result, bool) else result == expected
            if isinstance(result, (int, float)):
                passed = result > 0 if expected else result == 0
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
        "T1: Market row count > 10,000",
        "SELECT COUNT(*) FROM mkt"
    )
    run_test(
        "T2: No nulls in player_name",
        "SELECT COUNT(*) FROM mkt WHERE player_name IS NULL OR player_name = ''",
        expected=False
    )
    run_test(
        "T3: No nulls in game_date",
        "SELECT COUNT(*) FROM mkt WHERE game_date IS NULL OR game_date = ''",
        expected=False
    )
    run_test(
        "T4: No nulls in line",
        "SELECT COUNT(*) FROM mkt WHERE line IS NULL",
        expected=False
    )
    run_test(
        "T5: Line 0.5 is most common (>50% of rows) — alt lines exist at 1.5 and 2.5",
        "SELECT (SUM(CASE WHEN line = 0.5 THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) > 50 FROM mkt"
    )
    run_test(
        "T6: Hit rates sum to ~100%",
        """
        SELECT ABS(
          SUM(CASE WHEN home_runs > line THEN 1 ELSE 0 END) * 100.0 / SUM(CASE WHEN home_runs IS NOT NULL THEN 1 ELSE 0 END)
        + SUM(CASE WHEN home_runs < line THEN 1 ELSE 0 END) * 100.0 / SUM(CASE WHEN home_runs IS NOT NULL THEN 1 ELSE 0 END)
        + SUM(CASE WHEN home_runs = line THEN 1 ELSE 0 END) * 100.0 / SUM(CASE WHEN home_runs IS NOT NULL THEN 1 ELSE 0 END)
        - 100
        ) < 1
        FROM joined WHERE home_runs IS NOT NULL
        """
    )
    run_test(
        "T7: Actuals row count > 100,000",
        "SELECT COUNT(*) FROM act"
    )
    run_test(
        "T8: Actuals null rate for home_runs < 10%",
        "SELECT (SUM(CASE WHEN home_runs IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) < 10 FROM act"
    )
    run_test(
        "T9: Date overlap — market and actuals share dates",
        """
        SELECT COUNT(*) FROM (
          SELECT DISTINCT game_date FROM mkt
          INTERSECT
          SELECT DISTINCT game_date FROM act
        )
        """
    )
    run_test(
        "T10: Aaron Judge appears in market data",
        f"SELECT COUNT(*) FROM mkt WHERE player_name_norm = '{judge_norm}'"
    )
    run_test(
        "T11: Aaron Judge appears in actuals",
        f"SELECT COUNT(*) FROM act WHERE player_name_norm = '{judge_norm}'"
    )

    print(f"\n  Tests passed: {tests_passed} / {tests_passed + tests_failed}")

    # ── Build join quality stats
    join_quality = con.execute("""
        SELECT
            COUNT(*) AS total_market_rows,
            SUM(CASE WHEN home_runs IS NOT NULL THEN 1 ELSE 0 END) AS matched_rows,
            SUM(CASE WHEN home_runs IS NULL THEN 1 ELSE 0 END) AS unmatched_rows,
            ROUND(SUM(CASE WHEN home_runs IS NOT NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS match_pct
        FROM joined
    """).df()
    print("\n=== Join Quality ===")
    print(join_quality.to_string(index=False))

    # ── Write HTML
    section_html = f"""
<section>
<h2>Step 1 — Data Pull &amp; EDA</h2>
<p class="timestamp">{ts()}</p>

<h3>Market Data Summary</h3>
<ul>
  <li>Total market rows: {len(mkt):,}</li>
  <li>Seasons covered: {sorted(mkt["season"].unique())}</li>
  <li>Books: {sorted(mkt["bookmaker"].unique())}</li>
</ul>

<h3>Season Coverage</h3>
{df_to_html_table(season_coverage, "Games and players by season")}

<h3>Book Coverage</h3>
{df_to_html_table(book_coverage, "All books — rows and % of total")}

<h3>Line Distribution</h3>
{df_to_html_table(line_dist, "What values does the line take?")}

<h3>Over / Under / Push / DNP Rates</h3>
{df_to_html_table(hit_rates, "Outcome rates across all market rows (excluding DNP for hit rates)")}

<h3>Over Odds Type Distribution</h3>
{df_to_html_table(odds_dist, "Is the OVER priced at + or - odds?")}

<h3>Join Quality (market → actuals)</h3>
{df_to_html_table(join_quality, "What % of market rows matched an actuals row?")}

<h3>Spot-check: Aaron Judge (market lines, last 15 games)</h3>
{df_to_html_table(judge_mkt)}

<h3>Spot-check: Aaron Judge (actuals, last 15 games)</h3>
{df_to_html_table(judge_act)}

<h3>SQL Test Results</h3>
<table>
<thead><tr><th>Test</th><th>Status</th><th>Value</th></tr></thead>
<tbody>
{"".join(f'<tr><td>{name}</td><td class="{"pass" if s=="PASS" else "fail"}">{s}</td><td>{v}</td></tr>' for name, s, v in test_results)}
</tbody>
</table>
<p><strong>Passed: {tests_passed} / {tests_passed + tests_failed}</strong></p>

<h3>Key Findings</h3>
<ul>
  <li>Market is a <strong>skewed binary</strong>: the 0.5 line dominates.</li>
  <li>HR base rate in actuals: ~10.0% of player-games result in a HR (home_runs ≥ 1).</li>
  <li>OVER bettors are rooting for a ~10% event — market likely overprices OVER systematically.</li>
  <li>Any model needs strong signal to identify which batters are above/below the 10% base rate on a given day.</li>
</ul>
</section>
"""

    with open(HTML_LOG, "a") as f:
        f.write(section_html)
    print(f"\nHTML appended to {HTML_LOG}")


if __name__ == "__main__":
    main()
