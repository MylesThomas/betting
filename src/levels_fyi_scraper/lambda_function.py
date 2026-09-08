"""
Lambda handler for daily Levels.fyi overview scrape.

Uses curl_cffi directly (Chrome TLS impersonation, no Scrapling) for minimal deps.
After scraping, runs 3 validation checks and sends one SES email (TL;DR + full table).

Env vars:
  S3_BUCKET    required — "levels-fyi-mt"
  S3_KEY       optional — default "overview/daily.parquet"
  COMPANIES    optional — comma-separated slugs (default: all 29)
  SSL_VERIFY   optional — "false" to disable cert check (local dev only)
  SES_FROM     optional — default tqstrats@gmail.com
  SES_TO       optional — default mylescgthomas@gmail.com

Trigger: EventBridge cron(0 12 * * ? *) — 7am EST / 8am EDT
S3 layout:
  overview/daily.parquet          cumulative daily snapshots, all companies
  validation/{date}_log.parquet   per-company pass/fail, one file per day
"""

import io
import json
import os
import re
import time
from datetime import date

import boto3
import pandas as pd
from curl_cffi import requests as curl_requests

S3_BUCKET  = os.environ["S3_BUCKET"]
S3_KEY     = os.environ.get("S3_KEY", "overview/daily.parquet")
VAL_PREFIX = "validation"
SSL_VERIFY = os.environ.get("SSL_VERIFY", "true").lower() != "false"
SES_FROM   = os.environ.get("SES_FROM", "tqstrats@gmail.com")
SES_TO     = os.environ.get("SES_TO", "mylescgthomas@gmail.com")
COMPANIES  = [c.strip() for c in os.environ.get("COMPANIES", ",".join([
    "snap", "meta", "google", "microsoft", "nvidia", "apple", "amazon",
    "netflix", "uber", "lyft", "airbnb", "stripe", "palantir", "coinbase",
    "salesforce", "adobe", "oracle", "intel", "amd", "qualcomm",
    "linkedin", "pinterest", "reddit", "bytedance",
    "openai", "anthropic", "databricks", "snowflake", "cloudflare",
])).split(",")]

TC_MIN     = 50_000
TC_MAX     = 1_500_000
LEVELS_BASE = "https://www.levels.fyi"

s3  = boto3.client("s3")
ses = boto3.client("ses", region_name="us-east-2")


# ── scrape ────────────────────────────────────────────────────────────────────

def fetch_overview(slug: str) -> dict | None:
    url = f"{LEVELS_BASE}/companies/{slug}/salaries/"
    try:
        r = curl_requests.get(url, impersonate="chrome", verify=SSL_VERIFY, timeout=20)
        if r.status_code != 200:
            print(f"  {slug:<20} HTTP {r.status_code}")
            return None
        html = r.text
    except Exception as e:
        print(f"  {slug:<20} error: {e}")
        return None

    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL)
    if not m:
        print(f"  {slug:<20} no __NEXT_DATA__")
        return None

    try:
        props = json.loads(m.group(1))["props"]["pageProps"]
    except (json.JSONDecodeError, KeyError):
        print(f"  {slug:<20} JSON parse error")
        return None

    overview = props.get("overview", [])
    total_submissions = sum(
        sum(t.get("count", 0) for t in jf.get("titles", []))
        for jf in overview
    )
    highest = props.get("highestPayingJobFamilyAndLevel") or {}
    lowest  = props.get("lowestPayingJobFamilyAndLevel") or {}

    return {
        "scrape_date":       date.today().isoformat(),
        "company_slug":      slug,
        "total_submissions": total_submissions,
        "median_tc":         props.get("medianAcrossAllJobFamilies"),
        "num_job_families":  len(overview),
        "highest_role":      highest.get("jobFamily"),
        "highest_tc":        highest.get("totalCompensation"),
        "lowest_role":       lowest.get("jobFamily"),
        "lowest_tc":         lowest.get("totalCompensation"),
    }


# ── S3 I/O ────────────────────────────────────────────────────────────────────

def load_parquet(key: str) -> pd.DataFrame:
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
        return pd.read_parquet(io.BytesIO(obj["Body"].read()))
    except s3.exceptions.NoSuchKey:
        return pd.DataFrame()
    except Exception as e:
        print(f"Warning: could not load s3://{S3_BUCKET}/{key}: {e}")
        return pd.DataFrame()


def write_parquet(df: pd.DataFrame, key: str) -> None:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue())
    print(f"Written {len(df)} rows → s3://{S3_BUCKET}/{key}")


# ── validation ────────────────────────────────────────────────────────────────

def validate(today_df: pd.DataFrame, prev_df: pd.DataFrame) -> pd.DataFrame:
    prev = (
        {}
        if prev_df.empty
        else {r["company_slug"]: r for r in prev_df.to_dict("records")}
    )
    rows = []
    for slug in COMPANIES:
        match = today_df[today_df["company_slug"] == slug]
        if match.empty:
            rows.append({
                "date": date.today().isoformat(),
                "company_slug": slug,
                "total_submissions": None,
                "median_tc": None,
                "submissions_delta": None,
                "check_present": False,
                "check_nonnull": False,
                "check_monotone": False,
                "check_tc_range": False,
                "passed": False,
            })
            continue

        r        = match.iloc[0]
        subs     = r.get("total_submissions")
        tc       = r.get("median_tc")
        prev_row = prev.get(slug)
        prev_subs = prev_row["total_submissions"] if prev_row else None
        delta    = int(subs - prev_subs) if (subs is not None and prev_subs is not None) else None

        nonnull  = subs is not None and tc is not None and subs > 0
        monotone = delta is None or delta >= 0
        tc_range = tc is not None and TC_MIN <= tc <= TC_MAX

        rows.append({
            "date": date.today().isoformat(),
            "company_slug": slug,
            "total_submissions": subs,
            "median_tc": tc,
            "submissions_delta": delta,
            "check_present": True,
            "check_nonnull": nonnull,
            "check_monotone": monotone,
            "check_tc_range": tc_range,
            "passed": nonnull and monotone and tc_range,
        })
    return pd.DataFrame(rows)


# ── email ─────────────────────────────────────────────────────────────────────

def _icon(v: bool) -> str:
    return "✅" if v else "❌"


def send_email(val_df: pd.DataFrame, new_rows: int, total_rows: int) -> None:
    today   = date.today().isoformat()
    n_pass  = int(val_df["passed"].sum())
    n_fail  = len(val_df) - n_pass
    failures = val_df[~val_df["passed"]]

    subject = (
        f"✅ Levels.fyi daily scrape — {n_pass}/{len(val_df)} passed"
        if n_fail == 0
        else f"❌ Levels.fyi daily scrape — {n_fail} failure{'s' if n_fail > 1 else ''}"
    )

    if failures.empty:
        fail_html = "<span style='color:green'>None</span>"
    else:
        items = []
        for _, row in failures.iterrows():
            failed_checks = [
                c.replace("check_", "")
                for c in ["check_present", "check_nonnull", "check_monotone", "check_tc_range"]
                if not row[c]
            ]
            items.append(f"<li><b>{row['company_slug']}</b>: {', '.join(failed_checks)}</li>")
        fail_html = f"<ul style='color:red;margin:4px 0'>{''.join(items)}</ul>"

    tldr = f"""
<h2 style="margin:0 0 8px">TL;DR</h2>
<table style="border-collapse:collapse;font-family:monospace;font-size:13px;margin-bottom:8px">
  <tr><td style="padding:2px 16px 2px 0"><b>Date</b></td><td>{today}</td></tr>
  <tr><td style="padding:2px 16px 2px 0"><b>Companies scraped</b></td><td>{n_pass + n_fail} / {len(val_df)}</td></tr>
  <tr><td style="padding:2px 16px 2px 0"><b>Validation</b></td><td>{n_pass} passed &nbsp;·&nbsp; {n_fail} failed</td></tr>
  <tr><td style="padding:2px 16px 2px 0"><b>New rows added</b></td><td>{new_rows}</td></tr>
  <tr><td style="padding:2px 16px 2px 0"><b>Total rows in file</b></td><td>{total_rows}</td></tr>
  <tr><td style="padding:2px 16px 2px 0;vertical-align:top"><b>Failures</b></td><td>{fail_html}</td></tr>
</table>
<hr style="margin:16px 0;border:none;border-top:1px solid #ddd">
"""

    th = "style='padding:5px 12px;border:1px solid #ccc;background:#f5f5f5;text-align:left'"
    td = "style='padding:4px 12px;border:1px solid #ddd'"
    header = "".join(f"<th {th}>{h}</th>" for h in [
        "Company", "Submissions", "Δ Subs", "Median TC",
        "Present", "Non-null", "Monotone", "TC Range", "Status",
    ])

    def fmt_delta(v):
        if v is None:
            return "—"
        return f"+{v}" if v >= 0 else str(v)

    rows_html = ""
    for _, row in val_df.sort_values("company_slug").iterrows():
        bg   = "" if row["passed"] else " style='background:#fff0f0'"
        subs = str(int(row["total_submissions"])) if row["total_submissions"] is not None else "—"
        tc   = f"${row['median_tc']/1000:.0f}K" if row["median_tc"] else "—"
        rows_html += (
            f"<tr{bg}>"
            + "".join(f"<td {td}>{v}</td>" for v in [
                row["company_slug"], subs, fmt_delta(row["submissions_delta"]), tc,
                _icon(row["check_present"]),
                _icon(row["check_nonnull"]),
                _icon(row["check_monotone"]),
                _icon(row["check_tc_range"]),
                "✅ Pass" if row["passed"] else "❌ FAIL",
            ])
            + "</tr>"
        )

    full_table = f"""
<h2 style="margin:0 0 8px">Full Results — {today}</h2>
<table style="border-collapse:collapse;font-family:monospace;font-size:12px">
  <thead><tr>{header}</tr></thead>
  <tbody>{rows_html}</tbody>
</table>
"""

    body_html = (
        "<html><body style='font-family:sans-serif;padding:16px'>"
        + tldr + full_table
        + "</body></html>"
    )

    ses.send_email(
        Source=SES_FROM,
        Destination={"ToAddresses": [SES_TO]},
        Message={
            "Subject": {"Data": subject, "Charset": "UTF-8"},
            "Body":    {"Html": {"Data": body_html, "Charset": "UTF-8"}},
        },
    )
    print(f"Email sent: {subject}")


# ── handler ───────────────────────────────────────────────────────────────────

def handler(event, context):
    today = date.today().isoformat()
    print(f"Starting — {today} — {len(COMPANIES)} companies")

    existing = load_parquet(S3_KEY)

    new_rows = []
    for slug in COMPANIES:
        row = fetch_overview(slug)
        if row:
            new_rows.append(row)
            tc_str = f"  median ${row['median_tc']:,.0f}" if row.get("median_tc") else ""
            print(f"  {slug:<20} {row['total_submissions']:>5} subs{tc_str}")
        time.sleep(0.5)

    if not new_rows:
        all_fail = pd.DataFrame([{
            "date": today, "company_slug": s,
            "total_submissions": None, "median_tc": None, "submissions_delta": None,
            "check_present": False, "check_nonnull": False,
            "check_monotone": False, "check_tc_range": False, "passed": False,
        } for s in COMPANIES])
        send_email(all_fail, 0, len(existing) if not existing.empty else 0)
        return {"statusCode": 500, "body": "No rows scraped"}

    today_df = pd.DataFrame(new_rows)

    if not existing.empty:
        combined = pd.concat([existing, today_df], ignore_index=True).drop_duplicates(
            subset=["scrape_date", "company_slug"], keep="last"
        )
        net_new = len(combined) - len(existing)
    else:
        combined = today_df
        net_new  = len(combined)

    combined = combined.sort_values(["scrape_date", "company_slug"]).reset_index(drop=True)
    write_parquet(combined, S3_KEY)

    yesterday = (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    prev_df   = existing[existing["scrape_date"] == yesterday] if not existing.empty else pd.DataFrame()

    val_df  = validate(today_df, prev_df)
    val_key = f"{VAL_PREFIX}/{today}_log.parquet"
    write_parquet(val_df, val_key)

    n_fail = int((~val_df["passed"]).sum())
    send_email(val_df, net_new, len(combined))

    return {
        "statusCode": 200,
        "body": json.dumps({
            "date":                today,
            "scraped":             len(new_rows),
            "net_new_rows":        net_new,
            "total_rows":          len(combined),
            "validation_failures": n_fail,
        }),
    }
