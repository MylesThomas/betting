"""
Manual validation script — reads from s3://levels-fyi-mt and checks recent scrapes.

Usage:
  uv run python src/levels_fyi_scraper/validate_scrape.py
  uv run python src/levels_fyi_scraper/validate_scrape.py --days 3
  uv run python src/levels_fyi_scraper/validate_scrape.py --date 2026-09-06
"""

import argparse
import io
from datetime import date, timedelta

import boto3
import pandas as pd

S3_BUCKET   = "levels-fyi-mt"
S3_KEY      = "overview/daily.parquet"
VAL_PREFIX  = "validation"
TC_MIN      = 50_000
TC_MAX      = 1_500_000

s3 = boto3.client("s3")


def load_overview() -> pd.DataFrame:
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_KEY)
        return pd.read_parquet(io.BytesIO(obj["Body"].read()))
    except Exception as e:
        print(f"Could not load s3://{S3_BUCKET}/{S3_KEY}: {e}")
        return pd.DataFrame()


def validate_day(day_df: pd.DataFrame, prev_df: pd.DataFrame) -> pd.DataFrame:
    prev = {} if prev_df.empty else {r["company_slug"]: r for r in prev_df.to_dict("records")}
    rows = []
    for _, r in day_df.iterrows():
        slug     = r["company_slug"]
        subs     = r.get("total_submissions")
        tc       = r.get("median_tc")
        prev_row = prev.get(slug)
        delta    = int(subs - prev_row["total_submissions"]) if (prev_row and subs is not None and prev_row.get("total_submissions") is not None) else None

        nonnull  = subs is not None and tc is not None and subs > 0
        monotone = delta is None or delta >= 0
        tc_range = tc is not None and TC_MIN <= tc <= TC_MAX
        passed   = nonnull and monotone and tc_range

        rows.append({
            "company_slug":      slug,
            "total_submissions": subs,
            "submissions_delta": delta,
            "median_tc":         tc,
            "nonnull":           "✅" if nonnull  else "❌",
            "monotone":          "✅" if monotone else "❌",
            "tc_range":          "✅" if tc_range else "❌",
            "status":            "✅ PASS" if passed else "❌ FAIL",
        })
    return pd.DataFrame(rows)


def run(check_date: str, days: int) -> None:
    df = load_overview()
    if df.empty:
        print("No data in S3.")
        return

    dates = sorted(df["scrape_date"].unique())
    print(f"Overview file: {len(df)} rows, {len(dates)} days ({dates[0]} → {dates[-1]})\n")

    target_dates = sorted([d for d in dates if d <= check_date], reverse=True)[:days]
    if not target_dates:
        print(f"No data on or before {check_date}")
        return

    for target in reversed(target_dates):
        day_df  = df[df["scrape_date"] == target].copy()
        idx     = dates.index(target)
        prev_df = df[df["scrape_date"] == dates[idx - 1]].copy() if idx > 0 else pd.DataFrame()

        val = validate_day(day_df, prev_df)
        n_pass = (val["status"] == "✅ PASS").sum()
        n_fail = len(val) - n_pass

        status = "✅ ALL PASS" if n_fail == 0 else f"❌ {n_fail} FAIL{'S' if n_fail > 1 else ''}"
        print(f"{'─'*70}")
        print(f"  {target}   {n_pass}/{len(val)} passed   {status}")
        print(f"{'─'*70}")

        display = val.copy()
        if display["median_tc"].notna().any():
            display["median_tc"] = display["median_tc"].apply(
                lambda v: f"${v/1000:.0f}K" if pd.notna(v) else "—"
            )
        display["submissions_delta"] = display["submissions_delta"].apply(
            lambda v: f"+{int(v)}" if (pd.notna(v) and v >= 0) else (str(int(v)) if pd.notna(v) else "—")
        )
        display["total_submissions"] = display["total_submissions"].apply(
            lambda v: str(int(v)) if pd.notna(v) else "—"
        )

        failures = display[display["status"] == "❌ FAIL"]
        if not failures.empty:
            print("\n  FAILURES:")
            print(failures.to_string(index=False))
            print()

        print(display.to_string(index=False))
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=date.today().isoformat(), help="Check on or before this date")
    parser.add_argument("--days", type=int, default=1, help="How many recent days to show")
    args = parser.parse_args()
    run(args.date, args.days)
