"""
Analyze settled rebounds results from S3.

Reads:
  - Latest all_time.csv rollup for Section 1 (overall summary)
  - All rebounds_scored_settled_*.parquet files for Sections 2 & 3

Usage:
  python scripts/analyze_settled_results.py
  python scripts/analyze_settled_results.py --season 2025-26
  python scripts/analyze_settled_results.py --since 2026-01-01
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from io import BytesIO, StringIO
from pathlib import Path

import numpy as np
import pandas as pd


BUCKET = "nba-betting-mt"
RUNS_PREFIX = "rebounds/daily_runs"

PROB_BINS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 101]
PROB_LABELS = ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]

EDGE_BINS = [0, 5, 10, 15, 20, 25, 101]
EDGE_LABELS = ["0-5%", "5-10%", "10-15%", "15-20%", "20-25%", "25%+"]


def ensure_repo_root_on_syspath() -> Path:
    current = Path(__file__).resolve().parent
    while True:
        if (current / ".gitignore").exists() and (current / "src").exists():
            if str(current) not in sys.path:
                sys.path.insert(0, str(current))
            return current
        if current.parent == current:
            raise FileNotFoundError("Could not locate repo root")
        current = current.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze settled rebounds results.")
    p.add_argument("--season", type=str, default="", help="Filter to a specific season, e.g. 2025-26.")
    p.add_argument("--since", type=str, default="", help="Only include dates >= YYYY-MM-DD.")
    p.add_argument("--bucket", type=str, default=BUCKET)
    p.add_argument("--runs-prefix", type=str, default=RUNS_PREFIX)
    return p.parse_args()


def _s3():
    import boto3
    return boto3.client("s3")


def find_latest_rollup_key(bucket: str, rollups_prefix: str) -> str | None:
    s3 = _s3()
    paginator = s3.get_paginator("list_objects_v2")
    candidates: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=rollups_prefix.rstrip("/") + "/"):
        for item in page.get("Contents", []):
            key = item["Key"]
            if key.endswith("/all_time.csv"):
                candidates.append(key)
    return sorted(candidates)[-1] if candidates else None


def read_csv_s3(bucket: str, key: str) -> pd.DataFrame:
    body = _s3().get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
    return pd.read_csv(StringIO(body))


def read_parquet_s3(bucket: str, key: str) -> pd.DataFrame:
    body = _s3().get_object(Bucket=bucket, Key=key)["Body"].read()
    return pd.read_parquet(BytesIO(body))


def list_settled_keys(bucket: str, runs_prefix: str) -> list[str]:
    """Return latest-run-per-date settled parquet keys (excludes _rollups/)."""
    s3 = _s3()
    paginator = s3.get_paginator("list_objects_v2")
    all_keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=runs_prefix.rstrip("/") + "/"):
        for item in page.get("Contents", []):
            key = item["Key"]
            if (
                "rebounds_scored_settled_" in key
                and key.endswith(".parquet")
                and "_rollups/" not in key
            ):
                all_keys.append(key)

    latest: dict[str, tuple[str, str]] = {}
    for key in all_keys:
        parts = key.split("/")
        if len(parts) < 4:
            continue
        date_part = parts[-3]
        run_id = parts[-2]
        if date_part not in latest or run_id > latest[date_part][0]:
            latest[date_part] = (run_id, key)
    return sorted(v[1] for v in latest.values())


def load_settled_data(bucket: str, runs_prefix: str, season: str, since: str) -> pd.DataFrame:
    keys = list_settled_keys(bucket, runs_prefix)
    frames: list[pd.DataFrame] = []
    print(f"  Loading {len(keys)} settled parquet files...", flush=True)
    for key in keys:
        frames.append(read_parquet_s3(bucket, key))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if season:
        df = df[df["season"].astype(str) == season]
    if since:
        df = df[pd.to_datetime(df["date"]) >= pd.Timestamp(since)]
    return df


def pnl_fmt(v: float) -> str:
    return f"+{v:.3f}u" if v >= 0 else f"{v:.3f}u"


def roi_fmt(v: float) -> str:
    pct = v * 100
    return f"+{pct:.1f}%" if pct >= 0 else f"{pct:.1f}%"


def hr_fmt(v: float) -> str:
    return f"{v * 100:.1f}%"


def compute_bin_stats(
    df: pd.DataFrame,
    bin_col: str,
    bins: list,
    labels: list,
    extra_avg_cols: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    settled = df[df["result"].isin(["win", "loss", "push"])].copy()
    settled["_bin"] = pd.cut(
        settled[bin_col] * 100,
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )
    rows_list = []
    for label in labels:
        grp = settled[settled["_bin"] == label]
        n = len(grp)
        base: dict = {"bin": label, "rows": n, "wlp": "0-0-0", "hr": np.nan, "pnl": 0.0, "roi": np.nan}
        if extra_avg_cols:
            for col, key in extra_avg_cols:
                base[key] = grp[col].mean() if n > 0 else np.nan
        if n == 0:
            rows_list.append(base)
            continue
        n_win = int((grp["result"] == "win").sum())
        n_loss = int((grp["result"] == "loss").sum())
        n_push = int((grp["result"] == "push").sum())
        pnl = float(grp["pnl_units"].sum())
        settled_count = n_win + n_loss
        base.update({
            "wlp": f"{n_win}-{n_loss}-{n_push}",
            "hr": n_win / settled_count if settled_count > 0 else np.nan,
            "pnl": pnl,
            "roi": pnl / n,
        })
        rows_list.append(base)
    return pd.DataFrame(rows_list)


def print_bin_table(
    title: str,
    stats: pd.DataFrame,
    bin_header: str,
    extra_col_headers: list[str] | None = None,
) -> None:
    cw = [10, 6, 13, 10, 11, 9]
    extra_keys = [h for h in (extra_col_headers or [])]
    extra_cw = [9] * len(extra_keys)
    all_cw = cw + extra_cw
    headers = [bin_header, "rows", "W-L-P", "hit_rate", "PnL", "ROI"] + extra_keys
    print(f"  [{title}]")
    print("  " + "  ".join(h.ljust(w) for h, w in zip(headers, all_cw)))
    print("  " + "-" * (sum(all_cw) + 2 * (len(all_cw) - 1)))
    for _, row in stats.iterrows():
        cells = [
            str(row["bin"]).ljust(cw[0]),
            str(row["rows"]).ljust(cw[1]),
            str(row["wlp"]).ljust(cw[2]),
            (hr_fmt(row["hr"]) if not pd.isna(row["hr"]) else "-").ljust(cw[3]),
            pnl_fmt(row["pnl"]).ljust(cw[4]),
            (roi_fmt(row["roi"]) if not pd.isna(row["roi"]) else "-").ljust(cw[5]),
        ]
        for key, w in zip(extra_keys, extra_cw):
            val = row[key]
            cells.append((f"{val * 100:.1f}%" if not pd.isna(val) else "-").ljust(w))
        print("  " + "  ".join(cells))
    print()


def aggregate_rollup(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-run rollup rows into one row per strategy_bucket."""
    df = df.copy()
    df["_prob_sum"] = df["avg_implied_prob_taken"] * df["n_rows"]
    agg = (
        df.groupby("strategy_bucket", as_index=False)
        .agg(
            n_rows=("n_rows", "sum"),
            n_bets=("n_bets", "sum"),
            n_win=("n_win", "sum"),
            n_loss=("n_loss", "sum"),
            n_push=("n_push", "sum"),
            n_unsettled=("n_unsettled", "sum"),
            pnl_units=("pnl_units", "sum"),
            reference_pnl_units=("reference_pnl_units", "sum"),
            _prob_sum=("_prob_sum", "sum"),
        )
    )
    agg["avg_implied_prob_taken"] = agg["_prob_sum"] / agg["n_rows"]
    settled = agg["n_win"] + agg["n_loss"] + agg["n_push"]
    agg["hit_rate"] = np.where(
        (agg["n_win"] + agg["n_loss"]) > 0,
        agg["n_win"] / (agg["n_win"] + agg["n_loss"]),
        np.nan,
    )
    agg["roi_units_per_settled_bet"] = np.where(
        settled > 0,
        agg["reference_pnl_units"] / settled,
        np.nan,
    )
    return agg.drop(columns=["_prob_sum"])


def section1_summary(bucket: str, runs_prefix: str) -> None:
    rollups_prefix = f"{runs_prefix}/_rollups"
    key = find_latest_rollup_key(bucket, rollups_prefix)
    if key is None:
        print("  [ERROR] No all_time.csv rollup found.")
        return

    raw = read_csv_s3(bucket, key)
    df = aggregate_rollup(raw)

    order = {"both": 0, "ols": 1, "xgb": 2, "neither": 3}
    df["_ord"] = df["strategy_bucket"].map(order).fillna(99)
    df = df.sort_values("_ord").reset_index(drop=True)

    for _, row in df.iterrows():
        label = str(row["strategy_bucket"]).upper()
        is_neither = row["strategy_bucket"] == "neither"
        pnl_val = float(row["reference_pnl_units"] if is_neither else row["pnl_units"])
        hr = float(row["hit_rate"]) if not pd.isna(row["hit_rate"]) else np.nan
        roi = float(row["roi_units_per_settled_bet"]) if not pd.isna(row["roi_units_per_settled_bet"]) else np.nan
        pnl_str = pnl_fmt(pnl_val) + (" (reference @ line odds)" if is_neither else "")
        print(f"  [{label}]")
        print(f"    Rows: {int(row['n_rows'])} | Bets: {int(row['n_bets'])} | W-L-P: {int(row['n_win'])}-{int(row['n_loss'])}-{int(row['n_push'])} | Unsettled: {int(row['n_unsettled'])}")
        print(f"    PnL: {pnl_str} | Hit Rate: {hr_fmt(hr) if not pd.isna(hr) else '-'} | ROI: {roi_fmt(roi) if not pd.isna(roi) else '-'}")
        print()

    rollup_date = key.split("/")[-3] if len(key.split("/")) >= 3 else "unknown"
    print(f"  Source: s3://{bucket}/{key}")
    print(f"  Rollup date: {rollup_date}")


def section2_prob_bins(df: pd.DataFrame) -> None:
    xgb_df = df[df["strategy_bucket"] == "xgb"].copy()
    ols_df = df[df["strategy_bucket"] == "ols"].copy()
    both_df = df[df["strategy_bucket"] == "both"].copy()
    if "p_under_ols" in both_df.columns and "p_under_xgb" in both_df.columns:
        both_df["_p_mean"] = (both_df["p_under_ols"] + both_df["p_under_xgb"]) / 2

    if len(xgb_df) > 0 and "p_under_xgb" in xgb_df.columns:
        print_bin_table("XGB ONLY", compute_bin_stats(xgb_df, "p_under_xgb", PROB_BINS, PROB_LABELS), "p_under")
    else:
        print("  [XGB ONLY] No data.\n")

    if len(ols_df) > 0 and "p_under_ols" in ols_df.columns:
        print_bin_table("OLS ONLY", compute_bin_stats(ols_df, "p_under_ols", PROB_BINS, PROB_LABELS), "p_under")
    else:
        print("  [OLS ONLY] No data.\n")

    if len(both_df) > 0 and "_p_mean" in both_df.columns:
        stats = compute_bin_stats(
            both_df, "_p_mean", PROB_BINS, PROB_LABELS,
            extra_avg_cols=[("p_under_ols", "avg_p_ols"), ("p_under_xgb", "avg_p_xgb")],
        )
        print_bin_table("BOTH", stats, "p_under", extra_col_headers=["avg_p_ols", "avg_p_xgb"])
    else:
        print("  [BOTH] No data.\n")


def section3_edge_bins(df: pd.DataFrame) -> None:
    xgb_df = df[df["strategy_bucket"] == "xgb"].copy()
    ols_df = df[df["strategy_bucket"] == "ols"].copy()
    both_df = df[df["strategy_bucket"] == "both"].copy()
    if "edge_under_ols" in both_df.columns and "edge_under_xgb" in both_df.columns:
        both_df["_edge_mean"] = (both_df["edge_under_ols"] + both_df["edge_under_xgb"]) / 2

    if len(xgb_df) > 0 and "edge_under_xgb" in xgb_df.columns:
        print_bin_table("XGB ONLY", compute_bin_stats(xgb_df, "edge_under_xgb", EDGE_BINS, EDGE_LABELS), "edge")
    else:
        print("  [XGB ONLY] No data.\n")

    if len(ols_df) > 0 and "edge_under_ols" in ols_df.columns:
        print_bin_table("OLS ONLY", compute_bin_stats(ols_df, "edge_under_ols", EDGE_BINS, EDGE_LABELS), "edge")
    else:
        print("  [OLS ONLY] No data.\n")

    if len(both_df) > 0 and "_edge_mean" in both_df.columns:
        stats = compute_bin_stats(
            both_df, "_edge_mean", EDGE_BINS, EDGE_LABELS,
            extra_avg_cols=[("edge_under_ols", "avg_e_ols"), ("edge_under_xgb", "avg_e_xgb")],
        )
        print_bin_table("BOTH", stats, "edge", extra_col_headers=["avg_e_ols", "avg_e_xgb"])
    else:
        print("  [BOTH] No data.\n")


def main() -> None:
    ensure_repo_root_on_syspath()
    args = parse_args()
    bucket = args.bucket
    runs_prefix = args.runs_prefix.rstrip("/")

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sep = "=" * 70

    print(sep)
    print("NBA REBOUNDS — SETTLED RESULTS ANALYSIS")
    print(f"Generated: {now_utc}")
    if args.season:
        print(f"Season filter: {args.season}")
    if args.since:
        print(f"Since filter: {args.since}")
    print(sep)

    print("\nALL-TIME SUMMARY (from latest rollup)\n")
    section1_summary(bucket, runs_prefix)

    print(f"\n{sep}")
    print("WIN PROBABILITY BINS\n")
    df = load_settled_data(bucket, runs_prefix, args.season, args.since)
    if df.empty:
        print("  No settled data found.")
        return

    n_settled = int(df["result"].isin(["win", "loss", "push"]).sum())
    n_unsettled = int((df["result"] == "unsettled").sum())
    print(f"  Total rows: {len(df)} | Settled: {n_settled} | Unsettled: {n_unsettled}\n")

    section2_prob_bins(df)

    print(sep)
    print("EDGE BINS\n")
    section3_edge_bins(df)


if __name__ == "__main__":
    main()
