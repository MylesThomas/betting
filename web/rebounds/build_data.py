"""
Generate data.json for the rebounds strategy dashboard.

Reads from S3:
  - Latest all_time.csv rollup  → bucket summary (ROI, hit rate, PnL)
  - Settled parquet files       → 30-day ROI trend + probability calibration
  - Today's scored parquet      → current plays

Outputs: web/rebounds/data.json  (or --output path)

Usage:
  python web/rebounds/build_data.py
  python web/rebounds/build_data.py --season 2025-26
  python web/rebounds/build_data.py --output /tmp/data.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta, timezone
from io import BytesIO, StringIO
from pathlib import Path

import numpy as np
import pandas as pd


BUCKET = "nba-betting-mt"
RUNS_PREFIX = "rebounds/daily_runs"

PROB_BINS = [(50, 60), (60, 70), (70, 80), (80, 90), (90, 101)]
PROB_BIN_LABELS = ["50–60%", "60–70%", "70–80%", "80–90%", "90–100%"]
PROB_BIN_MIDS = [55, 65, 75, 85, 95]


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
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", default=BUCKET)
    p.add_argument("--runs-prefix", default=RUNS_PREFIX)
    p.add_argument("--season", default="", help="Filter to one season, e.g. 2025-26")
    p.add_argument("--output", default="", help="Output path (default: same dir as this script)")
    return p.parse_args()


def _s3():
    import boto3
    return boto3.client("s3")


def _read_csv_s3(bucket: str, key: str) -> pd.DataFrame:
    body = _s3().get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
    return pd.read_csv(StringIO(body))


def _read_parquet_s3(bucket: str, key: str) -> pd.DataFrame:
    body = _s3().get_object(Bucket=bucket, Key=key)["Body"].read()
    return pd.read_parquet(BytesIO(body))


# ── S3 key discovery ──────────────────────────────────────

def find_latest_rollup(bucket: str, runs_prefix: str) -> str | None:
    paginator = _s3().get_paginator("list_objects_v2")
    candidates: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=runs_prefix.rstrip("/") + "/"):
        for item in page.get("Contents", []):
            key = item["Key"]
            if key.endswith("/all_time.csv"):
                candidates.append(key)
    return sorted(candidates)[-1] if candidates else None


def _latest_per_date(all_keys: list[str]) -> list[str]:
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


def list_settled_keys(bucket: str, runs_prefix: str) -> list[str]:
    paginator = _s3().get_paginator("list_objects_v2")
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
    return _latest_per_date(all_keys)


def list_scored_keys(bucket: str, runs_prefix: str) -> list[str]:
    paginator = _s3().get_paginator("list_objects_v2")
    all_keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=runs_prefix.rstrip("/") + "/"):
        for item in page.get("Contents", []):
            key = item["Key"]
            fname = key.split("/")[-1]
            if (
                fname.startswith("rebounds_scored_")
                and "settled" not in fname
                and key.endswith(".parquet")
                and "_rollups/" not in key
            ):
                all_keys.append(key)
    return _latest_per_date(all_keys)


# ── Rollup aggregation (mirrors analyze_settled_results.py) ──

def _aggregate_rollup(df: pd.DataFrame) -> pd.DataFrame:
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
            pnl_units=("pnl_units", "sum"),
            _prob_sum=("_prob_sum", "sum"),
        )
    )
    settled = agg["n_win"] + agg["n_loss"] + agg["n_push"]
    agg["hit_rate"] = np.where(
        (agg["n_win"] + agg["n_loss"]) > 0,
        agg["n_win"] / (agg["n_win"] + agg["n_loss"]),
        np.nan,
    )
    agg["roi"] = np.where(settled > 0, agg["pnl_units"] / settled, np.nan)
    return agg.drop(columns=["_prob_sum"])


def build_bucket_stats(bucket: str, runs_prefix: str) -> dict:
    key = find_latest_rollup(bucket, runs_prefix)
    if key is None:
        print("  [WARN] No rollup found — bucket stats will be empty")
        return {}
    raw = _read_csv_s3(bucket, key)
    df = _aggregate_rollup(raw)

    result: dict = {}
    for _, row in df.iterrows():
        b = str(row["strategy_bucket"])
        if b == "neither":
            continue
        result[b] = {
            "roi":      round(float(row["roi"]) * 100, 2) if not pd.isna(row["roi"]) else None,
            "hit_rate": round(float(row["hit_rate"]) * 100, 1) if not pd.isna(row["hit_rate"]) else None,
            "wins":     int(row["n_win"]),
            "losses":   int(row["n_loss"]),
            "pushes":   int(row["n_push"]),
            "pnl":      round(float(row["pnl_units"]), 2),
            "bets":     int(row["n_bets"]),
        }
    return result


def build_overall_stats(buckets: dict) -> dict:
    """Hero KPIs: use BOTH bucket for ROI/hit-rate; sum all buckets for PnL/bets."""
    hero = buckets.get("both", {})
    total_pnl = sum(b["pnl"] for b in buckets.values())
    total_bets = sum(b["bets"] for b in buckets.values())
    return {
        "roi":        hero.get("roi"),
        "hit_rate":   hero.get("hit_rate"),
        "total_pnl":  round(total_pnl, 2),
        "total_bets": total_bets,
    }


# ── Settled data for trend + calibration ─────────────────

def load_recent_settled(bucket: str, runs_prefix: str, days: int = 67) -> pd.DataFrame:
    cutoff = (datetime.now(timezone.utc).date() - timedelta(days=days)).isoformat()
    keys = list_settled_keys(bucket, runs_prefix)
    recent = [k for k in keys if k.split("/")[-3] >= cutoff]
    frames: list[pd.DataFrame] = []
    for key in recent:
        try:
            frames.append(_read_parquet_s3(bucket, key))
        except Exception as e:
            print(f"  [WARN] Skipping {key}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def compute_roi_trend(df: pd.DataFrame, n_days: int = 30, window: int = 7) -> list[float | None]:
    if df.empty:
        return [None] * n_days
    both = df[df["strategy_bucket"] == "both"].copy()
    if both.empty:
        return [None] * n_days

    both["date"] = pd.to_datetime(both["date"]).dt.date
    end_date = both["date"].max()

    values: list[float | None] = []
    for i in range(n_days):
        day = end_date - timedelta(days=(n_days - 1 - i))
        win_start = day - timedelta(days=window - 1)
        mask = (both["date"] >= win_start) & (both["date"] <= day)
        grp = both[mask & both["result"].isin(["win", "loss", "push"])]
        if len(grp) < 3:
            values.append(None)
        else:
            roi = float(grp["pnl_units"].sum() / len(grp) * 100)
            values.append(round(roi, 2))
    return values


def compute_calibration(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    both = df[df["strategy_bucket"] == "both"].copy()
    settled = both[both["result"].isin(["win", "loss"])].copy()
    if settled.empty:
        return []

    if "p_under_xgb" in settled.columns and "p_under_ols" in settled.columns:
        settled["p_model"] = (settled["p_under_xgb"] + settled["p_under_ols"]) / 2
    elif "p_under_xgb" in settled.columns:
        settled["p_model"] = settled["p_under_xgb"]
    elif "p_under_ols" in settled.columns:
        settled["p_model"] = settled["p_under_ols"]
    else:
        return []

    rows = []
    for (lo, hi), label, mid in zip(PROB_BINS, PROB_BIN_LABELS, PROB_BIN_MIDS):
        mask = (settled["p_model"] * 100 >= lo) & (settled["p_model"] * 100 < hi)
        grp = settled[mask]
        n = len(grp)
        actual = round(float((grp["result"] == "win").mean() * 100), 1) if n > 0 else None
        rows.append({"bin": label, "actual": actual, "expected": float(mid), "n": n})
    return rows


# ── Today's plays ─────────────────────────────────────────

_BUCKET_RANK = {"both": 0, "ols": 1, "xgb": 2}


def _derive_bucket(row: pd.Series) -> str:
    if row.get("play_both", False):     return "both"
    if row.get("play_ols_only", False): return "ols"
    if row.get("play_xgb_only", False): return "xgb"
    return ""


def _row_edge(row: pd.Series, bucket: str) -> float:
    if bucket == "both":
        return ((row.get("edge_under_ols") or 0) + (row.get("edge_under_xgb") or 0)) / 2
    if bucket == "xgb":
        return row.get("edge_under_xgb") or 0
    return row.get("edge_under_ols") or 0


def _load_game_context(bucket: str, run_prefix: str, target_date: str) -> dict[str, dict]:
    """Return {odds_event_id: {away_team, home_team, game_time}} from the raw props CSV."""
    s3 = _s3()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=run_prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if f"live_rebounds_props_raw_{target_date}.csv" in key:
                body = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
                raw = pd.read_csv(StringIO(body))
                ctx: dict[str, dict] = {}
                for _, r in raw.drop_duplicates("odds_api_event_id").iterrows():
                    ctx[str(r["odds_api_event_id"])] = {
                        "away_team": str(r.get("away_team", "")),
                        "home_team": str(r.get("home_team", "")),
                        "game_time": str(r.get("game_time", "")),
                    }
                return ctx
    return {}


def build_todays_plays(bucket: str, runs_prefix: str, target_date: str) -> list[dict]:
    keys = list_scored_keys(bucket, runs_prefix)
    todays_keys = [k for k in keys if "/" + target_date + "/" in k]
    if not todays_keys:
        print(f"  [INFO] No scored parquet found for {target_date}")
        return []

    try:
        df = _read_parquet_s3(bucket, todays_keys[-1])
    except Exception as e:
        print(f"  [WARN] Could not read today's scored parquet: {e}")
        return []

    # Filter to qualifying plays using boolean columns
    # (strategy_bucket is only present in settled parquets, not scored)
    if "play_both" in df.columns:
        mask = df["play_both"] | df["play_ols_only"] | df["play_xgb_only"]
        df = df[mask].copy()
    elif "strategy_bucket" in df.columns:
        df = df[df["strategy_bucket"].isin(["both", "ols", "xgb"])].copy()

    if df.empty:
        return []

    # Derive bucket and sort edge
    df["_bucket"] = df.apply(_derive_bucket, axis=1)
    df["_edge"]   = df.apply(lambda r: _row_edge(r, r["_bucket"]), axis=1)
    df["_brank"]  = df["_bucket"].map(_BUCKET_RANK).fillna(3)

    # One row per player: best bucket, then best edge within bucket
    df = df.sort_values(["_brank", "_edge"], ascending=[True, False])
    df = df.drop_duplicates(subset=["player_normalized"], keep="first")

    # Load game context (team names, game time) from raw props CSV
    run_prefix = "/".join(todays_keys[-1].split("/")[:-1]) + "/"
    game_ctx = _load_game_context(bucket, run_prefix, target_date)

    plays: list[dict] = []
    for _, row in df.iterrows():
        bkt  = str(row["_bucket"])
        edge = float(row["_edge"])

        if bkt == "both":
            prob = ((row.get("p_under_ols") or 0) + (row.get("p_under_xgb") or 0)) / 2
        elif bkt == "xgb":
            prob = row.get("p_under_xgb") or 0
        else:
            prob = row.get("p_under_ols") or 0

        line = row.get("consensus_reb_line") or row.get("min_line") or row.get("max_line")

        # Game context from raw CSV
        event_id = str(row.get("odds_event_id", ""))
        ctx = game_ctx.get(event_id, {})
        away = ctx.get("away_team", "")
        home = ctx.get("home_team", "")
        matchup = f"{away} @ {home}" if away and home else ""
        game_time = ctx.get("game_time", "")

        plays.append({
            "player":    str(row.get("player_normalized", "Unknown")),
            "matchup":   matchup,
            "game_time": game_time,
            "line":      float(line) if line is not None else None,
            "prob":      round(prob * 100, 1),
            "edge":      round(edge * 100, 1),
            "bucket":    bkt,
        })

    plays.sort(key=lambda p: p["prob"] or 0, reverse=True)
    return plays


# ── Entry point ───────────────────────────────────────────

def main() -> None:
    ensure_repo_root_on_syspath()
    args = parse_args()
    bucket = args.bucket
    runs_prefix = args.runs_prefix.rstrip("/")
    today = date.today().isoformat()

    print(f"build_data.py — {today}")

    print("  [1/5] Bucket stats from rollup...")
    buckets = build_bucket_stats(bucket, runs_prefix)

    print("  [2/5] Overall stats...")
    overall = build_overall_stats(buckets)

    print("  [3/5] Loading recent settled data (last 67 days)...")
    settled_df = load_recent_settled(bucket, runs_prefix)
    if args.season and not settled_df.empty and "season" in settled_df.columns:
        settled_df = settled_df[settled_df["season"].astype(str) == args.season]
    print(f"         {len(settled_df)} rows loaded")

    print("  [4/5] ROI trend + calibration...")
    roi_trend = compute_roi_trend(settled_df)
    calibration = compute_calibration(settled_df)

    print("  [5/5] Today's plays...")
    todays_plays = build_todays_plays(bucket, runs_prefix, today)
    print(f"         {len(todays_plays)} plays for {today}")

    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "todays_date":  today,
        "overall":      overall,
        "buckets":      buckets,
        "roi_trend":    roi_trend,
        "calibration":  calibration,
        "todays_plays": todays_plays,
    }

    out_path = Path(args.output) if args.output else Path(__file__).parent / "data.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"  → {out_path}")

    # Also write data.js so dashboard works when opened as file:// (fetch blocked by browser)
    js_path = out_path.with_suffix(".js")
    js_path.write_text(f"window.DASHBOARD_DATA = {json.dumps(payload, indent=2)};")
    print(f"  → {js_path}")


if __name__ == "__main__":
    main()
