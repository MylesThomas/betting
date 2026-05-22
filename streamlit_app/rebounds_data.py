"""
Data loading functions for the NBA rebounds strategy dashboard.

S3 path convention:  rebounds/daily_runs/{slate_date}/{run_id}/{filename}
Run IDs are UTC ISO timestamps (e.g. 20260422T103045Z) — lexicographic max = latest run.

No st.* rendering here. No try/except. No defensive column checks.
"""

from __future__ import annotations

import json
from datetime import date
from io import BytesIO

import boto3
import pandas as pd
import streamlit as st

S3_BUCKET: str = "nba-betting-mt"
DAILY_RUNS_PREFIX: str = "rebounds/daily_runs"
PLAYED_BUCKETS: frozenset[str] = frozenset({"both", "ols", "xgb"})
# First date the pipeline ran in production (git: "rebounds: productionize daily run" 2026-04-01)
PROD_GO_LIVE_DATE: str = "2026-04-07"


def _s3_client() -> boto3.client:
    return boto3.client("s3")


def _read_parquet_from_s3(bucket: str, key: str) -> pd.DataFrame:
    body: bytes = _s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()
    return pd.read_parquet(BytesIO(body))


def _read_json_from_s3(bucket: str, key: str) -> dict:
    body: bytes = _s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()
    return json.loads(body)


@st.cache_data(ttl=900, show_spinner=False)
def _list_all_run_keys() -> list[str]:
    """Enumerate every key under rebounds/daily_runs/. 15-min TTL so today's files appear promptly."""
    paginator = _s3_client().get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=DAILY_RUNS_PREFIX + "/"):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def _parse_key_parts(key: str) -> tuple[str, str, str] | None:
    """
    Return (slate_date, run_id, filename) from a key shaped:
      rebounds/daily_runs/{slate_date}/{run_id}/{filename}
    Returns None for keys that don't match the expected three-part structure.
    """
    relative: str = key.removeprefix(DAILY_RUNS_PREFIX + "/")
    parts: list[str] = relative.split("/")
    if len(parts) != 3:
        return None
    slate_date, run_id, filename = parts
    return slate_date, run_id, filename


def _latest_key_for_date(
    all_keys: list[str], slate_date: str, filename_contains: str
) -> str | None:
    """
    Return the S3 key for the lexicographically latest run_id on slate_date whose
    filename contains filename_contains. Returns None if no match found.
    """
    candidates: list[tuple[str, str]] = []  # (run_id, full_key)
    for key in all_keys:
        parsed = _parse_key_parts(key)
        if parsed is None:
            continue
        key_date, run_id, filename = parsed
        if key_date == slate_date and filename_contains in filename:
            candidates.append((run_id, key))
    if not candidates:
        return None
    return max(candidates, key=lambda pair: pair[0])[1]


def _unique_slate_dates(all_keys: list[str]) -> list[str]:
    """Return sorted list of unique slate dates present in S3."""
    dates: set[str] = set()
    for key in all_keys:
        parsed = _parse_key_parts(key)
        if parsed is not None:
            dates.add(parsed[0])
    return sorted(dates)


@st.cache_data(ttl=3600, show_spinner=False)
def load_settled_plays() -> pd.DataFrame:
    """
    Concatenate all rebounds_scored_settled_{date}.parquet files from S3.
    Adds a float `diff` column (reb_actual - line) for display.
    Converts `date` column to datetime.
    Only dates that have a settled file are included — unsettled days are excluded entirely.
    """
    all_keys: list[str] = _list_all_run_keys()
    slate_dates: list[str] = _unique_slate_dates(all_keys)

    frames: list[pd.DataFrame] = []
    for slate_date in slate_dates:
        settled_key: str | None = _latest_key_for_date(
            all_keys, slate_date, f"rebounds_scored_settled_{slate_date}.parquet"
        )
        if settled_key is None:
            continue
        frames.append(_read_parquet_from_s3(S3_BUCKET, settled_key))

    if not frames:
        return pd.DataFrame()

    combined: pd.DataFrame = pd.concat(frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined["diff"] = combined["reb_actual"] - combined["line"]
    return combined


@st.cache_data(ttl=900, show_spinner=False)
def load_todays_scored() -> pd.DataFrame | None:
    """
    Load today's scored (unsettled) slate. Returns None if the pipeline hasn't run yet.
    15-min TTL so new files are picked up quickly.

    rebounds_scored_settled_{date}.parquet won't be matched because "settled_" is inserted
    between "rebounds_scored_" and the date, so f"rebounds_scored_{today}.parquet" is unique
    to the unsettled file.
    """
    today: str = str(date.today())
    all_keys: list[str] = _list_all_run_keys()
    scored_key: str | None = _latest_key_for_date(
        all_keys, today, f"rebounds_scored_{today}.parquet"
    )
    if scored_key is None:
        return None
    scored: pd.DataFrame = _read_parquet_from_s3(S3_BUCKET, scored_key)
    scored["date"] = pd.to_datetime(scored["date"])
    return scored


@st.cache_data(ttl=3600, show_spinner=False)
def load_run_manifests(n: int = 7) -> list[dict]:
    """Load the settlement_manifest.json for the n most recent settled dates."""
    all_keys: list[str] = _list_all_run_keys()
    recent_dates: list[str] = sorted(_unique_slate_dates(all_keys), reverse=True)

    manifests: list[dict] = []
    for slate_date in recent_dates:
        if len(manifests) >= n:
            break
        manifest_key: str | None = _latest_key_for_date(
            all_keys, slate_date, "settlement_manifest.json"
        )
        if manifest_key is None:
            continue
        manifest: dict = _read_json_from_s3(S3_BUCKET, manifest_key)
        manifest["date"] = slate_date
        manifests.append(manifest)

    return manifests
