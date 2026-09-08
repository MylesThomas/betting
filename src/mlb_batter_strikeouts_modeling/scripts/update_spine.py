"""
Incremental spine update for the MLB batter strikeouts model.

Steps:
  1. Load existing Statcast actuals from S3 (or full-refresh if --full)
  2. Fetch any new games since the last date via pybaseball
  3. Upload refreshed actuals to S3
  4. Rebuild rolling-feature spine
  5. Upload spine to S3

S3 paths:
  s3://the-odds-api-mt/mlb/batter_strikeouts_model/actuals/mlb_batter_strikeouts_actuals.parquet
  s3://the-odds-api-mt/mlb/batter_strikeouts_model/spine/mlb_batter_strikeouts_spine.parquet

Usage:
  python src/mlb_batter_strikeouts_modeling/scripts/update_spine.py
  python src/mlb_batter_strikeouts_modeling/scripts/update_spine.py --full
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
import unicodedata
from datetime import date, datetime, timedelta
from io import BytesIO
from pathlib import Path

import boto3
import botocore.exceptions
import numpy as np
import pandas as pd
import pybaseball as pb
from dotenv import load_dotenv
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

pb.cache.enable()

S3_BUCKET   = "the-odds-api-mt"
ACTUALS_KEY = "mlb/batter_strikeouts_model/actuals/mlb_batter_strikeouts_actuals.parquet"
SPINE_KEY   = "mlb/batter_strikeouts_model/spine/mlb_batter_strikeouts_spine.parquet"

SES_SOURCE  = os.environ.get("SES_SOURCE", "").strip()
SES_TO_RAW  = os.environ.get("SES_TO", "mylescgthomas@gmail.com").strip()
ET          = ZoneInfo("America/New_York")

K_EVENTS = {"strikeout", "strikeout_double_play"}
PA_EVENTS = {
    "single", "double", "triple", "home_run",
    "strikeout", "strikeout_double_play",
    "field_out", "force_out", "grounded_into_double_play",
    "double_play", "triple_play", "field_error",
    "fielders_choice", "fielders_choice_out",
    "walk", "hit_by_pitch", "sac_fly", "sac_bunt",
    "intent_walk",
}

NAME_MAP = {
    "daniel vogelbach": "Dan Vogelbach",
    "michael a taylor": "Michael Taylor",
}


def _s3():
    return boto3.client("s3")


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = name.lower()
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = re.sub(r"[-]", " ", name)
    name = re.sub(r"[.,']", "", name)
    name = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", name)
    # collapse spaced initials (j. p. → jp)
    name = re.sub(r"\b([a-z])\s([a-z])\b", r"\1\2", name)
    # strip lone middle initials
    name = re.sub(r"\b([a-z])\b(?=\s+[a-z]{2})", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def fetch_statcast_range(start: str, end: str) -> pd.DataFrame:
    chunks = []
    cur = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    while cur <= end_ts:
        chunk_end = min(cur + pd.DateOffset(days=29), end_ts)
        s, e = cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        print(f"    chunk {s} → {e}", end=" ... ", flush=True)
        try:
            df = pb.statcast(start_dt=s, end_dt=e)
            print(f"{len(df):,} pitches")
            if not df.empty:
                chunks.append(df)
        except Exception as ex:
            print(f"ERROR: {ex}")
        cur = chunk_end + pd.DateOffset(days=1)
        time.sleep(0.5)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def statcast_to_batter_games(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw[raw["game_type"] == "R"].copy()
    raw["game_date"] = pd.to_datetime(raw["game_date"]).dt.date.astype(str)
    raw["season"] = raw["game_date"].str[:4].astype(int)

    events = raw[raw["events"].notna()].copy()
    events["k_count"] = events["events"].isin(K_EVENTS).astype(int)
    events["pa"]      = events["events"].isin(PA_EVENTS).astype(int)
    events["is_home_batter"] = (events["inning_topbot"] == "Bot").astype(int)

    agg = events.groupby(["game_date", "season", "batter", "game_pk", "home_team", "away_team"]).agg(
        strikeouts  = ("k_count", "sum"),
        plate_appearances = ("pa", "sum"),
        is_home     = ("is_home_batter", "first"),
    ).reset_index()

    # Name lookup
    batter_ids = agg["batter"].dropna().unique().tolist()
    if batter_ids:
        nl = pb.playerid_reverse_lookup(batter_ids, key_type="mlbam")
        nl["player_name"] = nl["name_first"].str.title() + " " + nl["name_last"].str.title()
        agg = agg.merge(nl[["key_mlbam", "player_name"]].rename(columns={"key_mlbam": "batter"}),
                        on="batter", how="left")

    # stand (batting side) — take first occurrence per batter-game
    stand_df = raw[raw["stand"].notna()].groupby(["game_date","batter","game_pk"])["stand"].first().reset_index()
    agg = agg.merge(stand_df, on=["game_date","batter","game_pk"], how="left")

    return agg


def build_rolling_features(actuals: pd.DataFrame) -> pd.DataFrame:
    manual_norm = {normalize_name(k): v for k, v in NAME_MAP.items()}

    # Drop rows with no plate appearances (DNP)
    actuals = actuals[actuals["plate_appearances"] >= 1].copy()
    actuals["name_norm"] = actuals["player_name"].map(normalize_name)
    actuals["name_norm"] = actuals["name_norm"].map(lambda n: manual_norm.get(n, n))
    actuals["game_date"] = pd.to_datetime(actuals["game_date"])

    # Collapse doubleheaders
    actuals = (
        actuals.groupby(["name_norm", "game_date"], sort=False)
        .agg(
            strikeouts        = ("strikeouts", "sum"),
            plate_appearances = ("plate_appearances", "sum"),
            player_name       = ("player_name", "first"),
            season            = ("season", "first"),
            is_home           = ("is_home", "first"),
            stand             = ("stand", "first"),
            home_team         = ("home_team", "first"),
            away_team         = ("away_team", "first"),
        )
        .reset_index()
    )
    actuals = actuals.sort_values(["name_norm", "game_date"]).reset_index(drop=True)

    frames = []
    for player, grp in actuals.groupby("name_norm", sort=False):
        grp = grp.sort_values("game_date").reset_index(drop=True)
        shifted = grp["strikeouts"].shift(1)
        pa_shifted = grp["plate_appearances"].shift(1)

        for w in [1, 5, 10, 20]:
            grp[f"k_roll_L{w}"] = shifted.rolling(w, min_periods=1).mean()

        grp["k_roll_season"]  = grp.groupby("season")["strikeouts"].transform(
            lambda s: s.shift(1).expanding().mean()
        )
        grp["k_roll_career"]  = shifted.expanding().mean()
        grp["pa_roll_L5"]     = pa_shifted.rolling(5, min_periods=1).mean()
        grp["pa_roll_career"] = pa_shifted.expanding().mean()
        grp["k_rate_L5"]      = (
            grp["strikeouts"].shift(1).rolling(5, min_periods=1).sum() /
            grp["plate_appearances"].shift(1).rolling(5, min_periods=1).sum()
        )
        grp["k_rate_career"]  = (
            grp["strikeouts"].shift(1).expanding().sum() /
            grp["plate_appearances"].shift(1).expanding().sum()
        )
        frames.append(grp)

    return pd.concat(frames, ignore_index=True)


def load_actuals_s3() -> pd.DataFrame | None:
    try:
        body = _s3().get_object(Bucket=S3_BUCKET, Key=ACTUALS_KEY)["Body"].read()
        return pd.read_parquet(BytesIO(body))
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    existing = None if args.full else load_actuals_s3()

    if existing is None or args.full:
        start_date = "2024-03-20"
        print(f"Full fetch from {start_date}")
    else:
        last_date = existing["game_date"].max()
        start_date = (pd.Timestamp(last_date) + timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"Incremental fetch from {start_date} (existing: {len(existing):,} rows)")

    end_date = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    if start_date > end_date:
        print("Already up to date.")
        new_rows = pd.DataFrame()
    else:
        print(f"Fetching Statcast {start_date} → {end_date}...")
        raw = fetch_statcast_range(start_date, end_date)
        new_rows = statcast_to_batter_games(raw) if not raw.empty else pd.DataFrame()
        print(f"  New batter-games: {len(new_rows):,}")

    if not new_rows.empty and existing is not None:
        combined = pd.concat([existing, new_rows], ignore_index=True)
        combined = combined.drop_duplicates(subset=["batter", "game_pk"]).reset_index(drop=True)
    elif not new_rows.empty:
        combined = new_rows
    else:
        combined = existing if existing is not None else pd.DataFrame()

    if combined.empty:
        print("No actuals data — exiting.")
        return

    combined = combined.sort_values(["game_date", "player_name"]).reset_index(drop=True)
    print(f"Total actuals: {len(combined):,}  |  {combined['game_date'].min()} → {combined['game_date'].max()}")

    # Upload actuals
    buf = BytesIO()
    combined.to_parquet(buf, index=False)
    buf.seek(0)
    _s3().put_object(Bucket=S3_BUCKET, Key=ACTUALS_KEY, Body=buf.getvalue())
    print(f"Uploaded actuals → s3://{S3_BUCKET}/{ACTUALS_KEY}")

    # Build + upload spine
    print("Building rolling features...")
    spine = build_rolling_features(combined)
    spine["game_date"] = spine["game_date"].dt.strftime("%Y-%m-%d")
    print(f"  {len(spine):,} player-dates  |  {spine['name_norm'].nunique():,} players")

    buf2 = BytesIO()
    spine.to_parquet(buf2, index=False)
    buf2.seek(0)
    _s3().put_object(Bucket=S3_BUCKET, Key=SPINE_KEY, Body=buf2.getvalue())
    print(f"Uploaded spine → s3://{S3_BUCKET}/{SPINE_KEY}")

    # Confirmation email
    if SES_SOURCE and SES_TO_RAW:
        today = datetime.now(ET).strftime("%Y-%m-%d")
        subject = f"MLB Batter Strikeouts — Spine updated ({len(spine):,} rows) — {today}"
        body = (
            f"<html><body style='font-family:sans-serif;padding:20px'>"
            f"<h3>MLB Batter Strikeouts — Spine Update</h3>"
            f"<p><strong>{len(spine):,}</strong> player-dates &nbsp;·&nbsp; "
            f"<strong>{spine['name_norm'].nunique():,}</strong> players</p>"
            f"<p style='color:#555;font-size:12px'>Updated: {today}</p>"
            f"</body></html>"
        )
        to_list = [e.strip() for e in SES_TO_RAW.split(",") if e.strip()]
        boto3.client("ses", region_name="us-east-2").send_email(
            Source=SES_SOURCE,
            Destination={"ToAddresses": to_list},
            Message={
                "Subject": {"Data": subject, "Charset": "UTF-8"},
                "Body": {"Html": {"Data": body, "Charset": "UTF-8"}},
            },
        )
        print(f"  Spine email sent.")

    print("Done.")


if __name__ == "__main__":
    main()
