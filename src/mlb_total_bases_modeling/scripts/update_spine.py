"""
Incremental spine update for the MLB batter total-bases model.

Steps:
  1. Load existing Statcast actuals from S3 (or full-refresh if --full)
  2. Fetch any new games since the last date in the actuals via pybaseball
  3. Upload refreshed actuals parquet to S3
  4. Rebuild rolling-feature spine from updated actuals
  5. Upload spine to S3

S3 paths read/written:
  s3://the-odds-api-mt/mlb/total_bases_model/actuals/mlb_batting_statcast.parquet
  s3://the-odds-api-mt/mlb/total_bases_model/spine/mlb_total_bases_spine.parquet

Usage:
  python src/mlb_total_bases_modeling/scripts/update_spine.py
  python src/mlb_total_bases_modeling/scripts/update_spine.py --full
  python src/mlb_total_bases_modeling/scripts/update_spine.py --verify
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

S3_BUCKET    = "the-odds-api-mt"
ACTUALS_KEY  = "mlb/total_bases_model/actuals/mlb_batting_statcast.parquet"
SPINE_KEY    = "mlb/total_bases_model/spine/mlb_total_bases_spine.parquet"

SES_SOURCE   = os.environ.get("SES_SOURCE", "").strip()
SES_TO_RAW   = os.environ.get("SES_TO", "mylescgthomas@gmail.com").strip()
ET           = ZoneInfo("America/New_York")

SEASON_START = {2024: "2024-03-20", 2025: "2025-03-18", 2026: "2026-03-25"}
ROLLING_WINDOWS = [1, 3, 5, 10, 20]

TB_EVENTS = {"single", "double", "triple", "home_run"}
AB_EVENTS = {
    "single", "double", "triple", "home_run",
    "strikeout", "strikeout_double_play",
    "field_out", "force_out", "grounded_into_double_play",
    "double_play", "triple_play", "field_error",
    "fielders_choice", "fielders_choice_out",
}

MANUAL_MAP = {
    "daniel vogelbach":   "Dan Vogelbach",
    "michael a taylor":   "Michael Taylor",
    "max muncy (2002)":   "Max Muncy",
    "diego a castillo":   "Diego Castillo",
    "james jarvis":       "Jim Jarvis",
    "donnie walton":      "Donovan Walton",
    "josh kuroda-grauer": "Joshua Kuroda-Grauer",
}


def _s3():
    return boto3.client("s3")


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = name.lower()
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = re.sub(r"[.,'\-]", "", name)
    name = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", name)
    name = re.sub(r"\s+", "", name)
    return name.strip()


# ── Statcast fetch ─────────────────────────────────────────────────────────────

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
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def statcast_to_batter_games(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw[raw["game_type"] == "R"].copy()
    raw["game_date"] = pd.to_datetime(raw["game_date"]).dt.date.astype(str)
    events = raw[raw["events"].notna()].copy()
    events["tb"] = (
        (events["events"] == "single").astype(int) * 1 +
        (events["events"] == "double").astype(int) * 2 +
        (events["events"] == "triple").astype(int) * 3 +
        (events["events"] == "home_run").astype(int) * 4
    )
    events["is_ab"]  = events["events"].isin(AB_EVENTS).astype(int)
    events["is_hit"] = events["events"].isin(TB_EVENTS).astype(int)
    agg = events.groupby(["game_date", "batter", "home_team", "away_team", "game_pk"]).agg(
        singles     = ("events", lambda x: (x == "single").sum()),
        doubles     = ("events", lambda x: (x == "double").sum()),
        triples     = ("events", lambda x: (x == "triple").sum()),
        home_runs   = ("events", lambda x: (x == "home_run").sum()),
        total_bases = ("tb", "sum"),
        ab          = ("is_ab", "sum"),
        hits        = ("is_hit", "sum"),
    ).reset_index()
    # Infer team
    events_team = (
        events[["game_date", "batter", "game_pk", "inning_topbot", "home_team", "away_team"]]
        .drop_duplicates(subset=["game_date", "batter", "game_pk"])
        .copy()
    )
    events_team["team"]     = events_team.apply(lambda r: r["home_team"] if r["inning_topbot"] == "Bot" else r["away_team"], axis=1)
    events_team["opponent"] = events_team.apply(lambda r: r["away_team"] if r["inning_topbot"] == "Bot" else r["home_team"], axis=1)
    agg = agg.merge(
        events_team[["game_date", "batter", "game_pk", "team", "opponent"]],
        on=["game_date", "batter", "game_pk"], how="left",
    ).drop_duplicates(subset=["game_date", "batter", "game_pk"])
    # Add batter name
    batter_ids = agg["batter"].dropna().unique().tolist()
    if batter_ids:
        nl = pb.playerid_reverse_lookup(batter_ids, key_type="mlbam")
        nl["player_name"] = nl["name_first"].str.title() + " " + nl["name_last"].str.title()
        agg = agg.merge(nl[["key_mlbam", "player_name"]].rename(columns={"key_mlbam": "batter"}), on="batter", how="left")
    # Season from game_date
    agg["season"] = agg["game_date"].str[:4].astype(int)
    return agg


# ── Load / save actuals ────────────────────────────────────────────────────────

def load_actuals_s3() -> pd.DataFrame | None:
    try:
        body = _s3().get_object(Bucket=S3_BUCKET, Key=ACTUALS_KEY)["Body"].read()
        return pd.read_parquet(BytesIO(body))
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise


def save_actuals_s3(df: pd.DataFrame) -> None:
    buf = BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    _s3().put_object(Bucket=S3_BUCKET, Key=ACTUALS_KEY, Body=buf.getvalue())
    print(f"Uploaded actuals → s3://{S3_BUCKET}/{ACTUALS_KEY}  ({len(df):,} rows)")


def load_spine_s3() -> pd.DataFrame | None:
    try:
        body = _s3().get_object(Bucket=S3_BUCKET, Key=SPINE_KEY)["Body"].read()
        return pd.read_parquet(BytesIO(body))
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise


# ── Rolling features ────────────────────────────────────────────────────────────

def build_rolling_features(actuals: pd.DataFrame) -> pd.DataFrame:
    manual_norm = {normalize_name(k): normalize_name(v) for k, v in MANUAL_MAP.items()}
    actuals = actuals[actuals["ab"] >= 1].copy()
    actuals["name_norm"] = actuals["player_name"].map(normalize_name).map(lambda n: manual_norm.get(n, n))
    actuals["game_date"] = pd.to_datetime(actuals["game_date"])

    # Aggregate doubleheaders at player-date level
    sum_cols  = ["total_bases", "home_runs", "singles", "doubles", "triples", "ab", "hits"]
    meta_cols = ["player_name", "season", "team", "opponent"]
    actuals = (
        actuals.groupby(["name_norm", "game_date"], sort=False)
        .agg(
            **{c: (c, "sum") for c in sum_cols if c in actuals.columns},
            **{c: (c, "first") for c in meta_cols if c in actuals.columns},
            is_doubleheader=("game_pk", "count"),
        )
        .reset_index()
    )
    actuals["is_doubleheader"] = (actuals["is_doubleheader"] > 1).astype(int)
    actuals = actuals.sort_values(["name_norm", "game_date"]).reset_index(drop=True)

    frames = []
    for player, grp in actuals.groupby("name_norm", sort=False):
        grp = grp.sort_values("game_date").reset_index(drop=True)
        grp["games_played_career"] = range(len(grp))
        for w in ROLLING_WINDOWS:
            grp[f"tb_L{w}"]   = grp["total_bases"].shift(1).rolling(w, min_periods=1).mean()
            grp[f"hr_L{w}"]   = grp["home_runs"].shift(1).rolling(w, min_periods=1).mean()
            grp[f"ab_L{w}"]   = grp["ab"].shift(1).rolling(w, min_periods=1).mean()
            grp[f"hits_L{w}"] = grp["hits"].shift(1).rolling(w, min_periods=1).mean()
        grp["tb_Lcareer"]   = grp["total_bases"].shift(1).expanding().mean()
        grp["hr_Lcareer"]   = grp["home_runs"].shift(1).expanding().mean()
        grp["ab_Lcareer"]   = grp["ab"].shift(1).expanding().mean()
        grp["tb_Lseason"]   = grp.groupby("season")["total_bases"].transform(lambda s: s.shift(1).expanding().mean())
        grp["days_rest"]    = grp["game_date"].diff().dt.days.fillna(0).astype(int)
        frames.append(grp)

    return pd.concat(frames, ignore_index=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def send_ses_spine(n_rows: int, n_players: int) -> None:
    if not SES_SOURCE or not SES_TO_RAW:
        return
    today = datetime.now(ET).strftime("%Y-%m-%d")
    subject = f"MLB Total Bases — Spine updated ({n_rows:,} rows · {n_players:,} players) — {today}"
    body = (
        f"<html><body style=\"font-family:sans-serif;color:#222;padding:20px\">"
        f"<h3 style='color:#2c3e50'>MLB Total Bases — Spine Update</h3>"
        f"<p><strong>{n_rows:,}</strong> player-dates &nbsp;·&nbsp; <strong>{n_players:,}</strong> players</p>"
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
    print(f"  Spine email sent to {to_list}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full",   action="store_true", help="Full re-fetch from 2024-03-20")
    parser.add_argument("--verify", action="store_true", help="Rebuild locally, compare vs S3")
    args = parser.parse_args()

    # Step 1: Load existing actuals
    existing = None if args.full else load_actuals_s3()

    if existing is None or args.full:
        start_date = "2024-03-20"
        print(f"Full fetch from {start_date}")
    else:
        last_date = existing["game_date"].max()
        start_date = (pd.Timestamp(last_date) + timedelta(days=1)).strftime("%Y-%m-%d")
        print(f"Incremental fetch from {start_date} (existing: {len(existing):,} rows, last={last_date})")

    end_date = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    if start_date > end_date:
        print("Already up to date — no new data to fetch.")
        new_rows = pd.DataFrame()
    else:
        print(f"Fetching Statcast {start_date} → {end_date} ...")
        raw = fetch_statcast_range(start_date, end_date)
        new_rows = statcast_to_batter_games(raw) if not raw.empty else pd.DataFrame()
        print(f"  New batter-games: {len(new_rows):,}")

    # Step 2: Combine and dedup
    if not new_rows.empty and existing is not None:
        combined = pd.concat([existing, new_rows], ignore_index=True)
        combined = combined.drop_duplicates(subset=["batter", "game_pk"]).reset_index(drop=True)
    elif not new_rows.empty:
        combined = new_rows
    else:
        combined = existing if existing is not None else pd.DataFrame()

    if combined.empty:
        print("No actuals data.")
        return

    combined = combined.sort_values(["game_date", "player_name"]).reset_index(drop=True)
    print(f"Total actuals: {len(combined):,} batter-games  |  {combined['game_date'].min()} → {combined['game_date'].max()}")

    if not args.verify:
        save_actuals_s3(combined)

    # Step 3: Build rolling-feature spine
    print("\nBuilding rolling features ...")
    spine = build_rolling_features(combined)
    spine["game_date"] = spine["game_date"].dt.strftime("%Y-%m-%d")
    print(f"  {len(spine):,} player-dates  |  {spine['name_norm'].nunique():,} players")

    if args.verify:
        existing_spine = load_spine_s3()
        if existing_spine is not None:
            print(f"S3 spine: {len(existing_spine):,} rows | Local: {len(spine):,} rows | Diff: {len(spine)-len(existing_spine):+,}")
        return

    # Step 4: Upload spine
    buf = BytesIO()
    spine.to_parquet(buf, index=False)
    buf.seek(0)
    _s3().put_object(Bucket=S3_BUCKET, Key=SPINE_KEY, Body=buf.getvalue())
    n_spine_rows    = len(spine)
    n_spine_players = spine["name_norm"].nunique()
    print(f"Uploaded spine → s3://{S3_BUCKET}/{SPINE_KEY}  ({n_spine_rows:,} rows)")
    print("Done.")

    send_ses_spine(n_spine_rows, n_spine_players)


if __name__ == "__main__":
    main()
