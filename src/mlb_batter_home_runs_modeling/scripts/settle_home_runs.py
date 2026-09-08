"""
Settle yesterday's MLB Batter Home Runs bets.

  1. Load yesterday's recommendations CSV from S3
  2. Load Statcast actuals from S3
  3. For each bet: over → home_runs >= 1? (hits line 0.5 over)
  4. Compute P&L: win → +(decimal_price - 1), lose → -1
  5. Update settled_bets parquet in S3
  6. Print/email settlement summary

S3 paths read:
  s3://the-odds-api-mt/mlb/batter_home_runs_model/daily_runs/{gameday}/recommendations.csv
  s3://the-odds-api-mt/mlb/total_bases_model/actuals/mlb_batting_statcast.parquet

S3 paths written:
  s3://the-odds-api-mt/mlb/batter_home_runs_model/settled/mlb_batter_hr_settled_bets.parquet

Usage:
  python src/mlb_batter_home_runs_modeling/scripts/settle_home_runs.py
  python src/mlb_batter_home_runs_modeling/scripts/settle_home_runs.py --gameday 2026-03-27
  python src/mlb_batter_home_runs_modeling/scripts/settle_home_runs.py --output /tmp/hr_settle.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import unicodedata
from datetime import date, datetime, timedelta
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import boto3
import numpy as np
import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

ET = ZoneInfo("America/New_York")

SES_SOURCE = os.environ.get("SES_SOURCE", "").strip()
SES_TO_RAW = os.environ.get("SES_TO", "mylescgthomas@gmail.com").strip()

S3_BUCKET       = "the-odds-api-mt"
DAILY_PREFIX    = "mlb/batter_home_runs_model/daily_runs"
SETTLED_KEY     = "mlb/batter_home_runs_model/settled/mlb_batter_hr_settled_bets.parquet"
ACTUALS_KEY     = "mlb/total_bases_model/actuals/mlb_batting_statcast.parquet"


def normalize_name(name: str) -> str:
    import re
    if not isinstance(name, str):
        return ""
    name = name.lower().strip()
    name = unicodedata.normalize("NFD", name)
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = re.sub(r"\s*\(\d{4}\)", "", name)
    name = re.sub(r"[''`]", "", name)
    name = re.sub(r"[-]", " ", name)
    name = re.sub(r"\.", "", name)
    name = re.sub(r"\s+(jr|sr|ii|iii|iv)$", "", name)
    name = re.sub(r"\b([a-z]) ([a-z])\b", r"\1\2", name)
    name = re.sub(r"\b([a-z]) ([a-z])\b", r"\1\2", name)
    name = re.sub(r"(?<=\s)[a-z](?=\s)", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    NAME_MAP = {
        "daniel vogelbach":   "dan vogelbach",
        "donnie walton":      "donovan walton",
        "eddy alvarez":       "francisco alvarez",
        "josh kuroda grauer": "joshua kuroda grauer",
    }
    return NAME_MAP.get(name, name)


def load_recommendations(gameday: str) -> pd.DataFrame:
    s3  = boto3.client("s3")
    key = f"{DAILY_PREFIX}/{gameday}/recommendations.csv"
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
        df  = pd.read_csv(BytesIO(obj["Body"].read()))
        print(f"  Loaded {len(df):,} recommendations for {gameday}")
        return df
    except s3.exceptions.NoSuchKey:
        print(f"  No recommendations found for {gameday} at s3://{S3_BUCKET}/{key}")
        return pd.DataFrame()
    except Exception as e:
        print(f"  Error loading recommendations: {e}")
        return pd.DataFrame()


def load_actuals(gameday: str) -> pd.DataFrame:
    """Load Statcast batting stats for the given date."""
    s3  = boto3.client("s3")
    obj = s3.get_object(Bucket=S3_BUCKET, Key=ACTUALS_KEY)
    df  = pd.read_parquet(BytesIO(obj["Body"].read()))
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime("%Y-%m-%d")
    day = df[df["game_date"] == gameday]
    print(f"  Actuals for {gameday}: {len(day):,} rows")
    return day


def settle(recs: pd.DataFrame, actuals: pd.DataFrame, gameday: str) -> pd.DataFrame:
    """Join recommendations to actuals and compute P&L."""
    actuals = actuals.copy()
    actuals["player_key"] = actuals["player_name"].apply(normalize_name) if "player_name" in actuals.columns else actuals.get("player_key", pd.Series())
    actuals = actuals[["player_key", "game_date", "home_runs"]].dropna(subset=["player_key"])

    # Sum HRs per player-game (might be multiple rows from PBP)
    act_agg = actuals.groupby(["player_key", "game_date"])["home_runs"].sum().reset_index()

    recs = recs.copy()
    if "player_key" not in recs.columns and "player_name" in recs.columns:
        recs["player_key"] = recs["player_name"].apply(normalize_name)
    recs["game_date"] = gameday

    merged = recs.merge(act_agg, on=["player_key", "game_date"], how="left")

    def _settle_row(r):
        hr   = r.get("home_runs")
        line = float(r.get("line", 0.5))
        if pd.isna(hr):
            return "pending", np.nan
        if hr == line:
            return "push", 0.0
        direction = str(r.get("tier", "play")).lower()
        is_over   = hr > line
        if is_over:
            price = float(r.get("over_price", 2.0)) if not pd.isna(r.get("over_price", np.nan)) else np.nan
            pnl   = price - 1.0 if not pd.isna(price) else np.nan
            return "win", pnl
        else:
            price = float(r.get("over_price", 2.0)) if not pd.isna(r.get("over_price", np.nan)) else np.nan
            return "loss", -1.0

    results = merged.apply(_settle_row, axis=1, result_type="expand")
    merged["result"] = results[0]
    merged["pnl"]    = results[1]
    merged["home_runs_actual"] = merged["home_runs"]
    return merged


def update_settled_history(new_settled: pd.DataFrame) -> pd.DataFrame:
    s3 = boto3.client("s3")
    try:
        obj  = s3.get_object(Bucket=S3_BUCKET, Key=SETTLED_KEY)
        hist = pd.read_parquet(BytesIO(obj["Body"].read()))
    except Exception:
        hist = pd.DataFrame()

    # Remove any previously settled rows for the same gameday to avoid duplication
    gameday = new_settled["game_date"].iloc[0] if len(new_settled) > 0 else None
    if gameday and len(hist) > 0 and "game_date" in hist.columns:
        hist = hist[hist["game_date"] != gameday]

    combined = pd.concat([hist, new_settled], ignore_index=True)

    buf = BytesIO()
    combined.to_parquet(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=S3_BUCKET, Key=SETTLED_KEY, Body=buf.read())
    print(f"  Settled history updated: {len(combined):,} total rows → s3://{S3_BUCKET}/{SETTLED_KEY}")
    return combined


def print_summary(settled: pd.DataFrame, gameday: str) -> dict:
    finished = settled[settled["result"].isin(["win", "loss", "push"])]
    n_total  = len(settled)
    n_settled = len(finished)
    n_win    = (finished["result"] == "win").sum()
    n_loss   = (finished["result"] == "loss").sum()
    n_push   = (finished["result"] == "push").sum()
    net      = finished["pnl"].sum() if "pnl" in finished.columns else 0.0

    print(f"\n{'='*60}")
    print(f"Settlement for {gameday}")
    print(f"  Total recs: {n_total}  |  Settled: {n_settled}  |  Pending: {n_total - n_settled}")
    print(f"  Results: {n_win}W / {n_loss}L / {n_push}P  |  Net: {net:+.2f}u")
    if n_settled > 0:
        print(f"  Win rate: {n_win/n_settled:.1%}  |  ROI: {net/n_settled*100:+.1f}%")
    print(f"{'='*60}\n")

    # Season totals from S3
    try:
        s3   = boto3.client("s3")
        obj  = s3.get_object(Bucket=S3_BUCKET, Key=SETTLED_KEY)
        hist = pd.read_parquet(BytesIO(obj["Body"].read()))
        prod = hist[hist["result"].isin(["win", "loss"])]
        s_wins = (prod["result"] == "win").sum()
        s_loss = (prod["result"] == "loss").sum()
        s_net  = prod["pnl"].sum() if "pnl" in prod.columns else 0.0
        print(f"  Season totals: {s_wins}W / {s_loss}L  |  {s_net:+.2f}u")
    except Exception:
        s_wins, s_loss, s_net = 0, 0, 0.0

    return {
        "gameday": gameday,
        "yesterday_wins":   int(n_win),
        "yesterday_losses": int(n_loss),
        "yesterday_units":  float(net),
        "season_wins":      int(s_wins),
        "season_losses":    int(s_loss),
        "season_units":     float(s_net),
    }


def main(gameday: str | None = None, output: str | None = None) -> dict:
    if not gameday:
        yesterday = date.today() - timedelta(days=1)
        gameday   = yesterday.strftime("%Y-%m-%d")
    print(f"\nMLB Batter Home Runs settle | gameday={gameday}")

    recs    = load_recommendations(gameday)
    if recs.empty:
        result = {"gameday": gameday, "yesterday_wins": 0, "yesterday_losses": 0,
                  "yesterday_units": 0.0, "season_wins": 0, "season_losses": 0, "season_units": 0.0}
        if output:
            Path(output).write_text(json.dumps(result))
        return result

    actuals = load_actuals(gameday)
    if actuals.empty:
        print("  No actuals yet — bets remain pending")
        result = {"gameday": gameday, "yesterday_wins": 0, "yesterday_losses": 0,
                  "yesterday_units": 0.0, "season_wins": 0, "season_losses": 0, "season_units": 0.0}
        if output:
            Path(output).write_text(json.dumps(result))
        return result

    settled = settle(recs, actuals, gameday)
    update_settled_history(settled)
    result  = print_summary(settled, gameday)

    if output:
        Path(output).write_text(json.dumps(result))

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gameday", default=None)
    parser.add_argument("--output",  default=None)
    args = parser.parse_args()
    main(gameday=args.gameday, output=args.output)
