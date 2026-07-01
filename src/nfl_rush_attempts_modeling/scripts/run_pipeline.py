"""
Live gameday pipeline for NFL rush attempts props.

Production strategy: QB only, book_line >= 6.5, UNDER, edge >= 0.03
  edge = p_model - p_market  (negative = model favors under vs market)
  Bet filter: position=QB, book_line>=6.5, -edge >= 0.03

For each qualifying QB on the given gameday:
  1. Fetch today's event IDs and rush attempt props from The Odds API
  2. Join with rolling features from the spine (downloaded from S3)
  3. Score with Ridge model (downloaded from S3)
  4. Compute P(over) via stratified residual KDE CDFs
  5. Filter to production strategy bets
  6. Send SES email + SNS notification with bet sheet
  7. Save recommendations CSV to S3

S3 paths read:
  s3://the-odds-api-mt/nfl/rush_attempts_model/spine/nfl_rush_attempts_spine.parquet
  s3://the-odds-api-mt/nfl/rush_attempts_model/artifacts/best_model.pkl
  s3://the-odds-api-mt/nfl/rush_attempts_model/artifacts/residual_cdfs.pkl
  s3://the-odds-api-mt/nfl/rush_attempts_model/settled/settled_bets.parquet  (optional)

S3 paths written:
  s3://the-odds-api-mt/nfl/rush_attempts_model/daily_runs/{gameday}/recommendations.csv

Run:
  python src/nfl_rush_attempts_modeling/scripts/run_pipeline.py --gameday 2026-09-11
  python src/nfl_rush_attempts_modeling/scripts/run_pipeline.py  # defaults to today ET
"""

from __future__ import annotations

import argparse
import html as html_module
import os
import pickle
import re
import sys
import time
import warnings
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import boto3
import botocore.exceptions
import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from src.nfl_rush_attempts_modeling.scripts.step2_build_spine import _norm

ODDS_API_KEY  = os.environ.get("ODDS_API_KEY", "").strip()
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT         = "americanfootball_nfl"
MARKET        = "player_rush_attempts"
BOOKMAKERS    = "draftkings,espnbet,betmgm,hardrockbet,fliff,betonlineag,bovada,betrivers,ballybet,betparx,williamhill_us,fanatics,fanduel"
REGIONS       = "us"
SLEEP_S       = 0.25

S3_BUCKET   = "the-odds-api-mt"
S3_PREFIX   = "nfl/rush_attempts_model"
SES_SOURCE  = os.environ.get("SETTLEMENT_SES_SOURCE", "").strip()
SES_TO_RAW  = os.environ.get("SETTLEMENT_SES_TO", "mylescgthomas@gmail.com").strip()
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "").strip()
ENABLE_SNS  = os.environ.get("ENABLE_SNS", "").strip().lower() in ("1", "true", "yes")

ET = ZoneInfo("America/New_York")

# ── Production strategy params ──────────────────────────────────────────────
PROD_POSITION  = "QB"
PROD_MIN_LINE  = 6.5
PROD_DIRECTION = "under"
PROD_EDGE_THRESH = 0.03   # -edge >= 0.03

PRED_BINS   = [0, 5, 10, 15, 20, np.inf]
PRED_LABELS = ["lt5", "5to9", "10to14", "15to19", "20plus"]
CARRY_BINS  = PRED_BINS
CARRY_LABELS = PRED_LABELS

_MONO = "ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace"
_SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif"

_SUFFIX_RE  = re.compile(r"\s*,?\s*(Jr\.?|Sr\.?|II{1,2}|IV|V)\.?$", re.IGNORECASE)
_SPECIAL_RE = re.compile(r"['\.\-,]")
_INIT_RE    = re.compile(r"(?<!\w)([a-z])\s([a-z])(?=\s|\b)")


# ── Helpers ──────────────────────────────────────────────────────────────────

def today_et() -> str:
    return datetime.now(ET).strftime("%Y-%m-%d")


def current_nfl_season() -> int:
    now = datetime.now(ET)
    return now.year if now.month >= 8 else now.year - 1


def _s3():
    return boto3.client("s3")


def load_s3_parquet(key: str) -> pd.DataFrame | None:
    try:
        body = _s3().get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
        return pd.read_parquet(BytesIO(body))
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise


def load_s3_pkl(key: str) -> object:
    body = _s3().get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
    return pickle.loads(body)


def save_s3_csv(df: pd.DataFrame, key: str) -> None:
    buf = BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    _s3().put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue())
    print(f"  Saved → s3://{S3_BUCKET}/{key}")


# ── Feature engineering (mirrors step3_train.py exactly) ─────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    carry_tier = pd.cut(df["carry_rate_L8"], bins=CARRY_BINS,
                        labels=CARRY_LABELS, right=False)
    for label in CARRY_LABELS[1:]:
        df[f"carry_bucket_L8_{label}"] = (carry_tier == label).astype(int)
    line_tier = pd.cut(df["consensus_point"], bins=CARRY_BINS,
                       labels=CARRY_LABELS, right=False)
    for label in CARRY_LABELS[1:]:
        df[f"line_bucket_{label}"] = (line_tier == label).astype(int)
    df["line_deviation"]  = df["consensus_point"] - df["carry_rate_L8"]
    df["carry_trend"]     = df["carry_rate_L3"] - df["carry_rate_L5"]
    df["bell_cow_flag"]   = (df["carry_rate_Lcareer"] >= 12).astype(int)
    df["line_x_pos_RB"]   = df["consensus_point"] * df["pos_RB"]
    df["line_x_bell_cow"] = df["consensus_point"] * df["bell_cow_flag"]
    return df


# ── P(over) from residual CDF ─────────────────────────────────────────────────

def p_over(shortfall: float, bucket: str, cdfs: dict) -> float:
    return float(1.0 - cdfs[bucket](shortfall))


# ── Odds API calls ────────────────────────────────────────────────────────────

def _get(url: str, params: dict) -> dict:
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    time.sleep(SLEEP_S)
    return resp.json()


def fetch_events(gameday: str) -> list[dict]:
    """Fetch today's NFL events and filter to gameday."""
    events = _get(
        f"{ODDS_API_BASE}/sports/{SPORT}/events",
        {"apiKey": ODDS_API_KEY, "dateFormat": "iso"},
    )
    gd = pd.to_datetime(gameday).date()
    day_events = [
        e for e in events
        if pd.to_datetime(e["commence_time"]).astimezone(ET).date() == gd
    ]
    print(f"  Events on {gameday}: {len(day_events)}")
    return day_events


def fetch_rush_attempt_props(event_ids: list[str]) -> list[dict]:
    """Fetch player_rush_attempts props for given events."""
    all_props = []
    for eid in event_ids:
        try:
            data = _get(
                f"{ODDS_API_BASE}/sports/{SPORT}/events/{eid}/odds",
                {
                    "apiKey":     ODDS_API_KEY,
                    "regions":    REGIONS,
                    "markets":    MARKET,
                    "bookmakers": BOOKMAKERS,
                    "oddsFormat": "american",
                },
            )
            all_props.append(data)
        except requests.HTTPError as e:
            print(f"  WARN: event {eid} props failed ({e})")
    return all_props


def parse_props(raw_events: list[dict], gameday: str) -> pd.DataFrame:
    """Parse raw Odds API response into per-player-book rows."""
    rows = []
    for event in raw_events:
        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")
        for book in event.get("bookmakers", []):
            bookmaker = book["key"]
            for market in book.get("markets", []):
                if market["key"] != MARKET:
                    continue
                for outcome in market.get("outcomes", []):
                    side = outcome.get("name", "").lower()  # "over" or "under"
                    price = outcome.get("price")
                    player = outcome.get("description", "") or outcome.get("name", "")
                    line  = outcome.get("point")
                    if line is None or price is None:
                        continue
                    rows.append({
                        "gameday":   gameday,
                        "event_id":  event["id"],
                        "home_team": home_team,
                        "away_team": away_team,
                        "bookmaker": bookmaker,
                        "player_raw": player,
                        "player_norm": _norm(player),
                        "side":       side,
                        "line":       float(line),
                        "price":      int(price),
                    })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Pivot to one row per player-bookmaker-line with over/under prices
    over  = df[df["side"] == "over"].rename(columns={"price": "over_price", "player_raw": "player_display_name"})
    under = df[df["side"] == "under"].rename(columns={"price": "under_price"})

    merged = over[["gameday", "event_id", "home_team", "away_team", "bookmaker",
                   "player_display_name", "player_norm", "line", "over_price"]].merge(
        under[["event_id", "bookmaker", "player_norm", "line", "under_price"]],
        on=["event_id", "bookmaker", "player_norm", "line"],
        how="inner",
    )

    # Market-implied P(over) — no-vig
    def american_to_prob(price: int) -> float:
        p = abs(price) / (abs(price) + 100) if price < 0 else 100 / (price + 100)
        return p

    merged["over_prob_raw"]  = merged["over_price"].apply(american_to_prob)
    merged["under_prob_raw"] = merged["under_price"].apply(american_to_prob)
    total = merged["over_prob_raw"] + merged["under_prob_raw"]
    merged["book_over_prob"] = merged["over_prob_raw"] / total  # no-vig

    # Consensus line and implied prob across books
    consensus = (
        merged.groupby("player_norm")
        .agg(
            consensus_point  = ("line", "median"),
            consensus_prob   = ("book_over_prob", "mean"),
            n_books          = ("bookmaker", "nunique"),
        )
        .reset_index()
    )
    merged = merged.merge(consensus, on="player_norm", how="left")

    return merged


# ── Spine join ────────────────────────────────────────────────────────────────

def get_latest_features(spine: pd.DataFrame) -> pd.DataFrame:
    """
    For each player, return the rolling features from their most recent game row.
    These are valid as inputs for the upcoming game (rolling features are lag-1).
    Spine may use player_name_norm or player_norm — normalize to player_norm.
    """
    if "player_name_norm" in spine.columns and "player_norm" not in spine.columns:
        spine = spine.rename(columns={"player_name_norm": "player_norm"})
    latest = (
        spine.sort_values(["season", "week"])
             .groupby("player_norm", as_index=False)
             .last()
    )
    return latest


def join_spine_features(props: pd.DataFrame, spine_latest: pd.DataFrame) -> pd.DataFrame:
    """
    Join prop rows to latest spine features on player_norm.
    Unmatched players get NaN rolling features (filled with 0 below).
    """
    feature_cols = [
        "player_norm", "position", "pos_RB", "pos_QB", "is_home", "game_total",
        "games_played", "is_playoff",
        "carry_rate_L1", "carry_rate_L3", "carry_rate_L5", "carry_rate_L8",
        "carry_rate_L16", "carry_rate_Lcareer",
        "rush_yards_L1", "rush_yards_L3", "rush_yards_L5", "rush_yards_L8",
        "rush_yards_L16", "rush_yards_Lcareer",
        "over_rate_L3", "over_rate_L5", "over_rate_L8", "over_rate_L16",
        "over_rate_Lcareer",
        "opp_carry_allowed_L8", "opp_carry_allowed_L16", "opp_carry_allowed_Lcareer",
    ]
    feature_cols = [c for c in feature_cols if c in spine_latest.columns]
    df = props.merge(spine_latest[feature_cols], on="player_norm", how="left")

    n_unmatched = df["carry_rate_L8"].isna().sum()
    if n_unmatched:
        print(f"  WARN: {n_unmatched} player-book rows unmatched in spine "
              f"— rolling features set to 0 (first-time players)")

    rolling_cols = [c for c in df.columns if c.startswith("carry_rate_") or
                    c.startswith("rush_yards_") or c.startswith("opp_carry_")]
    for col in rolling_cols:
        df[col] = df[col].fillna(0)
    for col in [c for c in df.columns if c.startswith("over_rate_")]:
        df[col] = df[col].fillna(0.5)
    for col in ["pos_RB", "pos_QB", "is_home", "games_played", "game_total", "is_playoff"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


# ── Scoring ──────────────────────────────────────────────────────────────────

def score(df: pd.DataFrame, model_artifact: dict, cdfs: dict) -> pd.DataFrame:
    """Score each player-book row. Returns df with p_model, p_market, edge."""
    df = engineer_features(df)

    model    = model_artifact["model"]
    scaler   = model_artifact["scaler"]
    features = model_artifact["features"]

    X = df[features].values
    if scaler is not None:
        X = scaler.transform(X)
    df["predicted_carries"] = model.predict(X)

    df["pred_bucket"] = pd.cut(
        df["predicted_carries"], bins=PRED_BINS, labels=PRED_LABELS, right=False
    ).astype(str)

    df["shortfall"] = df["line"] - df["predicted_carries"]

    p_model_vals = np.empty(len(df))
    for bucket in PRED_LABELS:
        mask = df["pred_bucket"] == bucket
        if mask.sum() == 0:
            continue
        p_model_vals[mask] = np.array([
            p_over(s, bucket, cdfs)
            for s in df.loc[mask, "shortfall"].values
        ])
    df["p_model"]  = p_model_vals
    df["p_market"] = df["book_over_prob"]
    df["edge"]     = df["p_model"] - df["p_market"]

    return df


# ── Bet filter ────────────────────────────────────────────────────────────────

def filter_bets(df: pd.DataFrame) -> pd.DataFrame:
    """Apply production strategy: QB, line >= 6.5, -edge >= 0.03."""
    mask = (
        (df["position"] == PROD_POSITION) &
        (df["line"] >= PROD_MIN_LINE) &
        ((-df["edge"]) >= PROD_EDGE_THRESH)
    )
    bets = df[mask].copy()
    bets["direction"] = "UNDER"
    bets["offered_price"] = bets["under_price"]
    return bets.sort_values("edge", ascending=True)  # most negative edge first


# ── Email ────────────────────────────────────────────────────────────────────

def _odds_badge(price: int) -> str:
    color = "#065f46" if price > 0 else "#1e40af"
    return f'<span style="background:#dbeafe;color:{color};padding:1px 6px;border-radius:3px;font-size:11px;font-weight:700">{price:+d}</span>'


def build_bet_html(gameday: str, bets: pd.DataFrame, n_qb_high: int) -> str:
    if bets.empty:
        body = f'<p style="color:#6b7280;font-size:13px">No qualifying bets found for {gameday}.</p>'
    else:
        rows_html = ""
        for _, r in bets.iterrows():
            edge_pct = abs(float(r["edge"])) * 100
            mkt_pct  = float(r["p_market"]) * 100
            mdl_pct  = float(r["p_model"]) * 100
            rows_html += f"""
<tr style="border-bottom:1px solid #f3f4f6">
  <td style="padding:8px 12px;font-weight:600">{html_module.escape(str(r['player_display_name']))}</td>
  <td style="padding:8px 12px;text-align:center;font-family:{_MONO}">{r['line']:.1f}</td>
  <td style="padding:8px 12px;text-align:center;font-family:{_MONO}">{r['predicted_carries']:.1f}</td>
  <td style="padding:8px 12px;text-align:center">
    <span style="background:#dbeafe;color:#1e40af;padding:1px 6px;border-radius:3px;font-size:11px;font-weight:700">UNDER</span>
  </td>
  <td style="padding:8px 12px;text-align:center;font-family:{_MONO}">{_odds_badge(int(r['under_price']))}</td>
  <td style="padding:8px 12px;text-align:center;font-family:{_MONO}">{mkt_pct:.1f}%</td>
  <td style="padding:8px 12px;text-align:center;font-family:{_MONO}">{mdl_pct:.1f}%</td>
  <td style="padding:8px 12px;text-align:center;font-family:{_MONO};color:#991b1b;font-weight:600">−{edge_pct:.1f}pp</td>
  <td style="padding:8px 12px">{html_module.escape(str(r['bookmaker']))}</td>
</tr>"""

        body = f"""
<table style="width:100%;border-collapse:collapse;font-size:13px;margin-bottom:20px">
<thead><tr style="background:#1d2d44;color:#fff">
  <th style="padding:9px 12px;text-align:left">Player</th>
  <th style="padding:9px 12px;text-align:center">Line</th>
  <th style="padding:9px 12px;text-align:center">Model Proj</th>
  <th style="padding:9px 12px;text-align:center">Side</th>
  <th style="padding:9px 12px;text-align:center">Odds</th>
  <th style="padding:9px 12px;text-align:center">Mkt P(over)</th>
  <th style="padding:9px 12px;text-align:center">Model P(over)</th>
  <th style="padding:9px 12px;text-align:center">Edge</th>
  <th style="padding:9px 12px;text-align:left">Book</th>
</tr></thead>
<tbody>{rows_html}</tbody>
</table>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>NFL Rush Attempts — {gameday}</title></head>
<body style="margin:0;padding:16px;background:#f4f4f5;font-family:{_SANS};font-size:13px;color:#1a1a1a">
<div style="max-width:700px;margin:0 auto;background:#fff;padding:24px;border-radius:8px;border:1px solid #e2e2e4">
  <h2 style="font-size:18px;margin:0 0 4px">NFL Rush Attempts — QB Under Bets</h2>
  <p style="color:#6b7280;font-size:12px;margin:0 0 4px">
    Gameday: {gameday} &nbsp;|&nbsp; Generated {datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')}
  </p>
  <p style="color:#6b7280;font-size:12px;margin:0 0 20px">
    Strategy: QB + line ≥ 6.5 + UNDER + edge ≥ 3pp &nbsp;|&nbsp;
    QB props (line ≥ 6.5) scanned: {n_qb_high} &nbsp;|&nbsp; Qualifying bets: {len(bets)}
  </p>
  {body}
</div>
</body>
</html>"""


def build_bet_text(gameday: str, bets: pd.DataFrame) -> str:
    if bets.empty:
        return f"NFL Rush Attempts — {gameday}\n\nNo qualifying bets.\n"
    lines = [f"NFL Rush Attempts QB Unders — {gameday}\n"]
    lines.append(f"{'Player':<25} {'Line':>5} {'Proj':>6} {'Odds':>6} {'Edge':>7} {'Book'}")
    lines.append("-" * 70)
    for _, r in bets.iterrows():
        edge_pct = abs(float(r["edge"])) * 100
        lines.append(
            f"{str(r['player_display_name']):<25} {r['line']:>5.1f} "
            f"{r['predicted_carries']:>6.1f} {int(r['under_price']):>+6d} "
            f"−{edge_pct:>5.1f}pp {r['bookmaker']}"
        )
    return "\n".join(lines)


def send_email(subject: str, html_body: str, text_body: str) -> None:
    to_list = [a.strip() for a in SES_TO_RAW.split(",") if a.strip()]
    if SES_SOURCE and to_list:
        try:
            boto3.client("ses", region_name="us-east-2").send_email(
                Source=SES_SOURCE,
                Destination={"ToAddresses": to_list},
                Message={
                    "Subject": {"Data": subject, "Charset": "UTF-8"},
                    "Body": {
                        "Html": {"Data": html_body,  "Charset": "UTF-8"},
                        "Text": {"Data": text_body,  "Charset": "UTF-8"},
                    },
                },
            )
            print(f"  SES email sent: {subject}")
        except Exception as e:
            print(f"  SES send failed: {e}")
    if ENABLE_SNS and SNS_TOPIC_ARN:
        boto3.client("sns").publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject=subject[:100],
            Message=text_body[:256_000],
        )
        print(f"  SNS published: {subject}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gameday", type=str, default=None)
    args    = parser.parse_args()
    gameday = args.gameday or today_et()

    if not ODDS_API_KEY:
        sys.exit("ODDS_API_KEY not set")

    print(f"\nNFL Rush Attempts Pipeline — {gameday}")
    print("=" * 60)

    # ── 1. Load model artifacts ────────────────────────────────────────────
    print("Loading model artifacts from S3...")
    model_artifact = load_s3_pkl(f"{S3_PREFIX}/artifacts/best_model.pkl")
    cdfs           = load_s3_pkl(f"{S3_PREFIX}/artifacts/residual_cdfs.pkl")
    print(f"  Model: {model_artifact['model_type']}  "
          f"features: {model_artifact['features']}")

    # ── 2. Load spine ──────────────────────────────────────────────────────
    print("Loading spine from S3...")
    spine = load_s3_parquet(f"{S3_PREFIX}/spine/nfl_rush_attempts_spine.parquet")
    if spine is None or spine.empty:
        sys.exit("Spine not found in S3 — run update_spine.py first")
    spine_latest = get_latest_features(spine)
    print(f"  Spine: {len(spine_latest):,} unique players")

    # ── 3. Fetch props ─────────────────────────────────────────────────────
    print(f"Fetching props for {gameday}...")
    events    = fetch_events(gameday)
    if not events:
        print("No games today. Nothing to score.")
        subject   = f"NFL Rush Attempts — {gameday} — No games"
        html_body = build_bet_html(gameday, pd.DataFrame(), 0)
        text_body = f"NFL Rush Attempts — {gameday}\n\nNo games scheduled today.\n"
        send_email(subject, html_body, text_body)
        return

    event_ids = [e["id"] for e in events]
    raw_props = fetch_rush_attempt_props(event_ids)
    props     = parse_props(raw_props, gameday)

    if props.empty:
        print("No rush attempts props available yet.")
        subject   = f"NFL Rush Attempts — {gameday} — No props available"
        html_body = build_bet_html(gameday, pd.DataFrame(), 0)
        text_body = f"NFL Rush Attempts — {gameday}\n\nNo props posted yet.\n"
        send_email(subject, html_body, text_body)
        return

    print(f"  Props: {len(props):,} player-book rows "
          f"| {props['player_norm'].nunique():,} unique players")

    # ── 4. Join spine features ─────────────────────────────────────────────
    print("Joining spine features...")
    scored_df = join_spine_features(props, spine_latest)

    # ── 5. Score ───────────────────────────────────────────────────────────
    print("Scoring...")
    scored_df = score(scored_df, model_artifact, cdfs)

    # ── 6. Filter to production strategy ──────────────────────────────────
    qb_high = scored_df[
        (scored_df["position"] == PROD_POSITION) &
        (scored_df["line"] >= PROD_MIN_LINE)
    ]
    print(f"  QB props (line ≥ 6.5): {len(qb_high):,} player-book rows "
          f"| {qb_high['player_norm'].nunique():,} unique QBs")

    bets = filter_bets(scored_df)
    print(f"  Qualifying bets: {len(bets)}")

    if not bets.empty:
        for _, r in bets.iterrows():
            print(f"    ✓ {r['player_display_name']:<25} line={r['line']:.1f}  "
                  f"proj={r['predicted_carries']:.1f}  "
                  f"edge={r['edge']:+.3f}  odds={int(r['under_price']):+d}  "
                  f"book={r['bookmaker']}")

    # ── 7. Save recommendations ────────────────────────────────────────────
    save_cols = [
        "gameday", "player_display_name", "player_norm", "bookmaker",
        "line", "predicted_carries", "direction", "offered_price",
        "under_price", "over_price", "book_over_prob",
        "p_model", "p_market", "edge",
        "consensus_point", "n_books", "position",
    ]
    save_cols = [c for c in save_cols if c in bets.columns]
    if not bets.empty:
        rec_key = f"{S3_PREFIX}/daily_runs/{gameday}/recommendations.csv"
        save_s3_csv(bets[save_cols], rec_key)
    else:
        print("  No bets to save.")

    # ── 8. Email ───────────────────────────────────────────────────────────
    n_qb_high = len(qb_high)
    if bets.empty:
        subject = f"NFL Rush Attempts — {gameday} — No qualifying bets ({n_qb_high} QB props scanned)"
    else:
        subject = (f"NFL Rush Attempts — {gameday} — "
                   f"{len(bets)} QB UNDER bet{'s' if len(bets) > 1 else ''}")

    html_body = build_bet_html(gameday, bets, n_qb_high)
    text_body = build_bet_text(gameday, bets)
    send_email(subject, html_body, text_body)

    print(f"\n{'='*60}")
    print(f"  Pipeline complete — {gameday}")
    print(f"  Qualifying bets: {len(bets)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
