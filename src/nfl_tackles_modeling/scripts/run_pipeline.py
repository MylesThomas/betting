"""
Live gameday pipeline for NFL tackles props.

For each player with a player_tackles_assists prop on the given gameday:
  1. Fetches live event IDs, tackle props, and game lines from The Odds API
  2. Joins with rolling features from the spine (downloaded from S3)
  3. Scores with OLS + NegBin hybrid model (downloaded from S3)
  4. Filters to UNDER bets: edge >= 5pp, lines 4.5-9.5, min 1 book
  5. Checks cold streaks from settled history (flags players at streak <= -3)
  6. Sends SES email + SNS notification with bet sheet
  7. Saves recommendations CSV to S3

S3 paths read:
  s3://the-odds-api-mt/nfl/tackles_model/spine/nfl_tackles_historical_spine.parquet
  s3://the-odds-api-mt/nfl/tackles_model/artifacts/{ols_pipeline.joblib, nb_*.npy, residuals.npy}
  s3://the-odds-api-mt/nfl/tackles_model/settled/settled_bets.parquet  (optional)

S3 paths written:
  s3://the-odds-api-mt/nfl/tackles_model/daily_runs/{gameday}/recommendations.csv

Run:
  python src/nfl_tackles_modeling/scripts/run_pipeline.py --gameday 2026-09-11
  python src/nfl_tackles_modeling/scripts/run_pipeline.py  # defaults to today ET
"""

import argparse
import html as html_module
import os
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
import joblib
import numpy as np
import pandas as pd
import requests
from scipy.stats import nbinom

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

ODDS_API_KEY  = os.environ.get("ODDS_API_KEY", "").strip()
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT         = "americanfootball_nfl"
MARKET        = "player_tackles_assists"
BOOKMAKERS    = "draftkings,espnbet,betmgm,hardrockbet,fliff,betonlineag,bovada,betrivers,ballybet,betparx,williamhill_us,fanatics,fanduel"
REGIONS       = "us"
SLEEP_S       = 0.25

S3_BUCKET      = "the-odds-api-mt"
S3_PREFIX      = "nfl/tackles_model"
SNS_TOPIC_ARN  = os.environ.get("SNS_TOPIC_ARN", "").strip()
ENABLE_SNS     = os.environ.get("ENABLE_SNS", "").strip().lower() in ("1", "true", "yes")
SES_SOURCE     = os.environ.get("SETTLEMENT_SES_SOURCE", "").strip()
SES_TO_RAW     = os.environ.get("SETTLEMENT_SES_TO", "").strip()

ET  = ZoneInfo("America/New_York")

# ── Production config (mirrors infer.py) ──────────────────────────────────────
DIRECTION      = "UNDER"
EDGE_THRESHOLD = 0.05
LINE_MIN       = 4.5
LINE_MAX       = 9.5
MIN_BOOKS      = 1
COLD_STREAK_THRESHOLD = -3

# ── Model constants ───────────────────────────────────────────────────────────
HYBRID_NEGBIN_THRESHOLD = 4.5
N_BOOT = 10_000
RNG    = np.random.default_rng(42)

POS_GROUP_MAP = {
    "LB": "LB", "CB": "CB", "DB": "CB",
    "S": "S",  "FS": "S",  "SS": "S",
    "DE": "DL", "DT": "DL", "DL": "DL", "NT": "DL",
}
DROP_POSITIONS = ["WR", "FB"]

BEST_FEATS = [
    "offered_line", "game_total", "proj_opp_score", "tackle_rate_L16",
    "pos_LB", "pos_CB", "pos_S", "pos_DL", "market_under_prob",
]

TEAM_NAME_MAP = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN",
    "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE",
    "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}

_SUFFIX_RE  = re.compile(r"\s*,?\s*(Jr\.?|Sr\.?|II{1,2}|IV|V)\.?$", re.IGNORECASE)
_SPECIAL_RE = re.compile(r"['\.\-,]")

_SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif"
_MONO = "ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace"


# ── Helpers ───────────────────────────────────────────────────────────────────

def today_et() -> str:
    return datetime.now(ET).strftime("%Y-%m-%d")


def current_nfl_season() -> int:
    now = datetime.now(ET)
    return now.year if now.month >= 8 else now.year - 1


def normalize_name(name: str) -> str:
    s = str(name).strip()
    s = _SUFFIX_RE.sub("", s)
    s = _SPECIAL_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def amer_to_imp(price: float) -> float:
    if price < 0:
        return -price / (-price + 100)
    return 100 / (price + 100)


def imp_to_amer(p: float) -> int:
    if p >= 0.5:
        return int(round(-p / (1 - p) * 100))
    return int(round((1 - p) / p * 100))


# ── S3 helpers ────────────────────────────────────────────────────────────────

def s3_get_parquet(key: str) -> pd.DataFrame:
    body = boto3.client("s3").get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
    return pd.read_parquet(BytesIO(body))


def s3_get_bytes(key: str) -> bytes:
    return boto3.client("s3").get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()


def s3_put_csv(key: str, df: pd.DataFrame) -> None:
    boto3.client("s3").put_object(
        Bucket=S3_BUCKET, Key=key, Body=df.to_csv(index=False).encode()
    )


def load_settled_history() -> pd.DataFrame:
    key = f"{S3_PREFIX}/settled/settled_bets.parquet"
    try:
        return s3_get_parquet(key)
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return pd.DataFrame()
        raise


# ── Odds API ──────────────────────────────────────────────────────────────────

def api_get(url: str, params: dict) -> dict:
    params["apiKey"] = ODDS_API_KEY
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    time.sleep(SLEEP_S)
    return r.json()


def fetch_events(gameday: str) -> list[dict]:
    """Return NFL events scheduled on gameday (ET date string)."""
    data = api_get(
        f"{ODDS_API_BASE}/sports/{SPORT}/events",
        {"commenceTimeFrom": f"{gameday}T00:00:00Z",
         "commenceTimeTo":   f"{gameday}T23:59:59Z"},
    )
    events = [e for e in data if e.get("sport_key") == SPORT]
    print(f"  Events on {gameday}: {len(events)}")
    return events


def fetch_game_lines(events: list[dict]) -> pd.DataFrame:
    """Fetch totals + spreads for each event to get game_total + proj_opp_score."""
    rows = []
    for ev in events:
        eid  = ev["id"]
        home = TEAM_NAME_MAP.get(ev.get("home_team", ""), ev.get("home_team", ""))
        away = TEAM_NAME_MAP.get(ev.get("away_team", ""), ev.get("away_team", ""))
        try:
            data = api_get(
                f"{ODDS_API_BASE}/sports/{SPORT}/events/{eid}/odds",
                {"markets": "totals,spreads", "regions": REGIONS,
                 "bookmakers": BOOKMAKERS, "oddsFormat": "american"},
            )
        except Exception:
            continue
        totals = []
        spreads_home = []
        for book in data.get("bookmakers", []):
            for mkt in book.get("markets", []):
                if mkt["key"] == "totals":
                    for oc in mkt.get("outcomes", []):
                        if oc["name"] == "Over":
                            totals.append(oc["point"])
                elif mkt["key"] == "spreads":
                    for oc in mkt.get("outcomes", []):
                        if oc["name"] == ev.get("home_team", ""):
                            spreads_home.append(oc["point"])
        if not totals:
            continue
        game_total   = float(np.median(totals))
        spread_home  = float(np.median(spreads_home)) if spreads_home else 0.0
        # proj_opp_score from each team's defensive perspective
        rows.append({"event_id": eid, "team": home, "opponent": away,
                     "game_total": game_total,
                     "proj_opp_score": (game_total - spread_home) / 2})
        rows.append({"event_id": eid, "team": away, "opponent": home,
                     "game_total": game_total,
                     "proj_opp_score": (game_total + spread_home) / 2})
    return pd.DataFrame(rows)


def fetch_props(events: list[dict]) -> pd.DataFrame:
    """Fetch player_tackles_assists props from DK + FD for all events."""
    records = []
    for ev in events:
        eid  = ev["id"]
        home = TEAM_NAME_MAP.get(ev.get("home_team", ""), ev.get("home_team", ""))
        away = TEAM_NAME_MAP.get(ev.get("away_team", ""), ev.get("away_team", ""))
        try:
            data = api_get(
                f"{ODDS_API_BASE}/sports/{SPORT}/events/{eid}/odds",
                {"markets": MARKET, "regions": REGIONS,
                 "bookmakers": BOOKMAKERS, "oddsFormat": "american"},
            )
        except Exception as e:
            print(f"    Event {eid}: {e}")
            continue
        for book in data.get("bookmakers", []):
            book_key = book["key"]
            for mkt in book.get("markets", []):
                if mkt["key"] != MARKET:
                    continue
                for oc in mkt.get("outcomes", []):
                    player = oc.get("name", "")
                    desc   = oc.get("description", "")   # "Over" / "Under"
                    point  = oc.get("point")
                    price  = oc.get("price")
                    if not player or point is None or price is None:
                        continue
                    # Infer team from home/away roster — unknown, fill later from spine
                    records.append({
                        "event_id":    eid,
                        "home_team":   home,
                        "away_team":   away,
                        "bookmaker":   book_key,
                        "player_name": player,
                        "player_norm": normalize_name(player),
                        "side":        desc,
                        "point":       float(point),
                        "price":       float(price),
                    })
    return pd.DataFrame(records)


# ── Feature engineering ───────────────────────────────────────────────────────

def build_per_book_rows(props: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (player, book) — each row scores that book's specific line.
    This avoids blending prices across different lines into a synthetic consensus.
    """
    rows = []
    for player_norm, grp in props.groupby("player_norm"):
        sample = grp.iloc[0]
        for book_key, bgrp in grp.groupby("bookmaker"):
            book_over  = bgrp[bgrp["side"] == "Over"]
            book_under = bgrp[bgrp["side"] == "Under"]
            if book_over.empty or book_under.empty:
                continue
            point       = float(book_over.iloc[0]["point"])
            over_price  = float(book_over.iloc[0]["price"])
            under_price = float(book_under.iloc[0]["price"])
            op  = amer_to_imp(over_price)
            up  = amer_to_imp(under_price)
            total = op + up
            if total <= 0:
                continue
            rows.append({
                "player_name":           sample["player_name"],
                "player_norm":           player_norm,
                "event_id":              sample["event_id"],
                "home_team":             sample["home_team"],
                "away_team":             sample["away_team"],
                "book":                  book_key,
                "offered_line":          point,
                "market_under_prob":     up / total,
                "market_over_prob":      op / total,
                "consensus_under_price": int(under_price),
                "consensus_over_price":  int(over_price),
            })

    # n_books = total books with a two-sided market for this player-game
    per_book = pd.DataFrame(rows)
    if not per_book.empty:
        n_books = per_book.groupby("player_norm")["book"].transform("count")
        per_book["n_books"] = n_books
    return per_book


def add_spine_features(consensus: pd.DataFrame, spine: pd.DataFrame) -> pd.DataFrame:
    """Join latest rolling features + position from spine onto consensus rows."""
    spine_clean = spine[
        spine["position"].notna() &
        ~spine["position"].isin(DROP_POSITIONS)
    ].copy()

    # Latest row per player (most recent completed game → freshest rolling features)
    latest = (
        spine_clean
        .sort_values(["season", "week"])
        .groupby("player_name_norm", as_index=False)
        .last()
    )

    feat_cols = ["player_name_norm", "position", "team",
                 "tackle_rate_L16", "defense_pct"] + \
                [f"tackle_rate_L{w}" for w in [3, 5, 8] if f"tackle_rate_L{w}" in latest.columns]
    latest = latest[[c for c in feat_cols if c in latest.columns]]

    merged = consensus.merge(latest, left_on="player_norm", right_on="player_name_norm", how="left")

    # Position dummies
    merged["position_group"] = merged["position"].map(POS_GROUP_MAP)
    for g in ["LB", "CB", "S", "DL"]:
        merged[f"pos_{g}"] = (merged["position_group"] == g).astype(int)

    return merged


def add_game_context(df: pd.DataFrame, game_lines: pd.DataFrame) -> pd.DataFrame:
    """Join game_total + proj_opp_score from the team's perspective."""
    if game_lines.empty:
        df["game_total"]     = np.nan
        df["proj_opp_score"] = np.nan
        return df

    # Match by team; fall back to event_id + home/away logic if team unknown
    merged = df.merge(
        game_lines[["event_id", "team", "game_total", "proj_opp_score"]],
        left_on=["event_id", "team"],
        right_on=["event_id", "team"],
        how="left",
    )
    return merged


# ── Inference ─────────────────────────────────────────────────────────────────

def load_artifacts() -> dict:
    print(f"  Loading artifacts from s3://{S3_BUCKET}/{S3_PREFIX}/artifacts/...")
    prefix = f"{S3_PREFIX}/artifacts"
    ols         = joblib.load(BytesIO(s3_get_bytes(f"{prefix}/ols_pipeline.joblib")))
    residuals   = np.load(BytesIO(s3_get_bytes(f"{prefix}/residuals.npy")))
    nb_coefs    = np.load(BytesIO(s3_get_bytes(f"{prefix}/nb_coefs.npy")))
    nb_alpha    = float(np.load(BytesIO(s3_get_bytes(f"{prefix}/nb_alpha.npy")))[0])
    return {"ols": ols, "residuals": residuals, "nb_coefs": nb_coefs, "nb_alpha": nb_alpha}


def run_inference(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    warnings.filterwarnings("ignore")
    ols, residuals = artifacts["ols"], artifacts["residuals"]
    nb_coefs, nb_alpha = artifacts["nb_coefs"], artifacts["nb_alpha"]

    result = df.copy()
    mask   = result[BEST_FEATS].notna().all(axis=1)
    idx    = result.index[mask]

    if idx.empty:
        result["ols_pred"] = np.nan
        result["p_hybrid"] = np.nan
        result["p_market"] = result["market_under_prob"]
        result["edge"]     = np.nan
        result["recommendation"] = "PASS"
        return result

    X    = result.loc[idx, BEST_FEATS].to_numpy(dtype=float)
    line = result.loc[idx, "offered_line"].to_numpy(dtype=float)

    ols_pred = ols.predict(X)
    X_const  = np.column_stack([np.ones(len(X)), X])
    nb_mu    = np.exp(X_const @ nb_coefs)
    mu_c     = np.clip(nb_mu, 1e-3, None)
    n_nb     = 1.0 / nb_alpha
    p_nb     = nbinom.cdf(np.floor(line).astype(int), n=n_nb, p=n_nb / (n_nb + mu_c))
    samp     = RNG.choice(residuals, size=(len(ols_pred), N_BOOT))
    p_bt     = ((ols_pred[:, None] + samp) <= line[:, None]).mean(axis=1)
    p_hyb    = np.where(line < HYBRID_NEGBIN_THRESHOLD, p_bt, p_nb)
    p_mkt    = result.loc[idx, "market_under_prob"].to_numpy(dtype=float)
    edge     = p_hyb - p_mkt
    rec      = np.select(
        [edge > EDGE_THRESHOLD, edge < -EDGE_THRESHOLD],
        ["UNDER", "OVER"], default="PASS",
    )

    result.loc[idx, "ols_pred"]       = np.round(ols_pred, 3)
    result.loc[idx, "p_hybrid"]       = np.round(p_hyb, 4)
    result.loc[idx, "p_market"]       = np.round(p_mkt, 4)
    result.loc[idx, "edge"]           = np.round(edge, 4)
    result.loc[idx, "recommendation"] = rec
    return result


def filter_bets(results: pd.DataFrame) -> pd.DataFrame:
    mask = (
        (results["recommendation"] == DIRECTION) &
        (results["offered_line"] >= LINE_MIN) &
        (results["offered_line"] <= LINE_MAX) &
        (results["edge"].abs() >= EDGE_THRESHOLD) &
        results["ols_pred"].notna() &
        (results["n_books"] >= MIN_BOOKS)
    )
    return results[mask].copy()


# ── Cold streak detection ─────────────────────────────────────────────────────

def compute_player_streak(history: pd.DataFrame, player_norm: str) -> int:
    """Compute current win/loss streak for a player from settled history."""
    ph = history[history["player_norm"] == player_norm].sort_values(["season", "week"])
    if ph.empty:
        return 0
    hits = ph["hit"].tolist()
    if not hits:
        return 0
    streak = 1 if hits[-1] else -1
    for h in reversed(hits[:-1]):
        if bool(h) == bool(hits[-1]):
            streak += (1 if hits[-1] else -1)
        else:
            break
    return streak


def check_cold_streaks(bets: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Add streak column to bets; flag COLD_STREAK if streak <= threshold."""
    if history.empty:
        bets["streak"] = 0
        bets["cold_streak_warning"] = False
        return bets

    streaks = {
        pn: compute_player_streak(history, pn)
        for pn in bets["player_norm"].unique()
    }
    bets = bets.copy()
    bets["streak"] = bets["player_norm"].map(streaks).fillna(0).astype(int)
    bets["cold_streak_warning"] = bets["streak"] <= COLD_STREAK_THRESHOLD
    return bets


# ── HTML email ────────────────────────────────────────────────────────────────

def fmt_edge(edge: float) -> str:
    return f"+{abs(edge)*100:.1f}pp"


def fmt_odds(price) -> str:
    if pd.isna(price):
        return "—"
    return f"{int(price):+d}"


def build_recommendations_html(bets: pd.DataFrame, gameday: str,
                                n_players_scored: int) -> str:
    bets = bets.sort_values("edge")   # most negative edge = biggest UNDER advantage

    cold_players = bets[bets["cold_streak_warning"]]

    rows_html = ""
    for _, r in bets.iterrows():
        cold = r.get("cold_streak_warning", False)
        streak = int(r.get("streak", 0))
        streak_badge = (
            f' <span style="background:#fef3c7;color:#92400e;padding:1px 6px;'
            f'border-radius:3px;font-size:10px;font-weight:700">⚠ streak {streak}</span>'
            if cold else ""
        )
        edge_pp  = abs(r["edge"]) * 100
        edge_col = "#065f46" if edge_pp >= 10 else "#1d6fa4"
        rows_html += f"""
<tr style="border-bottom:1px solid #f3f4f6">
  <td style="padding:8px 12px;font-weight:600">{html_module.escape(str(r['player_name']))}{streak_badge}</td>
  <td style="padding:8px 12px;color:#6b7280">{html_module.escape(str(r.get('team','—')))}</td>
  <td style="padding:8px 12px;color:#6b7280">{html_module.escape(str(r.get('position','—')))}</td>
  <td style="padding:8px 12px;text-align:center;font-weight:600">{r['offered_line']:.1f}</td>
  <td style="padding:8px 12px;text-align:center;font-family:{_MONO}">{r['p_hybrid']*100:.1f}%</td>
  <td style="padding:8px 12px;text-align:center;font-family:{_MONO}">{r['p_market']*100:.1f}%</td>
  <td style="padding:8px 12px;text-align:center;font-weight:600;color:{edge_col}">{edge_pp:.1f}pp</td>
  <td style="padding:8px 12px;text-align:center;font-family:{_MONO}">{fmt_odds(r.get('consensus_under_price'))}</td>
  <td style="padding:8px 12px;text-align:center;color:#6b7280">{html_module.escape(str(r.get('book','—')))}</td>
</tr>"""

    no_bets_msg = "" if len(bets) else (
        '<p style="color:#6b7280">No qualifying UNDER bets today '
        f'(edge &lt; {EDGE_THRESHOLD*100:.0f}pp or no lines in {LINE_MIN}–{LINE_MAX} range).</p>'
    )

    table = "" if not len(bets) else f"""
<table style="width:100%;border-collapse:collapse;font-size:13px">
<thead><tr style="background:#1d2d44;color:#fff">
  <th style="padding:9px 12px;text-align:left">Player</th>
  <th style="padding:9px 12px;text-align:left">Team</th>
  <th style="padding:9px 12px;text-align:left">Pos</th>
  <th style="padding:9px 12px;text-align:center">Line</th>
  <th style="padding:9px 12px;text-align:center">Model P(Under)</th>
  <th style="padding:9px 12px;text-align:center">Mkt P(Under)</th>
  <th style="padding:9px 12px;text-align:center">Edge</th>
  <th style="padding:9px 12px;text-align:center">Under Odds</th>
  <th style="padding:9px 12px;text-align:center">Book</th>
</tr></thead>
<tbody>{rows_html}</tbody>
</table>"""

    cold_section = ""
    if not cold_players.empty:
        cold_rows = ""
        for _, r in cold_players.iterrows():
            cold_rows += (
                f'<li style="margin-bottom:4px">'
                f'<strong>{html_module.escape(str(r["player_name"]))}</strong> — '
                f'streak {int(r["streak"])} · line {r["offered_line"]:.1f} · '
                f'edge {abs(r["edge"])*100:.1f}pp · '
                f'consider reviewing before betting'
                f'</li>'
            )
        cold_section = f"""
<div style="background:#fff8e6;border:1px solid #f0c040;border-radius:6px;
            padding:14px 18px;margin-bottom:20px">
  <div style="font-weight:600;font-size:13px;color:#92400e;margin-bottom:8px">
    ⚠ Cold Streak Alert — {len(cold_players)} player{'s' if len(cold_players) > 1 else ''}
  </div>
  <p style="font-size:12px;color:#78350f;margin:0 0 8px">
    These players have gone Over (bet lost) {abs(COLD_STREAK_THRESHOLD)}+ times in a row.
    IS hit rate at this streak depth is ~52% (near breakeven). Review before betting.
  </p>
  <ul style="font-size:12px;color:#78350f;margin:0;padding-left:18px">
    {cold_rows}
  </ul>
</div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>NFL Tackles — {gameday}</title></head>
<body style="margin:0;padding:16px;background:#f4f4f5;font-family:{_SANS};font-size:13px;color:#1a1a1a">
<div style="max-width:800px;margin:0 auto;background:#fff;padding:24px;border-radius:8px;border:1px solid #e2e2e4">
  <h2 style="font-size:18px;margin:0 0 4px">NFL Tackles — {gameday}</h2>
  <p style="color:#6b7280;font-size:12px;margin:0 0 6px">
    Generated {datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')}
    &nbsp;|&nbsp; {n_players_scored} players scored
    &nbsp;|&nbsp; <strong>{len(bets)} qualifying bet{'s' if len(bets) != 1 else ''}</strong>
  </p>
  <p style="color:#9ca3af;font-size:11px;margin:0 0 20px">
    Strategy: UNDER · edge ≥ {EDGE_THRESHOLD*100:.0f}pp · lines {LINE_MIN}–{LINE_MAX} · min {MIN_BOOKS} book
  </p>
  {cold_section}
  {no_bets_msg}{table}
  <p style="color:#9ca3af;font-size:11px;margin-top:20px">
    In-sample hit rate: 57.7% (2024–2025, 2,370 bets) — OOS validation pending first live season.
  </p>
</div>
</body>
</html>"""


# ── SES / SNS send ────────────────────────────────────────────────────────────

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
                        "Html": {"Data": html_body, "Charset": "UTF-8"},
                        "Text": {"Data": text_body, "Charset": "UTF-8"},
                    },
                },
            )
            print(f"  SES sent: {subject}")
        except Exception as e:
            print(f"  SES failed: {e}")
    if ENABLE_SNS and SNS_TOPIC_ARN:
        try:
            boto3.client("sns").publish(
                TopicArn=SNS_TOPIC_ARN,
                Subject=subject[:100],
                Message=text_body[:256_000],
            )
            print(f"  SNS published")
        except Exception as e:
            print(f"  SNS failed: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gameday", type=str, default=None,
                        help="Gameday to run (YYYY-MM-DD, default: today ET)")
    args    = parser.parse_args()
    gameday = args.gameday or today_et()
    season  = current_nfl_season()

    print(f"\nNFL Tackles Pipeline — gameday={gameday}  season={season}")
    print("=" * 60)

    if not ODDS_API_KEY:
        sys.exit("ODDS_API_KEY not set")

    # ── 1. Fetch events + props ────────────────────────────────────────────────
    print("\n  Fetching events...")
    events = fetch_events(gameday)
    if not events:
        msg = f"No NFL events found for {gameday}."
        print(f"  {msg}")
        send_email(f"NFL Tackles — {gameday} — No games", f"<p>{msg}</p>", msg)
        return

    print("\n  Fetching game lines...")
    game_lines = fetch_game_lines(events)
    print(f"    {len(game_lines)} team-game rows")

    print(f"\n  Fetching {MARKET} props ({BOOKMAKERS})...")
    props_raw = fetch_props(events)
    if props_raw.empty:
        msg = f"No {MARKET} props available for {gameday}."
        print(f"  {msg}")
        send_email(f"NFL Tackles — {gameday} — No props", f"<p>{msg}</p>", msg)
        return
    print(f"    {len(props_raw)} prop rows  ({props_raw['player_norm'].nunique()} players)")

    # ── 2. Build per-book rows ─────────────────────────────────────────────────
    print("\n  Building per-book rows...")
    per_book = build_per_book_rows(props_raw)
    print(f"    {per_book['player_norm'].nunique()} players  ×  {len(per_book)} player-book rows")

    # ── 3. Load spine + join features ─────────────────────────────────────────
    print(f"\n  Loading spine from S3...")
    spine = s3_get_parquet(f"{S3_PREFIX}/spine/nfl_tackles_historical_spine.parquet")
    print(f"    Spine: {len(spine):,} rows")

    df = add_spine_features(per_book, spine)
    df = add_game_context(df, game_lines)

    n_with_feats = df[BEST_FEATS].notna().all(axis=1).sum()
    print(f"    {n_with_feats}/{len(df)} players have all required features")

    # ── 4. Inference ───────────────────────────────────────────────────────────
    print("\n  Loading model artifacts...")
    artifacts = load_artifacts()

    print("  Running inference...")
    results = run_inference(df, artifacts)

    bets = filter_bets(results)
    print(f"  Qualifying UNDER bets: {len(bets)}")

    # ── 5. Cold streak check ───────────────────────────────────────────────────
    print("\n  Loading settled history for streak check...")
    history = load_settled_history()
    if not history.empty:
        print(f"    Settled history: {len(history):,} rows")
        bets = check_cold_streaks(bets, history)
        n_cold = bets["cold_streak_warning"].sum()
        if n_cold:
            print(f"    ⚠ Cold streak alerts: {n_cold} player(s)")
    else:
        print("    No settled history yet (first run)")
        bets["streak"] = 0
        bets["cold_streak_warning"] = False

    # ── 6. Save recommendations to S3 ─────────────────────────────────────────
    save_cols = [
        "player_name", "player_norm", "team", "position", "event_id",
        "book", "offered_line", "p_hybrid", "p_market", "edge",
        "consensus_under_price", "consensus_over_price", "n_books", "streak", "cold_streak_warning",
    ]
    rec_df = bets[[c for c in save_cols if c in bets.columns]].copy()
    rec_df["gameday"] = gameday
    rec_df["season"]  = season
    rec_key = f"{S3_PREFIX}/daily_runs/{gameday}/recommendations.csv"
    s3_put_csv(rec_key, rec_df)
    print(f"\n  Recommendations saved → s3://{S3_BUCKET}/{rec_key}")

    # ── 7. Email ───────────────────────────────────────────────────────────────
    html_body = build_recommendations_html(bets, gameday, n_with_feats)
    n_cold    = int(bets["cold_streak_warning"].sum()) if "cold_streak_warning" in bets.columns else 0
    subject   = (
        f"NFL Tackles — {gameday} — {len(bets)} bet{'s' if len(bets) != 1 else ''}"
        + (f" ⚠ {n_cold} cold streak" if n_cold else "")
    )
    text_body = (
        f"NFL Tackles — {gameday}\n\n"
        f"Qualifying UNDER bets: {len(bets)}\n"
        + "\n".join(
            f"  {r['player_name']} | {r.get('book','?')} | line {r['offered_line']:.1f} | "
            f"UNDER {fmt_odds(r.get('consensus_under_price'))} | edge {abs(r['edge'])*100:.1f}pp"
            + (" ⚠COLD" if r.get("cold_streak_warning") else "")
            for _, r in bets.iterrows()
        )
    )
    send_email(subject, html_body, text_body)

    print(f"\n{'='*60}")
    print(f"  Done — {len(bets)} bets for {gameday}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
