"""
Live gameday pipeline for NFL WR/TE receiving yards props.

For each player with a player_reception_yds prop on the given gameday:
  1. Fetches live event IDs, rec yards props, and game lines from The Odds API
  2. Joins with rolling features from the spine (downloaded from S3)
  3. Scores with OLS + NegBin hybrid model (downloaded from S3)
  4. Filters to OVER bets: edge >= 3pp, all lines, min 3 books
  5. Sends SES email + SNS notification with bet sheet
  6. Saves recommendations CSV to S3

S3 paths read:
  s3://the-odds-api-mt/nfl/rec_yards_model/spine/nfl_rec_yards_historical_spine.parquet
  s3://the-odds-api-mt/nfl/rec_yards_model/artifacts/{ols_pipeline.joblib, nb_*.npy, ...}
  s3://the-odds-api-mt/nfl/rec_yards_model/settled/settled_bets.parquet  (optional)

S3 paths written:
  s3://the-odds-api-mt/nfl/rec_yards_model/daily_runs/{gameday}/recommendations.csv

Run:
  python src/nfl_rec_yards_modeling/scripts/run_pipeline.py --gameday 2026-09-11
  python src/nfl_rec_yards_modeling/scripts/run_pipeline.py  # defaults to today ET
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
import yaml
from scipy.stats import nbinom

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

ODDS_API_KEY  = os.environ.get("ODDS_API_KEY", "").strip()
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT         = "americanfootball_nfl"
MARKET        = "player_reception_yds"
BOOKMAKERS    = "draftkings,espnbet,betmgm,hardrockbet,fliff,betonlineag,bovada,betrivers,ballybet,betparx,williamhill_us,fanatics,fanduel"
REGIONS       = "us"
SLEEP_S       = 0.25

S3_BUCKET     = "the-odds-api-mt"
S3_PREFIX     = "nfl/rec_yards_model"
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "").strip()
ENABLE_SNS    = os.environ.get("ENABLE_SNS", "").strip().lower() in ("1", "true", "yes")
SES_SOURCE    = os.environ.get("SETTLEMENT_SES_SOURCE", "").strip()
SES_TO_RAW    = os.environ.get("SETTLEMENT_SES_TO", "").strip()

ET = ZoneInfo("America/New_York")

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "model_config.yaml"
with open(_CONFIG_PATH) as _f:
    _CFG = yaml.safe_load(_f)["nfl_rec_yards_model"]

NUMERIC_FEATURES     = _CFG["features"]["numeric"]
CATEGORICAL_FEATURES = _CFG["features"]["categorical"]
BEST_FEATS           = _CFG["features"]["order"]
assert set(BEST_FEATS) == set(NUMERIC_FEATURES + CATEGORICAL_FEATURES), (
    f"features.order doesn't match numeric+categorical: {BEST_FEATS}"
)

DIRECTION      = _CFG["inference"]["direction"]
EDGE_THRESHOLD = _CFG["inference"]["edge_threshold"]
LINE_MIN       = _CFG["inference"]["line_min"]
LINE_MAX       = _CFG["inference"]["line_max"]
MIN_BOOKS      = _CFG["inference"]["min_books"]

COLD_STREAK_THRESHOLD   = -3
HYBRID_NEGBIN_THRESHOLD = _CFG["negbin_threshold"]
N_BOOT = 10_000
RNG    = np.random.default_rng(42)

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


def today_et() -> str:
    return datetime.now(ET).strftime("%Y-%m-%d")


def current_nfl_season() -> int:
    now = datetime.now(ET)
    return now.year if now.month >= 8 else now.year - 1


def _normalize_name(name: str) -> str:
    s = str(name).strip()
    s = _SUFFIX_RE.sub("", s)
    s = _SPECIAL_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


# ── S3 helpers ────────────────────────────────────────────────────────────────

def s3_get_bytes(key: str) -> bytes:
    return boto3.client("s3").get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()


def s3_get_parquet(key: str) -> pd.DataFrame:
    return pd.read_parquet(BytesIO(s3_get_bytes(key)))


def s3_put_csv(key: str, df: pd.DataFrame) -> None:
    buf = BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    boto3.client("s3").put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue())


def load_settled_history() -> pd.DataFrame:
    try:
        return s3_get_parquet(f"{S3_PREFIX}/settled/settled_bets.parquet")
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return pd.DataFrame()
        raise


# ── Odds API ──────────────────────────────────────────────────────────────────

def _api_get(path: str, params: dict) -> dict:
    resp = requests.get(f"{ODDS_API_BASE}/{path}", params={**params, "apiKey": ODDS_API_KEY})
    resp.raise_for_status()
    return resp.json()


def fetch_events(gameday: str) -> list[dict]:
    try:
        events = _api_get(f"sports/{SPORT}/events", {
            "dateFormat": "iso", "eventIds": "",
        })
        return [
            e for e in events
            if e.get("commence_time", "")[:10] == gameday
        ]
    except Exception as e:
        print(f"  fetch_events error: {e}")
        return []


def fetch_game_lines(events: list[dict]) -> pd.DataFrame:
    rows = []
    for ev in events:
        try:
            data = _api_get(f"sports/{SPORT}/events/{ev['id']}/odds", {
                "regions": "us", "markets": "spreads,totals",
                "oddsFormat": "american", "dateFormat": "iso",
            })
            home  = TEAM_NAME_MAP.get(ev.get("home_team", ""), ev.get("home_team", ""))
            away  = TEAM_NAME_MAP.get(ev.get("away_team", ""), ev.get("away_team", ""))
            total_line = spread_line = None
            for book in data.get("bookmakers", []):
                for mkt in book.get("markets", []):
                    if mkt["key"] == "totals":
                        for o in mkt.get("outcomes", []):
                            if o.get("name") == "Over" and total_line is None:
                                total_line = float(o.get("point", 0))
                    elif mkt["key"] == "spreads":
                        for o in mkt.get("outcomes", []):
                            if o.get("name") == home and spread_line is None:
                                spread_line = float(o.get("point", 0))
                if total_line and spread_line is not None:
                    break
            if total_line:
                for team, spread in [(home, spread_line or 0), (away, -(spread_line or 0))]:
                    rows.append({
                        "event_id": ev["id"],
                        "team": team,
                        "game_total": total_line,
                        "team_spread": spread,
                        "proj_own_score": (total_line + spread) / 2,
                    })
            time.sleep(SLEEP_S)
        except Exception as e:
            print(f"  fetch_game_lines error for {ev['id']}: {e}")
    return pd.DataFrame(rows)


def fetch_props(events: list[dict]) -> pd.DataFrame:
    rows = []
    for ev in events:
        try:
            data = _api_get(f"sports/{SPORT}/events/{ev['id']}/odds", {
                "regions": REGIONS, "markets": MARKET,
                "bookmakers": BOOKMAKERS,
                "oddsFormat": "american", "dateFormat": "iso",
            })
            home_team = TEAM_NAME_MAP.get(ev.get("home_team", ""), ev.get("home_team", ""))
            away_team = TEAM_NAME_MAP.get(ev.get("away_team", ""), ev.get("away_team", ""))
            for book in data.get("bookmakers", []):
                bk = book["key"]
                for mkt in book.get("markets", []):
                    if mkt["key"] != MARKET:
                        continue
                    outcomes = mkt.get("outcomes", [])
                    players: dict[str, dict] = {}
                    for o in outcomes:
                        pname = o.get("description", "")
                        side  = o.get("name", "")
                        if side not in ("Over", "Under"):
                            continue
                        if pname not in players:
                            players[pname] = {}
                        players[pname][side] = {"price": o["price"], "point": o["point"]}
                    for pname, sides in players.items():
                        if "Over" not in sides or "Under" not in sides:
                            continue
                        rows.append({
                            "event_id": ev["id"],
                            "home_team": home_team,
                            "away_team": away_team,
                            "player_name": pname,
                            "player_norm": _normalize_name(pname),
                            "bookmaker": bk,
                            "over_price": sides["Over"]["price"],
                            "under_price": sides["Under"]["price"],
                            "point": sides["Over"]["point"],
                        })
            time.sleep(SLEEP_S)
        except Exception as e:
            print(f"  fetch_props error for {ev['id']}: {e}")
    return pd.DataFrame(rows)


# ── Name map loading ──────────────────────────────────────────────────────────

def load_name_map() -> dict[str, str]:
    map_path = Path("data/nfl/rec_yards_name_map.csv")
    if not map_path.exists():
        return {}
    df = pd.read_csv(map_path)
    return {
        _normalize_name(row["odds_name_raw"]): row["name_norm"]
        for _, row in df.iterrows()
    }


# ── Per-book rows ─────────────────────────────────────────────────────────────

def _amer_to_imp(price: float) -> float:
    return -price / (-price + 100) if price < 0 else 100 / (price + 100)


def build_per_book_rows(props: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (player_name, player_norm, event_id, home, away), grp in props.groupby(
        ["player_name", "player_norm", "event_id", "home_team", "away_team"]
    ):
        book_rows = []
        for _, r in grp.iterrows():
            imp_o = _amer_to_imp(float(r["over_price"]))
            imp_u = _amer_to_imp(float(r["under_price"]))
            total = imp_o + imp_u
            if total <= 0:
                continue
            book_rows.append({
                "book":                  r["bookmaker"],
                "offered_line":          float(r["point"]),
                "over_price":            r["over_price"],
                "under_price":           r["under_price"],
                "market_under_prob":     imp_u / total,
                "market_over_prob":      imp_o / total,
                "consensus_under_price": int(r["under_price"]),
                "consensus_over_price":  int(r["over_price"]),
            })
        if not book_rows:
            continue

        all_points = [br["offered_line"] for br in book_rows]
        for br in book_rows:
            rows.append({
                "player_name":     player_name,
                "player_norm":     player_norm,
                "event_id":        event_id,
                "home_team":       home,
                "away_team":       away,
                "n_books":         len(book_rows),
                "mkt_consensus_point": float(np.median(all_points)),
                **br,
            })
    return pd.DataFrame(rows)


# ── Spine feature join ────────────────────────────────────────────────────────

def add_spine_features(df: pd.DataFrame, spine: pd.DataFrame,
                       name_map: dict[str, str]) -> pd.DataFrame:
    latest = (
        spine.sort_values(["player_id", "season", "week"])
             .groupby("player_id")
             .last()
             .reset_index()
    )

    feat_cols = [
        "player_id", "player_name_norm", "position", "team",
        "receiving_yards_L8", "target_share_L8", "snap_pct_L8",
        "receiving_yards_L3", "receiving_yards_L5", "receiving_yards_L16",
    ]
    latest = latest[[c for c in feat_cols if c in latest.columns]]

    df = df.copy()
    df["player_norm_mapped"] = df["player_norm"].map(lambda n: name_map.get(n, n))

    merged = df.merge(
        latest.rename(columns={"player_name_norm": "player_norm_mapped"}),
        on="player_norm_mapped", how="left",
    )
    # TE dummy
    merged["pos_TE"] = (merged["position"] == "TE").astype(int)
    return merged


def add_game_context(df: pd.DataFrame, game_lines: pd.DataFrame) -> pd.DataFrame:
    if game_lines.empty:
        df["game_total"]     = np.nan
        df["proj_own_score"] = np.nan
        return df

    merged = df.merge(
        game_lines[["event_id", "team", "game_total", "proj_own_score"]],
        left_on=["event_id", "team"], right_on=["event_id", "team"], how="left",
    )
    return merged


# ── Model artifacts ───────────────────────────────────────────────────────────

def load_artifacts() -> dict:
    print(f"  Loading artifacts from s3://{S3_BUCKET}/{S3_PREFIX}/artifacts/...")
    prefix = f"{S3_PREFIX}/artifacts"
    ols       = joblib.load(BytesIO(s3_get_bytes(f"{prefix}/ols_pipeline.joblib")))
    residuals = np.load(BytesIO(s3_get_bytes(f"{prefix}/residuals.npy")))
    nb_coefs  = np.load(BytesIO(s3_get_bytes(f"{prefix}/nb_coefs.npy")))
    nb_alpha  = float(np.load(BytesIO(s3_get_bytes(f"{prefix}/nb_alpha.npy")))[0])
    return {"ols": ols, "residuals": residuals, "nb_coefs": nb_coefs, "nb_alpha": nb_alpha}


def run_inference(df: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    warnings.filterwarnings("ignore")
    ols, residuals   = artifacts["ols"], artifacts["residuals"]
    nb_coefs, nb_alpha = artifacts["nb_coefs"], artifacts["nb_alpha"]

    result = df.copy()
    mask   = result[BEST_FEATS].notna().all(axis=1)
    idx    = result.index[mask]

    if idx.empty:
        result["ols_pred"] = np.nan
        result["p_hybrid"] = np.nan
        result["p_market"] = result.get("market_under_prob", np.nan)
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
    filtered = results[mask].copy()
    # one row per player+line — keep the book with the best over price
    if not filtered.empty:
        filtered = (
            filtered
            .sort_values("consensus_over_price", ascending=False)
            .drop_duplicates(subset=["player_norm", "offered_line"])
            .sort_values("edge")
        )
    return filtered


# ── Cold streak ───────────────────────────────────────────────────────────────

def compute_player_streak(history: pd.DataFrame, player_norm: str) -> int:
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
    if history.empty:
        bets["streak"] = 0
        bets["cold_streak_warning"] = False
        return bets
    streaks = {pn: compute_player_streak(history, pn) for pn in bets["player_norm"].unique()}
    bets = bets.copy()
    bets["streak"] = bets["player_norm"].map(streaks).fillna(0).astype(int)
    bets["cold_streak_warning"] = bets["streak"] <= COLD_STREAK_THRESHOLD
    return bets


# ── Odds bucket helpers ───────────────────────────────────────────────────────

# In-sample characterization benchmarks (from Step 6 research, 3,618 bets)
_IS_BUCKETS = {
    "dog (+odds)": {"pct": "14.4%", "n": 522,  "wr": "35.1%", "roi": "3.35%",  "avg_odds": "+197", "breakeven": "33.7%"},
    "even":        {"pct": "2.5%",  "n": 90,   "wr": "65.6%", "roi": "31.11%", "avg_odds": "+100", "breakeven": "50.0%"},
    "fav (-odds)": {"pct": "83.1%", "n": 3006, "wr": "57.8%", "roi": "4.65%",  "avg_odds": "-124", "breakeven": "55.4%"},
}


def _odds_bucket(over_price) -> str:
    """Classify a bet's over_price into dog / even / fav."""
    try:
        p = float(over_price)
    except (TypeError, ValueError):
        return "unknown"
    if p > 0:
        return "dog (+odds)"
    if p == 100 or p == -100:
        return "even"
    return "fav (-odds)"


# ── HTML email ────────────────────────────────────────────────────────────────

_FEAT_DISPLAY = {
    "offered_line":       ("Line",          lambda v: f"{v:.1f}"),
    "game_total":         ("Game Tot",      lambda v: f"{v:.1f}"),
    "proj_own_score":     ("Proj Score",    lambda v: f"{v:.1f}"),
    "receiving_yards_L8": ("Rec Yds L8",    lambda v: f"{v:.1f}"),
    "target_share_L8":    ("Tgt Shr L8",    lambda v: f"{v*100:.1f}%"),
    "snap_pct_L8":        ("Snap% L8",      lambda v: f"{v*100:.1f}%"),
    "market_under_prob":  ("Mkt P(U)",      lambda v: f"{v*100:.1f}%"),
    "pos_TE":             ("Pos",           lambda v: "TE" if v == 1 else "WR"),
}

_SUSPICIOUS_ZERO = {"receiving_yards_L8", "target_share_L8", "snap_pct_L8"}


def _feat_cell(feat: str, val) -> str:
    is_nan = val is None or (isinstance(val, float) and np.isnan(val))
    is_sus_zero = (not is_nan) and (val == 0) and (feat in _SUSPICIOUS_ZERO)
    bg = "background:#fee2e2;" if (is_nan or is_sus_zero) else ""
    _, fmt = _FEAT_DISPLAY.get(feat, (feat, str))
    display = "NaN" if is_nan else fmt(val)
    return f'<td style="padding:6px 10px;text-align:center;font-family:{_MONO};{bg}">{display}</td>'


def fmt_odds(price) -> str:
    return "—" if pd.isna(price) else f"{int(price):+d}"


def build_recommendations_html(bets: pd.DataFrame, gameday: str,
                                n_players_scored: int,
                                history: pd.DataFrame | None = None) -> str:
    bets = bets.sort_values("edge")

    cold_players = bets[bets.get("cold_streak_warning", pd.Series(False, index=bets.index))]

    rows_html = ""
    for _, r in bets.iterrows():
        cold    = r.get("cold_streak_warning", False)
        streak  = int(r.get("streak", 0))
        badge   = (
            f' <span style="background:#fef3c7;color:#92400e;padding:1px 6px;'
            f'border-radius:3px;font-size:10px;font-weight:700">⚠ streak {streak}</span>'
            if cold else ""
        )
        bkt = _odds_bucket(r.get("consensus_over_price"))
        if bkt == "dog (+odds)":
            bkt_badge = (' <span style="background:#fee2e2;color:#991b1b;padding:1px 6px;'
                         'border-radius:3px;font-size:10px;font-weight:700">dog</span>')
        elif bkt == "fav (-odds)":
            bkt_badge = (' <span style="background:#d1fae5;color:#065f46;padding:1px 6px;'
                         'border-radius:3px;font-size:10px;font-weight:700">fav</span>')
        else:
            bkt_badge = ""
        edge_pp  = abs(r["edge"]) * 100
        edge_col = "#065f46" if edge_pp >= 10 else "#1d6fa4"
        rows_html += f"""
<tr style="border-bottom:1px solid #f3f4f6">
  <td style="padding:8px 12px;font-weight:600">{html_module.escape(str(r['player_name']))}{badge}{bkt_badge}</td>
  <td style="padding:8px 12px;color:#6b7280">{html_module.escape(str(r.get('team','—')))}</td>
  <td style="padding:8px 12px;text-align:center;font-weight:600;color:#1d6fa4">OVER</td>
  <td style="padding:8px 12px;color:#6b7280">{html_module.escape(str(r.get('position','—')))}</td>
  <td style="padding:8px 12px;text-align:center;font-weight:600">{r['offered_line']:.1f}</td>
  <td style="padding:8px 12px;text-align:center;font-family:{_MONO}">{r['p_hybrid']*100:.1f}%</td>
  <td style="padding:8px 12px;text-align:center;font-family:{_MONO}">{r['p_market']*100:.1f}%</td>
  <td style="padding:8px 12px;text-align:center;font-weight:600;color:{edge_col}">{edge_pp:.1f}pp</td>
  <td style="padding:8px 12px;text-align:center;font-family:{_MONO}">{fmt_odds(r.get('consensus_over_price'))}</td>
  <td style="padding:8px 12px;text-align:center;color:#6b7280">{html_module.escape(str(r.get('book','—')))}</td>
</tr>"""

    no_bets_msg = "" if len(bets) else (
        f'<p style="color:#6b7280">No qualifying OVER bets today '
        f'(edge &lt; {EDGE_THRESHOLD*100:.0f}pp or no lines in {LINE_MIN}–{LINE_MAX} range).</p>'
    )

    table = "" if not len(bets) else f"""
<table style="width:100%;border-collapse:collapse;font-size:13px">
<thead><tr style="background:#1d2d44;color:#fff">
  <th style="padding:9px 12px;text-align:left">Player</th>
  <th style="padding:9px 12px;text-align:left">Team</th>
  <th style="padding:9px 12px;text-align:center">Bet Direction</th>
  <th style="padding:9px 12px;text-align:left">Pos</th>
  <th style="padding:9px 12px;text-align:center">Line</th>
  <th style="padding:9px 12px;text-align:center">Model P(U)</th>
  <th style="padding:9px 12px;text-align:center">Mkt P(U)</th>
  <th style="padding:9px 12px;text-align:center">Edge</th>
  <th style="padding:9px 12px;text-align:center">Over Odds</th>
  <th style="padding:9px 12px;text-align:center">Book</th>
</tr></thead>
<tbody>{rows_html}</tbody>
</table>"""

    # ── Model inputs debug table ──────────────────────────────────────────────
    debug_section = ""
    if len(bets):
        feat_headers = "".join(
            f'<th style="padding:7px 10px;text-align:center;white-space:nowrap">'
            f'{_FEAT_DISPLAY.get(f, (f, None))[0]}</th>'
            for f in BEST_FEATS
        )
        debug_rows = ""
        for _, r in bets.iterrows():
            feat_cells = "".join(_feat_cell(f, r.get(f)) for f in BEST_FEATS)
            debug_rows += (
                f'<tr style="border-bottom:1px solid #f3f4f6">'
                f'<td style="padding:6px 10px;font-weight:600;white-space:nowrap">'
                f'{html_module.escape(str(r["player_name"]))}</td>'
                f'{feat_cells}</tr>'
            )
        debug_section = f"""
<details style="margin-top:20px">
  <summary style="cursor:pointer;font-size:12px;font-weight:600;color:#6b7280;
                  padding:8px 0;user-select:none">
    Model inputs ({len(BEST_FEATS)} features) — click to expand
  </summary>
  <div style="overflow-x:auto;margin-top:8px">
  <table style="width:100%;border-collapse:collapse;font-size:12px">
  <thead><tr style="background:#374151;color:#fff">
    <th style="padding:7px 10px;text-align:left">Player</th>
    {feat_headers}
  </tr></thead>
  <tbody>{debug_rows}</tbody>
  </table>
  <p style="font-size:11px;color:#9ca3af;margin:6px 0 0">
    Red cells = NaN or suspicious zero. Numeric features: {", ".join(NUMERIC_FEATURES)}.
    Categorical: {", ".join(CATEGORICAL_FEATURES)}.
  </p>
  </div>
</details>"""

    cold_section = ""
    if not cold_players.empty:
        cold_rows = ""
        for _, r in cold_players.iterrows():
            cold_rows += (
                f'<li style="margin-bottom:4px">'
                f'<strong>{html_module.escape(str(r["player_name"]))}</strong> — '
                f'streak {int(r["streak"])} · line {r["offered_line"]:.1f} · '
                f'edge {abs(r["edge"])*100:.1f}pp</li>'
            )
        cold_section = f"""
<div style="background:#fff8e6;border:1px solid #f0c040;border-radius:6px;
            padding:14px 18px;margin-bottom:20px">
  <div style="font-weight:600;font-size:13px;color:#92400e;margin-bottom:8px">
    ⚠ Cold Streak Alert — {len(cold_players)} player{'s' if len(cold_players) > 1 else ''}
  </div>
  <ul style="font-size:12px;color:#78350f;margin:0;padding-left:18px">{cold_rows}</ul>
</div>"""

    # ── Strategy context: in-sample odds bucket reference ────────────────────────────
    is_rows = "".join(
        f'<tr style="border-bottom:1px solid #e5e7eb">'
        f'<td style="padding:5px 10px">{bkt}</td>'
        f'<td style="padding:5px 10px;text-align:center">{v["pct"]}</td>'
        f'<td style="padding:5px 10px;text-align:center">{v["n"]}</td>'
        f'<td style="padding:5px 10px;text-align:center">{v["wr"]}</td>'
        f'<td style="padding:5px 10px;text-align:center">{v["breakeven"]}</td>'
        f'<td style="padding:5px 10px;text-align:center">{v["roi"]}</td>'
        f'<td style="padding:5px 10px;text-align:center">{v["avg_odds"]}</td>'
        f'</tr>'
        for bkt, v in _IS_BUCKETS.items()
    )
    context_section = f"""
<details style="margin-bottom:18px">
  <summary style="cursor:pointer;font-size:12px;font-weight:600;color:#6b7280;padding:8px 0;user-select:none">
    Strategy context (in-sample odds bucket benchmarks) — click to expand
  </summary>
  <div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:6px;padding:12px 14px;margin-top:6px">
    <p style="font-size:12px;color:#374151;margin:0 0 8px">
      <strong>83% of bets are -odds favorites</strong> (avg -124). Edge comes from win rate (57.8%) beating
      breakeven (55.4%), not from finding dogs. Dog bets (+odds, 14.4% of bets) are historically weaker —
      3.35% ROI at avg +197 vs 4.65% for favorites. Dog bets are flagged <span style="background:#fee2e2;
      color:#991b1b;padding:1px 5px;border-radius:3px;font-size:10px;font-weight:700">dog</span> below.
    </p>
    <table style="width:100%;border-collapse:collapse;font-size:12px">
    <thead><tr style="background:#374151;color:#fff">
      <th style="padding:5px 10px;text-align:left">Bucket</th>
      <th style="padding:5px 10px;text-align:center">% of strat</th>
      <th style="padding:5px 10px;text-align:center">In-sample n</th>
      <th style="padding:5px 10px;text-align:center">In-sample win%</th>
      <th style="padding:5px 10px;text-align:center">Breakeven</th>
      <th style="padding:5px 10px;text-align:center">In-sample ROI</th>
      <th style="padding:5px 10px;text-align:center">Avg odds</th>
    </tr></thead>
    <tbody>{is_rows}</tbody>
    </table>
  </div>
</details>"""

    # ── Season-to-date by odds bucket (from settled history) ──────────────────
    ytd_section = ""
    if history is not None and not history.empty and "outcome" in history.columns:
        settled = history.copy()
        settled["bkt"] = settled["consensus_over_price"].apply(_odds_bucket)

        def _ytd_units(row):
            op = float(row.get("consensus_over_price", -110))
            if row["outcome"] == "push": return 0.0
            if row["outcome"] == "loss": return -1.0
            return op / 100 if op > 0 else 100 / abs(op)

        settled["u"] = settled.apply(_ytd_units, axis=1)
        n_total = len(settled)

        ytd_rows = ""
        for bkt in ["dog (+odds)", "even", "fav (-odds)"]:
            sub = settled[settled["bkt"] == bkt]
            if len(sub) == 0:
                continue
            n    = len(sub)
            wins = (sub["outcome"] == "win").sum()
            wr   = wins / n
            u    = sub["u"].sum()
            roi  = u / n * 100
            be   = _IS_BUCKETS.get(bkt, {}).get("breakeven", "—")
            wr_color = "#065f46" if wr * 100 > float(be.rstrip("%")) else "#991b1b"
            ytd_rows += (
                f'<tr style="border-bottom:1px solid #e5e7eb">'
                f'<td style="padding:5px 10px">{bkt}</td>'
                f'<td style="padding:5px 10px;text-align:center">{n}</td>'
                f'<td style="padding:5px 10px;text-align:center;color:{wr_color};font-weight:600">{wr*100:.1f}%</td>'
                f'<td style="padding:5px 10px;text-align:center">{be} (in-sample)</td>'
                f'<td style="padding:5px 10px;text-align:center">{u:+.2f}u</td>'
                f'<td style="padding:5px 10px;text-align:center">{roi:.1f}%</td>'
                f'</tr>'
            )
        if ytd_rows:
            ytd_section = f"""
<div style="background:#f0f9ff;border:1px solid #bae6fd;border-radius:6px;padding:12px 14px;margin-bottom:18px">
  <div style="font-weight:600;font-size:12px;color:#0369a1;margin-bottom:8px">
    Season-to-date by odds bucket ({n_total} settled bets)
  </div>
  <table style="width:100%;border-collapse:collapse;font-size:12px">
  <thead><tr style="background:#0369a1;color:#fff">
    <th style="padding:5px 10px;text-align:left">Bucket</th>
    <th style="padding:5px 10px;text-align:center">n bets</th>
    <th style="padding:5px 10px;text-align:center">Win rate</th>
    <th style="padding:5px 10px;text-align:center">Breakeven</th>
    <th style="padding:5px 10px;text-align:center">Units</th>
    <th style="padding:5px 10px;text-align:center">ROI</th>
  </tr></thead>
  <tbody>{ytd_rows}</tbody>
  </table>
</div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>NFL Rec Yards — {gameday}</title></head>
<body style="margin:0;padding:16px;background:#f4f4f5;font-family:{_SANS};font-size:13px;color:#1a1a1a">
<div style="max-width:900px;margin:0 auto;background:#fff;padding:24px;border-radius:8px;border:1px solid #e2e2e4">
  <h2 style="font-size:18px;margin:0 0 4px">NFL Receiving Yards — {gameday}</h2>
  <p style="color:#6b7280;font-size:12px;margin:0 0 6px">
    Generated {datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')}
    &nbsp;|&nbsp; {n_players_scored} players scored
    &nbsp;|&nbsp; <strong>{len(bets)} qualifying bet{'s' if len(bets) != 1 else ''}</strong>
  </p>
  <p style="color:#9ca3af;font-size:11px;margin:0 0 20px">
    Strategy: OVER · edge ≥ {EDGE_THRESHOLD*100:.0f}pp · lines {LINE_MIN}–{LINE_MAX} · min {MIN_BOOKS} books
  </p>
  {ytd_section}{cold_section}{context_section}
  {no_bets_msg}{table}{debug_section}
</div>
</body>
</html>"""


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
                TopicArn=SNS_TOPIC_ARN, Subject=subject[:100], Message=text_body[:256_000],
            )
            print(f"  SNS published")
        except Exception as e:
            print(f"  SNS failed: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gameday", type=str, default=None)
    args    = parser.parse_args()
    gameday = args.gameday or today_et()
    season  = current_nfl_season()

    print(f"\nNFL Rec Yards Pipeline — gameday={gameday}  season={season}")
    print("=" * 60)

    if not ODDS_API_KEY:
        sys.exit("ODDS_API_KEY not set")

    print("\n  Fetching events...")
    events = fetch_events(gameday)
    if not events:
        msg = f"No NFL events found for {gameday}."
        print(f"  {msg}")
        send_email(f"NFL Rec Yards — {gameday} — No games", f"<p>{msg}</p>", msg)
        return

    print("\n  Fetching game lines...")
    game_lines = fetch_game_lines(events)
    print(f"    {len(game_lines)} team-game rows")

    print(f"\n  Fetching {MARKET} props...")
    props_raw = fetch_props(events)
    if props_raw.empty:
        msg = f"No {MARKET} props available for {gameday}."
        print(f"  {msg}")
        send_email(f"NFL Rec Yards — {gameday} — No props", f"<p>{msg}</p>", msg)
        return
    print(f"    {len(props_raw)} prop rows  ({props_raw['player_norm'].nunique()} players)")

    print("\n  Building per-book rows...")
    per_book = build_per_book_rows(props_raw)
    print(f"    {per_book['player_norm'].nunique()} players  ×  {len(per_book)} player-book rows")

    print("\n  Loading spine from S3...")
    spine    = s3_get_parquet(f"{S3_PREFIX}/spine/nfl_rec_yards_historical_spine.parquet")
    name_map = load_name_map()
    print(f"    Spine: {len(spine):,} rows  |  name map: {len(name_map)} entries")

    df = add_spine_features(per_book, spine, name_map)
    df = add_game_context(df, game_lines)

    n_with_feats = df[BEST_FEATS].notna().all(axis=1).sum()
    print(f"    {n_with_feats}/{len(df)} players have all required features")

    print("\n  Loading model artifacts...")
    artifacts = load_artifacts()

    print("  Running inference...")
    results = run_inference(df, artifacts)
    bets    = filter_bets(results)
    print(f"  Qualifying OVER bets: {len(bets)}")

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

    save_cols = [
        "player_name", "player_norm", "team", "position", "event_id",
        "book", "offered_line", "p_hybrid", "p_market", "edge",
        "consensus_under_price", "consensus_over_price", "n_books",
        "streak", "cold_streak_warning",
        *BEST_FEATS,
    ]
    rec_df = bets[[c for c in save_cols if c in bets.columns]].copy()
    rec_df["gameday"] = gameday
    rec_df["season"]  = season
    rec_key = f"{S3_PREFIX}/daily_runs/{gameday}/recommendations.csv"
    s3_put_csv(rec_key, rec_df)
    print(f"\n  Recommendations saved → s3://{S3_BUCKET}/{rec_key}")

    html_body = build_recommendations_html(bets, gameday, n_with_feats, history=history)
    n_cold    = int(bets.get("cold_streak_warning", pd.Series(0)).sum())
    subject   = (
        f"NFL Rec Yards — {gameday} — {len(bets)} bet{'s' if len(bets) != 1 else ''}"
        + (f" ⚠ {n_cold} cold streak" if n_cold else "")
    )
    text_body = (
        f"NFL Rec Yards — {gameday}\n\n"
        f"Qualifying OVER bets: {len(bets)}\n"
        + "\n".join(
            f"  {r['player_name']} | {r.get('book','?')} | line {r['offered_line']:.1f} | "
            f"OVER {fmt_odds(r.get('consensus_over_price'))} | edge {abs(r['edge'])*100:.1f}pp"
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
