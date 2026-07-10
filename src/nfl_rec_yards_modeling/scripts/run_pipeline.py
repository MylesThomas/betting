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
from datetime import date, datetime, timedelta
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


def _commence_to_et(commence_time: str) -> str:
    """Convert ISO UTC time string (from Odds API) to 'H:MM AM ET' display format."""
    try:
        dt_utc = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
        dt_et  = dt_utc.astimezone(ET)
        return dt_et.strftime("%I:%M %p ET").lstrip("0")
    except Exception:
        return "TBD"

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


def yesterday_et() -> str:
    return (datetime.now(ET) - timedelta(days=1)).strftime("%Y-%m-%d")


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
                "raw_over_prob":         imp_o,
                "raw_under_prob":        imp_u,
                "raw_total":             total,
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
    # Step 4: clip p_model to [0.01, 0.99] and log boundary hits
    n_low  = int((p_hyb < 0.01).sum())
    n_high = int((p_hyb > 0.99).sum())
    p_hyb  = np.clip(p_hyb, 0.01, 0.99)
    if n_low + n_high:
        print(f"  ⚠ p_model clip: {n_low} rows → 0.01, {n_high} rows → 0.99 "
              f"({(n_low + n_high) / len(p_hyb):.1%} of scored rows) — "
              f"review for feature drift before sending email")
    else:
        print(f"  p_model clip [0.01, 0.99]: 0 rows hit boundary ✓")

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




# ── HTML email ────────────────────────────────────────────────────────────────

def _american_to_payout(price: float) -> float:
    return 100.0 / abs(price) if price < 0 else price / 100.0


_BOOK_DISPLAY: dict[str, str] = {
    "draftkings":      "DraftKings",
    "fanduel":         "FanDuel",
    "betmgm":          "BetMGM",
    "williamhill_us":  "Caesars",
    "espnbet":         "ESPN BET",
    "betrivers":       "BetRivers",
    "fliff":           "Fliff",
    "betonlineag":     "BetOnline",
    "bovada":          "Bovada",
    "betparx":         "BetParx",
    "hardrockbet":     "Hard Rock Bet",
    "fanatics":        "Fanatics",
    "ballybet":        "Bally Bet",
}


def _book_display(key: str) -> str:
    return _BOOK_DISPLAY.get(str(key).lower(), str(key).title())


# Rolling features shown as columns in the email table
_EMAIL_FEAT_COLS: list[tuple[str, str, str]] = [
    # (DataFrame column, header label, Python format string)
    ("receiving_yards_L8", "Rec Yds L8", "{:.1f}"),
    ("target_share_L8",    "Tgt% L8",    "{:.1%}"),
    ("snap_pct_L8",        "Snap% L8",   "{:.1%}"),
    ("game_total",         "Game Tot",   "{:.1f}"),
    ("proj_own_score",     "Proj Sc",    "{:.1f}"),
]

# Model inputs reference table (Feature | Shown as | What it measures | Role)
_MODEL_INPUTS_REF: list[tuple[str, str, str, str]] = [
    ("offered_line",       "Line",         "Sportsbook's offered line for this player-game",                          "Context"),
    ("game_total",         "Game Tot",     "Game over/under total",                                                    "Context"),
    ("proj_own_score",     "Proj Sc",      "Projected team score = (game_total + team_spread) / 2",                   "Context"),
    ("receiving_yards_L8", "Rec Yds L8",   "Player rolling avg receiving yards, last 8 games",                        "Primary signal"),
    ("target_share_L8",    "Tgt% L8",      "Player rolling avg target share (targets ÷ team targets), last 8 games",  "Primary signal"),
    ("snap_pct_L8",        "Snap% L8",     "Player rolling avg snap pct, last 8 games",                               "Primary signal"),
    ("pos_TE",             "Pos",          "Player position: 1 = TE, 0 = WR",                                        "Context"),
    ("market_under_prob",  "Mkt P(U)",     "De-vigged market P(under) for this specific book and line (per-book)",    "Market anchor"),
]

# In-sample odds bucket benchmarks from Step 6 research (breakeven as float %)
_IS_BUCKETS = {
    "dog (+odds)": {"pct": "14.4%", "n": 522,  "wr": 35.1, "roi": 3.35,  "avg_odds": "+197", "breakeven": 33.7},
    "even":        {"pct": "2.5%",  "n": 90,   "wr": 65.6, "roi": 31.11, "avg_odds": "+100", "breakeven": 50.0},
    "fav (-odds)": {"pct": "83.1%", "n": 3006, "wr": 57.8, "roi": 4.65,  "avg_odds": "-124", "breakeven": 55.4},
}


def _odds_bucket(over_price) -> str:
    try:
        p = float(over_price)
    except (TypeError, ValueError):
        return "unknown"
    if p > 0:
        return "dog (+odds)"
    if p == 100 or p == -100:
        return "even"
    return "fav (-odds)"


def fmt_odds(price) -> str:
    return "—" if pd.isna(price) else f"{int(price):+d}"


def _fv(val, fmt: str = "{:.1f}") -> str:
    """Format a scalar value, returning '—' for NaN/None."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    try:
        return fmt.format(float(val))
    except Exception:
        return str(val)


def _td(content: str, align: str = "center", style: str = "") -> str:
    return (
        f'<td style="padding:5px 8px;text-align:{align};'
        f'border:1px solid #e5e7eb;{style}">'
        f'{html_module.escape(str(content))}</td>'
    )


def _build_yesterday_section(yesterday: str, rows: pd.DataFrame) -> str:
    if rows.empty:
        return f'<p style="color:#6b7280;font-size:13px;margin:8px 0">No qualifying bets placed on {yesterday}.</p>'

    def _badge(outcome: str) -> str:
        sty = {
            "win":       "background:#d1fae5;color:#065f46;padding:2px 8px;border-radius:4px;font-weight:600",
            "loss":      "background:#fee2e2;color:#991b1b;padding:2px 8px;border-radius:4px;font-weight:600",
            "unmatched": "background:#f3f4f6;color:#6b7280;padding:2px 8px;border-radius:4px",
        }
        lbl = {"win": "WIN", "loss": "LOSS", "unmatched": "N/A"}
        return f'<span style="{sty.get(outcome, "")}">{lbl.get(outcome, outcome.upper())}</span>'

    def _pnl_html(outcome: str, price: float) -> str:
        if outcome == "win":
            v = _american_to_payout(price)
            return (f'<td style="text-align:right;padding:6px 10px;font-family:{_MONO};'
                    f'font-weight:600;color:#065f46">+{v:.3f}u</td>')
        if outcome == "loss":
            return (f'<td style="text-align:right;padding:6px 10px;font-family:{_MONO};'
                    f'font-weight:600;color:#991b1b">−1.000u</td>')
        return f'<td style="text-align:right;padding:6px 10px;color:#9ca3af">—</td>'

    rows = rows.copy()
    rows["_ord"] = rows["outcome"].map({"win": 0, "loss": 1, "unmatched": 2}).fillna(3)
    rows = rows.sort_values(["_ord", "edge"], ascending=[True, False]).drop(columns=["_ord"])

    bet_rows_html = ""
    for _, r in rows.iterrows():
        actual_s  = f"{float(r['actual_yards']):.0f}" if not pd.isna(r.get("actual_yards", float("nan"))) else "—"
        odds_s    = f"{int(r['consensus_over_price']):+d}" if not pd.isna(r.get("consensus_over_price", float("nan"))) else "—"
        edge_s    = f"+{abs(float(r['edge']))*100:.1f}pp" if not pd.isna(r.get("edge", float("nan"))) else "—"
        opponent  = html_module.escape(str(r.get("opponent") or "—"))
        direction = html_module.escape(str(r.get("recommendation") or "OVER"))
        outcome   = str(r.get("outcome", "unmatched"))
        price     = float(r.get("consensus_over_price", -110) or -110)
        book      = html_module.escape(str(r.get("book") or "—"))
        bet_rows_html += f"""
<tr style="border-bottom:1px solid #f3f4f6">
  <td style="padding:6px 10px;font-weight:600">{html_module.escape(str(r.get('player_name', '')))}</td>
  <td style="padding:6px 10px">{html_module.escape(str(r.get('team', '')))}</td>
  <td style="padding:6px 10px">{opponent}</td>
  <td style="padding:6px 10px;text-align:center;font-weight:600;color:#1d6fa4">{direction}</td>
  <td style="padding:6px 10px;text-align:center;font-family:{_MONO}">{_fv(r.get('offered_line'), '{:.1f}')}</td>
  <td style="padding:6px 10px;text-align:center;font-size:11px;color:#374151">{book}</td>
  <td style="padding:6px 10px;text-align:center;font-family:{_MONO}">{odds_s}</td>
  <td style="padding:6px 10px;text-align:center;font-family:{_MONO}">{edge_s}</td>
  <td style="padding:6px 10px;text-align:center;font-family:{_MONO}">{actual_s}</td>
  <td style="padding:6px 10px;text-align:center">{_badge(outcome)}</td>
  {_pnl_html(outcome, price)}
</tr>"""

    settled_rows = rows[rows["outcome"].isin(["win", "loss"])]
    n_win   = int((settled_rows["outcome"] == "win").sum())
    n_loss  = int((settled_rows["outcome"] == "loss").sum())
    wins    = settled_rows[settled_rows["outcome"] == "win"]
    pnl     = wins["consensus_over_price"].apply(lambda p: _american_to_payout(float(p))).sum() - n_loss
    pnl_s   = f"+{pnl:.2f}u" if pnl >= 0 else f"{pnl:.2f}u"
    pnl_c   = "#065f46" if pnl >= 0 else "#991b1b"

    game_summary: dict[str, dict] = {}
    for _, r in rows.iterrows():
        gk = f"{r.get('team', '?')} vs {r.get('opponent', '?')}"
        if gk not in game_summary:
            game_summary[gk] = {"bets": 0, "win": 0, "loss": 0, "pnl": 0.0}
        g = game_summary[gk]
        g["bets"] += 1
        oc = str(r.get("outcome", ""))
        if oc == "win":
            g["win"] += 1
            g["pnl"] += _american_to_payout(float(r.get("consensus_over_price", -110) or -110))
        elif oc == "loss":
            g["loss"] += 1
            g["pnl"] -= 1.0

    game_rows_html = "".join(
        f'<tr style="border-bottom:1px solid #e5e7eb">'
        f'<td style="padding:5px 10px;font-weight:600">{html_module.escape(gk)}</td>'
        f'<td style="padding:5px 10px;text-align:center">{g["bets"]}</td>'
        f'<td style="padding:5px 10px;text-align:center;color:#065f46">{g["win"]}</td>'
        f'<td style="padding:5px 10px;text-align:center;color:#991b1b">{g["loss"]}</td>'
        f'<td style="padding:5px 10px;text-align:right;font-family:{_MONO};font-weight:600;'
        f'color:{"#065f46" if g["pnl"] >= 0 else "#991b1b"}">'
        f'{"+" if g["pnl"] >= 0 else ""}{g["pnl"]:.2f}u</td>'
        f'</tr>'
        for gk, g in game_summary.items()
    )

    return f"""<h3 style="font-size:15px;font-weight:700;color:#1a1a2e;margin:0 0 10px">
  Yesterday's Results — {yesterday}
  &nbsp;<span style="font-size:13px;font-weight:400;color:{pnl_c}">{n_win}W–{n_loss}L &nbsp;{pnl_s}</span>
</h3>
<table style="width:100%;border-collapse:collapse;font-size:13px;margin-bottom:16px">
<thead><tr style="background:#1d2d44;color:#fff">
  <th style="padding:7px 10px;text-align:left">Player</th>
  <th style="padding:7px 10px;text-align:left">Team</th>
  <th style="padding:7px 10px;text-align:left">Opponent</th>
  <th style="padding:7px 10px;text-align:center">Direction</th>
  <th style="padding:7px 10px;text-align:center">Line</th>
  <th style="padding:7px 10px;text-align:center">Book</th>
  <th style="padding:7px 10px;text-align:center">Odds</th>
  <th style="padding:7px 10px;text-align:center">Edge</th>
  <th style="padding:7px 10px;text-align:center">Actual Yards</th>
  <th style="padding:7px 10px;text-align:center">Outcome</th>
  <th style="padding:7px 10px;text-align:right">P&amp;L</th>
</tr></thead>
<tbody>{bet_rows_html}</tbody>
</table>
<table style="border-collapse:collapse;font-size:12px;margin-bottom:8px">
<thead><tr style="background:#374151;color:#fff">
  <th style="padding:5px 10px;text-align:left">Game</th>
  <th style="padding:5px 10px;text-align:center">Bets</th>
  <th style="padding:5px 10px;text-align:center">W</th>
  <th style="padding:5px 10px;text-align:center">L</th>
  <th style="padding:5px 10px;text-align:right">Net</th>
</tr></thead>
<tbody>{game_rows_html}</tbody>
</table>"""


def _build_alltime_section(history: pd.DataFrame) -> str:
    settled = history[history["outcome"].isin(["win", "loss"])].copy() if not history.empty else pd.DataFrame()
    if settled.empty:
        return '<p style="color:#6b7280;font-size:13px;margin:8px 0">No settled bets yet.</p>'

    n_bets   = len(settled)
    n_win    = int((settled["outcome"] == "win").sum())
    n_loss   = int((settled["outcome"] == "loss").sum())
    wins     = settled[settled["outcome"] == "win"]
    pnl      = wins["consensus_over_price"].apply(lambda p: _american_to_payout(float(p))).sum() - n_loss
    hit_rate = n_win / n_bets
    roi      = pnl / n_bets
    pnl_c    = "#065f46" if pnl >= 0 else "#991b1b"
    roi_c    = "#065f46" if roi >= 0 else "#991b1b"

    def _card(label: str, value: str, color: str = "#111827") -> str:
        return (
            f'<div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:6px;'
            f'padding:12px 16px;min-width:90px;text-align:center">'
            f'<div style="font-size:10px;color:#6b7280;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px">{label}</div>'
            f'<div style="font-size:17px;font-weight:700;color:{color}">{html_module.escape(value)}</div>'
            f'</div>'
        )

    stat_cards = (
        f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px">'
        + _card("Record", f"{n_win}W–{n_loss}L")
        + _card("Win %", f"{hit_rate:.1%}")
        + _card("P&L", f"+{pnl:.2f}u" if pnl >= 0 else f"{pnl:.2f}u", pnl_c)
        + _card("ROI", f"{roi:+.1%}", roi_c)
        + _card("Bets", str(n_bets))
        + f'</div>'
    )

    season_rows_html = ""
    if "season" in settled.columns:
        for season, sg in settled.sort_values("season").groupby("season"):
            sn   = len(sg)
            sw   = int((sg["outcome"] == "win").sum())
            sl   = int((sg["outcome"] == "loss").sum())
            sg_w = sg[sg["outcome"] == "win"]
            sp   = sg_w["consensus_over_price"].apply(lambda p: _american_to_payout(float(p))).sum() - sl
            sr   = sp / sn
            spc  = "#065f46" if sp >= 0 else "#991b1b"
            src  = "#065f46" if sr >= 0 else "#991b1b"
            season_rows_html += (
                f'<tr style="border-bottom:1px solid #e5e7eb">'
                f'<td style="padding:5px 10px;font-weight:600">{season}</td>'
                f'<td style="padding:5px 10px;text-align:center">{sn}</td>'
                f'<td style="padding:5px 10px;text-align:center">{sw}W–{sl}L</td>'
                f'<td style="padding:5px 10px;text-align:center">{sw/sn:.1%}</td>'
                f'<td style="padding:5px 10px;text-align:right;font-family:{_MONO};font-weight:600;color:{spc}">'
                f'{"+" if sp >= 0 else ""}{sp:.2f}u</td>'
                f'<td style="padding:5px 10px;text-align:right;font-weight:600;color:{src}">{sr:+.1%}</td>'
                f'</tr>'
            )

    season_table = ""
    if season_rows_html:
        season_table = f"""<table style="border-collapse:collapse;font-size:12px;margin-bottom:12px">
<thead><tr style="background:#374151;color:#fff">
  <th style="padding:5px 10px;text-align:left">Season</th>
  <th style="padding:5px 10px;text-align:center">Bets</th>
  <th style="padding:5px 10px;text-align:center">Record</th>
  <th style="padding:5px 10px;text-align:center">Win %</th>
  <th style="padding:5px 10px;text-align:right">Units</th>
  <th style="padding:5px 10px;text-align:right">ROI</th>
</tr></thead>
<tbody>{season_rows_html}</tbody>
</table>"""

    # Odds bucket breakdown
    bucket_rows = ""
    if not history.empty and "consensus_over_price" in history.columns:
        h2 = history[history["outcome"].isin(["win", "loss", "push"])].copy()
        if not h2.empty:
            h2["bkt"] = h2["consensus_over_price"].apply(_odds_bucket)
            h2["u"]   = h2.apply(
                lambda r: (
                    0.0 if r["outcome"] == "push"
                    else (-1.0 if r["outcome"] == "loss"
                          else _american_to_payout(float(r["consensus_over_price"])))
                ),
                axis=1,
            )
            for bkt in ["dog (+odds)", "even", "fav (-odds)"]:
                sub = h2[h2["bkt"] == bkt]
                if not len(sub):
                    continue
                n   = len(sub)
                wr  = (sub["outcome"] == "win").sum() / n
                u   = sub["u"].sum()
                be  = _IS_BUCKETS.get(bkt, {}).get("breakeven", 50.0)
                wrc = "#065f46" if wr * 100 > be else "#991b1b"
                bucket_rows += (
                    f'<tr style="border-bottom:1px solid #e5e7eb">'
                    f'<td style="padding:5px 10px">{bkt}</td>'
                    f'<td style="padding:5px 10px;text-align:center">{n}</td>'
                    f'<td style="padding:5px 10px;text-align:center;color:{wrc};font-weight:600">{wr*100:.1f}%</td>'
                    f'<td style="padding:5px 10px;text-align:center">{be:.1f}% (IS)</td>'
                    f'<td style="padding:5px 10px;text-align:center">{u:+.2f}u</td>'
                    f'<td style="padding:5px 10px;text-align:center">{u/n*100:.1f}%</td>'
                    f'</tr>'
                )

    bucket_table = ""
    if bucket_rows:
        bucket_table = f"""<div style="background:#f0f9ff;border:1px solid #bae6fd;border-radius:6px;padding:12px 14px;margin-top:8px">
  <div style="font-weight:600;font-size:12px;color:#0369a1;margin-bottom:8px">By odds bucket ({n_bets} settled bets)</div>
  <table style="width:100%;border-collapse:collapse;font-size:12px">
  <thead><tr style="background:#0369a1;color:#fff">
    <th style="padding:5px 10px;text-align:left">Bucket</th>
    <th style="padding:5px 10px;text-align:center">n bets</th>
    <th style="padding:5px 10px;text-align:center">Win rate</th>
    <th style="padding:5px 10px;text-align:center">Breakeven (IS)</th>
    <th style="padding:5px 10px;text-align:center">Units</th>
    <th style="padding:5px 10px;text-align:center">ROI</th>
  </tr></thead>
  <tbody>{bucket_rows}</tbody>
  </table>
</div>"""

    footer = (
        f'<p style="font-size:11px;color:#9ca3af;margin-top:10px">'
        f'Flat-bet 1u &nbsp;|&nbsp; Strategy: OVER · edge ≥ {EDGE_THRESHOLD*100:.0f}pp · '
        f'lines {LINE_MIN}–{LINE_MAX} · min {MIN_BOOKS} books &nbsp;|&nbsp; '
        f'OOS baseline (2023–2025): 4,978 bets · 57.0% hit · +344.4u · +6.92% ROI &nbsp;|&nbsp; IS/OOS ratio: 1.74x'
        f'</p>'
    )

    return f"""<h3 style="font-size:15px;font-weight:700;color:#1a1a2e;margin:0 0 12px">All-Time Results</h3>
{stat_cards}
{season_table}
{bucket_table}
{footer}"""


def build_recommendations_html(
    all_scored: pd.DataFrame,
    bets: pd.DataFrame,
    gameday: str,
    history: pd.DataFrame | None = None,
    yesterday_settled: pd.DataFrame | None = None,
) -> str:
    # ── 1. Mark qualifying rows, add derived display columns ─────────────────
    scored = all_scored[all_scored["ols_pred"].notna()].copy()

    qual_keys: set = set()
    if not bets.empty:
        qual_keys = set(zip(
            bets["player_norm"],
            bets["book"],
            bets["offered_line"].round(1),
        ))

    def _is_play(r) -> bool:
        return (r.get("player_norm"), r.get("book"),
                round(float(r.get("offered_line") or 0), 1)) in qual_keys

    scored["status"]    = scored.apply(lambda r: "PLAY" if _is_play(r) else "", axis=1)
    scored["direction"] = scored.apply(
        lambda r: str(r.get("recommendation", "") or "") if r["status"] == "PLAY" else "",
        axis=1,
    )

    ph = scored["p_hybrid"].fillna(float("nan"))
    scored["mdl_over_prob"]  = 1.0 - ph
    scored["mdl_under_prob"] = ph

    edge = scored["edge"].fillna(float("nan"))   # p_hybrid - p_market; positive → under favored
    scored["over_edge"]  = -edge                  # positive when OVER favored
    scored["under_edge"] = edge                   # positive when UNDER favored

    # Propagate cold-streak warning from bets back into scored for badge display
    if not bets.empty and "cold_streak_warning" in bets.columns:
        cold_map = dict(zip(bets["player_norm"], bets["cold_streak_warning"]))
        streak_map = dict(zip(bets["player_norm"], bets.get("streak", pd.Series(dtype=int))))
        scored["cold_streak_warning"] = scored["player_norm"].map(cold_map).fillna(False)
        scored["streak"] = scored["player_norm"].map(streak_map).fillna(0).astype(int)
    else:
        scored["cold_streak_warning"] = False
        scored["streak"] = 0

    # ── 2. Deduplicate: PLAY rows stay; non-play → one row per (player, event) ─
    play_rows = scored[scored["status"] == "PLAY"].copy()
    nonplay   = scored[scored["status"] != "PLAY"].copy()
    if not nonplay.empty:
        nonplay = (
            nonplay
            .sort_values("offered_line")
            .groupby(["player_norm", "event_id"], as_index=False)
            .first()
        )
    display_df = pd.concat([play_rows, nonplay], ignore_index=True)

    # ── 3. Sort events by game time ───────────────────────────────────────────
    if "game_sort_key" in display_df.columns:
        event_order = (
            display_df[["event_id", "game_sort_key"]]
            .drop_duplicates("event_id")
            .sort_values("game_sort_key")["event_id"]
            .tolist()
        )
    elif "game_time_et" in display_df.columns:
        event_order = (
            display_df[["event_id", "game_time_et"]]
            .drop_duplicates("event_id")
            .sort_values("game_time_et")["event_id"]
            .tolist()
        )
    else:
        event_order = display_df["event_id"].drop_duplicates().tolist() if "event_id" in display_df.columns else []

    n_scored_players = scored["player_norm"].nunique()
    n_plays          = int((scored["status"] == "PLAY").sum())
    n_games          = len(event_order)

    # ── 4. Per-game section builder ───────────────────────────────────────────
    # Groups: Player/Game(5) Book(1) AmOdds(2) Implied(3) NoVig(4) Model(3) Edge(2) Inputs(n) Status(1)
    n_feat_cols = len(_EMAIL_FEAT_COLS)
    total_cols  = 5 + 1 + 2 + 3 + 4 + 3 + 2 + n_feat_cols + 1

    def _thead() -> str:
        feat_headers = "".join(
            f'<th style="padding:5px 8px;border:1px solid #6b7280;white-space:nowrap">{lbl}</th>'
            for _, lbl, _ in _EMAIL_FEAT_COLS
        )
        return f"""<thead>
  <tr style="background:#2d3748;color:#fff;font-size:11px;font-weight:700">
    <th colspan="5"           style="padding:5px 8px;text-align:center;border:1px solid #4a5568">Player / Game</th>
    <th colspan="1"           style="padding:5px 8px;text-align:center;border:1px solid #4a5568">Book</th>
    <th colspan="2"           style="padding:5px 8px;text-align:center;border:1px solid #4a5568">American Odds</th>
    <th colspan="3"           style="padding:5px 8px;text-align:center;border:1px solid #4a5568">Implied</th>
    <th colspan="4"           style="padding:5px 8px;text-align:center;border:1px solid #4a5568">No-Vig</th>
    <th colspan="1"           style="padding:5px 8px;text-align:center;border:1px solid #4a5568">Strategy</th>
    <th colspan="4"           style="padding:5px 8px;text-align:center;border:1px solid #4a5568">Model Prediction</th>
    <th colspan="2"           style="padding:5px 8px;text-align:center;border:1px solid #4a5568">Edge</th>
    <th colspan="{n_feat_cols}" style="padding:5px 8px;text-align:center;border:1px solid #4a5568">Model Inputs</th>
    <th colspan="1"           style="padding:5px 8px;text-align:center;border:1px solid #4a5568"></th>
  </tr>
  <tr style="background:#4a5568;color:#e2e8f0;font-size:11px">
    <th style="padding:5px 8px;border:1px solid #6b7280;text-align:left">Player</th>
    <th style="padding:5px 8px;border:1px solid #6b7280">Team</th>
    <th style="padding:5px 8px;border:1px solid #6b7280">Opp</th>
    <th style="padding:5px 8px;border:1px solid #6b7280;white-space:nowrap">Time (ET)</th>
    <th style="padding:5px 8px;border:1px solid #6b7280">Line</th>
    <th style="padding:5px 8px;border:1px solid #6b7280">Book</th>
    <th style="padding:5px 8px;border:1px solid #6b7280">Over</th>
    <th style="padding:5px 8px;border:1px solid #6b7280">Under</th>
    <th style="padding:5px 8px;border:1px solid #6b7280">Raw Over</th>
    <th style="padding:5px 8px;border:1px solid #6b7280">Raw Under</th>
    <th style="padding:5px 8px;border:1px solid #6b7280">Raw Total</th>
    <th style="padding:5px 8px;border:1px solid #6b7280">Fair Over</th>
    <th style="padding:5px 8px;border:1px solid #6b7280">Fair Under</th>
    <th style="padding:5px 8px;border:1px solid #6b7280">Fair Total</th>
    <th style="padding:5px 8px;border:1px solid #6b7280">Vig</th>
    <th style="padding:5px 8px;border:1px solid #6b7280">Dog?</th>
    <th style="padding:5px 8px;border:1px solid #6b7280">Prediction</th>
    <th style="padding:5px 8px;border:1px solid #6b7280">Delta</th>
    <th style="padding:5px 8px;border:1px solid #6b7280">Pred Over</th>
    <th style="padding:5px 8px;border:1px solid #6b7280">Pred Under</th>
    <th style="padding:5px 8px;border:1px solid #6b7280">Over Edge</th>
    <th style="padding:5px 8px;border:1px solid #6b7280">Under Edge</th>
    {feat_headers}
    <th style="padding:5px 8px;border:1px solid #6b7280">Status</th>
  </tr>
</thead>"""

    def _game_section(game_rows: pd.DataFrame) -> str:
        s          = game_rows.sort_values("player_name").iloc[0]
        game_time  = str(s.get("game_time_et") or "TBD")
        home       = str(s.get("home_team") or "")
        away       = str(s.get("away_team") or "")
        n_pl       = game_rows["player_norm"].nunique()
        n_pl_plays = int((game_rows["status"] == "PLAY").sum())
        game_label = f"{game_time} · {away} @ {home}" if home and away else game_time
        plays_txt  = (f"{n_pl_plays} PLAY{'S' if n_pl_plays != 1 else ''}"
                      if n_pl_plays else "no plays")

        tbody_rows = ""
        for idx, (_, r) in enumerate(game_rows.sort_values("player_name").iterrows()):
            is_play   = r.get("status") == "PLAY"
            direction = str(r.get("direction") or "")
            is_over   = direction == "OVER"
            is_under  = direction == "UNDER"
            is_cold   = bool(r.get("cold_streak_warning", False))

            if is_play and is_over:
                row_bg = "background:#e6f4ea"
            elif is_play and is_under:
                row_bg = "background:#fce8e6"
            else:
                row_bg = "background:#f9f9f9" if idx % 2 == 0 else "background:#fff"

            cold_badge = (
                f' <span style="background:#fef3c7;color:#92400e;padding:1px 5px;'
                f'border-radius:3px;font-size:10px;font-weight:700">'
                f'⚠ streak {int(r.get("streak", 0))}</span>'
            ) if is_cold else ""

            player_cell = (
                f'<td style="padding:5px 8px;text-align:left;border:1px solid #e5e7eb;'
                f'{"font-weight:700;" if is_play else ""}">'
                f'{html_module.escape(str(r.get("player_name") or "—"))}{cold_badge}</td>'
            )

            oe = r.get("over_edge")
            ue = r.get("under_edge")
            oe_str = "—" if oe is None or (isinstance(oe, float) and np.isnan(oe)) else f"{oe*100:+.1f}pp"
            ue_str = "—" if ue is None or (isinstance(ue, float) and np.isnan(ue)) else f"{ue*100:+.1f}pp"
            oe_sty = "color:#065f46;font-weight:700;" if is_play and is_over else ""
            ue_sty = "color:#065f46;font-weight:700;" if is_play and is_under else ""

            raw_total = r.get("raw_total")
            vig = (raw_total - 1.0) if raw_total is not None and not (isinstance(raw_total, float) and np.isnan(raw_total)) else None

            _pred = r.get("ols_pred")
            _line = r.get("offered_line")
            _pred_ok = _pred is not None and not (isinstance(_pred, float) and np.isnan(_pred))
            _line_ok = _line is not None and not (isinstance(_line, float) and np.isnan(_line))
            delta = (_pred - _line) if (_pred_ok and _line_ok) else None

            feat_cells = ""
            for col, _, fmt in _EMAIL_FEAT_COLS:
                val    = r.get(col)
                is_nan = val is None or (isinstance(val, float) and np.isnan(val))
                is_sz  = (not is_nan) and (float(val) == 0) and col in {"receiving_yards_L8", "target_share_L8", "snap_pct_L8"}
                bg     = "background:#fee2e2;" if (is_nan or is_sz) else ""
                disp   = "—" if is_nan else _fv(val, fmt)
                feat_cells += (
                    f'<td style="padding:5px 8px;text-align:center;border:1px solid #e5e7eb;'
                    f'font-family:{_MONO};{bg}">{disp}</td>'
                )

            status_val = (f"PLAY - {direction}" if is_play and direction else
                          "PLAY" if is_play else "")
            status_sty = "font-weight:700;color:#065f46;" if is_play else "color:#d1d5db"

            tbody_rows += f"""<tr style="{row_bg}">
  {player_cell}
  {_td(r.get("team") or "—")}
  {_td(r.get("opponent") or "—")}
  {_td(r.get("game_time_et") or "—", style="font-size:11px;white-space:nowrap")}
  {_td(_fv(r.get("offered_line"), "{:.1f}"))}
  {_td(_book_display(r.get("book") or ""), style="font-size:11px;white-space:nowrap")}
  {_td(fmt_odds(r.get("over_price")), style=f"font-family:{_MONO}")}
  {_td(fmt_odds(r.get("under_price")), style=f"font-family:{_MONO}")}
  {_td(_fv(r.get("raw_over_prob"), "{:.1%}"), style=f"font-family:{_MONO}")}
  {_td(_fv(r.get("raw_under_prob"), "{:.1%}"), style=f"font-family:{_MONO}")}
  {_td(_fv(raw_total, "{:.1%}"), style=f"font-family:{_MONO}")}
  {_td(_fv(r.get("market_over_prob"), "{:.1%}"), style=f"font-family:{_MONO}")}
  {_td(_fv(r.get("market_under_prob"), "{:.1%}"), style=f"font-family:{_MONO}")}
  {_td("100.0%", style=f"font-family:{_MONO};color:#9ca3af")}
  {_td(_fv(vig, "{:.1%}"), style=f"font-family:{_MONO}")}
  {_td("✓" if (r.get("over_price") or 0) > 0 else "—", style="text-align:center")}
  {_td(_fv(r.get("ols_pred"), "{:.1f}"), style=f"font-family:{_MONO}")}
  {_td(_fv(delta, "{:+.1f}"), style=f"font-family:{_MONO}")}
  {_td(_fv(r.get("mdl_over_prob"), "{:.1%}"), style=f"font-family:{_MONO}")}
  {_td(_fv(r.get("mdl_under_prob"), "{:.1%}"), style=f"font-family:{_MONO}")}
  <td style="padding:5px 8px;text-align:center;border:1px solid #e5e7eb;font-family:{_MONO};{oe_sty}">{oe_str}</td>
  <td style="padding:5px 8px;text-align:center;border:1px solid #e5e7eb;font-family:{_MONO};{ue_sty}">{ue_str}</td>
  {feat_cells}
  {_td(status_val, style=status_sty)}
</tr>"""

        return f"""<div style="margin-bottom:24px">
  <div style="background:#1a1a2e;color:#fff;padding:10px 14px;border-radius:6px 6px 0 0;
              font-weight:600;font-size:13px">
    {html_module.escape(game_label)}
    &nbsp;·&nbsp; {n_pl} player{"s" if n_pl != 1 else ""} scored
    &nbsp;·&nbsp; <span style="color:#86efac">{plays_txt}</span>
  </div>
  <div style="overflow-x:auto">
  <table style="width:100%;border-collapse:collapse;font-size:12px">
  {_thead()}
  <tbody>{tbody_rows}</tbody>
  </table>
  </div>
</div>"""

    # ── 5. Assemble game sections ─────────────────────────────────────────────
    game_sections_html = ""
    for eid in event_order:
        game_rows = display_df[display_df["event_id"] == eid]
        game_sections_html += _game_section(game_rows)

    if not game_sections_html:
        game_sections_html = '<p style="color:#6b7280">No scored players today.</p>'

    # ── 6. Model inputs reference table ──────────────────────────────────────
    inputs_rows = "".join(
        f'<tr style="border-bottom:1px solid #e5e7eb">'
        f'<td style="padding:5px 10px;font-family:{_MONO};font-size:11px;color:#374151">{col}</td>'
        f'<td style="padding:5px 10px;font-weight:600">{shown_as}</td>'
        f'<td style="padding:5px 10px;color:#374151">{measures}</td>'
        f'<td style="padding:5px 10px;color:#6b7280">{role}</td>'
        f'</tr>'
        for col, shown_as, measures, role in _MODEL_INPUTS_REF
    )
    inputs_table = f"""<div style="margin-top:24px">
  <div style="font-size:12px;font-weight:600;color:#374151;margin-bottom:6px">Model Inputs</div>
  <table style="width:100%;border-collapse:collapse;font-size:12px">
  <thead><tr style="background:#374151;color:#fff">
    <th style="padding:5px 10px;text-align:left">Feature</th>
    <th style="padding:5px 10px;text-align:left">Shown as</th>
    <th style="padding:5px 10px;text-align:left">What it measures</th>
    <th style="padding:5px 10px;text-align:left">Role</th>
  </tr></thead>
  <tbody>{inputs_rows}</tbody>
  </table>
</div>"""

    # ── 7. Build sections 2 and 3 ─────────────────────────────────────────────
    _yesterday = yesterday_et()
    section2 = _build_yesterday_section(
        _yesterday,
        yesterday_settled if yesterday_settled is not None else pd.DataFrame(),
    )
    section3 = _build_alltime_section(
        history if history is not None else pd.DataFrame()
    )

    # ── 8. Assemble ───────────────────────────────────────────────────────────
    header_summary = (
        f'<div style="font-size:16px;font-weight:700;color:#1a1a2e;margin-bottom:16px">'
        f'{n_plays} play{"s" if n_plays != 1 else ""} today across '
        f'{n_games} game{"s" if n_games != 1 else ""}'
        f'</div>'
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>NFL Rec Yards — {gameday}</title></head>
<body style="margin:0;padding:16px;background:#f4f4f5;font-family:{_SANS};font-size:13px;color:#1a1a1a">
<div style="max-width:1200px;margin:0 auto;background:#fff;padding:24px;border-radius:8px;border:1px solid #e2e2e4">
  <h2 style="font-size:18px;margin:0 0 4px">NFL Receiving Yards — {gameday}</h2>
  <p style="color:#6b7280;font-size:12px;margin:0 0 16px">
    Generated {datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')}
    &nbsp;|&nbsp; {n_scored_players} players scored
    &nbsp;|&nbsp; Strategy: OVER · edge ≥ {EDGE_THRESHOLD*100:.0f}pp · lines {LINE_MIN}–{LINE_MAX} · min {MIN_BOOKS} books
  </p>
  {header_summary}
  {game_sections_html}
  {inputs_table}
  <hr style="border:none;border-top:2px solid #e5e7eb;margin:32px 0">
  {section2}
  <hr style="border:none;border-top:2px solid #e5e7eb;margin:32px 0">
  {section3}
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

    # Attach game time (ET display) and UTC sort key from Odds API event metadata
    game_times     = {ev["id"]: _commence_to_et(ev.get("commence_time", "")) for ev in events}
    game_sort_keys = {ev["id"]: ev.get("commence_time", "") for ev in events}
    per_book["game_time_et"]  = per_book["event_id"].map(game_times).fillna("TBD")
    per_book["game_sort_key"] = per_book["event_id"].map(game_sort_keys).fillna("")

    print("\n  Loading spine from S3...")
    spine    = s3_get_parquet(f"{S3_PREFIX}/spine/nfl_rec_yards_historical_spine.parquet")
    name_map = load_name_map()
    print(f"    Spine: {len(spine):,} rows  |  name map: {len(name_map)} entries")

    df = add_spine_features(per_book, spine, name_map)
    df = add_game_context(df, game_lines)

    # Derive opponent: if player's team matches home_team → opponent is away_team (and vice versa)
    df["opponent"] = df.apply(
        lambda r: (
            "—" if pd.isna(r.get("team"))
            else (
                str(r.get("away_team") or "—")
                if str(r.get("team", "")).upper() == str(r.get("home_team", "")).upper()
                else str(r.get("home_team") or "—")
            )
        ),
        axis=1,
    )

    n_with_feats = df[BEST_FEATS].notna().all(axis=1).sum()
    print(f"    {n_with_feats}/{len(df)} players have all required features")

    print("\n  Loading model artifacts...")
    artifacts = load_artifacts()

    print("  Running inference...")
    results = run_inference(df, artifacts)

    # ── Step 3 assert: yhat book-invariant ───────────────────────────────────
    scored_mask = results["ols_pred"].notna()
    if scored_mask.any():
        yhat_check = (
            results[scored_mask]
            .groupby(["player_norm", "event_id", "offered_line"])["ols_pred"]
            .nunique()
        )
        n_viol = int((yhat_check > 1).sum())
        if n_viol:
            print(f"  ⚠ ols_pred NOT book-invariant: {n_viol} (player, event, line) groups "
                  f"have varying predictions. market_under_prob is a per-book feature in "
                  f"BEST_FEATS — this is the expected cause.")
        else:
            print(f"  ✓ ols_pred book-invariant: 0 violations")

    # ── Step 4 assert: line monotonicity ─────────────────────────────────────
    scored_df = results[results["p_hybrid"].notna()].copy()
    pg_counts = scored_df.groupby(["player_norm", "event_id"])["offered_line"].nunique()
    multi_pg  = set(pg_counts[pg_counts > 1].index)
    n_inv, inv_examples = 0, []
    for (pn, eid), grp in scored_df.groupby(["player_norm", "event_id"]):
        if (pn, eid) not in multi_pg:
            continue
        grp_s = grp.sort_values("offered_line")
        pu = grp_s["p_hybrid"].values
        ls = grp_s["offered_line"].values
        for i in range(len(pu) - 1):
            if pu[i + 1] < pu[i]:
                n_inv += 1
                if len(inv_examples) < 3:
                    inv_examples.append(
                        f"{pn} event={eid} line {ls[i]:.1f}→{ls[i+1]:.1f} "
                        f"p_u {pu[i]:.3f}→{pu[i+1]:.3f}"
                    )
    if multi_pg:
        rate = n_inv / len(multi_pg)
        flag = " ⚠ rate > 2% — investigate" if rate > 0.02 else " ✓ OK"
        print(f"  Line monotonicity: {n_inv} inversions / {len(multi_pg)} multi-line "
              f"player-games ({rate:.1%}){flag}")
        for ex in inv_examples:
            print(f"    example: {ex}")
    else:
        print("  Line monotonicity: no multi-line player-games found on this gameday")

    bets    = filter_bets(results)
    print(f"  Qualifying OVER bets: {len(bets)}")

    print("\n  Loading settled history for streak check...")
    history = load_settled_history()
    yesterday = yesterday_et()
    yest_rows = pd.DataFrame()
    if not history.empty:
        print(f"    Settled history: {len(history):,} rows")
        bets = check_cold_streaks(bets, history)
        n_cold = bets["cold_streak_warning"].sum()
        if n_cold:
            print(f"    ⚠ Cold streak alerts: {n_cold} player(s)")
        if "gameday" in history.columns:
            yest_rows = history[history["gameday"] == yesterday].copy()
            print(f"    Yesterday ({yesterday}): {len(yest_rows)} settled bets for Section 2")
    else:
        print("    No settled history yet (first run)")
        bets["streak"] = 0
        bets["cold_streak_warning"] = False

    save_cols = [
        "player_name", "player_norm", "team", "opponent", "position", "event_id",
        "game_time_et", "book", "offered_line", "p_hybrid", "p_market", "edge",
        "recommendation", "consensus_under_price", "consensus_over_price", "n_books",
        "streak", "cold_streak_warning",
        *BEST_FEATS,
    ]
    rec_df = bets[[c for c in save_cols if c in bets.columns]].copy()
    rec_df["gameday"] = gameday
    rec_df["season"]  = season
    rec_key = f"{S3_PREFIX}/daily_runs/{gameday}/recommendations.csv"
    s3_put_csv(rec_key, rec_df)
    print(f"\n  Recommendations saved → s3://{S3_BUCKET}/{rec_key}")

    html_body = build_recommendations_html(results, bets, gameday, history=history,
                                           yesterday_settled=yest_rows)
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
