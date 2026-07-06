"""
Live gameday pipeline for NBA player points props.

Strategy: S3 — UNDER only · shrinkage=0.25 · edge≥5pp · fav_only · all lines
  Out-of-sample: 1,396 bets · 56.2% win rate · +149.6u · 10.71% ROI · 15.2u max drawdown

For each player with a player_points prop on the given gameday:
  1. Fetches live events + props from The Odds API
  2. Joins with rolling features from the spine (S3)
  3. Scores: yhat = OLS.predict(features); mean_adj = line + 0.75*(yhat-line)
  4. Bootstraps P(under): sample 10K residuals, fraction ≤ offered_line
  5. Computes no-vig P(market_under) from posted odds
  6. edge_under = P(model_under) - P(market_under)
  7. Bet UNDER if edge_under >= 0.05 AND P(market_under) >= 0.50
  8. Sends SES HTML email + SNS notification
  9. Saves recommendations CSV to S3

S3 paths read:
  s3://the-odds-api-mt/nba/points_model/spine/nba_points_spine.parquet
  s3://the-odds-api-mt/nba/points_model/model/nba_points_model_ols.joblib
  s3://the-odds-api-mt/nba/points_model/model/nba_points_residuals.npy

S3 paths written:
  s3://the-odds-api-mt/nba/points_model/daily_runs/{gameday}/recommendations.csv

Run:
    python src/nba_points_modeling/scripts/run_pipeline.py
    python src/nba_points_modeling/scripts/run_pipeline.py --gameday 2026-10-29
"""
from __future__ import annotations

import argparse
import html as html_module
import os
import sys
import time
import warnings
from datetime import datetime
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import boto3
import botocore.exceptions
import joblib
import numpy as np
import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

ET = ZoneInfo("America/New_York")

ODDS_API_KEY  = os.environ.get("ODDS_API_KEY", "").strip()
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT         = "basketball_nba"
MARKET        = "player_points"
REGIONS       = "us"
SLEEP_S       = 0.25

S3_BUCKET     = "the-odds-api-mt"
MODEL_KEY     = "nba/points_model/model/nba_points_model_ols.joblib"
RESIDUALS_KEY = "nba/points_model/model/nba_points_residuals.npy"
SPINE_KEY     = "nba/points_model/spine/nba_points_spine.parquet"

SES_SOURCE    = os.environ.get("SETTLEMENT_SES_SOURCE", "").strip()
SES_TO_RAW    = os.environ.get("SETTLEMENT_SES_TO", "mylescgthomas@gmail.com").strip()
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "").strip()

# Strategy S3
SHRINKAGE      = 0.25
EDGE_THRESHOLD = 0.05
BACKUP_EDGE    = 0.03    # show near-misses between 3–5pp edge
FAV_ONLY       = True    # only bet when p_market_under >= 0.50
MIN_BOOKS      = 1
N_BOOT         = 10_000
RNG            = np.random.default_rng(42)

SETTLED_KEY    = "nba/points_model/settled/settled_bets.parquet"

FEATURES = [
    "pts_L1", "pts_L3", "pts_L5", "pts_L10", "pts_L20", "pts_career",
    "min_L5", "min_L20", "fga_L5",
    "is_home", "days_rest", "games_into_season",
    "opp_pts_allowed_L10",
    "offered_line", "novig_prob_over",
]

_MONO = "ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace"
_SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif"


# ── Helpers ───────────────────────────────────────────────────────────────────

def today_et() -> str:
    return datetime.now(ET).strftime("%Y-%m-%d")


def current_nba_season(gameday: str) -> str:
    yr = int(gameday[:4])
    mo = int(gameday[5:7])
    yr = yr if mo >= 10 else yr - 1
    return f"{yr}-{str(yr+1)[-2:]}"


def _normalize_name(name: str) -> str:
    import unicodedata, re
    name = unicodedata.normalize("NFD", str(name))
    name = "".join(c for c in name if unicodedata.category(c) != "Mn")
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+(jr|sr|ii|iii|iv)$", "", name.strip().lower())
    return re.sub(r"\s+", " ", name).strip()


def american_profit(odds: float) -> float:
    return odds / 100.0 if odds >= 0 else 100.0 / abs(odds)


def p_market_to_american(p: float) -> float:
    if p >= 0.5:
        return -(p / (1 - p) * 100)
    return (1 - p) / p * 100


def fmt_odds(price) -> str:
    return "—" if (price is None or pd.isna(price)) else f"{int(price):+d}"


def fmt_pct(p) -> str:
    return "—" if (p is None or pd.isna(p)) else f"{p*100:.1f}%"


# ── S3 ────────────────────────────────────────────────────────────────────────

def _s3():
    return boto3.client("s3")


def s3_get(key: str) -> bytes:
    return _s3().get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()


def s3_put_csv(key: str, df: pd.DataFrame) -> None:
    buf = BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    _s3().put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue())


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model_artifacts() -> tuple:
    print("Loading model artifacts from S3...", flush=True)
    model = joblib.load(BytesIO(s3_get(MODEL_KEY)))
    residuals = np.load(BytesIO(s3_get(RESIDUALS_KEY)))
    print(f"  Residuals: {len(residuals):,}  σ={residuals.std():.4f}")
    return model, residuals


# ── Odds API ──────────────────────────────────────────────────────────────────

def _api_get(path: str, params: dict) -> dict:
    if not ODDS_API_KEY:
        raise RuntimeError("ODDS_API_KEY not set")
    resp = requests.get(
        f"{ODDS_API_BASE}{path}",
        params={**params, "apiKey": ODDS_API_KEY},
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_events(gameday: str) -> list[dict]:
    data = _api_get(f"/sports/{SPORT}/events", {
        "dateFormat": "iso",
        "commenceTimeFrom": f"{gameday}T00:00:00Z",
        "commenceTimeTo":   f"{gameday}T23:59:59Z",
    })
    events = [e for e in data if gameday in e.get("commence_time", "")]
    print(f"  Events on {gameday}: {len(events)}")
    return events


def fetch_props(events: list[dict]) -> pd.DataFrame:
    rows = []
    for ev in events:
        time.sleep(SLEEP_S)
        try:
            data = _api_get(f"/sports/{SPORT}/events/{ev['id']}/odds", {
                "regions": REGIONS,
                "markets": MARKET,
                "oddsFormat": "american",
            })
        except requests.HTTPError as e:
            print(f"  Skipping {ev.get('id')} — {e}")
            continue
        for bk in data.get("bookmakers", []):
            for mkt in bk.get("markets", []):
                if mkt["key"] != MARKET:
                    continue
                for outcome in mkt.get("outcomes", []):
                    rows.append({
                        "event_id":  ev["id"],
                        "game_date": ev.get("commence_time", "")[:10],
                        "home_team": ev.get("home_team", ""),
                        "away_team": ev.get("away_team", ""),
                        "bookmaker": bk["key"],
                        "player":    outcome["description"],
                        "side":      outcome["name"].lower(),
                        "prop_line": float(outcome["point"]),
                        "odds":      float(outcome["price"]),
                    })
    return pd.DataFrame(rows)


# ── Feature assembly ──────────────────────────────────────────────────────────

def load_spine_latest(spine: pd.DataFrame, gameday: str) -> pd.DataFrame:
    """
    For each player, return their most recent row strictly before gameday.
    Provides rolling features (no lookahead). Market features are added below
    from the live API call.
    """
    spine = spine[spine["game_date"] < gameday].sort_values("game_date")
    roll_cols = [c for c in spine.columns if c.startswith("pts_") or c.startswith("min_") or c.startswith("fga_")]
    # Need at least one rolling feature populated
    any_feat = spine[roll_cols].notna().any(axis=1)
    latest = (
        spine[any_feat]
        .groupby("player_key", as_index=False)
        .last()
    )
    keep = ["player_key", "game_date", "season",
            "pts_L1", "pts_L3", "pts_L5", "pts_L10", "pts_L20", "pts_career",
            "min_L5", "min_L20", "fga_L5",
            "days_rest", "games_into_season", "opp_pts_allowed_L10"]
    return latest[[c for c in keep if c in latest.columns]].rename(
        columns={"game_date": "last_game_date", "season": "last_season"}
    )


def _novig_p_over(over_odds: float, under_odds: float) -> float:
    p_o = 1 / (1 + american_profit(over_odds))
    p_u = 1 / (1 + american_profit(under_odds))
    return p_o / (p_o + p_u)


def build_bet_rows(
    props: pd.DataFrame,
    spine_latest: pd.DataFrame,
    gameday: str,
) -> pd.DataFrame:
    """One consensus row per player. Market features from live API."""
    if props.empty:
        return pd.DataFrame()

    props["player_key"] = props["player"].apply(_normalize_name)

    over_df  = props[props["side"] == "over"].copy()
    under_df = props[props["side"] == "under"].copy()
    if over_df.empty or under_df.empty:
        return pd.DataFrame()

    # Consensus line per player
    line_agg = (
        over_df.groupby("player_key")
        .agg(
            offered_line = ("prop_line", "median"),
            min_line     = ("prop_line", "min"),
            max_line     = ("prop_line", "max"),
            n_books      = ("bookmaker", "nunique"),
            home_team    = ("home_team", "first"),
            away_team    = ("away_team", "first"),
        )
        .reset_index()
    )

    # Best over/under odds (highest profit = best value for bettor)
    over_df["_profit"]  = over_df["odds"].apply(american_profit)
    under_df["_profit"] = under_df["odds"].apply(american_profit)

    best_over = (
        over_df.sort_values("odds", ascending=False)
        .groupby("player_key")
        .agg(best_over_odds=("odds", "first"), best_over_book=("bookmaker", "first"))
        .reset_index()
    )
    best_under = (
        under_df.sort_values("odds", ascending=False)
        .groupby("player_key")
        .agg(best_under_odds=("odds", "first"), best_under_book=("bookmaker", "first"))
        .reset_index()
    )

    # Avg over/under profit for no-vig calculation
    avg_over  = over_df.groupby("player_key").agg(avg_over_profit=("_profit", "mean")).reset_index()
    avg_under = under_df.groupby("player_key").agg(avg_under_profit=("_profit", "mean")).reset_index()

    df = (
        line_agg
        .merge(best_over,  on="player_key", how="left")
        .merge(best_under, on="player_key", how="left")
        .merge(avg_over,   on="player_key", how="left")
        .merge(avg_under,  on="player_key", how="left")
        .merge(spine_latest, on="player_key", how="left")
    )

    # No-vig P(over) from market
    def _mkt_prob(row):
        op = row.get("avg_over_profit")
        up = row.get("avg_under_profit")
        if pd.notna(op) and pd.notna(up) and op > 0 and up > 0:
            p_o = 1 / (1 + op)
            p_u = 1 / (1 + up)
            return p_o / (p_o + p_u)
        return 0.5
    df["novig_prob_over"] = df.apply(_mkt_prob, axis=1)

    # is_home: 1 if player's team is the home team
    def _is_home(row):
        team_name = props.loc[
            (props["player_key"] == row["player_key"]) & (props["side"] == "over"),
            "away_team"
        ].values
        # We don't know the player's team directly — default to 0.5 (unknown)
        # The spine's last known home/away flag can serve as a rough proxy
        return np.nan

    # days_rest: days from last game to gameday
    df["days_rest"] = (
        pd.to_datetime(gameday) - pd.to_datetime(df["last_game_date"])
    ).dt.days.clip(upper=14).fillna(3)

    # games_into_season: approximate from last known value +1
    df["games_into_season"] = df["games_into_season"].fillna(0) + 1

    df["game_date"] = gameday
    df["player_name"] = df["player_key"]

    return df


# ── Scoring (Strategy S3) ─────────────────────────────────────────────────────

def bootstrap_p_under(yhat_arr: np.ndarray, line_arr: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    samples = RNG.choice(residuals, size=(len(yhat_arr), N_BOOT), replace=True)
    sims = yhat_arr[:, None] + samples
    return (sims <= line_arr[:, None]).mean(axis=1)


def score(df: pd.DataFrame, model, residuals: np.ndarray) -> pd.DataFrame:
    df = df.copy()

    # Fill is_home with 0.5 if unknown (spine rolling features absorb most of this)
    df["is_home"] = df.get("is_home", pd.Series(0.5, index=df.index)).fillna(0.5)

    missing_feats = [f for f in FEATURES if f not in df.columns]
    if missing_feats:
        print(f"  Warning: missing features {missing_feats} — filling with NaN")
        for f in missing_feats:
            df[f] = np.nan

    valid_mask = df[FEATURES].notna().all(axis=1)
    df["yhat"] = np.nan
    if valid_mask.sum() > 0:
        df.loc[valid_mask, "yhat"] = model.predict(df.loc[valid_mask, FEATURES])

    # Shrinkage
    df["mean_adj"] = df["offered_line"] + (1 - SHRINKAGE) * (df["yhat"] - df["offered_line"])

    # Bootstrap P(under)
    valid2 = df["mean_adj"].notna() & df["offered_line"].notna()
    df["p_model_under"] = np.nan
    if valid2.sum() > 0:
        df.loc[valid2, "p_model_under"] = bootstrap_p_under(
            df.loc[valid2, "mean_adj"].values,
            df.loc[valid2, "offered_line"].values,
            residuals,
        )

    df["p_market_under"] = 1.0 - df["novig_prob_over"]
    df["edge_under"]     = df["p_model_under"] - df["p_market_under"]

    # Strategy S3: UNDER if edge >= 5pp AND p_market_under >= 0.50 (even or better)
    df["bet"]    = (
        (df["edge_under"] >= EDGE_THRESHOLD) &
        (df["p_market_under"] >= 0.50) &
        df["edge_under"].notna()
    )
    df["direction"] = "UNDER"

    return df


# ── HTML email ────────────────────────────────────────────────────────────────

def load_season_stats(gameday: str) -> dict:
    """Read settled_bets.parquet and return cumulative stats for the current season."""
    season = current_nba_season(gameday)
    try:
        raw = s3_get(SETTLED_KEY)
        df  = pd.read_parquet(BytesIO(raw))
        df  = df[df.get("season", pd.Series("", index=df.index)) == season] if "season" in df.columns else df
        if df.empty:
            return {}
        wins   = (df["result"] == "WIN").sum()
        losses = (df["result"] == "LOSS").sum()
        pushes = (df.get("result", pd.Series()) == "PUSH").sum()
        dnps   = (df.get("result", pd.Series()) == "DNP").sum()
        units  = df["pnl_units"].sum() if "pnl_units" in df.columns else 0.0
        decided = wins + losses
        return {
            "season": season, "wins": int(wins), "losses": int(losses),
            "pushes": int(pushes), "dnps": int(dnps), "units": float(units),
            "roi": units / decided if decided > 0 else 0.0,
            "win_pct": wins / decided if decided > 0 else 0.0,
        }
    except Exception as e:
        print(f"  Could not load season stats: {e}")
        return {}


def build_html(bets: pd.DataFrame, all_scored: pd.DataFrame, gameday: str, season_stats: dict | None = None) -> str:
    now_str = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")
    n_with_props = len(all_scored)
    n_no_spine   = all_scored["pts_L5"].isna().sum()
    n_bets       = bets["bet"].sum()

    def _td(val, mono=True, warn=False, bold=False, color=None):
        style = "padding:6px 10px;text-align:center;"
        if mono:
            style += f"font-family:{_MONO};font-size:12px;"
        if warn:
            style += "background:#fee2e2;"
        if bold:
            style += "font-weight:bold;"
        if color:
            style += f"color:{color};"
        return f'<td style="{style}">{html_module.escape(str(val))}</td>'

    def _bet_rows(subset: pd.DataFrame) -> str:
        if subset.empty:
            return f'<tr><td colspan="14" style="padding:10px;color:#6b7280;font-style:italic;">None</td></tr>'
        rows = ""
        for _, row in subset.sort_values("edge_under", ascending=False).iterrows():
            edge_pct   = f"{row['edge_under']*100:.1f}pp"
            am_odds    = p_market_to_american(row["p_market_under"])
            warn_spine = pd.isna(row.get("pts_L5"))
            rows += f"""
            <tr>
              {_td(html_module.escape(str(row.get("player_name",""))), mono=False, bold=True)}
              {_td(row.get("home_team",""), mono=False)}
              {_td(f"{row['offered_line']:.1f}")}
              {_td(f"{row['yhat']:.1f}" if pd.notna(row.get('yhat')) else "—")}
              {_td(f"{row['mean_adj']:.1f}" if pd.notna(row.get('mean_adj')) else "—")}
              {_td(fmt_pct(row.get("p_model_under")), warn=row.get("p_model_under",0.5) < 0.50)}
              {_td(fmt_pct(row.get("p_market_under")))}
              {_td(edge_pct, bold=True)}
              {_td(fmt_odds(am_odds))}
              {_td(fmt_odds(row.get("best_under_odds")), warn=pd.isna(row.get("best_under_odds")))}
              {_td(str(row.get("best_under_book","—")), mono=False)}
              {_td(f"{row.get('n_books',0):.0f}")}
              {_td(f"{row.get('pts_L5','—'):.1f}" if pd.notna(row.get('pts_L5')) else "—", warn=warn_spine)}
              {_td(f"{row.get('pts_L20','—'):.1f}" if pd.notna(row.get('pts_L20')) else "—", warn=warn_spine)}
            </tr>"""
        return rows

    header = """<tr>
        <th>Player</th><th>Game</th><th>Line</th><th>yhat</th><th>mean_adj</th>
        <th>P(und) model</th><th>P(und) mkt</th><th>Edge</th>
        <th>Mkt odds (und)</th><th>Best odds</th><th>Best book</th>
        <th>n_books</th><th>PTS L5</th><th>PTS L20</th>
    </tr>"""

    primary_rows = _bet_rows(bets[bets["bet"]])
    backup_mask  = (
        bets["edge_under"].notna() &
        (bets["edge_under"] >= BACKUP_EDGE) &
        (bets["edge_under"] < EDGE_THRESHOLD) &
        (bets["p_market_under"] >= 0.50)
    )
    backup_rows  = _bet_rows(bets[backup_mask])

    # ── Season running total ──────────────────────────────────────────────────
    if season_stats:
        ss = season_stats
        units_color = "#16a34a" if ss["units"] >= 0 else "#dc2626"
        season_html = f"""
<div style="display:flex;gap:16px;margin:16px 0 24px;flex-wrap:wrap;">
  <div style="background:#f0f7ff;border:1px solid #bfdbfe;border-radius:6px;padding:10px 16px;min-width:100px;">
    <div style="font-size:10px;color:#6b7280;text-transform:uppercase;">{ss['season']} PnL</div>
    <div style="font-size:20px;font-weight:700;color:{units_color};">{ss['units']:+.2f}u</div>
  </div>
  <div style="background:#f0f7ff;border:1px solid #bfdbfe;border-radius:6px;padding:10px 16px;min-width:100px;">
    <div style="font-size:10px;color:#6b7280;text-transform:uppercase;">Record</div>
    <div style="font-size:16px;font-weight:700;">{ss['wins']}W–{ss['losses']}L{f"–{ss['pushes']}P" if ss["pushes"] else ""}</div>
  </div>
  <div style="background:#f0f7ff;border:1px solid #bfdbfe;border-radius:6px;padding:10px 16px;min-width:100px;">
    <div style="font-size:10px;color:#6b7280;text-transform:uppercase;">Win %</div>
    <div style="font-size:16px;font-weight:700;">{ss['win_pct']*100:.1f}%</div>
  </div>
  <div style="background:#f0f7ff;border:1px solid #bfdbfe;border-radius:6px;padding:10px 16px;min-width:100px;">
    <div style="font-size:10px;color:#6b7280;text-transform:uppercase;">ROI</div>
    <div style="font-size:16px;font-weight:700;color:{units_color};">{ss['roi']*100:+.1f}%</div>
  </div>
</div>"""
    else:
        season_html = ""

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body {{ font-family:{_SANS}; font-size:14px; color:#111; background:#fff; margin:20px; }}
    h2 {{ color:#1d4ed8; margin-bottom:4px; }}
    h3 {{ color:#374151; margin:24px 0 6px; }}
    table {{ border-collapse:collapse; width:100%; margin-bottom:20px; }}
    th {{ background:#1d4ed8; color:#fff; padding:8px 10px; text-align:center; font-size:12px; }}
    td {{ border-bottom:1px solid #e5e7eb; }}
    tr:hover td {{ background:#f9fafb; }}
    .backup th {{ background:#6b7280; }}
  </style>
</head>
<body>
  <h2>NBA Player Points — Bet Recommendations</h2>
  <p style="color:#6b7280;margin-top:0;">{gameday} &nbsp;·&nbsp; Generated {now_str}</p>
  <p>
    Strategy: <strong>S3 — UNDER only · shrinkage=0.25 · edge≥5pp · fav_only · all lines</strong><br>
    Players with props: <strong>{n_with_props}</strong> &nbsp;·&nbsp;
    Bets today: <strong>{int(n_bets)}</strong> &nbsp;·&nbsp;
    Missing spine: <strong>{n_no_spine}</strong>
  </p>

  {season_html}

  <h3>Bets (edge ≥ {EDGE_THRESHOLD*100:.0f}pp)</h3>
  <table>
    <thead>{header}</thead>
    <tbody>{primary_rows}</tbody>
  </table>

  <h3 style="color:#6b7280;">Near-Misses ({BACKUP_EDGE*100:.0f}–{EDGE_THRESHOLD*100:.0f}pp — context only, not recommended)</h3>
  <table class="backup">
    <thead>{header}</thead>
    <tbody>{backup_rows}</tbody>
  </table>

  <p style="font-size:11px;color:#9ca3af;">
    Edge = P(model_under) − P(market_under). Bet UNDER when edge ≥ {EDGE_THRESHOLD*100:.0f}pp and p_mkt_under ≥ 50%.
    OLS + bootstrap residuals (σ=6.61 pts). Out-of-sample backtest: 1,396 bets · 56.2% win · +149.6u · 10.71% ROI.
  </p>
</body>
</html>"""


def send_email(subject: str, html_body: str) -> None:
    if not SES_SOURCE:
        print("  SES_SOURCE not set — skipping email")
        return
    recipients = [r.strip() for r in SES_TO_RAW.split(",") if r.strip()]
    boto3.client("ses", region_name="us-east-1").send_email(
        Source=SES_SOURCE,
        Destination={"ToAddresses": recipients},
        Message={
            "Subject": {"Data": subject, "Charset": "UTF-8"},
            "Body": {"Html": {"Data": html_body, "Charset": "UTF-8"}},
        },
    )
    print(f"  Email sent to {recipients}")


def _publish_sns(subject: str, message: str) -> None:
    if not SNS_TOPIC_ARN:
        return
    boto3.client("sns").publish(
        TopicArn=SNS_TOPIC_ARN, Subject=subject[:100], Message=message
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gameday", default=today_et())
    parser.add_argument("--dry-run", action="store_true", help="Skip email and S3 write")
    args = parser.parse_args()
    gameday = args.gameday

    print(f"NBA Points Pipeline | gameday={gameday}", flush=True)

    model, residuals = load_model_artifacts()

    print(f"Loading spine from S3...", flush=True)
    spine = pd.read_parquet(BytesIO(s3_get(SPINE_KEY)))
    spine_latest = load_spine_latest(spine, gameday)
    print(f"  Spine players: {len(spine_latest):,}")

    print(f"Fetching props from Odds API...", flush=True)
    events = fetch_events(gameday)
    if not events:
        print("No events found — exiting.")
        return

    props = fetch_props(events)
    print(f"  Raw prop rows: {len(props):,}")
    if props.empty:
        print("No props found — exiting.")
        return

    print(f"Building bet rows...", flush=True)
    bet_rows = build_bet_rows(props, spine_latest, gameday)
    if bet_rows.empty:
        print("No scoreable rows — exiting.")
        return
    print(f"  Players with props: {len(bet_rows):,}")

    print(f"Scoring (Strategy S3)...", flush=True)
    scored = score(bet_rows, model, residuals)
    n_bets = scored["bet"].sum()
    print(f"  Bets generated: {n_bets}")

    print("Loading season stats...", flush=True)
    season_stats = load_season_stats(gameday)

    html_body = build_html(scored, scored, gameday, season_stats)
    subject   = f"NBA Points — {int(n_bets)} UNDER bet{'s' if n_bets != 1 else ''} — {gameday}"

    if args.dry_run:
        print(f"  Dry run — skipping S3 write and email")
        print(scored[scored["bet"]][
            ["player_name", "offered_line", "yhat", "mean_adj",
             "p_model_under", "p_market_under", "edge_under"]
        ].to_string(index=False))
        return

    recs_key = f"nba/points_model/daily_runs/{gameday}/recommendations.csv"
    s3_put_csv(recs_key, scored)
    print(f"  Saved → s3://{S3_BUCKET}/{recs_key}")

    send_email(subject, html_body)
    _publish_sns(subject, f"{int(n_bets)} UNDER bets on {gameday}.\n\nStrategy S3: shrinkage=0.25, edge≥5pp, fav_only, all lines.")

    print("Done.")


if __name__ == "__main__":
    main()
