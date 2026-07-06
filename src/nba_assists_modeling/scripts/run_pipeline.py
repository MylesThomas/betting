"""
Live gameday pipeline for NBA player assists props.

For each player with a player_assists prop on the given gameday:
  1. Fetches live events + props from The Odds API
  2. Joins with rolling features from the spine (S3)
  3. Scores with OLS + Normal CDF: P(over) = 1 - CDF((line - yhat) / resid_std)
  4. Computes no-vig market P(over) from posted odds
  5. Filters to edge >= 0.10 (show all); highlights edge >= 0.15 (primary)
  6. Sends SES HTML email + SNS notification
  7. Saves recommendations CSV to S3

Strategy: OVER only · edge >= 0.15 primary · edge >= 0.10 backup
Books:    BetMGM and William Hill are primary value books

S3 paths read:
  s3://the-odds-api-mt/nba/assists_model/spine/nba_assists_spine.parquet

S3 paths written:
  s3://the-odds-api-mt/nba/assists_model/daily_runs/{gameday}/recommendations.csv

Run:
    python src/nba_assists_modeling/scripts/run_pipeline.py
    python src/nba_assists_modeling/scripts/run_pipeline.py --gameday 2026-10-29
"""
from __future__ import annotations

import argparse
import html as html_module
import os
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
from scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

ET = ZoneInfo("America/New_York")

ODDS_API_KEY  = os.environ.get("ODDS_API_KEY", "").strip()
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT         = "basketball_nba"
MARKET        = "player_assists"
REGIONS       = "us"
SLEEP_S       = 0.25

S3_BUCKET    = "the-odds-api-mt"
S3_PREFIX    = "nba/assists_model"
SPINE_KEY    = f"{S3_PREFIX}/spine/nba_assists_spine.parquet"
SETTLED_KEY  = f"{S3_PREFIX}/settled/settled_bets.parquet"

SES_SOURCE   = os.environ.get("SETTLEMENT_SES_SOURCE", "").strip()
SES_TO_RAW   = os.environ.get("SETTLEMENT_SES_TO", "mylescgthomas@gmail.com").strip()
SNS_TOPIC_ARN = os.environ.get("SNS_TOPIC_ARN", "").strip()

# ── OLS model (inline — stable, no pickle dependency) ────────────────────────
OLS_COEF = {
    "const":       0.2876,
    "min_line":    0.4478,
    "max_line":    0.3403,
    "ast_roll_20": 0.1833,
}
RESID_STD             = 2.1153
EDGE_THRESHOLD_PRIMARY = 0.15   # highlighted in email
EDGE_THRESHOLD_SHOW    = 0.10   # also shown (backup context)
MIN_BOOKS             = 1

_MONO = "ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,monospace"
_SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif"


# ── Helpers ───────────────────────────────────────────────────────────────────

def today_et() -> str:
    return datetime.now(ET).strftime("%Y-%m-%d")


def current_nba_season() -> str:
    now = datetime.now(ET)
    yr  = now.year if now.month >= 10 else now.year - 1
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


def no_vig_prob(over_odds: float, under_odds: float) -> float:
    """Strip vig, return P(over)."""
    p_o = 1 / (1 + american_profit(over_odds))
    p_u = 1 / (1 + american_profit(under_odds))
    return p_o / (p_o + p_u)


def fmt_odds(price) -> str:
    return "—" if (price is None or pd.isna(price)) else f"{int(price):+d}"


# ── S3 ────────────────────────────────────────────────────────────────────────

def _s3():
    return boto3.client("s3")


def s3_get_parquet(key: str) -> pd.DataFrame:
    body = _s3().get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
    return pd.read_parquet(BytesIO(body))


def s3_get_bytes(key: str) -> bytes:
    return _s3().get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()


def s3_put_csv(key: str, df: pd.DataFrame) -> None:
    buf = BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    _s3().put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue())


# ── Odds API ──────────────────────────────────────────────────────────────────

def _api_get(path: str, params: dict) -> dict:
    if not ODDS_API_KEY:
        raise RuntimeError("ODDS_API_KEY not set")
    resp = requests.get(f"{ODDS_API_BASE}{path}", params={**params, "apiKey": ODDS_API_KEY}, timeout=20)
    resp.raise_for_status()
    return resp.json()


def fetch_events(gameday: str) -> list[dict]:
    data = _api_get("/sports/{sport}/events".format(sport=SPORT), {
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
                        "event_id":   ev["id"],
                        "game_date":  gameday,
                        "home_team":  ev.get("home_team", ""),
                        "away_team":  ev.get("away_team", ""),
                        "bookmaker":  bk["key"],
                        "player":     outcome["description"],
                        "side":       outcome["name"].lower(),   # "Over" / "Under"
                        "prop_line":  float(outcome["point"]),
                        "odds":       float(outcome["price"]),
                    })
    return pd.DataFrame(rows)


# ── Feature assembly ──────────────────────────────────────────────────────────

def load_spine_latest(spine: pd.DataFrame) -> pd.DataFrame:
    """Return the most recent row per player (latest ast_roll_20 available)."""
    spine = spine.sort_values("game_date")
    latest = (
        spine.dropna(subset=["ast_roll_20"])
        .groupby("player_key", as_index=False)
        .last()
    )
    return latest[["player_key", "game_date", "ast_roll_20"]].rename(
        columns={"game_date": "last_game_date"}
    )


def build_bet_rows(props: pd.DataFrame, spine_latest: pd.DataFrame) -> pd.DataFrame:
    """One row per player-game (consensus across books)."""
    if props.empty:
        return pd.DataFrame()

    props["player_key"] = props["player"].apply(_normalize_name)

    over_props  = props[props["side"] == "over"].copy()
    under_props = props[props["side"] == "under"].copy()

    # Consensus line per player: median across books
    line_agg = (
        over_props.groupby("player_key")
        .agg(
            consensus_line = ("prop_line", "median"),
            min_line       = ("prop_line", "min"),
            max_line       = ("prop_line", "max"),
            n_books        = ("bookmaker", "nunique"),
            game_date      = ("game_date", "first"),
            home_team      = ("home_team", "first"),
            away_team      = ("away_team", "first"),
        )
        .reset_index()
    )

    # Best over/under odds per player (highest profit for over, highest profit for under)
    best_over = (
        over_props.sort_values("odds", ascending=False)
        .groupby("player_key")
        .agg(best_over_odds=("odds", "first"), best_over_book=("bookmaker", "first"))
        .reset_index()
    )
    over_props  = over_props.copy()
    under_props = under_props.copy()
    over_props["_profit"]  = over_props["odds"].apply(american_profit)
    under_props["_profit"] = under_props["odds"].apply(american_profit)
    avg_over = (
        over_props.groupby("player_key")
        .agg(avg_over_profit=("_profit", "mean"))
        .reset_index()
    )
    avg_under = (
        under_props.groupby("player_key")
        .agg(avg_under_profit=("_profit", "mean"))
        .reset_index()
    )

    df = line_agg.merge(best_over, on="player_key", how="left")
    df = df.merge(avg_over,  on="player_key", how="left")
    df = df.merge(avg_under, on="player_key", how="left")
    df = df.merge(spine_latest, on="player_key", how="left")
    df["player"]       = df["player_key"]
    df["season"]       = current_nba_season()

    return df


# ── Scoring ───────────────────────────────────────────────────────────────────

def score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c  = OLS_COEF
    df["yhat"] = (
        c["const"]
        + c["min_line"]    * df["min_line"]
        + c["max_line"]    * df["max_line"]
        + c["ast_roll_20"] * df["ast_roll_20"].fillna(df["consensus_line"] * 0.95)
    )
    df["p_over_model"] = df.apply(
        lambda r: float(1 - norm.cdf((r["consensus_line"] - r["yhat"]) / RESID_STD)),
        axis=1,
    )
    # Market probability (no-vig, using avg profit — never arithmetically average American odds)
    def mkt_prob(row):
        op = row.get("avg_over_profit")
        up = row.get("avg_under_profit")
        if pd.notna(op) and pd.notna(up) and op > 0 and up > 0:
            p_o = 1 / (1 + op)
            p_u = 1 / (1 + up)
            return p_o / (p_o + p_u)
        return 0.5   # no odds → neutral
    df["p_over_market"] = df.apply(mkt_prob, axis=1)
    df["edge"]          = df["p_over_model"] - df["p_over_market"]
    df["is_primary"]    = df["edge"] >= EDGE_THRESHOLD_PRIMARY
    return df


def filter_bets(df: pd.DataFrame) -> pd.DataFrame:
    mask = (
        (df["edge"] >= EDGE_THRESHOLD_SHOW) &
        df["ast_roll_20"].notna() &
        (df["n_books"] >= MIN_BOOKS)
    )
    return df[mask].sort_values("edge", ascending=False).copy()


# ── Season stats ─────────────────────────────────────────────────────────────

def load_season_stats(gameday: str) -> dict:
    season = current_nba_season()
    try:
        df = pd.read_parquet(BytesIO(s3_get_bytes(SETTLED_KEY)))
        df = df[df.get("season", pd.Series("", index=df.index)) == season] if "season" in df.columns else df
        if df.empty:
            return {}
        outcome_col = "outcome" if "outcome" in df.columns else "result"
        wins    = (df[outcome_col] == "WIN").sum()
        losses  = (df[outcome_col] == "LOSS").sum()
        pushes  = (df[outcome_col] == "PUSH").sum()
        pnl_col = "pnl" if "pnl" in df.columns else "pnl_units"
        units   = df[pnl_col].sum() if pnl_col in df.columns else 0.0
        decided = wins + losses
        return {
            "season": season, "wins": int(wins), "losses": int(losses),
            "pushes": int(pushes), "units": float(units),
            "win_pct": wins / decided if decided > 0 else 0.0,
            "roi": units / decided if decided > 0 else 0.0,
        }
    except Exception as e:
        print(f"  Could not load season stats: {e}")
        return {}


# ── HTML email ────────────────────────────────────────────────────────────────

def _feat_cell(label: str, warn: bool = False) -> str:
    bg = "background:#fee2e2;" if warn else ""
    return f'<td style="padding:6px 10px;text-align:center;font-family:{_MONO};font-size:12px;{bg}">{label}</td>'


def build_html(bets: pd.DataFrame, gameday: str, n_scored: int, season_stats: dict | None = None) -> str:
    now_str = datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")
    primary = bets[bets["is_primary"]]
    backup  = bets[~bets["is_primary"]]

    def bet_rows(subset: pd.DataFrame, tier: str) -> str:
        if subset.empty:
            return f'<tr><td colspan="12" style="padding:10px;color:#6b7280;font-style:italic;">No {tier} bets today</td></tr>'
        html_rows = []
        for _, row in subset.iterrows():
            roll_warn = pd.isna(row.get("ast_roll_20"))
            html_rows.append(f"""
            <tr>
              <td style="padding:6px 10px;font-weight:600;">{html_module.escape(str(row['player']))}</td>
              <td style="padding:6px 10px;text-align:center;">{row['consensus_line']:.1f}</td>
              <td style="padding:6px 10px;text-align:center;font-weight:bold;color:#1d4ed8;">OVER</td>
              <td style="padding:6px 10px;text-align:center;">{fmt_odds(row.get('best_over_odds'))}</td>
              <td style="padding:6px 10px;text-align:center;font-size:11px;">{html_module.escape(str(row.get('best_over_book','—')))}</td>
              {_feat_cell(f"{row['p_over_model']*100:.1f}%")}
              {_feat_cell(f"{row['p_over_market']*100:.1f}%")}
              {_feat_cell(f"+{row['edge']*100:.1f}pp")}
              {_feat_cell(f"{row['yhat']:.2f}")}
              {_feat_cell("—" if pd.isna(row.get('ast_roll_20')) else f"{row['ast_roll_20']:.2f}", warn=roll_warn)}
              {_feat_cell(f"{row['min_line']:.1f} / {row['max_line']:.1f}")}
              <td style="padding:6px 10px;text-align:center;font-size:11px;">{int(row['n_books'])}</td>
            </tr>""")
        return "\n".join(html_rows)

    header = """
    <tr style="background:#1e3a5f;">
      <th style="padding:8px 10px;text-align:left;color:#93c5fd;">Player</th>
      <th style="padding:8px 10px;text-align:center;color:#93c5fd;">Line</th>
      <th style="padding:8px 10px;text-align:center;color:#93c5fd;">Side</th>
      <th style="padding:8px 10px;text-align:center;color:#93c5fd;">Best Odds</th>
      <th style="padding:8px 10px;text-align:center;color:#93c5fd;">Book</th>
      <th style="padding:8px 10px;text-align:center;color:#93c5fd;">P(over) model</th>
      <th style="padding:8px 10px;text-align:center;color:#93c5fd;">P(over) mkt</th>
      <th style="padding:8px 10px;text-align:center;color:#93c5fd;">Edge</th>
      <th style="padding:8px 10px;text-align:center;color:#93c5fd;">yhat</th>
      <th style="padding:8px 10px;text-align:center;color:#93c5fd;">ast_roll_20</th>
      <th style="padding:8px 10px;text-align:center;color:#93c5fd;">min/max line</th>
      <th style="padding:8px 10px;text-align:center;color:#93c5fd;">#Books</th>
    </tr>"""

    if season_stats:
        ss = season_stats
        u_color = "#4ade80" if ss["units"] >= 0 else "#f87171"
        season_html = f"""
<div style="display:flex;gap:12px;margin:12px 0 20px;flex-wrap:wrap;">
  <div style="background:#1a1f2e;border:1px solid #2d3748;border-radius:6px;padding:10px 16px;min-width:100px;">
    <div style="font-size:10px;color:#6b7280;text-transform:uppercase;">{ss['season']} PnL</div>
    <div style="font-size:20px;font-weight:700;color:{u_color};">{ss['units']:+.2f}u</div>
  </div>
  <div style="background:#1a1f2e;border:1px solid #2d3748;border-radius:6px;padding:10px 16px;min-width:100px;">
    <div style="font-size:10px;color:#6b7280;text-transform:uppercase;">Record</div>
    <div style="font-size:16px;font-weight:700;">{ss['wins']}W–{ss['losses']}L{f"–{ss['pushes']}P" if ss["pushes"] else ""}</div>
  </div>
  <div style="background:#1a1f2e;border:1px solid #2d3748;border-radius:6px;padding:10px 16px;min-width:100px;">
    <div style="font-size:10px;color:#6b7280;text-transform:uppercase;">Win %</div>
    <div style="font-size:16px;font-weight:700;">{ss['win_pct']*100:.1f}%</div>
  </div>
  <div style="background:#1a1f2e;border:1px solid #2d3748;border-radius:6px;padding:10px 16px;min-width:100px;">
    <div style="font-size:10px;color:#6b7280;text-transform:uppercase;">ROI</div>
    <div style="font-size:16px;font-weight:700;color:{u_color};">{ss['roi']*100:+.1f}%</div>
  </div>
</div>"""
    else:
        season_html = ""

    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"/></head>
<body style="font-family:{_SANS};background:#0f1117;color:#e2e8f0;margin:0;padding:24px;">
<h2 style="color:#93c5fd;margin-bottom:4px;">NBA Assists Props — {gameday}</h2>
<p style="color:#6b7280;font-size:13px;margin-top:0;">{now_str} · {n_scored} players scored · {len(bets)} qualifying bets</p>
{season_html}
<h3 style="color:#4ade80;margin-bottom:6px;">Primary Bets (edge ≥ {EDGE_THRESHOLD_PRIMARY*100:.0f}pp)</h3>
<p style="color:#9ca3af;font-size:12px;margin-top:0;">
  Strategy: OVER · OLS + Normal CDF · BetMGM/WilliamHill are primary value books.<br>
  avg_implied_prob ≈ 40% at edge≥15pp — breakeven win rate 39.7%; model hits ~42%.
</p>
<table style="border-collapse:collapse;width:100%;font-size:13px;margin-bottom:24px;">
  {header}
  {bet_rows(primary, "primary")}
</table>

<h3 style="color:#fbbf24;margin-bottom:6px;">Backup Bets (edge {EDGE_THRESHOLD_SHOW*100:.0f}–{EDGE_THRESHOLD_PRIMARY*100:.0f}pp)</h3>
<p style="color:#9ca3af;font-size:12px;margin-top:0;">Thinner edge — use as context only.</p>
<table style="border-collapse:collapse;width:100%;font-size:13px;margin-bottom:24px;">
  {header}
  {bet_rows(backup, "backup")}
</table>

<hr style="border-color:#2d3748;margin:20px 0;"/>
<p style="font-size:11px;color:#6b7280;">
  Model: OLS [min_line={OLS_COEF['min_line']}, max_line={OLS_COEF['max_line']},
  ast_roll_20={OLS_COEF['ast_roll_20']}] · resid_std={RESID_STD}<br>
  Out-of-sample (2 folds): edge≥15pp OVER → 1,188 bets · 2.44% ROI · +29.0u · max_dd=27.7u<br>
  Research log: knowledge-base/raw/20260702-nba-player-assists.html
</p>
</body></html>"""


# ── Send / notify ─────────────────────────────────────────────────────────────

def send_email(subject: str, html_body: str) -> None:
    to_list = [a.strip() for a in SES_TO_RAW.split(",") if a.strip()]
    if not SES_SOURCE or not to_list:
        print(f"  SES not configured (SES_SOURCE={SES_SOURCE!r}), skipping email")
        return
    try:
        boto3.client("ses", region_name="us-east-2").send_email(
            Source=SES_SOURCE,
            Destination={"ToAddresses": to_list},
            Message={
                "Subject": {"Data": subject, "Charset": "UTF-8"},
                "Body": {
                    "Html": {"Data": html_body, "Charset": "UTF-8"},
                    "Text": {"Data": subject, "Charset": "UTF-8"},
                },
            },
        )
        print(f"  Email sent: {subject}")
    except Exception as e:
        print(f"  Email failed: {e}")


def publish_sns(subject: str, message: str) -> None:
    if not SNS_TOPIC_ARN:
        return
    boto3.client("sns").publish(TopicArn=SNS_TOPIC_ARN, Subject=subject[:100], Message=message)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gameday", default=today_et())
    args = parser.parse_args()
    gameday = args.gameday

    print(f"\nNBA Assists Pipeline | gameday={gameday}", flush=True)

    # 1. Load spine
    print("Loading spine from S3...", flush=True)
    spine = s3_get_parquet(SPINE_KEY)
    spine_latest = load_spine_latest(spine)
    print(f"  Spine players: {len(spine_latest):,}")

    # 2. Fetch today's events + props
    print("Fetching events...", flush=True)
    events = fetch_events(gameday)
    if not events:
        msg = f"No NBA events found for {gameday}"
        print(msg)
        send_email(f"NBA Assists — {gameday} — No games", f"<p>{msg}</p>")
        return

    print("Fetching props...", flush=True)
    props = fetch_props(events)
    print(f"  Raw prop rows: {len(props):,}")

    if props.empty:
        msg = f"No {MARKET} props found for {gameday}"
        print(msg)
        send_email(f"NBA Assists — {gameday} — No props", f"<p>{msg}</p>")
        return

    # 3. Build rows + score
    df  = build_bet_rows(props, spine_latest)
    df  = score(df)
    n_scored = len(df)
    print(f"  Players scored: {n_scored}")

    bets = filter_bets(df)
    print(f"  Qualifying bets (edge>={EDGE_THRESHOLD_SHOW}): {len(bets)}")
    print(f"  Primary bets   (edge>={EDGE_THRESHOLD_PRIMARY}): {bets['is_primary'].sum()}")

    # 4. Save to S3
    rec_key = f"{S3_PREFIX}/daily_runs/{gameday}/recommendations.csv"
    if not bets.empty:
        save_cols = [
            "player", "player_key", "game_date", "season",
            "consensus_line", "min_line", "max_line", "n_books",
            "best_over_odds", "best_over_book", "avg_over_profit", "avg_under_profit",
            "ast_roll_20", "yhat", "p_over_model", "p_over_market", "edge", "is_primary",
        ]
        s3_put_csv(rec_key, bets[[c for c in save_cols if c in bets.columns]])
        print(f"  Saved → s3://{S3_BUCKET}/{rec_key}")
    else:
        print("  No qualifying bets — skipping S3 save")

    # 5. Email
    print("Loading season stats...", flush=True)
    season_stats = load_season_stats(gameday)

    n_primary = int(bets["is_primary"].sum()) if not bets.empty else 0
    subject   = f"NBA Assists {gameday} — {n_primary} primary · {len(bets)} total bets"
    html_body = build_html(bets, gameday, n_scored, season_stats)
    send_email(subject, html_body)
    publish_sns(subject, f"{n_primary} primary bets (edge>={EDGE_THRESHOLD_PRIMARY*100:.0f}pp). See email.")

    print(f"\nDone. {n_primary} primary bets, {len(bets)} total.")


if __name__ == "__main__":
    main()
