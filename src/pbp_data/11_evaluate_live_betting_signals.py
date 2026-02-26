"""
Evaluate Monte Carlo live betting signals: Brier score, W-L, ROI.

Reads signals parquet from S3 (DuckDB), fetches ESPN final box scores for each
game_id, joins to get actual final points, then computes:
- Brier score (mean squared error of model_prob vs binary outcome)
- Win-loss record
- ROI / profitability (flat $100 per bet)

S3 path: s3://nba-betting-mt/data/04_output/live_betting_signals/player_points/{date}.parquet
Uses DuckDB with httpfs to read parquet from S3 into a pandas DataFrame.
AWS credentials for DuckDB are obtained via `aws configure get aws_access_key_id` and
`aws configure get aws_secret_access_key`.

Usage:
    python src/pbp_data/11_evaluate_live_betting_signals.py 20260223
    python src/pbp_data/11_evaluate_live_betting_signals.py --date 2026-02-23
    python src/pbp_data/11_evaluate_live_betting_signals.py --date all
    python src/pbp_data/11_evaluate_live_betting_signals.py --date all --simulate-manual   # one best per player per 60s, then dedupe
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Add src to path so player_team_history resolves (same as 10_live_betting_signal_generator)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import requests

try:
    requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass

SESSION = requests.Session()
SESSION.verify = False
SESSION.headers.update({"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"})

S3_BUCKET = "nba-betting-mt"
S3_SIGNALS_PREFIX = "data/04_output/live_betting_signals/player_points"
BET_AMOUNT = 100
# Window (seconds) for "one best bet per player per iteration" when --simulate-manual
ITERATION_SECONDS = 60
# Exclude these bookmakers from measurement (e.g. Bovada lines often stale). Set EXCLUDED_BOOKMAKERS env or default bovada.
EXCLUDED_BOOKMAKERS = [
    k.strip().lower() for k in os.environ.get("EXCLUDED_BOOKMAKERS", "bovada").split(",") if k.strip()
]


def _american_to_decimal(american: int) -> float:
    if american >= 100:
        return american / 100.0 + 1.0
    return 100.0 / abs(american) + 1.0


def _get_aws_credentials_for_duckdb():
    """Get AWS credentials via `aws configure get` for DuckDB httpfs. Returns (access_key, secret_key)."""
    def run_configure_get(key: str) -> str:
        out = subprocess.run(
            ["aws", "configure", "get", key],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode != 0 or not out.stdout:
            return ""
        return out.stdout.strip()

    access_key = run_configure_get("aws_access_key_id")
    secret_key = run_configure_get("aws_secret_access_key")
    if not access_key or not secret_key:
        raise RuntimeError(
            "AWS credentials not found. Run: aws configure (and set aws_access_key_id, aws_secret_access_key)."
        )
    return access_key, secret_key


def load_signals_from_s3(date_str: str) -> pd.DataFrame:
    """Load signals parquet from S3 into a pandas DataFrame using DuckDB (httpfs + aws configure credentials)."""
    import duckdb

    s3_path = f"s3://{S3_BUCKET}/{S3_SIGNALS_PREFIX}/{date_str}.parquet"
    con = duckdb.connect(database=":memory:")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET s3_region='us-east-2';")
    access_key, secret_key = _get_aws_credentials_for_duckdb()
    # Escape single quotes for DuckDB string literals (double the quote)
    def esc(s: str) -> str:
        return s.replace("'", "''")

    con.execute(f"SET s3_access_key_id='{esc(access_key)}';")
    con.execute(f"SET s3_secret_access_key='{esc(secret_key)}';")
    df = con.execute(f"SELECT * FROM read_parquet('{s3_path}')").fetchdf()
    con.close()
    return df


def fetch_espn_boxscore_final_points(game_id: str) -> dict:
    """
    Fetch ESPN summary for a completed game; return dict mapping normalized player name -> final points.
    Uses ESPN displayName and normalizes so we can join to signals (which use Odds API–style names).
    """
    from player_team_history.name_normalization import normalize_from_espn_api

    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={game_id}"
    resp = SESSION.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    boxscore = data.get("boxscore", {})
    players_data = boxscore.get("players", [])
    out = {}
    for team_data in players_data:
        stats = team_data.get("statistics", [])
        if not stats:
            continue
        labels = stats[0].get("labels", [])
        try:
            pts_idx = labels.index("PTS")
        except ValueError:
            continue
        for athlete_entry in stats[0].get("athletes", []):
            athlete = athlete_entry.get("athlete", {})
            display_name = athlete.get("displayName", "")
            stats_list = athlete_entry.get("stats", [])
            if pts_idx >= len(stats_list):
                continue
            try:
                pts = int(stats_list[pts_idx])
            except (ValueError, TypeError):
                pts = 0
            normalized = normalize_from_espn_api(display_name)
            if normalized:
                out[normalized] = pts
    return out


def _filter_stale_and_dedupe(df: pd.DataFrame, simulate_manual: bool = False):
    """
    Drop stale lines (over already cleared, or under on a line already over). Optionally simulate manual
    placement: one best bet per player per iteration (by EV), then dedupe to one per (game, player, bookmaker, line, side).
    Returns filtered DataFrame and prints counts dropped.
    """
    df = df.copy()
    start_n = len(df)
    # Stale: OVER when player already had more than line (line already cleared); UNDER when current_points >= live_line (over already hit)
    if "current_points" in df.columns and "live_line" in df.columns and "bet_side" in df.columns:
        stale_over = (df["bet_side"] == "OVER") & (df["current_points"] > df["live_line"])
        stale_under = (df["bet_side"] == "UNDER") & (df["current_points"] >= df["live_line"])
        stale = stale_over | stale_under
        if stale.any():
            df = df[~stale].copy()
            print(f"   Dropped {stale.sum()} stale line(s) (over already cleared or under on decided line)")
    after_stale_n = len(df)
    # Optional: per-iteration best per player (simulate "place best bet per player each scan"), then dedupe
    if simulate_manual and after_stale_n > 0:
        need = ["save_timestamp_utc", "ev", "game_id", "player_name"]
        missing = [k for k in need if k not in df.columns]
        if not missing:
            df["_iteration"] = pd.to_datetime(df["save_timestamp_utc"], utc=True).dt.floor(f"{ITERATION_SECONDS}s")
            idx_best = df.groupby(["_iteration", "game_id", "player_name"])["ev"].idxmax()
            df = df.loc[idx_best].drop(columns=["_iteration"]).copy()
            after_iter_n = len(df)
            print(f"   Per-iteration best per player ({ITERATION_SECONDS}s window): {after_iter_n} decision(s) (from {after_stale_n} after stale)")
        else:
            print(f"   --simulate-manual skipped (missing columns: {missing}); using standard dedupe")
    after_before_dedupe = len(df)
    # Dedupe: same (game, player, bookmaker, line, side) — keep first (earliest)
    dedupe_keys = ["game_id", "player_name", "bookmaker", "live_line", "bet_side"]
    if all(k in df.columns for k in dedupe_keys):
        if "save_timestamp_utc" in df.columns:
            df = df.sort_values("save_timestamp_utc").drop_duplicates(subset=dedupe_keys, keep="first")
        else:
            df = df.drop_duplicates(subset=dedupe_keys, keep="first")
        dedupe_dropped = after_before_dedupe - len(df)
        if dedupe_dropped > 0:
            print(f"   Deduped to one bet per (game, player, bookmaker, line, side): removed {dedupe_dropped} duplicate(s)")
    if len(df) < start_n:
        print(f"   Signals after filters: {len(df)} (from {start_n} raw)")
    return df


def evaluate_signals(df: pd.DataFrame, date_str: str, simulate_manual: bool = False):
    """
    Fetch final points per game, join to signals, compute Brier, W-L, ROI.
    Excludes stale lines; optionally --simulate-manual (one best per player per iteration then dedupe).
    Returns dict with n, wins, total_staked, total_profit, brier_sum for aggregation, or None if nothing evaluated.
    """
    from player_team_history.name_normalization import normalize_from_odds_api

    df = df.copy()
    if "game_id" not in df.columns or "player_name" not in df.columns:
        print("Missing game_id or player_name in signals; aborting.")
        return None
    if "bookmaker" in df.columns and EXCLUDED_BOOKMAKERS:
        before = len(df)
        df = df[~df["bookmaker"].str.lower().isin(EXCLUDED_BOOKMAKERS)].copy()
        dropped = before - len(df)
        if dropped:
            print(f"   Excluded {dropped} signal(s) from bookmaker(s): {', '.join(EXCLUDED_BOOKMAKERS)} (measurement only)")
    df = _filter_stale_and_dedupe(df, simulate_manual=simulate_manual)
    if len(df) == 0:
        print("   No signals left after stale/dedupe filters.")
        return None
    game_ids = df["game_id"].astype(str).unique().tolist()
    game_final_pts = {}
    for gid in game_ids:
        try:
            game_final_pts[gid] = fetch_espn_boxscore_final_points(gid)
        except Exception as e:
            print(f"   Could not fetch box score for game_id={gid}: {e}")
            game_final_pts[gid] = {}
    final_pts_list = []
    for _, row in df.iterrows():
        gid = str(row["game_id"])
        pname = row["player_name"]
        pts_map = game_final_pts.get(gid, {})
        fp = pts_map.get(pname)
        if fp is None and pname:
            fp = pts_map.get(normalize_from_odds_api(pname))
        final_pts_list.append(fp)
    df["final_points"] = final_pts_list
    evaluated = df[df["final_points"].notna()].copy()
    if len(evaluated) == 0:
        print("   No signals could be matched to box score final points; check game_id and player names.")
        return None
    if len(evaluated) < len(df):
        print(f"   Matched {len(evaluated)} of {len(df)} signals to final box scores.")
    evaluated["actual_over"] = evaluated["final_points"] > evaluated["live_line"]
    evaluated["win"] = (
        ((evaluated["bet_side"] == "OVER") & evaluated["actual_over"])
        | ((evaluated["bet_side"] == "UNDER") & ~evaluated["actual_over"])
    )
    evaluated["odds_bet"] = evaluated.apply(
        lambda r: r["over_odds"] if r["bet_side"] == "OVER" else r["under_odds"], axis=1
    )
    evaluated["decimal_bet"] = evaluated["odds_bet"].map(_american_to_decimal)
    evaluated["profit"] = evaluated.apply(
        lambda r: (r["decimal_bet"] - 1) * BET_AMOUNT if r["win"] else -BET_AMOUNT, axis=1
    )
    evaluated["outcome"] = evaluated["win"].astype(int)
    evaluated["brier"] = (evaluated["model_prob"] - evaluated["outcome"]) ** 2
    n = len(evaluated)
    wins = evaluated["win"].sum()
    total_staked = n * BET_AMOUNT
    total_profit = evaluated["profit"].sum()
    roi = total_profit / total_staked if total_staked else 0
    mean_brier = evaluated["brier"].mean()
    brier_sum = evaluated["brier"].sum()
    print()
    print("=" * 60)
    title = f"  EVALUATION: {date_str} — Monte Carlo live signals"
    if simulate_manual:
        title += " (manual sim: one best per player per iteration)"
    print(title)
    print("=" * 60)
    print(f"  Signals evaluated:     {n}")
    print(f"  W–L:                   {int(wins)}–{int(n - wins)}")
    print(f"  Total staked:          ${total_staked:,.0f} (${BET_AMOUNT} × {n})")
    print(f"  Total profit:          ${total_profit:+,.2f}")
    print(f"  ROI:                   {roi:+.1%}")
    print(f"  Mean Brier score:      {mean_brier:.4f} (lower is better)")
    print("=" * 60)
    print()
    print("  Per-signal summary (first 20):")
    cols = ["player_name", "bet_side", "bookmaker", "game_state", "current_points", "live_line", "odds_bet", "final_points", "model_prob", "win", "profit", "brier"]
    subset = [c for c in cols if c in evaluated.columns]
    print(evaluated[subset].head(20).to_string(index=False))
    print()
    return {"n": n, "wins": int(wins), "total_staked": total_staked, "total_profit": total_profit, "brier_sum": brier_sum}


def list_signal_dates_from_s3() -> list:
    """List all YYYYMMDD dates that have a parquet file under the signals prefix. Sorted ascending."""
    import boto3

    prefix = S3_SIGNALS_PREFIX.rstrip("/") + "/"
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    dates = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".parquet"):
                base = key.split("/")[-1]
                date_str = base.replace(".parquet", "")
                if len(date_str) == 8 and date_str.isdigit():
                    dates.append(date_str)
    return sorted(set(dates))


def main():
    parser = argparse.ArgumentParser(description="Evaluate live betting signals (Brier, W-L, ROI)")
    parser.add_argument("date", nargs="?", help="Date YYYYMMDD (e.g. 20260223) or 'all'")
    parser.add_argument("--date", dest="date_alt", help="Date: YYYY-MM-DD, YYYYMMDD, or 'all'")
    parser.add_argument("--simulate-manual", action="store_true", help="One best bet per player per 60s window, then dedupe; can differ from default when some bets were never 'best' in any window")
    args = parser.parse_args()
    simulate_manual = args.simulate_manual
    date_in = args.date or args.date_alt
    if not date_in:
        parser.error("Provide date as positional arg or --date (e.g. 20260223, 2026-02-23, or all)")
    date_in = date_in.strip().lower()
    if date_in == "all":
        date_list = list_signal_dates_from_s3()
        if not date_list:
            print("No signal parquet files found in S3.")
            return
        print(f"Evaluating {len(date_list)} date(s): {date_list[0]} .. {date_list[-1]}" + (" (--simulate-manual)" if simulate_manual else ""))
        agg = {"n": 0, "wins": 0, "total_staked": 0.0, "total_profit": 0.0, "brier_sum": 0.0}
        for date_str in date_list:
            print(f"\n--- {date_str} ---")
            print(f"Loading signals from s3://{S3_BUCKET}/{S3_SIGNALS_PREFIX}/{date_str}.parquet ...")
            df = load_signals_from_s3(date_str)
            print(f"   Loaded {len(df)} signal(s)")
            stats = evaluate_signals(df, date_str, simulate_manual=simulate_manual)
            if stats:
                agg["n"] += stats["n"]
                agg["wins"] += stats["wins"]
                agg["total_staked"] += stats["total_staked"]
                agg["total_profit"] += stats["total_profit"]
                agg["brier_sum"] += stats["brier_sum"]
        if agg["n"] > 0:
            agg_roi = agg["total_profit"] / agg["total_staked"]
            agg_brier = agg["brier_sum"] / agg["n"]
            print()
            print("=" * 60)
            print("  AGGREGATE (all dates)" + (" [manual sim]" if simulate_manual else ""))
            print("=" * 60)
            print(f"  Signals evaluated:     {agg['n']}")
            print(f"  W–L:                   {agg['wins']}–{agg['n'] - agg['wins']}")
            print(f"  Total staked:          ${agg['total_staked']:,.0f} (${BET_AMOUNT} × {agg['n']})")
            print(f"  Total profit:          ${agg['total_profit']:+,.2f}")
            print(f"  ROI:                   {agg_roi:+.1%}")
            print(f"  Mean Brier score:      {agg_brier:.4f} (lower is better)")
            print("=" * 60)
        return
    date_str = date_in.replace("-", "")  # YYYYMMDD
    if len(date_str) != 8 or not date_str.isdigit():
        parser.error("Date must be YYYYMMDD, YYYY-MM-DD, or 'all'")
    print(f"Loading signals from s3://{S3_BUCKET}/{S3_SIGNALS_PREFIX}/{date_str}.parquet ...")
    df = load_signals_from_s3(date_str)
    print(f"   Loaded {len(df)} signal(s)")
    evaluate_signals(df, date_str, simulate_manual=simulate_manual)


if __name__ == "__main__":
    main()
