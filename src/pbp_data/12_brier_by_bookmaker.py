"""
Brier score breakdown by sportsbook for live Monte Carlo signals.

Loads signals parquet from S3, fetches ESPN final box scores, joins actuals,
then prints overall summary + per-bookmaker table (brier, W-L, hit rate, ROI).

S3 path: s3://nba-betting-mt/data/04_output/live_betting_signals/player_points/{date}.parquet

Usage:
    python src/pbp_data/12_brier_by_bookmaker.py 20260520
    python src/pbp_data/12_brier_by_bookmaker.py --date all
    python src/pbp_data/12_brier_by_bookmaker.py --date all --simulate-manual --min-n 10
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

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
ITERATION_SECONDS = 60
EXCLUDED_BOOKMAKERS = [
    k.strip().lower() for k in os.environ.get("EXCLUDED_BOOKMAKERS", "bovada").split(",") if k.strip()
]


def _american_to_decimal(american: int) -> float:
    if american >= 100:
        return american / 100.0 + 1.0
    return 100.0 / abs(american) + 1.0


def _get_aws_credentials_for_duckdb():
    def run_configure_get(key: str) -> str:
        out = subprocess.run(["aws", "configure", "get", key], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else ""

    access_key = run_configure_get("aws_access_key_id")
    secret_key = run_configure_get("aws_secret_access_key")
    if not access_key or not secret_key:
        raise RuntimeError("AWS credentials not found. Run: aws configure")
    return access_key, secret_key


def load_signals_from_s3(date_str: str) -> pd.DataFrame:
    import duckdb

    s3_path = f"s3://{S3_BUCKET}/{S3_SIGNALS_PREFIX}/{date_str}.parquet"
    con = duckdb.connect(database=":memory:")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("SET s3_region='us-east-2';")
    access_key, secret_key = _get_aws_credentials_for_duckdb()

    def esc(s: str) -> str:
        return s.replace("'", "''")

    con.execute(f"SET s3_access_key_id='{esc(access_key)}';")
    con.execute(f"SET s3_secret_access_key='{esc(secret_key)}';")
    df = con.execute(f"SELECT * FROM read_parquet('{s3_path}')").fetchdf()
    con.close()
    return df


def fetch_espn_boxscore_final_points(game_id: str) -> dict:
    from player_team_history.name_normalization import normalize_from_espn_api

    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={game_id}"
    resp = SESSION.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    boxscore = data.get("boxscore", {})
    out = {}
    for team_data in boxscore.get("players", []):
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


def _filter_stale_and_dedupe(df: pd.DataFrame, simulate_manual: bool = False) -> pd.DataFrame:
    df = df.copy()
    start_n = len(df)
    if "current_points" in df.columns and "live_line" in df.columns and "bet_side" in df.columns:
        stale_over = (df["bet_side"] == "OVER") & (df["current_points"] > df["live_line"])
        stale_under = (df["bet_side"] == "UNDER") & (df["current_points"] >= df["live_line"])
        stale = stale_over | stale_under
        if stale.any():
            df = df[~stale].copy()
            print(f"   Dropped {stale.sum()} stale line(s) (already decided)")
    after_stale_n = len(df)
    if simulate_manual and after_stale_n > 0:
        need = ["save_timestamp_utc", "ev", "game_id", "player_name"]
        if not any(k not in df.columns for k in need):
            df["_iteration"] = pd.to_datetime(df["save_timestamp_utc"], utc=True).dt.floor(f"{ITERATION_SECONDS}s")
            idx_best = df.groupby(["_iteration", "game_id", "player_name"])["ev"].idxmax()
            df = df.loc[idx_best].drop(columns=["_iteration"]).copy()
            after_iter_n = len(df)
            print(f"   Per-iteration best per player ({ITERATION_SECONDS}s): {after_iter_n} (from {after_stale_n})")
        else:
            print("   --simulate-manual skipped (missing columns); using standard dedupe")
    after_before_dedupe = len(df)
    dedupe_keys = ["game_id", "player_name", "bookmaker", "live_line", "bet_side"]
    if all(k in df.columns for k in dedupe_keys):
        if "save_timestamp_utc" in df.columns:
            df = df.sort_values("save_timestamp_utc").drop_duplicates(subset=dedupe_keys, keep="first")
        else:
            df = df.drop_duplicates(subset=dedupe_keys, keep="first")
        dropped = after_before_dedupe - len(df)
        if dropped:
            print(f"   Deduped: removed {dropped} duplicate(s)")
    if len(df) < start_n:
        print(f"   Signals after filters: {len(df)} (from {start_n} raw)")
    return df


def _build_bookmaker_table(evaluated: pd.DataFrame, min_n: int) -> pd.DataFrame:
    rows = []
    for bk, grp in evaluated.groupby("bookmaker"):
        n = len(grp)
        wins = int(grp["win"].sum())
        brier = grp["brier"].mean()
        hit_rate = wins / n if n else 0.0
        staked = n * BET_AMOUNT
        profit = grp["profit"].sum()
        roi = profit / staked if staked else 0.0
        rows.append({
            "bookmaker": bk,
            "n": n,
            "W": wins,
            "L": n - wins,
            "hit_rate": hit_rate,
            "brier_score": brier,
            "roi": roi,
            "low_n": n < min_n,
            # keep sums for aggregate
            "_brier_sum": grp["brier"].sum(),
            "_total_staked": staked,
            "_total_profit": profit,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["brier_rank"] = df["brier_score"].rank(ascending=True, method="min").astype("Int64")
    return df.sort_values("brier_score")


def _print_bookmaker_table(bk_df: pd.DataFrame, min_n: int):
    print()
    print("  By bookmaker (sorted by Brier, lower = better calibrated):")
    display = bk_df[["bookmaker", "n", "W", "L", "hit_rate", "brier_score", "roi", "brier_rank"]].copy()
    display["hit_rate"] = display["hit_rate"].map("{:.1%}".format)
    display["brier_score"] = display["brier_score"].map("{:.4f}".format)
    display["roi"] = display["roi"].map("{:+.1%}".format)
    # Flag low-n rows
    display["bookmaker"] = display.apply(
        lambda r: bk_df.loc[r.name, "bookmaker"] + ("*" if bk_df.loc[r.name, "low_n"] else ""), axis=1
    )
    print(display.to_string(index=False))
    if any(bk_df["low_n"]):
        print(f"  * n < {min_n} (low sample, interpret with caution)")


def evaluate_signals(df: pd.DataFrame, date_str: str, simulate_manual: bool = False, min_n: int = 5):
    """
    Returns dict with overall + per-bookmaker aggregation sums for `--date all`, or None if nothing evaluated.
    """
    from player_team_history.name_normalization import normalize_from_odds_api

    df = df.copy()
    if "game_id" not in df.columns or "player_name" not in df.columns:
        print("Missing game_id or player_name; aborting.")
        return None

    if "bookmaker" in df.columns and EXCLUDED_BOOKMAKERS:
        before = len(df)
        df = df[~df["bookmaker"].str.lower().isin(EXCLUDED_BOOKMAKERS)].copy()
        dropped = before - len(df)
        if dropped:
            print(f"   Excluded {dropped} signal(s) from: {', '.join(EXCLUDED_BOOKMAKERS)}")

    stale_col = "stale" if "stale" in df.columns else ("bookmaker_stale" if "bookmaker_stale" in df.columns else None)
    if stale_col is not None:
        is_stale = (df[stale_col] == True) | df[stale_col].isna()
        stale_count = is_stale.sum()
        if stale_count:
            df = df[~is_stale].copy()
            print(f"   Excluded {stale_count} signal(s) (stale at signal time)")

    df = _filter_stale_and_dedupe(df, simulate_manual=simulate_manual)
    if len(df) == 0:
        print("   No signals left after filters.")
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
        print("   No signals matched to box score final points.")
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
    roi = total_profit / total_staked if total_staked else 0.0
    mean_brier = evaluated["brier"].mean()

    print()
    print("=" * 60)
    title = f"  EVALUATION: {date_str} — Brier by bookmaker"
    if simulate_manual:
        title += " (manual sim)"
    print(title)
    print("=" * 60)
    print(f"  Signals evaluated:     {n}")
    print(f"  W–L:                   {int(wins)}–{int(n - wins)}")
    print(f"  Total staked:          ${total_staked:,.0f} (${BET_AMOUNT} × {n})")
    print(f"  Total profit:          ${total_profit:+,.2f}")
    print(f"  ROI:                   {roi:+.1%}")
    print(f"  Mean Brier score:      {mean_brier:.4f} (lower is better)")
    print("=" * 60)

    bk_df = _build_bookmaker_table(evaluated, min_n=min_n)
    _print_bookmaker_table(bk_df, min_n=min_n)
    print()

    # Build per-bookmaker sums for aggregate rollup
    bk_sums = {}
    for _, row in bk_df.iterrows():
        bk_sums[row["bookmaker"]] = {
            "n": int(row["n"]),
            "wins": int(row["W"]),
            "brier_sum": float(row["_brier_sum"]),
            "total_staked": float(row["_total_staked"]),
            "total_profit": float(row["_total_profit"]),
        }

    return {
        "n": n,
        "wins": int(wins),
        "total_staked": total_staked,
        "total_profit": total_profit,
        "brier_sum": evaluated["brier"].sum(),
        "bookmaker_sums": bk_sums,
    }


def list_signal_dates_from_s3() -> list:
    import boto3

    prefix = S3_SIGNALS_PREFIX.rstrip("/") + "/"
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    dates = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".parquet"):
                base = key.split("/")[-1].replace(".parquet", "")
                if len(base) == 8 and base.isdigit():
                    dates.append(base)
    return sorted(set(dates))


def _print_aggregate_bookmaker_table(agg_bk: dict, min_n: int):
    rows = []
    for bk, sums in agg_bk.items():
        n = sums["n"]
        wins = sums["wins"]
        brier = sums["brier_sum"] / n if n else float("nan")
        hit_rate = wins / n if n else 0.0
        staked = sums["total_staked"]
        profit = sums["total_profit"]
        roi = profit / staked if staked else 0.0
        rows.append({
            "bookmaker": bk,
            "n": n,
            "W": wins,
            "L": n - wins,
            "hit_rate": hit_rate,
            "brier_score": brier,
            "roi": roi,
            "low_n": n < min_n,
        })
    df = pd.DataFrame(rows).sort_values("brier_score")
    df["brier_rank"] = df["brier_score"].rank(ascending=True, method="min").astype("Int64")

    display = df[["bookmaker", "n", "W", "L", "hit_rate", "brier_score", "roi", "brier_rank"]].copy()
    display["hit_rate"] = display["hit_rate"].map("{:.1%}".format)
    display["brier_score"] = display["brier_score"].map("{:.4f}".format)
    display["roi"] = display["roi"].map("{:+.1%}".format)
    display["bookmaker"] = display.apply(
        lambda r: df.loc[r.name, "bookmaker"] + ("*" if df.loc[r.name, "low_n"] else ""), axis=1
    )
    print()
    print("  By bookmaker — aggregate (sorted by Brier):")
    print(display.to_string(index=False))
    if any(df["low_n"]):
        print(f"  * n < {min_n} (low sample)")


def main():
    parser = argparse.ArgumentParser(description="Brier score by sportsbook for live MC signals")
    parser.add_argument("date", nargs="?", help="Date YYYYMMDD or 'all'")
    parser.add_argument("--date", dest="date_alt", help="Date: YYYY-MM-DD, YYYYMMDD, or 'all'")
    parser.add_argument("--simulate-manual", action="store_true", help="One best bet per player per 60s, then dedupe")
    parser.add_argument("--min-n", type=int, default=5, help="Min signals per bookmaker to rank (default 5)")
    args = parser.parse_args()

    date_in = args.date or args.date_alt
    if not date_in:
        parser.error("Provide date as positional arg or --date (e.g. 20260520 or all)")
    date_in = date_in.strip().lower()
    simulate_manual = args.simulate_manual
    min_n = args.min_n

    if date_in == "all":
        date_list = list_signal_dates_from_s3()
        if not date_list:
            print("No signal parquet files found in S3.")
            return
        print(f"Evaluating {len(date_list)} date(s): {date_list[0]} .. {date_list[-1]}" + (" (--simulate-manual)" if simulate_manual else ""))
        agg = {"n": 0, "wins": 0, "total_staked": 0.0, "total_profit": 0.0, "brier_sum": 0.0}
        agg_bk: dict = {}
        for date_str in date_list:
            print(f"\n--- {date_str} ---")
            print(f"Loading s3://{S3_BUCKET}/{S3_SIGNALS_PREFIX}/{date_str}.parquet ...")
            df = load_signals_from_s3(date_str)
            print(f"   Loaded {len(df)} signal(s)")
            stats = evaluate_signals(df, date_str, simulate_manual=simulate_manual, min_n=min_n)
            if not stats:
                continue
            agg["n"] += stats["n"]
            agg["wins"] += stats["wins"]
            agg["total_staked"] += stats["total_staked"]
            agg["total_profit"] += stats["total_profit"]
            agg["brier_sum"] += stats["brier_sum"]
            for bk, sums in stats["bookmaker_sums"].items():
                if bk not in agg_bk:
                    agg_bk[bk] = {"n": 0, "wins": 0, "brier_sum": 0.0, "total_staked": 0.0, "total_profit": 0.0}
                agg_bk[bk]["n"] += sums["n"]
                agg_bk[bk]["wins"] += sums["wins"]
                agg_bk[bk]["brier_sum"] += sums["brier_sum"]
                agg_bk[bk]["total_staked"] += sums["total_staked"]
                agg_bk[bk]["total_profit"] += sums["total_profit"]

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
            _print_aggregate_bookmaker_table(agg_bk, min_n=min_n)
            print("=" * 60)
        return

    date_str = date_in.replace("-", "")
    if len(date_str) != 8 or not date_str.isdigit():
        parser.error("Date must be YYYYMMDD, YYYY-MM-DD, or 'all'")
    print(f"Loading s3://{S3_BUCKET}/{S3_SIGNALS_PREFIX}/{date_str}.parquet ...")
    df = load_signals_from_s3(date_str)
    print(f"   Loaded {len(df)} signal(s)")
    evaluate_signals(df, date_str, simulate_manual=simulate_manual, min_n=min_n)


if __name__ == "__main__":
    main()
