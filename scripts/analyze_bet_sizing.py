"""
Bet sizing analysis: streak distribution, Kelly, and martingale viability per strategy.

Strategies analyzed:
  both           — both models agree (most selective)
  ols            — OLS-only plays
  xgb            — XGB-only plays
  both_high_edge — both bucket filtered to mean edge >= --high-edge-threshold

Sizing simulations run on the chronological bet sequence for each strategy:
  flat      — 1u per bet
  kelly025  — 0.25x fractional Kelly, sized to current bankroll
  martingale — 1u base, double on loss, reset on win; bust = can't fund next double

NOTE: same-day bets are ordered alphabetically by player (deterministic but arbitrary).
      With ~5 bets/day on average, observed streak stats have wide CIs — interpret
      directionally.

Usage:
  python scripts/analyze_bet_sizing.py
  python scripts/analyze_bet_sizing.py --since 2026-04-01 --bankroll 100
  python scripts/analyze_bet_sizing.py --high-edge-threshold 0.20
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd


BUCKET = "nba-betting-mt"
RUNS_PREFIX = "rebounds/daily_runs"
DEFAULT_BANKROLL = 100.0
DEFAULT_BASE_UNIT = 1.0
DEFAULT_KELLY_MULT = 0.25
DEFAULT_HIGH_EDGE = 0.15


def ensure_repo_root_on_syspath() -> Path:
    current = Path(__file__).resolve().parent
    while True:
        if (current / ".gitignore").exists() and (current / "src").exists():
            if str(current) not in sys.path:
                sys.path.insert(0, str(current))
            return current
        if current.parent == current:
            raise FileNotFoundError("Could not locate repo root")
        current = current.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bet sizing and martingale viability analysis.")
    p.add_argument("--since", type=str, default="", help="Only include dates >= YYYY-MM-DD.")
    p.add_argument("--bankroll", type=float, default=DEFAULT_BANKROLL, help="Starting bankroll in units (default: 100).")
    p.add_argument("--base-unit", type=float, default=DEFAULT_BASE_UNIT, help="Martingale base bet in units (default: 1).")
    p.add_argument("--kelly-mult", type=float, default=DEFAULT_KELLY_MULT, help="Kelly multiplier (default: 0.25).")
    p.add_argument("--high-edge-threshold", type=float, default=DEFAULT_HIGH_EDGE, help="Min mean edge for both_high_edge strategy (default: 0.15).")
    p.add_argument("--bucket", type=str, default=BUCKET)
    p.add_argument("--runs-prefix", type=str, default=RUNS_PREFIX)
    return p.parse_args()


# ---------------------------------------------------------------------------
# S3 helpers (same pattern as analyze_settled_results.py)
# ---------------------------------------------------------------------------

def _s3():
    import boto3
    return boto3.client("s3", region_name="us-east-2")


def read_parquet_s3(bucket: str, key: str) -> pd.DataFrame:
    body = _s3().get_object(Bucket=bucket, Key=key)["Body"].read()
    return pd.read_parquet(BytesIO(body))


def list_settled_keys(bucket: str, runs_prefix: str) -> list[str]:
    paginator = _s3().get_paginator("list_objects_v2")
    all_keys: list[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=runs_prefix.rstrip("/") + "/"):
        for item in page.get("Contents", []):
            key = item["Key"]
            if (
                "rebounds_scored_settled_" in key
                and key.endswith(".parquet")
                and "_rollups/" not in key
            ):
                all_keys.append(key)

    latest: dict[str, tuple[str, str]] = {}
    for key in all_keys:
        parts = key.split("/")
        if len(parts) < 4:
            continue
        date_part = parts[-3]
        run_id = parts[-2]
        if date_part not in latest or run_id > latest[date_part][0]:
            latest[date_part] = (run_id, key)
    return sorted(v[1] for v in latest.values())


def load_settled_data(bucket: str, runs_prefix: str, since: str) -> pd.DataFrame:
    keys = list_settled_keys(bucket, runs_prefix)
    frames: list[pd.DataFrame] = []
    print(f"  Loading {len(keys)} settled parquet files...", flush=True)
    for key in keys:
        frames.append(read_parquet_s3(bucket, key))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    if since:
        df = df[df["date"] >= pd.Timestamp(since)]
    return df


# ---------------------------------------------------------------------------
# Strategy definitions
# ---------------------------------------------------------------------------

def build_strategies(df: pd.DataFrame, high_edge_thresh: float) -> dict[str, pd.DataFrame]:
    bets = df[df["is_bet"].astype(bool) & df["result"].isin(["win", "loss", "push"])].copy()

    def _add_p_model(sub: pd.DataFrame, bucket: str) -> pd.DataFrame:
        sub = sub.copy()
        if bucket in ("both", "both_high_edge"):
            sub["p_model"] = (sub["p_under_ols"] + sub["p_under_xgb"]) / 2
            sub["edge_model"] = (sub["edge_under_ols"] + sub["edge_under_xgb"]) / 2
        elif bucket == "ols":
            sub["p_model"] = sub["p_under_ols"]
            sub["edge_model"] = sub["edge_under_ols"]
        else:  # xgb
            sub["p_model"] = sub["p_under_xgb"]
            sub["edge_model"] = sub["edge_under_xgb"]
        return sub

    both = _add_p_model(bets[bets["strategy_bucket"] == "both"], "both")
    ols = _add_p_model(bets[bets["strategy_bucket"] == "ols"], "ols")
    xgb = _add_p_model(bets[bets["strategy_bucket"] == "xgb"], "xgb")
    both_hi = _add_p_model(
        bets[(bets["strategy_bucket"] == "both") & ((bets["edge_under_ols"] + bets["edge_under_xgb"]) / 2 >= high_edge_thresh)],
        "both_high_edge",
    )

    strategies: dict[str, pd.DataFrame] = {}
    for name, sub in [("both", both), ("ols", ols), ("xgb", xgb), (f"both_edge≥{high_edge_thresh:.0%}", both_hi)]:
        if len(sub) == 0:
            continue
        # sort chronologically, then by player for same-day determinism
        strategies[name] = sub.sort_values(["date", "player_normalized"]).reset_index(drop=True)
    return strategies


# ---------------------------------------------------------------------------
# Kelly
# ---------------------------------------------------------------------------

def kelly_fraction(p: float, american_odds: float) -> float:
    if american_odds < 0:
        b = 100.0 / abs(american_odds)
    else:
        b = american_odds / 100.0
    q = 1.0 - p
    f = (b * p - q) / b
    return max(0.0, f)


# ---------------------------------------------------------------------------
# Streak analysis
# ---------------------------------------------------------------------------

def losing_streaks(outcomes: list[str]) -> list[int]:
    streaks: list[int] = []
    current = 0
    for o in outcomes:
        if o == "loss":
            current += 1
        elif o == "win":
            if current > 0:
                streaks.append(current)
            current = 0
        # push: neutral, don't break or extend
    if current > 0:
        streaks.append(current)
    return streaks


def streak_stats(outcomes: list[str]) -> dict:
    streaks = losing_streaks(outcomes)
    n_win = outcomes.count("win")
    n_loss = outcomes.count("loss")
    win_rate = n_win / (n_win + n_loss) if (n_win + n_loss) > 0 else 0.0

    if not streaks:
        return {
            "max": 0, "mean": 0.0, "p95": 0,
            "win_rate": win_rate, "n_streaks": 0, "distribution": {},
        }

    dist: dict[int, int] = {}
    for s in streaks:
        dist[s] = dist.get(s, 0) + 1

    return {
        "max": max(streaks),
        "mean": float(np.mean(streaks)),
        "p95": int(np.percentile(streaks, 95)) if len(streaks) >= 5 else max(streaks),
        "win_rate": win_rate,
        "n_streaks": len(streaks),
        "distribution": dict(sorted(dist.items())),
    }


def bankroll_needed_to_survive(max_streak: int, base_unit: float) -> float:
    """Minimum bankroll so you can fund all doubles through max_streak losses."""
    # Total outlay: base * (2^0 + 2^1 + ... + 2^(n-1)) = base * (2^n - 1)
    return base_unit * (2 ** max_streak - 1)


# ---------------------------------------------------------------------------
# Sizing simulations
# ---------------------------------------------------------------------------

def simulate_flat(bets: pd.DataFrame, base_unit: float, bankroll: float) -> dict:
    b = bankroll
    peak = bankroll
    max_dd = 0.0
    curve = [bankroll]
    for pnl in bets["pnl_units"]:
        b += base_unit * pnl
        peak = max(peak, b)
        dd = (peak - b) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
        curve.append(b)
    return {
        "final": b,
        "total_return": (b - bankroll) / bankroll,
        "max_drawdown": max_dd,
        "curve": curve,
    }


def simulate_kelly(bets: pd.DataFrame, kelly_mult: float, bankroll: float) -> dict:
    b = bankroll
    peak = bankroll
    max_dd = 0.0
    curve = [bankroll]
    for _, row in bets.iterrows():
        f = kelly_fraction(float(row["p_model"]), float(row["under_odds"]))
        bet_size = kelly_mult * f * b
        bet_size = min(bet_size, 0.20 * b)  # cap at 20% bankroll
        bet_size = max(0.0, bet_size)
        b += bet_size * float(row["pnl_units"])
        b = max(0.0, b)
        peak = max(peak, b)
        dd = (peak - b) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
        curve.append(b)
    return {
        "final": b,
        "total_return": (b - bankroll) / bankroll,
        "max_drawdown": max_dd,
        "curve": curve,
    }


def simulate_martingale(bets: pd.DataFrame, base_unit: float, bankroll: float) -> dict:
    b = bankroll
    peak = bankroll
    max_dd = 0.0
    curve = [bankroll]
    current_bet = base_unit
    n_busts = 0
    streak = 0
    max_streak_hit = 0

    for _, row in bets.iterrows():
        result = str(row["result"])

        if result == "push":
            curve.append(b)
            continue

        if current_bet > b:
            # Can't fund the required double — bust, reset series
            n_busts += 1
            current_bet = base_unit
            streak = 0

        b += current_bet * float(row["pnl_units"])
        b = max(0.0, b)

        if result == "win":
            max_streak_hit = max(max_streak_hit, streak)
            streak = 0
            current_bet = base_unit
        else:  # loss
            streak += 1
            current_bet *= 2

        peak = max(peak, b)
        dd = (peak - b) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
        curve.append(b)

    return {
        "final": b,
        "total_return": (b - bankroll) / bankroll,
        "max_drawdown": max_dd,
        "n_busts": n_busts,
        "max_streak_hit": max_streak_hit,
        "curve": curve,
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def pct(v: float) -> str:
    s = f"{v * 100:.1f}%"
    return f"+{s}" if v > 0 else s


def u_fmt(v: float) -> str:
    s = f"{v:.2f}u"
    return f"+{s}" if v > 0 else s


SEP = "=" * 70
SEC = "-" * 70


# ---------------------------------------------------------------------------
# Per-strategy report
# ---------------------------------------------------------------------------

def report_strategy(name: str, bets: pd.DataFrame, args: argparse.Namespace) -> None:
    bankroll = args.bankroll
    base_unit = args.base_unit
    kelly_mult = args.kelly_mult

    n = len(bets)
    n_win = int((bets["result"] == "win").sum())
    n_loss = int((bets["result"] == "loss").sum())
    n_push = int((bets["result"] == "push").sum())
    win_rate = n_win / (n_win + n_loss) if (n_win + n_loss) > 0 else 0.0
    avg_odds = float(bets["under_odds"].mean())
    avg_edge = float(bets["edge_model"].mean())
    flat_ev = float(bets["pnl_units"].mean())

    # Kelly fraction (median across bets as a guide)
    kf_vals = bets.apply(lambda r: kelly_fraction(float(r["p_model"]), float(r["under_odds"])), axis=1)
    median_kf = float(kf_vals.median())

    date_range = f"{bets['date'].min().date()} → {bets['date'].max().date()}"
    n_days = bets["date"].nunique()

    print(f"\n{'=' * 70}")
    print(f"STRATEGY: {name.upper()}")
    print(f"{'=' * 70}")
    print(f"  Period : {date_range}  ({n_days} slate days)")
    print(f"  Bets   : {n}  ({n / n_days:.1f}/day avg)")
    print(f"  W-L-P  : {n_win}-{n_loss}-{n_push}  |  Win rate: {win_rate * 100:.1f}%")
    print(f"  Avg odds: {avg_odds:+.0f}  |  Avg edge: {avg_edge * 100:.1f}%  |  Flat EV/bet: {u_fmt(flat_ev)}")
    print(f"  Median Kelly fraction: {median_kf * 100:.2f}%")

    # Streak analysis
    outcomes = bets["result"].tolist()
    ss = streak_stats(outcomes)
    dist = ss["distribution"]
    bankroll_required = bankroll_needed_to_survive(ss["max"], base_unit)

    print(f"\n  STREAK ANALYSIS")
    print(f"  {SEC[:50]}")
    print(f"  Total losing streaks observed : {ss['n_streaks']}")
    print(f"  Max consecutive losses        : {ss['max']}")
    print(f"  Mean streak length            : {ss['mean']:.2f}")
    print(f"  P95 streak length             : {ss['p95']}")
    if ss["n_streaks"] < 20:
        print(f"  ⚠  Only {ss['n_streaks']} losing streaks — P95 estimate unreliable")

    print(f"\n  Streak distribution:")
    for length, count in dist.items():
        bar = "█" * count
        p_geq = (1 - win_rate) ** length if win_rate > 0 else 0.0
        print(f"    streak={length}: {count:3d}x  {bar[:40]}  P(≥{length}) geometric={p_geq * 100:.1f}%")

    print(f"\n  Martingale bankroll to survive observed max streak of {ss['max']}:")
    print(f"    = {base_unit}u × (2^{ss['max']} - 1) = {bankroll_required:.0f}u  (with {base_unit}u base)")
    if bankroll_required > bankroll:
        print(f"  ⚠  Exceeds starting bankroll ({bankroll:.0f}u) — busts possible at this base unit")
    else:
        print(f"  ✓  Fits within starting bankroll ({bankroll:.0f}u)")

    # Sizing simulations
    flat_res = simulate_flat(bets, base_unit, bankroll)
    kelly_res = simulate_kelly(bets, kelly_mult, bankroll)
    mart_res = simulate_martingale(bets, base_unit, bankroll)

    print(f"\n  SIZING SIMULATIONS  (start={bankroll:.0f}u)")
    print(f"  {SEC[:50]}")
    header = f"  {'Strategy':<18}  {'Final':>8}  {'Return':>8}  {'MaxDD':>8}"
    print(header)
    print(f"  {'-'*18}  {'-'*8}  {'-'*8}  {'-'*8}")

    def sim_row(label: str, res: dict, extra: str = "") -> str:
        return (
            f"  {label:<18}  {res['final']:>7.1f}u  "
            f"{pct(res['total_return']):>8}  "
            f"{pct(-res['max_drawdown']):>8}"
            + (f"  {extra}" if extra else "")
        )

    print(sim_row(f"flat {base_unit}u", flat_res))
    print(sim_row(f"Kelly {kelly_mult}x", kelly_res))
    busts = mart_res["n_busts"]
    bust_note = f"  busts={busts}  max_streak_hit={mart_res['max_streak_hit']}"
    print(sim_row("martingale", mart_res, bust_note))

    # Martingale verdict
    print()
    if busts == 0 and ss["max"] <= 6:
        print(f"  ✓  MARTINGALE VIABLE: no busts observed, max streak {ss['max']}")
    elif busts == 0:
        print(f"  ⚠  No busts in backtest but max streak={ss['max']} — high bankroll req ({bankroll_required:.0f}u)")
    else:
        print(f"  ✗  MARTINGALE RISKY: {busts} bust(s) in backtest — strategy too volatile")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ensure_repo_root_on_syspath()
    args = parse_args()
    bucket = args.bucket
    runs_prefix = args.runs_prefix.rstrip("/")

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    print(SEP)
    print("NBA REBOUNDS — BET SIZING & MARTINGALE VIABILITY")
    print(f"Generated : {now_utc}")
    if args.since:
        print(f"Since     : {args.since}")
    print(f"Bankroll  : {args.bankroll:.0f}u  |  Base unit: {args.base_unit}u  |  Kelly mult: {args.kelly_mult}x")
    print(f"High-edge threshold: {args.high_edge_threshold:.0%}")
    print(SEP)

    print("\nLoading settled data...")
    df = load_settled_data(bucket, runs_prefix, args.since)
    if df.empty:
        print("No settled data found.")
        return

    total_bets = int(df["is_bet"].astype(bool).sum())
    print(f"  Total rows: {len(df)} | Bets placed: {total_bets} | Dates: {df['date'].nunique()}\n")

    strategies = build_strategies(df, args.high_edge_threshold)
    if not strategies:
        print("No strategies with bet data found.")
        return

    for name, bets in strategies.items():
        report_strategy(name, bets, args)

    # Summary ranking table
    print(f"\n{SEP}")
    print("MARTINGALE VIABILITY RANKING")
    print(SEP)
    print(f"  (ranked by win rate — higher = more stable = safer to martingale)")
    print()
    print(f"  {'Strategy':<25}  {'Bets':>5}  {'WinRate':>8}  {'MaxStreak':>10}  {'BankrollReq':>12}  {'MartReturn':>11}  {'MartDD':>8}  {'Busts':>6}")
    print(f"  {'-'*25}  {'-'*5}  {'-'*8}  {'-'*10}  {'-'*12}  {'-'*11}  {'-'*8}  {'-'*6}")

    rows = []
    for name, bets in strategies.items():
        n_win = int((bets["result"] == "win").sum())
        n_loss = int((bets["result"] == "loss").sum())
        wr = n_win / (n_win + n_loss) if (n_win + n_loss) > 0 else 0.0
        ss = streak_stats(bets["result"].tolist())
        br = bankroll_needed_to_survive(ss["max"], args.base_unit)
        mart = simulate_martingale(bets, args.base_unit, args.bankroll)
        rows.append((name, len(bets), wr, ss["max"], br, mart["total_return"], mart["max_drawdown"], mart["n_busts"]))

    rows.sort(key=lambda r: r[2], reverse=True)
    for name, n, wr, ms, br, mr, mdd, busts in rows:
        print(
            f"  {name:<25}  {n:>5}  {wr * 100:>7.1f}%  {ms:>10}  {br:>11.0f}u  "
            f"{pct(mr):>11}  {pct(-mdd):>8}  {busts:>6}"
        )

    print()


if __name__ == "__main__":
    main()
