"""
Aggregate CLV summary across all play bets for a given game date.

Answers: on average, how much better/worse is our 9am ET entry vs open (first-seen) and close?

Usage:
  uv run python src/mlb_total_bases_modeling/analysis/agg_clv_summary.py --gameday 2026-09-08
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

import yaml
from src.mlb_total_bases_modeling.scripts.compute_clv import compute_clv

DAILY_RUNS  = Path.home() / "Downloads/tmp/total_bases/daily_runs"
SUMMARY_DIR = Path.home() / "Downloads/tmp/total_bases"
LOG_PATH    = SUMMARY_DIR / "clv_summary_log.csv"

_CFG = yaml.safe_load(open(REPO_ROOT / "src/mlb_total_bases_modeling/config.yaml"))
_MARKETS = _CFG["strategy"]["markets"]
_LINES   = [float(x) for x in _CFG["strategy"]["lines"]]


def _load_recs(gameday: str) -> pd.DataFrame:
    path = DAILY_RUNS / gameday / "recommendations.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["game_date"] = gameday
    return df


def _merge_key(clv_df: pd.DataFrame, recs: pd.DataFrame) -> list[str]:
    """Use event_id in merge key if recs has it (forward-compat); fallback for old files."""
    if "event_id" in recs.columns and "event_id" in clv_df.columns:
        return ["player_name", "bookmaker", "event_id", "game_date"]
    return ["player_name", "bookmaker", "game_date"]


def _fmt(val) -> str:
    if val is None or pd.isna(val):
        return "  n/a"
    sign = "+" if val >= 0 else ""
    return f"{sign}{val:.1f}¢"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gameday", required=True, help="Game date YYYY-MM-DD")
    args = parser.parse_args()
    gameday = args.gameday
    season  = int(gameday[:4])

    clv_df = compute_clv(gameday, season, markets=_MARKETS, lines=_LINES)
    if clv_df.empty:
        print(f"No CLV data for {gameday} — not enough snapshots yet.")
        return

    recs = _load_recs(gameday)
    key  = _merge_key(clv_df, recs)

    # ── All books in CLV data ──────────────────────────────────────────────────
    print(f"\n{'='*58}")
    print(f"CLV Summary  |  {gameday}  |  all books in snapshot")
    print(f"{'='*58}")
    _print_stats(clv_df, "All snapshot rows")

    if recs.empty:
        print(f"\n  No recommendations found at {DAILY_RUNS / gameday}")
        _save_outputs(gameday, clv_df, pd.DataFrame(), pd.DataFrame(), None, None)
        return

    plays  = recs[recs["tier"] == "play"]
    tracks = recs[recs["tier"] == "track"]

    play_clv  = pd.DataFrame()
    track_clv = pd.DataFrame()

    # ── Play bets ──────────────────────────────────────────────────────────────
    if not plays.empty:
        play_clv = clv_df.merge(
            plays[[c for c in key if c in plays.columns]].drop_duplicates(),
            on=[c for c in key if c in plays.columns],
            how="inner",
        )
        print(f"\n{'='*58}")
        print(f"Play bets only  (n={len(plays)} recs → {len(play_clv)} CLV rows)")
        print(f"{'='*58}")
        _print_stats(play_clv, "Play bets")

    # ── Track bets ─────────────────────────────────────────────────────────────
    if not tracks.empty:
        track_clv = clv_df.merge(
            tracks[[c for c in key if c in tracks.columns]].drop_duplicates(),
            on=[c for c in key if c in tracks.columns],
            how="inner",
        )
        print(f"\n{'='*58}")
        print(f"Track bets only  (n={len(tracks)} recs → {len(track_clv)} CLV rows)")
        print(f"{'='*58}")
        _print_stats(track_clv, "Track bets")

    # ── Key question: cost of waiting from open → 9am ─────────────────────────
    fs_mean  = None
    am9_mean = None
    if not play_clv.empty:
        fs   = play_clv["clv_first_seen_cents"].dropna()
        am9  = play_clv["clv_9am_cents"].dropna()
        fs_mean  = fs.mean()  if not fs.empty  else None
        am9_mean = am9.mean() if not am9.empty else None
        if pd.notna(fs_mean) and pd.notna(am9_mean):
            cost = fs_mean - am9_mean
            print(f"\n{'='*58}")
            print(f"Edge lost waiting from open → 9am ET (play bets)")
            print(f"  Avg CLV at open  (first-seen): {_fmt(fs_mean)}")
            print(f"  Avg CLV at 9am ET:             {_fmt(am9_mean)}")
            print(f"  Edge left on table:            {_fmt(cost)}")
            verdict = "SIGNIFICANT — consider betting at first-seen" if cost > 5 else "minimal — 9am timing is fine"
            print(f"  Verdict: {verdict}")
        print(f"{'='*58}\n")

    _save_outputs(gameday, clv_df, play_clv, track_clv, fs_mean, am9_mean)


def _print_stats(df: pd.DataFrame, label: str) -> None:
    for col, name in [
        ("clv_first_seen_cents", "CLV at open (first-seen)"),
        ("clv_9am_cents",        "CLV at 9am ET           "),
        ("clv_line_shift",       "Line shift (open→close) "),
    ]:
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if vals.empty:
            print(f"  {name}: n/a")
            continue
        pct_positive = (vals > 0).mean() * 100
        print(f"  {name}:  avg={_fmt(vals.mean())}  median={_fmt(vals.median())}  "
              f"beat_close={pct_positive:.0f}%  (n={len(vals)})")

    tier_col = "clv_tier"
    if tier_col in df.columns:
        tiers = df[tier_col].value_counts().to_dict()
        tier_str = "  ".join(f"{k}:{v}" for k, v in sorted(tiers.items()))
        print(f"  Tiers: {tier_str}")


def _save_outputs(
    gameday: str,
    clv_df: pd.DataFrame,
    play_clv: pd.DataFrame,
    track_clv: pd.DataFrame,
    fs_mean: float | None,
    am9_mean: float | None,
) -> None:
    # ── Per-day detail CSV ─────────────────────────────────────────────────────
    detail_path = SUMMARY_DIR / f"clv_summary_{gameday}.csv"
    if not play_clv.empty:
        play_clv.assign(segment="play").to_csv(detail_path, index=False)
    elif not clv_df.empty:
        clv_df.to_csv(detail_path, index=False)
    print(f"  Saved detail → {detail_path}")

    # ── Running aggregate log ──────────────────────────────────────────────────
    cost = (fs_mean - am9_mean) if (pd.notna(fs_mean) and pd.notna(am9_mean)) else None

    def _stat(df: pd.DataFrame, col: str):
        if df.empty or col not in df.columns:
            return None
        v = df[col].dropna()
        return round(v.mean(), 2) if not v.empty else None

    log_row = pd.DataFrame([{
        "gameday":               gameday,
        "n_play_clv_rows":       len(play_clv),
        "n_track_clv_rows":      len(track_clv),
        "avg_clv_first_seen":    _stat(play_clv, "clv_first_seen_cents"),
        "avg_clv_9am":           _stat(play_clv, "clv_9am_cents"),
        "avg_clv_close":         _stat(play_clv, "clv_line_shift"),
        "edge_left_on_table":    round(cost, 2) if cost is not None else None,
        "pct_beat_close_fs":     round((play_clv["clv_first_seen_cents"].dropna() > 0).mean() * 100, 1) if not play_clv.empty else None,
        "pct_beat_close_9am":    round((play_clv["clv_9am_cents"].dropna() > 0).mean() * 100, 1) if not play_clv.empty else None,
    }])

    if LOG_PATH.exists():
        existing = pd.read_csv(LOG_PATH)
        # Overwrite row for this gameday if re-run
        existing = existing[existing["gameday"] != gameday]
        log_df = pd.concat([existing, log_row], ignore_index=True)
    else:
        log_df = log_row

    log_df.to_csv(LOG_PATH, index=False)
    print(f"  Saved log   → {LOG_PATH}  ({len(log_df)} days)")


if __name__ == "__main__":
    main()
