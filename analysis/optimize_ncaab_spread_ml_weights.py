"""
NCAAB Spread/ML weight optimizer: optimal split between spread and ML for underdog bets.

Underdog = side getting points on the spread (from game_lines), not ml_odds > 0.
Join: outcomes + game_ml_odds + game_lines on (game_date, home_team, away_team); all in ESPN names.
Stratify by implied_prob_bucket only. Objective: maximize log growth; report total return too.

Algorithm (empirical simulation, 1u per play / infinite bankroll):
  1. Load games (outcomes + ML odds + lines), merge.
  2. Filter to underdogs (dog POV: when betting the dog, do we go full unit spread, full unit ML, or in between).
  3. Bucket each game into 10 bins by implied win probability: 0-10%%, 10-20%%, ..., 90-100%%.
  4. For each bin:
     - Take all games in that subset.
     - For each weight x in [0, 0.1, 0.2, ..., 1] (spread weight = x, ML weight = 1 - x):
       - For each game: bet x units on spread, (1-x) units on ML (total 1 unit).
       - PnL: if weight is 0 that leg contributes 0; else win => payout PnL, lose => -stake.
       - Aggregate PnL over games in the bin for this x.
     - Maximize log growth (mean log(1 + R)) over x; record optimal split for that bin.
  5. Report optimal ML weight (and spread weight) per bin. No scipy; grid search only.
  Note: We assume infinite bankroll so we can bet 1u every play regardless of running PnL (drawdowns / unit sizing later).

Usage:
  # Test: one day (no cache)
  python analysis/optimize_ncaab_spread_ml_weights.py --test 2026-02-19

  # Real run: one or more seasons (cache used by default: ~/Downloads/tmp/ncaab_spread_ml_underdogs_<seasons>.parquet)
  python analysis/optimize_ncaab_spread_ml_weights.py --seasons 2025-26
  python analysis/optimize_ncaab_spread_ml_weights.py --seasons 2024-25 2025-26 --mode all
  python analysis/optimize_ncaab_spread_ml_weights.py --seasons 2025-26 --analyze-only

  # Force reload from S3 (ignore cache)
  python analysis/optimize_ncaab_spread_ml_weights.py --seasons 2025-26 --no-cache

  # Only true underdogs (implied_win_prob < 50%%); always logs implied >= 50%% counts
  python analysis/optimize_ncaab_spread_ml_weights.py --seasons 2025-26 --underdog-only

  # Empirical + fractional bankroll (bet 1%% of bankroll per game); real risk of ruin
  python analysis/optimize_ncaab_spread_ml_weights.py --seasons 2025-26 --mode all --underdog-only --unit-fraction 0.01 --bankroll-mode both

  # Generate plots (saved to content/viz/ncaab/ or --plot-dir)
  python analysis/optimize_ncaab_spread_ml_weights.py --seasons 2025-26 --mode all --underdog-only --plot
"""

import argparse
import shutil
import ssl
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# Repo root for src imports
def _find_repo_root():
    d = Path(__file__).resolve().parent.parent
    if (d / ".gitignore").exists():
        return d
    raise RuntimeError("Could not find repo root")
ROOT = _find_repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ncaab_team_name_mapping import ODDS_API_TO_ESPN_NCAAB
from src.odds_utils import odds_to_implied_probability, did_cover_spread
from src.season_utils import get_season_dates
from src.s3_utils import read_df_from_s3, list_s3_files

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
BUCKET = "ncaab-betting-mt"
OUTCOMES_PREFIX = "data/01_input/historical_game_results/"
ML_PREFIX = "data/01_input/the-odds-api/ncaab/game_ml_odds/"
LINES_PREFIX = "data/01_input/the-odds-api/ncaab/game_lines/"

DEFAULT_SPREAD_ODDS = -110  # when game_lines has no spread-odds column
MIN_SAMPLE_SIZE = 20
# Grid step for spread/ML weight: x in 0, GRID_STEP, 2*GRID_STEP, ..., 1 (spread weight = x, ML weight = 1-x)
GRID_STEP = 0.1
# 10 buckets: 0-10%, 10-20%, ..., 90-100%
IMPLIED_PROB_BUCKETS = [
    (0.0, 0.10),
    (0.10, 0.20),
    (0.20, 0.30),
    (0.30, 0.40),
    (0.40, 0.50),
    (0.50, 0.60),
    (0.60, 0.70),
    (0.70, 0.80),
    (0.80, 0.90),
    (0.90, 1.0),
]

ODDS_TO_ESPN = {k.lower(): v for k, v in ODDS_API_TO_ESPN_NCAAB.items()}

CACHE_DIR = Path.home() / "Downloads" / "tmp"

# Plot defaults (16:9 aspect)
PLOT_WIDTH_IN = 16
PLOT_HEIGHT_IN = 9
NCAA_LOGO_URL = "https://raw.githubusercontent.com/MylesThomas/shot-quality/main/Logos/NCAA_logo.png"
# Caption on plot (bottom left) and suggested tweet copy when --plot is used
PLOT_CAPTION = "6 seasons NCAAB underdogs, spread vs ML weight | @TQSLabs"
SUGGESTED_TWEET = """NCAAB underdogs: how much to weight spread vs ML? Grid search over 6 seasons.
Big underdogs (0-10% implied) get crushed on ML; mixing in spread helps. Charts: ROI by bucket + optimal weight by bucket.
@TQSLabs"""


def _cache_path_for_season(season: str) -> Path:
    """Path for one season's cache: ncaab_spread_ml_underdogs_2025-26.parquet"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"ncaab_spread_ml_underdogs_{season}.parquet"


def _underdogs_for_season_dates(
    bucket: str,
    outcomes_prefix: str,
    ml_prefix: str,
    lines_prefix: str,
    dates: list[str],
) -> pd.DataFrame | None:
    """Load from S3 for given dates, merge, create underdogs, add buckets. Returns None if empty."""
    outcomes, _ = load_outcomes(bucket, outcomes_prefix, dates)
    ml, _ = load_game_ml_odds(bucket, ml_prefix, dates)
    lines, _ = load_game_lines(bucket, lines_prefix, dates)
    if outcomes.empty or ml.empty:
        return None
    merged = merge_outcomes_ml_lines(outcomes, ml, lines)
    if merged.empty:
        return None
    underdogs = create_underdog_dataset(merged)
    if underdogs.empty:
        return None
    return add_implied_prob_buckets(underdogs)


def _odds_to_espn(name: str) -> str:
    if pd.isna(name):
        return ""
    return ODDS_TO_ESPN.get(str(name).lower().strip(), name)


def _american_to_decimal(american_odds: float) -> float:
    if american_odds < 0:
        return 1 + (100 / abs(american_odds))
    return 1 + (american_odds / 100)


# -----------------------------------------------------------------------------
# Load from S3 (by date list)
# -----------------------------------------------------------------------------
def load_outcomes(bucket: str, prefix: str, dates: list[str]) -> tuple[pd.DataFrame, list[str]]:
    dfs = []
    missing = []
    for d in dates:
        key = f"{prefix}{d}.csv"
        try:
            df = read_df_from_s3(bucket, key)
            if df is not None and not df.empty:
                df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.date
                dfs.append(df)
            else:
                missing.append(d)
        except Exception:
            missing.append(d)
    if not dfs:
        return pd.DataFrame(), dates
    out = pd.concat(dfs, ignore_index=True)
    out = out.drop_duplicates(subset=["GAME_DATE", "HOME_TEAM", "AWAY_TEAM"])
    return out, missing


def load_game_ml_odds(bucket: str, prefix: str, dates: list[str]) -> tuple[pd.DataFrame, list[str]]:
    dfs = []
    missing = []
    for d in dates:
        key = f"{prefix}{d}.csv"
        try:
            df = read_df_from_s3(bucket, key)
            if df is not None and not df.empty:
                df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
                if "error" in df.columns:
                    df = df[df["error"].isna() | (df["error"] == "")]
                if not df.empty:
                    dfs.append(df)
                else:
                    missing.append(d)
            else:
                missing.append(d)
        except Exception:
            missing.append(d)
    if not dfs:
        return pd.DataFrame(), dates
    out = pd.concat(dfs, ignore_index=True)
    out = out.drop_duplicates(subset=["game_date", "home_team", "away_team"])
    return out, missing


def load_game_lines(bucket: str, prefix: str, dates: list[str]) -> tuple[pd.DataFrame, list[str]]:
    dfs = []
    missing = []
    for d in dates:
        key = f"{prefix}{d}.csv"
        try:
            df = read_df_from_s3(bucket, key)
            if df is not None and not df.empty:
                df["date"] = pd.to_datetime(df["date"]).dt.date
                df["home_team"] = df["home_team"].astype(str).apply(_odds_to_espn)
                df["away_team"] = df["away_team"].astype(str).apply(_odds_to_espn)
                dfs.append(df)
            else:
                missing.append(d)
        except Exception:
            missing.append(d)
    if not dfs:
        return pd.DataFrame(), dates
    out = pd.concat(dfs, ignore_index=True)
    out = out.drop_duplicates(subset=["date", "home_team", "away_team"])
    return out, missing


# -----------------------------------------------------------------------------
# Merge and underdog rows (underdog = side getting points on spread)
# -----------------------------------------------------------------------------
def merge_outcomes_ml_lines(outcomes: pd.DataFrame, ml: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    """Inner join on (game_date, home_team, away_team). Outcomes and ML use ESPN names; lines normalized above."""
    if outcomes.empty or ml.empty:
        return pd.DataFrame()
    merged = outcomes.merge(
        ml,
        left_on=["GAME_DATE", "HOME_TEAM", "AWAY_TEAM"],
        right_on=["game_date", "home_team", "away_team"],
        how="inner",
    )
    if lines.empty:
        merged["consensus_spread"] = np.nan
        return merged
    lines_sub = lines[["date", "home_team", "away_team", "consensus_spread"]].copy()
    merged = merged.merge(
        lines_sub,
        left_on=["GAME_DATE", "HOME_TEAM", "AWAY_TEAM"],
        right_on=["date", "home_team", "away_team"],
        how="left",
        suffixes=("", "_line"),
    )
    # Keep a single consensus_spread (from line merge)
    if "consensus_spread_line" in merged.columns:
        merged["consensus_spread"] = merged["consensus_spread_line"]
    drop = [c for c in merged.columns if c.endswith("_line") or c == "date"]
    merged = merged.drop(columns=[c for c in drop if c in merged.columns], errors="ignore")
    return merged


def create_underdog_dataset(merged: pd.DataFrame) -> pd.DataFrame:
    """
    One row per underdog. Underdog = side getting points on the spread (consensus_spread = home spread).
    - consensus_spread > 0 => home gets points => home underdog
    - consensus_spread < 0 => away gets points => away underdog
    - consensus_spread == 0 or NaN => skip (no clear underdog)
    """
    rows = []
    for _, row in merged.iterrows():
        home_spread = row.get("consensus_spread")
        if pd.isna(home_spread) or home_spread == 0:
            continue
        home_score = row["HOME_SCORE"]
        away_score = row["AWAY_SCORE"]
        if home_spread > 0:
            # Home underdog (home getting points)
            spread_covered = did_cover_spread(home_score, away_score, home_spread, bet_home=True)
            rows.append({
                "game_date": row["GAME_DATE"],
                "home_team": row["HOME_TEAM"],
                "away_team": row["AWAY_TEAM"],
                "home_score": home_score,
                "away_score": away_score,
                "bet_team_name": row["HOME_TEAM"],
                "opponent": row["AWAY_TEAM"],
                "is_home": True,
                "bet_team_spread": home_spread,
                "spread_odds": DEFAULT_SPREAD_ODDS,
                "ml_odds": row["home_ml_odds"],
                "ml_won": row["HOME_WL"] == "W",
                "spread_covered": spread_covered if spread_covered is not None else False,
            })
        else:
            # Away underdog (away getting points; bet_team_spread = points away gets)
            spread_covered = did_cover_spread(home_score, away_score, home_spread, bet_home=False)
            rows.append({
                "game_date": row["GAME_DATE"],
                "home_team": row["HOME_TEAM"],
                "away_team": row["AWAY_TEAM"],
                "home_score": home_score,
                "away_score": away_score,
                "bet_team_name": row["AWAY_TEAM"],
                "opponent": row["HOME_TEAM"],
                "is_home": False,
                "bet_team_spread": -home_spread,
                "spread_odds": DEFAULT_SPREAD_ODDS,
                "ml_odds": row["away_ml_odds"],
                "ml_won": row["AWAY_WL"] == "W",
                "spread_covered": spread_covered if spread_covered is not None else False,
            })
    return pd.DataFrame(rows)


def add_implied_prob_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Add implied_win_prob and implied_prob_bucket."""
    df = df.copy()
    df["implied_win_prob"] = df["ml_odds"].astype(float).apply(odds_to_implied_probability)
    def bucket(p):
        for low, high in IMPLIED_PROB_BUCKETS:
            if low <= p < high:
                return f"{int(low*100)}-{int(high*100)}%"
        return None
    df["implied_prob_bucket"] = df["implied_win_prob"].apply(bucket)
    return df


# -----------------------------------------------------------------------------
# Optimization
# -----------------------------------------------------------------------------
def calculate_log_growth(df: pd.DataFrame, w_ml: float) -> float:
    """Per-game return on 1 unit total: w_ml on ML, (1-w_ml) on spread. If weight is 0 that leg PnL is 0. Mean(log(1+R)) = log growth for flat 1u/game (infinite bankroll)."""
    w_spread = 1 - w_ml
    log_returns = []
    for _, row in df.iterrows():
        r_ml = (w_ml * (_american_to_decimal(row["ml_odds"]) - 1)) if row["ml_won"] else -w_ml
        r_spread = (w_spread * (_american_to_decimal(row["spread_odds"]) - 1)) if row["spread_covered"] else -w_spread
        total = r_ml + r_spread  # PnL on the 1 unit we risked this game
        log_returns.append(np.log(1 + total) if total > -1 else -10.0)
    return np.mean(log_returns)


def calculate_total_return_pct(df: pd.DataFrame, w_ml: float) -> float:
    """Compound return: bet w_ml and w_spread of bankroll each game. Can hit -100% when we go bust."""
    w_spread = 1 - w_ml
    bankroll = 1.0
    for _, row in df.iterrows():
        ml_ret = (w_ml * (_american_to_decimal(row["ml_odds"]) - 1)) if row["ml_won"] else -w_ml
        spread_ret = (w_spread * (_american_to_decimal(row["spread_odds"]) - 1)) if row["spread_covered"] else -w_spread
        bankroll *= 1 + ml_ret + spread_ret
    return (bankroll - 1.0) * 100


def calculate_flat_return_pct(df: pd.DataFrame, w_ml: float) -> float:
    """Flat 1 unit per game (w_ml on ML, w_spread on spread). ROI = total profit / num_games, as pct."""
    w_spread = 1 - w_ml
    total_profit = 0.0
    for _, row in df.iterrows():
        ml_ret = (w_ml * (_american_to_decimal(row["ml_odds"]) - 1)) if row["ml_won"] else -w_ml
        spread_ret = (w_spread * (_american_to_decimal(row["spread_odds"]) - 1)) if row["spread_covered"] else -w_spread
        total_profit += ml_ret + spread_ret
    n = len(df)
    return (total_profit / n * 100) if n else 0.0


def cumulative_units_series(df: pd.DataFrame, w_ml: float) -> list[float]:
    """Running PnL in units for 1u/game split (w_ml on ML, 1-w_ml on spread). Starts at 0; length = len(df)+1."""
    w_spread = 1 - w_ml
    units = [0.0]
    for _, row in df.iterrows():
        ml_ret = (w_ml * (_american_to_decimal(row["ml_odds"]) - 1)) if row["ml_won"] else -w_ml
        spread_ret = (w_spread * (_american_to_decimal(row["spread_odds"]) - 1)) if row["spread_covered"] else -w_spread
        units.append(units[-1] + ml_ret + spread_ret)
    return units


def units_stats(cumulative_units: list[float]) -> dict[str, float]:
    """From list of cumulative units (start 0), return max_units, min_units, avg_units, std_units."""
    arr = np.array(cumulative_units)
    return {
        "max_units": float(np.max(arr)),
        "min_units": float(np.min(arr)),
        "avg_units": float(np.mean(arr)),
        "std_units": float(np.std(arr, ddof=0)),
    }


def simulate_fractional_bankroll(df: pd.DataFrame, w_ml: float, unit_fraction: float) -> dict:
    """Real-life path: B starts at 1.0, each game we risk unit_fraction*B (split w_ml/w_spread). Returns final_bankroll_multiple, went_bust, min_bankroll_ratio, max_bankroll_ratio, max_drawdown_pct."""
    w_spread = 1 - w_ml
    bankrolls = [1.0]
    peak = 1.0
    max_drawdown_pct = 0.0
    for _, row in df.iterrows():
        ml_ret = (w_ml * (_american_to_decimal(row["ml_odds"]) - 1)) if row["ml_won"] else -w_ml
        spread_ret = (w_spread * (_american_to_decimal(row["spread_odds"]) - 1)) if row["spread_covered"] else -w_spread
        R = ml_ret + spread_ret
        B = bankrolls[-1] * (1 + unit_fraction * R)
        bankrolls.append(max(B, 0.0))  # cap at 0 so we don't go negative
        if B > peak:
            peak = B
        if peak > 0:
            max_drawdown_pct = max(max_drawdown_pct, (peak - B) / peak * 100)
    final = bankrolls[-1]
    return {
        "final_bankroll_multiple": final,
        "went_bust": final <= 0,
        "min_bankroll_ratio": float(np.min(bankrolls)),
        "max_bankroll_ratio": float(np.max(bankrolls)),
        "max_drawdown_pct": float(max_drawdown_pct),
    }


def optimize_group(df: pd.DataFrame) -> dict:
    """Find w_ml that maximizes log growth for 1 unit/game split. Grid search only: w_ml = 0, GRID_STEP, ..., 1; evaluate mean(log(1+R)) at each, keep the best. No scipy."""
    # n=1: cannot estimate optimal split from one game; skip optimization and return NaNs
    if len(df) < 2:
        return {
            "optimal_ml_weight": np.nan,
            "optimal_spread_weight": np.nan,
            "log_growth": np.nan,
            "total_return_pct": np.nan,
            "flat_return_pct": np.nan,
            "sample_size": len(df),
            "spread_cover_rate": np.nan,
            "ml_win_rate": np.nan,
            "max_units": np.nan,
            "min_units": np.nan,
            "avg_units": np.nan,
            "std_units": np.nan,
        }
    best_w = 0.0
    best_lg = -np.inf
    # Grid search: w_ml in [0, 1] with step GRID_STEP (spread weight = 1 - w_ml); no scipy
    for w in np.arange(0.0, 1.0 + 1e-9, GRID_STEP):
        lg = calculate_log_growth(df, float(w))
        if lg > best_lg:
            best_lg = lg
            best_w = w
    w_ml = float(best_w)
    total_return_pct = calculate_total_return_pct(df, w_ml)
    flat_return_pct = calculate_flat_return_pct(df, w_ml)
    cum_units = cumulative_units_series(df, w_ml)
    u = units_stats(cum_units)
    return {
        "optimal_ml_weight": w_ml,
        "optimal_spread_weight": 1 - w_ml,
        "log_growth": best_lg,
        "total_return_pct": total_return_pct,
        "flat_return_pct": flat_return_pct,
        "sample_size": len(df),
        "spread_cover_rate": df["spread_covered"].mean(),
        "ml_win_rate": df["ml_won"].mean(),
        "max_units": u["max_units"],
        "min_units": u["min_units"],
        "avg_units": u["avg_units"],
        "std_units": u["std_units"],
    }


# -----------------------------------------------------------------------------
# Main: date list, run modes
# -----------------------------------------------------------------------------
def get_dates_for_seasons(seasons: list[str]) -> list[str]:
    dates = []
    for s in seasons:
        cfg = get_season_dates("ncaab", s)
        start = pd.to_datetime(cfg["season_start"]).date()
        end = pd.to_datetime(cfg["tournament_end"]).date()
        for d in pd.date_range(start=start, end=end, freq="D"):
            dates.append(d.strftime("%Y-%m-%d"))
    return sorted(set(dates))


def run_analyze_only(underdogs: pd.DataFrame) -> None:
    """Print cover-rate summary."""
    if underdogs.empty:
        print("No underdog rows.")
        return
    print("\n--- Cover / ML win rates ---")
    print(f"  N = {len(underdogs)}")
    print(f"  Spread cover rate: {underdogs['spread_covered'].mean():.1%}")
    print(f"  ML win rate: {underdogs['ml_won'].mean():.1%}")
    print(f"  Both: {(underdogs['spread_covered'] & underdogs['ml_won']).mean():.1%}")


def _debug_test_run(underdogs: pd.DataFrame) -> None:
    """When --test: log sample rows and step-through of compound so we can see why log_growth is negative and total_return_pct is -100."""
    if underdogs.empty or len(underdogs) < 2:
        return
    print("\n" + "=" * 60)
    print("DEBUG (--test): why log_growth negative / total_return_pct -100%")
    print("=" * 60)
    print("\n1) Sample underdog rows (first 5) — game context then spread/ML (bet_team = underdog):")
    cols = [
        "game_date", "home_team", "away_team", "home_score", "away_score",
        "bet_team_name", "bet_team_spread", "ml_odds", "spread_odds",
        "spread_covered", "ml_won", "implied_win_prob",
    ]
    cols = [c for c in cols if c in underdogs.columns]
    print(underdogs[cols].head().to_string(index=False))
    print("\n2) Rates: spread_cover = {:.1%}, ml_win = {:.1%}. At spread_odds=-110 break-even cover rate = 52.38%.".format(
        underdogs["spread_covered"].mean(), underdogs["ml_won"].mean()))
    w_ml = 0.0
    w_spread = 1.0
    print("\n3) Compound with w_ml=0 (all spread): we bet 100% of bankroll on spread each game.")
    print("   One spread LOSS => bankroll *= (1 + spread_ret) = (1 - 1) = 0 => bust. So total_return_pct = -100% after first loss.")
    bankroll = 1.0
    for i, (_, row) in enumerate(underdogs.iterrows()):
        spread_ret = (w_spread * (_american_to_decimal(row["spread_odds"]) - 1)) if row["spread_covered"] else -w_spread
        prev = bankroll
        bankroll *= 1 + spread_ret
        if i < 8:
            print(f"   game {i+1}: spread_covered={row['spread_covered']}, spread_ret={spread_ret:.4f}, bankroll {prev:.4f} -> {bankroll:.4f}")
        if bankroll <= 0:
            print(f"   -> BUST at game {i+1}. Remaining games leave bankroll at 0. So total_return_pct = (0 - 1)*100 = -100%.")
            break
    if bankroll > 0:
        print(f"   After {len(underdogs)} games bankroll = {bankroll:.4f} => total_return_pct = {(bankroll-1)*100:.2f}%")
    print("\n4) Log growth: mean(log(1+R)) per game. When we bust, 1+R=0 so log(0)=-inf (we clamp to -10). So mean log is very negative.")
    print("\n5) Takeaway: total_return_pct (compound) assumes we bet w_ml and w_spread of *current bankroll* each game.")
    print("   So w_ml=0 means we bet 100%% of bankroll on spread => one loss => bust => -100%%. Use flat_return_pct for per-game ROI.")
    print("=" * 60)


def run_generalized(
    underdogs: pd.DataFrame,
    unit_fraction: float | None = None,
    bankroll_mode: str = "empirical",
) -> dict:
    """Single optimal weight over all underdogs. If bankroll_mode in (fractional, both), add fractional-bankroll sim stats."""
    gen = optimize_group(underdogs)
    if unit_fraction is not None and bankroll_mode in ("fractional", "both") and len(underdogs) >= 2:
        frac = simulate_fractional_bankroll(underdogs, gen["optimal_ml_weight"], unit_fraction)
        gen.update(frac)
    return gen


def _log_implied_above_50(underdogs: pd.DataFrame) -> None:
    """Log count of underdog rows with implied_win_prob >= 50% (spread dog but ML favorite)."""
    if underdogs.empty or "implied_win_prob" not in underdogs.columns:
        return
    above = underdogs[underdogs["implied_win_prob"] >= 0.50]
    n = len(above)
    if n == 0:
        return
    print(f"  Implied >= 50%: {n} underdog rows (spread dog but ML favorite; use --underdog-only to filter)")
    # Bucket breakdown
    bins = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.0)]
    for lo, hi in bins:
        m = ((above["implied_win_prob"] >= lo) & (above["implied_win_prob"] < hi)).sum()
        if m > 0:
            print(f"    {int(lo*100)}-{int(hi*100)}%: {m}")
    _debug_high_implied(underdogs)


def _debug_high_implied(underdogs: pd.DataFrame) -> None:
    """Print rows in 80-90% and 90-100% implied buckets (underdog by spread but ML says heavy favorite — likely data bug)."""
    if underdogs.empty or "implied_win_prob" not in underdogs.columns:
        return
    high = underdogs[underdogs["implied_win_prob"] >= 0.80].copy()
    if high.empty:
        return
    high = high.sort_values(["implied_win_prob", "game_date"])
    cols = ["game_date", "home_team", "away_team", "bet_team_name", "bet_team_spread", "ml_odds", "implied_win_prob", "implied_prob_bucket"]
    cols = [c for c in cols if c in high.columns]
    print("\n  [DEBUG] Underdog rows with implied_win_prob >= 80% (spread says dog, ML says big favorite — check spread/ML alignment):")
    print(high[cols].to_string(index=False))


def run_by_bucket(
    underdogs: pd.DataFrame,
    min_sample_override: int | None = None,
    unit_fraction: float | None = None,
    bankroll_mode: str = "empirical",
) -> pd.DataFrame:
    """Optimal weight per implied_prob_bucket. If bankroll_mode fractional/both, add fractional sim stats per bucket."""
    underdogs = underdogs.dropna(subset=["implied_prob_bucket"])
    min_n = min_sample_override if min_sample_override is not None else MIN_SAMPLE_SIZE
    results = []
    for bucket, grp in underdogs.groupby("implied_prob_bucket"):
        row = optimize_group(grp)
        row["implied_prob_bucket"] = bucket
        row["note"] = "" if len(grp) >= min_n else f"n<{min_n} (noisy)"
        if unit_fraction is not None and bankroll_mode in ("fractional", "both"):
            if len(grp) >= 2:
                frac = simulate_fractional_bankroll(grp, row["optimal_ml_weight"], unit_fraction)
                row.update(frac)
            else:
                row["final_bankroll_multiple"] = np.nan
                row["went_bust"] = np.nan
                row["min_bankroll_ratio"] = np.nan
                row["max_bankroll_ratio"] = np.nan
                row["max_drawdown_pct"] = np.nan
        results.append(row)
    return pd.DataFrame(results)


def run_weight_combos_by_bucket(underdogs: pd.DataFrame) -> pd.DataFrame:
    """For each bucket, evaluate all grid weights (0, GRID_STEP, ..., 1); return rows (bucket, w_ml, w_spread, log_g, flat_roi) sorted best to worst by log_g within each bucket. Max rows = 10 buckets * 11 = 110."""
    underdogs = underdogs.dropna(subset=["implied_prob_bucket"])
    rows = []
    for bucket, grp in underdogs.groupby("implied_prob_bucket"):
        if len(grp) < 2:
            continue
        for w in np.arange(0.0, 1.0 + 1e-9, GRID_STEP):
            w_ml = float(w)
            lg = calculate_log_growth(grp, w_ml)
            flat = calculate_flat_return_pct(grp, w_ml)
            rows.append({
                "implied_prob_bucket": bucket,
                "w_ml": w_ml,
                "w_spread": 1 - w_ml,
                "log_growth": lg,
                "flat_return_pct": flat,
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # Best to worst within each bucket (by log_growth descending)
    df = df.sort_values(["implied_prob_bucket", "log_growth"], ascending=[True, False])
    return df.reset_index(drop=True)


def _download_ncaa_logo(plot_dir: Path) -> Path:
    """Download NCAA logo to plot_dir/NCAA_logo.png, or use repo-bundled copy. Returns path."""
    plot_dir = Path(plot_dir)
    logo_path = plot_dir / "NCAA_logo.png"
    if logo_path.exists():
        return logo_path

    # Prefer repo-bundled logo so SSL issues (e.g. macOS Python 3.13) don't block plots.
    bundled = ROOT / "content" / "viz" / "ncaab" / "NCAA_logo.png"
    if bundled.exists():
        shutil.copy2(bundled, logo_path)
        return logo_path

    # Try download: default SSL, then certifi bundle (often fixes macOS "Basic Constraints" / issuer errors).
    last_error = None
    try:
        urllib.request.urlretrieve(NCAA_LOGO_URL, logo_path)
        return logo_path
    except OSError as e:
        last_error = e
    if logo_path.exists():
        logo_path.unlink()  # remove partial file from failed urlretrieve

    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(NCAA_LOGO_URL, timeout=15, context=ctx) as resp:
            logo_path.write_bytes(resp.read())
        return logo_path
    except Exception as e:
        last_error = e
    if logo_path.exists():
        logo_path.unlink()

    print(f"  Warning: could not download NCAA logo: {last_error}")
    print(f"  To show the logo: place NCAA_logo.png at {bundled.resolve()}")
    return logo_path


def generate_plots(
    by_bucket: pd.DataFrame,
    combos: pd.DataFrame,
    plot_dir: Path,
) -> None:
    """Generate plots when --plot is set. Uses R/ggplot2; saves PNGs at 16:9, with NCAA logo."""
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Export data for R
    bucket_cols = ["implied_prob_bucket", "spread_cover_rate", "ml_win_rate", "flat_return_pct"]
    bucket_csv = plot_dir / "by_bucket.csv"
    by_bucket[[c for c in bucket_cols if c in by_bucket.columns]].to_csv(bucket_csv, index=False)

    if not combos.empty:
        combo_cols = ["implied_prob_bucket", "w_ml", "w_spread", "flat_return_pct"]
        combos[[c for c in combo_cols if c in combos.columns]].to_csv(plot_dir / "combos.csv", index=False)

    logo_path = _download_ncaa_logo(plot_dir)
    logo_path_str = str(logo_path.resolve()).replace("\\", "/")
    plot_dir_str = str(plot_dir.resolve()).replace("\\", "/")
    plot_caption_str = PLOT_CAPTION.replace('"', '\\"')  # safe for R string

    r_script = f'''
.libPaths(c("~/R/library", .libPaths()))
if (!requireNamespace("ggplot2", quietly = TRUE)) install.packages("ggplot2", repos = "https://cran.rstudio.com/")
if (!requireNamespace("png", quietly = TRUE)) install.packages("png", repos = "https://cran.rstudio.com/")
if (!requireNamespace("ggthemes", quietly = TRUE)) install.packages("ggthemes", repos = "https://cran.rstudio.com/")
library(ggplot2)
library(png)
library(grid)
library(ggthemes)

plot_dir <- "{plot_dir_str}"
logo_path <- "{logo_path_str}"
plot_caption <- "{plot_caption_str}"
by_bucket <- read.csv(file.path(plot_dir, "by_bucket.csv"))
bucket_order <- c("0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70-80%", "80-90%", "90-100%")
by_bucket$bucket <- factor(by_bucket$implied_prob_bucket, levels = intersect(bucket_order, by_bucket$implied_prob_bucket))

# Logo grob: flush top-right corner (right-aligned with last bucket, top of y-range)
add_logo_bar <- function(dat) {{
  if (!file.exists(logo_path)) return(list())
  img <- readPNG(logo_path)
  g <- rasterGrob(img, width = unit(1, "inch"), height = unit(1, "inch"))
  n <- nrow(dat)
  annotation_custom(g, xmin = n - 0.3, xmax = n + 0.7, ymin = 3.2, ymax = 5)
}}
add_logo_line <- function(dat) {{
  if (!file.exists(logo_path)) return(list())
  img <- readPNG(logo_path)
  g <- rasterGrob(img, width = unit(1, "inch"), height = unit(1, "inch"))
  xmax <- max(dat$w_ml, 1)
  ymax <- max(dat$flat_return_pct, 0) + 2
  annotation_custom(g, xmin = xmax - 0.25, xmax = xmax + 0.35, ymin = ymax - 3, ymax = ymax)
}}

# Plot 1: ROI by bucket (bar); y-axis fixed -5 to 5
p1 <- ggplot(by_bucket, aes(x = bucket, y = flat_return_pct, fill = flat_return_pct >= 0)) +
  geom_col() +
  geom_hline(yintercept = 0, linewidth = 0.5) +
  scale_fill_manual(values = c("TRUE" = "#2ecc71", "FALSE" = "#e74c3c"), guide = "none") +
  coord_cartesian(ylim = c(-5, 5)) +
  labs(title = "Blindly betting underdog spreads is negative EV", subtitle = "Blind 1u spread per game (6 seasons); 10-20%% slightly positive", caption = plot_caption, x = "Implied win prob bucket", y = "Flat ROI (%)") +
  theme_fivethirtyeight(base_size = 14) +
  theme(axis.line = element_line(color = "gray30", linewidth = 0.5), axis.title = element_text(size = 12, color = "gray20"), axis.text.x = element_text(angle = 45, hjust = 1), plot.caption = element_text(hjust = 0, size = 10)) +
  add_logo_bar(by_bucket)
ggsave(file.path(plot_dir, "ncaab_roi_by_bucket.png"), p1, width = {PLOT_WIDTH_IN}, height = {PLOT_HEIGHT_IN}, dpi = 150, bg = "white")
cat("Saved:", file.path(plot_dir, "ncaab_roi_by_bucket.png"), "\\n")
'''

    if not combos.empty:
        r_script += f'''
combos <- read.csv(file.path(plot_dir, "combos.csv"))

# Plot 2: Weight combos 0-10% (11 rows)
c1 <- combos[combos$implied_prob_bucket == "0-10%", ]
if (nrow(c1) > 0) {{
  p2 <- ggplot(c1, aes(x = w_ml, y = flat_return_pct)) +
    geom_col(fill = "#3498db") +
    geom_hline(yintercept = 0, linewidth = 0.5) +
    scale_x_continuous(breaks = seq(0, 1, 0.1)) +
    coord_cartesian(ylim = c(-50, 0)) +
    labs(title = "Heavy underdogs get crushed on ML; spread-only is the least bad strategy and nearly breaks even", subtitle = "0 = all spread (best), 1 = all ML (worst)", caption = plot_caption, x = "ML weight", y = "Flat ROI (%)") +
    theme_fivethirtyeight(base_size = 14) +
    theme(axis.line = element_line(color = "gray30", linewidth = 0.5), axis.title = element_text(size = 12, color = "gray20"), plot.caption = element_text(hjust = 0, size = 10)) +
    add_logo_line(c1)
  ggsave(file.path(plot_dir, "ncaab_weight_combos_0_10.png"), p2, width = {PLOT_WIDTH_IN}, height = {PLOT_HEIGHT_IN}, dpi = 150, bg = "white")
  cat("Saved:", file.path(plot_dir, "ncaab_weight_combos_0_10.png"), "\\n")
}}

# Plot 3: Weight combos 40-50%
c2 <- combos[combos$implied_prob_bucket == "40-50%", ]
if (nrow(c2) > 0) {{
  p3 <- ggplot(c2, aes(x = w_ml, y = flat_return_pct)) +
    geom_col(fill = "#9b59b6") +
    geom_hline(yintercept = 0, linewidth = 0.5) +
    scale_x_continuous(breaks = seq(0, 1, 0.1)) +
    coord_cartesian(ylim = c(-50, 0)) +
    labs(title = "Near-even underdogs still do worse on ML, although only slightly", subtitle = "0 = all spread (best), 1 = all ML (worst)", caption = plot_caption, x = "ML weight", y = "Flat ROI (%)") +
    theme_fivethirtyeight(base_size = 14) +
    theme(axis.line = element_line(color = "gray30", linewidth = 0.5), axis.title = element_text(size = 12, color = "gray20"), plot.caption = element_text(hjust = 0, size = 10)) +
    add_logo_line(c2)
  ggsave(file.path(plot_dir, "ncaab_weight_combos_40_50.png"), p3, width = {PLOT_WIDTH_IN}, height = {PLOT_HEIGHT_IN}, dpi = 150, bg = "white")
  cat("Saved:", file.path(plot_dir, "ncaab_weight_combos_40_50.png"), "\\n")
}}
'''

    r_file = plot_dir / "_plot_ncaab_optimizer.R"
    r_file.write_text(r_script, encoding="utf-8")

    try:
        result = subprocess.run(
            ["Rscript", str(r_file)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(ROOT),
        )
        if result.returncode != 0:
            print(f"  --plot Rscript failed: {result.stderr or result.stdout}")
            return
        if result.stdout:
            print(result.stdout.strip())
    except FileNotFoundError:
        print("  --plot: Rscript not found. Install R and ensure Rscript is on PATH.")
        return
    except subprocess.TimeoutExpired:
        print("  --plot: Rscript timed out.")
        return

    pngs = sorted(plot_dir.glob("ncaab_*.png"))
    if pngs:
        paths_str = " ".join(str(p.resolve()) for p in pngs)
        print(f"  --plot: saved to {plot_dir.resolve()}")
        print(f"  Open with: open {paths_str}")
        print()
        print("  Suggested tweet copy:")
        print("-" * 50)
        for line in SUGGESTED_TWEET.strip().splitlines():
            print(f"  {line}")
        print("-" * 50)
    return


def main():
    ap = argparse.ArgumentParser(description="NCAAB spread/ML weight optimizer")
    ap.add_argument("--seasons", nargs="+", help="Seasons e.g. 2024-25 2025-26")
    ap.add_argument("--test", type=str, metavar="DATE", help="Single date YYYY-MM-DD to run on one day of data (e.g. 2026-02-19)")
    ap.add_argument("--mode", choices=["generalized", "by_bucket", "all"], default="all")
    ap.add_argument("--analyze-only", action="store_true", help="Only print cover/ML rates, no optimization")
    ap.add_argument("--no-cache", action="store_true", help="Force reload from S3 (ignore cache); only applies to --seasons")
    ap.add_argument("--underdog-only", action="store_true", help="Keep only rows with implied_win_prob < 50%% (true underdogs by ML)")
    ap.add_argument("--bankroll-mode", choices=["empirical", "fractional", "both"], default="empirical",
                    help="empirical=flat 1u/game (unlimited $); fractional=bet unit_fraction of bankroll each game; both=run and print both")
    ap.add_argument("--unit-fraction", type=float, default=0.01, metavar="F",
                    help="Fraction of current bankroll risked per game in fractional mode (default 0.01 = 1%%)")
    ap.add_argument("--plot", action="store_true", help="Generate plots (saved to --plot-dir)")
    ap.add_argument("--plot-dir", type=str, default=None, metavar="DIR",
                    help="Directory to save plot(s); default: content/viz/ncaab/")
    args = ap.parse_args()

    if args.test:
        dates = [args.test]
        print(f"Test mode: single date {args.test} (small sample — results noisy)")
        use_cache = False
    elif args.seasons:
        dates = get_dates_for_seasons(args.seasons)
        print(f"Seasons {args.seasons}: {len(dates)} dates")
        use_cache = not args.no_cache
    else:
        ap.error("Provide either --seasons or --test DATE")

    underdogs = None
    if use_cache and args.seasons:
        parts = []
        missing_seasons = []
        for season in args.seasons:
            path = _cache_path_for_season(season)
            if path.exists():
                df = pd.read_parquet(path)
                if "game_date" in df.columns:
                    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
                parts.append(df)
                print(f"  Cache {season}: {path.name} ({len(df)} rows)")
            else:
                parts.append(None)
                missing_seasons.append(season)
        if not missing_seasons:
            underdogs = pd.concat(parts, ignore_index=True)
            print(f"  Underdog rows: {len(underdogs)} (all from cache)")
        else:
            loaded = {}
            for season in missing_seasons:
                season_dates = get_dates_for_seasons([season])
                print(f"  Loading from S3 for season {season} ({len(season_dates)} dates)...")
                ud = _underdogs_for_season_dates(BUCKET, OUTCOMES_PREFIX, ML_PREFIX, LINES_PREFIX, season_dates)
                if ud is None or ud.empty:
                    print(f"  No underdog data for {season}; skipping cache write.")
                    continue
                _cache_path_for_season(season).parent.mkdir(parents=True, exist_ok=True)
                ud.to_parquet(_cache_path_for_season(season), index=False)
                print(f"  Saved cache: {_cache_path_for_season(season).name} ({len(ud)} rows)")
                loaded[season] = ud
            final_parts = []
            for i, season in enumerate(args.seasons):
                if parts[i] is not None:
                    final_parts.append(parts[i])
                elif season in loaded:
                    final_parts.append(loaded[season])
            underdogs = pd.concat(final_parts, ignore_index=True) if final_parts else None
            if underdogs is not None:
                print(f"  Underdog rows: {len(underdogs)}")
        if underdogs is not None and underdogs.empty:
            underdogs = None

    if underdogs is None:
        print("Loading outcomes, ML, lines from S3...")
        outcomes, out_miss = load_outcomes(BUCKET, OUTCOMES_PREFIX, dates)
        ml, ml_miss = load_game_ml_odds(BUCKET, ML_PREFIX, dates)
        lines, line_miss = load_game_lines(BUCKET, LINES_PREFIX, dates)
        print(f"  Outcomes: {len(outcomes)}, ML: {len(ml)}, Lines: {len(lines)}")

        if out_miss:
            print(f"  Issue: missing outcomes for {len(out_miss)} date(s): {out_miss[:10]}{'...' if len(out_miss) > 10 else ''}")
        if ml_miss:
            print(f"  Issue: missing ML for {len(ml_miss)} date(s): {ml_miss[:10]}{'...' if len(ml_miss) > 10 else ''}")
        if line_miss:
            print(f"  Issue: missing lines for {len(line_miss)} date(s): {line_miss[:10]}{'...' if len(line_miss) > 10 else ''}")
        if args.test and (outcomes.empty or ml.empty or lines.empty):
            print("  Error: --test requires outcomes, ML, and lines for that date. Exit.")
            return

        merged = merge_outcomes_ml_lines(outcomes, ml, lines)
        if merged.empty:
            print("No merged games. Exit.")
            return
        n_outcomes = len(outcomes)
        n_merged = len(merged)
        n_with_spread = int(merged["consensus_spread"].notna().sum())
        print(f"  Merged: {n_merged} games, with spread: {n_with_spread}")
        if n_outcomes != n_merged:
            print(f"  Issue: {n_outcomes} outcomes but {n_merged} merged ({n_outcomes - n_merged} games missing ML or team match)")
        if n_merged != n_with_spread:
            print(f"  Issue: {n_merged} merged but {n_with_spread} have spread ({n_merged - n_with_spread} missing spread)")

        underdogs = create_underdog_dataset(merged)
        if underdogs.empty:
            print("No underdog rows (spread required). Exit.")
            return
        n_underdogs = len(underdogs)
        underdogs = add_implied_prob_buckets(underdogs)
        print(f"  Underdog rows: {n_underdogs}")
        if n_with_spread != n_underdogs:
            print(f"  Issue: {n_with_spread} games with spread but {n_underdogs} underdog rows ({n_with_spread - n_underdogs} dropped: spread push or NaN)")

        if args.seasons and not args.test:
            for season in args.seasons:
                cfg = get_season_dates("ncaab", season)
                start = pd.to_datetime(cfg["season_start"]).date()
                end = pd.to_datetime(cfg["tournament_end"]).date()
                mask = (underdogs["game_date"] >= start) & (underdogs["game_date"] <= end)
                season_ud = underdogs[mask]
                if not season_ud.empty:
                    _cache_path_for_season(season).parent.mkdir(parents=True, exist_ok=True)
                    season_ud.to_parquet(_cache_path_for_season(season), index=False)
                    print(f"  Saved cache: {_cache_path_for_season(season).name} ({len(season_ud)} rows)")

    _log_implied_above_50(underdogs)
    if args.underdog_only:
        before = len(underdogs)
        underdogs = underdogs[underdogs["implied_win_prob"] < 0.50].copy()
        print(f"  --underdog-only: kept {len(underdogs)} rows (dropped {before - len(underdogs)} with implied >= 50%%)")
    if args.underdog_only and underdogs.empty:
        print("No rows left after --underdog-only. Exit.")
        return

    if args.test:
        _debug_test_run(underdogs)

    if args.analyze_only:
        run_analyze_only(underdogs)
        return

    unit_frac = args.unit_fraction if args.bankroll_mode in ("fractional", "both") else None
    if args.bankroll_mode == "both":
        print(f"\nBankroll mode: empirical (flat 1u/game) + fractional (unit_fraction={args.unit_fraction})")

    if args.mode in ("generalized", "all"):
        gen = run_generalized(underdogs, unit_fraction=unit_frac, bankroll_mode=args.bankroll_mode)
        print("\n--- Generalized ---")
        print(f"  optimal_ml_weight={gen['optimal_ml_weight']:.2f}, log_growth={gen['log_growth']:.3f}, n={gen['sample_size']}")
        print(f"  spread_cover_rate={gen.get('spread_cover_rate', np.nan):.1%}, ml_win_rate={gen.get('ml_win_rate', np.nan):.1%}")
        print(f"  total_return_pct (compound)={gen['total_return_pct']:.1f}%  flat_return_pct (1u/game ROI)={gen.get('flat_return_pct', np.nan):.2f}%")
        print(f"  units: max={gen.get('max_units', np.nan):.2f}, min={gen.get('min_units', np.nan):.2f}, avg={gen.get('avg_units', np.nan):.2f}, std={gen.get('std_units', np.nan):.2f}")
        if unit_frac is not None and "final_bankroll_multiple" in gen:
            print(f"  fractional (unit_fraction={unit_frac}): final_bankroll={gen['final_bankroll_multiple']:.3f}, went_bust={gen['went_bust']}, min_ratio={gen['min_bankroll_ratio']:.3f}, max_drawdown_pct={gen['max_drawdown_pct']:.1f}%")

    if args.mode in ("by_bucket", "all"):
        by_bucket = run_by_bucket(underdogs, unit_fraction=unit_frac, bankroll_mode=args.bankroll_mode)
        if not by_bucket.empty:
            bucket_order = [f"{int(l*100)}-{int(h*100)}%" for l, h in IMPLIED_PROB_BUCKETS]
            by_bucket["_order"] = by_bucket["implied_prob_bucket"].apply(lambda b: bucket_order.index(b) if b in bucket_order else 999)
            by_bucket = by_bucket.sort_values("_order").drop(columns=["_order"])
            print("\n--- By implied_prob_bucket ---")
            cols = ["implied_prob_bucket", "sample_size", "spread_cover_rate", "ml_win_rate", "optimal_ml_weight", "optimal_spread_weight", "log_growth", "flat_return_pct", "max_units", "min_units", "avg_units", "std_units", "total_return_pct", "note"]
            if unit_frac is not None:
                cols = cols + ["final_bankroll_multiple", "went_bust", "min_bankroll_ratio", "max_drawdown_pct"]
            cols = [c for c in cols if c in by_bucket.columns]
            # Round numeric columns for readable output (2-3 decimals)
            round_cols = {c: 2 for c in ["spread_cover_rate", "ml_win_rate", "optimal_ml_weight", "optimal_spread_weight", "flat_return_pct", "max_units", "min_units", "avg_units", "std_units", "max_drawdown_pct"] if c in cols}
            round_cols.update({c: 3 for c in ["log_growth", "final_bankroll_multiple", "min_bankroll_ratio"] if c in cols})
            round_cols["total_return_pct"] = 1
            to_print = by_bucket[cols].round(round_cols) if round_cols else by_bucket[cols]
            # Short display names for terminal fit
            display_rename = {
                "implied_prob_bucket": "bucket",
                "sample_size": "n",
                "optimal_ml_weight": "opt_ml_wt",
                "optimal_spread_weight": "opt_spread_wt",
                "log_growth": "log_g",
                "flat_return_pct": "flat_roi",
                "max_units": "max_u", "min_units": "min_u", "avg_units": "avg_u", "std_units": "std_u",
                "total_return_pct": "total_return_pct_compound",
                "final_bankroll_multiple": "final_br",
                "min_bankroll_ratio": "min_br",
                "max_drawdown_pct": "max_dd",
            }
            to_print = to_print.rename(columns={k: v for k, v in display_rename.items() if k in to_print.columns})
            print(to_print.to_string(index=False))

            # Second table: per bucket, all weight combos sorted best to worst (max 10 buckets * 11 = 110 rows)
            combos = run_weight_combos_by_bucket(underdogs)
            if not combos.empty:
                print("\n--- By bucket: weight combos (best to worst per bucket) ---")
                combo_cols = ["implied_prob_bucket", "w_ml", "w_spread", "log_growth", "flat_return_pct"]
                combo_round = {"w_ml": 2, "w_spread": 2, "log_growth": 3, "flat_return_pct": 2}
                combos_print = combos[combo_cols].round(combo_round).rename(columns={"implied_prob_bucket": "bucket", "log_growth": "log_g", "flat_return_pct": "flat_roi"})
                print(combos_print.to_string(index=False))
            if args.plot:
                out_dir = Path(args.plot_dir) if args.plot_dir else ROOT / "content" / "viz" / "ncaab"
                generate_plots(by_bucket, combos if not combos.empty else pd.DataFrame(), out_dir)
        else:
            print("\nNo implied_prob_bucket groups (all underdog rows had no bucket?).")


if __name__ == "__main__":
    main()
