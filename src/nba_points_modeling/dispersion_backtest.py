"""
NBA Points Dispersion Backtest
==============================
Strategy: bet UNDER on non-star teammates in a team's next game after a star
posts a dominant individual performance (residual > σ threshold above rolling mean).

Derived from v2_dispersion_backtest.py (strict controls, closing lines, name match
audit) and v3_dispersion_signal.py (σ sensitivity grid, spread/HHI conditioning).
σ=1.0 selected as the max-units configuration (+1,734u pooled across 3 seasons).
Rolling spread and HHI are not applied as filters — r²=0.001 for both (not signal).

Controls:
  - Closing prop lines only — no rolling mean fallback
  - All rolling stats use shift(1) — no current-game data leaks
  - Minimum 10 prior games required before a bet is placed
  - Next game must be within 5 calendar days (no bets after a long break)
  - Stars = top 3 players by games played per team×season

Line timing: CLOSING. If books partially adjust teammate lines after a star's
dominant game, the measured edge here is a lower bound on the opening-line edge.

Usage:
    cd /path/to/repo
    python src/nba_points_modeling/dispersion_backtest.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# CONSTANTS
# =============================================================================

SEASONS = ["2023-24", "2024-25", "2025-26"]
MIN_MINUTES = 5
POINTS_MARKET_SUBSTR = "point"
ROLLING_WINDOW = 10
MIN_GAMES_FOR_ROLLING = 10
STAR_THRESHOLD_SIGMA = 1.0
MAX_GAME_GAP_DAYS = 5
SPREAD_ROLL_WINDOW = 15
BREAKEVEN_110 = 1 / (1 + 1 / 1.1)  # 0.5238

SPREAD_Q_LABELS = {
    1: "Q1 — Elite favorite   (roll spread most negative, dominant team)",
    2: "Q2 — Moderate favorite",
    3: "Q3 — Neutral / toss-up",
    4: "Q4 — Moderate underdog",
    5: "Q5 — Heavy underdog   (roll spread most positive, weak team)",
}


def _find_repo_root() -> Path:
    current = Path.cwd().resolve()
    while True:
        if (current / ".gitignore").exists() and (current / "src").exists():
            if str(current) not in sys.path:
                sys.path.insert(0, str(current))
            return current
        if current.parent == current:
            raise FileNotFoundError("Could not locate repo root")
        current = current.parent


REPO_ROOT = _find_repo_root()
OUTPUT_DIR = REPO_ROOT / "src" / "nba_points_modeling" / "research" / "outputs" / "dispersion"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from src.nba_data.get_data import get_data

    print("Loading NBA data ...")
    data = get_data(min_minutes=MIN_MINUTES)

    logs = data.logs.copy()
    logs.columns = logs.columns.str.lower()
    logs = logs.rename(columns={"min": "minutes"})
    logs["team_name"] = logs["team_normalized"]
    logs["game_date"] = pd.to_datetime(logs["game_date"]).dt.date
    logs = logs.sort_values(["player_id", "game_date"]).reset_index(drop=True)

    props_raw = data.props.copy()
    pts_props = props_raw[
        props_raw["market"].str.lower().str.contains(POINTS_MARKET_SUBSTR, na=False)
    ].copy()
    pts_props["game_date"] = pd.to_datetime(pts_props["game_date"]).dt.date

    # Median closing line per player×game across bookmakers
    props = (
        pts_props.groupby(["player_normalized", "game_date", "season"])
        .agg(prop_line=("prop_line", "median"), n_books=("bookmaker", "nunique"))
        .reset_index()
    )

    lines = data.lines.copy()
    lines["game_date"] = pd.to_datetime(lines["game_date"]).dt.date

    print(f"  Logs:  {len(logs):,} player-games | {logs['player_id'].nunique()} players")
    print(f"  Props: {len(props):,} player×game lines ({props['n_books'].mean():.1f} books avg)")
    print(f"  Lines: {len(lines):,} rows")
    return logs, props, lines


def build_roll_spread(lines: pd.DataFrame, logs: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling median signed spread per team-game over prior SPREAD_ROLL_WINDOW games.
    Negative = team is favored; positive = team is underdog.
    Returns a game_id × team_name DataFrame with roll_spread column.
    """
    spread = lines[lines["market"] == "spread"].copy()
    home = (
        spread.groupby(["home_team", "game_date", "season"])["home_line"]
        .median().reset_index()
        .rename(columns={"home_team": "team_name", "home_line": "spread_signed"})
    )
    away = (
        spread.groupby(["away_team", "game_date", "season"])["away_line"]
        .median().reset_index()
        .rename(columns={"away_team": "team_name", "away_line": "spread_signed"})
    )
    team_spreads = pd.concat([home, away], ignore_index=True)
    team_spreads["game_date"] = pd.to_datetime(team_spreads["game_date"]).dt.date

    team_games = (
        logs[["game_id", "team_name", "game_date", "season"]]
        .drop_duplicates()
        .merge(team_spreads, on=["team_name", "game_date", "season"], how="left")
        .sort_values(["team_name", "season", "game_date"])
    )
    team_games["roll_spread"] = (
        team_games.groupby(["team_name", "season"])["spread_signed"]
        .transform(lambda s: s.shift(1).rolling(SPREAD_ROLL_WINDOW, min_periods=max(3, SPREAD_ROLL_WINDOW // 3)).mean())
    )
    return team_games[["game_id", "team_name", "roll_spread"]]


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def build_features(logs: pd.DataFrame) -> pd.DataFrame:
    df = logs.sort_values(["player_id", "game_date"]).copy()

    team_totals = (
        df.groupby(["game_id", "team_name"])["pts"].sum().reset_index(name="team_pts")
    )
    df = df.merge(team_totals, on=["game_id", "team_name"], how="left")
    df["pts_share"] = df["pts"] / df["team_pts"].replace(0, np.nan)

    def _roll(series: pd.Series, window: int) -> pd.Series:
        return series.shift(1).rolling(window, min_periods=MIN_GAMES_FOR_ROLLING).mean()

    df[f"roll{ROLLING_WINDOW}_pts"] = (
        df.groupby("player_id")["pts"].transform(lambda s: _roll(s, ROLLING_WINDOW))
    )
    df["games_played_prior"] = df.groupby("player_id").cumcount()
    df["resid10"] = df["pts"] - df[f"roll{ROLLING_WINDOW}_pts"]
    return df


def identify_stars(df: pd.DataFrame) -> pd.DataFrame:
    gp = (
        df.groupby(["player_id", "team_name", "season"])["game_id"]
        .count()
        .reset_index(name="gp")
    )
    stars = (
        gp.sort_values(["team_name", "season", "gp"], ascending=[True, True, False])
        .groupby(["team_name", "season"])
        .head(3)
        .assign(is_star=True)
    )
    df = df.merge(
        stars[["player_id", "team_name", "season", "is_star"]],
        on=["player_id", "team_name", "season"],
        how="left",
    )
    df["is_star"] = df["is_star"].fillna(False)
    return df


# =============================================================================
# BET CONSTRUCTION
# =============================================================================

def build_bets(df: pd.DataFrame, props: pd.DataFrame) -> pd.DataFrame:
    star_resid_std = df[df["is_star"] & df["resid10"].notna()]["resid10"].std()
    threshold = STAR_THRESHOLD_SIGMA * star_resid_std
    print(f"\n  Star residual σ={star_resid_std:.2f}  threshold={threshold:.2f} pts  (σ={STAR_THRESHOLD_SIGMA})")

    star_nights = df[
        df["is_star"]
        & (df["resid10"] > threshold)
        & df[f"roll{ROLLING_WINDOW}_pts"].notna()
        & (df["games_played_prior"] >= MIN_GAMES_FOR_ROLLING)
    ][["game_id", "team_name", "season", "game_date", "player_id"]].rename(
        columns={"player_id": "star_id"}
    )
    print(f"  Star nights (history gate applied): {len(star_nights):,}")

    team_games = (
        df[["team_name", "season", "game_date", "game_id"]]
        .drop_duplicates()
        .sort_values(["team_name", "season", "game_date"])
        .copy()
    )
    team_games["next_game_date"] = (
        team_games.groupby(["team_name", "season"])["game_date"].shift(-1)
    )
    team_games["next_game_id"] = (
        team_games.groupby(["team_name", "season"])["game_id"].shift(-1)
    )

    signal = star_nights.merge(
        team_games[["team_name", "season", "game_date", "next_game_date", "next_game_id"]],
        on=["team_name", "season", "game_date"],
    ).dropna(subset=["next_game_id"])

    signal["game_date"] = pd.to_datetime(signal["game_date"])
    signal["next_game_date"] = pd.to_datetime(signal["next_game_date"])
    signal["gap_days"] = (signal["next_game_date"] - signal["game_date"]).dt.days
    signal = signal[signal["gap_days"] <= MAX_GAME_GAP_DAYS]
    print(f"  Signal events after {MAX_GAME_GAP_DAYS}-day gap filter: {len(signal):,}")

    # Rename so trigger columns don't collide with df columns in the merge below
    signal = signal.rename(columns={"game_id": "trigger_game_id", "game_date": "trigger_game_date"})

    bets = df.merge(
        signal[["team_name", "season", "next_game_id", "star_id", "trigger_game_id", "trigger_game_date", "gap_days"]],
        left_on=["team_name", "season", "game_id"],
        right_on=["team_name", "season", "next_game_id"],
    )
    bets = bets[
        (bets["player_id"] != bets["star_id"])
        & (~bets["is_star"])
        & bets[f"roll{ROLLING_WINDOW}_pts"].notna()
        & (bets["games_played_prior"] >= MIN_GAMES_FOR_ROLLING)
    ]

    bets = bets.merge(
        props[["player_normalized", "game_date", "prop_line"]],
        on=["player_normalized", "game_date"],
        how="left",
    )
    bets_with_line = bets.dropna(subset=["prop_line"]).copy()
    coverage = len(bets_with_line) / len(bets) if len(bets) > 0 else 0
    print(f"  Candidates: {len(bets):,}  with prop line: {len(bets_with_line):,} ({coverage:.1%} coverage)")

    bets_with_line["under_win"] = (bets_with_line["pts"] < bets_with_line["prop_line"]).astype(int)
    return bets_with_line


# =============================================================================
# RESULTS
# =============================================================================

def _pnl_series(under_win: pd.Series) -> pd.Series:
    """Cumulative units P&L at -110: win=+10/11, loss=-1."""
    return under_win.map(lambda x: 10 / 11 if x == 1 else -1).cumsum()


def _drawdown(pnl: pd.Series) -> float:
    return (pnl - pnl.cummax()).min()


def print_results(bets: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    if bets.empty:
        print("  No bets.")
        return

    header = f"  {'Season':<10} {'N':>6}  {'WR':>7}  {'Edge':>7}  {'n_units':>9}  {'max_up':>8}  {'max_dd':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for season in SEASONS:
        sub = bets[bets["season"] == season].sort_values("game_date").copy()
        if sub.empty:
            continue
        wr = sub["under_win"].mean()
        edge = wr - BREAKEVEN_110
        pnl = _pnl_series(sub["under_win"])
        n_units = pnl.iloc[-1]
        max_up = pnl.max()
        max_dd = _drawdown(pnl)
        print(f"  {season:<10} {len(sub):>6,}  {wr:>7.4f}  {edge:>+7.4f}  {n_units:>+9.1f}u  {max_up:>+8.1f}u  {max_dd:>8.1f}u")

    pooled = bets.sort_values("game_date").copy()
    wr_all = pooled["under_win"].mean()
    edge_all = wr_all - BREAKEVEN_110
    pnl_all = _pnl_series(pooled["under_win"])
    n_units_all = pnl_all.iloc[-1]
    max_up_all = pnl_all.max()
    max_dd_all = _drawdown(pnl_all)

    print("  " + "-" * (len(header) - 2))
    print(f"  {'Pooled':<10} {len(pooled):>6,}  {wr_all:>7.4f}  {edge_all:>+7.4f}  {n_units_all:>+9.1f}u  {max_up_all:>+8.1f}u  {max_dd_all:>8.1f}u")
    print(f"\n  Breakeven at -110: {BREAKEVEN_110:.4f}  |  σ threshold: {STAR_THRESHOLD_SIGMA}")


def plot_pnl(bets: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Pooled cumulative P&L
    pooled = bets.sort_values("game_date").copy()
    pnl_all = _pnl_series(pooled["under_win"])
    axes[0].plot(np.arange(len(pnl_all)), pnl_all.values, linewidth=1.2)
    axes[0].axhline(0, color="red", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("Bet number (chronological)")
    axes[0].set_ylabel("Cumulative units")
    axes[0].set_title(f"Cumulative P&L — All Seasons (σ={STAR_THRESHOLD_SIGMA})")

    # Per-season cumulative P&L
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    for i, season in enumerate(SEASONS):
        sub = bets[bets["season"] == season].sort_values("game_date").copy()
        if sub.empty:
            continue
        pnl = _pnl_series(sub["under_win"])
        axes[1].plot(np.arange(len(pnl)), pnl.values, color=colors[i], label=season)
    axes[1].axhline(0, color="red", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Bet number within season")
    axes[1].set_ylabel("Cumulative units")
    axes[1].set_title("Cumulative P&L — By Season")
    axes[1].legend()

    fig.tight_layout()
    out = OUTPUT_DIR / "dispersion_pnl.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\n  Chart → {out}")


# =============================================================================
# EXAMPLES
# =============================================================================

def show_examples(bets: pd.DataFrame, df: pd.DataFrame, n: int) -> None:
    if n == 0 or bets.empty:
        return

    print("\n" + "=" * 60)
    print(f"EXAMPLES  (most recent {n})")
    print("=" * 60)

    star_resid_std = df[df["is_star"] & df["resid10"].notna()]["resid10"].std()
    threshold = STAR_THRESHOLD_SIGMA * star_resid_std

    # Compute spread quintile labels from the full bets pool (display only)
    bets_with_spread = bets.dropna(subset=["roll_spread"])
    if len(bets_with_spread) > 0:
        _, q_bins = pd.qcut(bets_with_spread["roll_spread"], q=5, retbins=True, labels=False)
    else:
        q_bins = None

    def _spread_q(roll_spread_val: float) -> str:
        if q_bins is None or pd.isna(roll_spread_val):
            return "Q? (no spread data)"
        q = int(np.searchsorted(q_bins[1:-1], roll_spread_val)) + 1
        q = max(1, min(5, q))
        return f"Q{q}  {roll_spread_val:+.1f} pts   {SPREAD_Q_LABELS[q]}"

    events = (
        bets[["trigger_game_id", "trigger_game_date", "star_id", "team_name", "season", "gap_days"]]
        .drop_duplicates(subset=["trigger_game_id", "star_id"])
        .sort_values("trigger_game_date", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )

    for i, event in events.iterrows():
        star_row = df[
            (df["game_id"] == event["trigger_game_id"]) &
            (df["player_id"] == event["star_id"])
        ]
        if star_row.empty:
            continue

        star = star_row.iloc[0]
        star_name = star["player_normalized"]
        event_bets = (
            bets[
                (bets["trigger_game_id"] == event["trigger_game_id"]) &
                (bets["star_id"] == event["star_id"])
            ]
            .sort_values("player_normalized")
            .copy()
        )

        bet_date = event_bets["game_date"].iloc[0]
        gap = int(event["gap_days"])
        n_won = int(event_bets["under_win"].sum())
        n_total = len(event_bets)
        star_prior = int(star["games_played_prior"])
        team_roll_spread = event_bets["roll_spread"].iloc[0]
        spread_q_str = _spread_q(team_roll_spread)

        print(f"\n  {'─' * 56}")
        print(f"  Example {i + 1}")
        print(f"  Trigger  {star_name} ({event['team_name']})  —  {event['trigger_game_date'].date()}")
        print(f"           roll_avg={star[f'roll{ROLLING_WINDOW}_pts']:.1f} pts  actual={int(star['pts'])} pts")
        print(f"  Hurdles  "
              f"[1] resid={star['resid10']:+.1f} > {threshold:+.1f} (σ={STAR_THRESHOLD_SIGMA}) ✓  "
              f"[2] {star_prior} prior games ≥ {MIN_GAMES_FOR_ROLLING} ✓  "
              f"[3] next game {gap}d ≤ {MAX_GAME_GAP_DAYS}d ✓")
        print(f"  Team Q   {spread_q_str}")
        print(f"  Bets     {event['team_name']} next game  —  {bet_date}")
        print()

        name_w = max(len(str(r["player_normalized"])) for _, r in event_bets.iterrows()) + 2
        print(f"  {'Player':<{name_w}}  {'prior_g':>7}  {'roll_avg':>9}  {'prop_line':>9}  {'actual':>6}  result")
        print(f"  {'-' * name_w}  {'-' * 7}  {'-' * 9}  {'-' * 9}  {'-' * 6}  ------")
        for _, row in event_bets.iterrows():
            result = "WIN " if row["under_win"] == 1 else "LOSS"
            print(f"  {row['player_normalized']:<{name_w}}  "
                  f"{int(row['games_played_prior']):>7}  "
                  f"{row[f'roll{ROLLING_WINDOW}_pts']:>9.1f}  "
                  f"{row['prop_line']:>9.1f}  "
                  f"{int(row['pts']):>6}  {result}")

        print(f"\n  Result: {n_won}/{n_total} won")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="NBA points dispersion backtest")
    parser.add_argument("--examples", type=int, default=0, metavar="N",
                        help="print N most-recent example events (default: 0)")
    args = parser.parse_args()

    print("=" * 60)
    print("NBA POINTS DISPERSION BACKTEST")
    print(f"  σ={STAR_THRESHOLD_SIGMA}         star night trigger: player residual must exceed σ × rolling std")
    print(f"  min_games={MIN_GAMES_FOR_ROLLING}   require this many prior games before any rolling stat is trusted")
    print(f"  max_gap={MAX_GAME_GAP_DAYS}d       skip bets if the next team game is more than {MAX_GAME_GAP_DAYS} days away (breaks/bye)")
    print("=" * 60)

    logs, props, lines = load_data()
    df = build_features(logs)
    df = identify_stars(df)

    print("\n" + "=" * 60)
    print("BET CONSTRUCTION")
    print("=" * 60)
    bets = build_bets(df, props)

    # Attach roll_spread to bets (display-only — not used as a filter)
    roll_spread = build_roll_spread(lines, logs)
    bets = bets.merge(roll_spread, on=["game_id", "team_name"], how="left")

    print_results(bets)
    plot_pnl(bets)
    show_examples(bets, df, args.examples)

    print("\n" + "=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
