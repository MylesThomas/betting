"""
NBA Points Dispersion — v2 Backtest with Strict Controls
=========================================================
Date: 2026-05-26
Follows: 20260526_dispersion.py (v1 exploratory)

Phase 2 gates (from plan):
  Walk-forward ROI > 0 across all 3 seasons at n >= 200/season

Controls added vs v1:
  - Actual closing prop lines only — no rolling mean fallback
  - Name match rate audit runs first; bad match rate kills the analysis
  - Strict temporal ordering: all rolling stats use shift(1) — no current-game data leaks
  - 'Next game' = team's immediately next game_date (gap capped at 5 days)
  - Min 10 prior games required for rolling stats to be valid before a bet is placed
  - Selection bias probe: compare prop-covered vs uncovered players

Line timing: CLOSING (confirmed). If the book partially adjusts teammate lines
downward after a star's prior dominant game, our measured edge is a lower bound
on the true opening-line edge.

Usage:
    cd /path/to/repo
    python src/nba_points_modeling/research/scripts/v2_dispersion_backtest.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# CONSTANTS
# =============================================================================

SEASONS = ["2023-24", "2024-25", "2025-26"]
MIN_MINUTES = 5
POINTS_MARKET_SUBSTR = "point"
ROLLING_WINDOW = 10
MIN_GAMES_FOR_ROLLING = 10        # require this many prior games before a bet is placed
STAR_THRESHOLD_SIGMA = 1.5        # star night = resid > 1.5σ above their own rolling mean
MAX_GAME_GAP_DAYS = 5             # next game must be within 5 days (not after a bye/break)
BREAKEVEN_110 = 1 / (1 + 1 / 1.1)  # 0.5238 at -110


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
OUTPUT_DIR = REPO_ROOT / "src" / "nba_points_modeling" / "research" / "outputs" / "dispersion_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# SECTION 1 — DATA LOADING
# =============================================================================

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    from src.nba_data.get_data import get_data

    print("Loading NBA data from cache ...")
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

    # Closing line: median across bookmakers per player×game (already closing per confirmation)
    props = (
        pts_props.groupby(["player_normalized", "game_date", "season"])
        .agg(prop_line=("prop_line", "median"), n_books=("bookmaker", "nunique"))
        .reset_index()
    )

    print(f"  Logs:  {len(logs):,} player-games | {logs['player_id'].nunique()} players")
    print(f"  Props: {len(props):,} player×game lines ({props['n_books'].mean():.1f} books avg)")
    return logs, props


# =============================================================================
# SECTION 2 — NAME MATCH RATE AUDIT
# =============================================================================

def section2_name_match_audit(logs: pd.DataFrame, props: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("SECTION 2: NAME MATCH RATE AUDIT")
    print("=" * 60)

    merged = logs.merge(
        props[["player_normalized", "game_date", "prop_line"]],
        on=["player_normalized", "game_date"],
        how="left",
    )
    merged["has_prop"] = merged["prop_line"].notna()

    total = len(merged)
    matched = merged["has_prop"].sum()
    print(f"\n  Overall: {matched:,} / {total:,} player-games have a prop line ({matched/total:.1%})")

    by_season = merged.groupby("season")["has_prop"].agg(["sum", "count"])
    by_season["pct"] = by_season["sum"] / by_season["count"]
    print("\n  By season:")
    for season, row in by_season.iterrows():
        print(f"    {season}: {int(row['sum']):,} / {int(row['count']):,} ({row['pct']:.1%})")

    # Players with most unmatched games — likely normalization failures
    player_match = (
        merged.groupby("player_normalized")["has_prop"]
        .agg(matched="sum", total="count")
        .assign(pct=lambda d: d["matched"] / d["total"])
        .reset_index()
    )
    never_matched = player_match[player_match["matched"] == 0].sort_values("total", ascending=False)
    low_match = player_match[
        (player_match["pct"] < 0.30) & (player_match["total"] >= 20)
    ].sort_values("pct")

    print(f"\n  Players with 0 prop matches (top 20 by games played):")
    print(never_matched.head(20).to_string(index=False))

    print(f"\n  Players with <30% match rate and >= 20 games (likely norm failures):")
    if low_match.empty:
        print("  None — name normalization looks clean.")
    else:
        print(low_match.to_string(index=False))

    # Cross-check: how many unique player_normalized in props don't appear in logs?
    props_players = set(props["player_normalized"].dropna().unique())
    log_players = set(logs["player_normalized"].dropna().unique())
    props_only = props_players - log_players
    logs_only = log_players - props_players
    print(f"\n  player_normalized in props but not in logs: {len(props_only)}")
    if props_only:
        sample = sorted(props_only)[:20]
        print(f"    Sample: {sample}")
    print(f"  player_normalized in logs but not in props: {len(logs_only)}")

    return merged


# =============================================================================
# SECTION 3 — FEATURE ENGINEERING (STRICT TEMPORAL)
# =============================================================================

def build_features(logs: pd.DataFrame) -> pd.DataFrame:
    """
    All rolling stats use shift(1) — no current-game data leaks into any feature.
    Requires MIN_GAMES_FOR_ROLLING prior games before a feature is considered valid.
    """
    df = logs.sort_values(["player_id", "game_date"]).copy()

    # Team total per game (needed for share)
    team_totals = (
        df.groupby(["game_id", "team_name"])["pts"].sum().reset_index(name="team_pts")
    )
    df = df.merge(team_totals, on=["game_id", "team_name"], how="left")
    df["pts_share"] = df["pts"] / df["team_pts"].replace(0, np.nan)

    # Rolling stats (all shift(1) — prior games only)
    def _roll(series: pd.Series, window: int) -> pd.Series:
        return series.shift(1).rolling(window, min_periods=MIN_GAMES_FOR_ROLLING).mean()

    df[f"roll{ROLLING_WINDOW}_pts"] = (
        df.groupby("player_id")["pts"].transform(lambda s: _roll(s, ROLLING_WINDOW))
    )
    df[f"roll{ROLLING_WINDOW}_share"] = (
        df.groupby("player_id")["pts_share"].transform(lambda s: _roll(s, ROLLING_WINDOW))
    )

    # Games played counter (prior games, for minimum-history gate)
    df["games_played_prior"] = (
        df.groupby("player_id").cumcount()  # 0-indexed, so game 0 = 0 prior games
    )

    df["resid10"] = df["pts"] - df[f"roll{ROLLING_WINDOW}_pts"]
    return df


def identify_stars(df: pd.DataFrame) -> pd.DataFrame:
    """
    Star = top 3 players by total games played per team×season.
    Minor season-level lookahead for early-season games (< ~20 games in).
    Flagged as a known caveat — does not affect mid/late-season bets.
    """
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
# SECTION 4 — BET CONSTRUCTION
# =============================================================================

def build_bets(df: pd.DataFrame, props: pd.DataFrame) -> pd.DataFrame:
    """
    Strategy: bet UNDER on non-star teammates in a team's next game
    after a star posted resid10 > STAR_THRESHOLD_SIGMA × σ.

    Strict controls:
    - Rolling stats must be non-null (requires MIN_GAMES_FOR_ROLLING prior games)
    - Next game must be within MAX_GAME_GAP_DAYS days
    - Actual closing prop line required — no fallback
    """
    # Sigma computed once from all star residuals (blind to current game)
    star_resid_std = df[df["is_star"] & df["resid10"].notna()]["resid10"].std()
    threshold = STAR_THRESHOLD_SIGMA * star_resid_std
    print(f"\n  Star residual σ = {star_resid_std:.2f} pts | threshold = {threshold:.2f} pts")

    # Identify star nights
    star_nights = df[
        df["is_star"]
        & (df["resid10"] > threshold)
        & df[f"roll{ROLLING_WINDOW}_pts"].notna()
        & (df["games_played_prior"] >= MIN_GAMES_FOR_ROLLING)
    ][["game_id", "team_name", "season", "game_date", "player_id"]].rename(
        columns={"player_id": "star_id"}
    )
    print(f"  Star nights qualifying (history gate applied): {len(star_nights):,}")

    # Map each star night to the team's next game
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

    # Enforce max gap — no bets after a long break
    signal["game_date"] = pd.to_datetime(signal["game_date"])
    signal["next_game_date"] = pd.to_datetime(signal["next_game_date"])
    signal["gap_days"] = (signal["next_game_date"] - signal["game_date"]).dt.days
    signal = signal[signal["gap_days"] <= MAX_GAME_GAP_DAYS]
    print(f"  Signal events after gap filter ({MAX_GAME_GAP_DAYS}d): {len(signal):,}")

    # Bet: UNDER on non-star teammates in the next game
    bets = df.merge(
        signal[["team_name", "season", "next_game_id", "star_id"]],
        left_on=["team_name", "season", "game_id"],
        right_on=["team_name", "season", "next_game_id"],
    )
    bets = bets[
        (bets["player_id"] != bets["star_id"])  # exclude the triggering star
        & (~bets["is_star"])                     # exclude other stars (role players only)
        & bets[f"roll{ROLLING_WINDOW}_pts"].notna()  # require rolling history
        & (bets["games_played_prior"] >= MIN_GAMES_FOR_ROLLING)
    ]

    # Attach actual closing prop lines — no fallback
    bets = bets.merge(
        props[["player_normalized", "game_date", "prop_line"]],
        on=["player_normalized", "game_date"],
        how="left",
    )
    bets_with_line = bets.dropna(subset=["prop_line"])
    prop_coverage = len(bets_with_line) / len(bets) if len(bets) > 0 else 0
    print(f"  Candidate bets: {len(bets):,} | with prop line: {len(bets_with_line):,} ({prop_coverage:.1%} coverage)")

    bets_with_line = bets_with_line.copy()
    bets_with_line["under_win"] = (bets_with_line["pts"] < bets_with_line["prop_line"]).astype(int)
    return bets_with_line


# =============================================================================
# SECTION 5 — WALK-FORWARD RESULTS
# =============================================================================

def section5_results(bets: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("SECTION 5: WALK-FORWARD RESULTS")
    print("=" * 60)

    if bets.empty:
        print("  No bets — check upstream filters.")
        return

    overall_wr = bets["under_win"].mean()
    edge = overall_wr - BREAKEVEN_110
    roi = edge * 2 * 100
    print(f"\n  Pooled: n={len(bets):,}  win_rate={overall_wr:.4f}  edge={edge:+.4f}  implied_ROI={roi:+.1f}%")

    print("\n  By season:")
    def _units_pnl(series: pd.Series) -> pd.Series:
        """Cumulative units P&L at -110: win = +10/11, loss = -1."""
        return series.map(lambda x: 10 / 11 if x == 1 else -1).cumsum()

    season_rows = []
    for season in SEASONS:
        sub = bets[bets["season"] == season].sort_values("game_date").copy()
        if sub.empty:
            continue
        wr = sub["under_win"].mean()
        e = wr - BREAKEVEN_110
        r = e * 2 * 100
        pnl = _units_pnl(sub["under_win"])
        max_up = pnl.max()
        # max drawdown: largest peak-to-trough in cumulative units
        roll_max = pnl.cummax()
        max_dd = (pnl - roll_max).min()
        n_units = pnl.iloc[-1]
        print(f"    {season}: n={len(sub):,}  win_rate={wr:.4f}  edge={e:+.4f}  "
              f"implied_ROI={r:+.1f}%  n_units={n_units:+.1f}u  "
              f"max_up={max_up:+.1f}u  max_dd={max_dd:.1f}u")
        season_rows.append({
            "season": season, "n": len(sub), "win_rate": wr, "edge": e, "roi": r,
            "n_units": n_units, "max_up": max_up, "max_dd": max_dd,
        })

    # Pooled drawdown
    bets_sorted_pnl = bets.sort_values("game_date").copy()
    pnl_all = _units_pnl(bets_sorted_pnl["under_win"])
    max_up_all = pnl_all.max()
    roll_max_all = pnl_all.cummax()
    max_dd_all = (pnl_all - roll_max_all).min()
    n_units_all = pnl_all.iloc[-1]
    print(f"    Pooled:  n_units={n_units_all:+.1f}u  max_up={max_up_all:+.1f}u  max_dd={max_dd_all:.1f}u")

    # Rolling win rate over time (pooled, sorted by date)
    bets_sorted = bets.sort_values("game_date").copy()
    bets_sorted["cum_wr"] = bets_sorted["under_win"].expanding().mean()
    bets_sorted["bet_index"] = np.arange(len(bets_sorted))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(bets_sorted["bet_index"], bets_sorted["cum_wr"], linewidth=1.2)
    axes[0].axhline(BREAKEVEN_110, color="red", linestyle="--", label=f"breakeven ({BREAKEVEN_110:.3f})")
    axes[0].axhline(0.5, color="grey", linestyle=":", linewidth=0.8, label="50%")
    axes[0].set_xlabel("Bet number (chronological)")
    axes[0].set_ylabel("Cumulative win rate")
    axes[0].set_title("Cumulative Win Rate — All Seasons")
    axes[0].legend()

    colors = ["#4C72B0", "#55A868", "#C44E52"]
    for i, season in enumerate(SEASONS):
        sub = bets[bets["season"] == season].sort_values("game_date").copy()
        if sub.empty:
            continue
        sub["cum_wr"] = sub["under_win"].expanding().mean()
        sub["bet_index"] = np.arange(len(sub))
        axes[1].plot(sub["bet_index"], sub["cum_wr"], color=colors[i], label=season)
    axes[1].axhline(BREAKEVEN_110, color="red", linestyle="--", label="breakeven")
    axes[1].set_xlabel("Bet number within season")
    axes[1].set_ylabel("Cumulative win rate")
    axes[1].set_title("Cumulative Win Rate — By Season")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "v2_cumulative_win_rate.png", dpi=120)
    plt.close(fig)
    print("  → v2_cumulative_win_rate.png")


# =============================================================================
# SECTION 6 — SELECTION BIAS PROBE
# =============================================================================

def section6_selection_bias(logs: pd.DataFrame, bets: pd.DataFrame, props: pd.DataFrame) -> None:
    """
    Compare players who appear in bets (had a prop line) vs those who didn't.
    If prop-covered players systematically under-perform vs no-prop players,
    it suggests the coverage itself is a confound.
    """
    print("\n" + "=" * 60)
    print("SECTION 6: SELECTION BIAS PROBE")
    print("=" * 60)

    # Rebuild candidate bets pool (with and without prop)
    logs_f = logs.copy()
    logs_f["game_date"] = pd.to_datetime(logs_f["game_date"]).dt.date
    logs_f = logs_f.merge(
        props[["player_normalized", "game_date", "prop_line"]],
        on=["player_normalized", "game_date"],
        how="left",
    )
    logs_f["has_prop"] = logs_f["prop_line"].notna()

    # Compare minutes and pts distribution: prop-covered vs not
    print("\n  Distribution comparison (prop-covered vs not):")
    for col in ["minutes", "pts"]:
        g1 = logs_f[logs_f["has_prop"]][col].dropna()
        g0 = logs_f[~logs_f["has_prop"]][col].dropna()
        t, p = stats.ttest_ind(g1, g0, equal_var=False)
        print(f"    {col:10}  with_prop: μ={g1.mean():.2f}  no_prop: μ={g0.mean():.2f}  "
              f"diff={g1.mean()-g0.mean():+.2f}  p={p:.4f}")

    # Among bet candidates (non-star teammates after star night): prop vs no-prop
    if not bets.empty:
        wr = bets["under_win"].mean()
        print(f"\n  Bet universe (has prop): n={len(bets):,}  win_rate={wr:.4f}")

    # Check: do players with higher rolling pts (more prominent) have higher prop coverage?
    logs_f["roll10_pts"] = (
        logs_f.sort_values(["player_id", "game_date"])
        .groupby("player_id")["pts"]
        .transform(lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=3).mean())
    )
    coverage_by_bucket = (
        logs_f.dropna(subset=["roll10_pts"])
        .assign(pts_bucket=lambda d: pd.qcut(d["roll10_pts"], q=5, labels=False))
        .groupby("pts_bucket")["has_prop"]
        .mean()
    )
    print("\n  Prop coverage rate by rolling-pts quintile (0=lowest, 4=highest):")
    print(coverage_by_bucket.round(3).to_string())
    print("  (Higher quintile = more prominent player = higher prop coverage expected)")


# =============================================================================
# SECTION 7 — SENSITIVITY ANALYSIS
# =============================================================================

def section7_sensitivity(df: pd.DataFrame, props: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("SECTION 7: SENSITIVITY ANALYSIS")
    print("=" * 60)

    star_resid_std = df[df["is_star"] & df["resid10"].notna()]["resid10"].std()

    results = []

    # Vary star threshold
    for sigma in [1.0, 1.25, 1.5, 1.75, 2.0]:
        threshold = sigma * star_resid_std
        bets = build_bets_silent(df, props, threshold=threshold)
        if len(bets) < 50:
            continue
        wr = bets["under_win"].mean()
        results.append({
            "variable": "sigma_threshold",
            "value": sigma,
            "n": len(bets),
            "win_rate": wr,
            "edge": wr - BREAKEVEN_110,
        })

    # Vary max gap
    for gap in [2, 3, 5, 7]:
        bets = build_bets_silent(df, props, max_gap=gap)
        if len(bets) < 50:
            continue
        wr = bets["under_win"].mean()
        results.append({
            "variable": "max_gap_days",
            "value": gap,
            "n": len(bets),
            "win_rate": wr,
            "edge": wr - BREAKEVEN_110,
        })

    res_df = pd.DataFrame(results)
    print("\n  Sensitivity results:")
    print(res_df.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, var in zip(axes, ["sigma_threshold", "max_gap_days"]):
        sub = res_df[res_df["variable"] == var]
        ax.bar(sub["value"].astype(str), sub["edge"] * 100, color="#4C72B0")
        ax.axhline(0, color="red", linestyle="--", linewidth=0.8)
        ax.set_xlabel(var)
        ax.set_ylabel("Implied edge (%)")
        ax.set_title(f"Edge by {var}")
        for _, row in sub.iterrows():
            ax.text(str(row["value"]), row["edge"] * 100 + 0.2, f"n={int(row['n'])}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "v2_sensitivity.png", dpi=120)
    plt.close(fig)
    print("  → v2_sensitivity.png")


def build_bets_silent(
    df: pd.DataFrame,
    props: pd.DataFrame,
    threshold: float | None = None,
    max_gap: int = MAX_GAME_GAP_DAYS,
) -> pd.DataFrame:
    """Like build_bets but without print output — used by sensitivity analysis."""
    star_resid_std = df[df["is_star"] & df["resid10"].notna()]["resid10"].std()
    _threshold = threshold if threshold is not None else STAR_THRESHOLD_SIGMA * star_resid_std

    star_nights = df[
        df["is_star"]
        & (df["resid10"] > _threshold)
        & df[f"roll{ROLLING_WINDOW}_pts"].notna()
        & (df["games_played_prior"] >= MIN_GAMES_FOR_ROLLING)
    ][["game_id", "team_name", "season", "game_date", "player_id"]].rename(
        columns={"player_id": "star_id"}
    )

    team_games = (
        df[["team_name", "season", "game_date", "game_id"]]
        .drop_duplicates()
        .sort_values(["team_name", "season", "game_date"])
        .copy()
    )
    team_games["next_game_date"] = team_games.groupby(["team_name", "season"])["game_date"].shift(-1)
    team_games["next_game_id"] = team_games.groupby(["team_name", "season"])["game_id"].shift(-1)

    signal = star_nights.merge(
        team_games[["team_name", "season", "game_date", "next_game_date", "next_game_id"]],
        on=["team_name", "season", "game_date"],
    ).dropna(subset=["next_game_id"])
    signal["game_date"] = pd.to_datetime(signal["game_date"])
    signal["next_game_date"] = pd.to_datetime(signal["next_game_date"])
    signal["gap_days"] = (signal["next_game_date"] - signal["game_date"]).dt.days
    signal = signal[signal["gap_days"] <= max_gap]

    bets = df.merge(
        signal[["team_name", "season", "next_game_id", "star_id"]],
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
    ).dropna(subset=["prop_line"])
    bets = bets.copy()
    bets["under_win"] = (bets["pts"] < bets["prop_line"]).astype(int)
    return bets


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("=" * 60)
    print("NBA POINTS DISPERSION v2 BACKTEST — STRICT CONTROLS")
    print("=" * 60)

    logs, props = load_data()

    merged = section2_name_match_audit(logs, props)

    # Check match rate is high enough to proceed
    match_rate = merged["has_prop"].mean()
    if match_rate < 0.40:
        print(f"\nFATAL: overall prop match rate {match_rate:.1%} < 40% — name normalization likely broken.")
        sys.exit(1)
    print(f"\n  Match rate {match_rate:.1%} — proceeding with backtest.")

    df = build_features(logs)
    df = identify_stars(df)

    print("\n" + "=" * 60)
    print("SECTION 3: FEATURE ENGINEERING (STRICT TEMPORAL)")
    print("=" * 60)
    valid = df[df[f"roll{ROLLING_WINDOW}_pts"].notna()]
    print(f"  Player-games with valid rolling stats: {len(valid):,} / {len(df):,} ({len(valid)/len(df):.1%})")
    print(f"  Stars identified: {df['is_star'].sum():,} player-game slots")

    print("\n" + "=" * 60)
    print("SECTION 4: BET CONSTRUCTION")
    print("=" * 60)
    bets = build_bets(df, props)

    section5_results(bets)
    section6_selection_bias(logs, bets, props)
    section7_sensitivity(df, props)

    print("\n" + "=" * 60)
    print("DONE — outputs written to:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
