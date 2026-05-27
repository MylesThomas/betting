"""
NBA Points Dispersion — v3 Signal Conditioning
===============================================
Date: 2026-05-26
Follows: v2_dispersion_backtest.py (Phase 2 gate passed)

Question: does the v2 signal (UNDER on role players after star night) concentrate
in specific team contexts?

Two conditioning dimensions:
  1. Team quality — rolling signed spread (negative = elite favorite, positive = bad underdog)
  2. Team type   — rolling HHI (low = balanced scoring, high = star-concentrated)

Hypothesis: the zero-sum suppression effect should be cleanest on:
  - Balanced teams: teammates actually share possessions; star dominance crowds others out
  - Weaker teams: books may price role player props less precisely; more mispricing available

Usage:
    cd /path/to/repo
    python src/nba_points_modeling/research/scripts/v3_dispersion_signal.py
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

SEASONS = ["2023-24", "2024-25", "2025-26"]
MIN_MINUTES = 5
POINTS_MARKET_SUBSTR = "point"
ROLLING_WINDOW = 10
MIN_GAMES_FOR_ROLLING = 10
STAR_THRESHOLD_SIGMA = 1.5
MAX_GAME_GAP_DAYS = 5
SPREAD_ROLL_WINDOW = 15        # rolling games for team quality signal
HHI_ROLL_WINDOW = 15
BREAKEVEN_110 = 1 / (1 + 1 / 1.1)


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
OUTPUT_DIR = REPO_ROOT / "src" / "nba_points_modeling" / "research" / "outputs" / "dispersion_v3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# SECTION 1 — DATA LOADING
# =============================================================================

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    props = (
        pts_props.groupby(["player_normalized", "game_date", "season"])
        .agg(prop_line=("prop_line", "median"), n_books=("bookmaker", "nunique"))
        .reset_index()
    )

    lines = data.lines.copy()
    lines["game_date"] = pd.to_datetime(lines["game_date"]).dt.date

    print(f"  Logs:  {len(logs):,} player-games")
    print(f"  Props: {len(props):,} player×game lines")
    print(f"  Lines: {len(lines):,} rows")
    return logs, props, lines


# =============================================================================
# SECTION 2 — TEAM CONTEXT FEATURES
# =============================================================================

def build_team_spread(lines: pd.DataFrame) -> pd.DataFrame:
    """
    One signed spread per team per game (median across bookmakers).
    Negative = team is favored; positive = team is underdog.
    """
    spread = lines[lines["market"] == "spread"].copy()

    home = (
        spread.groupby(["home_team", "game_date", "season"])["home_line"]
        .median()
        .reset_index()
        .rename(columns={"home_team": "team_name", "home_line": "spread_signed"})
    )
    away = (
        spread.groupby(["away_team", "game_date", "season"])["away_line"]
        .median()
        .reset_index()
        .rename(columns={"away_team": "team_name", "away_line": "spread_signed"})
    )
    team_spreads = pd.concat([home, away], ignore_index=True)
    team_spreads["game_date"] = pd.to_datetime(team_spreads["game_date"]).dt.date
    return team_spreads


def build_team_hhi(logs: pd.DataFrame) -> pd.DataFrame:
    """
    One HHI per team per game (same computation as v1/v2).
    """
    team_totals = logs.groupby(["game_id", "team_name"])["pts"].sum().reset_index(name="team_pts")
    merged = logs.merge(team_totals, on=["game_id", "team_name"])
    merged["pts_share"] = merged["pts"] / merged["team_pts"].replace(0, np.nan)

    def _hhi(grp):
        shares = grp["pts_share"].dropna().values
        return (shares ** 2).sum() if len(shares) >= 2 else np.nan

    hhi = (
        merged.groupby(["game_id", "team_name", "game_date", "season"])
        .apply(_hhi)
        .reset_index(name="hhi")
    )
    return hhi


def attach_rolling_team_context(
    logs: pd.DataFrame,
    team_spreads: pd.DataFrame,
    team_hhi: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each player-game, attach:
      roll_spread  — team's rolling median signed spread over prior SPREAD_ROLL_WINDOW games
      roll_hhi     — team's rolling mean HHI over prior HHI_ROLL_WINDOW games
    Both use shift(1) — no current-game leakage.
    """
    # Merge game-level team context onto logs (one row per team-game)
    team_game_ctx = (
        logs[["game_id", "team_name", "game_date", "season"]]
        .drop_duplicates()
        .merge(team_spreads, on=["team_name", "game_date", "season"], how="left")
        .merge(team_hhi, on=["game_id", "team_name", "game_date", "season"], how="left")
        .sort_values(["team_name", "season", "game_date"])
    )

    def _roll(s, w):
        return s.shift(1).rolling(w, min_periods=max(3, w // 3)).mean()

    team_game_ctx["roll_spread"] = (
        team_game_ctx.groupby(["team_name", "season"])["spread_signed"]
        .transform(lambda s: _roll(s, SPREAD_ROLL_WINDOW))
    )
    team_game_ctx["roll_hhi"] = (
        team_game_ctx.groupby(["team_name", "season"])["hhi"]
        .transform(lambda s: _roll(s, HHI_ROLL_WINDOW))
    )

    logs = logs.merge(
        team_game_ctx[["game_id", "team_name", "roll_spread", "roll_hhi"]],
        on=["game_id", "team_name"],
        how="left",
    )
    return logs


# =============================================================================
# SECTION 3 — FEATURES + BET CONSTRUCTION (v2 core, reused)
# =============================================================================

def build_features(logs: pd.DataFrame) -> pd.DataFrame:
    df = logs.sort_values(["player_id", "game_date"]).copy()

    team_totals = df.groupby(["game_id", "team_name"])["pts"].sum().reset_index(name="team_pts")
    df = df.merge(team_totals, on=["game_id", "team_name"], how="left")

    def _roll(s, w):
        return s.shift(1).rolling(w, min_periods=MIN_GAMES_FOR_ROLLING).mean()

    df[f"roll{ROLLING_WINDOW}_pts"] = (
        df.groupby("player_id")["pts"].transform(lambda s: _roll(s, ROLLING_WINDOW))
    )
    df["games_played_prior"] = df.groupby("player_id").cumcount()
    df["resid10"] = df["pts"] - df[f"roll{ROLLING_WINDOW}_pts"]
    return df


def identify_stars(df: pd.DataFrame) -> pd.DataFrame:
    gp = (
        df.groupby(["player_id", "team_name", "season"])["game_id"]
        .count().reset_index(name="gp")
    )
    stars = (
        gp.sort_values(["team_name", "season", "gp"], ascending=[True, True, False])
        .groupby(["team_name", "season"]).head(3)
        .assign(is_star=True)
    )
    df = df.merge(
        stars[["player_id", "team_name", "season", "is_star"]],
        on=["player_id", "team_name", "season"], how="left",
    )
    df["is_star"] = df["is_star"].fillna(False)
    return df


def build_bets(df: pd.DataFrame, props: pd.DataFrame) -> pd.DataFrame:
    star_resid_std = df[df["is_star"] & df["resid10"].notna()]["resid10"].std()
    threshold = STAR_THRESHOLD_SIGMA * star_resid_std

    star_nights = df[
        df["is_star"]
        & (df["resid10"] > threshold)
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
    signal = signal[
        (signal["next_game_date"] - signal["game_date"]).dt.days <= MAX_GAME_GAP_DAYS
    ]

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
        on=["player_normalized", "game_date"], how="left",
    ).dropna(subset=["prop_line"])
    bets = bets.copy()
    bets["under_win"] = (bets["pts"] < bets["prop_line"]).astype(int)
    return bets


# =============================================================================
# SECTION 4 — STRATIFIED RESULTS
# =============================================================================

def _summary(sub: pd.DataFrame, label: str) -> dict:
    if len(sub) < 30:
        return {}
    wr = sub["under_win"].mean()
    e = wr - BREAKEVEN_110
    pnl = sub.sort_values("game_date")["under_win"].map(lambda x: 10/11 if x == 1 else -1).cumsum()
    roll_max = pnl.cummax()
    return {
        "label": label,
        "n": len(sub),
        "win_rate": round(wr, 4),
        "edge": round(e, 4),
        "n_units": round(pnl.iloc[-1], 1),
        "max_up": round(pnl.max(), 1),
        "max_dd": round((pnl - roll_max).min(), 1),
    }


def section4_stratified(bets: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("SECTION 4: STRATIFIED RESULTS")
    print("=" * 60)

    # ── 4a: by spread quintile ─────────────────────────────────────
    print("\n── 4a: Team quality (rolling signed spread quintiles) ──")
    print("    Negative spread = elite favorite | Positive = weak underdog")

    bets_s = bets.dropna(subset=["roll_spread"]).copy()
    if bets_s.empty:
        print("  WARNING: no roll_spread values — check lines join")
    else:
        bets_s["spread_q"] = pd.qcut(bets_s["roll_spread"], q=5, labels=["Q1 elite","Q2","Q3","Q4","Q5 weak"])
        rows = []
        for q in ["Q1 elite", "Q2", "Q3", "Q4", "Q5 weak"]:
            sub = bets_s[bets_s["spread_q"] == q]
            r = _summary(sub, q)
            if r:
                spread_range = f"{sub['roll_spread'].min():.1f} to {sub['roll_spread'].max():.1f}"
                r["spread_range"] = spread_range
                rows.append(r)
                print(f"  {q:12} | n={r['n']:,}  wr={r['win_rate']:.3f}  "
                      f"edge={r['edge']:+.3f}  units={r['n_units']:+.1f}u  "
                      f"max_dd={r['max_dd']:.1f}u  spread=[{spread_range}]")

        # Overall for reference
        r_all = _summary(bets_s, "All")
        print(f"  {'All':12} | n={r_all['n']:,}  wr={r_all['win_rate']:.3f}  "
              f"edge={r_all['edge']:+.3f}  units={r_all['n_units']:+.1f}u")

    # ── 4b: by HHI tertile ────────────────────────────────────────
    print("\n── 4b: Team type (rolling HHI tertiles) ──")
    print("    Low HHI = balanced scoring | High HHI = star-concentrated")

    bets_h = bets.dropna(subset=["roll_hhi"]).copy()
    if bets_h.empty:
        print("  WARNING: no roll_hhi values — check HHI join")
    else:
        bets_h["hhi_t"] = pd.qcut(bets_h["roll_hhi"], q=3, labels=["T1 balanced","T2 mid","T3 concentrated"])
        for t in ["T1 balanced", "T2 mid", "T3 concentrated"]:
            sub = bets_h[bets_h["hhi_t"] == t]
            r = _summary(sub, t)
            if r:
                hhi_range = f"{sub['roll_hhi'].min():.3f} to {sub['roll_hhi'].max():.3f}"
                print(f"  {t:18} | n={r['n']:,}  wr={r['win_rate']:.3f}  "
                      f"edge={r['edge']:+.3f}  units={r['n_units']:+.1f}u  "
                      f"max_dd={r['max_dd']:.1f}u  HHI=[{hhi_range}]")

    # ── 4c: cross-tab spread quintile × HHI tertile ──────────────
    print("\n── 4c: Cross-tab — spread quintile × HHI tertile ──")
    bets_c = bets.dropna(subset=["roll_spread", "roll_hhi"]).copy()
    if not bets_c.empty:
        bets_c["spread_q"] = pd.qcut(bets_c["roll_spread"], q=5, labels=["Q1","Q2","Q3","Q4","Q5"])
        bets_c["hhi_t"] = pd.qcut(bets_c["roll_hhi"], q=3, labels=["T1","T2","T3"])
        print(f"  {'':6}  {'T1 (balanced)':>20}  {'T2 (mid)':>20}  {'T3 (concentrated)':>22}")
        for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
            row_parts = [f"  {q}    "]
            for t in ["T1", "T2", "T3"]:
                sub = bets_c[(bets_c["spread_q"] == q) & (bets_c["hhi_t"] == t)]
                if len(sub) < 30:
                    row_parts.append(f"{'n<30':>20}  ")
                else:
                    wr = sub["under_win"].mean()
                    e = wr - BREAKEVEN_110
                    row_parts.append(f"{f'wr={wr:.3f} e={e:+.3f} n={len(sub)}':>20}  ")
            print("".join(row_parts))

    _plot_stratified(bets)


def _plot_stratified(bets: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Spread quintile bar chart
    bets_s = bets.dropna(subset=["roll_spread"]).copy()
    if not bets_s.empty:
        bets_s["spread_q"] = pd.qcut(bets_s["roll_spread"], q=5,
                                      labels=["Q1\nelite", "Q2", "Q3", "Q4", "Q5\nweak"])
        q_stats = bets_s.groupby("spread_q")["under_win"].agg(
            wr="mean", n="count"
        ).reset_index()
        q_stats["edge"] = q_stats["wr"] - BREAKEVEN_110
        colors = ["#55A868" if e > 0 else "#C44E52" for e in q_stats["edge"]]
        axes[0].bar(q_stats["spread_q"].astype(str), q_stats["edge"] * 100, color=colors)
        axes[0].axhline(0, color="red", linestyle="--", linewidth=0.8)
        axes[0].set_xlabel("Team quality (spread quintile)")
        axes[0].set_ylabel("Edge (%)")
        axes[0].set_title("Edge by Team Quality\n(rolling signed spread)")
        for _, row in q_stats.iterrows():
            axes[0].text(str(row["spread_q"]), row["edge"] * 100 + 0.2,
                         f"n={int(row['n'])}", ha="center", fontsize=8)

    # HHI tertile bar chart
    bets_h = bets.dropna(subset=["roll_hhi"]).copy()
    if not bets_h.empty:
        bets_h["hhi_t"] = pd.qcut(bets_h["roll_hhi"], q=3,
                                   labels=["T1\nbalanced", "T2\nmid", "T3\nconcentrated"])
        h_stats = bets_h.groupby("hhi_t")["under_win"].agg(
            wr="mean", n="count"
        ).reset_index()
        h_stats["edge"] = h_stats["wr"] - BREAKEVEN_110
        colors = ["#55A868" if e > 0 else "#C44E52" for e in h_stats["edge"]]
        axes[1].bar(h_stats["hhi_t"].astype(str), h_stats["edge"] * 100, color=colors)
        axes[1].axhline(0, color="red", linestyle="--", linewidth=0.8)
        axes[1].set_xlabel("Team type (HHI tertile)")
        axes[1].set_ylabel("Edge (%)")
        axes[1].set_title("Edge by Team Type\n(rolling HHI)")
        for _, row in h_stats.iterrows():
            axes[1].text(str(row["hhi_t"]), row["edge"] * 100 + 0.2,
                         f"n={int(row['n'])}", ha="center", fontsize=8)

    fig.suptitle("v3: Signal Conditioning — Team Quality & Team Type", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "v3_stratified.png", dpi=120)
    plt.close(fig)
    print(f"\n  → v3_stratified.png")


# =============================================================================
# SECTION 5 — SEASON TREND BY SEGMENT
# =============================================================================

def section5_season_trend(bets: pd.DataFrame) -> None:
    """Does the edge decay uniformly, or is a specific segment driving the decline?"""
    print("\n" + "=" * 60)
    print("SECTION 5: SEASON TREND BY SEGMENT")
    print("=" * 60)

    bets_s = bets.dropna(subset=["roll_spread"]).copy()
    if bets_s.empty:
        return

    bets_s["spread_q"] = pd.qcut(bets_s["roll_spread"], q=5,
                                  labels=["Q1 elite", "Q2", "Q3", "Q4", "Q5 weak"])

    print("\n  Win rate by season × spread quintile:")
    pivot = (
        bets_s.groupby(["season", "spread_q"])["under_win"]
        .mean()
        .unstack("spread_q")
        .round(3)
    )
    print(pivot.to_string())

    bets_h = bets.dropna(subset=["roll_hhi"]).copy()
    bets_h["hhi_t"] = pd.qcut(bets_h["roll_hhi"], q=3,
                               labels=["T1 balanced", "T2 mid", "T3 concentrated"])
    print("\n  Win rate by season × HHI tertile:")
    pivot_h = (
        bets_h.groupby(["season", "hhi_t"])["under_win"]
        .mean()
        .unstack("hhi_t")
        .round(3)
    )
    print(pivot_h.to_string())


# =============================================================================
# SECTION 6 — YEAR × QUINTILE × SIGMA GRID
# =============================================================================

SIGMAS = [1.0, 1.25, 1.5, 1.75, 2.0]
Q_LABELS = ["Q1", "Q2", "Q3", "Q4", "Q5"]


def _build_bets_at_sigma(df: pd.DataFrame, props: pd.DataFrame, sigma: float) -> pd.DataFrame:
    star_resid_std = df[df["is_star"] & df["resid10"].notna()]["resid10"].std()
    threshold = sigma * star_resid_std

    star_nights = df[
        df["is_star"]
        & (df["resid10"] > threshold)
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
    signal = signal[(signal["next_game_date"] - signal["game_date"]).dt.days <= MAX_GAME_GAP_DAYS]

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
        on=["player_normalized", "game_date"], how="left",
    ).dropna(subset=["prop_line"])
    bets = bets.copy()
    bets["under_win"] = (bets["pts"] < bets["prop_line"]).astype(int)
    return bets


def section6_sigma_grid(df: pd.DataFrame, props: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("SECTION 6: YEAR × QUINTILE × SIGMA GRID")
    print("=" * 60)
    print("  n_units at -110. Negative cells in [ ]. Min 20 bets to show value.")

    # Compute quintile breakpoints from the base sigma (1.5) bets so labels are consistent
    base_bets = _build_bets_at_sigma(df, props, 1.5).dropna(subset=["roll_spread"])
    _, q_bins = pd.qcut(base_bets["roll_spread"], q=5, retbins=True, labels=False)

    rows = []
    for sigma in SIGMAS:
        bets = _build_bets_at_sigma(df, props, sigma).dropna(subset=["roll_spread"]).copy()
        bets["q"] = pd.cut(bets["roll_spread"], bins=q_bins, labels=Q_LABELS, include_lowest=True)
        for season in SEASONS:
            for q in Q_LABELS:
                sub = bets[(bets["season"] == season) & (bets["q"] == q)]
                n = len(sub)
                if n < 20:
                    units = None
                    wr = None
                else:
                    wr = sub["under_win"].mean()
                    units = round(n * (wr * 21 / 11 - 1), 1)
                rows.append({"sigma": sigma, "season": season, "q": q, "n": n, "wr": wr, "units": units})

    grid = pd.DataFrame(rows)

    # Print as a clean table: rows = season × Q, columns = sigma
    print()
    sigma_cols = [f"σ={s}" for s in SIGMAS]
    header = f"  {'Season':<10} {'Q':<4}" + "".join(f"  {c:>14}" for c in sigma_cols)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for season in SEASONS:
        for q in Q_LABELS:
            row_str = f"  {season:<10} {q:<4}"
            for sigma in SIGMAS:
                cell = grid[(grid["sigma"] == sigma) & (grid["season"] == season) & (grid["q"] == q)]
                if cell.empty or cell["units"].iloc[0] is None:
                    row_str += f"  {'n<20':>14}"
                else:
                    u = cell["units"].iloc[0]
                    n = int(cell["n"].iloc[0])
                    val = f"{u:+.0f}u (n={n})"
                    # Flag negative
                    if u < 0:
                        val = f"[{val}]"
                    row_str += f"  {val:>14}"
            print(row_str)
        print()

    # Also save as CSV for easy inspection
    grid.to_csv(OUTPUT_DIR / "v3_sigma_grid.csv", index=False)
    print(f"  → v3_sigma_grid.csv")

    # Heatmap of units: rows = season × Q, columns = sigma (base σ=1.5 only, for readability)
    _plot_units_heatmap(grid)


def _plot_units_heatmap(grid: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(SIGMAS), figsize=(4 * len(SIGMAS), 5), sharey=True)
    fig.suptitle("n_units by Season × Spread Quintile × Sigma", fontsize=11)

    all_units = grid["units"].dropna()
    vmin, vmax = all_units.min(), all_units.max()
    # Centre colormap at 0
    abs_max = max(abs(vmin), abs(vmax))

    for ax, sigma in zip(axes, SIGMAS):
        sub = grid[grid["sigma"] == sigma].copy()
        pivot = sub.pivot(index="q", columns="season", values="units")
        pivot = pivot.reindex(index=Q_LABELS, columns=SEASONS)

        im = ax.imshow(
            pivot.values.astype(float),
            aspect="auto",
            cmap="RdYlGn",
            vmin=-abs_max,
            vmax=abs_max,
        )
        ax.set_xticks(range(len(SEASONS)))
        ax.set_xticklabels([s[-5:] for s in SEASONS], fontsize=8)
        ax.set_yticks(range(len(Q_LABELS)))
        if sigma == SIGMAS[0]:
            ax.set_yticklabels(Q_LABELS, fontsize=8)
        ax.set_title(f"σ={sigma}", fontsize=9)

        for i, q in enumerate(Q_LABELS):
            for j, season in enumerate(SEASONS):
                val = pivot.loc[q, season] if q in pivot.index and season in pivot.columns else None
                if val is not None and not np.isnan(float(val)):
                    ax.text(j, i, f"{val:+.0f}", ha="center", va="center", fontsize=7,
                            color="black" if abs(float(val)) < abs_max * 0.6 else "white")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "v3_sigma_heatmap.png", dpi=120)
    plt.close(fig)
    print(f"  → v3_sigma_heatmap.png")


# =============================================================================
# SECTION 7 — ROLLING SPREAD: CONTINUOUS CORRELATION CHECK
# =============================================================================

def section7_spread_correlation(bets: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("SECTION 7: ROLLING SPREAD — CONTINUOUS CORRELATION CHECK")
    print("=" * 60)
    print("  H0: roll_spread has no linear relationship with under_win")

    clean = bets.dropna(subset=["roll_spread", "under_win"]).copy()
    print(f"\n  n = {len(clean):,} bets with roll_spread")

    # Point-biserial correlation (equivalent to Pearson for binary outcome)
    r, p = stats.pointbiserialr(clean["under_win"], clean["roll_spread"])
    print(f"\n  Pooled:  r={r:+.4f}  p={p:.4f}  {'significant' if p < 0.05 else 'NOT significant'} at p<0.05")

    # Per season — does it strengthen or weaken over time?
    print("\n  By season:")
    for season in SEASONS:
        sub = clean[clean["season"] == season]
        if len(sub) < 50:
            continue
        r_s, p_s = stats.pointbiserialr(sub["under_win"], sub["roll_spread"])
        sig = "significant" if p_s < 0.05 else "NOT significant"
        print(f"    {season}: r={r_s:+.4f}  p={p_s:.4f}  {sig}  (n={len(sub):,})")

    # Also check roll_hhi for comparison
    clean_h = bets.dropna(subset=["roll_hhi", "under_win"])
    r_h, p_h = stats.pointbiserialr(clean_h["under_win"], clean_h["roll_hhi"])
    print(f"\n  roll_hhi (for comparison): r={r_h:+.4f}  p={p_h:.4f}  "
          f"{'significant' if p_h < 0.05 else 'NOT significant'} at p<0.05")

    r_sq = r ** 2
    print(f"\n  r² = {r_sq:.4f} — spread explains {r_sq*100:.2f}% of variance in under_win")
    print("\n  Verdict:")
    # Pooled p<0.05 is purely a sample-size artefact at n=3,830 with r~0.03.
    # Check whether it holds in any individual season.
    season_sigs = []
    for season in SEASONS:
        sub = clean[clean["season"] == season]
        if len(sub) < 50:
            continue
        _, p_s = stats.pointbiserialr(sub["under_win"], sub["roll_spread"])
        if p_s < 0.05:
            season_sigs.append(season)

    if not season_sigs and r_sq < 0.005:
        print(f"  → roll_spread is NOT a meaningful predictor (r={r:+.4f}, r²<0.1%).")
        print("    Pooled p<0.05 is a sample-size artefact — not significant in any single season.")
        print("    Use only as a kill switch (drop Q4/Q5 based on season-trend decay), not as a signal.")
    else:
        print(f"  → roll_spread shows a relationship worth investigating (r={r:+.4f}, sig in {season_sigs}).")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("=" * 60)
    print("NBA POINTS DISPERSION v3 — SIGNAL CONDITIONING")
    print("=" * 60)

    logs, props, lines = load_data()

    print("\n[Building team context features ...]")
    team_spreads = build_team_spread(lines)
    team_hhi = build_team_hhi(logs)
    logs = attach_rolling_team_context(logs, team_spreads, team_hhi)

    spread_coverage = logs["roll_spread"].notna().mean()
    hhi_coverage = logs["roll_hhi"].notna().mean()
    print(f"  roll_spread coverage: {spread_coverage:.1%}")
    print(f"  roll_hhi coverage:    {hhi_coverage:.1%}")

    print("\n[Building player features + stars ...]")
    df = build_features(logs)
    df = identify_stars(df)

    print("\n[Constructing bets ...]")
    bets = build_bets(df, props)
    print(f"  Total bets with prop line: {len(bets):,}")

    spread_attached = bets["roll_spread"].notna().mean()
    hhi_attached = bets["roll_hhi"].notna().mean()
    print(f"  Bets with roll_spread: {spread_attached:.1%}")
    print(f"  Bets with roll_hhi:    {hhi_attached:.1%}")

    section4_stratified(bets)
    section5_season_trend(bets)
    section6_sigma_grid(df, props)
    section7_spread_correlation(bets)

    print("\n" + "=" * 60)
    print("DONE — outputs written to:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
