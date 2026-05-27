"""
NBA Points Dispersion — High-Level Analysis
============================================
Date: 2026-05-26
Plan: src/nba_points_modeling/research/plans/20260526_dispersion.py

Core idea: borrow dispersion-trading logic from finance.
  - "Index"        = team total
  - "Constituents" = individual player props
  - Hard constraint: Σ player_pts = team_pts  (zero-sum budget)

When a star over-performs their line, teammates collectively under-perform.
If the book prices player props semi-independently, the zero-sum relationship
is systematically mispriced in the tails.

Sections:
  1  Data loading & overview
  2  Team-game dispersion metrics (CV, HHI, Gini, top-N share)
  3  Concentration stability — team profiles across seasons
  4  Intra-team player correlation
  5  Star over-performance → teammate response (core dispersion signal)
  6  Dispersion context features & correlation with over/under outcome
  7  Preliminary ROI sketch of the raw signal

Usage:
    cd /path/to/repo
    python src/nba_points_modeling/research/scripts/20260526_dispersion.py
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
# REPO ROOT
# =============================================================================

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
# SECTION 1 — DATA LOADING & OVERVIEW
# =============================================================================

SEASONS = ["2023-24", "2024-25", "2025-26"]
MIN_MINUTES_CUTOFF = 5
POINTS_MARKET_SUBSTR = "point"  # matches 'player_points', 'points', etc.


def load_player_game_logs() -> pd.DataFrame:
    """Load logs via src/nba_data cache (S3 parquet)."""
    from src.nba_data.get_data import get_data

    print("Loading NBA data from cache (src/nba_data)...")
    data = get_data(min_minutes=MIN_MINUTES_CUTOFF)

    df = data.logs.copy()
    df.columns = df.columns.str.lower()
    df = df.rename(columns={"min": "minutes"})
    # team_normalized is the consistently-normalised full team name
    df["team_name"] = df["team_normalized"]
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["player_id", "game_date"]).reset_index(drop=True)

    for season in SEASONS:
        n = (df["season"] == season).sum()
        print(f"  {season}: {n:,} player-games")
    print(f"  Total: {len(df):,} player-games | {df['player_id'].nunique()} unique players")
    return df


def load_points_props() -> pd.DataFrame:
    """
    Load player points props from the nba_data cache.
    Returns one row per player × game × bookmaker with columns:
      player_normalized, game_date, prop_line, bookmaker, over_odds, under_odds, season
    Returns empty DataFrame if no props data found.
    """
    from src.nba_data.get_data import get_data

    data = get_data()
    props = data.props.copy()
    if props.empty:
        return props

    pts = props[props["market"].str.lower().str.contains(POINTS_MARKET_SUBSTR, na=False)].copy()
    if pts.empty:
        print("  WARNING: no points market found in props — check market names")
        print("  Available markets:", props["market"].unique()[:10])
        return pts

    pts["game_date"] = pd.to_datetime(pts["game_date"])
    # Best line per player × game: median across bookmakers
    best = (
        pts.groupby(["player_normalized", "game_date", "season"])
        .agg(
            prop_line=("prop_line", "median"),
            n_books=("bookmaker", "nunique"),
        )
        .reset_index()
    )
    print(f"  Props loaded: {len(best):,} player×game lines ({best['n_books'].mean():.1f} books avg)")
    return best


def section1_overview(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("SECTION 1: DATA OVERVIEW")
    print("=" * 60)

    summary = df.groupby("season").agg(
        player_games=("pts", "count"),
        unique_players=("player_id", "nunique"),
        unique_games=("game_id", "nunique"),
        mean_pts=("pts", "mean"),
        median_pts=("pts", "median"),
        std_pts=("pts", "std"),
        mean_min=("minutes", "mean"),
    ).round(2)
    print("\nSeason summary:")
    print(summary.to_string())

    # Distribution of points per player-game
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, season in enumerate(SEASONS):
        subset = df[df["season"] == season]["pts"]
        axes[i].hist(subset, bins=40, edgecolor="white", linewidth=0.4)
        axes[i].axvline(subset.mean(), color="red", linestyle="--", label=f"mean={subset.mean():.1f}")
        axes[i].axvline(subset.median(), color="orange", linestyle="--", label=f"median={subset.median():.1f}")
        axes[i].set_title(f"{season}")
        axes[i].set_xlabel("Points")
        axes[i].set_ylabel("Count")
        axes[i].legend(fontsize=8)
    fig.suptitle("Distribution of Player Points per Game (MIN ≥ 5)", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "s1_points_distribution.png", dpi=120)
    plt.close()
    print("  → s1_points_distribution.png")


# =============================================================================
# SECTION 2 — TEAM-GAME DISPERSION METRICS
# =============================================================================

def gini(arr: np.ndarray) -> float:
    arr = np.sort(np.abs(arr))
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * arr).sum() / (n * arr.sum())) - (n + 1) / n)


def compute_team_game_dispersion(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (game_id, team_name), compute dispersion metrics over the roster.
    Returns one row per team-game.
    """
    def _metrics(grp: pd.DataFrame) -> pd.Series:
        pts = grp["pts"].values.astype(float)
        total = pts.sum()
        n = len(pts)
        if total == 0 or n < 2:
            return pd.Series(dtype=float)
        shares = pts / total
        sorted_desc = np.sort(pts)[::-1]
        return pd.Series({
            "n_players":   n,
            "team_pts":    total,
            "mean_pts":    pts.mean(),
            "std_pts":     pts.std(ddof=1),
            "cv":          pts.std(ddof=1) / pts.mean() if pts.mean() > 0 else np.nan,
            "hhi":         float((shares ** 2).sum()),
            "top1_share":  float(shares.max()),
            "top3_share":  float(sorted_desc[:3].sum() / total),
            "gini":        gini(pts),
            "max_pts":     float(pts.max()),
        })

    tg = (
        df.groupby(["game_id", "team_name", "game_date", "season"])
        .apply(_metrics)
        .reset_index()
        .dropna(subset=["cv"])
    )
    return tg


def section2_dispersion_metrics(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("SECTION 2: TEAM-GAME DISPERSION METRICS")
    print("=" * 60)

    tg = compute_team_game_dispersion(df)

    metric_cols = ["cv", "hhi", "top1_share", "top3_share", "gini"]
    print("\nDispersion metric distributions (all seasons):")
    print(tg[metric_cols].describe().round(3).to_string())

    # Season-level comparison
    print("\nBy season:")
    print(tg.groupby("season")[metric_cols].mean().round(3).to_string())

    # Plot metric distributions
    fig, axes = plt.subplots(1, len(metric_cols), figsize=(18, 4))
    labels = {"cv": "CV (std/mean)", "hhi": "HHI", "top1_share": "Top-1 Share",
              "top3_share": "Top-3 Share", "gini": "Gini"}
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    for i, col in enumerate(metric_cols):
        axes[i].hist(tg[col], bins=40, color=colors[i], edgecolor="white", linewidth=0.3)
        mu = tg[col].mean()
        axes[i].axvline(mu, color="black", linestyle="--", linewidth=1.2, label=f"μ={mu:.3f}")
        axes[i].set_title(labels[col])
        axes[i].legend(fontsize=8)
    fig.suptitle("Team-Game Scoring Dispersion Metrics (all 3 seasons)", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "s2_dispersion_distributions.png", dpi=120)
    plt.close()
    print("  → s2_dispersion_distributions.png")

    # Plot HHI vs team_pts scatter
    fig, ax = plt.subplots(figsize=(7, 5))
    for season, color in zip(SEASONS, ["#4C72B0", "#55A868", "#C44E52"]):
        sub = tg[tg["season"] == season]
        ax.scatter(sub["team_pts"], sub["hhi"], alpha=0.15, s=8, label=season, color=color)
    ax.set_xlabel("Team Points (actual)")
    ax.set_ylabel("HHI (scoring concentration)")
    ax.set_title("Scoring Concentration vs Team Points")
    ax.legend()
    corr = tg[["team_pts", "hhi"]].corr().iloc[0, 1]
    ax.annotate(f"r = {corr:.3f}", xy=(0.05, 0.92), xycoords="axes fraction", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "s2_hhi_vs_team_pts.png", dpi=120)
    plt.close()
    print("  → s2_hhi_vs_team_pts.png")

    return tg


# =============================================================================
# SECTION 3 — CONCENTRATION STABILITY (TEAM PROFILES)
# =============================================================================

def section3_concentration_stability(tg: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("SECTION 3: CONCENTRATION STABILITY — TEAM PROFILES")
    print("=" * 60)

    team_season = (
        tg.groupby(["team_name", "season"])
        .agg(
            mean_hhi=("hhi", "mean"),
            mean_top1=("top1_share", "mean"),
            mean_cv=("cv", "mean"),
            n_games=("hhi", "count"),
        )
        .reset_index()
        .round(4)
    )

    # Most and least concentrated teams
    for season in SEASONS:
        sub = team_season[team_season["season"] == season].sort_values("mean_hhi", ascending=False)
        print(f"\n{season} — Top-5 concentrated (high HHI):")
        print(sub[["team_name", "mean_hhi", "mean_top1", "mean_cv"]].head(5).to_string(index=False))
        print(f"\n{season} — Top-5 balanced (low HHI):")
        print(sub[["team_name", "mean_hhi", "mean_top1", "mean_cv"]].tail(5).to_string(index=False))

    # HHI lag-1 autocorrelation per team — is concentration persistent game-to-game?
    tg_sorted = tg.sort_values(["team_name", "season", "game_date"])
    tg_sorted["hhi_lag1"] = tg_sorted.groupby(["team_name", "season"])["hhi"].shift(1)
    lag_corr = (
        tg_sorted.dropna(subset=["hhi_lag1"])
        .groupby(["team_name", "season"])
        .apply(lambda g: g["hhi"].corr(g["hhi_lag1"]))
        .reset_index(name="hhi_autocorr")
    )
    print(f"\nHHI lag-1 autocorrelation — distribution across team×seasons:")
    print(lag_corr["hhi_autocorr"].describe().round(3).to_string())
    median_ac = lag_corr["hhi_autocorr"].median()
    print(f"  Median autocorr: {median_ac:.3f}  (> 0 = persistent, ≈ 0 = no memory)")

    # Pivot teams to see their HHI across seasons
    pivot = team_season.pivot(index="team_name", columns="season", values="mean_hhi").round(4)
    print("\nTeam HHI across seasons (top/bottom 10 by 2024-25):")
    col = "2024-25" if "2024-25" in pivot.columns else pivot.columns[-1]
    print(pivot.sort_values(col).to_string())

    # Plot top-10 concentrated teams across seasons
    if "2024-25" in pivot.columns:
        top10 = pivot[col].nlargest(10).index
        fig, ax = plt.subplots(figsize=(10, 5))
        for team in top10:
            vals = pivot.loc[team].dropna()
            ax.plot(vals.index, vals.values, marker="o", label=team)
        ax.set_title("HHI Across Seasons — Top-10 Concentrated Teams (2024-25)")
        ax.set_ylabel("Mean HHI")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "s3_hhi_by_team_season.png", dpi=120)
        plt.close(fig)
        print("  → s3_hhi_by_team_season.png")


# =============================================================================
# SECTION 4 — INTRA-TEAM PLAYER CORRELATION
# =============================================================================

def section4_intra_team_correlation(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("SECTION 4: INTRA-TEAM PLAYER CORRELATION")
    print("=" * 60)

    all_pairwise = []

    for (team, season), grp in df.groupby(["team_name", "season"]):
        # Take players with at least 20 games
        player_games = grp.groupby("player_id")["pts"].count()
        active = player_games[player_games >= 20].index
        if len(active) < 2:
            continue
        pivot = (
            grp[grp["player_id"].isin(active)]
            .pivot_table(index="game_id", columns="player_id", values="pts")
            .dropna(thresh=2)
        )
        if pivot.shape[0] < 20:
            continue
        corr_matrix = pivot.corr()
        # Extract upper triangle (pairwise correlations)
        n = corr_matrix.shape[0]
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                r = corr_matrix.iloc[i, j]
                if not np.isnan(r):
                    pairs.append(r)
        if pairs:
            all_pairwise.append({
                "team_name": team,
                "season": season,
                "n_players": len(active),
                "n_pairs": len(pairs),
                "mean_corr": np.mean(pairs),
                "median_corr": np.median(pairs),
                "pct_negative": np.mean(np.array(pairs) < 0),
            })

    corr_df = pd.DataFrame(all_pairwise)
    print("\nIntra-team pairwise correlation summary (across team×seasons):")
    print(corr_df[["mean_corr", "median_corr", "pct_negative"]].describe().round(3).to_string())

    overall_median = corr_df["median_corr"].median()
    print(f"\n  Overall median intra-team pairwise correlation: {overall_median:.4f}")

    # T-test: is the population median significantly different from zero?
    t_stat, p_val = stats.ttest_1samp(corr_df["median_corr"].dropna(), 0)
    print(f"  1-sample t-test (H0: median corr = 0): t={t_stat:.3f}, p={p_val:.4f}")
    if p_val < 0.05:
        direction = "NEGATIVE" if overall_median < 0 else "POSITIVE"
        print(f"  → Significant {direction} intra-team correlation confirmed")
    else:
        print("  → Cannot reject H0 (correlation not distinguishable from zero)")

    # Distribution plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(corr_df["median_corr"], bins=30, edgecolor="white", linewidth=0.4)
    axes[0].axvline(0, color="red", linestyle="--", label="zero")
    axes[0].axvline(overall_median, color="blue", linestyle="--",
                    label=f"median={overall_median:.3f}")
    axes[0].set_title("Median Intra-Team Pairwise Correlation\n(each dot = one team×season)")
    axes[0].set_xlabel("Pearson r")
    axes[0].legend()

    axes[1].hist(corr_df["pct_negative"], bins=20, edgecolor="white", linewidth=0.4)
    axes[1].axvline(0.5, color="red", linestyle="--", label="50%")
    axes[1].set_title("% of Pairs with Negative Correlation\n(per team×season)")
    axes[1].set_xlabel("Fraction")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "s4_intra_team_correlation.png", dpi=120)
    plt.close(fig)
    print("  → s4_intra_team_correlation.png")

    # Top/bottom teams by mean pairwise correlation
    print("\nMost negatively correlated teams (2024-25):")
    sub = corr_df[corr_df["season"] == "2024-25"].sort_values("mean_corr")
    print(sub[["team_name", "mean_corr", "median_corr", "pct_negative"]].head(8).to_string(index=False))
    print("\nMost positively correlated teams (2024-25):")
    print(sub[["team_name", "mean_corr", "median_corr", "pct_negative"]].tail(8).to_string(index=False))


# =============================================================================
# SECTION 5 — STAR OVER-PERFORMANCE → TEAMMATE RESPONSE
# =============================================================================

def compute_rolling_mean(df: pd.DataFrame, window: int = 10, col: str = "pts") -> pd.DataFrame:
    """Add rolling mean (excluding current game) per player."""
    df = df.sort_values(["player_id", "game_date"]).copy()
    df[f"roll{window}_{col}"] = (
        df.groupby("player_id")[col]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=3).mean())
    )
    return df


def section5_star_overperformance(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("SECTION 5: STAR OVER-PERFORMANCE → TEAMMATE RESPONSE")
    print("=" * 60)

    df = compute_rolling_mean(df, window=10)
    df = compute_rolling_mean(df, window=20)

    df["resid10"] = df["pts"] - df["roll10_pts"]

    # Identify "star" players: top-3 by games played per team×season
    games_played = (
        df.groupby(["player_id", "team_name", "season"])["game_id"]
        .count()
        .reset_index(name="gp")
    )
    stars = (
        games_played.sort_values(["team_name", "season", "gp"], ascending=[True, True, False])
        .groupby(["team_name", "season"])
        .head(3)
        .assign(is_star=True)
    )
    df = df.merge(
        stars[["player_id", "team_name", "season", "is_star"]],
        on=["player_id", "team_name", "season"],
        how="left"
    )
    df["is_star"] = df["is_star"].fillna(False)

    # Compute star residual σ for threshold
    star_df = df[df["is_star"] & df["resid10"].notna()]
    resid_std = star_df["resid10"].std()
    THRESHOLD = 1.5 * resid_std
    print(f"\nStar residual σ = {resid_std:.2f} pts  |  Threshold (1.5σ) = {THRESHOLD:.2f} pts")

    # Find "star night" = star posts resid10 > threshold
    star_nights = star_df[star_df["resid10"] > THRESHOLD][
        ["game_id", "team_name", "player_id", "season", "resid10"]
    ].rename(columns={"player_id": "star_player_id", "resid10": "star_resid"})

    print(f"  Star nights (resid > +1.5σ): {len(star_nights):,} events")

    # For each star night, get same-team non-star teammates' residuals
    teammate_residuals = df.merge(
        star_nights[["game_id", "team_name", "star_player_id"]],
        on=["game_id", "team_name"],
    )
    # Exclude the star themselves
    teammate_residuals = teammate_residuals[
        teammate_residuals["player_id"] != teammate_residuals["star_player_id"]
    ]
    teammate_residuals = teammate_residuals.dropna(subset=["resid10"])

    # Teammate residuals on star nights
    tm_resid_on_star_night = teammate_residuals["resid10"].values
    print(f"  Teammate observations on star nights: {len(tm_resid_on_star_night):,}")
    print(f"  Teammate mean resid on star nights: {tm_resid_on_star_night.mean():.3f} pts")
    print(f"  Teammate median resid on star nights: {np.median(tm_resid_on_star_night):.3f} pts")

    # Baseline: all non-star teammate residuals on non-star nights
    non_star_nights = df[~df["game_id"].isin(star_nights["game_id"])]
    non_star_tm = non_star_nights[~non_star_nights["is_star"] & non_star_nights["resid10"].notna()]
    baseline_resid = non_star_tm["resid10"].values
    print(f"\n  Baseline (non-star nights) teammate mean resid: {baseline_resid.mean():.3f} pts")

    # T-test: is star-night teammate residual significantly lower than baseline?
    t_stat, p_val = stats.ttest_ind(tm_resid_on_star_night, baseline_resid, equal_var=False)
    effect = tm_resid_on_star_night.mean() - baseline_resid.mean()
    print(f"\n  Effect (star night - baseline): {effect:.3f} pts")
    print(f"  Welch t-test: t={t_stat:.3f}, p={p_val:.4f}")
    if p_val < 0.05:
        print("  → Statistically significant: teammates score LESS on star nights")
    else:
        print("  → Not statistically significant at p<0.05")

    # Also check: top decile star night (most extreme over-performances)
    top_decile_threshold = np.percentile(star_df["resid10"].dropna(), 90)
    top_star_nights = star_df[star_df["resid10"] > top_decile_threshold][
        ["game_id", "team_name", "player_id"]
    ].rename(columns={"player_id": "star_player_id"})
    top_tm = df.merge(top_star_nights[["game_id", "team_name", "star_player_id"]], on=["game_id", "team_name"])
    top_tm = top_tm[top_tm["player_id"] != top_tm["star_player_id"]].dropna(subset=["resid10"])
    print(f"\n  Top-decile star nights (resid > {top_decile_threshold:.1f} pts): {len(top_star_nights):,} events")
    print(f"  Teammate mean resid on top-decile star nights: {top_tm['resid10'].mean():.3f} pts")
    t2, p2 = stats.ttest_ind(top_tm["resid10"].values, baseline_resid, equal_var=False)
    print(f"  Welch t-test vs baseline: t={t2:.3f}, p={p2:.4f}")

    # Distribution plot
    fig5, ax5 = plt.subplots(figsize=(9, 5))
    bins = np.linspace(-20, 20, 50)
    ax5.hist(baseline_resid, bins=bins, alpha=0.5, label=f"Baseline (μ={baseline_resid.mean():.2f})", density=True)
    ax5.hist(tm_resid_on_star_night, bins=bins, alpha=0.5,
             label=f"Star night teammates (μ={tm_resid_on_star_night.mean():.2f})", density=True)
    ax5.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax5.set_xlabel("Points Residual (actual − rolling-10 mean)")
    ax5.set_ylabel("Density")
    ax5.set_title("Teammate Residuals: Star Night vs Baseline\n(zero-sum dispersion signal)")
    ax5.legend()
    ax5.annotate(f"Effect: {effect:.2f} pts  p={p_val:.4f}", xy=(0.55, 0.88),
                 xycoords="axes fraction", fontsize=10)
    fig5.tight_layout()
    fig5.savefig(OUTPUT_DIR / "s5_star_night_teammate_residuals.png", dpi=120)
    plt.close(fig5)
    print("  → s5_star_night_teammate_residuals.png")


# =============================================================================
# SECTION 6 — DISPERSION CONTEXT FEATURES
# =============================================================================

def section6_dispersion_features(df: pd.DataFrame, tg: pd.DataFrame, props: pd.DataFrame | None = None) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("SECTION 6: DISPERSION CONTEXT FEATURES")
    print("=" * 60)

    # Add team_pts to player-game rows via merge
    df = df.merge(
        tg[["game_id", "team_name", "team_pts", "hhi", "cv", "top1_share"]],
        on=["game_id", "team_name"],
        how="left",
    )
    df["pts_share"] = df["pts"] / df["team_pts"].replace(0, np.nan)

    # Rolling player share (how much of team's scoring this player typically claims)
    df = df.sort_values(["player_id", "game_date"])
    for window in [10, 20]:
        df[f"roll{window}_share"] = (
            df.groupby("player_id")["pts_share"]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=3).mean())
        )
        df[f"roll{window}_hhi"] = (
            df.groupby(["team_name", "season"])["hhi"]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=3).mean())
        )

    # Implied share: how does player's rolling share × team total compare to pts?
    df["implied_pts_via_share"] = df["roll10_share"] * df["team_pts"]
    df["share_residual"] = df["pts"] - df["implied_pts_via_share"]

    df = compute_rolling_mean(df, window=10)

    # is_under: use actual prop line if available, else rolling mean as proxy
    if props is not None and not props.empty and "player_normalized" in df.columns:
        df = df.merge(
            props[["player_normalized", "game_date", "prop_line"]],
            on=["player_normalized", "game_date"],
            how="left",
        )
        covered = df["prop_line"].notna().sum()
        print(f"\n  Using ACTUAL prop lines for is_under ({covered:,} / {len(df):,} rows matched)")
        df["is_under"] = (df["pts"] < df["prop_line"]).astype(float)
        is_under_label = "pts < actual prop line"
    else:
        df["is_under"] = (df["pts"] < df["roll10_pts"]).astype(float)
        is_under_label = "pts < rolling-10 mean (proxy)"
        print("\n  NOTE: no prop lines available — using rolling mean as is_under proxy.")

    feature_cols = ["roll10_share", "roll20_share", "roll10_hhi", "roll20_hhi", "share_residual"]
    avail = [c for c in feature_cols if c in df.columns]

    print(f"\nFeature → is_under correlations ({is_under_label}):")
    clean = df.dropna(subset=avail + ["is_under"])
    for col in avail:
        r, p = stats.pointbiserialr(clean["is_under"], clean[col])
        print(f"  {col:<25} r={r:+.4f}   p={p:.4f}")

    print(f"\n  n = {len(clean):,} player-games with complete feature set")

    return df


# =============================================================================
# SECTION 7 — PRELIMINARY ROI SKETCH
# =============================================================================

def section7_roi_sketch(df: pd.DataFrame, props: pd.DataFrame | None = None) -> None:
    """
    Prototype rule: bet UNDER on non-star co-starters after a star's dominant
    performance (resid10 > 1.5σ in the PRIOR game).

    Uses actual prop lines when available; falls back to rolling-10 mean.
    """
    use_real_lines = props is not None and not props.empty and "player_normalized" in df.columns

    print("\n" + "=" * 60)
    if use_real_lines:
        print("SECTION 7: PRELIMINARY ROI SKETCH (USING ACTUAL PROP LINES)")
    else:
        print("SECTION 7: PRELIMINARY ROI SKETCH (USING ROLLING MEAN AS LINE PROXY)")
    print("=" * 60)
    if not use_real_lines:
        print("  WARNING: Rolling mean ≠ actual book line. This is directional only.")

    # Re-derive star nights with resid10 relative to σ
    df = compute_rolling_mean(df, window=10)
    df["resid10"] = df["pts"] - df["roll10_pts"]
    resid_std = df[df.get("is_star", df["pts"] > 0)]["resid10"].dropna().std()
    THRESHOLD = 1.5 * resid_std

    # Identify star players (top 3 by games per team×season)
    gp = (
        df.groupby(["player_id", "team_name", "season"])["game_id"]
        .count()
        .reset_index(name="gp")
    )
    stars = (
        gp.sort_values(["team_name", "season", "gp"], ascending=[True, True, False])
        .groupby(["team_name", "season"])
        .head(3)
    )
    star_set = set(zip(stars["player_id"], stars["team_name"], stars["season"]))
    df["is_star"] = df.apply(
        lambda r: (r["player_id"], r["team_name"], r["season"]) in star_set, axis=1
    )

    # Find prior-game star nights per team
    star_df = df[df["is_star"] & df["resid10"].notna()].copy()
    star_df["star_date"] = star_df["game_date"]
    dominant = star_df[star_df["resid10"] > THRESHOLD][
        ["team_name", "season", "game_date", "player_id"]
    ].rename(columns={"player_id": "star_id"})

    # Next game for that team after a dominant star night
    team_games = (
        df[["team_name", "season", "game_date", "game_id"]]
        .drop_duplicates()
        .sort_values(["team_name", "season", "game_date"])
    )
    team_games["next_game_date"] = (
        team_games.groupby(["team_name", "season"])["game_date"].shift(-1)
    )
    team_games["next_game_id"] = (
        team_games.groupby(["team_name", "season"])["game_id"].shift(-1)
    )

    # Merge: after a dominant star game, find the next team game
    signal = dominant.merge(
        team_games[["team_name", "season", "game_date", "next_game_date", "next_game_id"]],
        on=["team_name", "season", "game_date"],
    ).dropna(subset=["next_game_id"])

    # Bet: UNDER on all non-star teammates in the next game
    bets = df.merge(
        signal[["team_name", "season", "next_game_id", "star_id"]],
        left_on=["team_name", "season", "game_id"],
        right_on=["team_name", "season", "next_game_id"],
    )
    bets = bets[bets["player_id"] != bets["star_id"]]
    bets = bets[~bets["is_star"]]  # only role players (exclude other stars)

    if use_real_lines:
        # prop_line may already be present if df came from section6; merge only if missing
        if "prop_line" not in bets.columns:
            bets = bets.merge(
                props[["player_normalized", "game_date", "prop_line"]],
                on=["player_normalized", "game_date"],
                how="left",
            )
        bets = bets.dropna(subset=["pts", "prop_line"])
        bets["under_win"] = (bets["pts"] < bets["prop_line"]).astype(int)
        line_col = "prop_line"
    else:
        bets = bets.dropna(subset=["pts", "roll10_pts"])
        bets["under_win"] = (bets["pts"] < bets["roll10_pts"]).astype(int)
        line_col = "roll10_pts"  # noqa: F841

    print(f"\n  Strategy: UNDER on non-star teammates after a star's dominant game")
    print(f"  Total bets: {len(bets):,}")
    print(f"  Overall under win rate: {bets['under_win'].mean():.4f} (breakeven ≈ 0.5238 at -110)")

    # By season
    season_results = bets.groupby("season").agg(
        n_bets=("under_win", "count"),
        win_rate=("under_win", "mean"),
    ).round(4)
    print("\n  By season:")
    print(season_results.to_string())

    # Implied edge vs -110
    breakeven = 1 / (1 + 1 / 1.1)  # = 0.5238
    for season, row in season_results.iterrows():
        edge = row["win_rate"] - breakeven
        roi = edge * 2 * 100  # approximate ROI %
        print(f"    {season}: edge={edge:+.4f}  implied ROI≈{roi:+.1f}%  n={int(row['n_bets'])}")

    print("\n  CAVEATS:")
    if not use_real_lines:
        print("  - Rolling mean ≠ actual prop line. Market line accounts for matchup, rest, etc.")
        print("  - Actual edge will likely be lower once real lines are incorporated.")
    print("  - Requires minimum n≥200 per season for stable win rate estimates.")
    print("  - Next step: full walk-forward backtest in v2_dispersion_backtest.py")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("=" * 60)
    print("NBA POINTS DISPERSION RESEARCH — 20260526")
    print("=" * 60)

    df = load_player_game_logs()
    props = load_points_props()

    section1_overview(df)

    tg = section2_dispersion_metrics(df)
    section3_concentration_stability(tg)
    section4_intra_team_correlation(df)
    section5_star_overperformance(df)
    df_feat = section6_dispersion_features(df, tg, props=props)
    section7_roi_sketch(df_feat, props=props)

    print("\n" + "=" * 60)
    print("DONE — outputs written to:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
