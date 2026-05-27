"""
Local integration test for dispersion_signal.py vs dispersion_backtest.py.

Two modes:
  --days N   Day-by-day comparison for the most recent N bet-dates.
             Checks that Lambda produces the same players/teams as the backtest.

  --full     Full-season validation across all historical bet-dates.
             Runs Lambda on every date, joins outcomes from logs, and
             checks whether total units matches the backtest's +1,733.5u.

Usage:
    python lambda/nba_dispersion_daily/test_local.py            # --days 25
    python lambda/nba_dispersion_daily/test_local.py --days 10
    python lambda/nba_dispersion_daily/test_local.py --full
"""
from __future__ import annotations

import argparse
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from dispersion_signal import (
    MAX_GAME_GAP_DAYS,
    MIN_GAMES_FOR_ROLLING,
    POINTS_MARKET,
    ROLLING_WINDOW,
    SPREAD_ROLL_WINDOW,
    STAR_THRESHOLD_SIGMA,
    _spread_q,
    build_features,
    build_roll_spread_map,
    compute_todays_plays,
    identify_stars,
)


# =============================================================================
# DATA LOADING  (uses get_data cache — same path as backtest)
# =============================================================================

def load_all(min_minutes: int = 5):
    from src.nba_data.get_data import get_data
    from src.player_team_history.team_normalization import normalize_team_name_from_odds_api

    print("Loading from cache ...")
    data = get_data(min_minutes=min_minutes)

    logs = data.logs.copy()
    logs.columns = logs.columns.str.lower()
    logs = logs.rename(columns={"min": "minutes"})
    logs["team_name"] = logs["team_normalized"]
    logs["game_date"] = pd.to_datetime(logs["game_date"]).dt.date
    logs = logs.sort_values(["player_id", "game_date"]).reset_index(drop=True)

    # Raw props — keep home/away/game_time_et for Lambda compatibility
    props_raw = data.props.copy()
    pts_props = props_raw[
        props_raw["market"].str.lower() == POINTS_MARKET
    ].copy()
    pts_props["game_date"] = pd.to_datetime(pts_props["game_date"]).dt.date
    pts_props["game_time_et"] = (
        pd.to_datetime(pts_props["game_time"], utc=True)
        .dt.tz_convert("America/New_York")
        .dt.strftime("%H:%M")
    )
    # Full props with home/away (for Lambda's compute_todays_plays)
    props_full = (
        pts_props.groupby(
            ["player_normalized", "game_date", "home_team", "away_team",
             "game_time_et", "season"],
            as_index=False,
        ).agg(prop_line=("prop_line", "median"))
    )
    # Simple props (player×date only) for backtest join
    props_simple = (
        pts_props.groupby(["player_normalized", "game_date"], as_index=False)
        .agg(prop_line=("prop_line", "median"))
    )

    lines = data.lines.copy()
    lines["game_date"] = pd.to_datetime(lines["game_date"]).dt.date

    print(f"  Logs:  {len(logs):,} player-games")
    print(f"  Props: {len(props_full):,} player×game lines (full), {len(props_simple):,} (simple)")
    print(f"  Lines: {len(lines):,} rows")
    return logs, props_full, props_simple, lines


# =============================================================================
# BACKTEST GROUND-TRUTH (mirrors dispersion_backtest.py build_bets exactly)
# =============================================================================

def run_backtest(
    logs: pd.DataFrame,
    props_simple: pd.DataFrame,
    lines: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return full historical bets DataFrame identical to dispersion_backtest.py.

    Adds two extra columns used for stratification:
      star_sigma_multiple  — trigger star's residual expressed in σ units
      spread_q             — team's rolling spread quintile at bet time (1=fav, 5=dog)
    """
    df = build_features(logs)
    df = identify_stars(df)

    star_resid_std = df[df["is_star"] & df["resid10"].notna()]["resid10"].std()
    threshold = STAR_THRESHOLD_SIGMA * star_resid_std
    print(f"  σ={star_resid_std:.2f}  threshold={threshold:.2f} pts")

    star_nights = df[
        df["is_star"]
        & (df["resid10"] > threshold)
        & df[f"roll{ROLLING_WINDOW}_pts"].notna()
        & (df["games_played_prior"] >= MIN_GAMES_FOR_ROLLING)
    ][["game_id", "team_name", "season", "game_date", "player_id", "resid10"]].rename(
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
    signal = signal[signal["gap_days"] <= MAX_GAME_GAP_DAYS]
    signal["star_sigma_multiple"] = signal["resid10"] / star_resid_std

    bets = df.merge(
        signal[["team_name", "season", "next_game_id", "star_id", "star_sigma_multiple"]],
        left_on=["team_name", "season", "game_id"],
        right_on=["team_name", "season", "next_game_id"],
    )
    bets = bets[
        (bets["player_id"] != bets["star_id"])
        & (~bets["is_star"])
        & bets[f"roll{ROLLING_WINDOW}_pts"].notna()
        & (bets["games_played_prior"] >= MIN_GAMES_FOR_ROLLING)
    ]
    bets = bets.merge(props_simple, on=["player_normalized", "game_date"], how="left")
    bets = bets.dropna(subset=["prop_line"]).copy()
    bets["game_date"] = pd.to_datetime(bets["game_date"]).dt.date

    # Keep best (highest σ) trigger per player×game when multiple stars fired.
    bets = bets.sort_values("star_sigma_multiple", ascending=False)
    bets_dedup = bets.drop_duplicates(subset=["player_id", "game_date", "team_name"])
    n_dupes = len(bets) - len(bets_dedup)
    if n_dupes:
        print(f"  Note: {n_dupes} duplicate rows removed (multiple stars firing same game)")

    # Add spread_q (1=elite fav … 5=heavy dog) for stratification
    if lines is not None and not lines.empty:
        roll_spread_map = build_roll_spread_map(lines, df)
        bets_dedup = bets_dedup.copy()
        bets_dedup["roll_spread"] = bets_dedup.apply(
            lambda r: roll_spread_map.get((r["team_name"], r["game_date"]), float("nan")),
            axis=1,
        )
        bets_dedup["spread_q"] = bets_dedup["roll_spread"].apply(
            lambda v: _spread_q(None if (isinstance(v, float) and np.isnan(v)) else v)
        )
    else:
        bets_dedup = bets_dedup.copy()
        bets_dedup["spread_q"] = 3

    return bets_dedup


# =============================================================================
# STRATIFIED BACKTEST TABLE
# =============================================================================

def print_stratified_table(bets: pd.DataFrame) -> None:
    """Print Season × Spread-Q × σ threshold breakdown of backtest units."""
    WIN_ODDS = 100 / 110
    SIGMAS = [1.0, 1.25, 1.5, 1.75, 2.0]
    Q_LABELS = {1: "Q1 fav", 2: "Q2", 3: "Q3 neu", 4: "Q4", 5: "Q5 dog"}
    seasons = sorted(bets["season"].unique())

    print(f"\n{'='*72}")
    print("STRATIFIED BACKTEST  —  Season × Spread-Q × σ threshold")
    print(f"{'='*72}")

    cell_w = 16
    header_cells = "  ".join(f"{'σ='+str(s):<{cell_w}}" for s in SIGMAS)
    print(f"\n  {'Season':<8}  {'Q':<7}  {header_cells}")
    print(f"  {'-'*8}  {'-'*7}  {'  '.join(['-'*cell_w]*len(SIGMAS))}")

    for season in seasons:
        season_bets = bets[bets["season"] == season]
        for q in range(1, 6):
            q_bets = season_bets[season_bets["spread_q"] == q]
            cells = []
            for sigma in SIGMAS:
                subset = q_bets[q_bets["star_sigma_multiple"] >= sigma]
                n = len(subset)
                if n == 0:
                    cells.append(f"{'—':<{cell_w}}")
                else:
                    wins = (subset["pts"] < subset["prop_line"]).sum()
                    losses = (subset["pts"] >= subset["prop_line"]).sum()
                    units = wins * WIN_ODDS + losses * (-1.0)
                    cell = f"{units:>+.0f}u (n={n})"
                    cells.append(f"{cell:<{cell_w}}")
            print(f"  {season:<8}  {Q_LABELS[q]:<7}  {'  '.join(cells)}")
        print()

    # Summary row: all seasons pooled
    print(f"  {'ALL':<8}  {'':7}  ", end="")
    row_parts = []
    for sigma in SIGMAS:
        subset = bets[bets["star_sigma_multiple"] >= sigma]
        n = len(subset)
        wins = (subset["pts"] < subset["prop_line"]).sum()
        losses = (subset["pts"] >= subset["prop_line"]).sum()
        wr = wins / (wins + losses) if (wins + losses) > 0 else 0.0
        units = wins * WIN_ODDS + losses * (-1.0)
        cell = f"{units:>+.0f}u ({wr:.1%})"
        row_parts.append(f"{cell:<{cell_w}}")
    print("  ".join(row_parts))
    print()


# =============================================================================
# COMPARISON
# =============================================================================

def compare(
    logs: pd.DataFrame,
    props_full: pd.DataFrame,
    lines: pd.DataFrame,
    backtest_bets: pd.DataFrame,
    n_days: int,
):
    all_dates = sorted(backtest_bets["game_date"].unique(), reverse=True)
    test_dates = sorted(all_dates[:n_days])

    print(f"\n{'='*72}")
    print(f"COMPARISON — last {len(test_dates)} bet-dates  (all dates ET)")
    print(f"{'='*72}")
    print(f"\n  {'Date':<12}  {'BT':>5}  {'LM':>5}  {'cnt':>5}  {'teams':>6}  {'players':>8}  skipped")
    print(f"  {'-'*12}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*6}  {'-'*8}  {'-'*30}")

    pass_count = 0
    total_bt_n = 0
    total_lm_n = 0
    mismatches = []

    for d in test_dates:
        bt_day = backtest_bets[backtest_bets["game_date"] == d]
        bt_n = len(bt_day)
        bt_teams = set(bt_day["team_name"].tolist())
        bt_players = set(bt_day["player_normalized"].tolist())
        total_bt_n += bt_n

        props_for_date = props_full[props_full["game_date"] == d].copy()
        if props_for_date.empty:
            print(f"  {str(d):<12}  {bt_n:>5}  {'—':>5}  {'—':>5}  {'—':>6}  {'no props':>8}")
            continue

        lm, lm_skipped = compute_todays_plays(logs, props_for_date, lines, d, verbose=False)
        lm_n = len(lm) if not lm.empty else 0
        total_lm_n += lm_n

        lm_teams = set(lm["team"].tolist()) if not lm.empty else set()
        lm_players = set(lm["player"].tolist()) if not lm.empty else set()

        n_ok = bt_n == lm_n
        teams_ok = bt_teams == lm_teams
        players_ok = bt_players == lm_players
        full_pass = n_ok and teams_ok and players_ok

        if full_pass:
            pass_count += 1

        # Summarise skipped on the same line: "n skipped (teams)"
        if not lm_skipped.empty:
            n_skip = len(lm_skipped)
            skip_teams = lm_skipped["team"].nunique()
            skip_summary = f"{n_skip} across {skip_teams}t"
        else:
            skip_summary = ""

        print(
            f"  {str(d):<12}  {bt_n:>5}  {lm_n:>5}  "
            f"{'✓' if n_ok else '✗':>5}  "
            f"{'✓' if teams_ok else '✗':>6}  "
            f"{'✓' if players_ok else '✗':>8}  "
            f"{skip_summary}"
        )

        if not full_pass:
            mismatches.append({
                "date": d,
                "bt_n": bt_n, "lm_n": lm_n,
                "missing_teams": sorted(bt_teams - lm_teams),
                "extra_teams": sorted(lm_teams - bt_teams),
                "missing_players": sorted(bt_players - lm_players),
                "extra_players": sorted(lm_players - bt_players),
                "skipped": lm_skipped,
            })

    print(f"\n  {'TOTAL':<12}  {total_bt_n:>5}  {total_lm_n:>5}")
    print(f"\n  Pass: {pass_count}/{len(test_dates)} dates  ({pass_count/len(test_dates):.0%})")

    if mismatches:
        print(f"\n{'='*72}")
        print("MISMATCH DETAILS")
        print(f"{'='*72}")
        for m in mismatches:
            print(f"\n  {m['date']}")
            if m["missing_teams"]:
                print(f"    missing teams:   {m['missing_teams']}")
            if m["extra_teams"]:
                print(f"    extra teams:     {m['extra_teams']}")
            if m["missing_players"]:
                print(f"    missing players: {m['missing_players']}")
            if m["extra_players"]:
                print(f"    extra players:   {m['extra_players']}")
            skipped = m["skipped"]
            if not skipped.empty:
                print(f"    no prop (skipped):")
                for team, grp in skipped.groupby("team"):
                    names = ", ".join(sorted(grp["player"].tolist()))
                    print(f"      {team}: {names}")

    return pass_count, len(test_dates)


# =============================================================================
# FULL VALIDATION  (--full flag)
# =============================================================================

def run_full_validation(
    logs: pd.DataFrame,
    props_full: pd.DataFrame,
    lines: pd.DataFrame,
    backtest_bets: pd.DataFrame,
):
    """
    Run Lambda signal over every historical bet-date, compute outcomes from
    logs, and report total units vs the backtest reference of +1,733.5u.
    """
    WIN_ODDS = 100 / 110  # -110 payout per unit risked

    all_dates = sorted(backtest_bets["game_date"].unique())
    n_dates = len(all_dates)

    print(f"\n{'='*72}")
    print(f"FULL VALIDATION — {n_dates} bet-dates")
    print(f"{'='*72}")

    # Pre-compute features once — avoids re-running build_features per date
    print("\nPre-computing features (one-time) ...")
    df_pre = build_features(logs)
    df_pre = identify_stars(df_pre)
    print(f"  Done. {len(df_pre):,} player-game rows")

    print(f"\nRunning Lambda signal across {n_dates} dates ...")
    all_plays = []
    for i, d in enumerate(all_dates):
        if i % 100 == 0 and i > 0:
            print(f"  {i}/{n_dates} ...")
        props_for_date = props_full[props_full["game_date"] == d].copy()
        if props_for_date.empty:
            continue
        plays, _ = compute_todays_plays(
            logs, props_for_date, lines, d,
            verbose=False, _precomputed_df=df_pre,
        )
        if not plays.empty:
            all_plays.append(plays)

    if not all_plays:
        print("  No plays generated.")
        return

    lm = pd.concat(all_plays, ignore_index=True)
    lm["game_date_dt"] = pd.to_datetime(lm["date"]).dt.date

    # Join actual pts from logs on bet date
    actuals = (
        logs[["player_normalized", "game_date", "pts"]]
        .rename(columns={"pts": "actual_pts"})
    )
    lm = lm.merge(
        actuals,
        left_on=["player", "game_date_dt"],
        right_on=["player_normalized", "game_date"],
        how="left",
    )

    # Compute units: push if no log entry (DNP), win if actual < line, else loss
    lm["push"] = lm["actual_pts"].isna()
    lm["win"]  = (~lm["push"]) & (lm["actual_pts"] < lm["prop_line"])
    lm["loss"] = (~lm["push"]) & (~lm["win"])
    lm["units"] = lm["win"].astype(float) * WIN_ODDS + lm["loss"].astype(float) * -1.0

    lm_n     = len(lm)
    lm_wins  = int(lm["win"].sum())
    lm_losses = int(lm["loss"].sum())
    lm_pushes = int(lm["push"].sum())
    lm_u     = lm["units"].sum()
    lm_wr    = lm_wins / (lm_wins + lm_losses) if (lm_wins + lm_losses) > 0 else 0.0

    # Compute backtest units from data (pts vs prop_line already in backtest_bets)
    bt_n    = len(backtest_bets)
    bt_win  = (backtest_bets["pts"] < backtest_bets["prop_line"]).sum()
    bt_loss = (backtest_bets["pts"] >= backtest_bets["prop_line"]).sum()
    bt_wr   = bt_win / (bt_win + bt_loss) if (bt_win + bt_loss) > 0 else 0.0
    bt_u    = bt_win * WIN_ODDS + bt_loss * (-1.0)

    print(f"\n{'='*72}")
    print(f"  {'':30}  {'BACKTEST':>12}  {'LAMBDA':>12}")
    print(f"  {'-'*30}  {'-'*12}  {'-'*12}")
    print(f"  {'Bets':30}  {bt_n:>12,}  {lm_n:>12,}")
    print(f"  {'Bet-dates':30}  {backtest_bets['game_date'].nunique():>12,}  {lm['game_date_dt'].nunique():>12,}")
    print(f"  {'Wins':30}  {int(bt_win):>12,}  {lm_wins:>12,}")
    print(f"  {'Losses':30}  {int(bt_loss):>12,}  {lm_losses:>12,}")
    print(f"  {'Pushes (DNP)':30}  {'—':>12}  {lm_pushes:>12,}")
    print(f"  {'Win rate (excl. pushes)':30}  {bt_wr:>12.1%}  {lm_wr:>12.1%}")
    print(f"  {'Total units at −110':30}  {bt_u:>+12.1f}  {lm_u:>+12.1f}")
    print(f"  {'Delta':30}  {'':>12}  {lm_u - bt_u:>+12.1f}")
    print(f"{'='*72}")

    delta_pct = abs(lm_u - bt_u) / max(abs(bt_u), 1) * 100
    if delta_pct < 5:
        print(f"\nRESULT: ✓  Lambda within {delta_pct:.1f}% of backtest units")
    else:
        print(f"\nRESULT: ✗  Lambda is {delta_pct:.1f}% off backtest — investigate")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=25,
                        help="Number of most-recent bet-dates to compare day-by-day (default 25)")
    parser.add_argument("--full", action="store_true",
                        help="Run full-season validation and compute total units")
    args = parser.parse_args()

    if args.full:
        print("=" * 72)
        print("DISPERSION SIGNAL — FULL UNITS VALIDATION")
        print("=" * 72)
        print()
        logs, props_full, props_simple, lines = load_all()
        print("\nRunning backtest (ground truth) ...")
        backtest_bets = run_backtest(logs, props_simple, lines)
        print(f"  Backtest total bets: {len(backtest_bets):,} across {backtest_bets['game_date'].nunique()} dates")
        run_full_validation(logs, props_full, lines, backtest_bets)
        print_stratified_table(backtest_bets)
    else:
        print("=" * 72)
        print("DISPERSION SIGNAL — LOCAL INTEGRATION TEST")
        print(f"Comparing Lambda signal vs backtest for last {args.days} bet-dates")
        print("=" * 72)
        print()
        logs, props_full, props_simple, lines = load_all()
        print("\nRunning backtest (ground truth) ...")
        backtest_bets = run_backtest(logs, props_simple, lines)
        print(f"  Backtest total bets: {len(backtest_bets):,} across {backtest_bets['game_date'].nunique()} dates")
        pass_count, total = compare(logs, props_full, lines, backtest_bets, args.days)
        print()
        if pass_count == total:
            print("RESULT: ALL PASS ✓ — Lambda signal matches backtest exactly")
        else:
            print(f"RESULT: {total - pass_count} date(s) have mismatches — review details above")


if __name__ == "__main__":
    main()
