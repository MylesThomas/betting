"""
Dispersion signal for daily Lambda.

Adapted from src/nba_points_modeling/dispersion_backtest.py.
Computes today's qualifying UNDER bets on non-star teammates after a star night.
All feature engineering is identical to the backtest (shift(1) rolling, min 10 games, etc.).
"""
from __future__ import annotations

import sys
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import boto3
import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Mirror backtest constants exactly
ROLLING_WINDOW = 10
MIN_GAMES_FOR_ROLLING = 10
STAR_THRESHOLD_SIGMA = 1.0
MAX_GAME_GAP_DAYS = 5
SPREAD_ROLL_WINDOW = 15
POINTS_MARKET = "player_points"
MIN_MINUTES = 5

LOGS_BUCKET = "nba-api-mt"
ODDS_BUCKET = "the-odds-api-mt"

SPREAD_Q_LABELS = {
    1: "Elite favorite",
    2: "Moderate favorite",
    3: "Neutral / toss-up",
    4: "Moderate underdog",
    5: "Heavy underdog",
}


# =============================================================================
# HELPERS
# =============================================================================

def current_season(d: date) -> str:
    """Return NBA season string for a given date, e.g. '2025-26'."""
    year = d.year
    if d.month >= 10:
        return f"{year}-{str(year + 1)[2:]}"
    return f"{year - 1}-{str(year)[2:]}"


def _spread_q(roll_spread: float | None) -> int:
    if roll_spread is None or np.isnan(roll_spread):
        return 3
    if roll_spread < -7.0:
        return 1
    if roll_spread < -2.0:
        return 2
    if roll_spread <= 2.0:
        return 3
    if roll_spread <= 7.0:
        return 4
    return 5


def _read_s3_csv(bucket: str, key: str) -> pd.DataFrame | None:
    s3 = boto3.client("s3")
    try:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        return pd.read_csv(BytesIO(body))
    except s3.exceptions.NoSuchKey:
        return None
    except Exception as exc:
        print(f"  warn: could not read s3://{bucket}/{key} — {exc}")
        return None


# =============================================================================
# DATA LOADING
# =============================================================================

def load_logs(season: str) -> pd.DataFrame:
    """Load all player game logs for the season via DuckDB httpfs glob."""
    from src.nba_rebounds_modeling.duckdb_s3_creds import connect_duckdb_s3
    from src.player_team_history.name_normalization import normalize_from_nba_api
    from src.player_team_history.team_normalization import (
        TEAM_ABBR_TO_NAME,
        normalize_team_code,
        normalize_team_name_from_odds_api,
    )

    def _abbr_to_full(abbr: str) -> str | None:
        if not abbr:
            return None
        n = normalize_team_code(str(abbr).strip())
        return TEAM_ABBR_TO_NAME.get(n, n)

    print(f"  Loading logs for {season} ...")
    con = connect_duckdb_s3()
    try:
        df = con.execute(f"""
            SELECT
                '{season}' AS season,
                PLAYER_ID, PLAYER_NAME, TEAM_NAME,
                GAME_ID, GAME_DATE, MATCHUP, MIN, PTS
            FROM read_csv_auto(
                's3://{LOGS_BUCKET}/player_game_logs/{season}/*.csv',
                union_by_name=true
            )
        """).df()
    finally:
        con.close()

    df.columns = df.columns.str.lower()
    df = df.rename(columns={"min": "minutes"})
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    df["team_abbr"] = df["matchup"].str.split(r" vs\. | @ ", regex=True).str[0].str.strip()
    df["team_name"] = df["team_abbr"].apply(_abbr_to_full)
    df["player_normalized"] = df["player_name"].apply(normalize_from_nba_api)

    is_home = df["matchup"].str.contains(" vs. ", regex=False)
    opponent = df["matchup"].str.split(r" vs\. | @ ", regex=True).str[1].str.strip()
    df["opponent"] = opponent.apply(_abbr_to_full)
    df["is_home"] = is_home

    df = df[df["minutes"].notna()].copy()
    try:
        df = df[pd.to_numeric(df["minutes"], errors="coerce") >= MIN_MINUTES].copy()
    except Exception:
        pass

    print(f"    {len(df):,} player-games loaded")
    return df.sort_values(["player_id", "game_date"]).reset_index(drop=True)


def load_props_today(season: str, today: date) -> pd.DataFrame | None:
    """Load today's prop lines from S3 (single date CSV)."""
    from src.player_team_history.name_normalization import normalize_from_odds_api
    from src.player_team_history.team_normalization import normalize_team_name_from_odds_api

    key = f"nba/historical_player_props/{season}/{today}.csv"
    print(f"  Loading props: s3://{ODDS_BUCKET}/{key}")
    df = _read_s3_csv(ODDS_BUCKET, key)
    if df is None or df.empty:
        print("    No props found for today.")
        return None

    df = df[df["market"].str.lower() == POINTS_MARKET].copy()
    if df.empty:
        return None

    df["game_time"] = pd.to_datetime(df["game_time"], utc=True)
    df["game_time_et"] = df["game_time"].dt.tz_convert("America/New_York").dt.strftime("%H:%M")
    df["game_date"] = df["game_time"].dt.tz_convert("America/New_York").dt.date
    df["player_normalized"] = df["player"].apply(normalize_from_odds_api)
    df["home_team"] = df["home_team"].apply(normalize_team_name_from_odds_api)
    df["away_team"] = df["away_team"].apply(normalize_team_name_from_odds_api)

    # Median line per player×game across bookmakers
    props = (
        df.groupby(["player_normalized", "game_date", "home_team", "away_team", "game_time_et"])
        .agg(prop_line=("prop_line", "median"))
        .reset_index()
    )
    print(f"    {len(props):,} player prop lines found")
    return props


def load_lines_season(season: str) -> pd.DataFrame | None:
    """Load all game lines for the season for roll_spread computation."""
    from src.player_team_history.team_normalization import normalize_team_name_from_odds_api
    from pathlib import Path as _Path

    print(f"  Loading lines for {season} ...")
    from src.nba_rebounds_modeling.duckdb_s3_creds import connect_duckdb_s3
    con = connect_duckdb_s3()
    try:
        df = con.execute(f"""
            SELECT home_team, away_team, market, home_line, away_line, filename
            FROM read_csv_auto(
                's3://{ODDS_BUCKET}/nba/historical_game_lines/{season}/nba_game_lines_*.csv',
                union_by_name=true,
                filename=true
            )
        """).df()
    finally:
        con.close()

    if df.empty:
        return None

    df["game_date"] = pd.to_datetime(
        df["filename"].apply(lambda p: _Path(str(p)).stem.replace("nba_game_lines_", ""))
    ).dt.date
    df["home_team"] = df["home_team"].apply(normalize_team_name_from_odds_api)
    df["away_team"] = df["away_team"].apply(normalize_team_name_from_odds_api)
    print(f"    {len(df):,} line rows loaded")
    return df.drop(columns=["filename"])


# =============================================================================
# FEATURE ENGINEERING  (identical to dispersion_backtest.py)
# =============================================================================

def build_features(logs: pd.DataFrame) -> pd.DataFrame:
    df = logs.sort_values(["player_id", "game_date"]).copy()

    team_totals = df.groupby(["game_id", "team_name"])["pts"].sum().reset_index(name="team_pts")
    df = df.merge(team_totals, on=["game_id", "team_name"], how="left")
    df["pts_share"] = df["pts"] / df["team_pts"].replace(0, np.nan)

    def _roll(s: pd.Series, w: int) -> pd.Series:
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
        .groupby(["team_name", "season"])
        .head(3)
        .assign(is_star=True)
    )
    df = df.merge(
        stars[["player_id", "team_name", "season", "is_star"]],
        on=["player_id", "team_name", "season"],
        how="left",
    )
    df["is_star"] = df["is_star"].fillna(False).astype(bool)
    return df


def build_roll_spread_map(lines: pd.DataFrame, logs: pd.DataFrame) -> dict[tuple, float]:
    """
    Returns a dict keyed by (team_name, game_date) → roll_spread for today's games.
    Only computes for teams present in logs.
    """
    spread = lines[lines["market"] == "spread"].copy()
    home = (
        spread.groupby(["home_team", "game_date"])["home_line"]
        .median().reset_index()
        .rename(columns={"home_team": "team_name", "home_line": "spread_signed"})
    )
    away = (
        spread.groupby(["away_team", "game_date"])["away_line"]
        .median().reset_index()
        .rename(columns={"away_team": "team_name", "away_line": "spread_signed"})
    )
    team_spreads = pd.concat([home, away], ignore_index=True)

    team_games = (
        logs[["game_id", "team_name", "game_date", "season"]]
        .drop_duplicates()
        .merge(team_spreads, on=["team_name", "game_date"], how="left")
        .sort_values(["team_name", "season", "game_date"])
    )
    team_games["roll_spread"] = (
        team_games.groupby(["team_name", "season"])["spread_signed"]
        .transform(lambda s: s.shift(1).rolling(SPREAD_ROLL_WINDOW, min_periods=max(3, SPREAD_ROLL_WINDOW // 3)).mean())
    )
    result = {}
    for _, row in team_games.iterrows():
        result[(row["team_name"], row["game_date"])] = row["roll_spread"]
    return result


# =============================================================================
# MAIN SIGNAL
# =============================================================================

def compute_todays_plays(
    logs: pd.DataFrame,
    props_today: pd.DataFrame,
    lines: pd.DataFrame | None,
    today: date,
    verbose: bool = True,
    _precomputed_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (plays, skipped) where:
      plays   — flat DataFrame, one row per UNDER bet for today
      skipped — flat DataFrame, one row per eligible teammate with no prop line
                columns: team, player
    Both are empty DataFrames when there are no qualifying plays.
    """
    def _log(msg: str) -> None:
        if verbose:
            print(msg)

    if _precomputed_df is not None:
        df = _precomputed_df
    else:
        df = build_features(logs)
        df = identify_stars(df)

    star_resid_std = df[df["is_star"] & df["resid10"].notna()]["resid10"].std()
    threshold = STAR_THRESHOLD_SIGMA * star_resid_std
    _log(f"  Star σ={star_resid_std:.2f}  threshold={threshold:.2f} pts")

    # Build roll_spread lookup (display-only)
    roll_spread_map: dict = {}
    if lines is not None and not lines.empty:
        roll_spread_map = build_roll_spread_map(lines, df)

    # Teams playing today (union of home + away from props)
    teams_today = set(props_today["home_team"].tolist()) | set(props_today["away_team"].tolist())
    _log(f"  Teams with props today: {len(teams_today)}")

    # Restrict to completed games — today's games haven't been played yet.
    # This keeps historical testing correct when full logs are passed.
    completed = df[df["game_date"] < today]

    # Most recent completed game per team, gap-checked against today
    team_last = (
        completed[completed["team_name"].isin(teams_today)]
        .groupby("team_name")["game_date"].max()
        .reset_index().rename(columns={"game_date": "last_game_date"})
    )
    team_last["gap_days"] = (
        pd.to_datetime(today) - pd.to_datetime(team_last["last_game_date"])
    ).dt.days
    team_last = team_last[team_last["gap_days"] <= MAX_GAME_GAP_DAYS]

    if team_last.empty:
        _log("  No teams within gap window.")
        return pd.DataFrame(), pd.DataFrame()

    # Find qualifying star nights in those last games
    trigger_pool = completed.merge(
        team_last[["team_name", "last_game_date"]].rename(columns={"last_game_date": "game_date"}),
        on=["team_name", "game_date"],
    )
    star_nights = trigger_pool[
        trigger_pool["is_star"]
        & (trigger_pool["resid10"] > threshold)
        & trigger_pool[f"roll{ROLLING_WINDOW}_pts"].notna()
        & (trigger_pool["games_played_prior"] >= MIN_GAMES_FOR_ROLLING)
    ].copy()

    if star_nights.empty:
        _log("  No qualifying star nights.")
        return pd.DataFrame(), pd.DataFrame()

    # One trigger per team: highest resid wins if multiple stars fired
    star_nights["sigma_multiple"] = star_nights["resid10"] / star_resid_std
    best_trigger = (
        star_nights.sort_values("sigma_multiple", ascending=False)
        .drop_duplicates(subset=["team_name"])
    )
    _log(f"  Qualifying triggers: {len(best_trigger)} team(s)")

    # Opponent map for today's games
    opponent_map: dict[str, str] = {}
    game_time_map: dict[str, str] = {}
    for _, row in props_today.drop_duplicates(subset=["home_team", "away_team"]).iterrows():
        ht, at = row["home_team"], row["away_team"]
        opponent_map[ht] = at
        opponent_map[at] = ht
        game_time_map[ht] = row["game_time_et"]
        game_time_map[at] = row["game_time_et"]

    # Deduplicate props by player — one prop_line per player for today's bet
    props_dedup = (
        props_today[["player_normalized", "prop_line"]]
        .groupby("player_normalized", as_index=False)
        .median()
    )

    # Build bet rows
    rows = []
    skipped_rows = []
    generated_at = datetime.now(ET).strftime("%Y-%m-%dT%H:%M:%SZ")
    season = current_season(today)

    for _, trigger in best_trigger.iterrows():
        team = trigger["team_name"]
        trigger_game_date = trigger["game_date"]
        gap = int(
            team_last[team_last["team_name"] == team]["gap_days"].iloc[0]
        )

        roll_spread_val = roll_spread_map.get((team, trigger_game_date), float("nan"))
        roll_spread_display = round(float(roll_spread_val), 1) if not np.isnan(roll_spread_val) else None
        q = _spread_q(roll_spread_val if not np.isnan(roll_spread_val) else None)

        # Non-star teammates who played in the trigger game — the dispersion
        # mechanism only applies to players who were on the floor that night
        # and had their shots compressed by the star's big game.
        # "Prop available tonight" is the proxy for "will play tonight" (inner
        # join below); DNPs settle as push.
        eligible = completed[
            (completed["team_name"] == team)
            & (completed["game_date"] == trigger_game_date)
            & (completed["player_id"] != trigger["player_id"])
            & (~completed["is_star"])
            & completed[f"roll{ROLLING_WINDOW}_pts"].notna()
            & (completed["games_played_prior"] >= MIN_GAMES_FOR_ROLLING)
        ].copy()

        # Split into those with / without a prop line tonight
        has_prop = eligible.merge(props_dedup, on="player_normalized", how="inner")
        no_prop_names = sorted(
            set(eligible["player_normalized"]) - set(has_prop["player_normalized"])
        )
        for name in no_prop_names:
            skipped_rows.append({"team": team, "player": name})

        if has_prop.empty:
            continue

        for _, bet in has_prop.iterrows():
            rows.append({
                "date": str(today),
                "generated_at": generated_at,
                "sigma": STAR_THRESHOLD_SIGMA,
                "threshold_pts": round(threshold, 2),
                "team": team,
                "opponent": opponent_map.get(team, ""),
                "game_time_et": game_time_map.get(team, ""),
                "gap_days": gap,
                "trigger_player": trigger["player_normalized"],
                "trigger_pts": int(trigger["pts"]),
                "trigger_roll_avg": round(float(trigger[f"roll{ROLLING_WINDOW}_pts"]), 1),
                "trigger_resid": round(float(trigger["resid10"]), 1),
                "trigger_sigma_multiple": round(float(trigger["sigma_multiple"]), 2),
                "trigger_game_date": str(trigger_game_date),
                "team_roll_spread": roll_spread_display,
                "spread_q": q,
                "spread_q_label": SPREAD_Q_LABELS[q],
                "player": bet["player_normalized"],
                "prop_line": float(bet["prop_line"]),
                "roll_avg": round(float(bet[f"roll{ROLLING_WINDOW}_pts"]), 1),
                "prior_games": int(bet["games_played_prior"]),
                "direction": "UNDER",
            })

    skipped = pd.DataFrame(skipped_rows, columns=["team", "player"]) if skipped_rows else pd.DataFrame(columns=["team", "player"])

    if not rows:
        _log("  No bets after prop line join.")
        return pd.DataFrame(), skipped

    plays = pd.DataFrame(rows)
    _log(f"  {len(plays)} bet(s) across {plays['team'].nunique()} team(s)")
    return plays, skipped
