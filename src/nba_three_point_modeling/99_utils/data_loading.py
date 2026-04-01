"""Data loading and contract construction utilities for v1 player_threes workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.player_team_history.name_normalization import normalize_from_nba_api
from src.player_team_history.name_normalization import normalize_from_odds_api
from src.player_team_history.team_normalization import normalize_team_name_from_odds_api
try:
    from .duckdb_s3 import connect_duckdb_s3
    from .odds import american_to_implied_prob
    from .odds import implied_prob_to_american
    from .odds import remove_vig_two_way
except ImportError:
    from duckdb_s3 import connect_duckdb_s3
    from odds import american_to_implied_prob
    from odds import implied_prob_to_american
    from odds import remove_vig_two_way


@dataclass
class V1DataBundle:
    """Container for model input and line contract data."""

    player_games_df: pd.DataFrame
    lines_df: pd.DataFrame


SPREAD_BUCKET_EDGES = [float("-inf"), -12.0, -8.0, -4.0, -1.0, 1.0, 4.0, 8.0, 12.0, float("inf")]
SPREAD_BUCKET_LABELS = [
    "(-inf,-12]",
    "(-12,-8]",
    "(-8,-4]",
    "(-4,-1]",
    "(-1,1]",
    "(1,4]",
    "(4,8]",
    "(8,12]",
    "(12,inf)",
]


def load_raw_player_game_logs_from_s3(season: str) -> pd.DataFrame:
    """Load raw player game logs for a season from S3 via DuckDB."""
    con = connect_duckdb_s3()
    query = f"""
    SELECT
      PLAYER_ID AS player_id,
      PLAYER_NAME AS player_name,
      TEAM_NAME AS team_name,
      GAME_ID AS game_id,
      GAME_DATE AS game_date,
      MATCHUP AS matchup,
      FG3M AS actual_fg3m,
      FG3A AS actual_fg3a,
      MIN AS actual_min
    FROM read_csv_auto('s3://nba-api-mt/player_game_logs/{season}/*.csv', union_by_name=true)
    """
    df = con.execute(query).fetchdf()
    con.close()
    return df


def load_raw_player_props_from_s3(season: str) -> pd.DataFrame:
    """Load raw player props for a season from S3 via DuckDB."""
    con = connect_duckdb_s3()
    query = f"""
    SELECT
      player,
      away_team,
      home_team,
      game_time,
      market,
      prop_line,
      bookmaker,
      over_odds,
      under_odds,
      filename
    FROM read_csv_auto(
      's3://the-odds-api-mt/nba/historical_player_props/{season}/*.csv',
      union_by_name=true,
      filename=true
    )
    """
    raw = con.execute(query).fetchdf()
    con.close()
    return raw


def load_raw_game_lines_from_s3(season: str) -> pd.DataFrame:
    """Load raw game lines for a season from S3 via DuckDB."""
    con = connect_duckdb_s3()
    query = f"""
    SELECT
      home_team,
      away_team,
      bookmaker,
      bookmaker_key,
      market,
      home_line,
      away_line,
      home_odds,
      away_odds,
      filename
    FROM read_csv_auto(
      's3://the-odds-api-mt/nba/historical_game_lines/{season}/nba_game_lines_*.csv',
      union_by_name=true,
      filename=true
    )
    """
    raw = con.execute(query).fetchdf()
    con.close()
    return raw


def prepare_player_game_logs(raw_game_logs_df: pd.DataFrame, player_name: str) -> pd.DataFrame:
    """Normalize and filter raw game logs for one player."""
    df = raw_game_logs_df.copy()
    df["player_normalized"] = df["player_name"].apply(normalize_from_nba_api)
    target_normalized = normalize_from_nba_api(player_name)
    df = df[df["player_normalized"] == target_normalized].copy()
    if df.empty:
        raise ValueError(f"No player game logs found for {player_name}")

    df["date"] = pd.to_datetime(df["game_date"]).dt.date.astype(str)
    df = df.sort_values("game_date").reset_index(drop=True)
    df["season_avg_3pm"] = (
        df["actual_fg3m"].astype(float).expanding(min_periods=1).mean().shift(1)
    )
    df["season_avg_3pm"] = df["season_avg_3pm"].fillna(df["actual_fg3m"].astype(float).mean())
    return df


def load_player_history_from_season_logs(
    player_name: str,
    history_seasons: list[str],
) -> pd.DataFrame:
    """Load and normalize player game history across one or more seasons."""
    frames = [load_raw_player_game_logs_from_s3(season=season) for season in history_seasons]
    raw = pd.concat(frames, ignore_index=True)
    df = raw.copy()
    df["player_normalized"] = df["player_name"].apply(normalize_from_nba_api)
    target_normalized = normalize_from_nba_api(player_name)
    df = df[df["player_normalized"] == target_normalized].copy()
    if df.empty:
        raise ValueError(f"No player history found for {player_name} in {history_seasons}")
    df["date"] = pd.to_datetime(df["game_date"]).dt.date.astype(str)
    df = df.sort_values("game_date").reset_index(drop=True)
    return df[
        [
            "player_id",
            "player_name",
            "game_id",
            "date",
            "actual_fg3m",
            "actual_fg3a",
            "actual_min",
        ]
    ].copy()


def prepare_player_props(raw_props_df: pd.DataFrame, player_name: str) -> pd.DataFrame:
    """Normalize and filter raw props for player_threes for one player."""
    df = raw_props_df.copy()
    df["player_normalized"] = df["player"].apply(normalize_from_odds_api)
    df["home_team"] = df["home_team"].apply(normalize_team_name_from_odds_api)
    df["away_team"] = df["away_team"].apply(normalize_team_name_from_odds_api)
    game_time_utc = pd.to_datetime(df["game_time"], utc=True)
    df["date"] = game_time_utc.dt.tz_convert("America/New_York").dt.date.astype(str)
    df["file_date"] = df["filename"].apply(lambda p: Path(str(p)).name.replace(".csv", ""))
    df = df[(df["market"] == "player_threes") & df["prop_line"].notna()].copy()

    target_normalized = normalize_from_nba_api(player_name)
    df = df[df["player_normalized"] == target_normalized].copy()
    if df.empty:
        raise ValueError(f"No player_threes props found for {player_name}")
    return df


def prepare_game_lines(raw_game_lines_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and aggregate raw game lines into per-game context rows."""
    df = raw_game_lines_df.copy()
    df["home_team"] = df["home_team"].apply(normalize_team_name_from_odds_api)
    df["away_team"] = df["away_team"].apply(normalize_team_name_from_odds_api)
    df["date"] = df["filename"].apply(
        lambda p: Path(str(p)).name.replace("nba_game_lines_", "").replace(".csv", "")
    )
    keep = df[df["market"].isin(["spread", "moneyline"])].copy()
    spread = (
        keep[keep["market"] == "spread"]
        .groupby(["date", "home_team", "away_team"], as_index=False)
        .agg(
            home_spread=("home_line", "median"),
            away_spread=("away_line", "median"),
            home_spread_odds=("home_odds", "median"),
            away_spread_odds=("away_odds", "median"),
        )
    )
    moneyline = (
        keep[keep["market"] == "moneyline"]
        .groupby(["date", "home_team", "away_team"], as_index=False)
        .agg(
            home_moneyline=("home_odds", "median"),
            away_moneyline=("away_odds", "median"),
        )
    )
    return spread.merge(moneyline, on=["date", "home_team", "away_team"], how="outer")


def build_consensus_and_contract_views(prepared_props_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build ladder contracts with consensus markers and median/best odds views.

    Consensus is chosen per player/date as the line closest to no-vig 50/50.
    """
    df = prepared_props_df.copy()
    df["implied_over"] = df["over_odds"].apply(american_to_implied_prob)
    df["implied_under"] = df["under_odds"].apply(american_to_implied_prob)
    no_vig = df.apply(
        lambda x: remove_vig_two_way(x["implied_over"], x["implied_under"]), axis=1
    )
    df["no_vig_over"] = [x[0] for x in no_vig]
    df["no_vig_under"] = [x[1] for x in no_vig]

    line_balance = (
        df.groupby(["date", "player_normalized", "prop_line"], as_index=False)["no_vig_over"]
        .median()
        .rename(columns={"no_vig_over": "median_no_vig_over"})
    )
    line_balance["distance_to_5050"] = (line_balance["median_no_vig_over"] - 0.5).abs()

    consensus_line = (
        line_balance.sort_values(["date", "player_normalized", "distance_to_5050"])
        .groupby(["date", "player_normalized"], as_index=False)
        .first()[["date", "player_normalized", "prop_line"]]
        .rename(columns={"prop_line": "consensus_line"})
    )
    df = df.merge(consensus_line, on=["date", "player_normalized"], how="left")
    df["is_consensus"] = (df["prop_line"] == df["consensus_line"]).astype(int)

    rows = []
    for keys, group in df.groupby(["date", "player_normalized", "prop_line"], dropna=False):
        date, player_normalized, prop_line = keys
        home_team = group["home_team"].iloc[0]
        away_team = group["away_team"].iloc[0]
        median_over_prob = group["over_odds"].apply(american_to_implied_prob).median()
        median_under_prob = group["under_odds"].apply(american_to_implied_prob).median()
        median_over_odds = implied_prob_to_american(float(median_over_prob))
        median_under_odds = implied_prob_to_american(float(median_under_prob))

        over_best_idx = group["over_odds"].idxmax()
        under_best_idx = group["under_odds"].idxmax()
        over_best_row = group.loc[over_best_idx]
        under_best_row = group.loc[under_best_idx]

        rows.append(
            {
                "date": date,
                "player_normalized": player_normalized,
                "line": float(prop_line),
                "home_team": home_team,
                "away_team": away_team,
                "is_consensus": int(group["is_consensus"].max()),
                "median_over_odds": float(median_over_odds),
                "median_under_odds": float(median_under_odds),
                "best_over_odds": float(over_best_row["over_odds"]),
                "best_under_odds": float(under_best_row["under_odds"]),
                "best_over_book": over_best_row["bookmaker"],
                "best_under_book": under_best_row["bookmaker"],
            }
        )
    contracts = pd.DataFrame(rows).sort_values(["date", "line"]).reset_index(drop=True)
    return contracts


def build_v1_data_bundle(season: str, player_name: str) -> V1DataBundle:
    """Load player games and ladder-aware line contracts for v1 run."""
    raw_logs = load_raw_player_game_logs_from_s3(season=season)
    raw_props = load_raw_player_props_from_s3(season=season)
    raw_game_lines = load_raw_game_lines_from_s3(season=season)

    player_games_df = prepare_player_game_logs(raw_logs, player_name=player_name)
    prepared_props_df = prepare_player_props(raw_props, player_name=player_name)
    game_lines_df = prepare_game_lines(raw_game_lines)
    lines_df = build_consensus_and_contract_views(prepared_props_df)

    lines_with_context = lines_df.merge(
        game_lines_df,
        on=["date", "home_team", "away_team"],
        how="left",
    )
    if lines_with_context.empty:
        raise ValueError("No joined line contracts and game line context for v1 run")

    return V1DataBundle(player_games_df=player_games_df, lines_df=lines_with_context)


def build_player_game_context_features(
    player_games_df: pd.DataFrame,
    lines_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build canonical game-context features for mean-model inputs.

    Context:
    - Production naming uses `team_point_spread` and `player_consensus_prop_line`.
    - Legacy aliases (`spread_signed`, `market_consensus_line`) are also emitted
      during transition to avoid breaking downstream scripts.
    """
    required_game_cols = ["date", "team_name"]
    required_line_cols = ["date", "line", "is_consensus", "home_team", "away_team", "home_spread", "away_spread"]
    for col in required_game_cols:
        if col not in player_games_df.columns:
            raise ValueError(f"Missing required player game column: {col}")
    for col in required_line_cols:
        if col not in lines_df.columns:
            raise ValueError(f"Missing required lines column: {col}")

    consensus_lines = lines_df[lines_df["is_consensus"] == 1].copy()
    context = consensus_lines[
        ["date", "line", "home_team", "away_team", "home_spread", "away_spread"]
    ].rename(columns={"line": "player_consensus_prop_line"})
    merged = player_games_df.merge(context, on="date", how="left")
    if merged["player_consensus_prop_line"].isna().any():
        raise ValueError("Missing player_consensus_prop_line for one or more game rows")

    home_mask = merged["team_name"] == merged["home_team"]
    away_mask = merged["team_name"] == merged["away_team"]
    if (~(home_mask | away_mask)).any():
        raise ValueError("Could not map team_name to home_team/away_team for spread assignment")

    merged["team_point_spread"] = np.where(
        home_mask,
        merged["home_spread"].astype(float),
        merged["away_spread"].astype(float),
    )
    merged["team_point_spread_abs"] = merged["team_point_spread"].abs()
    merged["team_point_spread_bucket"] = pd.cut(
        merged["team_point_spread"].astype(float),
        bins=SPREAD_BUCKET_EDGES,
        labels=SPREAD_BUCKET_LABELS,
        right=True,
    ).astype(str)

    # Legacy aliases during migration window.
    merged["spread_signed"] = merged["team_point_spread"]
    merged["market_consensus_line"] = merged["player_consensus_prop_line"]
    return merged

