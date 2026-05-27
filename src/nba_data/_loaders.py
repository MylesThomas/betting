"""Raw S3 loaders for player game logs, player props, and game lines."""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.nba_rebounds_modeling.duckdb_s3_creds import connect_duckdb_s3
from src.player_team_history.name_normalization import (
    normalize_from_nba_api,
    normalize_from_odds_api,
)
from src.player_team_history.team_normalization import (
    TEAM_ABBR_TO_NAME,
    normalize_team_code,
    normalize_team_name_from_odds_api,
)


def _abbr_to_full(abbr: str) -> str | None:
    if not abbr:
        return None
    normalized = normalize_team_code(str(abbr).strip())
    return TEAM_ABBR_TO_NAME.get(normalized, normalized)


def _add_matchup_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised: add home_team_normalized, away_team_normalized, is_home from MATCHUP."""
    is_home = df["MATCHUP"].str.contains(" vs. ", regex=False)
    # Opponent is always the token after "vs. " or "@ "
    opponent = df["MATCHUP"].str.split(r" vs\. | @ ", regex=True).str[1].str.strip()
    home_abbr = df["TEAM_ABBREVIATION"].where(is_home, opponent)
    away_abbr = opponent.where(is_home, df["TEAM_ABBREVIATION"])
    df["home_team_normalized"] = home_abbr.apply(_abbr_to_full)
    df["away_team_normalized"] = away_abbr.apply(_abbr_to_full)
    df["team_normalized"] = df["TEAM_ABBREVIATION"].apply(_abbr_to_full)
    df["is_home"] = is_home
    return df


def load_logs_raw(seasons: list[str]) -> pd.DataFrame:
    frames = []
    con = connect_duckdb_s3()
    try:
        for season in seasons:
            df = con.execute(f"""
                SELECT
                    '{season}' AS season,
                    PLAYER_ID, PLAYER_NAME, TEAM_NAME,
                    GAME_ID, GAME_DATE, MATCHUP, WL, MIN,
                    PTS, REB, AST, STL, BLK, TOV, PF, PLUS_MINUS,
                    FGM, FGA, FG3M, FG3A, FTM, FTA, OREB, DREB
                FROM read_csv_auto(
                    's3://nba-api-mt/player_game_logs/{season}/*.csv',
                    union_by_name=true
                )
            """).df()
            frames.append(df)
    finally:
        con.close()

    df = pd.concat(frames, ignore_index=True)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.date
    # Raw CSV has TEAM_NAME but not TEAM_ABBREVIATION; derive abbr from MATCHUP
    # MATCHUP format: "DEN vs. LAL" (home) or "DEN @ LAL" (away)
    df["TEAM_ABBREVIATION"] = df["MATCHUP"].str.split(r" vs\. | @ ", regex=True).str[0].str.strip()
    df["player_normalized"] = df["PLAYER_NAME"].apply(normalize_from_nba_api)
    df = _add_matchup_columns(df)
    return df


def load_props_raw(seasons: list[str]) -> pd.DataFrame:
    frames = []
    con = connect_duckdb_s3()
    try:
        for season in seasons:
            df = con.execute(f"""
                SELECT
                    '{season}' AS season,
                    player, home_team, away_team, game_time,
                    market, prop_line, bookmaker, over_odds, under_odds
                FROM read_csv_auto(
                    's3://the-odds-api-mt/nba/historical_player_props/{season}/*.csv',
                    union_by_name=true
                )
            """).df()
            frames.append(df)
    finally:
        con.close()

    df = pd.concat(frames, ignore_index=True)
    df["game_time"] = pd.to_datetime(df["game_time"], utc=True)
    df["game_date"] = df["game_time"].dt.tz_convert("America/New_York").dt.date
    df["player_normalized"] = df["player"].apply(normalize_from_odds_api)
    df["home_team"] = df["home_team"].apply(normalize_team_name_from_odds_api)
    df["away_team"] = df["away_team"].apply(normalize_team_name_from_odds_api)
    return df


def load_lines_raw(seasons: list[str]) -> pd.DataFrame:
    frames = []
    con = connect_duckdb_s3()
    try:
        for season in seasons:
            df = con.execute(f"""
                SELECT
                    '{season}' AS season,
                    home_team, away_team, bookmaker,
                    market, home_line, away_line, home_odds, away_odds,
                    filename
                FROM read_csv_auto(
                    's3://the-odds-api-mt/nba/historical_game_lines/{season}/nba_game_lines_*.csv',
                    union_by_name=true,
                    filename=true
                )
            """).df()
            frames.append(df)
    finally:
        con.close()

    df = pd.concat(frames, ignore_index=True)
    # Date lives in the filename: nba_game_lines_YYYY-MM-DD.csv
    df["game_date"] = pd.to_datetime(
        df["filename"].apply(lambda p: Path(str(p)).stem.replace("nba_game_lines_", ""))
    ).dt.date
    df = df.drop(columns=["filename"])
    df["home_team"] = df["home_team"].apply(normalize_team_name_from_odds_api)
    df["away_team"] = df["away_team"].apply(normalize_team_name_from_odds_api)
    return df
