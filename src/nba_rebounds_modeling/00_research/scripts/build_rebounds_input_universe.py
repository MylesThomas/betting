"""
Build rebounds input universe for feature engineering.

Context:
- This production script replaces dependency on legacy spread-universe artifacts.
- It builds canonical player-date-game inputs required by rebounds feature
  generation:
  - spread_signed
  - FGA
  - FG3A
  - FTA
- Supports full rebuild or append mode, and optional S3 publish.

Usage:
    python src/nba_rebounds_modeling/00_research/scripts/build_rebounds_input_universe.py \
        --season "*" \
        --output /tmp/rebounds_prod/cache/rebounds_input_universe.parquet \
        --s3-uri s3://nba-betting-mt/rebounds/input/rebounds_input_universe.parquet \
        --mode append
"""

from __future__ import annotations

import argparse
import os
import subprocess
from io import BytesIO
from pathlib import Path
import sys

import duckdb
import pandas as pd


def ensure_repo_root_on_syspath() -> Path:
    current = Path.cwd().resolve()
    while True:
        if (current / ".gitignore").exists() and (current / "src").exists():
            if str(current) not in sys.path:
                sys.path.insert(0, str(current))
            return current
        if current.parent == current:
            raise FileNotFoundError("Could not locate repo root")
        current = current.parent


ensure_repo_root_on_syspath()

from src.player_team_history.name_normalization import normalize_from_nba_api
from src.player_team_history.team_normalization import normalize_team_name_from_odds_api


TEAM_ABBR_TO_NAME = {
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",
}

KEY_COLS = ["season", "date", "player_normalized", "game_id"]
REQUIRED_COLS = KEY_COLS + ["spread_signed", "FGA", "FG3A", "FTA"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build rebounds input universe.")
    parser.add_argument("--season", type=str, default="*")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--s3-uri", type=str, default="")
    parser.add_argument("--mode", type=str, choices=("replace", "append"), default="append")
    return parser.parse_args()


def season_predicate(alias: str, season: str) -> str:
    if season.strip() in ("*", ""):
        return "TRUE"
    values = [x.strip() for x in season.split(",") if x.strip()]
    if len(values) == 1:
        return f"{alias}.season = '{values[0]}'"
    quoted = ", ".join(f"'{x}'" for x in values)
    return f"{alias}.season IN ({quoted})"


def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid s3 uri: {s3_uri}")
    rest = s3_uri[5:]
    bucket, _, key = rest.partition("/")
    if bucket == "" or key == "":
        raise ValueError(f"Invalid s3 uri: {s3_uri}")
    return bucket, key


def connect_duckdb_s3() -> duckdb.DuckDBPyConnection:
    if "AWS_ACCESS_KEY_ID" in os.environ and "AWS_SECRET_ACCESS_KEY" in os.environ:
        access_key = os.environ["AWS_ACCESS_KEY_ID"]
        secret_key = os.environ["AWS_SECRET_ACCESS_KEY"]
    else:
        access_key = subprocess.check_output(["aws", "configure", "get", "aws_access_key_id"], text=True).strip()
        secret_key = subprocess.check_output(["aws", "configure", "get", "aws_secret_access_key"], text=True).strip()
        if access_key == "" or secret_key == "":
            raise ValueError("Missing AWS credentials.")
    con = duckdb.connect()
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")
    con.execute("SET s3_region='us-east-2'")
    con.execute(f"SET s3_access_key_id='{access_key}'")
    con.execute(f"SET s3_secret_access_key='{secret_key}'")
    if "AWS_SESSION_TOKEN" in os.environ:
        con.execute(f"SET s3_session_token='{os.environ['AWS_SESSION_TOKEN']}'")
    return con


def load_player_day_inputs(season: str) -> pd.DataFrame:
    con = connect_duckdb_s3()
    query = f"""
    WITH logs AS (
      SELECT
        regexp_extract(filename, '/player_game_logs/([^/]+)/', 1) AS season,
        GAME_ID,
        GAME_DATE,
        PLAYER_NAME,
        FGA,
        FG3A,
        FTA
      FROM read_csv_auto(
        's3://nba-api-mt/player_game_logs/*/*.csv',
        union_by_name=true,
        filename=true,
        all_varchar=true,
        ignore_errors=true
      )
    ),
    history AS (
      SELECT
        player_normalized,
        team AS team_abbr,
        CAST(valid_from AS DATE) AS valid_from,
        CAST(valid_to AS DATE) AS valid_to
      FROM read_parquet('s3://nba-betting-mt/data/02_cache/player_team_history.parquet')
    )
    SELECT
      l.season,
      CAST(l.GAME_DATE AS DATE) AS game_date,
      l.GAME_ID AS game_id,
      l.PLAYER_NAME,
      l.FGA,
      l.FG3A,
      l.FTA,
      h.team_abbr
    FROM logs l
    INNER JOIN history h
      ON lower(trim(l.PLAYER_NAME)) = lower(trim(h.player_normalized))
      AND CAST(l.GAME_DATE AS DATE) >= h.valid_from
      AND (h.valid_to IS NULL OR CAST(l.GAME_DATE AS DATE) <= h.valid_to)
    WHERE {season_predicate('l', season)}
      AND l.GAME_ID IS NOT NULL
      AND l.PLAYER_NAME IS NOT NULL
      AND l.GAME_DATE IS NOT NULL
    """
    df = con.execute(query).fetchdf()
    con.close()
    df["player_normalized"] = df["PLAYER_NAME"].apply(normalize_from_nba_api)
    df["date"] = pd.to_datetime(df["game_date"]).dt.date.astype(str)
    df["team_normalized"] = df["team_abbr"].map(TEAM_ABBR_TO_NAME)
    missing = df["team_normalized"].isna()
    if missing.any():
        missing_abbr = sorted(df.loc[missing, "team_abbr"].dropna().unique().tolist())
        raise ValueError(f"Missing TEAM_ABBR_TO_NAME mapping for: {missing_abbr}")
    for c in ["FGA", "FG3A", "FTA"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    out = df[["season", "date", "player_normalized", "game_id", "team_normalized", "FGA", "FG3A", "FTA"]].copy()
    return out


def load_team_spreads(season: str) -> pd.DataFrame:
    con = connect_duckdb_s3()
    query = f"""
    WITH raw AS (
      SELECT
        home_team,
        away_team,
        market,
        home_line,
        away_line,
        regexp_extract(filename, '/historical_game_lines/([^/]+)/', 1) AS season,
        regexp_extract(filename, 'nba_game_lines_(\\d{{4}}-\\d{{2}}-\\d{{2}})\\.csv', 1) AS date
      FROM read_csv_auto(
        's3://the-odds-api-mt/nba/historical_game_lines/*/nba_game_lines_*.csv',
        union_by_name=true,
        filename=true,
        all_varchar=true
      )
    ),
    spread AS (
      SELECT
        season,
        date,
        home_team,
        away_team,
        median(CAST(home_line AS DOUBLE)) AS home_spread,
        median(CAST(away_line AS DOUBLE)) AS away_spread
      FROM raw r
      WHERE {season_predicate('r', season)}
        AND market = 'spread'
        AND home_line IS NOT NULL
        AND away_line IS NOT NULL
      GROUP BY season, date, home_team, away_team
    )
    SELECT
      season,
      date,
      home_team AS team_raw,
      home_spread AS spread_signed
    FROM spread
    UNION ALL
    SELECT
      season,
      date,
      away_team AS team_raw,
      away_spread AS spread_signed
    FROM spread
    """
    spread_df = con.execute(query).fetchdf()
    con.close()
    spread_df["team_normalized"] = spread_df["team_raw"].apply(normalize_team_name_from_odds_api)
    spread_df["spread_signed"] = pd.to_numeric(spread_df["spread_signed"], errors="coerce")
    return spread_df[["season", "date", "team_normalized", "spread_signed"]].drop_duplicates()


def load_team_spreads_for_calendar_date(season: str, calendar_date: str) -> pd.DataFrame:
    """
    Team-level spreads for a single calendar date (YYYY-MM-DD), Odds API historical lines.

    Used by pregame feature backfill so spread_signed matches the slate game, not the
    player's last completed game.
    """
    con = connect_duckdb_s3()
    date_lit = calendar_date.replace("'", "''")
    query = f"""
    WITH raw AS (
      SELECT
        home_team,
        away_team,
        market,
        home_line,
        away_line,
        regexp_extract(filename, '/historical_game_lines/([^/]+)/', 1) AS season,
        regexp_extract(filename, 'nba_game_lines_(\\d{{4}}-\\d{{2}}-\\d{{2}})\\.csv', 1) AS date
      FROM read_csv_auto(
        's3://the-odds-api-mt/nba/historical_game_lines/*/nba_game_lines_*.csv',
        union_by_name=true,
        filename=true,
        all_varchar=true
      )
    ),
    spread AS (
      SELECT
        season,
        date,
        home_team,
        away_team,
        median(CAST(home_line AS DOUBLE)) AS home_spread,
        median(CAST(away_line AS DOUBLE)) AS away_spread
      FROM raw r
      WHERE {season_predicate('r', season)}
        AND market = 'spread'
        AND home_line IS NOT NULL
        AND away_line IS NOT NULL
        AND r.date = '{date_lit}'
      GROUP BY season, date, home_team, away_team
    )
    SELECT
      season,
      date,
      home_team AS team_raw,
      home_spread AS spread_signed
    FROM spread
    UNION ALL
    SELECT
      season,
      date,
      away_team AS team_raw,
      away_spread AS spread_signed
    FROM spread
    """
    spread_df = con.execute(query).fetchdf()
    con.close()
    spread_df["team_normalized"] = spread_df["team_raw"].apply(normalize_team_name_from_odds_api)
    spread_df["spread_signed"] = pd.to_numeric(spread_df["spread_signed"], errors="coerce")
    return spread_df[["season", "date", "team_normalized", "spread_signed"]].drop_duplicates()


def validate_output(df: pd.DataFrame) -> None:
    for col in REQUIRED_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing required output column: {col}")
    null_keys = df[KEY_COLS].isna().any(axis=1)
    if null_keys.any():
        raise ValueError(f"Nulls detected in key columns: {int(null_keys.sum())} rows")
    dupes = df.duplicated(subset=KEY_COLS, keep=False)
    if dupes.any():
        raise ValueError(f"Duplicate key rows detected: {int(dupes.sum())}")


def read_s3_parquet_if_exists(s3_uri: str) -> pd.DataFrame:
    import boto3
    from botocore.exceptions import ClientError

    bucket, key = parse_s3_uri(s3_uri)
    s3 = boto3.client("s3")
    try:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except ClientError as exc:
        if exc.response["Error"]["Code"] in {"NoSuchKey", "404"}:
            return pd.DataFrame(columns=REQUIRED_COLS)
        raise
    return pd.read_parquet(BytesIO(body))


def upload_file_to_s3(local_path: Path, s3_uri: str) -> None:
    import boto3

    bucket, key = parse_s3_uri(s3_uri)
    boto3.client("s3").put_object(Bucket=bucket, Key=key, Body=local_path.read_bytes())


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve()

    player_inputs = load_player_day_inputs(args.season)
    spread_inputs = load_team_spreads(args.season)
    built = player_inputs.merge(
        spread_inputs,
        on=["season", "date", "team_normalized"],
        how="left",
    )
    built = built[REQUIRED_COLS].copy()

    if args.mode == "append" and args.s3_uri.strip() != "":
        existing = read_s3_parquet_if_exists(args.s3_uri.strip())
        merged = pd.concat([existing, built], ignore_index=True)
        merged = merged.drop_duplicates(subset=KEY_COLS, keep="last").copy()
        out = merged
    else:
        out = built

    out = out.sort_values(KEY_COLS).reset_index(drop=True)
    validate_output(out)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    if args.s3_uri.strip() != "":
        upload_file_to_s3(output_path, args.s3_uri.strip())
    print(
        "rebounds_input_universe_built",
        f"season={args.season}",
        f"mode={args.mode}",
        f"rows={len(out):,}",
        f"output={output_path}",
        f"s3_uri={args.s3_uri.strip()}",
        sep=" | ",
    )


if __name__ == "__main__":
    main()
