"""
Data loading for the NCAAB Away Revenge strategy dashboard.

S3 layout:
  Plays:    s3://ncaab-betting-mt/data/04_output/plays/fade-revenge-spot/{date}.csv
  Outcomes: s3://ncaab-betting-mt/data/01_input/historical_game_results/{date}.csv

No st.* rendering. No try/except. No defensive column checks.
"""

from __future__ import annotations

from datetime import date

import boto3
import duckdb
import numpy as np
import pandas as pd
import streamlit as st

S3_BUCKET: str = "ncaab-betting-mt"
PLAYS_PREFIX: str = "data/04_output/plays/fade-revenge-spot"
OUTCOMES_PREFIX: str = "data/01_input/historical_game_results"
PROD_GO_LIVE_DATE: str = "2026-02-19"
NCAAB_PAUSE_UNTIL: date = date(2026, 11, 3)
WIN_PAYOUT: float = 100.0 / 110.0  # -110 juice


def _s3_client():
    return boto3.client("s3")


def _duckdb_s3_conn() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB connection with S3 credentials sourced from boto3."""
    con = duckdb.connect()
    con.execute("LOAD httpfs")
    session = boto3.Session()
    creds = session.get_credentials()
    if creds:
        frozen = creds.get_frozen_credentials()
        con.execute(f"SET s3_access_key_id='{frozen.access_key}'")
        con.execute(f"SET s3_secret_access_key='{frozen.secret_key}'")
        if frozen.token:
            con.execute(f"SET s3_session_token='{frozen.token}'")
    con.execute(f"SET s3_region='{session.region_name or 'us-east-1'}'")
    return con


@st.cache_data(ttl=900, show_spinner=False)
def _list_plays_keys() -> list[str]:
    """All CSV keys under the plays prefix. 15-min TTL so today's file appears promptly."""
    paginator = _s3_client().get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=PLAYS_PREFIX + "/"):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".csv"):
                keys.append(obj["Key"])
    return keys


@st.cache_data(ttl=3600, show_spinner=False)
def _list_outcome_keys() -> set[str]:
    """All CSV keys under the outcomes prefix. 1-hour TTL; historical results don't change."""
    paginator = _s3_client().get_paginator("list_objects_v2")
    keys: set[str] = set()
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=OUTCOMES_PREFIX + "/"):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".csv"):
                keys.add(obj["Key"])
    return keys


@st.cache_data(ttl=900, show_spinner=False)
def load_all_plays() -> pd.DataFrame:
    """Read all plays CSVs in parallel via DuckDB httpfs glob."""
    if not _list_plays_keys():
        return pd.DataFrame()
    con = _duckdb_s3_conn()
    df: pd.DataFrame = con.execute(f"""
        SELECT * FROM read_csv_auto(
            's3://{S3_BUCKET}/{PLAYS_PREFIX}/*.csv',
            union_by_name=true,
            ignore_errors=true
        )
    """).df()
    con.close()
    if df.empty:
        return df
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def _load_outcomes_for_dates(game_dates: tuple[str, ...]) -> pd.DataFrame:
    """Read outcome CSVs for specific dates in parallel via DuckDB httpfs."""
    available_keys: set[str] = _list_outcome_keys()
    file_paths: list[str] = [
        f"s3://{S3_BUCKET}/{OUTCOMES_PREFIX}/{d}.csv"
        for d in game_dates
        if f"{OUTCOMES_PREFIX}/{d}.csv" in available_keys
    ]
    if not file_paths:
        return pd.DataFrame()
    con = _duckdb_s3_conn()
    paths_literal = "[" + ", ".join(f"'{p}'" for p in file_paths) + "]"
    df: pd.DataFrame = con.execute(f"""
        SELECT * FROM read_csv_auto(
            {paths_literal},
            union_by_name=true,
            ignore_errors=true
        )
    """).df()
    con.close()
    if df.empty:
        return df
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.drop_duplicates(subset=["GAME_DATE", "HOME_TEAM", "AWAY_TEAM"])
    return df


def settle_plays(plays: pd.DataFrame) -> pd.DataFrame:
    """
    Join plays with outcomes; compute result and pnl_units per play.
    Rows with bet_team null get result='no_play'.
    Rows missing a spread get result='no_line'.
    Rows whose outcome hasn't been fetched yet get result='pending'.
    """
    if plays.empty:
        return plays

    work: pd.DataFrame = plays.copy().reset_index(drop=True)

    is_play: pd.Series = (
        work["bet_team"].notna()
        & (work["bet_team"].astype(str).str.strip() != "")
        & (work["bet_team"].astype(str).str.strip().str.lower() != "nan")
    )
    has_line: pd.Series = work["consensus_spread_home"].notna()

    work["result"] = "no_play"
    work["pnl_units"] = float("nan")
    work["spread_margin"] = float("nan")
    work["home_score"] = float("nan")
    work["away_score"] = float("nan")

    work.loc[is_play & ~has_line, "result"] = "no_line"

    # Only load outcomes for past dates — today's games haven't been played yet
    past_play_dates: tuple[str, ...] = tuple(
        d for d in work.loc[is_play & has_line, "game_date"]
        .dt.strftime("%Y-%m-%d").unique().tolist()
        if d < str(date.today())
    )
    outcomes: pd.DataFrame = (
        _load_outcomes_for_dates(past_play_dates) if past_play_dates else pd.DataFrame()
    )

    if outcomes.empty:
        work.loc[is_play & has_line, "result"] = "pending"
        return work

    outcomes_j: pd.DataFrame = outcomes.copy()
    outcomes_j["_jdate"] = outcomes_j["GAME_DATE"].dt.strftime("%Y-%m-%d")
    outcomes_j["_jhome"] = outcomes_j["HOME_TEAM"].astype(str).str.strip()
    outcomes_j["_jaway"] = outcomes_j["AWAY_TEAM"].astype(str).str.strip()

    work["_jdate"] = work["game_date"].dt.strftime("%Y-%m-%d")
    work["_jhome"] = work["home_team"].astype(str).str.strip()
    work["_jaway"] = work["away_team"].astype(str).str.strip()

    merged: pd.DataFrame = work.merge(
        outcomes_j[["_jdate", "_jhome", "_jaway", "HOME_SCORE", "AWAY_SCORE"]],
        on=["_jdate", "_jhome", "_jaway"],
        how="left",
    ).reset_index(drop=True)

    has_outcome: np.ndarray = merged["HOME_SCORE"].notna().values

    settled_mask: pd.Series = is_play & has_line & has_outcome
    pending_mask: pd.Series = is_play & has_line & ~has_outcome

    work.loc[pending_mask, "result"] = "pending"
    work.loc[settled_mask, "home_score"] = merged.loc[settled_mask, "HOME_SCORE"].values
    work.loc[settled_mask, "away_score"] = merged.loc[settled_mask, "AWAY_SCORE"].values

    if settled_mask.any():
        sh: np.ndarray = work.loc[settled_mask, "consensus_spread_home"].astype(float).values
        hs: np.ndarray = merged.loc[settled_mask, "HOME_SCORE"].astype(float).values
        aws: np.ndarray = merged.loc[settled_mask, "AWAY_SCORE"].astype(float).values
        bet: np.ndarray = work.loc[settled_mask, "bet_team"].astype(str).str.strip().values
        home: np.ndarray = work.loc[settled_mask, "home_team"].astype(str).str.strip().values

        is_home_bet: np.ndarray = bet == home
        margin: np.ndarray = np.where(is_home_bet, (hs + sh) - aws, (aws - sh) - hs)

        work.loc[settled_mask, "spread_margin"] = margin

        settled_idx = work.loc[settled_mask].index
        work.loc[settled_idx[margin > 0], "result"] = "win"
        work.loc[settled_idx[margin > 0], "pnl_units"] = WIN_PAYOUT
        work.loc[settled_idx[margin < 0], "result"] = "loss"
        work.loc[settled_idx[margin < 0], "pnl_units"] = -1.0
        work.loc[settled_idx[margin == 0], "result"] = "push"
        work.loc[settled_idx[margin == 0], "pnl_units"] = 0.0

    return work.drop(columns=["_jdate", "_jhome", "_jaway"])


@st.cache_data(ttl=900, show_spinner=False)
def load_todays_plays() -> pd.DataFrame | None:
    """Today's plays CSV. Returns None if Lambda hasn't run yet."""
    today_str: str = str(date.today())
    key: str = f"{PLAYS_PREFIX}/{today_str}.csv"
    if key not in _list_plays_keys():
        return None
    con = _duckdb_s3_conn()
    df: pd.DataFrame = con.execute(f"""
        SELECT * FROM read_csv_auto(
            's3://{S3_BUCKET}/{key}',
            ignore_errors=true
        )
    """).df()
    con.close()
    if df.empty:
        return None
    df["game_date"] = pd.to_datetime(df["game_date"])
    # Score columns don't exist yet (games haven't been played); add as nan so
    # add_derived_columns can build the score string consistently.
    df["home_score"] = float("nan")
    df["away_score"] = float("nan")
    return df
