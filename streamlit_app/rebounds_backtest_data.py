"""
Data loading for the NBA Rebounds backtest dashboard.

S3: s3://nba-betting-mt/data/04_output/backtest/rebounds/multi.csv

strategy_bucket is MECE: "ols" | "xgb" | "both"
  - both: OLS and XGB both fired on that game (uses OLS row)
  - ols:  only OLS fired
  - xgb:  only XGB fired

No st.* rendering.
"""

from __future__ import annotations

import boto3
import duckdb
import numpy as np
import pandas as pd
import streamlit as st

S3_BUCKET: str = "nba-betting-mt"
BACKTEST_PREFIX: str = "data/04_output/backtest/rebounds"
PROD_GO_LIVE_DATE: str = "2026-04-07"
BUCKETS: list[str] = ["both", "ols", "xgb"]


def _duckdb_s3_conn() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    try:
        key_id = st.secrets["AWS_ACCESS_KEY_ID"]
        secret  = st.secrets["AWS_SECRET_ACCESS_KEY"]
        region  = st.secrets.get("AWS_DEFAULT_REGION", "us-east-2")
    except Exception:
        session = boto3.Session()
        frozen  = session.get_credentials().get_frozen_credentials()
        key_id  = frozen.access_key
        secret  = frozen.secret_key
        region  = session.region_name or "us-east-2"
    con.execute(f"""
        CREATE SECRET (
            TYPE S3,
            KEY_ID '{key_id}',
            SECRET '{secret}',
            REGION '{region}'
        )
    """)
    return con


@st.cache_data(ttl=3600, show_spinner=False)
def load_backtest_multi() -> pd.DataFrame:
    """Load game-level backtest CSV from S3. Returns empty DataFrame if not yet seeded."""
    try:
        con = _duckdb_s3_conn()
        df: pd.DataFrame = con.execute(f"""
            SELECT * FROM read_csv_auto(
                's3://{S3_BUCKET}/{BACKTEST_PREFIX}/multi.csv',
                ignore_errors=true
            )
        """).df()
        con.close()
    except Exception:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    df["season"] = df["season"].astype(str)
    df["strategy_bucket"] = df["strategy_bucket"].astype(str)
    return df


def filter_by_buckets(df: pd.DataFrame, selected_buckets: list[str]) -> pd.DataFrame:
    return df[df["strategy_bucket"].isin(selected_buckets)].reset_index(drop=True)


def settled_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["result"].isin({"win", "loss"})].reset_index(drop=True)


def compute_kpis(settled: pd.DataFrame) -> dict[str, float | int]:
    wins: int = int((settled["result"] == "win").sum())
    losses: int = int((settled["result"] == "loss").sum())
    total: int = len(settled)
    pnl: float = float(settled["pnl_units"].sum())
    hit_rate: float = wins / (wins + losses) if (wins + losses) > 0 else float("nan")
    roi: float = pnl / total if total > 0 else float("nan")
    return {
        "total_bets": total,
        "wins": wins,
        "losses": losses,
        "total_pnl": pnl,
        "hit_rate": hit_rate,
        "roi_per_bet": roi,
    }


def per_season_summary(df: pd.DataFrame) -> pd.DataFrame:
    s = settled_rows(df)
    if s.empty:
        return pd.DataFrame()
    rows = (
        s.groupby(["season", "strategy_bucket"], as_index=False)
        .agg(
            Bets   =("pnl_units", "count"),
            Wins   =("result",    lambda x: (x == "win").sum()),
            Losses =("result",    lambda x: (x == "loss").sum()),
            PnL    =("pnl_units", "sum"),
        )
    )
    rows["Hit Rate"] = rows["Wins"] / (rows["Wins"] + rows["Losses"])
    rows["ROI/Bet"]  = rows["PnL"] / rows["Bets"]
    rows = rows.rename(columns={"season": "Season", "strategy_bucket": "Bucket"})
    return rows.sort_values(["Season", "Bucket"], ascending=[False, True]).reset_index(drop=True)
