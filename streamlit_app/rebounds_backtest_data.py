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

import io

import boto3
import numpy as np
import pandas as pd
import streamlit as st

S3_BUCKET: str = "nba-betting-mt"
BACKTEST_PREFIX: str = "data/04_output/backtest/rebounds"
PROD_GO_LIVE_DATE: str = "2026-04-07"
BUCKETS: list[str] = ["both", "ols", "xgb"]


def _s3_client() -> boto3.client:
    return boto3.client("s3")


@st.cache_data(ttl=3600, show_spinner=False)
def load_backtest_multi() -> pd.DataFrame:
    """Load game-level backtest CSV from S3 via boto3."""
    try:
        body = _s3_client().get_object(
            Bucket=S3_BUCKET, Key=f"{BACKTEST_PREFIX}/multi.csv"
        )["Body"].read()
        df = pd.read_csv(io.BytesIO(body))
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
