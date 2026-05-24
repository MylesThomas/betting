"""
Data loading for the NCAAB Away Revenge backtest dashboard.

S3 layout:
  Multi (game-level): s3://ncaab-betting-mt/data/04_output/backtest/fade-revenge-spot/multi.csv

No st.* rendering. No try/except. No defensive column checks.
"""

from __future__ import annotations

import boto3
import duckdb
import numpy as np
import pandas as pd
import streamlit as st

S3_BUCKET: str = "ncaab-betting-mt"
BACKTEST_PREFIX: str = "data/04_output/backtest/fade-revenge-spot"
WIN_PAYOUT: float = 100.0 / 110.0  # -110 juice
PROD_GO_LIVE_DATE: str = "2026-02-19"
SEASON_ORDER: list[str] = [
    "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"
]


def _duckdb_s3_conn() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")
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
    """
    Load game-level backtest CSV from S3 and return enriched DataFrame.

    Adds: game_date, matchup, focal_spread, focal_ats_margin, pnl_units, result.
    """
    con = _duckdb_s3_conn()
    df: pd.DataFrame = con.execute(f"""
        SELECT * FROM read_csv_auto(
            's3://{S3_BUCKET}/{BACKTEST_PREFIX}/multi.csv',
            ignore_errors=true
        )
    """).df()
    con.close()

    df["game_date"] = pd.to_datetime(df["GAME_DATE"])
    df["season"] = df["season"].astype(str)
    df["matchup"] = df["AWAY_TEAM"].astype(str) + " @ " + df["HOME_TEAM"].astype(str)

    home_bet: np.ndarray = df["focal_was_home"].fillna(False).astype(bool).values
    spread: np.ndarray = pd.to_numeric(df["consensus_spread"], errors="coerce").values

    # Spread shown from focal team's perspective (negative = favored)
    df["focal_spread"] = np.where(home_bet, spread, -spread)

    hs: np.ndarray = pd.to_numeric(df["HOME_SCORE"], errors="coerce").values
    aws: np.ndarray = pd.to_numeric(df["AWAY_SCORE"], errors="coerce").values

    # ATS margin: positive means covered, negative means didn't cover
    df["focal_ats_margin"] = np.where(
        home_bet, (hs - aws) + spread, (aws - hs) - spread
    )

    cover: pd.Series = df["focal_ats_cover"]
    df["pnl_units"] = np.nan
    df.loc[cover.eq(True), "pnl_units"] = WIN_PAYOUT
    df.loc[cover.eq(False), "pnl_units"] = -1.0

    df["result"] = "no_line"
    df.loc[cover.eq(True), "result"] = "win"
    df.loc[cover.eq(False), "result"] = "loss"

    return df


def filter_by_side(df: pd.DataFrame, side: str) -> pd.DataFrame:
    """Filter rows by focal team's home/away side. side: 'Away' | 'Home' | 'All'."""
    if side == "Away":
        return df[df["focal_was_home"].eq(False)].reset_index(drop=True)
    if side == "Home":
        return df[df["focal_was_home"].eq(True)].reset_index(drop=True)
    return df.reset_index(drop=True)


def settled_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["result"].isin({"win", "loss"})].reset_index(drop=True)


def compute_kpis(settled: pd.DataFrame) -> dict[str, float | int]:
    non_push: pd.DataFrame = settled[settled["result"].isin({"win", "loss"})]
    wins: int = int((settled["result"] == "win").sum())
    losses: int = int((settled["result"] == "loss").sum())
    total: int = len(settled)
    pnl: float = float(settled["pnl_units"].sum())
    ats_pct: float = wins / len(non_push) if len(non_push) > 0 else float("nan")
    roi: float = pnl / total if total > 0 else float("nan")
    return {
        "total_bets": total,
        "wins": wins,
        "losses": losses,
        "total_pnl": pnl,
        "ats_pct": ats_pct,
        "roi_per_bet": roi,
    }


def per_season_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per season from settled game-level rows."""
    records: list[dict] = []
    for season in SEASON_ORDER:
        sub: pd.DataFrame = settled_rows(df[df["season"] == season])
        if sub.empty:
            continue
        kpis = compute_kpis(sub)
        records.append({
            "Season": season,
            "Bets": kpis["total_bets"],
            "W": kpis["wins"],
            "L": kpis["losses"],
            "ATS %": kpis["ats_pct"],
            "P&L": kpis["total_pnl"],
            "ROI/Bet": kpis["roi_per_bet"],
        })
    return pd.DataFrame(records)
