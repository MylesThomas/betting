"""DuckDB + S3 connection helpers for nba_three_point_modeling."""

from __future__ import annotations

import os

import duckdb


def connect_duckdb_s3() -> duckdb.DuckDBPyConnection:
    """Create a DuckDB connection configured for S3 access in us-east-2."""
    con = duckdb.connect()
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")
    con.execute("SET s3_region='us-east-2'")

    access_key = os.environ["AWS_ACCESS_KEY_ID"]
    secret_key = os.environ["AWS_SECRET_ACCESS_KEY"]
    con.execute(f"SET s3_access_key_id='{access_key}'")
    con.execute(f"SET s3_secret_access_key='{secret_key}'")

    session_token = os.environ.get("AWS_SESSION_TOKEN")
    if session_token:
        con.execute(f"SET s3_session_token='{session_token}'")

    return con

