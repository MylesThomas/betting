"""DuckDB + S3 connection helpers for nba_three_point_modeling."""

from __future__ import annotations

import os
import subprocess

import duckdb


def connect_duckdb_s3() -> duckdb.DuckDBPyConnection:
    """Create a DuckDB connection configured for S3 access in us-east-2."""
    access_key: str
    secret_key: str
    if "AWS_ACCESS_KEY_ID" in os.environ and "AWS_SECRET_ACCESS_KEY" in os.environ:
        access_key = os.environ["AWS_ACCESS_KEY_ID"]
        secret_key = os.environ["AWS_SECRET_ACCESS_KEY"]
    else:
        access_key = subprocess.check_output(
            ["aws", "configure", "get", "aws_access_key_id"], text=True
        ).strip()
        secret_key = subprocess.check_output(
            ["aws", "configure", "get", "aws_secret_access_key"], text=True
        ).strip()
        if access_key == "" or secret_key == "":
            raise ValueError(
                "Missing AWS credentials. Set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY "
                "or configure via `aws configure`."
            )

    con = duckdb.connect()
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")
    con.execute("SET s3_region='us-east-2'")
    con.execute(f"SET s3_access_key_id='{access_key}'")
    con.execute(f"SET s3_secret_access_key='{secret_key}'")

    if "AWS_SESSION_TOKEN" in os.environ:
        con.execute(f"SET s3_session_token='{os.environ['AWS_SESSION_TOKEN']}'")

    return con

