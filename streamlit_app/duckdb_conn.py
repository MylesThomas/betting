"""
Shared DuckDB S3 connection factory.

All data modules should import get_s3_conn() from here rather than
duplicating credential logic.
"""

from __future__ import annotations

import boto3
import duckdb
import streamlit as st


def get_s3_conn() -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection with httpfs loaded and S3 credentials set."""
    con = duckdb.connect()
    con.execute("LOAD httpfs")
    try:
        key_id = st.secrets["AWS_ACCESS_KEY_ID"]
        secret  = st.secrets["AWS_SECRET_ACCESS_KEY"]
        region  = st.secrets.get("AWS_DEFAULT_REGION", "us-east-2")
        cred_source = "st.secrets"
    except Exception:
        session = boto3.Session()
        frozen  = session.get_credentials().get_frozen_credentials()
        key_id  = frozen.access_key
        secret  = frozen.secret_key
        region  = session.region_name or "us-east-2"
        cred_source = "boto3"
    con.execute(f"""
        CREATE SECRET (
            TYPE S3,
            KEY_ID '{key_id}',
            SECRET '{secret}',
            REGION '{region}'
        )
    """)
    return con
