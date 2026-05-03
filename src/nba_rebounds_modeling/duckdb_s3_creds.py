"""
Configure DuckDB httpfs for S3 using the same credential resolution as boto3
(CLI default profile, SSO, assume-role, long-lived env keys).

DuckDB must receive access key, secret, and session token (when using temporary
creds) or reads against ``s3://...`` can fail with 403. `aws configure get` does
not return SSO/role credentials; using boto3 fixes local runs.
"""

from __future__ import annotations

import os
from typing import Optional

import duckdb


def _sql_set_string(value: str) -> str:
    return value.replace("'", "''")


def configure_duckdb_httpfs_s3(
    con: duckdb.DuckDBPyConnection,
    *,
    region: str = "us-east-2",
) -> None:
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute(f"SET s3_region='{region}'")

    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    token: Optional[str] = None

    def try_boto3() -> bool:
        nonlocal access_key, secret_key, token
        try:
            import boto3
            from botocore.exceptions import BotoCoreError, NoCredentialsError
        except ImportError:
            return False
        try:
            c = boto3.session.Session().get_credentials()
        except (BotoCoreError, NoCredentialsError, Exception):
            return False
        if c is None:
            return False
        frozen = c.get_frozen_credentials()
        if not (frozen and frozen.access_key and frozen.secret_key):
            return False
        access_key, secret_key, token = (
            str(frozen.access_key),
            str(frozen.secret_key),
            str(frozen.token) if frozen.token else None,
        )
        return True

    if not try_boto3():
        if "AWS_ACCESS_KEY_ID" in os.environ and "AWS_SECRET_ACCESS_KEY" in os.environ:
            access_key = os.environ["AWS_ACCESS_KEY_ID"]
            secret_key = os.environ["AWS_SECRET_ACCESS_KEY"]
            token = os.environ.get("AWS_SESSION_TOKEN")
        else:
            import subprocess

            access_key = subprocess.check_output(
                ["aws", "configure", "get", "aws_access_key_id"], text=True
            ).strip()
            secret_key = subprocess.check_output(
                ["aws", "configure", "get", "aws_secret_access_key"], text=True
            ).strip()
            if not access_key or not secret_key:
                raise ValueError(
                    "No AWS credentials for DuckDB S3. Run `aws sso login`, set "
                    "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY, or use a profile "
                    "boto3 can resolve."
                )
            if "AWS_SESSION_TOKEN" in os.environ:
                token = os.environ["AWS_SESSION_TOKEN"]

    assert access_key and secret_key
    con.execute(f"SET s3_access_key_id='{_sql_set_string(access_key)}'")
    con.execute(f"SET s3_secret_access_key='{_sql_set_string(secret_key)}'")
    if token:
        con.execute(f"SET s3_session_token='{_sql_set_string(token)}'")
    # Limit parallel threads to avoid exhausting S3 HTTP connections when
    # globbing over many objects (e.g. 163 CSVs).
    con.execute("SET threads=1")


def connect_duckdb_s3(*, region: str = "us-east-2") -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    configure_duckdb_httpfs_s3(con, region=region)
    return con
