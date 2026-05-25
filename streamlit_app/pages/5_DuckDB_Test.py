"""
DuckDB S3 Connection Test

Diagnostic page for verifying DuckDB httpfs works on Streamlit Cloud.
Tests against a known S3 file and reports exactly what succeeds or fails.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="DuckDB Test", layout="wide")
st.title("DuckDB S3 Connection Test")
st.caption("Diagnostic page — verifying httpfs + S3 credentials work end-to-end.")

TEST_URI = "s3://ncaab-betting-mt/data/04_output/backtest/fade-revenge-spot/multi.csv"

if st.button("Run test", type="primary"):
    results: list[tuple[str, bool, str]] = []

    # Step 1: install_extension
    with st.spinner("Step 1: installing httpfs extension..."):
        t0 = time.perf_counter()
        try:
            import duckdb
            duckdb.install_extension("httpfs")
            elapsed = (time.perf_counter() - t0) * 1000
            results.append(("install_extension('httpfs')", True, f"{elapsed:.0f} ms"))
        except Exception as e:
            results.append(("install_extension('httpfs')", False, traceback.format_exc()))

    # Step 2: LOAD httpfs
    con = None
    with st.spinner("Step 2: loading httpfs..."):
        t0 = time.perf_counter()
        try:
            con = duckdb.connect()
            con.execute("LOAD httpfs")
            elapsed = (time.perf_counter() - t0) * 1000
            results.append(("LOAD httpfs", True, f"{elapsed:.0f} ms"))
        except Exception:
            results.append(("LOAD httpfs", False, traceback.format_exc()))

    # Step 3: CREATE SECRET
    cred_source = "unknown"
    if con is not None:
        with st.spinner("Step 3: creating S3 secret..."):
            t0 = time.perf_counter()
            try:
                import boto3
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
                elapsed = (time.perf_counter() - t0) * 1000
                results.append((f"CREATE SECRET (creds from {cred_source})", True, f"{elapsed:.0f} ms"))
            except Exception:
                results.append((f"CREATE SECRET (creds from {cred_source})", False, traceback.format_exc()))

    # Step 4: SELECT COUNT(*)
    row_count = None
    if con is not None:
        with st.spinner("Step 4: querying S3..."):
            t0 = time.perf_counter()
            try:
                row_count = con.execute(
                    f"SELECT COUNT(*) FROM read_csv_auto('{TEST_URI}', ignore_errors=true)"
                ).fetchone()[0]
                elapsed = (time.perf_counter() - t0) * 1000
                results.append((f"SELECT COUNT(*) from multi.csv", True, f"{row_count:,} rows · {elapsed:.0f} ms"))
            except Exception:
                results.append(("SELECT COUNT(*) from multi.csv", False, traceback.format_exc()))

        con.close()

    # Render results
    st.markdown("---")
    all_passed = all(ok for _, ok, _ in results)
    if all_passed:
        st.success("All steps passed — DuckDB httpfs is working on this environment.")
    else:
        st.error("One or more steps failed — see details below.")

    for label, ok, detail in results:
        icon = "✅" if ok else "❌"
        st.markdown(f"**{icon} {label}**")
        if not ok:
            st.code(detail, language="text")
        else:
            st.caption(detail)

    st.markdown("---")
    st.markdown(f"**Test URI:** `{TEST_URI}`")
