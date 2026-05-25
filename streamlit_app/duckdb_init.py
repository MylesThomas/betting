"""
Pre-installs the DuckDB httpfs extension at app startup.

Imported at the top of app.py so this runs once on every cold start.
install_extension() is a no-op if the extension is already cached.
"""

import duckdb

duckdb.install_extension("httpfs")
