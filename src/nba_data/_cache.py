"""S3 cache read/write for the unified NBA data loader."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.io_utils import (
    read_json_any,
    read_parquet_any,
    uri_exists,
    write_json_any,
    write_parquet_any,
)

_BASE = "s3://nba-betting-mt/data/02_cache/nba_unified"
MANIFEST_URI = f"{_BASE}/manifest.json"
LOGS_URI = f"{_BASE}/player_game_logs.parquet"
PROPS_URI = f"{_BASE}/player_props.parquet"
LINES_URI = f"{_BASE}/game_lines.parquet"


def cache_exists() -> bool:
    return uri_exists(MANIFEST_URI)


def read_manifest() -> dict | None:
    if not cache_exists():
        return None
    try:
        return read_json_any(MANIFEST_URI)
    except Exception:
        return None


def read_cache():
    from .get_data import NBAData
    meta = read_manifest() or {}
    logs = read_parquet_any(LOGS_URI)
    props = read_parquet_any(PROPS_URI)
    lines = read_parquet_any(LINES_URI)
    # Restore date types lost in parquet round-trip
    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"]).dt.date
    props["game_date"] = pd.to_datetime(props["game_date"]).dt.date
    lines["game_date"] = pd.to_datetime(lines["game_date"]).dt.date
    return NBAData(logs=logs, props=props, lines=lines, meta=meta)


def write_cache(logs: pd.DataFrame, props: pd.DataFrame, lines: pd.DataFrame) -> dict:
    write_parquet_any(logs, LOGS_URI)
    write_parquet_any(props, PROPS_URI)
    write_parquet_any(lines, LINES_URI)
    meta = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "seasons": sorted(logs["season"].dropna().unique().tolist()),
        "max_game_date": str(logs["GAME_DATE"].max()),
        "max_prop_date": str(props["game_date"].max()),
        "row_counts": {
            "player_game_logs": len(logs),
            "player_props": len(props),
            "game_lines": len(lines),
        },
    }
    write_json_any(meta, MANIFEST_URI)
    print(
        f"[nba_data] cache written — "
        f"{meta['row_counts']['player_game_logs']:,} logs, "
        f"{meta['row_counts']['player_props']:,} props, "
        f"{meta['row_counts']['game_lines']:,} lines "
        f"(updated_at {meta['updated_at']})"
    )
    return meta
