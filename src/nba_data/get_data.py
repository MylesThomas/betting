"""Unified NBA data loader with S3 parquet cache."""
from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.nba_data._types import NBAData
from src.nba_data._cache import (
    cache_exists,
    read_cache,
    read_manifest,
    write_cache,
    LOGS_URI,
    PROPS_URI,
    LINES_URI,
)
from src.nba_data._loaders import load_logs_raw, load_props_raw, load_lines_raw
from src.io_utils import read_parquet_any

ALL_SEASONS = ["2023-24", "2024-25", "2025-26"]


def _current_season() -> str:
    today = date.today()
    year, month = today.year, today.month
    if month >= 10:
        return f"{year}-{str(year + 1)[2:]}"
    return f"{year - 1}-{str(year)[2:]}"


def _rebuild(seasons: list[str]) -> NBAData:
    print(f"[nba_data] rebuilding cache for seasons {seasons} ...")
    logs = load_logs_raw(seasons)
    props = load_props_raw(seasons)
    lines = load_lines_raw(seasons)
    meta = write_cache(logs, props, lines)
    return NBAData(logs=logs, props=props, lines=lines, meta=meta)


def _refresh(seasons: list[str]) -> NBAData:
    manifest = read_manifest()
    if manifest is None:
        return _rebuild(seasons)

    max_date = manifest.get("max_game_date")
    active = _current_season()

    if active not in seasons:
        # No active season in scope — nothing to append
        print("[nba_data] refresh: no active season in requested scope, returning cache")
        return _apply_filters(read_cache(), seasons, None)

    print(f"[nba_data] refresh: checking {active} for games after {max_date} ...")

    new_logs = load_logs_raw([active])
    new_props = load_props_raw([active])
    new_lines = load_lines_raw([active])

    if max_date:
        cutoff = pd.to_datetime(max_date).date()
        new_logs = new_logs[new_logs["GAME_DATE"] > cutoff]
        new_props = new_props[new_props["game_date"] > cutoff]
        new_lines = new_lines[new_lines["game_date"] > cutoff]

    if new_logs.empty:
        print("[nba_data] refresh: cache is already up to date")
        return _apply_filters(read_cache(), seasons, None)

    print(f"[nba_data] refresh: appending {len(new_logs):,} new log rows")
    cached = read_cache()
    combined_logs = pd.concat([cached.logs, new_logs], ignore_index=True).drop_duplicates(
        subset=["GAME_ID", "PLAYER_ID"]
    )
    combined_props = pd.concat([cached.props, new_props], ignore_index=True).drop_duplicates()
    combined_lines = pd.concat([cached.lines, new_lines], ignore_index=True).drop_duplicates()
    meta = write_cache(combined_logs, combined_props, combined_lines)
    full = NBAData(logs=combined_logs, props=combined_props, lines=combined_lines, meta=meta)
    return _apply_filters(full, seasons, None)


def _apply_filters(data: NBAData, seasons: list[str] | None, min_minutes: float | None) -> NBAData:
    logs, props, lines = data.logs, data.props, data.lines

    if seasons:
        logs = logs[logs["season"].isin(seasons)]
        props = props[props["season"].isin(seasons)]
        lines = lines[lines["season"].isin(seasons)]

    if min_minutes is not None:
        logs = logs[logs["MIN"].astype(float) >= min_minutes]

    return NBAData(logs=logs, props=props, lines=lines, meta=data.meta)


def get_data(
    seasons: list[str] | None = None,
    min_minutes: float | None = None,
    refresh: bool = False,
    rebuild: bool = False,
) -> NBAData:
    """
    Load NBA player game logs, player props, and game lines.

    Args:
        seasons: Seasons to include, e.g. ["2023-24", "2024-25"]. Defaults to all 3.
        min_minutes: If set, filter logs to rows where MIN >= this value.
        refresh: Check S3 for new games since last cache write and append them.
        rebuild: Full reload from raw S3 CSVs, overwrites cache.

    Returns:
        NBAData with .logs, .props, .lines, .meta
    """
    _seasons = seasons or ALL_SEASONS

    if rebuild:
        data = _rebuild(_seasons)
        return _apply_filters(data, seasons, min_minutes)

    if refresh:
        return _apply_filters(_refresh(_seasons), seasons, min_minutes)

    if not cache_exists():
        print("[nba_data] no cache found, building from scratch ...")
        data = _rebuild(_seasons)
        return _apply_filters(data, seasons, min_minutes)

    print("[nba_data] loading from cache ...")
    data = read_cache()
    return _apply_filters(data, seasons, min_minutes)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NBA unified data loader CLI")
    parser.add_argument("command", choices=["rebuild", "refresh", "status"])
    args = parser.parse_args()

    if args.command == "rebuild":
        get_data(rebuild=True)
    elif args.command == "refresh":
        get_data(refresh=True)
    elif args.command == "status":
        from src.nba_data._cache import read_manifest
        import json
        m = read_manifest()
        print(json.dumps(m, indent=2) if m else "No cache found.")
