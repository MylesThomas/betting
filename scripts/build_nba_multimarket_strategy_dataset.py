"""
Build a unified NBA dataset for top-down strategy discovery and backtesting.

Goal: Find profitable betting strategies by seeing which variables matter most
per market and deriving backtestable signals.

Joins:
- Left: Game results (ESPN/NBA API) from s3://nba-betting-mt/data/01_input/historical_game_results/
- Right 1: Pregame player prop lines (all markets) from s3://the-odds-api-mt/nba/historical_player_props/{season}/{date}.csv
- Right 2: Pregame team game lines (spread, ML) from s3://the-odds-api-mt/nba/historical_game_lines/{season}/
Uses src/player_team_history (player_team_history.parquet) to get each player's team
at that date so props can be joined to the correct game and game lines. Team names
are normalized via src/player_team_history/team_normalization (Odds API → ESPN/NBA
canonical, e.g. LA Clippers → Los Angeles Clippers) so joins match.

Also joins player game logs (s3://nba-api-mt/player_game_logs/{season}/) to attach
actual stats (PTS, REB, AST, etc.) for each player-game so you can compare line vs actual.

Output: Single .parquet in ~/Downloads/tmp (default: nba_strategy_3seasons.parquet). One row per player-game: for each of 9 markets, market_median_value_{market} (median line, can be null) and actual_{stat} (from game results; null only if no game data).

Why so many nulls? Rows with GAME_ID null are props for games that did not match game_results
(often future games: props are posted before the game, so game_date can be ahead of when we have
results/logs). For those rows we have no home_team, away_team, game lines, or actuals. actual_*
are also null whenever we don't have that player's game log for that date (e.g. game not played yet).
To inspect: filter WHERE "GAME_ID" IS NOT NULL for backtests; or use:
  SELECT "GAME_ID" IS NOT NULL AS matched, game_date, count(*) FROM parquet GROUP BY 1, 2 ORDER BY 2 DESC LIMIT 20;

Uses 3 seasons by default (2023-24, 2024-25, 2025-26). Player names normalized via
src/player_team_history/name_normalization (Odds API for props, NBA API for game logs).

Usage:
    python scripts/build_nba_multimarket_strategy_dataset.py
    python scripts/build_nba_multimarket_strategy_dataset.py --seasons 2024-25 2025-26
    python scripts/build_nba_multimarket_strategy_dataset.py --output ~/Downloads/tmp/nba_strategy.parquet
    python scripts/build_nba_multimarket_strategy_dataset.py   # single parquet; filter GAME_ID.isna() for non-joins

When name_normalization changes (e.g. Jr/Sr exceptions in src/player_team_history/name_normalization.py):
    1. Rebuild player team history so history.parquet has correct player_normalized names:
           python src/player_team_history/01_build.py --no-cache
       (Or clear cache for affected players via src/player_team_history/03_cache.py --clear "Jabari Smith" "Gary Trent" ...)
    2. Upload the new history.parquet to S3 (nba-betting-mt, nba/player_team_history/history.parquet) if that is your source.
    3. Delete the strategy build caches for player_team_history so this script re-fetches from S3.
       From repo root, or use absolute paths so it works from any cwd:
           rm -f data/02_cache/nba_strategy_build/player_team_history.parquet
           rm -f ~/Downloads/tmp/nba_strategy_build/player_team_history.parquet
       Or run this script with --no-cache once to bypass all caches and re-download from S3.
"""

import argparse
import functools
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import boto3
import pandas as pd
import yaml
from io import BytesIO, StringIO

# Repo root (find via .gitignore per workspace rules)
def _repo_root():
    p = Path(__file__).resolve().parent.parent
    if (p / ".gitignore").exists():
        return p
    raise RuntimeError("Repo root not found (no .gitignore)")

REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.player_team_history.utils import load_team_history
from src.player_team_history.team_normalization import normalize_team_name_from_odds_api
from src.team_utils import NBA_TEAMS

# Exactly 30 NBA teams; canonical full names for assert
NBA_TEAMS_COUNT = 30
CANONICAL_TEAM_NAMES = set(NBA_TEAMS.values())

# S3
S3_BETTING = "nba-betting-mt"
S3_ODDS = "the-odds-api-mt"
S3_NBA_API = "nba-api-mt"
HISTORY_S3_KEY = "nba/player_team_history/history.parquet"  # canonical (trades, etc.); also at data/02_cache/ for legacy
GAME_RESULTS_PREFIX = "data/01_input/historical_game_results/"

OUTPUT_DIR = Path.home() / "Downloads" / "tmp"

# Cache for each dataset so re-runs skip already-fetched data (e.g. game results ok, props failed → only re-fetch props)
CACHE_DIR = REPO_ROOT / "data" / "02_cache" / "nba_strategy_build"

# Default: 3 seasons for strategy discovery and backtesting
DEFAULT_SEASONS = ["2023-24", "2024-25", "2025-26"]

# Retry S3 reads on transient failures (timeout, connection reset)
S3_GET_RETRIES = 3
S3_GET_RETRY_DELAY_SEC = 2


def _s3_read_csv_with_retry(s3, bucket: str, key: str):
    """Get S3 object body as decoded UTF-8 string; retry on timeout/connection errors."""
    last_err = None
    for attempt in range(S3_GET_RETRIES):
        try:
            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
            return body
        except Exception as e:
            last_err = e
            # Retry on typical transient errors (read timeout, connection reset)
            if attempt < S3_GET_RETRIES - 1 and (
                "timeout" in str(e).lower()
                or "Connection reset" in str(e)
                or "Connection broken" in str(e)
            ):
                time.sleep(S3_GET_RETRY_DELAY_SEC * (attempt + 1))
                continue
            raise
    raise last_err


def timed(f):
    """Decorator: print function name and elapsed seconds after each call."""
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = f(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        print(f"   ⏱  {f.__name__}: {elapsed:.2f}s")
        return result
    return wrapper


def _seasons_cache_key(seasons: list[str]) -> str:
    """Stable key for cache filenames from season list."""
    return "_".join(sorted(seasons))


def _assert_30_teams(
    df: pd.DataFrame,
    source_name: str,
    columns: list[str],
    *,
    use_abbr: bool = False,
) -> None:
    """Assert that the union of unique team values across columns has exactly 30 NBA teams (after normalization if full names)."""
    if df.empty:
        return
    uniq = set()
    for col in columns:
        if col not in df.columns:
            continue
        vals = df[col].dropna().unique()
        if use_abbr:
            uniq.update(str(v) for v in vals)
        else:
            uniq.update(normalize_team_name_from_odds_api(str(v)) for v in vals)
    if use_abbr:
        extra = uniq - set(NBA_TEAMS.keys())
        missing = set(NBA_TEAMS.keys()) - uniq
        if extra or missing or len(uniq) != NBA_TEAMS_COUNT:
            print(f"DEBUG {source_name}: len(uniq)={len(uniq)}, extra abbrs={sorted(extra)}, missing abbrs={sorted(missing)}")
        assert len(uniq) == NBA_TEAMS_COUNT and uniq == set(NBA_TEAMS.keys()), (
            f"{source_name}: expected 30 team abbrs, got {len(uniq)}: extra={uniq - set(NBA_TEAMS.keys())} missing={set(NBA_TEAMS.keys()) - uniq}"
        )
    else:
        extra = uniq - CANONICAL_TEAM_NAMES
        missing = CANONICAL_TEAM_NAMES - uniq
        if extra or missing or len(uniq) != NBA_TEAMS_COUNT:
            print(f"DEBUG {source_name}: len(uniq)={len(uniq)}")
            print(f"  extra (not in canonical 30): {sorted(extra)}")
            print(f"  missing (canonical not in data): {sorted(missing)}")
        assert len(uniq) == NBA_TEAMS_COUNT and uniq == CANONICAL_TEAM_NAMES, (
            f"{source_name}: expected 30 canonical team names, got {len(uniq)}"
        )


def _cache_filename(cache_name: str, seasons_key: str | None) -> str:
    if seasons_key:
        return f"{cache_name}_{seasons_key}.parquet"
    return f"{cache_name}.parquet"


def _load_with_cache(
    cache_name: str,
    loader,
    loader_args: tuple,
    use_cache: bool,
    seasons_key: str | None = None,
    output_cache_dir: Path | None = None,
) -> pd.DataFrame:
    """Load from cache (output dir first, then CACHE_DIR) if present; else run loader and write cache to both CACHE_DIR and output_cache_dir."""
    fname = _cache_filename(cache_name, seasons_key)
    # Prefer output dir cache, then repo CACHE_DIR
    read_candidates = []
    if output_cache_dir is not None:
        read_candidates.append(output_cache_dir / fname)
    read_candidates.append(CACHE_DIR / fname)

    if use_cache:
        for path in read_candidates:
            if path.exists():
                df = pd.read_parquet(path)
                if cache_name == "player_team_history" and not df.empty:
                    df["valid_from"] = pd.to_datetime(df["valid_from"]).dt.date
                    df["valid_to"] = pd.to_datetime(df["valid_to"], errors="coerce").dt.date
                print(f"✅ {cache_name}: loaded from cache ({len(df):,} rows)")
                return df

    df = loader(*loader_args)
    if use_cache and not df.empty:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path_repo = CACHE_DIR / fname
        df.to_parquet(cache_path_repo, index=False)
        if output_cache_dir is not None:
            output_cache_dir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(output_cache_dir / fname, index=False)
        print(f"   💾 Cached {cache_name} to {fname} (CACHE_DIR + output dir)")
    return df


def _season_date_range(season: str) -> tuple[str, str]:
    with open(REPO_ROOT / "config" / "season_dates.yaml") as f:
        data = yaml.safe_load(f)
    nba = data["nba"][season]
    return nba["season_start"], nba["playoff_end"]


def _season_for_game_date(seasons: list[str], date_str: str) -> str | None:
    """Return which of the given seasons contains this game_date (by season_start..playoff_end). Used so props season matches game_results."""
    try:
        d = pd.to_datetime(date_str).date()
    except Exception:
        return None
    for season in seasons:
        start, end = _season_date_range(season)
        if pd.to_datetime(start).date() <= d <= pd.to_datetime(end).date():
            return season
    return None


def _normalize_team_for_join(name):
    """Normalize team name to ESPN/NBA canonical form (e.g. LA Clippers → Los Angeles Clippers) so joins match."""
    if pd.isna(name):
        return name
    return normalize_team_name_from_odds_api(name)


def _normalize_team_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """In-place: normalize all listed columns that exist using _normalize_team_for_join (e.g. LA Clippers → Los Angeles Clippers)."""
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(_normalize_team_for_join)


@timed
def load_game_results(seasons: list[str], max_files: int | None = None) -> pd.DataFrame:
    """Load ESPN game results for all given seasons from S3; filter by season date range. If max_files set, stop after that many files (dry run)."""
    all_dfs = []
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    n_loaded = 0
    for season in seasons:
        start, end = _season_date_range(season)
        start_dt = pd.to_datetime(start).date()
        end_dt = pd.to_datetime(end).date()
        for page in paginator.paginate(Bucket=S3_BETTING, Prefix=GAME_RESULTS_PREFIX):
            for obj in page.get("Contents", []):
                if max_files is not None and n_loaded >= max_files:
                    break
                key = obj["Key"]
                if not key.endswith(".csv"):
                    continue
                fn = key.split("/")[-1]
                try:
                    date_str = fn.replace(".csv", "")
                    file_date = pd.to_datetime(date_str).date()
                except Exception:
                    continue
                if file_date < start_dt or file_date > end_dt:
                    continue
                try:
                    body = s3.get_object(Bucket=S3_BETTING, Key=key)["Body"].read()
                    df = pd.read_csv(BytesIO(body))
                    df["game_date"] = pd.to_datetime(df["GAME_DATE"]).dt.date.astype(str)
                    df["home_team"] = df["HOME_TEAM"].apply(_normalize_team_for_join)
                    df["away_team"] = df["AWAY_TEAM"].apply(_normalize_team_for_join)
                    df["season"] = season
                    all_dfs.append(df)
                    n_loaded += 1
                except Exception as e:
                    print(f"  ⚠️  Skip {key}: {e}")
            if max_files is not None and n_loaded >= max_files:
                break
        if max_files is not None and n_loaded >= max_files:
            break
    if not all_dfs:
        return pd.DataFrame()
    non_empty = [df for df in all_dfs if not df.empty]
    if not non_empty:
        return pd.DataFrame()
    out = pd.concat(non_empty, ignore_index=True)
    print(f"✅ Game results: {len(out)} games, seasons {seasons}, {out['game_date'].min()} to {out['game_date'].max()}")
    return out


@timed
def load_player_props(seasons: list[str], max_files: int | None = None) -> pd.DataFrame:
    """Load all player props (all markets) from S3; add game_date from filename, player_normalized. If max_files set, stop after that many files (dry run)."""
    from src.player_team_history.name_normalization import normalize_from_odds_api
    all_dfs = []
    s3 = boto3.client("s3")
    n_loaded = 0
    for season in seasons:
        prefix = f"nba/historical_player_props/{season}/"
        resp = s3.list_objects_v2(Bucket=S3_ODDS, Prefix=prefix)
        if "Contents" not in resp:
            continue
        for obj in resp["Contents"]:
            if max_files is not None and n_loaded >= max_files:
                break
            key = obj["Key"]
            if not key.endswith(".csv"):
                continue
            date_str = key.split("/")[-1].replace(".csv", "")
            try:
                body = _s3_read_csv_with_retry(s3, S3_ODDS, key)
                df = pd.read_csv(StringIO(body))
                df["game_date"] = date_str
                # Align with ESPN/game results: derive game_date from game_time (ET) when present
                # so late-night UTC games (e.g. 03:00 UTC = 22:00 ET prev day) join correctly.
                if "game_time" in df.columns:
                    gt = pd.to_datetime(df["game_time"], utc=True)
                    if gt.dt.tz is None:
                        gt = gt.dt.tz_localize("UTC")
                    game_date_et = gt.dt.tz_convert(ZoneInfo("America/New_York")).dt.date.astype(str)
                    df["game_date"] = game_date_et
                df["season"] = season
                if "player" not in df.columns:
                    continue
                df["player_normalized"] = df["player"].apply(normalize_from_odds_api)
                all_dfs.append(df)
                n_loaded += 1
            except Exception as e:
                print(f"  ⚠️  Skip {key}: {e}")
        if max_files is not None and n_loaded >= max_files:
            break
    if not all_dfs:
        return pd.DataFrame()
    non_empty = [df for df in all_dfs if not df.empty]
    if not non_empty:
        return pd.DataFrame()
    out = pd.concat(non_empty, ignore_index=True)
    # Derive season from game_date so props match game_results (same date -> same season). S3 folder season can be wrong (e.g. 2023-11-21 in 2025-26 folder).
    derived_season = out["game_date"].apply(lambda d: _season_for_game_date(seasons, d))
    out["season"] = derived_season.fillna(out["season"])
    print(f"✅ Player props: {len(out):,} rows, seasons {seasons}, markets: {out['market'].unique().tolist()[:10]}...")
    return out


@timed
def load_player_team_history() -> pd.DataFrame:
    """Load player_team_history from S3 (nba/player_team_history/history.parquet); valid_from/valid_to as date."""
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=S3_BETTING, Key=HISTORY_S3_KEY)
    df = pd.read_parquet(BytesIO(obj["Body"].read()))
    df["valid_from"] = pd.to_datetime(df["valid_from"]).dt.date
    df["valid_to"] = pd.to_datetime(df["valid_to"], errors="coerce").dt.date
    return df


@timed
def add_team_to_props(props_df: pd.DataFrame, history_df: pd.DataFrame) -> pd.DataFrame:
    """Add team_abbr and team_full to props using player_team_history (vectorized)."""
    props_df = props_df.copy()
    props_df["game_date_dt"] = pd.to_datetime(props_df["game_date"]).dt.date
    # Merge on player; then filter to valid range
    merged = props_df.merge(
        history_df[["player_normalized", "team", "valid_from", "valid_to"]],
        on="player_normalized",
        how="left",
    )
    # Keep row where valid_from <= game_date and (valid_to is null or valid_to >= game_date)
    in_range = (merged["game_date_dt"] >= merged["valid_from"]) & (
        merged["valid_to"].isna() | (merged["game_date_dt"] <= merged["valid_to"])
    )
    merged = merged.loc[in_range].copy()
    merged["team_abbr"] = merged["team"]
    merged["team_full"] = merged["team"].map(NBA_TEAMS)
    # Normalize to ESPN/NBA canonical so joins with game results and game lines match (e.g. LA Clippers → Los Angeles Clippers)
    merged["team_full"] = merged["team_full"].apply(_normalize_team_for_join)
    # If multiple stints (shouldn't happen), take first
    merged = merged.drop_duplicates(
        subset=[c for c in props_df.columns if c in merged.columns] + ["game_date"],
        keep="first",
    )
    merged = merged.drop(columns=["valid_from", "valid_to", "game_date_dt", "team"], errors="ignore")
    print(f"   Props with team resolved: {merged['team_full'].notna().sum():,} / {len(merged):,}")
    return merged


@timed
def load_game_lines(seasons: list[str], max_files: int | None = None) -> pd.DataFrame:
    """Load game lines (spread, moneyline) from S3. If max_files set, stop after that many files (dry run)."""
    all_dfs = []
    s3 = boto3.client("s3")
    n_loaded = 0
    for season in seasons:
        prefix = f"nba/historical_game_lines/{season}/"
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=S3_ODDS, Prefix=prefix):
            for obj in page.get("Contents", []):
                if max_files is not None and n_loaded >= max_files:
                    break
                key = obj["Key"]
                if not key.endswith(".csv") or "failed" in key.lower():
                    continue
                fn = key.split("/")[-1]
                if "nba_game_lines_" in fn:
                    date_str = fn.replace("nba_game_lines_", "").replace(".csv", "")
                else:
                    date_str = fn.replace(".csv", "")
                try:
                    body = _s3_read_csv_with_retry(s3, S3_ODDS, key)
                    df = pd.read_csv(StringIO(body))
                    df["game_date"] = date_str
                    df["season"] = season
                    all_dfs.append(df)
                    n_loaded += 1
                except Exception as e:
                    print(f"  ⚠️  Skip {key}: {e}")
            if max_files is not None and n_loaded >= max_files:
                break
        if max_files is not None and n_loaded >= max_files:
            break
    if not all_dfs:
        return pd.DataFrame()
    non_empty = [df for df in all_dfs if not df.empty]
    if not non_empty:
        return pd.DataFrame()
    raw = pd.concat(non_empty, ignore_index=True)
    # Consensus by game/market (per season we keep game_date, away_team, home_team; season in raw but drop in groupby then re-merge if needed)
    if "market" not in raw.columns:
        return raw
    spread = raw[raw["market"] == "spread"].groupby(["game_date", "away_team", "home_team", "season"]).agg(
        {"away_line": "mean", "home_line": "mean", "away_odds": "mean", "home_odds": "mean"}
    ).reset_index()
    spread.columns = ["game_date", "away_team", "home_team", "season", "away_spread", "home_spread", "away_spread_odds", "home_spread_odds"]
    ml = raw[raw["market"] == "moneyline"].groupby(["game_date", "away_team", "home_team", "season"]).agg(
        {"away_odds": "mean", "home_odds": "mean"}
    ).reset_index()
    ml.columns = ["game_date", "away_team", "home_team", "season", "away_moneyline", "home_moneyline"]
    out = spread.merge(ml, on=["game_date", "away_team", "home_team", "season"], how="outer")
    out["away_team"] = out["away_team"].apply(_normalize_team_for_join)
    out["home_team"] = out["home_team"].apply(_normalize_team_for_join)
    print(f"✅ Game lines: {len(out)} games with spread/ML, seasons {seasons}")
    return out


@timed
def load_player_game_logs(seasons: list[str], max_files: int | None = None) -> pd.DataFrame:
    """Load player game logs from S3 for actuals (PTS, REB, AST, etc.). If max_files set, stop after that many files (dry run)."""
    from src.player_team_history.name_normalization import normalize_from_nba_api
    all_dfs = []
    s3 = boto3.client("s3")
    n_loaded = 0
    for season in seasons:
        prefix = f"player_game_logs/{season}/"
        resp = s3.list_objects_v2(Bucket=S3_NBA_API, Prefix=prefix)
        if "Contents" not in resp:
            continue
        for obj in resp["Contents"]:
            if max_files is not None and n_loaded >= max_files:
                break
            key = obj["Key"]
            if not key.endswith(".csv"):
                continue
            try:
                body = _s3_read_csv_with_retry(s3, S3_NBA_API, key)
                df = pd.read_csv(StringIO(body))
                df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
                df["game_date"] = df["GAME_DATE"].dt.date.astype(str)
                df["player_normalized"] = df["PLAYER_NAME"].apply(normalize_from_nba_api)
                df["season"] = season
                all_dfs.append(df)
                n_loaded += 1
            except Exception as e:
                print(f"  ⚠️  Skip {key}: {e}")
        if max_files is not None and n_loaded >= max_files:
            break
    if not all_dfs:
        return pd.DataFrame()
    non_empty = [df for df in all_dfs if not df.empty]
    if not non_empty:
        return pd.DataFrame()
    out = pd.concat(non_empty, ignore_index=True)
    print(f"✅ Game logs: {len(out):,} player-game rows, seasons {seasons}")
    return out


def _list_s3_files_for_dates(bad_dates: list, seasons: list[str], max_dates_to_show: int = 25) -> None:
    """For each unmatched date (up to max_dates_to_show), list S3 keys for game_results and props so we can see if data exists for that date."""
    if not bad_dates:
        return
    s3 = boto3.client("s3")
    # Game results: list all keys under prefix; map date (filename without .csv) -> key
    game_results_by_date = {}
    print("   Listing S3 game_results (s3://nba-betting-mt/...)...")
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BETTING, Prefix=GAME_RESULTS_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".csv"):
                continue
            fn = key.split("/")[-1]
            date_str = fn.replace(".csv", "")
            game_results_by_date.setdefault(date_str, []).append(f"s3://{S3_BETTING}/{key}")
    print(f"   Listing S3 props (seasons {seasons})...")
    # Props: list each season prefix; map date -> keys (filename is often {date}.csv)
    props_by_date = {}
    bad_dates_set = set(bad_dates)
    for season in seasons:
        prefix = f"nba/historical_player_props/{season}/"
        resp = s3.list_objects_v2(Bucket=S3_ODDS, Prefix=prefix)
        for obj in resp.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".csv"):
                continue
            fn = key.split("/")[-1].replace(".csv", "")
            if fn in bad_dates_set:
                props_by_date.setdefault(fn, []).append(f"s3://{S3_ODDS}/{key}")
    dates_to_show = sorted(bad_dates)[:max_dates_to_show]
    print(f"   S3 files for unmatched dates (showing {len(dates_to_show)} of {len(bad_dates)}):")
    for date in dates_to_show:
        gr = game_results_by_date.get(date, [])
        pr = props_by_date.get(date, [])
        gr_str = gr[0] if len(gr) == 1 else (f"{len(gr)} files" if gr else "none")
        pr_str = pr[0] if len(pr) == 1 else (f"{len(pr)} files" if pr else "none")
        print(f"      {date}: game_results={gr_str} | props={pr_str}")


def _debug_nulls_after_pivot(pivoted_df: pd.DataFrame, seasons: list[str]) -> None:
    """Report null GAME_ID after pivot (one row per player-game) and list S3 files for each unmatched date."""
    nulls = pivoted_df[pivoted_df["GAME_ID"].isna()]
    if nulls.empty:
        return
    n_rows = len(nulls)
    distinct_keys = nulls[["game_date", "team_full"]].drop_duplicates() if "team_full" in nulls.columns else nulls[["game_date"]].drop_duplicates()
    n_distinct = len(distinct_keys)
    bad_dates = sorted(nulls["game_date"].dropna().astype(str).unique())
    print(f"   After pivot: {n_rows:,} player-game rows with null GAME_ID ({n_distinct:,} distinct (game_date, team_full))")
    print(f"   Unmatched game_dates (sample): {bad_dates[:15]} ... ({len(bad_dates)} total)")
    if "team_full" in nulls.columns:
        by_team = nulls.groupby("team_full", dropna=False).size().sort_values(ascending=False)
        print(f"   Unmatched by team_full (top 10): {by_team.head(10).to_dict()}")
    print("   Checking S3 for files on unmatched dates...")
    _list_s3_files_for_dates(bad_dates, seasons)


def join_all(
    games_df: pd.DataFrame,
    props_df: pd.DataFrame,
    lines_df: pd.DataFrame,
    logs_df: pd.DataFrame,
):
    """
    Join props -> games (via team_full), games -> game lines, then add actuals from logs.
    Uses left joins so unmatched props stay in the dataset with nulls in right-side columns
    (GAME_ID, home_team, scores, game lines, actuals). Filter where GAME_ID is null to inspect non-joins.
    """
    if games_df.empty or props_df.empty:
        return pd.DataFrame()

    # Props have game_date, team_full, season. Games have game_date, home_team, away_team, GAME_ID, season, etc.
    # Build one row per (game, team) so we can join props on team_full; keep both home_team and away_team on every row for the game-lines merge.
    base_cols = ["game_date", "GAME_ID", "home_team", "away_team", "HOME_SCORE", "AWAY_SCORE"]
    if "season" in games_df.columns:
        base_cols.append("season")
    home = games_df[base_cols].copy()
    home["team_full"] = home["home_team"]
    away = games_df[base_cols].copy()
    away["team_full"] = away["away_team"]
    games_long = pd.concat([home, away], ignore_index=True).drop_duplicates()

    # Match on date + team only. Do not join on season: props season can be wrong (S3 folder), which caused ~34k false unmatched.
    join_keys = ["game_date", "team_full"]
    # Drop prop-level home/away so we get canonical home_team/away_team from games only (avoids _x/_y suffix and missing key for lines merge)
    props_for_merge = props_df.drop(columns=["home_team", "away_team"], errors="ignore")
    props_with_game = props_for_merge.merge(games_long, on=join_keys, how="left")
    # Use game's season (from games_long); drop prop season to avoid duplicate column
    if "season_y" in props_with_game.columns:
        props_with_game["season"] = props_with_game["season_y"]
        props_with_game = props_with_game.drop(columns=["season_x", "season_y"], errors="ignore")
    matched = props_with_game["GAME_ID"].notna().sum()
    unmatched_count = props_with_game["GAME_ID"].isna().sum()
    print(f"   Props matched to games: {matched:,} | unmatched (null GAME_ID): {unmatched_count:,}")

    if not lines_df.empty:
        line_keys = ["game_date", "away_team", "home_team"]
        if "season" in lines_df.columns:
            line_keys.append("season")
        props_with_game = props_with_game.merge(
            lines_df,
            on=line_keys,
            how="left",
        )
        print(f"   Game lines attached")

    if not logs_df.empty:
        # Attach all game-log columns by (game_date, player_normalized). Stats as actual_* for pivot; rest as log_* to keep.
        # Game log schema (NBA API): PLAYER_ID, PLAYER_NAME, TEAM_ID, TEAM_NAME, GAME_ID, GAME_DATE, MATCHUP, WL, MIN, PTS, FGM, FGA, FG_PCT, FG3M, FG3A, FG3_PCT, FTM, FTA, FT_PCT, OREB, DREB, REB, AST, STL, BLK, TOV, PF, PLUS_MINUS
        log_sub = logs_df.copy()
        merge_keys = ["game_date", "player_normalized"]
        actual_rename = {
            "PTS": "actual_pts", "REB": "actual_reb", "AST": "actual_ast", "STL": "actual_stl",
            "BLK": "actual_blk", "TOV": "actual_tov", "MIN": "actual_min", "FG3M": "actual_threes",
        }
        renames = {k: v for k, v in actual_rename.items() if k in log_sub.columns}
        for col in log_sub.columns:
            if col in merge_keys or col in renames:
                continue
            renames[col] = f"log_{col.lower()}"
        log_sub = log_sub.rename(columns=renames)
        props_with_game = props_with_game.merge(
            log_sub,
            on=merge_keys,
            how="left",
        )
        n_with_pts = props_with_game["actual_pts"].notna().sum() if "actual_pts" in props_with_game.columns else 0
        print(f"   Actuals attached: {n_with_pts:,} rows with actual_pts ({len(renames)} log columns kept)")

    return props_with_game


# Nine prop markets (from distinct market in props cache). Each gets market_median_value_{market} (can be null) and actual_{stat} (from game results).
# DuckDB to list: SELECT DISTINCT market FROM 'data/02_cache/nba_strategy_build/player_props_....parquet' ORDER BY market;
PROP_MARKET_ORDER = [
    "player_assists",
    "player_blocks",
    "player_double_double",
    "player_points",
    "player_points_rebounds_assists",
    "player_rebounds",
    "player_steals",
    "player_threes",
    "player_triple_double",
]

# Map market -> actual column name (from game logs). None = no direct column (computed or null).
MARKET_TO_ACTUAL = {
    "player_points": "actual_points",  # from actual_pts
    "player_rebounds": "actual_rebounds",  # from actual_reb
    "player_assists": "actual_assists",  # from actual_ast
    "player_threes": "actual_threes",  # from FG3M in game logs
    "player_steals": "actual_steals",  # from actual_stl
    "player_blocks": "actual_blocks",  # from actual_blk
    "player_points_rebounds_assists": "actual_points_rebounds_assists",  # computed
    "player_double_double": "actual_double_double",  # computed
    "player_triple_double": "actual_triple_double",  # computed
}


def _actual_stat(row: dict, market: str) -> float | None:
    """Derive actual_* for this market from game-log actuals. Returns None if no data."""
    pts = row.get("actual_pts")
    reb = row.get("actual_reb")
    ast = row.get("actual_ast")
    stl = row.get("actual_stl")
    blk = row.get("actual_blk")
    if market == "player_points":
        return float(pts) if pts is not None and not pd.isna(pts) else None
    if market == "player_rebounds":
        return float(reb) if reb is not None and not pd.isna(reb) else None
    if market == "player_assists":
        return float(ast) if ast is not None and not pd.isna(ast) else None
    if market == "player_threes":
        threes = row.get("actual_threes")
        return float(threes) if threes is not None and not pd.isna(threes) else None
    if market == "player_steals":
        return float(stl) if stl is not None and not pd.isna(stl) else None
    if market == "player_blocks":
        return float(blk) if blk is not None and not pd.isna(blk) else None
    if market == "player_points_rebounds_assists":
        if pts is not None and reb is not None and ast is not None and not any(pd.isna(x) for x in (pts, reb, ast)):
            return float(pts) + float(reb) + float(ast)
        return None
    if market == "player_double_double":
        count = sum(1 for v in (pts, reb, ast, stl, blk) if v is not None and not pd.isna(v) and float(v) >= 10)
        return 1.0 if count >= 2 else 0.0
    if market == "player_triple_double":
        count = sum(1 for v in (pts, reb, ast, stl, blk) if v is not None and not pd.isna(v) and float(v) >= 10)
        return 1.0 if count >= 3 else 0.0
    return None


def pivot_props_long_to_wide(df: pd.DataFrame, n_markets: int = 9) -> pd.DataFrame:
    """
    One row per player-game. For each of 9 markets: market_median_value_{market} (median prop_line
    across bookmakers, can be null) and actual_{stat} (from game results; null only if no game data).
    """
    if df.empty or "market" not in df.columns or "prop_line" not in df.columns:
        return df
    key_cols = ["player", "game_date", "season"]
    key_cols = [c for c in key_cols if c in df.columns]
    pivot_cols = ["market", "prop_line", "over_odds", "under_odds"]
    # Drop bookmaker-specific columns when we aggregate across bookmakers; add list of bookmakers that contributed.
    drop_after_aggregate = ["bookmaker", "bookmaker_last_update", "market_last_update"]
    market_order = [m for m in PROP_MARKET_ORDER if m in MARKET_TO_ACTUAL][:n_markets]
    rows = []
    for _, g in df.groupby(key_cols, dropna=False):
        first = g.iloc[0]
        row = first.drop(pivot_cols, errors="ignore").to_dict()
        for col in drop_after_aggregate:
            row.pop(col, None)
        row["bookmakers"] = sorted(g["bookmaker"].dropna().unique().tolist()) if "bookmaker" in g.columns else []
        for m in market_order:
            vals = g.loc[g["market"] == m, "prop_line"].dropna()
            row[f"market_median_value_{m}"] = float(vals.median()) if len(vals) else None
            row[MARKET_TO_ACTUAL[m]] = _actual_stat(row, m)
        rows.append(row)
    out = pd.DataFrame(rows)
    print(f"   Pivoted long → wide: {len(df):,} rows → {len(out):,} rows (one per player-game, 9 markets: market_median_value_* + actual_*)")
    return out


def main():
    parser = argparse.ArgumentParser(description="Build NBA multi-market strategy dataset (game results + props + game lines + actuals)")
    parser.add_argument("--seasons", nargs="*", default=DEFAULT_SEASONS, help=f"Seasons to include (default: {' '.join(DEFAULT_SEASONS)})")
    parser.add_argument("--output", type=Path, default=None, help="Output parquet path (default: ~/Downloads/tmp/nba_strategy_3seasons.parquet)")
    parser.add_argument("--no-cache", action="store_true", help="Ignore and do not write cache; always fetch from S3")
    parser.add_argument("--max-files", type=int, default=None, metavar="N", help="Dry run: only load N files per source (skips cache for file-based sources)")
    args = parser.parse_args()
    seasons = args.seasons
    out_path = args.output or (OUTPUT_DIR / f"nba_strategy_{len(seasons)}seasons.parquet")
    out_path = Path(out_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    use_cache = not args.no_cache
    max_files = getattr(args, "max_files", None)
    if max_files is not None:
        use_cache = False  # dry run: don't use cache so we actually limit files
    seasons_key = _seasons_cache_key(seasons)
    output_cache_dir = out_path.parent / "nba_strategy_build"

    print("=" * 60)
    print(f"Building NBA strategy dataset — seasons {seasons}")
    if max_files is not None:
        print(f"Dry run: max {max_files} files per source (cache disabled)")
    if use_cache:
        print(f"Cache: {CACHE_DIR} and {output_cache_dir}")
    else:
        print("Cache disabled (--no-cache)")
    print("=" * 60)
    main_start = time.perf_counter()

    games_loader_args = (seasons,) if max_files is None else (seasons, max_files)
    games_df = _load_with_cache("game_results", load_game_results, games_loader_args, use_cache, seasons_key=seasons_key, output_cache_dir=output_cache_dir)
    _normalize_team_columns(games_df, ["home_team", "away_team"])
    # ESPN includes All-Star/exhibition (Team Shaq, Eastern Conf All-Stars, etc.); keep only NBA-vs-NBA games
    n_before = len(games_df)
    teams_in_data = set(games_df["home_team"].dropna()) | set(games_df["away_team"].dropna())
    dropped_teams = sorted(teams_in_data - CANONICAL_TEAM_NAMES)
    games_df = games_df[
        games_df["home_team"].isin(CANONICAL_TEAM_NAMES) & games_df["away_team"].isin(CANONICAL_TEAM_NAMES)
    ].copy()
    if dropped_teams:
        print(f"   Filtered game_results to NBA-only: {n_before} → {len(games_df)} games")
        print(f"   Dropped non-NBA teams ({len(dropped_teams)}): {dropped_teams}")
    if max_files is None:
        _assert_30_teams(games_df, "game_results", ["home_team", "away_team"])
    props_loader_args = (seasons,) if max_files is None else (seasons, max_files)
    props_df = _load_with_cache("player_props", load_player_props, props_loader_args, use_cache, seasons_key=seasons_key, output_cache_dir=output_cache_dir)
    history_df = _load_with_cache("player_team_history", load_player_team_history, (), use_cache, seasons_key=None, output_cache_dir=output_cache_dir)
    if max_files is None:
        _assert_30_teams(history_df, "player_team_history", ["team"], use_abbr=True)
    props_df = add_team_to_props(props_df, history_df)
    if max_files is None:
        _assert_30_teams(props_df, "player_props (team_full)", ["team_full"])
    lines_loader_args = (seasons,) if max_files is None else (seasons, max_files)
    lines_df = _load_with_cache("game_lines", load_game_lines, lines_loader_args, use_cache, seasons_key=seasons_key, output_cache_dir=output_cache_dir)
    _normalize_team_columns(lines_df, ["home_team", "away_team"])
    if max_files is None:
        _assert_30_teams(lines_df, "game_lines", ["away_team", "home_team"])
    logs_loader_args = (seasons,) if max_files is None else (seasons, max_files)
    logs_df = _load_with_cache("game_logs", load_player_game_logs, logs_loader_args, use_cache, seasons_key=seasons_key, output_cache_dir=output_cache_dir)
    if max_files is None:
        _assert_30_teams(logs_df, "game_logs", ["TEAM_NAME"])

    merged = join_all(games_df, props_df, lines_df, logs_df)
    if merged.empty:
        print("No data to write.")
        return
    print("   Pivoting long → wide (one row per player-game)...")
    merged = pivot_props_long_to_wide(merged, n_markets=9)
    if merged.empty:
        print("No data to write after pivot.")
        return
    _debug_nulls_after_pivot(merged, seasons)
    before = len(merged)
    merged = merged[merged["GAME_ID"].notna()].copy()
    dropped = before - len(merged)
    if dropped:
        print(f"   Dropped {dropped:,} rows with null GAME_ID (output has matched player-games only)")
    t0 = time.perf_counter()
    merged.to_parquet(out_path, index=False)
    print(f"   ⏱  to_parquet: {time.perf_counter() - t0:.2f}s")
    print(f"\n💾 Wrote {len(merged):,} rows to {out_path}")
    print(f"   ⏱  total: {time.perf_counter() - main_start:.2f}s")


if __name__ == "__main__":
    main()
