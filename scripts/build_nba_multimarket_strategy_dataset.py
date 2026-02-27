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

Output: Single .parquet in ~/Downloads/tmp (default: nba_strategy_3seasons.parquet).

Uses 3 seasons by default (2023-24, 2024-25, 2025-26). Player names normalized via
src/player_team_history/name_normalization (Odds API for props, NBA API for game logs).

Usage:
    python scripts/build_nba_multimarket_strategy_dataset.py
    python scripts/build_nba_multimarket_strategy_dataset.py --seasons 2024-25 2025-26
    python scripts/build_nba_multimarket_strategy_dataset.py --output ~/Downloads/tmp/nba_strategy.parquet
    python scripts/build_nba_multimarket_strategy_dataset.py   # single parquet; filter GAME_ID.isna() for non-joins
"""

import argparse
import functools
import sys
import time
from pathlib import Path

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


def _normalize_team_for_join(name):
    """Normalize team name to ESPN/NBA canonical form (Odds API → ESPN) so joins match."""
    if pd.isna(name):
        return name
    return normalize_team_name_from_odds_api(name)


@timed
def load_game_results(seasons: list[str]) -> pd.DataFrame:
    """Load ESPN game results for all given seasons from S3; filter by season date range."""
    all_dfs = []
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    for season in seasons:
        start, end = _season_date_range(season)
        start_dt = pd.to_datetime(start).date()
        end_dt = pd.to_datetime(end).date()
        for page in paginator.paginate(Bucket=S3_BETTING, Prefix=GAME_RESULTS_PREFIX):
            for obj in page.get("Contents", []):
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
                except Exception as e:
                    print(f"  ⚠️  Skip {key}: {e}")
    if not all_dfs:
        return pd.DataFrame()
    non_empty = [df for df in all_dfs if not df.empty]
    if not non_empty:
        return pd.DataFrame()
    out = pd.concat(non_empty, ignore_index=True)
    print(f"✅ Game results: {len(out)} games, seasons {seasons}, {out['game_date'].min()} to {out['game_date'].max()}")
    return out


@timed
def load_player_props(seasons: list[str]) -> pd.DataFrame:
    """Load all player props (all markets) from S3; add game_date from filename, player_normalized. Uses Odds API normalization."""
    from src.player_team_history.name_normalization import normalize_from_odds_api
    all_dfs = []
    s3 = boto3.client("s3")
    for season in seasons:
        prefix = f"nba/historical_player_props/{season}/"
        resp = s3.list_objects_v2(Bucket=S3_ODDS, Prefix=prefix)
        if "Contents" not in resp:
            continue
        for obj in resp["Contents"]:
            key = obj["Key"]
            if not key.endswith(".csv"):
                continue
            date_str = key.split("/")[-1].replace(".csv", "")
            try:
                body = _s3_read_csv_with_retry(s3, S3_ODDS, key)
                df = pd.read_csv(StringIO(body))
                df["game_date"] = date_str
                df["season"] = season
                if "player" not in df.columns:
                    continue
                df["player_normalized"] = df["player"].apply(normalize_from_odds_api)
                all_dfs.append(df)
            except Exception as e:
                print(f"  ⚠️  Skip {key}: {e}")
    if not all_dfs:
        return pd.DataFrame()
    non_empty = [df for df in all_dfs if not df.empty]
    if not non_empty:
        return pd.DataFrame()
    out = pd.concat(non_empty, ignore_index=True)
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
def load_game_lines(seasons: list[str]) -> pd.DataFrame:
    """Load game lines (spread, moneyline) from S3.
    S3 keys: nba/historical_game_lines/{season}/nba_game_lines_YYYY-MM-DD.csv (or YYYY-MM-DD.csv).
    Lists by prefix and uses returned keys; supports both filename patterns."""
    all_dfs = []
    s3 = boto3.client("s3")
    for season in seasons:
        prefix = f"nba/historical_game_lines/{season}/"
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=S3_ODDS, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith(".csv") or "failed" in key.lower():
                    continue
                fn = key.split("/")[-1]
                # Filenames are nba_game_lines_YYYY-MM-DD.csv (not plain YYYY-MM-DD.csv)
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
                except Exception as e:
                    print(f"  ⚠️  Skip {key}: {e}")
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
def load_player_game_logs(seasons: list[str]) -> pd.DataFrame:
    """Load player game logs from S3 for actuals (PTS, REB, AST, etc.). Uses NBA API name normalization."""
    from src.player_team_history.name_normalization import normalize_from_nba_api
    all_dfs = []
    s3 = boto3.client("s3")
    for season in seasons:
        prefix = f"player_game_logs/{season}/"
        resp = s3.list_objects_v2(Bucket=S3_NBA_API, Prefix=prefix)
        if "Contents" not in resp:
            continue
        for obj in resp["Contents"]:
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
            except Exception as e:
                print(f"  ⚠️  Skip {key}: {e}")
    if not all_dfs:
        return pd.DataFrame()
    non_empty = [df for df in all_dfs if not df.empty]
    if not non_empty:
        return pd.DataFrame()
    out = pd.concat(non_empty, ignore_index=True)
    print(f"✅ Game logs: {len(out):,} player-game rows, seasons {seasons}")
    return out


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

    join_keys = ["game_date", "team_full"]
    if "season" in props_df.columns and "season" in games_long.columns:
        join_keys.append("season")
    props_with_game = props_df.merge(games_long, on=join_keys, how="left")
    matched = props_with_game["GAME_ID"].notna().sum()
    unmatched = props_with_game["GAME_ID"].isna().sum()
    print(f"   Props matched to games: {matched:,} | unmatched (null GAME_ID): {unmatched:,}")

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
        # Attach actuals: PTS, REB, AST, etc. by (game_date, player_normalized)
        log_cols = ["game_date", "player_normalized", "PTS", "REB", "AST", "STL", "BLK", "TOV", "MIN"]
        have = [c for c in log_cols if c in logs_df.columns]
        log_sub = logs_df[have].copy()
        log_sub = log_sub.rename(columns={c: f"actual_{c.lower()}" for c in have if c != "game_date" and c != "player_normalized"})
        log_sub = log_sub.rename(columns={"PTS": "actual_pts", "REB": "actual_reb", "AST": "actual_ast", "STL": "actual_stl", "BLK": "actual_blk", "TOV": "actual_tov", "MIN": "actual_min"})
        if "actual_pts" not in log_sub.columns and "PTS" in logs_df.columns:
            log_sub["actual_pts"] = logs_df["PTS"].values
        props_with_game = props_with_game.merge(
            log_sub,
            on=["game_date", "player_normalized"],
            how="left",
        )
        print(f"   Actuals attached: {props_with_game['actual_pts'].notna().sum():,} rows with actual_pts")

    return props_with_game


def main():
    parser = argparse.ArgumentParser(description="Build NBA multi-market strategy dataset (game results + props + game lines + actuals)")
    parser.add_argument("--seasons", nargs="*", default=DEFAULT_SEASONS, help=f"Seasons to include (default: {' '.join(DEFAULT_SEASONS)})")
    parser.add_argument("--output", type=Path, default=None, help="Output parquet path (default: ~/Downloads/tmp/nba_strategy_3seasons.parquet)")
    parser.add_argument("--no-cache", action="store_true", help="Ignore and do not write cache; always fetch from S3")
    args = parser.parse_args()
    seasons = args.seasons
    out_path = args.output or (OUTPUT_DIR / f"nba_strategy_{len(seasons)}seasons.parquet")
    out_path = Path(out_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    use_cache = not args.no_cache
    seasons_key = _seasons_cache_key(seasons)
    output_cache_dir = out_path.parent / "nba_strategy_build"

    print("=" * 60)
    print(f"Building NBA strategy dataset — seasons {seasons}")
    if use_cache:
        print(f"Cache: {CACHE_DIR} and {output_cache_dir}")
    else:
        print("Cache disabled (--no-cache)")
    print("=" * 60)
    main_start = time.perf_counter()

    games_df = _load_with_cache("game_results", load_game_results, (seasons,), use_cache, seasons_key=seasons_key, output_cache_dir=output_cache_dir)
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
    _assert_30_teams(games_df, "game_results", ["home_team", "away_team"])
    props_df = _load_with_cache("player_props", load_player_props, (seasons,), use_cache, seasons_key=seasons_key, output_cache_dir=output_cache_dir)
    history_df = _load_with_cache("player_team_history", load_player_team_history, (), use_cache, seasons_key=None, output_cache_dir=output_cache_dir)
    _assert_30_teams(history_df, "player_team_history", ["team"], use_abbr=True)
    props_df = add_team_to_props(props_df, history_df)
    _assert_30_teams(props_df, "player_props (team_full)", ["team_full"])
    lines_df = _load_with_cache("game_lines", load_game_lines, (seasons,), use_cache, seasons_key=seasons_key, output_cache_dir=output_cache_dir)
    _assert_30_teams(lines_df, "game_lines", ["away_team", "home_team"])
    logs_df = _load_with_cache("game_logs", load_player_game_logs, (seasons,), use_cache, seasons_key=seasons_key, output_cache_dir=output_cache_dir)
    _assert_30_teams(logs_df, "game_logs", ["TEAM_NAME"])

    merged = join_all(games_df, props_df, lines_df, logs_df)
    if merged.empty:
        print("No data to write.")
        return
    t0 = time.perf_counter()
    merged.to_parquet(out_path, index=False)
    print(f"   ⏱  to_parquet: {time.perf_counter() - t0:.2f}s")
    print(f"\n💾 Wrote {len(merged):,} rows to {out_path}")
    print(f"   ⏱  total: {time.perf_counter() - main_start:.2f}s")


if __name__ == "__main__":
    main()
