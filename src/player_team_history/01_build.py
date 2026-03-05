"""
STEP 1: Build player team history and box score artifacts.

Context from Thomas (2026-03-05):
- Preserve existing player-team-history flow and caches.
- Extend to a second artifact with one row per player/game.
- Include PLAYER_INFO_* metadata fields on box score rows.
- Keep the process resumable and fail fast for required fields.

Outputs:
    ~/Downloads/tmp/player_team_history/
    ├── history.parquet            # Team stint history
    ├── box_scores.parquet         # Player game box scores
    ├── checkpoint.parquet         # Team history checkpoint
    ├── box_scores_checkpoint.parquet
    ├── failures.txt               # Detailed failure report
    └── cache/
        ├── seasons/*.parquet
        ├── players/*.parquet
        └── player_info/*.parquet
"""

from pathlib import Path
import argparse
import ssl
import sys
import time
import urllib3
import requests

import duckdb
import pandas as pd

# Fix SSL - must be done BEFORE importing nba_api
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

original_request = requests.Session.request


def patched_request(self, *args, **kwargs):
    kwargs["verify"] = False
    return original_request(self, *args, **kwargs)


requests.Session.request = patched_request

import requests.sessions

original_init = requests.sessions.Session.__init__


def patched_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    self.verify = False


requests.sessions.Session.__init__ = patched_init

repo_root = Path(__file__).resolve()
while not (repo_root / ".gitignore").exists():
    repo_root = repo_root.parent
sys.path.insert(0, str(repo_root))

from src.player_team_history.name_normalization import normalize_from_nba_api
from src.player_team_history.team_normalization import normalize_team_code
from src.config import CURRENT_NBA_SEASON, EMOJI

try:
    from nba_api.stats.endpoints import commonplayerinfo, playergamelog
    from nba_api.stats.static import players
except ImportError:
    print(f"{EMOJI['error']} nba_api not found. Install with: pip install nba_api")
    sys.exit(1)

OUTPUT_DIR = Path.home() / "Downloads" / "tmp" / "player_team_history"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.parquet"
BOX_CHECKPOINT_FILE = OUTPUT_DIR / "box_scores_checkpoint.parquet"
FINAL_HISTORY_OUTPUT = OUTPUT_DIR / "history.parquet"
FINAL_BOX_OUTPUT = OUTPUT_DIR / "box_scores.parquet"
FAILURE_REPORT = OUTPUT_DIR / "failures.txt"

CACHE_DIR = OUTPUT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
SEASON_CACHE_DIR = CACHE_DIR / "seasons"
PLAYER_CACHE_DIR = CACHE_DIR / "players"
PLAYER_INFO_CACHE_DIR = CACHE_DIR / "player_info"
SEASON_CACHE_DIR.mkdir(parents=True, exist_ok=True)
PLAYER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
PLAYER_INFO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

RATE_LIMIT = 0.1
TEAM_HISTORY_COLS = ["player_normalized", "team", "valid_from", "valid_to"]
REQUIRED_GAMELOG_COLS = ["Player_ID", "Game_ID", "GAME_DATE", "SEASON_ID", "MATCHUP", "TEAM"]
REQUIRED_BOX_COLS = ["player_normalized", "Player_ID", "Game_ID", "GAME_DATE", "SEASON_ID", "TEAM"]


def get_safe_player_name(player_name):
    return player_name.replace(" ", "_").replace("'", "").replace(".", "")


def get_player_cache_filename(player_name):
    return PLAYER_CACHE_DIR / f"{get_safe_player_name(player_name)}.parquet"


def get_season_cache_filename(player_name, season):
    return SEASON_CACHE_DIR / f"{get_safe_player_name(player_name)}_{season}.parquet"


def get_player_info_cache_filename(player_name):
    return PLAYER_INFO_CACHE_DIR / f"{get_safe_player_name(player_name)}.parquet"


def load_parquet_with_duckdb(path):
    return duckdb.sql(f"SELECT * FROM read_parquet('{path.as_posix()}')").df()


def atomic_write_parquet(df, target_path):
    temp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    df.to_parquet(temp_path, index=False)
    temp_path.replace(target_path)


def enforce_required_columns(df, columns, context):
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{context}: missing required columns {missing}")


def load_season_from_cache(player_name, season):
    cache_file = get_season_cache_filename(player_name, season)
    if cache_file.exists():
        try:
            return load_parquet_with_duckdb(cache_file)
        except Exception:
            cache_file.unlink()
    return None


def save_season_to_cache(player_name, season, game_logs_df):
    if game_logs_df.empty:
        return
    game_logs_df = game_logs_df.copy()
    game_logs_df["TEAM"] = game_logs_df["MATCHUP"].apply(extract_team_from_matchup)
    enforce_required_columns(game_logs_df, REQUIRED_GAMELOG_COLS, "save_season_to_cache")
    atomic_write_parquet(game_logs_df, get_season_cache_filename(player_name, season))


def load_player_from_cache(player_name):
    cache_file = get_player_cache_filename(player_name)
    if cache_file.exists():
        try:
            return load_parquet_with_duckdb(cache_file)
        except Exception:
            cache_file.unlink()
    return None


def save_player_to_cache(player_name, game_logs_df):
    if game_logs_df.empty:
        return
    atomic_write_parquet(game_logs_df, get_player_cache_filename(player_name))


def load_player_info_from_cache(player_name):
    cache_file = get_player_info_cache_filename(player_name)
    if cache_file.exists():
        try:
            cached = load_parquet_with_duckdb(cache_file)
            return cached.iloc[0].to_dict()
        except Exception:
            cache_file.unlink()
    return None


def save_player_info_to_cache(player_name, player_info_map):
    info_df = pd.DataFrame([player_info_map])
    atomic_write_parquet(info_df, get_player_info_cache_filename(player_name))


def discover_players_from_s3(sample_size=None):
    from src.player_team_history.discovery import discover_all_players

    players_set = discover_all_players(s3_sample_size=sample_size, verbose=True)
    return sorted(list(players_set))


def find_player_id(player_name):
    all_players = players.get_players()
    for player_record in all_players:
        nba_name = normalize_from_nba_api(player_record["full_name"])
        if nba_name == player_name:
            return player_record["id"]

    for player_record in all_players:
        nba_name = normalize_from_nba_api(player_record["full_name"])
        if nba_name and player_name in nba_name and player_record.get("is_active", False):
            return player_record["id"]

    parts = player_name.split()
    if len(parts) >= 2:
        reversed_name = f"{parts[-1]} {' '.join(parts[:-1])}"
        for player_record in all_players:
            nba_name = normalize_from_nba_api(player_record["full_name"])
            if nba_name and reversed_name in nba_name:
                return player_record["id"]
    return None


def get_career_seasons(player_id):
    try:
        player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id, timeout=5)
        info_df = player_info.get_data_frames()[0]
        if info_df.empty:
            return [CURRENT_NBA_SEASON]

        from_year = int(info_df["FROM_YEAR"].iloc[0])
        to_year = int(info_df["TO_YEAR"].iloc[0])
        seasons = []
        for year in range(from_year, to_year + 1):
            seasons.append(f"{year}-{str(year + 1)[-2:]}")
        return seasons
    except Exception:
        return []


def get_player_info(player_name, player_id, use_cache=True):
    cached = load_player_info_from_cache(player_name)
    if cached is not None:
        return cached

    first_name = player_name.split()[0]
    last_name = " ".join(player_name.split()[1:]) if len(player_name.split()) > 1 else ""
    fallback = {
        "PLAYER_INFO_PERSON_ID": player_id,
        "PLAYER_INFO_FIRST_NAME": first_name,
        "PLAYER_INFO_LAST_NAME": last_name,
        "PLAYER_INFO_DISPLAY_FIRST_LAST": player_name,
    }

    # For standard cached runs, avoid expensive player-info endpoint calls.
    if use_cache:
        save_player_info_to_cache(player_name, fallback)
        return fallback

    for attempt in range(3):
        try:
            player_info = commonplayerinfo.CommonPlayerInfo(player_id=player_id, timeout=5)
            info_df = player_info.get_data_frames()[0]
            if info_df.empty:
                raise ValueError(f"No player info returned for {player_name}")

            row = info_df.iloc[0].to_dict()
            prefixed = {}
            for key, value in row.items():
                prefixed[f"PLAYER_INFO_{key}"] = value
            save_player_info_to_cache(player_name, prefixed)
            return prefixed
        except Exception:
            if attempt < 2:
                time.sleep(0.3)
    save_player_info_to_cache(player_name, fallback)
    return fallback


def extract_team_from_matchup(matchup):
    if pd.isna(matchup):
        return None
    if "@" in matchup:
        return normalize_team_code(matchup.split("@")[0].strip())
    if "vs." in matchup:
        return normalize_team_code(matchup.split("vs.")[0].strip())
    return None


def fetch_player_game_log(player_name, player_id, verbose=False, use_cache=True):
    if use_cache:
        cached_logs = load_player_from_cache(player_name)
        if cached_logs is not None:
            if verbose:
                print("      [COMPLETE PLAYER CACHE]")
            enforce_required_columns(cached_logs, REQUIRED_GAMELOG_COLS, "player cache")
            return cached_logs, True

    if verbose:
        print("      [BUILDING FROM SEASONS]", flush=True)

    seasons = get_career_seasons(player_id)
    if verbose:
        print(f"      Expected: {len(seasons)} seasons...")

    all_games = []
    failed_seasons = []
    cached_count = 0
    fetched_count = 0

    for season in seasons:
        season_df = load_season_from_cache(player_name, season)
        if season_df is not None:
            all_games.append(season_df)
            cached_count += 1
            if verbose:
                print(f"      💾 {season}: {len(season_df)} games [from season cache]")
            continue

        try:
            gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season, timeout=5)
            df = gamelog.get_data_frames()[0]
            if not df.empty:
                save_season_to_cache(player_name, season, df)
                all_games.append(df)
                fetched_count += 1
                if verbose:
                    print(f"      ✓ {season}: {len(df)} games [fetched & saved]")
            time.sleep(RATE_LIMIT)
        except Exception as exc:
            failed_seasons.append((season, str(exc)[:40]))
            if verbose:
                print(f"      ✗ {season}: {str(exc)[:40]} [FAILED - try again later]")

    if verbose:
        print(
            f"      Summary: {cached_count} cached, {fetched_count} fetched, "
            f"{len(failed_seasons)} failed"
        )

    if not all_games:
        return pd.DataFrame(), False

    combined = pd.concat(all_games, ignore_index=True)
    combined["TEAM"] = combined["MATCHUP"].apply(extract_team_from_matchup)
    enforce_required_columns(combined, REQUIRED_GAMELOG_COLS, "fetch_player_game_log")

    if not failed_seasons:
        save_player_to_cache(player_name, combined)
        if verbose:
            print(f"      ✅ Saved complete player cache ({len(seasons)} seasons)")
    elif verbose:
        print(f"      ⚠️ NOT saving player cache - missing {len(failed_seasons)} seasons")
    return combined, False


def create_team_history_from_gamelogs(game_logs_df, player_name):
    if game_logs_df.empty:
        return pd.DataFrame(columns=TEAM_HISTORY_COLS)

    enforce_required_columns(game_logs_df, REQUIRED_GAMELOG_COLS, "create_team_history_from_gamelogs")

    history_input = game_logs_df.copy()
    history_input["GAME_DATE"] = pd.to_datetime(history_input["GAME_DATE"], format="mixed")
    history_input = history_input.sort_values("GAME_DATE")
    history_input["team_change"] = history_input["TEAM"] != history_input["TEAM"].shift()
    history_input["team_stint"] = history_input["team_change"].cumsum()

    history = []
    for stint_id, stint_games in history_input.groupby("team_stint"):
        team = stint_games["TEAM"].iloc[0]
        if pd.isna(team):
            continue
        first_game = stint_games["GAME_DATE"].min()
        last_game = stint_games["GAME_DATE"].max()
        is_last_stint = stint_id == history_input["team_stint"].max()
        history.append(
            {
                "player_normalized": player_name,
                "team": normalize_team_code(team),
                "valid_from": first_game.date(),
                "valid_to": None if is_last_stint else last_game.date(),
            }
        )
    return pd.DataFrame(history, columns=TEAM_HISTORY_COLS)


def season_id_to_str(season_id):
    season_id_str = str(int(season_id))
    start_year = int(season_id_str[-4:])
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def create_box_scores_from_gamelogs(game_logs_df, player_name, player_info_map):
    if game_logs_df.empty:
        return pd.DataFrame()

    box_df = game_logs_df.copy()
    enforce_required_columns(box_df, REQUIRED_GAMELOG_COLS, "create_box_scores_from_gamelogs")
    box_df["player_normalized"] = player_name
    box_df["GAME_DATE"] = pd.to_datetime(box_df["GAME_DATE"], format="mixed")
    box_df["SEASON_STR"] = box_df["SEASON_ID"].apply(season_id_to_str)

    for key, value in player_info_map.items():
        box_df[key] = value

    enforce_required_columns(box_df, REQUIRED_BOX_COLS, "box score required fields")
    box_df = box_df.sort_values(["player_normalized", "GAME_DATE", "Game_ID"])
    box_df = box_df.drop_duplicates(subset=["player_normalized", "Game_ID"], keep="last")
    return box_df


def load_checkpoint(path, columns):
    if not path.exists():
        return pd.DataFrame(columns=columns)
    return load_parquet_with_duckdb(path)


def get_completed_players(checkpoint_df):
    if checkpoint_df.empty:
        return set()
    return set(checkpoint_df["player_normalized"].unique())


def save_checkpoint(path, df):
    atomic_write_parquet(df, path)


def write_failure_section(file_handle, title, entries, description):
    if not entries:
        return
    file_handle.write("=" * 80 + "\n")
    file_handle.write(f"{title} ({len(entries)} players)\n")
    file_handle.write("=" * 80 + "\n")
    file_handle.write(description + "\n\n")
    for entry in sorted(entries):
        if isinstance(entry, tuple):
            file_handle.write(f"  - {entry[0]}\n")
            file_handle.write(f"    Error: {entry[1]}\n")
        else:
            file_handle.write(f"  - {entry}\n")
    file_handle.write("\n")


def generate_failure_report(failures, total_processed, successful):
    failed = sum(len(value) for value in failures.values())
    with open(FAILURE_REPORT, "w") as file_handle:
        file_handle.write("=" * 80 + "\n")
        file_handle.write("PLAYER TEAM HISTORY + BOX SCORES BUILD - FAILURE REPORT\n")
        file_handle.write("=" * 80 + "\n\n")
        file_handle.write(f"Total players processed: {total_processed}\n")
        file_handle.write(f"Successful: {successful}\n")
        file_handle.write(f"Failed: {failed}\n\n")

        write_failure_section(
            file_handle,
            "NOT FOUND IN NBA API",
            failures["not_found_in_nba"],
            "These players exist in Odds API but could not be matched to NBA API.",
        )
        write_failure_section(
            file_handle,
            "NO GAME LOGS",
            failures["no_game_logs"],
            "Found in NBA API but no game logs available.",
        )
        write_failure_section(
            file_handle,
            "PLAYER INFO ERRORS",
            failures["player_info_errors"],
            "Unable to fetch/player cache CommonPlayerInfo output.",
        )
        write_failure_section(
            file_handle,
            "NO HISTORY CREATED",
            failures["no_history_created"],
            "Game logs fetched but team history could not be built.",
        )
        write_failure_section(
            file_handle,
            "NO BOX SCORES CREATED",
            failures["no_box_scores_created"],
            "Game logs fetched but player/game box rows were not created.",
        )
        write_failure_section(
            file_handle,
            "DUPLICATE BOX GAME ROWS",
            failures["duplicate_box_rows"],
            "Duplicate rows found for (player_normalized, Game_ID).",
        )
        write_failure_section(
            file_handle,
            "BOX SCHEMA ERRORS",
            failures["box_schema_errors"],
            "Required fields missing while building box rows.",
        )
        write_failure_section(
            file_handle,
            "PROCESSING ERRORS",
            failures["processing_errors"],
            "Unexpected errors during player processing.",
        )


def build(resume=False, sample_size=None, verbose=False, use_cache=True):
    print("=" * 80)
    print(f"{EMOJI['nba']} BUILD PLAYER TEAM HISTORY + BOX SCORES")
    print("=" * 80)
    print()

    if use_cache:
        player_cache_count = len(list(PLAYER_CACHE_DIR.glob("*.parquet")))
        season_cache_count = len(list(SEASON_CACHE_DIR.glob("*.parquet")))
        player_info_cache_count = len(list(PLAYER_INFO_CACHE_DIR.glob("*.parquet")))
        if player_cache_count > 0 or season_cache_count > 0 or player_info_cache_count > 0:
            print(
                f"{EMOJI['info']} Cache: {player_cache_count} complete players, "
                f"{season_cache_count} seasons, {player_info_cache_count} player_info files"
            )
            print()

    history_checkpoint_df = load_checkpoint(CHECKPOINT_FILE, TEAM_HISTORY_COLS) if resume else pd.DataFrame(columns=TEAM_HISTORY_COLS)
    box_checkpoint_df = load_checkpoint(BOX_CHECKPOINT_FILE, REQUIRED_BOX_COLS) if resume else pd.DataFrame(columns=REQUIRED_BOX_COLS)
    completed_players = get_completed_players(history_checkpoint_df)

    if resume and completed_players:
        print(
            f"{EMOJI['success']} Resuming from checkpoint: "
            f"{len(completed_players)} players already done\n"
        )

    all_players = discover_players_from_s3(sample_size=sample_size)
    players_to_process = [player_name for player_name in all_players if player_name not in completed_players]

    print(f"\n{EMOJI['info']} Players to process: {len(players_to_process)}")
    print(f"{EMOJI['info']} Already completed: {len(completed_players)}")
    print()

    if not players_to_process:
        print(f"{EMOJI['success']} All players already processed!")
        return history_checkpoint_df, box_checkpoint_df

    new_history = []
    new_box_scores = []
    successful = 0
    start_time = time.time()
    failures = {
        "not_found_in_nba": [],
        "no_game_logs": [],
        "player_info_errors": [],
        "no_history_created": [],
        "no_box_scores_created": [],
        "duplicate_box_rows": [],
        "box_schema_errors": [],
        "processing_errors": [],
    }

    for index, player_name in enumerate(players_to_process, 1):
        player_start = time.time()
        print(f"[{index}/{len(players_to_process)}] {player_name}...", end=" ", flush=True)
        try:
            player_id = find_player_id(player_name)
            if not player_id:
                print(f"{EMOJI['warning']} Not found in NBA API")
                failures["not_found_in_nba"].append(player_name)
                continue

            game_logs, from_cache = fetch_player_game_log(
                player_name,
                player_id,
                verbose=verbose,
                use_cache=use_cache,
            )
            if game_logs.empty:
                print(f"{EMOJI['warning']} No game logs")
                failures["no_game_logs"].append(player_name)
                continue

            try:
                player_info_map = get_player_info(player_name, player_id, use_cache=use_cache)
            except Exception as exc:
                print(f"{EMOJI['warning']} Player info error")
                failures["player_info_errors"].append((player_name, str(exc)))
                continue

            player_history = create_team_history_from_gamelogs(game_logs, player_name)
            if player_history.empty:
                print(f"{EMOJI['warning']} No history created")
                failures["no_history_created"].append(player_name)
                continue

            try:
                player_box_scores = create_box_scores_from_gamelogs(game_logs, player_name, player_info_map)
            except Exception as exc:
                print(f"{EMOJI['warning']} Box schema error")
                failures["box_schema_errors"].append((player_name, str(exc)))
                continue

            if player_box_scores.empty:
                print(f"{EMOJI['warning']} No box scores created")
                failures["no_box_scores_created"].append(player_name)
                continue

            duplicate_rows = player_box_scores[
                player_box_scores.duplicated(subset=["player_normalized", "Game_ID"], keep=False)
            ]
            if not duplicate_rows.empty:
                print(f"{EMOJI['warning']} Duplicate box rows")
                failures["duplicate_box_rows"].append(player_name)
                continue

            new_history.append(player_history)
            new_box_scores.append(player_box_scores)
            successful += 1
            elapsed = time.time() - player_start
            cache_indicator = "💾 CACHED" if from_cache else "🔄 API"
            print(
                f"{cache_indicator} {EMOJI['success']} "
                f"{len(player_history)} stints, {len(player_box_scores)} box rows ({elapsed:.1f}s)"
            )
        except Exception as exc:
            elapsed = time.time() - player_start
            error_msg = str(exc)[:60]
            print(f"{EMOJI['error']} {error_msg} ({elapsed:.1f}s)")
            failures["processing_errors"].append((player_name, str(exc)))
            continue

        if index % 25 == 0 and new_history:
            elapsed = time.time() - start_time
            rate = index / elapsed
            remaining = len(players_to_process) - index
            eta_minutes = (remaining / rate) / 60 if rate > 0 else 0

            incremental_history = pd.concat(new_history, ignore_index=True)
            incremental_box = pd.concat(new_box_scores, ignore_index=True)
            checkpoint_history = (
                incremental_history
                if history_checkpoint_df.empty
                else pd.concat([history_checkpoint_df, incremental_history], ignore_index=True)
            )
            checkpoint_box = (
                incremental_box
                if box_checkpoint_df.empty
                else pd.concat([box_checkpoint_df, incremental_box], ignore_index=True)
            )
            save_checkpoint(CHECKPOINT_FILE, checkpoint_history)
            save_checkpoint(BOX_CHECKPOINT_FILE, checkpoint_box)
            print(f"\n{EMOJI['save']} Checkpoint saved")
            print(f"   Progress: {index}/{len(players_to_process)} ({index/len(players_to_process)*100:.1f}%)")
            print(f"   Speed: {rate:.1f} players/sec")
            print(f"   ETA: {eta_minutes:.1f} min\n")

    if new_history:
        incremental_history = pd.concat(new_history, ignore_index=True)
        final_history = (
            incremental_history
            if history_checkpoint_df.empty
            else pd.concat([history_checkpoint_df, incremental_history], ignore_index=True)
        )
        final_history = final_history.sort_values(["player_normalized", "valid_from"])
        final_history = final_history.drop_duplicates(
            subset=["player_normalized", "team", "valid_from"], keep="last"
        )

        incremental_box = pd.concat(new_box_scores, ignore_index=True)
        final_box = (
            incremental_box
            if box_checkpoint_df.empty
            else pd.concat([box_checkpoint_df, incremental_box], ignore_index=True)
        )
        final_box = final_box.sort_values(["player_normalized", "GAME_DATE", "Game_ID"])
        final_box = final_box.drop_duplicates(
            subset=["player_normalized", "Game_ID"], keep="last"
        )

        save_checkpoint(CHECKPOINT_FILE, final_history)
        save_checkpoint(BOX_CHECKPOINT_FILE, final_box)
        atomic_write_parquet(final_history, FINAL_HISTORY_OUTPUT)
        atomic_write_parquet(final_box, FINAL_BOX_OUTPUT)

        print()
        print("=" * 80)
        print(f"{EMOJI['success']} BUILD COMPLETE")
        print("=" * 80)
        print(f"Total players (history): {final_history['player_normalized'].nunique()}")
        print(f"Total stints: {len(final_history)}")
        print(f"Total box rows: {len(final_box)}")
        print(f"Successful players: {successful}")
        print(f"Failed players: {sum(len(value) for value in failures.values())}")
        print()
        print(f"History output: {FINAL_HISTORY_OUTPUT}")
        print(f"Box score output: {FINAL_BOX_OUTPUT}")
        print(f"History checkpoint: {CHECKPOINT_FILE}")
        print(f"Box checkpoint: {BOX_CHECKPOINT_FILE}")
        print()

        if any(failures.values()):
            generate_failure_report(failures, len(players_to_process), successful)
            print(f"{EMOJI['warning']} Failure report: {FAILURE_REPORT}")
            print("   Analyze: python src/player_team_history/02_analyze_failures.py")
            print()

        return final_history, final_box

    return history_checkpoint_df, box_checkpoint_df


def main():
    parser = argparse.ArgumentParser(
        description="Build player team history + player box scores from S3 betting data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/player_team_history/01_build.py
  python src/player_team_history/01_build.py --sample 100
  python src/player_team_history/01_build.py --resume
        """,
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints")
    parser.add_argument("--sample", type=int, help="Only process sample of S3 files (for testing)")
    parser.add_argument("--verbose", action="store_true", help="Show detailed logs for each player")
    parser.add_argument("--no-cache", action="store_true", help="Bypass caches and fetch fresh data")
    args = parser.parse_args()

    build(
        resume=args.resume,
        sample_size=args.sample,
        verbose=args.verbose,
        use_cache=not args.no_cache,
    )


if __name__ == "__main__":
    main()
