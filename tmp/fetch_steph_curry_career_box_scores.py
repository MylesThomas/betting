"""
Fetch Stephen Curry career regular-season box score game logs via nba_api.

Context:
- You are building `src/nba_three_point_modeling/00_research/notebooks/`
  research and need a working tmp script in parallel while iterating in the
  notebook.
- The local environment can fail SSL validation on `stats.nba.com`, so this
  script applies the same repo-standard requests SSL workaround used elsewhere.

What this script does:
1) Resolves player id from nba_api static player list
2) Pulls career season range from CommonPlayerInfo
3) Fetches one PlayerGameLog per season
4) Concatenates all seasons, sorts by date, prints sanity checks
5) Optionally merges all CommonPlayerInfo columns onto each game row
6) Writes a wide CSV plus a columns inventory file to `tmp/`

Default target:
- Stephen Curry, Regular Season, full career.
"""

from __future__ import annotations

import argparse
import ssl
import time
from pathlib import Path

import pandas as pd
import requests
import urllib3

# Must be applied before nba_api imports.
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
_ORIGINAL_REQUEST = requests.Session.request


def _patched_request(self, *args, **kwargs):
    kwargs["verify"] = False
    return _ORIGINAL_REQUEST(self, *args, **kwargs)


requests.Session.request = _patched_request

from nba_api.stats.endpoints import commonplayerinfo
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players


def get_player_id_exact(full_name: str) -> int:
    """Return nba_api player id for one exact full-name match."""
    matches = [p for p in players.get_players() if p["full_name"] == full_name]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one player named '{full_name}', found {len(matches)}"
        )
    return int(matches[0]["id"])


def build_season_strings(from_year: int, to_year: int) -> list[str]:
    """Return nba_api season labels, e.g. 2009-10."""
    return [f"{year}-{str(year + 1)[-2:]}" for year in range(from_year, to_year + 1)]


def fetch_career_game_logs(
    player_id: int, season_type: str, sleep_seconds: float
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Fetch and combine career game logs for one player."""
    info_df = commonplayerinfo.CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
    from_year = int(info_df["FROM_YEAR"].iloc[0])
    to_year = int(info_df["TO_YEAR"].iloc[0])
    seasons = build_season_strings(from_year=from_year, to_year=to_year)

    frames: list[pd.DataFrame] = []
    for i, season in enumerate(seasons, start=1):
        print(f"[{i}/{len(seasons)}] fetching {season}")
        frame = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star=season_type,
        ).get_data_frames()[0]
        frame["SEASON_STR"] = season
        frames.append(frame)
        time.sleep(sleep_seconds)

    combined = pd.concat(frames, ignore_index=True)
    combined["GAME_DATE"] = pd.to_datetime(combined["GAME_DATE"])
    combined = combined.sort_values("GAME_DATE").reset_index(drop=True)
    return combined, seasons, info_df


def build_output_path(player_name: str) -> Path:
    """Build output file path in tmp directory."""
    safe_name = player_name.lower().replace(" ", "_")
    return Path(f"/Users/thomasmyles/dev/betting/tmp/{safe_name}_career_box_scores.csv")


def attach_common_player_info(games_df: pd.DataFrame, info_df: pd.DataFrame) -> pd.DataFrame:
    """Attach CommonPlayerInfo columns to every game row (prefixed)."""
    info_row = info_df.iloc[0]
    for column_name in info_df.columns:
        prefixed = f"PLAYER_INFO_{column_name}"
        games_df[prefixed] = info_row[column_name]
    return games_df


def write_column_inventory(games_df: pd.DataFrame, output_path: Path) -> Path:
    """Write column inventory (name, dtype, non-null count) to CSV."""
    inventory_path = output_path.with_name(
        output_path.stem.replace("_career_box_scores", "_career_box_scores_columns")
        + ".csv"
    )
    inventory_df = pd.DataFrame(
        {
            "column_name": games_df.columns,
            "dtype": [str(games_df[col].dtype) for col in games_df.columns],
            "non_null_count": [int(games_df[col].notna().sum()) for col in games_df.columns],
        }
    )
    inventory_df.to_csv(inventory_path, index=False)
    return inventory_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch career nba_api player game logs to CSV."
    )
    parser.add_argument(
        "--player",
        type=str,
        default="Stephen Curry",
        help="Exact player full name in nba_api static players list.",
    )
    parser.add_argument(
        "--season-type",
        type=str,
        default="Regular Season",
        choices=["Regular Season", "Playoffs", "Pre Season", "All Star"],
        help="Season type passed to PlayerGameLog.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.6,
        help="Delay between season requests to reduce rate-limit risk.",
    )
    parser.add_argument(
        "--include-player-info",
        action="store_true",
        help=(
            "Attach all CommonPlayerInfo columns to each game row "
            "(wider output CSV)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    player_id = get_player_id_exact(args.player)
    games_df, seasons, info_df = fetch_career_game_logs(
        player_id=player_id,
        season_type=args.season_type,
        sleep_seconds=args.sleep_seconds,
    )

    if args.include_player_info:
        games_df = attach_common_player_info(games_df=games_df, info_df=info_df)

    output_path = build_output_path(args.player)
    games_df.to_csv(output_path, index=False)
    inventory_path = write_column_inventory(games_df=games_df, output_path=output_path)

    print()
    print(f"player_id: {player_id}")
    print(
        f"season_span: {seasons[0]} -> {seasons[-1]} "
        f"({len(seasons)} seasons, {args.season_type})"
    )
    print(f"games: {len(games_df)}")
    print(
        "avg_pts/ast/reb: "
        f"{games_df['PTS'].mean():.2f}/"
        f"{games_df['AST'].mean():.2f}/"
        f"{games_df['REB'].mean():.2f}"
    )
    print(f"column_count: {len(games_df.columns)}")
    print(f"saved_csv: {output_path}")
    print(f"saved_columns_inventory: {inventory_path}")


if __name__ == "__main__":
    main()
