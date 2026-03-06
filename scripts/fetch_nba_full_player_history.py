"""
Fetch full NBA game-log history for one player and store it in S3.

Context:
- V2 player_threes uncertainty modeling needs reusable player-history data for
  minutes and 3PA sampling.
- This script fetches complete game logs from nba_api for one player and writes:
  `s3://nba-api-mt/full_player_history/{full name}.csv`
- Current operational need is support for:
  `--player-name "Stephen Curry"`
"""

from __future__ import annotations

import argparse
import io
import ssl
import time
from urllib.parse import urlparse

import boto3
import pandas as pd
import requests
import urllib3

# Apply before nba_api imports.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch full player game history to S3.")
    parser.add_argument(
        "--player-name",
        required=True,
        help="Exact nba_api player full name, for example 'Stephen Curry'.",
    )
    parser.add_argument(
        "--season-type",
        default="Regular Season",
        choices=["Regular Season", "Playoffs", "Pre Season", "All Star"],
        help="Season type passed to nba_api PlayerGameLog.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.6,
        help="Delay between season requests to reduce rate-limit risk.",
    )
    parser.add_argument(
        "--s3-prefix",
        default="s3://nba-api-mt/full_player_history",
        help="S3 prefix for uploaded CSV files.",
    )
    return parser.parse_args()


def get_player_id(full_name: str) -> int:
    matches = [p for p in players.get_players() if p["full_name"] == full_name]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one player named '{full_name}', found {len(matches)}"
        )
    return int(matches[0]["id"])


def build_season_labels(from_year: int, to_year: int) -> list[str]:
    return [f"{year}-{str(year + 1)[-2:]}" for year in range(from_year, to_year + 1)]


def fetch_full_history(
    player_id: int,
    season_type: str,
    sleep_seconds: float,
) -> pd.DataFrame:
    info_df = commonplayerinfo.CommonPlayerInfo(player_id=player_id).get_data_frames()[0]
    from_year = int(info_df["FROM_YEAR"].iloc[0])
    to_year = int(info_df["TO_YEAR"].iloc[0])
    seasons = build_season_labels(from_year=from_year, to_year=to_year)

    frames = []
    for idx, season in enumerate(seasons, start=1):
        print(f"[{idx}/{len(seasons)}] fetching {season}")
        frame = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star=season_type,
        ).get_data_frames()[0]
        frame["SEASON_STR"] = season
        frames.append(frame)
        time.sleep(sleep_seconds)

    history_df = pd.concat(frames, ignore_index=True)
    history_df["GAME_DATE"] = pd.to_datetime(history_df["GAME_DATE"])
    history_df = history_df.sort_values("GAME_DATE").reset_index(drop=True)
    return history_df


def upload_csv_to_s3(df: pd.DataFrame, s3_uri: str) -> None:
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    payload = io.StringIO()
    df.to_csv(payload, index=False)
    boto3.client("s3").put_object(
        Bucket=bucket,
        Key=key,
        Body=payload.getvalue().encode("utf-8"),
        ContentType="text/csv",
    )


def main() -> None:
    args = parse_args()
    player_id = get_player_id(args.player_name)
    history_df = fetch_full_history(
        player_id=player_id,
        season_type=args.season_type,
        sleep_seconds=args.sleep_seconds,
    )
    output_uri = f"{args.s3_prefix}/{args.player_name}.csv"
    upload_csv_to_s3(df=history_df, s3_uri=output_uri)

    print(f"player_id: {player_id}")
    print(f"rows: {len(history_df)}")
    print(f"s3_uri: {output_uri}")


if __name__ == "__main__":
    main()
