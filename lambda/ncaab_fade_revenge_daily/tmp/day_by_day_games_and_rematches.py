"""
Day-by-day NCAAB: per-season min/max games and min/max rematch games.

For each season 2020-21 through 2025-26, scan every game day and report:
- min_games / max_games (fewest and most games on a single calendar day)
- min_rematch_games / max_rematch_games (fewest and most rematch games on a single day)

Rematch = 2nd or 3rd meeting between the same two teams in that season (pair = sorted home/away).

Uses the same cache as analyze_ncaab_conference_rematch_su_ats.py in this folder
(rematch_joined_{season}_{et_date}.parquet in ~/Downloads/tmp/ncaab_cache/).
Run from repo root so PROJECT_ROOT and tmp imports resolve.

Usage:
    cd ~/dev/betting
    python3 lambda/ncaab_fade_revenge_daily/tmp/day_by_day_games_and_rematches.py
"""

import sys
from pathlib import Path

import pandas as pd

# -----------------------------------------------------------------------------
# Project root and path setup (no parent-based sys.path per cursor rules)
# -----------------------------------------------------------------------------


def find_project_root():
    """Find project root by .gitignore."""
    current = Path.cwd()
    while current != current.parent:
        if (current / '.gitignore').exists():
            return current
        current = current.parent
    return Path.cwd()


PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'tmp'))
# Same tmp dir as this script so we can import the analyze script and reuse its cache
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from join_ncaab_outcomes_and_lines import SEASON_DATES
from analyze_ncaab_conference_rematch_su_ats import get_today_et, load_joined_from_cache_or_s3

# Seasons to report (order for output table)
SEASONS = ['2020-21', '2021-22', '2022-23', '2023-24', '2024-25', '2025-26']


def add_meeting_number_and_is_rematch(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each game add meeting_number (1, 2, 3, ...) per (home, away) pair and is_rematch (2nd or 3rd meeting).
    Pair is canonical sorted [HOME_TEAM, AWAY_TEAM] so same two teams count as one pair.
    """
    df = df.copy()
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE']).dt.date
    df['pair'] = df.apply(
        lambda r: tuple(sorted([r['HOME_TEAM'], r['AWAY_TEAM']])), axis=1
    )
    df = df.sort_values(['pair', 'GAME_DATE']).reset_index(drop=True)

    meeting_numbers = []
    for _, grp in df.groupby('pair', sort=False):
        meeting_numbers.extend(range(1, len(grp) + 1))
    df['meeting_number'] = meeting_numbers
    df['is_rematch'] = df['meeting_number'] >= 2
    return df


def daily_stats_for_season(joined_df: pd.DataFrame) -> tuple[int, int, int, int]:
    """
    Given joined df with is_rematch and GAME_DATE, aggregate by date.
    Returns (min_games, max_games, min_rematch_games, max_rematch_games).
    """
    daily = joined_df.groupby('GAME_DATE').agg(
        games=('GAME_DATE', 'size'),
        rematch_games=('is_rematch', 'sum'),
    ).reset_index()
    min_games = int(daily['games'].min())
    max_games = int(daily['games'].max())
    min_rematches = int(daily['rematch_games'].min())
    max_rematches = int(daily['rematch_games'].max())
    return min_games, max_games, min_rematches, max_rematches


def main():
    et_date = get_today_et()
    rows = []

    for season in SEASONS:
        if season not in SEASON_DATES:
            rows.append((season, '—', '—'))
            continue
        joined_df = load_joined_from_cache_or_s3(season, et_date)
        df = add_meeting_number_and_is_rematch(joined_df)
        min_g, max_g, min_r, max_r = daily_stats_for_season(df)
        rows.append((season, f"{min_g} / {max_g}", f"{min_r} / {max_r}"))

    # Print table
    print("Day-by-day NCAAB: min/max games and min/max rematch games per season")
    print("(Rematch = 2nd or 3rd meeting between same two teams in that season.)")
    print()
    col_year = "year"
    col_games = "min_games / max_games"
    col_rematch = "min_rematch_games / max_rematch_games"
    w_year = max(len(col_year), 8)
    w_games = max(len(col_games), 28)
    w_rematch = max(len(col_rematch), 35)
    fmt = f"{{:<{w_year}}}  {{:<{w_games}}}  {{:<{w_rematch}}}"
    print(fmt.format(col_year, col_games, col_rematch))
    print("-" * (w_year + w_games + w_rematch + 4))
    for season, games_str, rematch_str in rows:
        print(fmt.format(season, games_str, rematch_str))


if __name__ == '__main__':
    main()
