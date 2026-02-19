"""
NCAAB conference rematch: 2nd/3rd meeting SU and ATS for the losing team.

When conference opponents play 2nd or 3rd time: what is the record SU/ATS for
the team that lost the first meeting (2nd game) or lost both prior (3rd game)?

Data: S3 only. Cache keyed by ET date: same day (ET) reuses cache; next day loads fresh.
- Outcomes: s3://ncaab-betting-mt/data/01_input/historical_game_results/
- Lines:   s3://ncaab-betting-mt/data/01_input/the-odds-api/ncaab/game_lines/

Usage:
    python analysis/analyze_ncaab_conference_rematch_su_ats.py --conference "Big Ten" --season "2025-26"
    python analysis/analyze_ncaab_conference_rematch_su_ats.py --conference "Big Ten" --season "2024-25" --csv
    python analysis/analyze_ncaab_conference_rematch_su_ats.py --conferences all --season "2024-25" --csv
    python analysis/analyze_ncaab_conference_rematch_su_ats.py --all-games --season "2024-25" --csv
    python analysis/analyze_ncaab_conference_rematch_su_ats.py --all-games --seasons "2020-21,2021-22,2022-23,2023-24,2024-25" --csv

Output: Stdout summary; optional CSV to data/04_output/ with rematch_type column
('rematch 2nd', 'rematch 3rd', 'not rematch') for sense-checking.
"""

import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import duckdb

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

from join_ncaab_outcomes_and_lines import (
    load_game_outcomes,
    load_game_lines,
    join_outcomes_and_lines,
    SEASON_DATES,
)
from ncaab_conference_data import NCAAB_CONFERENCE_MAPPING_2025_26

# All conferences in the mapping (for --conferences all)
ALL_CONFERENCES = sorted(set(NCAAB_CONFERENCE_MAPPING_2025_26.values()))

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

LOG = logging.getLogger(__name__)


def setup_logging(verbose: bool = True):
    """Configure logging to stdout. INFO by default; boto at WARNING to avoid S3 debug flood."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger('boto3').setLevel(logging.WARNING)
    logging.getLogger('botocore').setLevel(logging.WARNING)


# -----------------------------------------------------------------------------
# Cache (ET day)
# -----------------------------------------------------------------------------

CACHE_DIR = Path.home() / 'Downloads' / 'tmp' / 'ncaab_cache'


def get_today_et() -> str:
    """Today's date in Eastern Time, YYYY-MM-DD."""
    return datetime.now(ZoneInfo('America/New_York')).strftime('%Y-%m-%d')


def get_cache_path(season: str, et_date: str) -> Path:
    """Cache path for joined data: rematch_joined_{season}_{et_date}.parquet."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f'rematch_joined_{season}_{et_date}.parquet'


def load_joined_from_cache_or_s3(season: str, et_date: str) -> pd.DataFrame:
    """
    If cache exists for (season, et_date), load and return. Else load from S3, join, write cache, return.
    """
    cache_path = get_cache_path(season, et_date)
    if cache_path.exists():
        LOG.info("Cache hit: loading joined data from %s", cache_path)
        return pd.read_parquet(cache_path)

    LOG.info("Cache miss: loading outcomes and lines from S3 for season %s", season)
    if season not in SEASON_DATES:
        raise ValueError(
            f"Unknown season: {season}. Available: {list(SEASON_DATES.keys())}"
        )
    start_date, end_date = SEASON_DATES[season]
    outcomes_df = load_game_outcomes(start_date, end_date, use_cache=False)
    lines_df = load_game_lines(start_date, end_date, use_cache=False)
    if outcomes_df.empty:
        raise RuntimeError(
            f"No game outcomes in S3 for {season} ({start_date} to {end_date})"
        )
    if lines_df.empty:
        LOG.warning("No game lines in S3 for %s; join will have no spreads", season)

    LOG.info("Joining outcomes and lines...")
    joined_df, stats = join_outcomes_and_lines(outcomes_df, lines_df, min_games=5)
    LOG.info(
        "Joined: %s games, %s with lines (%.1f%%)",
        len(joined_df),
        stats['matched'],
        stats['coverage_pct'],
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joined_df.to_parquet(cache_path, index=False)
    LOG.info("Wrote cache: %s", cache_path)
    return joined_df


# -----------------------------------------------------------------------------
# Conference filter
# -----------------------------------------------------------------------------


def filter_to_conference(
    joined_df: pd.DataFrame, conference: str, allow_empty: bool = False
) -> pd.DataFrame:
    """Keep games where both HOME_TEAM and AWAY_TEAM are in the given conference."""
    home_conf = joined_df['HOME_TEAM'].map(NCAAB_CONFERENCE_MAPPING_2025_26)
    away_conf = joined_df['AWAY_TEAM'].map(NCAAB_CONFERENCE_MAPPING_2025_26)
    mask = (home_conf == conference) & (away_conf == conference)
    out = joined_df.loc[mask].copy()
    if out.empty:
        if allow_empty:
            return out
        raise RuntimeError(
            f"No games found for conference '{conference}'. "
            "Check conference name and that both teams are in NCAAB_CONFERENCE_MAPPING_2025_26."
        )
    LOG.info(
        "Conference filter: %s games in %s (%s team-pairs)",
        len(out),
        conference,
        out.apply(
            lambda r: tuple(sorted([r['HOME_TEAM'], r['AWAY_TEAM']])), axis=1
        ).nunique(),
    )
    return out


# -----------------------------------------------------------------------------
# Meeting number and rematch type
# -----------------------------------------------------------------------------


def add_meeting_number_and_rematch_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each game, add meeting_number (1, 2, 3, ...) per pair and rematch_type:
    'rematch 2nd', 'rematch 3rd', or 'not rematch'.

    Pair is canonical (sorted [HOME, AWAY]) so the same two teams = one pair
    regardless of who was home/away; rematches are caught when away flips to home.
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

    def rematch_type(row):
        n = row['meeting_number']
        if n == 2:
            return 'rematch 2nd'
        if n == 3:
            return 'rematch 3rd'
        return 'not rematch'

    df['rematch_type'] = df.apply(rematch_type, axis=1)
    n2 = (df['rematch_type'] == 'rematch 2nd').sum()
    n3 = (df['rematch_type'] == 'rematch 3rd').sum()
    LOG.info("Rematch types: rematch 2nd=%s, rematch 3rd=%s, not rematch=%s", n2, n3, len(df) - n2 - n3)
    return df


def _log_rematch_home_away_flip(df: pd.DataFrame) -> None:
    """Log that we catch rematches when home/away flips (same pair, different venue). Requires home_1st_meeting."""
    if 'home_1st_meeting' not in df.columns or df.empty:
        return
    rematch2 = df[df['rematch_type'] == 'rematch 2nd']
    if rematch2.empty:
        return
    same_home = (rematch2['HOME_TEAM'] == rematch2['home_1st_meeting']).sum()
    flipped = (rematch2['HOME_TEAM'] != rematch2['home_1st_meeting']).sum()
    LOG.info("Rematch 2nd home/away: %s same venue as game 1, %s flipped (away→home or vice versa)", same_home, flipped)


# -----------------------------------------------------------------------------
# Focal team (loser of prior meeting(s))
# -----------------------------------------------------------------------------


def winner_of_game(row) -> str:
    """Return the team that won this game (higher score)."""
    if row['HOME_SCORE'] > row['AWAY_SCORE']:
        return row['HOME_TEAM']
    return row['AWAY_TEAM']


def loser_of_game(row) -> str:
    """Return the team that lost this game (lower score)."""
    if row['HOME_SCORE'] < row['AWAY_SCORE']:
        return row['HOME_TEAM']
    return row['AWAY_TEAM']


def add_pair_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each row, add spread/winner/date for 1st, 2nd, 3rd meeting of that pair.
    - spread_1st_meeting, spread_2nd_meeting, spread_3rd_meeting (null if no 3rd game yet)
    - winner_1st_meeting, winner_2nd_meeting, winner_3rd_meeting
    - date_1st_meeting, date_2nd_meeting, date_3rd_meeting
    """
    rows = []
    for pair, grp in df.groupby('pair', sort=False):
        grp = grp.sort_values('GAME_DATE').reset_index(drop=True)
        game1 = grp[grp['meeting_number'] == 1]
        game2 = grp[grp['meeting_number'] == 2]
        game3 = grp[grp['meeting_number'] == 3]
        g1 = game1.iloc[0] if len(game1) else None
        g2 = game2.iloc[0] if len(game2) else None
        g3 = game3.iloc[0] if len(game3) else None
        spread_1 = g1['consensus_spread'] if g1 is not None and pd.notna(g1.get('consensus_spread')) else None
        spread_2 = g2['consensus_spread'] if g2 is not None and pd.notna(g2.get('consensus_spread')) else None
        spread_3 = g3['consensus_spread'] if g3 is not None and pd.notna(g3.get('consensus_spread')) else None
        winner_1 = winner_of_game(g1) if g1 is not None else None
        winner_2 = winner_of_game(g2) if g2 is not None else None
        winner_3 = winner_of_game(g3) if g3 is not None else None
        date_1 = g1['GAME_DATE'] if g1 is not None else None
        date_2 = g2['GAME_DATE'] if g2 is not None else None
        date_3 = g3['GAME_DATE'] if g3 is not None else None
        home_1 = g1['HOME_TEAM'] if g1 is not None else None
        away_1 = g1['AWAY_TEAM'] if g1 is not None else None
        for _, row in grp.iterrows():
            r = row.to_dict()
            r['spread_1st_meeting'] = spread_1
            r['spread_2nd_meeting'] = spread_2
            r['spread_3rd_meeting'] = spread_3
            r['winner_1st_meeting'] = winner_1
            r['winner_2nd_meeting'] = winner_2
            r['winner_3rd_meeting'] = winner_3
            r['date_1st_meeting'] = date_1
            r['date_2nd_meeting'] = date_2
            r['date_3rd_meeting'] = date_3
            r['home_1st_meeting'] = home_1
            r['away_1st_meeting'] = away_1
            rows.append(r)
    return pd.DataFrame(rows)


def add_focal_and_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each row with meeting_number >= 2, set focal_team (loser of 1st; for 3rd, loser of both 1st and 2nd).
    Add focal_su_win, focal_ats_cover (bool; NaN when no spread), focal_was_home.
    """
    df = df.copy()
    rows = []
    for pair, grp in df.groupby('pair', sort=False):
        grp = grp.sort_values('GAME_DATE').reset_index(drop=True)
        for i, row in grp.iterrows():
            r = row.to_dict()
            r['focal_team'] = None
            r['focal_su_win'] = None
            r['focal_ats_cover'] = None
            r['focal_was_home'] = None
            if row['meeting_number'] == 2:
                first = grp[grp['meeting_number'] == 1].iloc[0]
                r['focal_team'] = loser_of_game(first)
            elif row['meeting_number'] == 3:
                first = grp[grp['meeting_number'] == 1].iloc[0]
                second = grp[grp['meeting_number'] == 2].iloc[0]
                loser1 = loser_of_game(first)
                loser2 = loser_of_game(second)
                if loser1 == loser2:
                    r['focal_team'] = loser1
                else:
                    r['focal_team'] = None
            if r['focal_team'] is not None:
                focal = r['focal_team']
                r['focal_was_home'] = focal == row['HOME_TEAM']
                focal_score = row['HOME_SCORE'] if r['focal_was_home'] else row['AWAY_SCORE']
                opp_score = row['AWAY_SCORE'] if r['focal_was_home'] else row['HOME_SCORE']
                r['focal_su_win'] = focal_score > opp_score
                spread = row.get('consensus_spread')
                if pd.notna(spread):
                    home_margin = row['HOME_SCORE'] - row['AWAY_SCORE']
                    if r['focal_was_home']:
                        diff = home_margin - spread
                    else:
                        diff = (-home_margin) + spread
                    r['focal_ats_cover'] = diff > 0 if diff != 0 else None
                else:
                    r['focal_ats_cover'] = None
            rows.append(r)
    out = pd.DataFrame(rows)
    return out


def _assert_rematch_dates_in_season(df: pd.DataFrame, season: str | None = None) -> None:
    """
    Assert that for every rematch 2nd/3rd row, date_1st_meeting (and date_2nd_meeting for 3rd)
    and the game date fall within the season's date range. Ensures we never mix seasons.
    Pass season= when df has no 'season' column (single-season run).
    """
    if df.empty:
        return
    rematch = df[df['rematch_type'].isin(('rematch 2nd', 'rematch 3rd'))]
    if rematch.empty:
        return
    LOG.info("Rematch date check: verifying 1st/2nd/3rd meeting dates are within season range (%s rows)", len(rematch))
    if 'season' not in df.columns and season is None:
        return
    start_dates = {}
    end_dates = {}
    seasons_to_check = {season} if season is not None else set(rematch['season'].dropna().unique())
    for s in seasons_to_check:
        if s not in SEASON_DATES:
            raise ValueError(f"Season {s} not in SEASON_DATES")
        start_dates[s], end_dates[s] = SEASON_DATES[s]
        start_dates[s] = pd.to_datetime(start_dates[s]).date()
        end_dates[s] = pd.to_datetime(end_dates[s]).date()

    def to_date(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        if hasattr(x, 'date'):
            return x.date() if hasattr(x, 'date') else x
        return pd.to_datetime(x).date()

    bad = []
    for idx, row in rematch.iterrows():
        s = row['season'] if season is None else season
        if s not in start_dates:
            continue
        start, end = start_dates[s], end_dates[s]
        d1 = to_date(row.get('date_1st_meeting'))
        game_date = to_date(row.get('GAME_DATE'))
        if d1 is None or game_date is None:
            continue
        if not (start <= d1 <= end):
            bad.append((idx, s, 'date_1st_meeting', d1, start, end))
        if not (start <= game_date <= end):
            bad.append((idx, s, 'GAME_DATE', game_date, start, end))
        if row['rematch_type'] == 'rematch 3rd':
            d2 = to_date(row.get('date_2nd_meeting'))
            if d2 is not None and not (start <= d2 <= end):
                bad.append((idx, s, 'date_2nd_meeting', d2, start, end))

    if bad:
        msg = "Rematch dates outside season range (season start–end):\n"
        for idx, s, field, val, start, end in bad[:10]:
            msg += f"  row {idx}: {field}={val} not in [{start}, {end}] (season {s})\n"
        if len(bad) > 10:
            msg += f"  ... and {len(bad) - 10} more\n"
        raise AssertionError(msg)
    LOG.info("Rematch date check: all 1st/2nd/3rd meeting dates fall within season range")
    return


# -----------------------------------------------------------------------------
# Aggregate and report
# -----------------------------------------------------------------------------


def aggregate_rematch_stats(df: pd.DataFrame) -> dict:
    """Compute SU and ATS records for rematch 2nd and rematch 3rd (focal team)."""
    second = df[df['rematch_type'] == 'rematch 2nd'].dropna(subset=['focal_team'])
    third = df[df['rematch_type'] == 'rematch 3rd'].dropna(subset=['focal_team'])

    def stats(sub):
        n = len(sub)
        su_w = sub['focal_su_win'].sum()
        su_l = n - su_w
        with_spread = sub['focal_ats_cover'].notna()
        ats_sub = sub.loc[with_spread]
        ats_n = len(ats_sub)
        ats_w = ats_sub['focal_ats_cover'].eq(True).sum()
        ats_l = ats_sub['focal_ats_cover'].eq(False).sum()
        ats_p = ats_sub['focal_ats_cover'].isna().sum()
        ats_resolved = ats_n - ats_p
        return {
            'n': n,
            'su_w': int(su_w),
            'su_l': int(su_l),
            'su_pct': 100 * su_w / n if n else 0,
            'ats_n': ats_n,
            'ats_w': int(ats_w),
            'ats_l': int(ats_l),
            'ats_p': int(ats_p),
            'ats_pct': 100 * ats_w / ats_resolved if ats_resolved else 0,
        }

    return {
        'rematch_2nd': stats(second) if len(second) else None,
        'rematch_3rd': stats(third) if len(third) else None,
    }


def print_summary(agg: dict, conference: str, season: str):
    """Print summary to stdout."""
    print()
    print("=" * 60)
    print(f"NCAAB CONFERENCE REMATCH: {conference} ({season})")
    print("=" * 60)
    for key, label in [('rematch_2nd', '2nd game (lost 1st)'), ('rematch_3rd', '3rd game (lost 1st & 2nd)')]:
        s = agg.get(key)
        if s is None:
            print(f"\n{label}: N=0")
            continue
        print(f"\n{label}: N={s['n']}")
        print(f"  SU:  {s['su_w']}-{s['su_l']} ({s['su_pct']:.1f}%)")
        p_str = f"-{s['ats_p']}" if s['ats_p'] else ""
        print(f"  ATS: {s['ats_w']}-{s['ats_l']}{p_str} (cover pct: {s['ats_pct']:.1f}%) [n with line: {s['ats_n']}]")
    print()


# -----------------------------------------------------------------------------
# Debug: per-conference date listing
# -----------------------------------------------------------------------------


def _log_matchup_quality_by_conference(combined: pd.DataFrame) -> None:
    """Log per-conference: total games, how many have a line (consensus_spread), pct."""
    rows = []
    for conf in combined['conference'].dropna().unique():
        sub = combined[combined['conference'] == conf]
        n = len(sub)
        with_line = sub['consensus_spread'].notna().sum()
        pct = 100.0 * with_line / n if n else 0
        rows.append({'conference': conf, 'n_games': n, 'n_with_line': with_line, 'line_pct': pct})
    df = pd.DataFrame(rows).sort_values('n_games', ascending=False)
    LOG.info("Matchup quality by conference (games with a line):")
    for _, r in df.iterrows():
        LOG.info("   %s: %s games, %s with line (%.1f%%)", r['conference'], r['n_games'], r['n_with_line'], r['line_pct'])


def _run_duckdb_queries(df: pd.DataFrame, label: str) -> None:
    """Run DuckDB sense-check queries on the selected rematch subset (in-memory, duckbox-style). label used in headers."""
    if df.empty:
        return
    con = duckdb.connect(':memory:')
    con.register('rematch', df)

    def run(title: str, sql: str) -> None:
        try:
            out = con.execute(sql).fetchdf()
            print(f"\n--- DuckDB: {title} ({label}) ---")
            print(out.to_string(index=False))
        except Exception as e:
            LOG.warning("DuckDB query failed (%s): %s", title, e)

    run("Counts by rematch_type", "SELECT rematch_type, COUNT(*) AS n FROM rematch GROUP BY rematch_type ORDER BY rematch_type")
    run(
        "Rematch 2nd: N total vs N with both lines (explains n=81 vs 119)",
        """
        SELECT
          COUNT(*) AS rematch_2nd_total,
          SUM(CASE WHEN spread_1st_meeting IS NOT NULL AND spread_2nd_meeting IS NOT NULL THEN 1 ELSE 0 END) AS n_with_both_lines
        FROM rematch WHERE rematch_type = 'rematch 2nd'
        """,
    )
    run(
        "SU/ATS for rematch 2nd (focal = loser of game 1)",
        """
        SELECT COUNT(*) AS n,
          SUM(CASE WHEN focal_su_win THEN 1 ELSE 0 END) AS su_w,
          SUM(CASE WHEN NOT focal_su_win THEN 1 ELSE 0 END) AS su_l,
          ROUND(100.0 * SUM(CASE WHEN focal_su_win THEN 1 ELSE 0 END) / COUNT(*), 1) AS su_win_pct,
          SUM(CASE WHEN focal_ats_cover = true THEN 1 ELSE 0 END) AS ats_w,
          SUM(CASE WHEN focal_ats_cover = false THEN 1 ELSE 0 END) AS ats_l,
          ROUND(100.0 * SUM(CASE WHEN focal_ats_cover = true THEN 1 ELSE 0 END) / NULLIF(SUM(CASE WHEN focal_ats_cover IS NOT NULL THEN 1 ELSE 0 END), 0), 1) AS ats_cover_pct
        FROM rematch
        WHERE rematch_type = 'rematch 2nd' AND spread_1st_meeting IS NOT NULL AND spread_2nd_meeting IS NOT NULL
        """,
    )
    run(
        "Avg spread change 1→2 (home perspective)",
        """
        SELECT ROUND(AVG(spread_2nd_meeting - spread_1st_meeting), 2) AS avg_spread_change_1_to_2,
          ROUND(MIN(spread_2nd_meeting - spread_1st_meeting), 2) AS min_change,
          ROUND(MAX(spread_2nd_meeting - spread_1st_meeting), 2) AS max_change, COUNT(*) AS n
        FROM rematch
        WHERE rematch_type = 'rematch 2nd' AND spread_1st_meeting IS NOT NULL AND spread_2nd_meeting IS NOT NULL
        """,
    )
    if 'home_1st_meeting' in df.columns:
        # CTE r: focal spreads + focal margin and margin vs spread for rematch game (positive = covered by that much)
        _r_cte = """
            WITH r AS (
              SELECT *,
                CASE WHEN focal_team = home_1st_meeting THEN spread_1st_meeting ELSE -spread_1st_meeting END AS focal_spread_1st,
                CASE WHEN focal_was_home THEN spread_2nd_meeting ELSE -spread_2nd_meeting END AS focal_spread_2nd,
                CASE WHEN focal_was_home THEN (HOME_SCORE - AWAY_SCORE) ELSE (AWAY_SCORE - HOME_SCORE) END AS focal_margin,
                (CASE WHEN focal_was_home THEN (HOME_SCORE - AWAY_SCORE) ELSE (AWAY_SCORE - HOME_SCORE) END)
                  + (CASE WHEN focal_was_home THEN spread_2nd_meeting ELSE -spread_2nd_meeting END) AS margin_vs_spread
              FROM rematch
              WHERE rematch_type = 'rematch 2nd' AND spread_1st_meeting IS NOT NULL AND spread_2nd_meeting IS NOT NULL AND home_1st_meeting IS NOT NULL
            )
        """
        _r_cte_ats = _r_cte.replace(
            "AND home_1st_meeting IS NOT NULL",
            "AND home_1st_meeting IS NOT NULL AND focal_ats_cover IS NOT NULL",
        )
        _metrics = "ROUND(100.0 * SUM(CASE WHEN focal_su_win THEN 1 ELSE 0 END) / COUNT(*), 1) AS su_win_pct, ROUND(100.0 * SUM(CASE WHEN focal_ats_cover = true THEN 1 ELSE 0 END) / NULLIF(SUM(CASE WHEN focal_ats_cover IS NOT NULL THEN 1 ELSE 0 END), 0), 1) AS ats_cover_pct, ROUND(AVG(margin_vs_spread), 2) AS avg_margin_vs_spread"
        run(
            "Avg line move toward focal (positive = in focal favor g1→g2)",
            _r_cte
            + """
            SELECT ROUND(AVG(focal_spread_1st - focal_spread_2nd), 2) AS avg_line_move_toward_focal, COUNT(*) AS n,
              """ + _metrics + """ FROM r
            """,
        )
        run(
            "Avg line move toward focal by focal W/L game 2",
            _r_cte
            + """
            SELECT focal_su_win AS focal_won_game2, ROUND(AVG(focal_spread_1st - focal_spread_2nd), 2) AS avg_line_move_toward_focal, COUNT(*) AS n,
              """ + _metrics + """
            FROM r GROUP BY focal_su_win ORDER BY focal_su_win
            """,
        )
        run(
            "Avg line move toward focal by focal cover/no-cover game 2",
            _r_cte_ats
            + """
            SELECT focal_ats_cover AS focal_covered_game2, ROUND(AVG(focal_spread_1st - focal_spread_2nd), 2) AS avg_line_move_toward_focal, COUNT(*) AS n,
              """ + _metrics + """
            FROM r GROUP BY focal_ats_cover ORDER BY focal_ats_cover
            """,
        )
        run(
            "Avg line move toward focal by favored in game 2 (Y/N)",
            _r_cte
            + """
            SELECT (focal_spread_2nd < 0) AS focal_favored_game2, ROUND(AVG(focal_spread_1st - focal_spread_2nd), 2) AS avg_line_move_toward_focal, COUNT(*) AS n,
              """ + _metrics + """
            FROM r GROUP BY (focal_spread_2nd < 0) ORDER BY focal_favored_game2
            """,
        )
        run(
            "Avg line move toward focal by dog g1 but fav g2 (Y/N)",
            _r_cte
            + """
            SELECT (focal_spread_1st > 0 AND focal_spread_2nd < 0) AS dog_g1_fav_g2, ROUND(AVG(focal_spread_1st - focal_spread_2nd), 2) AS avg_line_move_toward_focal, COUNT(*) AS n,
              """ + _metrics + """
            FROM r GROUP BY (focal_spread_1st > 0 AND focal_spread_2nd < 0) ORDER BY dog_g1_fav_g2
            """,
        )
    con.close()


def _all_games_summary_row(df: pd.DataFrame, season_label: str) -> dict:
    """
    One row of metrics for the all-games summary table: rematch 2nd and rematch 3rd.
    Short column names so table fits on one screen. r2 = 2nd meeting (focal = loser g1),
    r3 = 3rd meeting (focal = loser of both g1 and g2).
    """
    row = {'season': season_label}

    # ----- Rematch 2nd -----
    r2 = df[df['rematch_type'] == 'rematch 2nd']
    row['r2_tot'] = len(r2)
    with_both = r2['spread_1st_meeting'].notna() & r2['spread_2nd_meeting'].notna()
    row['r2_n'] = with_both.sum()
    r = r2[with_both]
    if 'home_1st_meeting' in df.columns:
        r = r[r['home_1st_meeting'].notna()]
    if r.empty:
        row['r2_su'] = None
        row['r2_ats'] = None
        row['r2_mrg'] = None
        row['r2_move'] = None
    else:
        n = len(r)
        row['r2_su'] = round(100.0 * r['focal_su_win'].sum() / n, 1)
        ats_valid = r['focal_ats_cover'].notna()
        row['r2_ats'] = round(100.0 * (r.loc[ats_valid, 'focal_ats_cover'] == True).sum() / ats_valid.sum(), 1) if ats_valid.sum() else None
        if 'home_1st_meeting' in r.columns:
            focal_margin = np.where(r['focal_was_home'], r['HOME_SCORE'] - r['AWAY_SCORE'], r['AWAY_SCORE'] - r['HOME_SCORE'])
            focal_spread_2nd = np.where(r['focal_was_home'], r['spread_2nd_meeting'], -r['spread_2nd_meeting'])
            row['r2_mrg'] = round(float((focal_margin + focal_spread_2nd).mean()), 2)
            focal_spread_1st = np.where(r['focal_team'] == r['home_1st_meeting'], r['spread_1st_meeting'], -r['spread_1st_meeting'])
            row['r2_move'] = round(float((focal_spread_1st - focal_spread_2nd).mean()), 2)
        else:
            row['r2_mrg'] = None
            row['r2_move'] = None

    # ----- Rematch 3rd -----
    r3 = df[df['rematch_type'] == 'rematch 3rd']
    row['r3_tot'] = len(r3)
    with_all3 = r3['spread_1st_meeting'].notna() & r3['spread_2nd_meeting'].notna() & r3['spread_3rd_meeting'].notna()
    row['r3_n'] = with_all3.sum()
    r3_sub = r3[with_all3].dropna(subset=['focal_team'])
    if r3_sub.empty:
        row['r3_su'] = None
        row['r3_ats'] = None
        row['r3_mrg'] = None
    else:
        n3 = len(r3_sub)
        row['r3_su'] = round(100.0 * r3_sub['focal_su_win'].sum() / n3, 1)
        ats3 = r3_sub['focal_ats_cover'].notna()
        row['r3_ats'] = round(100.0 * (r3_sub.loc[ats3, 'focal_ats_cover'] == True).sum() / ats3.sum(), 1) if ats3.sum() else None
        focal_margin_3 = np.where(r3_sub['focal_was_home'], r3_sub['HOME_SCORE'] - r3_sub['AWAY_SCORE'], r3_sub['AWAY_SCORE'] - r3_sub['HOME_SCORE'])
        focal_spread_3rd = np.where(r3_sub['focal_was_home'], r3_sub['spread_3rd_meeting'], -r3_sub['spread_3rd_meeting'])
        row['r3_mrg'] = round(float((focal_margin_3 + focal_spread_3rd).mean()), 2)

    return row


def _log_conference_dates(conf_name: str, conf_df: pd.DataFrame) -> None:
    """Log each date that has games in this conference (for debugging low rematch counts)."""
    dates = pd.to_datetime(conf_df['GAME_DATE']).dt.date
    date_counts = dates.value_counts().sort_index()
    date_list = [f"{d}({n})" for d, n in date_counts.items()]
    n_dates = len(date_list)
    if n_dates <= 25:
        LOG.info("%s: %s games on %s dates: %s", conf_name, len(conf_df), n_dates, ", ".join(str(x) for x in date_list))
    else:
        head = ", ".join(str(x) for x in date_list[:12])
        tail = ", ".join(str(x) for x in date_list[-12:])
        LOG.info("%s: %s games on %s dates (first 12): %s", conf_name, len(conf_df), n_dates, head)
        LOG.info("%s: (last 12): %s", conf_name, tail)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="NCAAB conference rematch: 2nd/3rd meeting SU & ATS for losing team."
    )
    parser.add_argument(
        '--conference',
        type=str,
        default='Big Ten',
        help='Single conference (used if --conferences not set)',
    )
    parser.add_argument(
        '--conferences',
        type=str,
        default=None,
        help='Comma-separated conferences or "all". Overrides --conference. Output CSV has conference column.',
    )
    parser.add_argument(
        '--season',
        type=str,
        default='2025-26',
        help='Season YYYY-YY',
    )
    parser.add_argument(
        '--all-games',
        action='store_true',
        help='No conference filter; analyze all rematches (any pair that plays 2+ times). Data back to 2020-21.',
    )
    parser.add_argument(
        '--seasons',
        type=str,
        default=None,
        help='Comma-separated seasons for --all-games (e.g. 2020-21,2021-22,2024-25). If set, runs each season and prints a summary table with row per season + "all".',
    )
    parser.add_argument(
        '--csv',
        action='store_true',
        help='Write game-level CSV to data/04_output/',
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Verbose logging (default: True)',
    )
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    et_date = get_today_et()

    if args.all_games:
        if args.seasons:
            seasons_list = [s.strip() for s in args.seasons.split(',') if s.strip()]
        else:
            seasons_list = [args.season]
        LOG.info("All-games mode: no conference filter, seasons=%s", seasons_list)
        LOG.info("Cache date (ET): %s", et_date)

        all_dfs = []
        for season in seasons_list:
            joined_df = load_joined_from_cache_or_s3(season, et_date)
            LOG.info("Season %s: %s games", season, len(joined_df))
            df = add_meeting_number_and_rematch_type(joined_df)
            df = add_pair_context(df)
            df = add_focal_and_results(df)
            df['season'] = season
            all_dfs.append(df)

        combined = pd.concat(all_dfs, ignore_index=True)
        _log_rematch_home_away_flip(combined)
        _assert_rematch_dates_in_season(combined)

        if len(seasons_list) == 1:
            df = all_dfs[0]
            one_season = seasons_list[0]
            agg = aggregate_rematch_stats(df)
            print("=" * 60)
            print(f"NCAAB REMATCH: all games ({one_season})")
            print("=" * 60)
            print_summary(agg, "all games", one_season)
            if args.csv:
                out_dir = PROJECT_ROOT / 'data' / '04_output'
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f'ncaab_rematch_all_games_{one_season}_{et_date}.csv'
                df.to_csv(out_path, index=False)
                LOG.info("Wrote CSV: %s", out_path)
                print(f"CSV written: {out_path}")
            _run_duckdb_queries(df, "all games")
            return

        # Multi-season: build summary table (one row per season + 'all')
        rows = []
        for season in seasons_list:
            sub = combined[combined['season'] == season]
            rows.append(_all_games_summary_row(sub, season))
        rows.append(_all_games_summary_row(combined, 'all'))
        summary_df = pd.DataFrame(rows)
        print()
        print("=" * 80)
        print("NCAAB REMATCH (all games): summary by season")
        print("=" * 80)
        print(summary_df.to_string(index=False))
        print()
        if args.csv:
            out_dir = PROJECT_ROOT / 'data' / '04_output'
            out_dir.mkdir(parents=True, exist_ok=True)
            combined_path = out_dir / f'ncaab_rematch_all_games_multi_{et_date}.csv'
            combined.to_csv(combined_path, index=False)
            LOG.info("Wrote CSV: %s", combined_path)
            summary_path = out_dir / f'ncaab_rematch_all_games_summary_{et_date}.csv'
            summary_df.to_csv(summary_path, index=False)
            LOG.info("Wrote summary: %s", summary_path)
            print(f"CSV written: {combined_path}")
            print(f"Summary table: {summary_path}")
        return

    if args.conferences is not None:
        if args.conferences.strip().lower() == 'all':
            conferences = ALL_CONFERENCES
        else:
            conferences = [c.strip() for c in args.conferences.split(',') if c.strip()]
        LOG.info("Multi-conference: season=%s, conferences=%s", args.season, len(conferences))
        LOG.info("Cache date (ET): %s", et_date)
        joined_df = load_joined_from_cache_or_s3(args.season, et_date)
        all_dfs = []
        for conf_name in conferences:
            conf_df = filter_to_conference(joined_df, conf_name, allow_empty=True)
            if conf_df.empty:
                LOG.info("Skipping %s (no games)", conf_name)
                continue
            # Log each date with game count for debugging (why so few rematches, etc.)
            _log_conference_dates(conf_name, conf_df)
            conf_df = add_meeting_number_and_rematch_type(conf_df)
            conf_df = add_pair_context(conf_df)
            conf_df = add_focal_and_results(conf_df)
            conf_df['conference'] = conf_name
            all_dfs.append(conf_df)
        if not all_dfs:
            LOG.warning("No conference had games; nothing to write.")
            return
        combined = pd.concat(all_dfs, ignore_index=True)
        _log_rematch_home_away_flip(combined)
        _assert_rematch_dates_in_season(combined, season=args.season)
        _log_matchup_quality_by_conference(combined)
        for conf_name in conferences:
            sub = combined[combined['conference'] == conf_name]
            if sub.empty:
                continue
            agg = aggregate_rematch_stats(sub)
            print_summary(agg, conf_name, args.season)
        agg_all = aggregate_rematch_stats(combined)
        print("=" * 60)
        print("OVERALL (all conferences combined)")
        print("=" * 60)
        print_summary(agg_all, "(all)", args.season)
        if args.csv:
            out_dir = PROJECT_ROOT / 'data' / '04_output'
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f'ncaab_rematch_all_conferences_{args.season}_{et_date}.csv'
            combined.to_csv(out_path, index=False)
            LOG.info("Wrote CSV: %s", out_path)
            print(f"CSV written: {out_path} (filter by conference, rematch_type)")
        _run_duckdb_queries(combined, "all conferences")
        return

    LOG.info("Starting rematch analysis: conference=%s, season=%s", args.conference, args.season)
    LOG.info("Cache date (ET): %s", et_date)

    joined_df = load_joined_from_cache_or_s3(args.season, et_date)
    conf_df = filter_to_conference(joined_df, args.conference)
    conf_df = add_meeting_number_and_rematch_type(conf_df)
    conf_df = add_pair_context(conf_df)
    conf_df = add_focal_and_results(conf_df)
    _log_rematch_home_away_flip(conf_df)
    _assert_rematch_dates_in_season(conf_df, season=args.season)

    agg = aggregate_rematch_stats(conf_df)
    print_summary(agg, args.conference, args.season)

    # Examples: last 5 of each rematch type
    for rtype in ['rematch 2nd', 'rematch 3rd']:
        sub = conf_df[conf_df['rematch_type'] == rtype].tail(5)
        if sub.empty:
            continue
        print(f"Example games ({rtype}), last 5:")
        for _, row in sub.iterrows():
            focal = row.get('focal_team') or '—'
            su = 'W' if row.get('focal_su_win') is True else ('L' if row.get('focal_su_win') is False else '—')
            ats = 'C' if row.get('focal_ats_cover') is True else 'L' if row.get('focal_ats_cover') is False else '—'
            print(f"  {row['GAME_DATE']} {row['pair'][0]} vs {row['pair'][1]} | focal={focal} SU={su} ATS={ats}")
        print()

    if args.csv:
        out_dir = PROJECT_ROOT / 'data' / '04_output'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f'ncaab_rematch_{args.conference.replace(" ", "_")}_{args.season}_{et_date}.csv'
        conf_df.to_csv(out_path, index=False)
        LOG.info("Wrote CSV: %s", out_path)
        print(f"CSV written: {out_path} (filter by rematch_type for sense-check)")

    # Run DuckDB queries on the given conference/season subset
    _run_duckdb_queries(conf_df, args.conference)


if __name__ == '__main__':
    main()
