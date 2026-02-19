"""
Debug why outcomes (n=59) + lines (n=59) join does not yield 59 matched rows.

Shows:
1. BEFORE normalization: join on raw names (outcome HOME_TEAM = line home_team, etc.). Stats.
2. AFTER normalization: join on (date, normalized home_key, away_key). Stats.
3. Mismatch table: for each game that didn't match after norm, show outcome names (ESPN) vs
   line raw -> normalized, and which side (home/away) differs.

Run from repo root:
    python tmp/debug_ncaab_outcomes_lines_join.py --date 2026-02-18
    python tmp/debug_ncaab_outcomes_lines_join.py --date all --season 2025-26

Workflow to catch and add missing team name mappings:
    Step 1: python tmp/debug_ncaab_outcomes_lines_join.py --date all --season 2025-26
            -> writes suggested Odds API -> ESPN by date to ~/Downloads/tmp/ncaab_teams_to_add_to_mapping.json
    Step 2: Review JSON (and/or spot-check single dates). Confirm mappings are correct (no flips).
    Step 3: Add confirmed entries to ODDS_API_ESPN_OVERRIDES in tmp/build_comprehensive_ncaab_mapping.py
    Step 4: python3 tmp/build_comprehensive_ncaab_mapping.py
            -> idempotent: uses S3 + patterns + overrides only; overwrites src/ncaab_team_name_mapping.py

Verification after mapping changes:
    1. python3 tmp/build_comprehensive_ncaab_mapping.py
    2. python tmp/debug_ncaab_outcomes_lines_join.py --date 2026-02-18
       -> expect "Matched (outcome row has consensus_spread_home): 59 / 59" and "Unmatched: 0"
    3. Run on other dates; 100% match on a given date depends on that day's games and
       whether all ESPN/Odds API name variants are in the mapping.
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Project root for src imports
def _find_project_root():
    current = Path.cwd()
    while current != current.parent:
        if (current / ".gitignore").exists():
            return current
        current = current.parent
    return Path.cwd()


ROOT = _find_project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ncaab_team_name_mapping import ODDS_API_TO_ESPN_NCAAB
from src.season_utils import get_season_dates

ODDS_TO_ESPN_LOWER = {k.lower(): v for k, v in ODDS_API_TO_ESPN_NCAAB.items()}

# ESPN display names that differ from our mapping value; use to find same-game line when both sides differ
ESPN_TO_NORMALIZED_KEY = {
    "American University Eagles": "American Eagles",
    "Army Black Knights": "Army Knights",
    "Loyola Chicago Ramblers": "Loyola (Chi) Ramblers",
    "Loyola Maryland Greyhounds": "Loyola (MD) Greyhounds",
    "Purdue Fort Wayne Mastodons": "Fort Wayne Mastodons",
    "Sam Houston Bearkats": "Sam Houston State Bearkats",
    "Seattle U Redhawks": "Seattle Redhawks",
}


def _odds_to_espn(name: str) -> str:
    if pd.isna(name):
        return ""
    key = str(name).lower().strip()
    return ODDS_TO_ESPN_LOWER.get(key, name)


def load_outcomes(path: str) -> pd.DataFrame:
    """Path: local path or s3://bucket/key."""
    if path.startswith("s3://"):
        import boto3
        from io import BytesIO
        parts = path.replace("s3://", "").split("/", 1)
        bucket, key = parts[0], parts[1]
        s3 = boto3.client("s3")
        r = s3.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(BytesIO(r["Body"].read()))
    else:
        df = pd.read_csv(path)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.date
    return df


def load_lines(path: str) -> pd.DataFrame:
    if path.startswith("s3://"):
        import boto3
        from io import BytesIO
        parts = path.replace("s3://", "").split("/", 1)
        bucket, key = parts[0], parts[1]
        s3 = boto3.client("s3")
        r = s3.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(BytesIO(r["Body"].read()))
    else:
        df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def _season_date_list(season: str) -> list[str]:
    """Return list of YYYY-MM-DD dates from NCAAB season_start through tournament_end."""
    dates_config = get_season_dates("ncaab", season)
    start = datetime.strptime(dates_config["season_start"], "%Y-%m-%d").date()
    end = datetime.strptime(dates_config["tournament_end"], "%Y-%m-%d").date()
    out = []
    d = start
    while d <= end:
        out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out


def join_outcomes_lines(outcomes: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    """Replicate Lambda join logic. Caller should not pass empty outcomes."""
    if outcomes.empty:
        return outcomes
    outcomes = outcomes.copy()
    outcomes["date_key"] = outcomes["GAME_DATE"]
    outcomes["home_key"] = outcomes["HOME_TEAM"].astype(str).str.strip()
    outcomes["away_key"] = outcomes["AWAY_TEAM"].astype(str).str.strip()

    if lines.empty:
        outcomes["consensus_spread_home"] = float("nan")
        return outcomes

    lines = lines.copy()
    lines["date_key"] = pd.to_datetime(lines["date"]).dt.date
    lines["home_key"] = lines["home_team"].apply(lambda t: _odds_to_espn(t).strip())
    lines["away_key"] = lines["away_team"].apply(lambda t: _odds_to_espn(t).strip())
    lines = lines[["date_key", "home_key", "away_key", "consensus_spread"]].drop_duplicates(
        subset=["date_key", "home_key", "away_key"]
    )
    lines = lines.rename(columns={"consensus_spread": "consensus_spread_home"})

    joined = outcomes.merge(
        lines,
        on=["date_key", "home_key", "away_key"],
        how="left",
        suffixes=("", "_line"),
    )
    if "_line" in joined.columns:
        joined = joined[[c for c in joined.columns if not c.endswith("_line")]]
    return joined


def run_for_date(
    date_str: str,
    bucket: str,
    outcomes_path: str | None = None,
    lines_path: str | None = None,
) -> dict | None:
    """
    Load outcomes and lines for one date, run join, collect suggested_additions from mismatches.
    Returns None if data cannot be loaded (e.g. missing S3 object); otherwise a dict with
    n_out, n_lin, n_before, n_after, suggested_additions, joined, missing, lines_with_keys.
    """
    op = outcomes_path or f"s3://{bucket}/data/01_input/historical_game_results/{date_str}.csv"
    lp = lines_path or f"s3://{bucket}/data/01_input/the-odds-api/ncaab/game_lines/{date_str}.csv"
    try:
        outcomes = load_outcomes(op)
        lines = load_lines(lp)
    except Exception:
        return None

    n_out = len(outcomes)
    n_lin = len(lines)
    outcomes = outcomes.copy()
    lines = lines.copy()

    if outcomes.empty:
        n_before = 0
        if lines.empty:
            lines_with_keys = pd.DataFrame()
        else:
            lines_with_keys = lines.copy()
            lines_with_keys["date_key"] = pd.to_datetime(lines_with_keys["date"]).dt.date
            lines_with_keys["home_key"] = lines_with_keys["home_team"].apply(
                lambda t: _odds_to_espn(t).strip()
            )
            lines_with_keys["away_key"] = lines_with_keys["away_team"].apply(
                lambda t: _odds_to_espn(t).strip()
            )
        joined = outcomes.copy()
        joined["date_key"] = pd.Series(dtype=object)
        joined["home_key"] = pd.Series(dtype=object)
        joined["away_key"] = pd.Series(dtype=object)
        joined["consensus_spread_home"] = pd.Series(dtype=float)
        return {
            "n_out": n_out,
            "n_lin": n_lin,
            "n_before": n_before,
            "n_after": 0,
            "suggested_additions": {},
            "joined": joined,
            "missing": joined,
            "lines_with_keys": lines_with_keys,
        }

    outcomes["_date"] = outcomes["GAME_DATE"].astype(str)
    outcomes["_home_raw"] = outcomes["HOME_TEAM"].astype(str).str.strip()
    outcomes["_away_raw"] = outcomes["AWAY_TEAM"].astype(str).str.strip()
    lines["_date"] = pd.to_datetime(lines["date"]).dt.date.astype(str)
    lines["_home_raw"] = lines["home_team"].astype(str).str.strip()
    lines["_away_raw"] = lines["away_team"].astype(str).str.strip()
    before_joined = outcomes.merge(
        lines[["_date", "_home_raw", "_away_raw", "consensus_spread"]],
        on=["_date", "_home_raw", "_away_raw"],
        how="left",
    )
    n_before = before_joined["consensus_spread"].notna().sum()
    outcomes.drop(columns=["_date", "_home_raw", "_away_raw"], inplace=True)
    lines.drop(columns=["_date", "_home_raw", "_away_raw"], inplace=True)

    lines_with_keys = lines.copy()
    lines_with_keys["date_key"] = pd.to_datetime(lines_with_keys["date"]).dt.date
    lines_with_keys["home_key"] = lines_with_keys["home_team"].apply(lambda t: _odds_to_espn(t).strip())
    lines_with_keys["away_key"] = lines_with_keys["away_team"].apply(lambda t: _odds_to_espn(t).strip())

    joined = join_outcomes_lines(outcomes, lines)
    n_after = joined["consensus_spread_home"].notna().sum()
    missing = joined[joined["consensus_spread_home"].isna()]

    # Only record suggestions when we're sure the outcome and line row are the same game.
    # "One side matches" can pair outcome with the wrong line (same date, same home/away in another game).
    # Safe cases: (1) matched via (h_canon, a_canon) fallback, or (2) exactly one missing outcome
    # and one unmatched line on this date (then they must be the same game).
    matched_keys = set(
        zip(
            joined[joined["consensus_spread_home"].notna()]["date_key"].astype(str),
            joined[joined["consensus_spread_home"].notna()]["home_key"].astype(str),
            joined[joined["consensus_spread_home"].notna()]["away_key"].astype(str),
        )
    )
    lines_with_keys["_matched"] = lines_with_keys.apply(
        lambda r: (str(r["date_key"]), r["home_key"], r["away_key"]) in matched_keys,
        axis=1,
    )
    unmatched_lines_by_date = (
        lines_with_keys[~lines_with_keys["_matched"]]
        .groupby(lines_with_keys["date_key"].astype(str))
        .size()
    )

    suggested_additions = {}
    for _, row in missing.iterrows():
        h_espn = row["HOME_TEAM"]
        a_espn = row["AWAY_TEAM"]
        dt_str = row["date_key"].isoformat() if hasattr(row["date_key"], "isoformat") else str(row["date_key"])
        same_date = lines_with_keys[lines_with_keys["date_key"].astype(str) == dt_str]
        n_missing_this_date = len(missing[missing["date_key"].astype(str) == dt_str])
        n_unmatched_lines_this_date = int(unmatched_lines_by_date.get(dt_str, 0))
        unique_pair = n_missing_this_date == 1 and n_unmatched_lines_this_date == 1

        lrow = None
        found_via_fallback = False
        one_side_home_matched = False  # True if we matched on home_key == h_espn
        one_side_away_matched = False  # True if we matched on away_key == a_espn
        for _, l in same_date.iterrows():
            if l["home_key"] == h_espn and l["away_key"] != a_espn:
                lrow = l
                one_side_home_matched = True
                break
            if l["away_key"] == a_espn and l["home_key"] != h_espn:
                lrow = l
                one_side_away_matched = True
                break
        line_home_espn = h_espn
        line_away_espn = a_espn
        if lrow is None:
            h_canon = ESPN_TO_NORMALIZED_KEY.get(h_espn, h_espn)
            a_canon = ESPN_TO_NORMALIZED_KEY.get(a_espn, a_espn)
            for _, l in same_date.iterrows():
                if l["home_key"] == h_canon and l["away_key"] == a_canon:
                    lrow = l
                    found_via_fallback = True
                    break
                if l["home_key"] == a_canon and l["away_key"] == h_canon:
                    lrow = l
                    found_via_fallback = True
                    line_home_espn = a_espn
                    line_away_espn = h_espn
                    break

        if lrow is None:
            continue
        # Safe to record: fallback (both sides correct), or unique_pair, or one-side match and
        # this line row is the only one with that home_key (or away_key) on this date.
        if found_via_fallback or unique_pair:
            pass
        elif one_side_home_matched:
            n_with_same_home = (same_date["home_key"] == h_espn).sum()
            if n_with_same_home != 1:
                continue
            # Only suggest the mismatched side (away)
            line_away_espn = a_espn
            line_home_espn = None
        elif one_side_away_matched:
            n_with_same_away = (same_date["away_key"] == a_espn).sum()
            if n_with_same_away != 1:
                continue
            line_home_espn = h_espn
            line_away_espn = None
        else:
            continue

        if line_home_espn is not None and line_home_espn != lrow["home_key"]:
            suggested_additions[lrow["home_team"]] = line_home_espn
        if line_away_espn is not None and line_away_espn != lrow["away_key"]:
            suggested_additions[lrow["away_team"]] = line_away_espn

    return {
        "n_out": n_out,
        "n_lin": n_lin,
        "n_before": n_before,
        "n_after": n_after,
        "suggested_additions": suggested_additions,
        "joined": joined,
        "missing": missing,
        "lines_with_keys": lines_with_keys,
    }


def main():
    ap = argparse.ArgumentParser(description="Debug NCAAB outcomes vs lines join for a date or full season.")
    ap.add_argument(
        "--date",
        default="2026-02-18",
        help="Date YYYY-MM-DD, or 'all' to run every day of the season",
    )
    ap.add_argument(
        "--season",
        default="2025-26",
        help="Season (e.g. 2025-26); used when --date all",
    )
    ap.add_argument(
        "--output",
        default=None,
        help="Write suggested additions to this JSON file (default: ~/Downloads/tmp/ncaab_teams_to_add_to_mapping.json when --date all)",
    )
    ap.add_argument(
        "--outcomes",
        default=None,
        help="Outcomes CSV path (default: s3://ncaab-betting-mt/data/01_input/historical_game_results/DATE.csv)",
    )
    ap.add_argument(
        "--lines",
        default=None,
        help="Lines CSV path (default: s3://ncaab-betting-mt/data/01_input/the-odds-api/ncaab/game_lines/DATE.csv)",
    )
    args = ap.parse_args()
    date_arg = args.date
    season = args.season
    bucket = "ncaab-betting-mt"

    if date_arg.lower() == "all":
        dates = _season_date_list(season)
        print(f"Running for {len(dates)} dates (season {season})...")
        by_date = {}
        total_dates_with_data = 0
        total_outcomes = 0
        total_matched = 0
        dates_100 = 0
        for d in dates:
            res = run_for_date(d, bucket)
            if res is None:
                print(f"  {d}: skip (no data)")
                continue
            total_dates_with_data += 1
            n_out, n_after = res["n_out"], res["n_after"]
            total_outcomes += n_out
            total_matched += n_after
            if n_out > 0 and n_after == n_out:
                dates_100 += 1
            by_date[d] = dict(sorted(res["suggested_additions"].items()))
            if n_out == 0:
                continue
            status = f"{n_after}/{n_out}" if n_after == n_out else f"{n_after}/{n_out} ({n_out - n_after} unmatched)"
            print(f"  {d}: {status}")
        print()
        print(f"--- Summary (season {season}) ---")
        print(f"  Dates with data: {total_dates_with_data} / {len(dates)}")
        print(f"  Dates at 100%% match: {dates_100} / {total_dates_with_data}")
        print(f"  Total outcomes: {total_outcomes}, total matched: {total_matched}")
        if total_outcomes:
            pct = 100.0 * total_matched / total_outcomes
            print(f"  Overall match rate: {pct:.1f}%")
        out_path = args.output or (Path.home() / "Downloads" / "tmp" / "ncaab_teams_to_add_to_mapping.json")
        out_path = Path(out_path).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(dict(sorted(by_date.items())), f, indent=2)
        total_suggestions = sum(len(v) for v in by_date.values())
        all_merged = {}
        for fixes in by_date.values():
            all_merged.update(fixes)
        unique_count = len(all_merged)
        print(f"\nWrote suggestions by date ({total_suggestions} total, {unique_count} unique Odds API → ESPN) to {out_path}")
        if total_suggestions:
            print("Next: add confirmed entries to tmp/build_comprehensive_ncaab_mapping.py ODDS_API_ESPN_OVERRIDES, then run python3 tmp/build_comprehensive_ncaab_mapping.py")
        return

    # Single date
    date_str = date_arg
    outcomes_path = args.outcomes or None
    lines_path = args.lines or None
    print("Loading outcomes and lines...")
    res = run_for_date(date_str, bucket, outcomes_path, lines_path)
    if res is None:
        print("Failed to load outcomes or lines for this date (missing or error).")
        return
    n_out = res["n_out"]
    n_lin = res["n_lin"]
    n_before = res["n_before"]
    n_after = res["n_after"]
    suggested_additions = res["suggested_additions"]
    joined = res["joined"]
    missing = res["missing"]
    lines_with_keys = res["lines_with_keys"]

    print(f"\nRow counts: outcomes = {n_out}, lines = {n_lin}")

    print("\n--- BEFORE normalization (join on raw team names) ---")
    print(f"  Matched (outcome row has a line): {n_before} / {n_out}")
    print(f"  Unmatched: {n_out - n_before}")

    print("\n--- AFTER normalization (join on mapped keys: Odds API -> ESPN) ---")
    print(f"  Matched (outcome row has consensus_spread_home): {n_after} / {n_out}")
    print(f"  Unmatched: {n_out - n_after}")

    if not missing.empty:
        print("\n--- Mismatches (game in outcomes but no line match after normalization) ---")
        for _, row in missing.iterrows():
            h_espn = row["HOME_TEAM"]
            a_espn = row["AWAY_TEAM"]
            dt_str = row["date_key"].isoformat() if hasattr(row["date_key"], "isoformat") else str(row["date_key"])
            same_date = lines_with_keys[lines_with_keys["date_key"].astype(str) == dt_str]
            lrow = None
            for _, l in same_date.iterrows():
                if l["home_key"] == h_espn and l["away_key"] != a_espn:
                    lrow = l
                    break
                if l["away_key"] == a_espn and l["home_key"] != h_espn:
                    lrow = l
                    break
            if lrow is None:
                h_canon = ESPN_TO_NORMALIZED_KEY.get(h_espn, h_espn)
                a_canon = ESPN_TO_NORMALIZED_KEY.get(a_espn, a_espn)
                for _, l in same_date.iterrows():
                    if (l["home_key"] == h_canon and l["away_key"] == a_canon) or (
                        l["home_key"] == a_canon and l["away_key"] == h_canon
                    ):
                        lrow = l
                        break
            print(f"  Game: {a_espn} @ {h_espn}")
            print(f"    Outcome date:      {dt_str} (from outcomes file for this run)")
            print(f"    Outcome (ESPN):     home={h_espn!r}, away={a_espn!r}")
            if lrow is not None:
                print(f"    Line (raw):        home={lrow['home_team']!r}, away={lrow['away_team']!r}")
                print(f"    Line (normalized): home={lrow['home_key']!r}, away={lrow['away_key']!r}")
                if h_espn == lrow["away_key"] and a_espn == lrow["home_key"]:
                    print(f"    -> home/away flipped in line feed (same teams); not a mapping issue.")
                else:
                    if h_espn != lrow["home_key"]:
                        print(f"    MISMATCH home: ESPN {h_espn!r}  vs  normalized {lrow['home_key']!r}")
                    if a_espn != lrow["away_key"]:
                        print(f"    MISMATCH away: ESPN {a_espn!r}  vs  normalized {lrow['away_key']!r}")
            else:
                print(f"    No line row found for this game (tried key match and ESPN->normalized alias fallback).")
            print()

    if len(lines_with_keys.columns) > 0:
        matched_keys = set(
            zip(
                joined[joined["consensus_spread_home"].notna()]["date_key"].astype(str),
                joined[joined["consensus_spread_home"].notna()]["home_key"].astype(str),
                joined[joined["consensus_spread_home"].notna()]["away_key"].astype(str),
            )
        )
        lines_with_keys["_matched"] = lines_with_keys.apply(
            lambda r: (str(r["date_key"]), r["home_key"], r["away_key"]) in matched_keys,
            axis=1,
        )
        unmatched_lines = lines_with_keys[~lines_with_keys["_matched"]]
        if not unmatched_lines.empty:
            print("--- Line rows that did NOT match any outcome (after normalization) ---")
            for _, row in unmatched_lines.iterrows():
                print(f"  {row['away_team']!r} @ {row['home_team']!r}  ->  normalized: ({row['home_key']!r}, {row['away_key']!r})")
            print()

    if suggested_additions:
        print("--- Add these rows to tmp/build_comprehensive_ncaab_mapping.py ODDS_API_ESPN_OVERRIDES ---")
        for odds_name, espn_name in sorted(suggested_additions.items()):
            print(f'    "{odds_name}": "{espn_name}",')
        print()


if __name__ == "__main__":
    main()
