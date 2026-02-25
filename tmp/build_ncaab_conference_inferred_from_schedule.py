"""
Build NCAAB conference mapping inferred from Jan+ schedule (second source of truth).

For every team that appears in S3 game results (Jan 1 onward):
- Count games played against each opponent conference (using primary wiki mapping for opponents).
- If team played one conference 10+ times -> that is their conference.
- Else if one conference is the most-played -> use it.
- Else -> do not add (tie/ambiguous or no known opponents).

This gives a second mapping that should agree with the primary (wiki) mapping. We compare
the two: agreement = validation; disagreements = fix primary or investigate; teams only in
inferred = gaps in primary to fill; teams only in primary but not inferred = investigate.

Requires: S3 access to ncaab-betting-mt/data/01_input/historical_game_results/
Run from repo root:
    python tmp/build_ncaab_conference_inferred_from_schedule.py
    python tmp/build_ncaab_conference_inferred_from_schedule.py --start-date 2026-01-01 --end-date 2026-02-24
    python tmp/build_ncaab_conference_inferred_from_schedule.py --season 2025-26 --end-date 2026-02-25
"""

import argparse
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

# Repo root: find via .git
def _repo_root():
    p = Path(__file__).resolve().parent.parent
    if (p / ".git").exists():
        return p
    raise FileNotFoundError("Repo root (parent of tmp/) not found")

REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import re
import boto3
import pandas as pd

try:
    from src.ncaab_team_name_mapping import ODDS_API_TO_ESPN_NCAAB
except ImportError:
    ODDS_API_TO_ESPN_NCAAB = {}

# ESPN -> Odds API (reverse of ODDS_API_TO_ESPN_NCAAB); outcomes use ESPN names, primary has Odds API keys
ESPN_TO_ODDS = {v: k for k, v in ODDS_API_TO_ESPN_NCAAB.items()} if ODDS_API_TO_ESPN_NCAAB else {}

# S3
BUCKET = "ncaab-betting-mt"
OUTCOMES_PREFIX = "data/01_input/historical_game_results/"


def _normalize_team_name_for_lookup(name: str) -> str:
    """Normalize for matching primary keys: lowercase, remove parentheticals, collapse spaces. Fallback for wiki-style keys."""
    if not name or not isinstance(name, str):
        return ""
    s = name.strip().lower()
    s = re.sub(r"\s*\([^)]*\)\s*", " ", s)
    s = " ".join(s.split())
    return s


def _primary_conference(
    team_name: str,
    primary_mapping: dict[str, str],
    norm_to_conf: dict[str, str],
) -> str:
    """Look up conference: 1) exact primary 2) ESPN->Odds API then primary 3) normalized primary key match."""
    t = team_name.strip()
    if t in primary_mapping:
        return primary_mapping[t]
    odds_key = ESPN_TO_ODDS.get(t)
    if odds_key and odds_key in primary_mapping:
        return primary_mapping[odds_key]
    norm = _normalize_team_name_for_lookup(t)
    return norm_to_conf.get(norm, "")

# Jan 1 cutoff: conference play. Season 2025-26 -> Jan 1, 2026.
def _jan1_cutoff(season: str) -> date:
    # season is "2025-26" -> second year is 2026
    year = int(season.split("-")[1])
    if year < 100:
        year += 2000  # 26 -> 2026
    return date(year, 1, 1)


def _list_s3_keys(client, bucket: str, prefix: str):
    out = []
    pag = client.get_paginator("list_objects_v2")
    for page in pag.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            out.append(obj["Key"])
    return out


def load_outcomes_s3(s3, start_date: date, end_date: date) -> pd.DataFrame:
    """Load NCAAB game results from S3; columns include GAME_DATE, HOME_TEAM, AWAY_TEAM."""
    keys = _list_s3_keys(s3, BUCKET, OUTCOMES_PREFIX)
    dfs = []
    for key in keys:
        try:
            fn = key.split("/")[-1].replace(".csv", "")
            d = pd.to_datetime(fn).date()
            if start_date <= d <= end_date:
                r = s3.get_object(Bucket=BUCKET, Key=key)
                df = pd.read_csv(r["Body"])
                if df is not None and not df.empty:
                    dfs.append(df)
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True)
    out["GAME_DATE"] = pd.to_datetime(out["GAME_DATE"]).dt.date
    out = out.drop_duplicates(subset=["GAME_DATE", "HOME_TEAM", "AWAY_TEAM"])
    return out


def build_inferred_mapping(
    outcomes: pd.DataFrame,
    primary_mapping: dict[str, str],
    jan1: date,
    min_conference_games: int = 10,
    verbose: bool = False,
) -> tuple[dict[str, str], list[tuple[str, str, str]]]:
    """
    For every team in outcomes, infer conference from Jan+ opponent conference counts.
    Returns (inferred_dict, problem_list). Inferred includes all teams we can assign;
    compare with primary to check agreement.
    """
    jan_plus = outcomes[outcomes["GAME_DATE"] >= jan1].copy()
    if jan_plus.empty:
        return {}, [("_no_data", "no_games", f"No games on or after {jan1}")]

    # Normalized primary key -> conference (so "purdue fort wayne mastodons" -> Horizon League)
    norm_to_conf = {}
    for key, conf in primary_mapping.items():
        if key and conf:
            n = _normalize_team_name_for_lookup(key)
            if n and n not in norm_to_conf:
                norm_to_conf[n] = conf

    # Per team: list of (opponent_name, opponent_conference) for each Jan+ game
    team_opp_pairs = defaultdict(list)
    for _, row in jan_plus.iterrows():
        h = str(row["HOME_TEAM"]).strip()
        a = str(row["AWAY_TEAM"]).strip()
        h_conf = _primary_conference(h, primary_mapping, norm_to_conf)
        a_conf = _primary_conference(a, primary_mapping, norm_to_conf)
        team_opp_pairs[h].append((a, a_conf))
        team_opp_pairs[a].append((h, h_conf))

    inferred = {}
    problems = []

    for team in sorted(team_opp_pairs.keys()):
        opp_pairs = team_opp_pairs[team]
        opp_confs = [c for _, c in opp_pairs]
        if verbose:
            print(f"\n--- {team} ---")
        for i, (opp_name, opp_conf) in enumerate(opp_pairs):
            if verbose:
                disp = opp_conf if opp_conf else "(unknown)"
                print(f"  game {i}, team {team}, opponent {opp_name}, opponent conference {disp}")

        known = [c for c in opp_confs if c]
        if not known:
            problems.append((team, "no_known_opp_confs", "All opponents unmapped"))
            if verbose:
                print(f"  Decision: (none) — no_known_opp_confs")
            continue
        counts = Counter(known)
        best_conf, best_count = counts.most_common(1)[0]
        if best_count >= min_conference_games:
            inferred[team] = best_conf
            if verbose:
                print(f"  Decision: {best_conf} (10+ games, count={best_count})")
            continue
        if len(counts) == 1:
            inferred[team] = best_conf
            if verbose:
                print(f"  Decision: {best_conf} (only one opponent conference, count={best_count})")
            continue
        second_count = counts.most_common(2)[1][1] if len(counts) >= 2 else 0
        if best_count > second_count:
            inferred[team] = best_conf
            if verbose:
                print(f"  Decision: {best_conf} (most-played, count={best_count})")
        else:
            problems.append((team, "tie_or_ambiguous", f"counts={dict(counts)}"))
            if verbose:
                print(f"  Decision: (none) — tie_or_ambiguous counts={dict(counts)}")

    return inferred, problems


def report_agreement(
    primary: dict[str, str],
    inferred: dict[str, str],
) -> None:
    """Print agreement between primary (wiki) and inferred (schedule) mappings."""
    both = set(primary) & set(inferred)
    agree = [t for t in both if primary[t] == inferred[t]]
    disagree = [t for t in both if primary[t] != inferred[t]]
    only_primary = set(primary) - set(inferred)
    only_inferred = set(inferred) - set(primary)

    print("\n--- Agreement (primary vs inferred) ---")
    print(f"  Both have mapping:  {len(both)}")
    print(f"  Agree:              {len(agree)}")
    print(f"  Disagree:           {len(disagree)}")
    if disagree:
        for t in sorted(disagree)[:30]:
            print(f"    {t}: primary={primary[t]}  inferred={inferred[t]}")
        if len(disagree) > 30:
            print(f"    ... and {len(disagree) - 30} more")
    print(f"  Only in primary:    {len(only_primary)} (inferred could not assign)")
    if only_primary and len(only_primary) <= 20:
        for t in sorted(only_primary):
            print(f"    {t}")
    elif only_primary:
        for t in sorted(only_primary)[:15]:
            print(f"    {t}")
        print(f"    ... and {len(only_primary) - 15} more")
    print(f"  Only in inferred:  {len(only_inferred)} (gaps in primary)")
    if only_inferred and len(only_inferred) <= 20:
        for t in sorted(only_inferred):
            print(f"    {t} -> {inferred[t]}")
    elif only_inferred:
        for t in sorted(only_inferred)[:15]:
            print(f"    {t} -> {inferred[t]}")
        print(f"    ... and {len(only_inferred) - 15} more")


def write_inferred_module(mapping: dict[str, str], out_path: Path, season: str) -> None:
    """Write src/ncaab_conference_inferred.py with NCAAB_CONFERENCE_INFERRED_* dict."""
    season_key = season.replace("-", "_")
    var_name = f"NCAAB_CONFERENCE_INFERRED_{season_key}"
    lines = [
        '"""',
        "NCAAB Conference Mappings (inferred from schedule).",
        "",
        "Second source of truth: inferred from Jan+ opponent conference counts (10+ games",
        "vs a conference, or most-played). Should agree with primary; use for validation and fallback.",
        "",
        "Generated by: tmp/build_ncaab_conference_inferred_from_schedule.py",
        "Source: S3 game results (historical_game_results), primary mapping for opponents.",
        '"""',
        "",
        f"{var_name} = {{",
    ]
    for k in sorted(mapping.keys()):
        key_esc = k.replace("'", "\\'")
        lines.append(f"    '{key_esc}': '{mapping[k]}',")
    lines.append("}")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Build inferred NCAAB conference mapping from Jan+ schedule")
    parser.add_argument("--season", default="2025-26", help="Season string e.g. 2025-26 (for output var name)")
    parser.add_argument("--start-date", default=None, help="Start date YYYY-MM-DD (optional; with --end-date overrides season range)")
    parser.add_argument("--end-date", default=None, help="End date YYYY-MM-DD (default: today ET)")
    parser.add_argument("--min-games", type=int, default=10, help="Min games vs a conference to assign (default 10)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Per-team logging: each game and decision")
    args = parser.parse_args()

    from datetime import datetime
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
    end_date = args.end_date
    if not end_date:
        end_date = datetime.now(ET).strftime("%Y-%m-%d")
    end_d = date.fromisoformat(end_date)

    if args.start_date:
        start_date = date.fromisoformat(args.start_date)
        jan1 = start_date  # use provided range as "conference window"
    else:
        season_starts = {
            "2025-26": "2025-11-03",
            "2024-25": "2024-11-03",
            "2023-24": "2023-11-06",
        }
        start_date_str = season_starts.get(args.season, "2025-11-03")
        start_date = date.fromisoformat(start_date_str)
        jan1 = _jan1_cutoff(args.season)

    from src.ncaab_conference_data import NCAAB_CONFERENCE_MAPPING_2025_26

    s3 = boto3.client("s3")
    outcomes = load_outcomes_s3(s3, start_date, end_d)
    if outcomes.empty:
        print("No outcomes in S3 for date range; cannot build inferred mapping.")
        sys.exit(1)
    print(f"Outcomes: {len(outcomes)} games, {outcomes['GAME_DATE'].min()} to {outcomes['GAME_DATE'].max()}")

    jan_plus = outcomes[outcomes["GAME_DATE"] >= jan1]
    print(f"Jan+ games (>= {jan1}): {len(jan_plus)}")

    inferred, problems = build_inferred_mapping(
        outcomes,
        NCAAB_CONFERENCE_MAPPING_2025_26,
        jan1,
        min_conference_games=args.min_games,
        verbose=args.verbose,
    )
    print(f"Inferred: {len(inferred)} teams")
    if problems:
        print(f"Could not infer: {len(problems)}")
        for team, reason, detail in problems[:20]:
            print(f"  - {team}: {reason} ({detail})")
        if len(problems) > 20:
            print(f"  ... and {len(problems) - 20} more")

    report_agreement(NCAAB_CONFERENCE_MAPPING_2025_26, inferred)

    out_path = REPO_ROOT / "src" / "ncaab_conference_inferred.py"
    write_inferred_module(inferred, out_path, args.season)
    print(f"\nWrote: {out_path}")
    return inferred


if __name__ == "__main__":
    main()
