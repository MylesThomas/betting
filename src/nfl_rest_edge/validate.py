"""
Validate computed rest metrics against Warren Sharp's 2026 NFL Preview (pages 25-40).

All expected values sourced from Sharp's published table and text (p.27-39).
Run after compute_rest_metrics() to confirm the pipeline is correct.
"""

import sys

import pandas as pd


# Net rest rankings from Warren Sharp's 2026 Football Preview heatmap table (p.27).
# Values taken from the image (authoritative) — the surrounding text had minor
# transcription differences that were confusing; the table image is the ground truth.
SHARP_NET_REST_2026 = {
    "CHI": 15,
    "BUF": 14,
    "DAL": 12,
    "WAS": 9,
    "SEA": 9,
    "CAR": 8,
    "HOU": 8,
    "NE": 8,
    "ATL": 7,
    "TEN": 6,
    "MIN": 4,
    "DEN": 3,
    "NYG": 3,
    "ARI": 1,
    "DET": 1,
    "CLE": 0,
    "SF": 0,
    "GB": -2,
    "KC": -2,
    "TB": -2,
    "BAL": -3,
    "JAX": -3,
    "CIN": -4,
    "IND": -6,
    "LAR": -6,
    "MIA": -6,
    "PIT": -6,
    "NO": -7,
    "NYJ": -9,
    "LV": -13,
    "PHI": -15,
    "LAC": -24,
}

# Sharp's published situational counts for select teams
SHARP_SHORT_WEEK_ROAD_2026 = {
    "BUF": 3,  # Only team with 3; no other team has more than 2
}

SHARP_NEGATED_BYES_2026 = {"LAR", "GB", "TEN"}

SHARP_4_IN_17_TEAMS_2026 = {"BUF"}  # Only team with 4-games-in-17-days stretch


def _check(condition: bool, label: str, expected, actual) -> bool:
    if condition:
        print(f"  PASS  {label}")
        return True
    else:
        print(f"  FAIL  {label}  |  expected={expected}  actual={actual}")
        return False


def validate_2026(team_games: pd.DataFrame, summary: pd.DataFrame) -> bool:
    """
    Run all assertions against Sharp's published 2026 values.
    Returns True if all pass, False if any fail.
    """
    net = summary.set_index("team")["net_rest"].to_dict()
    short_road = summary.set_index("team")["short_week_road"].to_dict()
    has_4_in_17 = summary.set_index("team")["in_4_in_17"].to_dict()

    failures = 0
    passes = 0

    print("=" * 60)
    print("VALIDATION: Warren Sharp 2026 NFL Preview (pp.25-40)")
    print("=" * 60)

    # --- Net rest rankings ---
    print("\nNet Rest Rankings:")
    for team, expected in SHARP_NET_REST_2026.items():
        actual = net.get(team, "MISSING")
        ok = _check(actual == expected, f"net_rest[{team}]", expected, actual)
        if ok:
            passes += 1
        else:
            failures += 1

    # --- 39-day swing ---
    print("\nSchedule Disparity:")
    swing = max(net.values()) - min(net.values())
    ok = _check(swing == 39, "best-to-worst swing == 39 days", 39, swing)
    passes += ok
    failures += not ok

    best_team = max(net, key=net.get)
    ok = _check(best_team == "CHI", "best net rest team == CHI", "CHI", best_team)
    passes += ok
    failures += not ok

    worst_team = min(net, key=net.get)
    ok = _check(worst_team == "LAC", "worst net rest team == LAC", "LAC", worst_team)
    passes += ok
    failures += not ok

    # --- Short-week road games ---
    print("\nShort-Week Road Games:")
    buf_swr = int(short_road.get("BUF", -1))
    ok = _check(buf_swr == 3, "BUF short_week_road == 3", 3, buf_swr)
    passes += ok
    failures += not ok

    # No other team has more than 2
    over_2 = [t for t, v in short_road.items() if int(v) > 2 and t != "BUF"]
    ok = _check(len(over_2) == 0, "No other team has 3+ short_week_road games", [], over_2)
    passes += ok
    failures += not ok

    # --- 4-games-in-17-days ---
    print("\n4 Games in 17 Days:")
    teams_with_4in17 = {t for t, v in has_4_in_17.items() if v}
    ok = _check(
        teams_with_4in17 == SHARP_4_IN_17_TEAMS_2026,
        "4-in-17 teams == {BUF}",
        SHARP_4_IN_17_TEAMS_2026,
        teams_with_4in17,
    )
    passes += ok
    failures += not ok

    # --- Negated byes ---
    print("\nNegated Byes:")
    negated = summary.set_index("team")["negated_bye"]
    teams_with_negated = {t for t, v in negated.items() if v > 0}
    ok = _check(
        teams_with_negated == SHARP_NEGATED_BYES_2026,
        f"negated bye teams == {SHARP_NEGATED_BYES_2026}",
        SHARP_NEGATED_BYES_2026,
        teams_with_negated,
    )
    passes += ok
    failures += not ok

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"Results: {passes} passed, {failures} failed")
    print("=" * 60)

    return failures == 0
