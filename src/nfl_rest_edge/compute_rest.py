"""
Compute NFL rest edge metrics from a schedule DataFrame.

Definitions (per Warren Sharp):
  days_rest     = (current_game_date - prev_game_date).days - 1
                  Season opener gets 6 (standard week, no prior game).
  rest_edge     = team's days_rest - opponent's days_rest (+ means team has more rest)
  net_rest      = sum of all rest_edge values across a team's full season

Situation flags:
  short_week_road   Away game with days_rest < 6
  post_road_prime   Previous game was an away SNF or MNF
  opp_extra_prep    Opponent has days_rest > 6 (extra time to prepare)
  negated_bye       Team's days_rest >= 13 (off full bye) AND opp_days_rest >= 9
                    (opponent also on extra rest, eroding the bye advantage)
  in_3_in_10        Part of a stretch of 3 games within 10 calendar days
  in_4_in_17        Part of a stretch of 4 games within 17 calendar days
"""

from datetime import date

import pandas as pd

# Mid-season game cancellations produce a week gap that looks identical to a real bye.
# Keys: (nfl_season, team, week_of_next_game_after_cancellation)
# 2022 BUF/CIN: Week 17 cancelled (Damar Hamlin) → both teams jump W16→W18
_CANCELLED_GAME_FALSE_BYES: set[tuple[int, str, int]] = {
    (2022, "BUF", 18),
    (2022, "CIN", 18),
}


def _add_days_rest(tg: pd.DataFrame) -> pd.DataFrame:
    """Vectorized days_rest calculation; avoids groupby.apply FutureWarning."""
    tg = tg.sort_values(["team", "game_date"]).copy()
    tg["_ord"] = tg["game_date"].apply(lambda d: d.toordinal())
    tg["days_rest"] = tg.groupby("team")["_ord"].diff() - 1
    first_game_mask = tg.groupby("team")["_ord"].transform("rank") == 1
    tg.loc[first_game_mask, "days_rest"] = 6.0  # season opener: standard rest
    tg["days_rest"] = tg["days_rest"].astype(int)
    tg = tg.drop(columns=["_ord"])
    return tg


def _compressed_schedule_flags(game_dates: list[date]) -> tuple[list[bool], list[bool]]:
    """Return per-game booleans for in_3_in_10 and in_4_in_17."""
    n = len(game_dates)
    in_3 = [False] * n
    in_4 = [False] * n

    for i in range(n - 2):
        if (game_dates[i + 2] - game_dates[i]).days <= 10:
            in_3[i] = in_3[i + 1] = in_3[i + 2] = True

    for i in range(n - 3):
        if (game_dates[i + 3] - game_dates[i]).days <= 17:
            in_4[i] = in_4[i + 1] = in_4[i + 2] = in_4[i + 3] = True

    return in_3, in_4


def build_team_games(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand schedule (272 game rows) into team-game format (544 rows, one per team per game).

    Returns columns: week, game_date, game_type, team, opponent, is_home, broadcast
    """
    home = schedule_df.assign(team=schedule_df["home_team"], opponent=schedule_df["away_team"], is_home=True)
    away = schedule_df.assign(team=schedule_df["away_team"], opponent=schedule_df["home_team"], is_home=False)
    keep = ["week", "game_date", "game_type", "team", "opponent", "is_home", "broadcast"]
    return pd.concat([home[keep], away[keep]], ignore_index=True)


def compute_rest_metrics(schedule_df: pd.DataFrame, season: int | None = None) -> pd.DataFrame:
    """
    Given a 272-row schedule DataFrame, return a 544-row team-game DataFrame
    with rest calculations and situation flags.
    """
    tg = build_team_games(schedule_df)
    tg = tg.sort_values(["team", "game_date"]).reset_index(drop=True)

    # Step 1: days_rest per team per game
    tg = _add_days_rest(tg)

    # Step 2: merge opponent's rest_days onto each row
    opp = (
        tg[["week", "team", "days_rest"]]
        .rename(columns={"team": "opponent", "days_rest": "opp_days_rest"})
    )
    tg = tg.merge(opp, on=["week", "opponent"], how="left")
    tg["rest_edge"] = tg["days_rest"] - tg["opp_days_rest"]

    # Step 3: situation flags computed per team (sorted by date)
    flag_rows = []
    for team, grp in tg.groupby("team"):
        grp = grp.sort_values("game_date").copy()
        game_dates = grp["game_date"].tolist()
        weeks = grp["week"].tolist()

        # short_week_road: away + fewer than 6 days rest
        grp["short_week_road"] = (~grp["is_home"]) & (grp["days_rest"] < 6)

        # post_road_prime: previous game was away SNF or MNF
        prev_is_away = (~grp["is_home"]).shift(1, fill_value=False)
        prev_game_type = grp["game_type"].shift(1, fill_value="")
        grp["post_road_prime"] = prev_is_away & prev_game_type.isin(["SNF", "MNF"])

        # opp_extra_prep: opponent had more than standard rest
        grp["opp_extra_prep"] = grp["opp_days_rest"] > 6

        # negated_bye: team had a bye in the preceding week AND opponent also has extra rest.
        # Detect bye from week gap (week[i] - week[i-1] > 1), not from days_rest, because
        # a Wednesday game after a bye gives only 9 days rest (not the usual 13).
        #
        # Edge case 1 — cancelled season opener (e.g. 2017 MIA/TB, Hurricane Irma):
        #   first game is not Week 1, so weeks[0] > 1 means they had a forced bye in Week 1.
        # Edge case 2 — cancelled mid-season game (e.g. 2022 BUF/CIN Week 17, Damar Hamlin):
        #   week gap looks identical to a real bye; suppress with KNOWN_GAME_CANCELLATIONS.
        had_bye = [weeks[0] > 1]
        for i in range(1, len(weeks)):
            had_bye.append(weeks[i] - weeks[i - 1] > 1)
        # Null out false byes caused by mid-season game cancellations
        if season is not None:
            for i, w in enumerate(weeks):
                if (season, team, w) in _CANCELLED_GAME_FALSE_BYES:
                    had_bye[i] = False
        grp["had_bye"] = had_bye
        grp["negated_bye"] = grp["had_bye"] & (grp["opp_days_rest"] > 6)

        # compressed schedule flags
        in_3, in_4 = _compressed_schedule_flags(game_dates)
        grp["in_3_in_10"] = in_3
        grp["in_4_in_17"] = in_4

        flag_rows.append(grp)

    result = pd.concat(flag_rows, ignore_index=True)
    return result.sort_values(["team", "game_date"]).reset_index(drop=True)


def compute_team_summary(team_games: pd.DataFrame) -> pd.DataFrame:
    """
    Compute season-level rest summary per team.

    Returns one row per team with: net_rest, rest_adv_games, rest_disadv_games,
    short_week_road_count, post_road_prime_count, opp_extra_prep_count,
    negated_bye_count, in_3_in_10_count, in_4_in_17_count
    """
    agg = (
        team_games.groupby("team")
        .agg(
            net_rest=("rest_edge", "sum"),
            rest_adv_games=("rest_edge", lambda x: (x > 0).sum()),
            rest_disadv_games=("rest_edge", lambda x: (x < 0).sum()),
            short_week_road=("short_week_road", "sum"),
            post_road_prime=("post_road_prime", "sum"),
            opp_extra_prep=("opp_extra_prep", "sum"),
            negated_bye=("negated_bye", "sum"),
            in_3_in_10=("in_3_in_10", lambda x: x.any()),
            in_4_in_17=("in_4_in_17", lambda x: x.any()),
        )
        .reset_index()
    )
    return agg.sort_values("net_rest", ascending=False).reset_index(drop=True)


def build_weekly_rest_edge_table(team_games: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot team_games to produce a teams × weeks matrix of rest_edge values.
    Mirrors the heatmap on p.27 of Warren Sharp's 2026 preview.
    """
    pivot = team_games.pivot_table(
        index="team", columns="week", values="rest_edge", aggfunc="first"
    )
    pivot.columns = [f"Wk{w}" for w in pivot.columns]

    # Add net_rest and sort
    net = team_games.groupby("team")["rest_edge"].sum().rename("Net")
    pivot = pivot.join(net).sort_values("Net", ascending=False)
    cols = ["Net"] + [c for c in pivot.columns if c != "Net"]
    return pivot[cols]
