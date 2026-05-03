"""Round-trip checks for rebounds audit list columns vs B_MIN_MAX_FEATS scalars."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.nba_rebounds_modeling.rebounds_audit_list_verify import verify_audit_lists_row


def _load_rebounds_universe_build_module():
    """Load build_rebounds_full_universe (research script) without package import path."""
    repo = Path(__file__).resolve().parents[2]
    path = repo / "src/nba_rebounds_modeling/00_research/scripts/build_rebounds_full_universe.py"
    spec = importlib.util.spec_from_file_location("rebounds_universe_build", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def universe_builder():
    return _load_rebounds_universe_build_module()


def test_rolling_tails_match_scalars(universe_builder):
    """Synthetic log: tail list aggregates match shift(1).rolling scalars."""
    dates = pd.date_range("2024-10-01", periods=15, freq="D")
    reb = np.linspace(2.0, 16.0, 15)
    fg3 = np.arange(15, dtype=float) * 0.5
    logs = pd.DataFrame(
        {
            "season": ["2024-25"] * 15,
            "date": dates.strftime("%Y-%m-%d"),
            "player_normalized": ["test_player"] * 15,
            "game_id": [f"g{i}" for i in range(15)],
            "MIN": [24.0] * 15,
            "OREB": [1.0] * 15,
            "DREB": [1.0] * 15,
            "REB": reb,
            "team_normalized": ["Boston Celtics"] * 15,
        }
    )
    shot_profile_df = logs[["season", "date", "player_normalized"]].assign(
        FGA=np.full(15, 12.0),
        FG3A=fg3,
        FTA=np.ones(15),
    )
    out = universe_builder.build_rolling_features(logs, shot_profile_df)
    last = out.iloc[-1]
    tail60 = last["input_reb_tail_60"]
    tail20 = last["input_fg3a_tail_20"]
    tail5 = last["input_reb_tail_5"]
    assert len(tail60) == 14
    assert math.isclose(float(np.nanmean(tail60)), float(last["roll_reb_mean_60"]), rel_tol=0, abs_tol=1e-9)
    assert math.isclose(float(np.nanmean(tail20)), float(last["roll_fg3a_mean_20"]), rel_tol=0, abs_tol=1e-9)
    s_list = float(pd.Series(tail5, dtype=float).std(ddof=1))
    s_scalar = float(last["roll_reb_std_5"])
    if math.isnan(s_list) and math.isnan(s_scalar):
        pass
    else:
        assert math.isclose(s_list, s_scalar, rel_tol=0, abs_tol=1e-9)


def test_market_lines_min_max(universe_builder):
    props = pd.DataFrame(
        {
            "season": ["2024-25"] * 6,
            "date": ["2024-11-01"] * 6,
            "player_normalized": ["p1"] * 6,
            "bookmaker": ["a", "a", "b", "b", "c", "c"],
            "line": [10.5, 10.5, 11.0, 11.5, 10.5, 12.0],
            "odds_over": [-110] * 6,
            "odds_under": [-110] * 6,
        }
    )
    logs = pd.DataFrame(
        {
            "season": ["2024-25"],
            "date": ["2024-11-01"],
            "player_normalized": ["p1"],
            "game_id": ["gid1"],
            "MIN": [30.0],
            "OREB": [1.0],
            "DREB": [1.0],
            "REB": [8.0],
            "team_normalized": ["Boston Celtics"],
        }
    )
    panel, _ = universe_builder.build_market_panel(props, logs)
    row = panel.iloc[0]
    lines = row["input_reb_prop_lines"]
    assert min(lines) == row["min_line"]
    assert max(lines) == row["max_line"]


def test_team_audit_kwargs_from_row_and_verify():
    from src.nba_rebounds_modeling.rebounds_audit_list_verify import (
        team_audit_kwargs_from_row,
        verify_audit_lists_row,
    )

    row = pd.Series(
        {
            "team_normalized": "Boston Celtics",
            "home_team_norm": "Boston Celtics",
            "away_team_norm": "Los Angeles Lakers",
            "spread_signed": -3.5,
            "input_spread_by_side": [-3.5, 3.5],
            "min_line": 1.0,
            "max_line": 2.0,
            "roll_reb_mean_60": 1.0,
            "roll_fg3a_mean_20": 1.0,
            "roll_reb_std_5": 0.0,
            "input_reb_prop_lines": [1.0, 2.0],
            "input_reb_tail_60": [1.0],
            "input_fg3a_tail_20": [1.0],
            "input_reb_tail_5": [1.0, 1.0],
        }
    )
    k = team_audit_kwargs_from_row(row)
    assert k["team_normalized"] == "Boston Celtics"
    verify_audit_lists_row(row, **k)


def test_verify_audit_lists_spread_side():
    row = pd.Series(
        {
            "spread_signed": -3.5,
            "input_spread_by_side": [-3.5, 3.5],
            "min_line": 1.0,
            "max_line": 2.0,
            "roll_reb_mean_60": 1.0,
            "roll_fg3a_mean_20": 1.0,
            "roll_reb_std_5": 0.0,
            "input_reb_prop_lines": [1.0, 2.0],
            "input_reb_tail_60": [1.0],
            "input_fg3a_tail_20": [1.0],
            "input_reb_tail_5": [1.0, 1.0],
        }
    )
    verify_audit_lists_row(
        row,
        team_normalized="Boston Celtics",
        home_team_norm="Boston Celtics",
        away_team_norm="Los Angeles Lakers",
    )


def test_verify_audit_lists_spread_side_accepts_two_element_ndarray():
    """Parquet/merge sometimes stores ``[home, away]`` as a length-2 ndarray (not list)."""
    row = pd.Series(
        {
            "spread_signed": -3.5,
            "input_spread_by_side": np.array([-3.5, 3.5], dtype=float),
            "min_line": 1.0,
            "max_line": 2.0,
            "roll_reb_mean_60": 1.0,
            "roll_fg3a_mean_20": 1.0,
            "roll_reb_std_5": 0.0,
            "input_reb_prop_lines": [1.0, 2.0],
            "input_reb_tail_60": [1.0],
            "input_fg3a_tail_20": [1.0],
            "input_reb_tail_5": [1.0, 1.0],
        }
    )
    verify_audit_lists_row(
        row,
        team_normalized="Boston Celtics",
        home_team_norm="Boston Celtics",
        away_team_norm="Los Angeles Lakers",
    )


def test_most_recent_date_rows_orders_by_calendar():
    from src.nba_rebounds_modeling import rebounds_audit_list_verify as m

    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-03", "2024-01-02", "2024-01-03"],
            "season": ["2023-24"] * 4,
            "player_normalized": ["a", "b", "c", "d"],
            "game_id": ["g1", "g2", "g3", "g4"],
        }
    )
    out = m._most_recent_date_rows(df, 2)
    assert len(out) == 2
    assert (out["date"] == "2024-01-03").all()
    assert set(out["player_normalized"]) == {"b", "d"}
