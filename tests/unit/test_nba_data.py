"""
Tests for src/nba_data.

Unit tests: pure Python, no S3.
Integration tests (marked): require AWS credentials and a warm cache.
  Run with: pytest -m integration tests/unit/test_nba_data.py
"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.nba_data._loaders import _abbr_to_full, _add_matchup_columns
from src.player_team_history.team_normalization import TEAM_ABBR_TO_NAME

import pandas as pd

ALL_30_FULL_NAMES = set(TEAM_ABBR_TO_NAME.values())


# ── Unit tests ────────────────────────────────────────────────────────────────

class TestAbbrToFull:
    def test_current_team(self):
        assert _abbr_to_full("BOS") == "Boston Celtics"

    def test_current_team_lakers(self):
        assert _abbr_to_full("LAL") == "Los Angeles Lakers"

    def test_historical_code_nets(self):
        # NJN (New Jersey Nets) → Brooklyn Nets
        assert _abbr_to_full("NJN") == "Brooklyn Nets"

    def test_historical_code_sonics(self):
        # SEA (Seattle SuperSonics) → Oklahoma City Thunder
        assert _abbr_to_full("SEA") == "Oklahoma City Thunder"

    def test_all_30_current_codes_map_to_known_names(self):
        for abbr in TEAM_ABBR_TO_NAME:
            assert _abbr_to_full(abbr) in ALL_30_FULL_NAMES, f"{abbr} did not map to a known team"


class TestAddMatchupColumns:
    def _make_df(self, rows: list[tuple[str, str]]) -> pd.DataFrame:
        """rows = [(MATCHUP, TEAM_ABBREVIATION), ...]"""
        return pd.DataFrame(rows, columns=["MATCHUP", "TEAM_ABBREVIATION"])

    def test_home_game(self):
        df = self._make_df([("BOS vs. MIA", "BOS")])
        result = _add_matchup_columns(df)
        assert result.iloc[0]["home_team_normalized"] == "Boston Celtics"
        assert result.iloc[0]["away_team_normalized"] == "Miami Heat"
        assert result.iloc[0]["is_home"] == True

    def test_away_game(self):
        df = self._make_df([("BOS @ MIA", "BOS")])
        result = _add_matchup_columns(df)
        assert result.iloc[0]["home_team_normalized"] == "Miami Heat"
        assert result.iloc[0]["away_team_normalized"] == "Boston Celtics"
        assert result.iloc[0]["is_home"] == False

    def test_historical_code_normalized(self):
        # Player on NJN (Brooklyn Nets) playing at home vs LAL
        df = self._make_df([("NJN vs. LAL", "NJN")])
        result = _add_matchup_columns(df)
        assert result.iloc[0]["home_team_normalized"] == "Brooklyn Nets"
        assert result.iloc[0]["away_team_normalized"] == "Los Angeles Lakers"
        assert result.iloc[0]["is_home"] == True

    def test_team_normalized_column(self):
        df = self._make_df([("GSW vs. LAC", "GSW")])
        result = _add_matchup_columns(df)
        assert result.iloc[0]["team_normalized"] == "Golden State Warriors"

    def test_multiple_rows(self):
        rows = [
            ("LAL vs. BOS", "LAL"),  # home
            ("LAL @ DEN", "LAL"),   # away
        ]
        df = self._make_df(rows)
        result = _add_matchup_columns(df)
        assert result.iloc[0]["home_team_normalized"] == "Los Angeles Lakers"
        assert result.iloc[0]["is_home"] == True
        assert result.iloc[1]["home_team_normalized"] == "Denver Nuggets"
        assert result.iloc[1]["is_home"] == False

    def test_all_30_teams_representable_as_home_and_away(self):
        """Every current team can appear as home and away in logs."""
        rows = []
        teams = list(TEAM_ABBR_TO_NAME.keys())
        for abbr in teams:
            opponent = "BOS" if abbr != "BOS" else "LAL"
            rows.append((f"{abbr} vs. {opponent}", abbr))  # home
            rows.append((f"{abbr} @ {opponent}", abbr))   # away
        df = self._make_df(rows)
        result = _add_matchup_columns(df)
        home_teams = set(result["home_team_normalized"].dropna())
        away_teams = set(result["away_team_normalized"].dropna())
        assert ALL_30_FULL_NAMES <= home_teams, f"Missing as home: {ALL_30_FULL_NAMES - home_teams}"
        assert ALL_30_FULL_NAMES <= away_teams, f"Missing as away: {ALL_30_FULL_NAMES - away_teams}"


# ── Integration tests (require S3 + warm cache) ───────────────────────────────

@pytest.fixture(scope="session")
def nba_data():
    from src.nba_data import get_data
    return get_data()


@pytest.mark.integration
def test_logs_all_30_teams_as_home(nba_data):
    """All 30 current NBA teams appear at least once as home_team_normalized in logs."""
    found = set(nba_data.logs["home_team_normalized"].dropna().unique())
    missing = ALL_30_FULL_NAMES - found
    assert not missing, f"Teams never appeared as home: {missing}"


@pytest.mark.integration
def test_logs_all_30_teams_as_away(nba_data):
    """All 30 current NBA teams appear at least once as away_team_normalized in logs."""
    found = set(nba_data.logs["away_team_normalized"].dropna().unique())
    missing = ALL_30_FULL_NAMES - found
    assert not missing, f"Teams never appeared as away: {missing}"


@pytest.mark.integration
def test_logs_row_count(nba_data):
    assert len(nba_data.logs) > 50_000, "Expected 50k+ player-game rows across 3 seasons"


@pytest.mark.integration
def test_props_has_all_three_markets(nba_data):
    markets = set(nba_data.props["market"].unique())
    for expected in ("player_points", "player_rebounds", "player_assists"):
        assert expected in markets, f"Missing market: {expected}"


@pytest.mark.integration
def test_lines_has_spread_market(nba_data):
    assert "spread" in nba_data.lines["market"].unique()


@pytest.mark.integration
def test_meta_has_required_keys(nba_data):
    for key in ("updated_at", "seasons", "max_game_date", "row_counts"):
        assert key in nba_data.meta, f"manifest missing key: {key}"


@pytest.mark.integration
def test_player_normalized_no_nulls_in_logs(nba_data):
    null_count = nba_data.logs["player_normalized"].isna().sum()
    assert null_count == 0, f"{null_count} null player_normalized values in logs"


@pytest.mark.integration
def test_logs_and_props_share_player_normalized_values(nba_data):
    """A meaningful intersection exists — player names are joining correctly."""
    log_players = set(nba_data.logs["player_normalized"].dropna().unique())
    prop_players = set(nba_data.props["player_normalized"].dropna().unique())
    overlap = log_players & prop_players
    assert len(overlap) > 200, f"Only {len(overlap)} players match between logs and props"
