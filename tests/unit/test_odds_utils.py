"""
Tests for src/odds_utils. Spread-cover logic is the single source of truth; these tests lock it in.
See docs/domain/spread-cover-rule.md.
"""

import pytest

from src.odds_utils import did_cover_spread


class TestDidCoverSpread:
    """Spread cover: home_spread = home team's line (negative when home favored). No reimplementing elsewhere."""

    def test_home_favored_does_not_cover_when_margin_too_small(self):
        # Arizona 75-68, home -10.5: margin 7, need > 10.5
        assert did_cover_spread(75, 68, -10.5, bet_home=True) is False

    def test_home_favored_covers_when_margin_exceeds_spread(self):
        # Home -10.5, margin 11
        assert did_cover_spread(79, 68, -10.5, bet_home=True) is True
        assert did_cover_spread(80, 68, -10.5, bet_home=True) is True

    def test_home_favored_margin_exactly_spread_is_push(self):
        # Margin 10.5 with -10.5: diff = 0, push
        assert did_cover_spread(78.5, 68, -10.5, bet_home=True) is None

    def test_away_underdog_covers_when_lose_by_less_than_spread(self):
        # Away +10.5, home 75-68: away lost by 7, 7 < 10.5 so away covers
        assert did_cover_spread(75, 68, -10.5, bet_home=False) is True

    def test_away_underdog_does_not_cover_when_lose_by_more_than_spread(self):
        # Away +10.5, home 85-68: away lost by 17
        assert did_cover_spread(85, 68, -10.5, bet_home=False) is False

    def test_spread_none_returns_none(self):
        assert did_cover_spread(75, 68, None, bet_home=True) is None

    def test_spread_nan_returns_none(self):
        assert did_cover_spread(75, 68, float('nan'), bet_home=True) is None

    def test_push_returns_none(self):
        # Home -3, final margin exactly 3 (e.g. 70-67)
        assert did_cover_spread(70, 67, -3.0, bet_home=True) is None
