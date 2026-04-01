"""
Lock Option A vector batch + under_only play rule to v3 semantics (strict edge > min_edge).
"""

import numpy as np
import pytest
from scipy.stats import norm

from src.nba_rebounds_modeling.option_a_scoring import (
    option_a_vector_batch,
    pick_side,
    play_under_only_mask,
)


class TestOptionAVectorBatch:
    def test_matches_manual_normal_cdf_single_row(self):
        consensus = np.array([8.0])
        yhat = np.array([7.0])
        line = np.array([8.5])
        sigma_raw = np.array([2.0])
        shrink = 0.0
        p_nov_o = np.array([0.48])
        p_nov_u = np.array([0.52])
        mean_adj, z, p_over, p_under, edge_o, edge_u = option_a_vector_batch(
            consensus, yhat, line, sigma_raw, shrink, p_nov_o, p_nov_u
        )
        assert mean_adj[0] == pytest.approx(7.0)
        assert z[0] == pytest.approx(0.75)
        assert p_under[0] == pytest.approx(norm.cdf(0.75))
        assert p_over[0] == pytest.approx(1.0 - norm.cdf(0.75))
        assert edge_o[0] == pytest.approx(p_over[0] - 0.48)
        assert edge_u[0] == pytest.approx(p_under[0] - 0.52)

    def test_sigma_floor_applied(self):
        consensus = np.array([10.0])
        yhat = np.array([10.0])
        line = np.array([10.0])
        sigma_raw = np.array([0.01])
        z = option_a_vector_batch(
            consensus, yhat, line, sigma_raw, 0.0,
            np.array([0.5]), np.array([0.5]),
        )[1]
        # (10-10)/0.25 = 0
        assert z[0] == pytest.approx(0.0)

    def test_shrink_half_moves_mean_halfway(self):
        c = np.array([8.0])
        y = np.array([10.0])
        ell = np.array([9.0])
        sig = np.array([1.0])
        mean_adj, *_ = option_a_vector_batch(
            c, y, ell, sig, 0.5, np.array([0.5]), np.array([0.5]),
        )
        assert mean_adj[0] == pytest.approx(9.0)


class TestPickSideUnderOnlyStrict:
    def test_edge_equal_min_edge_is_no_bet(self):
        edge_o = np.array([0.0])
        edge_u = np.array([0.05])
        assert pick_side(0, edge_o, edge_u, 0.05, "under_only") is None

    def test_edge_above_min_edge_is_under(self):
        edge_o = np.array([0.0])
        edge_u = np.array([0.0500001])
        assert pick_side(0, edge_o, edge_u, 0.05, "under_only") == "under"


class TestPlayUnderOnlyMask:
    def test_matches_pick_side_vectorized(self):
        edge_u = np.array([0.04, 0.051, 0.05])
        m = play_under_only_mask(edge_u, 0.05)
        assert (m == np.array([False, True, False])).all()
