import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_windowed_weights_sum_to_one_and_recency_emphasis():
    mod = _load_module(
        Path(
            "src/nba_three_point_modeling/02_probability_engine/uncertainty_models/v2_weighted_history_sampler.py"
        ),
        "v2_weighted_history_sampler_mod",
    )
    weights = mod.build_recency_weights(
        n_rows=5,
        weighting_mode="windowed_weighted",
        decay_alpha=0.03,
    )
    assert np.isclose(weights.sum(), 1.0)
    assert weights[-1] > weights[0]


def test_sampler_returns_non_negative_draws():
    mod = _load_module(
        Path(
            "src/nba_three_point_modeling/02_probability_engine/uncertainty_models/v2_weighted_history_sampler.py"
        ),
        "v2_weighted_history_sampler_mod_non_negative",
    )
    history_df = pd.DataFrame(
        {
            "date": [
                "2025-01-01",
                "2025-01-03",
                "2025-01-05",
                "2025-01-07",
                "2025-01-09",
            ],
            "actual_fg3m": [2.0, 4.0, 3.0, 5.0, 4.0],
            "actual_fg3a": [7.0, 10.0, 8.0, 12.0, 11.0],
            "actual_min": [29.0, 33.0, 31.0, 36.0, 34.0],
        }
    )
    model = mod.fit_v2_weighted_history_sampler(
        history_df=history_df,
        history_n=5,
        weighting_mode="exp_decay",
        decay_alpha=0.05,
    )
    sims = model.simulate_fg3m(y_hat=3.8, n_sims=500, rng=np.random.default_rng(11))
    assert len(sims) == 500
    assert float(sims.min()) >= 0.0
