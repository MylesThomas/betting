import importlib.util
from pathlib import Path

import numpy as np


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_global_variance_simulations_non_negative():
    mod = _load_module(
        Path("src/nba_three_point_modeling/02_probability_engine/uncertainty_models/global_variance.py"),
        "global_variance_mod",
    )
    model = mod.GlobalVarianceModel(sigma=1.2)
    sims = model.simulate_fg3m(y_hat=3.8, n_sims=1000, rng=np.random.default_rng(3))
    assert len(sims) == 1000
    assert float(sims.min()) >= 0.0

