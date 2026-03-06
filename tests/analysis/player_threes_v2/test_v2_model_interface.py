import importlib.util
from pathlib import Path

import pandas as pd


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v2_three_input_model_fit_predict_contract():
    mod = _load_module(
        Path(
            "src/nba_three_point_modeling/01_signal_discovery/models/v2_three_input_regression.py"
        ),
        "v2_three_input_regression_mod",
    )
    games = pd.DataFrame(
        {
            "date": [
                "2025-10-10",
                "2025-10-12",
                "2025-10-14",
                "2025-10-16",
                "2025-10-18",
            ],
            "actual_fg3m": [4.0, 5.0, 3.0, 6.0, 4.0],
            "actual_fg3a": [10.0, 11.0, 9.0, 13.0, 10.0],
            "actual_min": [33.0, 35.0, 31.0, 36.0, 34.0],
        }
    )
    features = mod.build_v2_feature_frame(games)
    model = mod.fit_v2_three_input_model(features)
    preds = model.predict(features)
    assert len(preds) == len(games)
    assert model.model_id == "v2_three_input_regression"
    assert all(col in features.columns for col in mod.V2_FEATURE_COLUMNS)
