import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_consensus_flag_marks_closest_to_5050():
    utils_dir = Path("src/nba_three_point_modeling/99_utils").resolve()
    if str(utils_dir) not in sys.path:
        sys.path.insert(0, str(utils_dir))

    data_loading = _load_module(
        Path("src/nba_three_point_modeling/99_utils/data_loading.py"),
        "data_loading_mod",
    )
    props_df = pd.DataFrame(
        [
            {"date": "2025-12-01", "player_normalized": "Stephen Curry", "prop_line": 3.5, "bookmaker": "a", "over_odds": -110.0, "under_odds": -110.0, "home_team": "Golden State Warriors", "away_team": "Los Angeles Lakers"},
            {"date": "2025-12-01", "player_normalized": "Stephen Curry", "prop_line": 2.5, "bookmaker": "a", "over_odds": -170.0, "under_odds": 130.0, "home_team": "Golden State Warriors", "away_team": "Los Angeles Lakers"},
        ]
    )
    out = data_loading.build_consensus_and_contract_views(props_df)
    consensus_lines = out[out["is_consensus"] == 1]["line"].tolist()
    assert consensus_lines == [3.5]

