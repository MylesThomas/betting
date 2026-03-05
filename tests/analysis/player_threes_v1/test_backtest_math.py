import importlib.util
from pathlib import Path


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_target_profit_stake_examples():
    odds_mod = _load_module(
        Path("src/nba_three_point_modeling/99_utils/odds.py"),
        "odds_mod",
    )
    assert abs(odds_mod.target_profit_stake(-110, 100.0) - 110.0) < 1e-9
    assert abs(odds_mod.target_profit_stake(200, 100.0) - 50.0) < 1e-9

