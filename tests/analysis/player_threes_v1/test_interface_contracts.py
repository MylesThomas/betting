import importlib.util
from pathlib import Path


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_schema_contract_column_presence():
    schemas_mod = _load_module(
        Path("src/nba_three_point_modeling/99_utils/schemas.py"),
        "schemas_mod",
    )
    assert "run_id" in schemas_mod.PREDICTIONS_COLUMNS
    assert "is_consensus" in schemas_mod.LINES_COLUMNS
    assert "p_over" in schemas_mod.PRICED_LINES_COLUMNS
    assert "pnl" in schemas_mod.BETS_COLUMNS

