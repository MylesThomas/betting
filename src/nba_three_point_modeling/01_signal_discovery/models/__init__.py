"""Model registry for signal discovery."""

from baseline import BaselineSeasonAvgOLSModel
from baseline import fit_baseline_model
from v2_three_input_regression import V2ThreeInputRegressionModel
from v2_three_input_regression import fit_v2_three_input_model
from v3_market_spread_regression import V3MarketSpreadRegressionModel
from v3_market_spread_regression import fit_v3_market_spread_model

__all__ = [
    "BaselineSeasonAvgOLSModel",
    "V2ThreeInputRegressionModel",
    "V3MarketSpreadRegressionModel",
    "fit_baseline_model",
    "fit_v2_three_input_model",
    "fit_v3_market_spread_model",
]

