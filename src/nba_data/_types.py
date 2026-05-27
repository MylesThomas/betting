from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class NBAData:
    logs: pd.DataFrame
    props: pd.DataFrame
    lines: pd.DataFrame
    meta: dict
