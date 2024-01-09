import pandas as pd

from dataclasses import dataclass
from typing import Dict, Union


@dataclass
class Dataset:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


@dataclass
class Metrics:
    model_params: Dict[str, Union[str, float]]
    r2: float
    mse: float
    mae: float
