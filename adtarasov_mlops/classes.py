import json
from dataclasses import dataclass
from typing import Dict, Union

import pandas as pd


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

    def to_json(self):
        return json.dumps(
            {
                'model_params': self.model_params,
                'r2_score': self.r2,
                'mse': self.mse,
                'mae': self.mae,
            },
            indent=4,
        )
