from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .config import ModelConfig

try:
    from lightgbm import LGBMRegressor
except ImportError:  # pragma: no cover - optional dependency
    LGBMRegressor = None


@dataclass(slots=True)
class ModelFitResult:
    model_name: str
    n_rows: int
    n_features: int


class TabularPriceForecaster:
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.model_name, estimator = self._build_estimator()
        self.pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", estimator),
            ]
        )

    def _build_estimator(self) -> tuple[str, object]:
        model_type = self.config.model_type.lower()

        if model_type in {"auto", "lightgbm"} and LGBMRegressor is not None:
            estimator = LGBMRegressor(
                n_estimators=self.config.max_iter,
                learning_rate=self.config.learning_rate,
                max_depth=self.config.max_depth,
                random_state=self.config.random_state,
                verbosity=-1,
            )
            return "lightgbm", estimator

        if model_type == "lightgbm" and LGBMRegressor is None:
            raise RuntimeError(
                "model_type='lightgbm' but lightgbm is not installed. "
                "Install optional dependencies or switch model_type='auto'."
            )

        estimator = HistGradientBoostingRegressor(
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
            max_iter=self.config.max_iter,
            random_state=self.config.random_state,
        )
        return "hist_gradient_boosting", estimator

    def fit(self, X: pd.DataFrame, y: pd.Series) -> ModelFitResult:
        self.pipeline.fit(X, y)
        return ModelFitResult(
            model_name=self.model_name,
            n_rows=len(X),
            n_features=X.shape[1],
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.pipeline.predict(X), dtype=float)
