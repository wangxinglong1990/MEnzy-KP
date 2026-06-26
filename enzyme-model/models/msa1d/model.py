"""MSA1DModel — ESM-C + SMILES + MSA1D(6) → XGBoost → y.

Inherits BaseModel and implements the four core methods
(fit, predict, save, load).
"""

import joblib
import numpy as np
from xgboost import XGBRegressor

from models.base_model import BaseModel


class MSA1DModel(BaseModel):
    """Model 2: ESMC + SMILES + MSA1D(6) → XGBoost → y.

    Parameters
    ----------
    model : XGBRegressor or None
        A fitted XGBoost regressor.  ``None`` creates an untrained instance.
    feature_dim : int
        Number of input features (default 1990 = 1024 SMILES + 960 ESM + 6 MSA1D).
    """

    def __init__(self, model=None, feature_dim: int = 1990):
        self._model = model
        self._feature_dim = feature_dim

    # ── BaseModel interface ─────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the XGBoost model."""
        self._model = XGBRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
            n_jobs=-1,
        )
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions for feature matrix X."""
        if self._model is None:
            raise RuntimeError("模型未加载或未训练")
        return self._model.predict(X)

    def save(self, path: str) -> None:
        """Persist the trained model to *path*."""
        if self._model is None:
            raise RuntimeError("没有可保存的模型")
        joblib.dump(self._model, path)

    @classmethod
    def load(cls, path: str) -> "MSA1DModel":
        """Load a model from a joblib file."""
        model = joblib.load(path)
        feat_dim = getattr(model, "n_features_in_", 1990)
        return cls(model=model, feature_dim=feat_dim)

    @property
    def input_dim(self) -> int:
        return self._feature_dim
