"""ConditionModel — ESM-C + SMILES + Temperature + pH → LightGBM → y.

Inherits ``BaseModel`` and implements the four core methods
(``fit``, ``predict``, ``save``, ``load``).
"""

import joblib
import numpy as np
from lightgbm import LGBMRegressor

from models.base_model import BaseModel


class ConditionModel(BaseModel):
    """Model 4: ESMC + SMILES + Temperature + pH → LightGBM → y.

    Parameters
    ----------
    model : LGBMRegressor or None
        A fitted LightGBM regressor.  ``None`` creates an untrained
        instance (for ``fit``).
    feature_dim : int
        Number of input features (default 1986 = 1024 SMILES + 960 ESM + 2 condition).
    """

    def __init__(self, model=None, feature_dim: int = 1986):
        self._model = model
        self._feature_dim = feature_dim

    # ── BaseModel interface ─────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the LightGBM model.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)
        """
        self._model = LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions for feature matrix X.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        if self._model is None:
            raise RuntimeError("模型未加载或未训练")
        return self._model.predict(X)

    def save(self, path: str) -> None:
        """Persist the trained LightGBM model to *path*.

        Parameters
        ----------
        path : str
            Filesystem path for the serialised model.
        """
        if self._model is None:
            raise RuntimeError("没有可保存的模型")
        joblib.dump(self._model, path)

    @classmethod
    def load(cls, path: str) -> "ConditionModel":
        """Load a LightGBM model from a joblib file.

        Parameters
        ----------
        path : str
            Path to a model saved with :meth:`save`.

        Returns
        -------
        ConditionModel
            Deserialised model instance, ready for ``predict``.
        """
        model = joblib.load(path)
        feat_dim = getattr(model, "n_features_in_", 1986)
        return cls(model=model, feature_dim=feat_dim)

    @property
    def input_dim(self) -> int:
        """Expected number of input features."""
        return self._feature_dim
