"""
BaselineModel — wraps a trained ExtraTreesRegressor (existing joblib).

Persistence is delegated to joblib, so existing ``.joblib`` files
remain directly usable both through this wrapper and via raw
``joblib.load()``.
"""

import joblib
import numpy as np
from models.base_model import BaseModel


class BaselineModel(BaseModel):
    """Model 1: ESMC + SMILES → ExtraTrees → y_base.

    Parameters
    ----------
    model : object or None
        A fitted sklearn-compatible estimator.  ``None`` creates an
        untrained instance (for ``fit``).
    feature_dim : int
        Number of input features expected by the model (default 1984
        = 1024 SMILES + 960 ESM).
    """

    def __init__(self, model=None, feature_dim: int = 1984):
        self._model = model
        self._feature_dim = feature_dim

    # ── BaseModel interface ────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        from sklearn.ensemble import ExtraTreesRegressor
        self._model = ExtraTreesRegressor(
            n_estimators=300,
            max_depth=30,
            min_samples_split=2,
            min_samples_leaf=1,
            n_jobs=6,
            random_state=42,
        )
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("模型未加载或未训练")
        return self._model.predict(X)

    def save(self, path: str) -> None:
        if self._model is None:
            raise RuntimeError("没有可保存的模型")
        joblib.dump(self._model, path)

    @classmethod
    def load(cls, path: str) -> "BaselineModel":
        """Load an ExtraTrees model from a joblib file.

        The file can be a model saved with :meth:`save`, or any
        existing ``*_predictor.joblib`` produced by the original
        training pipeline.
        """
        model = joblib.load(path)
        # Infer feature_dim from n_features_ if available
        feat_dim = getattr(model, "n_features_in_", 1984)
        return cls(model=model, feature_dim=feat_dim)

    @property
    def input_dim(self) -> int:
        return self._feature_dim
