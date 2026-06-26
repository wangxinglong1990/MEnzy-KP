"""StackingModel — [y_base, y_cond, y_msa1d] → Ridge → y.

Inherits BaseModel and implements the four core methods.
"""
import joblib
import numpy as np
from sklearn.linear_model import Ridge
from models.base_model import BaseModel


class StackingModel(BaseModel):
    """Meta-learner that ensembles Baseline, Condition, and MSA1D predictions.

    Parameters
    ----------
    model : Ridge or None
        A fitted Ridge regressor.  ``None`` creates an untrained instance.
    feature_dim : int
        Number of input features (default 3 = y_base, y_cond, y_msa1d).
    """

    def __init__(self, model=None, feature_dim: int = 3):
        self._model = model
        self._feature_dim = feature_dim

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model = Ridge(alpha=1.0, random_state=42)
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
    def load(cls, path: str) -> "StackingModel":
        model = joblib.load(path)
        feat_dim = getattr(model, "n_features_in_", 3)
        return cls(model=model, feature_dim=feat_dim)

    @property
    def input_dim(self) -> int:
        return self._feature_dim
