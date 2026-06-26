"""StackingV2Model — configurable meta learner ensemble.

Supports multiple base estimators via the ``meta_model`` config parameter.

Usage:
    model = StackingV2Model(meta_model="ridge")
    model.fit(X, y)
    model.predict(X)
    model.get_feature_importance()
"""

import joblib
import numpy as np
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import ExtraTreesRegressor
from lightgbm import LGBMRegressor

from models.base_model import BaseModel

META_ESTIMATORS = {
    "ridge": lambda: Ridge(alpha=1.0, random_state=42),
    "elasticnet": lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
    "lightgbm": lambda: LGBMRegressor(
        n_estimators=500, learning_rate=0.03, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
    ),
    "extratrees": lambda: ExtraTreesRegressor(
        n_estimators=500, max_depth=10, random_state=42, n_jobs=-1,
    ),
}


class StackingV2Model(BaseModel):
    """Meta-learner for ensembling base model predictions.

    Parameters
    ----------
    meta_model : str
        One of: ridge, elasticnet, lightgbm, extratrees.
    model : sklearn estimator or None
        Pre-fitted estimator. None creates a new instance.
    feature_dim : int
        Number of input features (default auto-detected).
    feature_names : list[str] or None
        Names of input features for importance analysis.
    """

    def __init__(self, meta_model: str = "ridge", model=None,
                 feature_dim: int = None, feature_names: list[str] = None):
        self._meta_model = meta_model
        self._model = model
        self._feature_dim = feature_dim
        self._feature_names = feature_names

        if meta_model not in META_ESTIMATORS:
            raise ValueError(
                f"Unknown meta_model '{meta_model}'. "
                f"Choose from: {list(META_ESTIMATORS.keys())}"
            )

    # ── BaseModel interface ─────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if self._model is None:
            self._model = META_ESTIMATORS[self._meta_model]()
        self._model.fit(X, y)
        if self._feature_dim is None:
            self._feature_dim = X.shape[1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted or loaded")
        return self._model.predict(X)

    def save(self, path: str) -> None:
        if self._model is None:
            raise RuntimeError("No model to save")
        joblib.dump({
            "model": self._model,
            "meta_model": self._meta_model,
            "feature_dim": self._feature_dim,
            "feature_names": self._feature_names,
        }, path)

    @classmethod
    def load(cls, path: str) -> "StackingV2Model":
        data = joblib.load(path)
        return cls(
            meta_model=data.get("meta_model", "ridge"),
            model=data["model"],
            feature_dim=data.get("feature_dim"),
            feature_names=data.get("feature_names"),
        )

    # ── Extended API ────────────────────────────────────────────

    def get_feature_importance(self) -> dict | None:
        """Return feature importance if supported by the base estimator."""
        if hasattr(self._model, "feature_importances_"):
            importances = self._model.feature_importances_
            names = self._feature_names or [f"f{i}" for i in range(len(importances))]
            return dict(sorted(zip(names, importances), key=lambda x: -x[1]))
        if hasattr(self._model, "coef_"):
            coefs = self._model.coef_.flatten() if self._model.coef_.ndim > 1 else self._model.coef_
            names = self._feature_names or [f"f{i}" for i in range(len(coefs))]
            return dict(sorted(zip(names, coefs), key=lambda x: -abs(x[1])))
        return None

    @property
    def input_dim(self) -> int:
        return self._feature_dim or -1
