import joblib
import numpy as np

from lightgbm import LGBMRegressor

from models.base_model import BaseModel


class MSA2DModel(BaseModel):

    def __init__(
        self,
        model=None,
        feature_dim=2030,
    ):
        self._model = model
        self._feature_dim = feature_dim

    def fit(self, X, y):

        self._model = LGBMRegressor(
            n_estimators=1500,
            learning_rate=0.03,
            num_leaves=127,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )

        self._model.fit(X, y)

    def predict(self, X):

        return self._model.predict(X)

    def save(self, path):

        joblib.dump(self._model, path)

    @classmethod
    def load(cls, path):

        model = joblib.load(path)

        feat_dim = getattr(
            model,
            "n_features_in_",
            2030
        )

        return cls(
            model=model,
            feature_dim=feat_dim,
        )

    @property
    def input_dim(self):

        return self._feature_dim

