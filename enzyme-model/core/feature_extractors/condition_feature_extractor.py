"""Condition Feature Extractor.

Encodes Temperature + pH into a 2-dimensional numeric feature vector
with missing-value imputation and standard scaling.

Usage
-----
    from core.feature_extractors.condition_feature_extractor import (
        ConditionFeatureExtractor,
    )

    extractor = ConditionFeatureExtractor()
    condition_feat = extractor.fit_transform(temperature, ph)
"""

import numpy as np
from sklearn.preprocessing import StandardScaler


class ConditionFeatureExtractor:
    """Encodes reaction-condition features (temperature, pH).

    Parameters
    ----------
    temp_fill : float or None
        Value to impute missing temperatures.  ``None`` → median.
    ph_fill : float or None
        Value to impute missing pH values.  ``None`` → median.
    """

    def __init__(self, temp_fill: float | None = None, ph_fill: float | None = None):
        self._temp_fill = temp_fill
        self._ph_fill = ph_fill
        self._scaler = StandardScaler()
        self._fitted = False

    # ── public API ──────────────────────────────────────────────

    def fit_transform(self, temperature: np.ndarray, ph: np.ndarray) -> np.ndarray:
        """Impute missing values, fit the scaler, and return scaled features.

        Parameters
        ----------
        temperature : np.ndarray, shape (n_samples,)
        ph : np.ndarray, shape (n_samples,)

        Returns
        -------
        np.ndarray, shape (n_samples, 2)
            Columns: [scaled_temperature, scaled_ph]
        """
        temp, pH = self._impute(temperature, ph)
        feat = np.column_stack([temp, pH])
        self._scaler.fit(feat)
        self._fitted = True
        return self._scaler.transform(feat)

    def transform(self, temperature: np.ndarray, ph: np.ndarray) -> np.ndarray:
        """Apply the pre-fitted imputation and scaling."""
        if not self._fitted:
            raise RuntimeError("ConditionFeatureExtractor has not been fitted yet")
        temp, pH = self._impute(temperature, ph, use_fitted=True)
        feat = np.column_stack([temp, pH])
        return self._scaler.transform(feat)

    # ── helpers ─────────────────────────────────────────────────

    def _impute(self, temperature: np.ndarray, ph: np.ndarray,
                use_fitted: bool = False):
        temp = np.asarray(temperature, dtype=float).copy()
        pH = np.asarray(ph, dtype=float).copy()

        fill_t = self._temp_fill if self._temp_fill is not None else np.nanmedian(temp)
        fill_p = self._ph_fill if self._ph_fill is not None else np.nanmedian(pH)

        temp[np.isnan(temp)] = fill_t
        pH[np.isnan(pH)] = fill_p

        return temp, pH
