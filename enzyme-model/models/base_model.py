"""
Base Model — abstract interface for all model variants.

All model variants (Baseline, MSA1D, MSA2D, Condition, Stacking)
inherit from ``BaseModel`` and implement the four core methods.
This guarantees call-site uniformity across the entire pipeline.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    """Abstract interface shared by every model variant.

    Concrete subclasses must implement ``fit``, ``predict``,
    ``save``, and ``load``.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model on feature matrix X and target vector y.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions for feature matrix X.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples,)
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist the trained model to *path*.

        Parameters
        ----------
        path : str
            Filesystem path for the serialised model.
        """
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseModel":
        """Load a previously saved model from *path*.

        Parameters
        ----------
        path : str
            Filesystem path to a model saved with :meth:`save`.

        Returns
        -------
        BaseModel
            Deserialised model instance, ready for ``predict``.
        """
        ...

    @property
    def input_dim(self) -> int:
        """Expected number of input features, or -1 if unknown."""
        return -1
