"""
Singleton model service - loads ESMC, SMILES Transformer, and enzyme-model
ExtraTreesRegressor models (KM + Kcat) once. Holds them in memory for the
lifetime of the FastAPI process.
"""
import threading

import joblib
import numpy as np

from config import KM_MODEL_PATH, KCAT_MODEL_PATH
from src.features.extractor import extract_joint_features

# enzyme-model feature order:  [SMILES(1024) | Protein(960)]
# Kinora-main feature order:   [Protein(960) | SMILES(1024)]
# ExtraTreesRegressor is column-order-sensitive, so we must reorder.
_PROTEIN_DIM = 960


def _reorder_features(features: np.ndarray) -> np.ndarray:
    """Swap [Protein|SMILES] → [SMILES|Protein] to match enzyme-model format."""
    return np.concatenate([features[:, _PROTEIN_DIM:], features[:, :_PROTEIN_DIM]], axis=1)


class ModelService:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        print("[ModelService] Loading enzyme-model baseline models ...")
        if not KM_MODEL_PATH.exists():
            raise FileNotFoundError(f"KM model not found: {KM_MODEL_PATH}")
        if not KCAT_MODEL_PATH.exists():
            raise FileNotFoundError(f"Kcat model not found: {KCAT_MODEL_PATH}")

        self.km_model = joblib.load(str(KM_MODEL_PATH))
        self.kcat_model = joblib.load(str(KCAT_MODEL_PATH))
        print(f"[ModelService] KM model:  {type(self.km_model).__name__}")
        print(f"[ModelService] Kcat model: {type(self.kcat_model).__name__}")

        # Eagerly warm up encoders by running a dummy extraction
        print("[ModelService] Warming up encoders (first call initializes singletons) ...")
        try:
            _ = extract_joint_features(["CCO"], ["AAA"])
            print("[ModelService] Encoders initialized.")
        except Exception as e:
            print(f"[ModelService] Encoder warmup warning: {e}")

        print("[ModelService] Ready.")

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def predict_single(self, protein: str, smiles: str) -> dict:
        """Predict KM and Kcat for a single enzyme-substrate pair."""
        features = extract_joint_features([smiles], [protein]).astype(np.float32)
        # Reorder columns: Kinora [Protein|SMILES] → enzyme-model [SMILES|Protein]
        features = _reorder_features(features)

        log10_km = float(self.km_model.predict(features)[0])
        log10_kcat = float(self.kcat_model.predict(features)[0])

        return {
            "log10_km": log10_km,
            "km": 10 ** log10_km,
            "log10_kcat": log10_kcat,
            "kcat": 10 ** log10_kcat,
            "kcat_over_km": 10 ** (log10_kcat - log10_km),
        }

    def predict_batch(self, sequences: list[str], smiles_list: list[str]) -> np.ndarray:
        """Returns (N, 2) array: log10_km, log10_kcat."""
        features = extract_joint_features(smiles_list, sequences).astype(np.float32)
        # Reorder columns: Kinora [Protein|SMILES] → enzyme-model [SMILES|Protein]
        features = _reorder_features(features)

        log10_km = self.km_model.predict(features).reshape(-1, 1)
        log10_kcat = self.kcat_model.predict(features).reshape(-1, 1)
        return np.concatenate([log10_km, log10_kcat], axis=1).astype(np.float32)
