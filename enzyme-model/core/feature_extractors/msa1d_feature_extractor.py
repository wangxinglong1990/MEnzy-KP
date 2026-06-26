"""MSA1D Feature Extractor.

Loads pre-computed 6-dimensional MSA1D features from the feature cache.

Features (6 dim):
    0: msa_depth           — number of homologous sequences
    1: conservation_mean   — avg per-position conservation
    2: conservation_std    — std of conservation
    3: entropy_mean        — avg Shannon entropy
    4: entropy_std         — std of Shannon entropy
    5: gap_ratio           — fraction of gaps in alignment

Usage:
    from core.feature_extractors.msa1d_feature_extractor import MSA1DFeatureExtractor
    extractor = MSA1DFeatureExtractor()
    feat = extractor.extract(protein_id)          # -> np.ndarray (6,) or None
    feats = extractor.extract_many(protein_ids)   # -> np.ndarray (N, 6) or None
"""

import numpy as np
from pathlib import Path


class MSA1DFeatureExtractor:
    """Loads pre-computed MSA1D features from disk cache.

    Expects files at FEAT_DIR/{protein_id}.npy (shape=(6,), float32).
    Falls back to on-the-fly extraction from A3M if cache missing.
    """

    def __init__(self, feat_dir: str | Path = "data/msa/features",
                 a3m_dir: str | Path = "data/msa/a3m"):
        self._feat_dir = Path(feat_dir)
        self._a3m_dir = Path(a3m_dir)

    # ── public API ──────────────────────────────────────────────

    def extract(self, protein_id: str) -> np.ndarray | None:
        """Return MSA1D feature vector (6,) for a single protein or None."""
        cache_path = self._feat_dir / f"{protein_id}.npy"
        if cache_path.exists():
            return np.load(cache_path)

        # On-the-fly extraction from A3M
        try:
            from scripts.extract_msa1d_features import extract
            return extract(protein_id, self._a3m_dir, self._feat_dir)
        except Exception:
            return None

    def extract_many(self, protein_ids: list[str]) -> np.ndarray | None:
        """Return MSA1D features (N, 6) for a list of protein IDs."""
        feats = []
        for pid in protein_ids:
            f = self.extract(pid)
            if f is not None:
                feats.append(f)
        if not feats:
            return None
        return np.stack(feats, axis=0)

    @property
    def feature_dim(self) -> int:
        return 6
