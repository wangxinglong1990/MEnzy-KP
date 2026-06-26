"""Contact Map Feature Extractor.

Loads pre-computed contact maps from disk cache.

Contact maps are (L, L) float32 matrices where entry (i,j)
represents the estimated interaction strength between residues i and j.

Cache location: data/msa_full/contact_maps/{protein_id}.npy

Usage:
    from core.feature_extractors.contact_map_feature_extractor import (
        ContactMapFeatureExtractor,
    )
    extractor = ContactMapFeatureExtractor()
    cm = extractor.extract(protein_id)       # -> (L, L) or None
"""

import numpy as np
from pathlib import Path


class ContactMapFeatureExtractor:
    """Loads pre-computed contact maps from disk cache.

    Parameters
    ----------
    contact_dir : str or Path
        Directory containing {protein_id}.npy contact map files.
    """

    def __init__(self, contact_dir: str | Path = "data/msa_full/contact_maps"):
        self._contact_dir = Path(contact_dir)

    def extract(self, protein_id: str) -> np.ndarray | None:
        """Return contact map (L, L) for a single protein, or None."""
        path = self._contact_dir / f"{protein_id}.npy"
        if path.exists():
            return np.load(path)
        return None

    def exists(self, protein_id: str) -> bool:
        """Check if a contact map exists for this protein."""
        return (self._contact_dir / f"{protein_id}.npy").exists()

    @property
    def contact_dir(self) -> Path:
        return self._contact_dir
