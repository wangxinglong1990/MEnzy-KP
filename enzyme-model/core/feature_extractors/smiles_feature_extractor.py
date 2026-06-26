"""
SMILES Feature Extractor

Unified interface for extracting SMILES embeddings via TrfmSeq2seq.

Usage:
    extractor = SmilesFeatureExtractor(vocab_path, model_path)
    embedding = extractor.extract("CCO")  # -> np.array(1, 1024)
"""

import torch
import numpy as np
from core.encoders import SmilesEncoder


class SmilesFeatureExtractor:
    """Wraps SmilesEncoder with a simplified extract() interface.

    Accepts the same vocabulary and model checkpoint paths used by
    the existing kcat/km prediction and training scripts.
    """

    def __init__(self, vocab_path, model_path, seq_len=220,
                 embed_dim=1024, device=None):
        self.vocab_path = vocab_path
        self.model_path = model_path
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self._device = device or torch.device("cpu")

    # ── public API ──────────────────────────────────────────────

    def extract(self, smiles):
        """Encode SMILES string(s) into an (N, 1024) numpy array.

        Parameters
        ----------
        smiles : str or list[str]
            One or more SMILES strings.

        Returns
        -------
        np.ndarray
            Shape (1, embed_dim) for a single string, (N, embed_dim) for a list.
            Returns zeros if model files are missing.
        """
        if not smiles:
            return np.zeros((1, self.embed_dim), dtype=np.float32)

        encoder = SmilesEncoder(
            self.vocab_path,
            self.model_path,
            seq_len=self.seq_len,
            embed_dim=self.embed_dim,
            device=self._device,
        )
        return encoder.encode(smiles)
