"""
Protein Feature Extractor

Unified interface for extracting protein embeddings via ESMC-300M.
Supports both single-sequence inference and dual-GPU training modes.

Usage:
    extractor = ProteinFeatureExtractor(device=torch.device("cpu"), use_direct_load=True)
    embedding = extractor.extract("MKLL...")  # -> np.array(1, 960)
"""

import torch
import numpy as np
from core.encoders import ESMEncoder, encode_sequences_dual_gpu, ESM_AVAILABLE


class ProteinFeatureExtractor:
    """Wraps ESMEncoder with a simplified extract() interface.

    For inference (single/batch) the extractor creates a lightweight
    ESMEncoder internally. For dual-GPU training pass ``use_dual_gpu=True``.
    """

    def __init__(self, model_name="esmc_300m", embed_dim=960,
                 device=None, use_direct_load=False, use_dual_gpu=False):
        self.model_name = model_name
        self.embed_dim = embed_dim
        self._device = device or torch.device("cpu")
        self._use_direct_load = use_direct_load
        self._use_dual_gpu = use_dual_gpu

        if not ESM_AVAILABLE:
            raise RuntimeError("ESM library not installed")

    # ── public API ──────────────────────────────────────────────

    def extract(self, sequence):
        """Encode protein sequence(s) into an (N, 960) numpy array.

        Parameters
        ----------
        sequence : str or list[str]
            One or more amino-acid sequences.

        Returns
        -------
        np.ndarray
            Shape (1, embed_dim) for a single string, (N, embed_dim) for a list.
        """
        if self._use_dual_gpu and isinstance(sequence, list) and len(sequence) > 1:
            # Training path: dual-GPU parallel
            return encode_sequences_dual_gpu(
                sequence,
                model_name=self.model_name,
                embed_dim=self.embed_dim,
            )
        else:
            # Inference path: single sequence or batch via ESMEncoder
            encoder = ESMEncoder(
                model_name=self.model_name,
                embed_dim=self.embed_dim,
                device=self._device,
                use_direct_load=self._use_direct_load,
            )
            result = encoder.encode(sequence)
            encoder.release()
            return result
