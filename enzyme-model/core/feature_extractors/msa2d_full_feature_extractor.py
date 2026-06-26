"""MSA2D Full Feature Extractor — Fuses Contact Map CNN + ESM + SMILES.

Pipeline:
    sample (dict)
    │
    ├── contact_map  →  ContactMapCNN.forward()  →  CNN_embed  (128,)
    │
    ├── protein_id   →  ESM cache lookup         →  ESM_embed  (960,)
    │
    └── smiles       →  SmilesEncoder            →  SMILES_embed (1024,)
                           │
                     np.concatenate([CNN, ESM, SMILES])
                           │
                     Final feature vector (2112,)

Usage:
    from core.feature_extractors.msa2d_full_feature_extractor import (
        MSA2DFullFeatureExtractor,
    )
    extractor = MSA2DFullFeatureExtractor()
    feat = extractor.extract(sample)  # np.ndarray (2112,)
"""

import numpy as np
import torch

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent

PRETRAINED_DIR = _ROOT / "data" / "pretrained"
CONTACT_DIR = _ROOT / "data" / "msa_full" / "contact_maps"


class MSA2DFullFeatureExtractor:
    """Fuses Contact Map CNN features + ESM + SMILES into a single vector.

    Parameters
    ----------
    device : torch.device or None
        Device for CNN inference. None → auto-detect.
    cnn_model : torch.nn.Module or None
        Pre-loaded ContactMapCNN. None → lazy load on first call.
    smiles_encoder : SmilesEncoder or None
        Pre-loaded SmilesEncoder. None → lazy load.
    esm_encoder : ESMEncoder or None
        Pre-loaded ESMEncoder. None → lazy load.
    """

    def __init__(self, device=None, cnn_model=None, smiles_encoder=None, esm_encoder=None):
        self._device = device or torch.device("cpu")
        self._cnn = cnn_model
        self._smiles_encoder = smiles_encoder
        self._esm_encoder = esm_encoder
        self._cnn_loaded = cnn_model is not None
        self._smiles_loaded = smiles_encoder is not None
        self._esm_loaded = esm_encoder is not None

    # ── public API ──────────────────────────────────────────────

    def extract(self, sample: dict) -> np.ndarray:
        """Extract fused feature vector from a sample dict.

        Parameters
        ----------
        sample : dict
            Must contain keys: contact_map (Tensor or None),
            protein_id (str), sequence (str), smiles (str).
            Typically from MSA2DFullDataset.__getitem__().

        Returns
        -------
        np.ndarray, shape (2112,)  or  (input_dim,) with fallback.
        """
        features = []

        # ── 1. Contact Map CNN embedding ──
        cm_embed = self._extract_cnn(sample)
        if cm_embed is not None:
            features.append(cm_embed)
        else:
            features.append(np.zeros(128, dtype=np.float32))

        # ── 2. ESM embedding ──
        esm_embed = self._extract_esm(sample)
        if esm_embed is not None:
            features.append(esm_embed)
        else:
            features.append(np.zeros(960, dtype=np.float32))

        # ── 3. SMILES embedding ──
        smi_embed = self._extract_smiles(sample)
        if smi_embed is not None:
            features.append(smi_embed)
        else:
            features.append(np.zeros(1024, dtype=np.float32))

        return np.concatenate(features, axis=0).astype(np.float32)

    @property
    def feature_dim(self) -> int:
        """Total fused feature dimension: 128 + 960 + 1024 = 2112."""
        return 128 + 960 + 1024

    # ── lazy-load helpers ───────────────────────────────────────

    def _extract_cnn(self, sample: dict) -> np.ndarray | None:
        if sample.get("contact_map") is None:
            return None
        if not self._cnn_loaded:
            from models.msa2d_full.model import ContactMapCNN
            self._cnn = ContactMapCNN().to(self._device).eval()
            self._cnn_loaded = True
        with torch.no_grad():
            cm = sample["contact_map"].unsqueeze(0).to(self._device)  # (1,1,L,L)
            # Get features before the final FC layers
            feat = self._cnn.features(cm)  # (1, 128, 1, 1)
            feat = feat.flatten().cpu().numpy()  # (128,)
        return feat

    def _extract_esm(self, sample: dict) -> np.ndarray | None:
        if not self._esm_loaded:
            try:
                from core.encoders.esm_encoder import ESMEncoder
                self._esm_encoder = ESMEncoder(device=self._device)
                self._esm_loaded = True
            except Exception:
                return None
        try:
            seq = sample.get("sequence", "")
            if not seq:
                return None
            emb = self._esm_encoder.encode(seq)
            if isinstance(emb, list):
                emb = np.array(emb)
            if emb.ndim == 1:
                emb = emb.reshape(1, -1)
            return emb[0].astype(np.float32)
        except Exception:
            return None

    def _extract_smiles(self, sample: dict) -> np.ndarray | None:
        if not self._smiles_loaded:
            try:
                from core.encoders.smiles_encoder import SmilesEncoder
                vocab = PRETRAINED_DIR / "smiles_vocab.pkl"
                model = PRETRAINED_DIR / "smiles_transformer.pkl"
                self._smiles_encoder = SmilesEncoder(str(vocab), str(model))
                self._smiles_loaded = True
            except Exception:
                return None
        try:
            smi = sample.get("smiles", "")
            if not smi:
                return None
            emb = self._smiles_encoder.encode(smi)
            if isinstance(emb, list):
                emb = np.array(emb)
            if emb.ndim == 1:
                emb = emb.reshape(1, -1)
            return emb[0].astype(np.float32)
        except Exception:
            return None
