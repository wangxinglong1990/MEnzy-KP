"""
Combined Feature Extractor

Orchestrates protein and SMILES feature extraction into a single call.
Returns a dict with all embeddings for downstream models (Baseline,
MSA1D, MSA2D, Condition).

Usage:
    extractor = CombinedFeatureExtractor(
        smiles_vocab_path, smiles_model_path,
        esm_device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    result = extractor.extract(sequence="MKLL...", smiles="CCO")
    # result = {
    #     "protein_embedding": np.array(1, 960),
    #     "smiles_embedding": np.array(1, 1024),
    #     "combined_embedding": np.array(1, 1984),
    # }
"""

import numpy as np
from core.feature_extractors.protein_feature_extractor import ProteinFeatureExtractor
from core.feature_extractors.smiles_feature_extractor import SmilesFeatureExtractor


class CombinedFeatureExtractor:
    """Orchestrates single-call feature extraction for both modalities.

    Parameters
    ----------
    smiles_vocab_path : str
        Path to ``smiles_vocab.pkl``.
    smiles_model_path : str
        Path to ``smiles_transformer.pkl``.
    esm_model_name : str
        ESM model identifier (default ``"esmc_300m"``).
    esm_embed_dim : int
        Expected ESM embedding dimension (default ``960``).
    smiles_seq_len : int
        Max SMILES token sequence length (default ``220``).
    smiles_embed_dim : int
        Expected SMILES embedding dimension (default ``1024``).
    esm_device : torch.device or None
        Device for ESM inference (default ``cpu``).
    esm_use_direct_load : bool
        ``True`` → ``ESMC.from_pretrained(name, device=device)``
        ``False`` → ``ESMC.from_pretrained(name).to(device)``
    use_dual_gpu : bool
        If True, use dual-GPU parallelism for protein encoding (training).
    """

    def __init__(self, smiles_vocab_path, smiles_model_path,
                 esm_model_name="esmc_300m", esm_embed_dim=960,
                 smiles_seq_len=220, smiles_embed_dim=1024,
                 esm_device=None, esm_use_direct_load=False,
                 use_dual_gpu=False):

        self._smiles_extractor = SmilesFeatureExtractor(
            smiles_vocab_path, smiles_model_path,
            seq_len=smiles_seq_len, embed_dim=smiles_embed_dim,
        )

        self._protein_extractor = ProteinFeatureExtractor(
            model_name=esm_model_name, embed_dim=esm_embed_dim,
            device=esm_device, use_direct_load=esm_use_direct_load,
            use_dual_gpu=use_dual_gpu,
        )

    # ── public API ──────────────────────────────────────────────

    def extract(self, sequence=None, smiles=None):
        """Extract protein, SMILES, and combined embeddings.

        Parameters
        ----------
        sequence : str or list[str] or None
            Protein amino-acid sequence(s).  May be None if only SMILES
            features are needed.
        smiles : str or list[str] or None
            SMILES string(s).  May be None if only protein features
            are needed.

        Returns
        -------
        dict
            ``{"protein_embedding": np.ndarray or None,
                "smiles_embedding": np.ndarray or None,
                "combined_embedding": np.ndarray or None}``

            ``combined_embedding`` is ``np.concatenate([protein, smiles], axis=1)``
            when both are present.  When only one modality is provided the
            missing one is ``None`` and ``combined_embedding`` is also ``None``.
        """
        protein_emb = None
        smiles_emb = None

        if sequence is not None:
            protein_emb = self._protein_extractor.extract(sequence)

        if smiles is not None:
            smiles_emb = self._smiles_extractor.extract(smiles)

        combined = None
        if protein_emb is not None and smiles_emb is not None:
            # Dimension safeguard
            if protein_emb.shape[0] != smiles_emb.shape[0]:
                min_n = min(protein_emb.shape[0], smiles_emb.shape[0])
                protein_emb = protein_emb[:min_n]
                smiles_emb = smiles_emb[:min_n]
            combined = np.concatenate((smiles_emb, protein_emb), axis=1)

        elif protein_emb is not None:
            combined = protein_emb

        elif smiles_emb is not None:
            combined = smiles_emb

        return {
            "protein_embedding": protein_emb,
            "smiles_embedding": smiles_emb,
            "combined_embedding": combined,
        }
