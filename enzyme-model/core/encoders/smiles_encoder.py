"""
Shared SMILES Encoder

Consolidates smiles_to_vec implementations from:
  - predict_kcat.py (lines 71-138)
  - predict_km.py (lines 233-325)
  - train/train_kcat.py (lines 127-206)
  - train/train_km.py (lines 112-207)

Usage:
    from core.encoders.smiles_encoder import SmilesEncoder

    encoder = SmilesEncoder(vocab_path, model_path)
    embedding = encoder.encode("CCO")
    # or
    embeddings = encoder.encode(["CCO", "CCCO"])
"""

import torch
import numpy as np
from core.shared_smiles.model import TrfmSeq2seq
from core.shared_smiles.vocab_builder import WordVocab
from core.shared_smiles.common import split


class SmilesEncoder:
    """Unified SMILES string → embedding vector encoder.

    Accepts a vocabulary file and a pre-trained SMILES Transformer checkpoint.
    Once loaded, encodes arbitrary SMILES strings into 1024-dim vectors.
    """

    def __init__(self, vocab_path, model_path, seq_len=220, embed_dim=1024, device=None):
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        # Resolve device (auto CUDA if available)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Load vocabulary
        self.vocab = WordVocab.load_vocab(vocab_path)

        # Build model architecture
        trfm_hidden_size = 256
        self.trfm = TrfmSeq2seq(len(self.vocab), trfm_hidden_size, len(self.vocab), 4)

        # Load model weights
        self.trfm.load_state_dict(torch.load(model_path, map_location=self.device))
        self.trfm.to(self.device)
        self.trfm.eval()

    # ── public API ──────────────────────────────────────────────

    def encode(self, smiles_input):
        """Encode SMILES string(s) into a (N, embed_dim) numpy array.

        Parameters
        ----------
        smiles_input : str or list[str]
            One or more SMILES strings.

        Returns
        -------
        np.ndarray
            Shape (1, embed_dim) for a single string, or (N, embed_dim) for a list.
        """
        if isinstance(smiles_input, str):
            smiles_input = [smiles_input]
            single = True
        else:
            single = False

        x_split = [split(sm) for sm in smiles_input]

        pad_index = 0
        unk_index = 1
        eos_index = 2
        sos_index = 3

        x_id = []
        for sm in x_split:
            if len(sm) > (self.seq_len - 2):
                sm = sm[:(self.seq_len // 2) - 1] + sm[-(self.seq_len // 2 - 1):]
            ids = [self.vocab.stoi.get(token, unk_index) for token in sm]
            ids = [sos_index] + ids + [eos_index]
            padding = [pad_index] * (self.seq_len - len(ids))
            ids.extend(padding)
            x_id.append(ids)

        xid = torch.tensor(x_id).to(self.device)

        with torch.no_grad():
            X = self.trfm.encode(xid.t())

        if isinstance(X, torch.Tensor):
            if X.device.type != 'cpu':
                X = X.cpu()
            X = X.numpy()

        return X
