"""
SMILES Transformer — TrfmSeq2seq model definition.

Migrated from kcat/src/models/smiles_transformer.py.
Internal imports updated: kcat/src/ → core/shared_smiles/.
"""

import math
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from core.shared_smiles.vocab_builder import WordVocab
from core.shared_smiles.dataset import Seq2seqDataset

PAD = 0
UNK = 1
EOS = 2
SOS = 3
MASK = 4


class PositionalEncoding(nn.Module):
    "Implement the PE function. No batch support?"
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)  # (T,H)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class TrfmSeq2seq(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, n_layers, dropout=0.1):
        super(TrfmSeq2seq, self).__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(in_size, hidden_size)
        self.pe = PositionalEncoding(hidden_size, dropout)
        self.trfm = nn.Transformer(
            d_model=hidden_size, nhead=4,
            num_encoder_layers=n_layers, num_decoder_layers=n_layers,
            dim_feedforward=hidden_size, batch_first=True,
        )
        self.out = nn.Linear(hidden_size, out_size)

    def forward(self, src):
        # src: (T,B) -> (B,T) for batch_first=True
        src = src.transpose(0, 1)  # (B,T)
        embedded = self.embed(src)  # (B,T,H)
        embedded = self.pe(embedded)  # (B,T,H)
        hidden = self.trfm(embedded, embedded)  # (B,T,H)
        out = self.out(hidden)  # (B,T,V)
        out = out.log_softmax(dim=2)  # (B,T,V)
        return out.transpose(0, 1)  # (T,B,V)

    def _encode(self, src):
        # src: (T,B) -> (B,T)
        src = src.transpose(0, 1)  # (B,T)
        embedded = self.embed(src)  # (B,T,H)
        embedded = self.pe(embedded)  # (B,T,H)
        output = embedded
        for i in range(self.trfm.encoder.num_layers - 1):
            output = self.trfm.encoder.layers[i](output, None)  # (B,T,H)
        penul = output.detach().numpy()
        output = self.trfm.encoder.layers[-1](output, None)  # (B,T,H)
        if self.trfm.encoder.norm:
            output = self.trfm.encoder.norm(output)  # (B,T,H)
        output = output.detach().numpy()
        # mean, max, first*2
        return np.hstack([
            np.mean(output, axis=1), np.max(output, axis=1),
            output[:, 0, :], penul[:, 0, :],
        ])  # (B, 4H)

    def encode(self, src):
        # src: (T,B)
        batch_size = src.shape[1]
        if batch_size <= 100:
            return self._encode(src)
        else:
            print('There are {:d} molecules. It will take a little time.'.format(batch_size))
            st, ed = 0, 100
            out = self._encode(src[:, st:ed])  # (B,4H)
            while ed < batch_size:
                st += 100
                ed += 100
                out = np.concatenate([out, self._encode(src[:, st:ed])], axis=0)
            return out
