import numpy as np
from pathlib import Path

FEATURE_DIM = 46


class MSA2DFeatureExtractor:

    def __init__(
        self,
        feat_dir="data/msa2d/features",
    ):
        self._feat_dir = Path(feat_dir)

    def extract(self, protein_id):

        fp = self._feat_dir / f"{protein_id}.npy"

        if not fp.exists():
            return None

        return np.load(fp)

    def extract_many(self, protein_ids):

        feats = []

        for pid in protein_ids:

            f = self.extract(pid)

            if f is not None:
                feats.append(f)

        if not feats:
            return None

        return np.stack(feats)

    @property
    def feature_dim(self):
        return FEATURE_DIM

