"""
MSA2D Full Dataset
"""

import hashlib
import sys

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset


_ROOT = Path(__file__).resolve().parent.parent

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


MASTER_DIR = _ROOT / "data" / "master"
CONTACT_DIR = _ROOT / "data" / "msa_full" / "contact_maps"


def make_protein_id(sequence: str):

    return hashlib.sha256(
        str(sequence).encode()
    ).hexdigest()[:16]


def load_contact_map(pid):

    path = CONTACT_DIR / f"{pid}.npy"

    if not path.exists():
        return None

    try:

        cm = np.load(path).astype(np.float32)

        if (
            cm.ndim == 2
            and cm.shape[0] == cm.shape[1]
        ):
            return cm

    except Exception:
        pass

    return None


class MSA2DFullDataset(Dataset):

    def __init__(
        self,
        task="kcat",
        split=None,
        pad_to=256,
    ):

        self.task = task
        self.split = split
        self.pad_to = pad_to

        self.df = pd.read_csv(
            MASTER_DIR / f"{task}.csv"
        )

        if split is not None:

            self.df = self.df[
                self.df["split"] == split
            ].reset_index(drop=True)

        self.protein_ids = [
            make_protein_id(seq)
            for seq in self.df["sequence"]
        ]

        print(
            f"Dataset: {task}/{split} "
            f"{len(self.df)} samples"
        )

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        pid = self.protein_ids[idx]

        cm = load_contact_map(pid)

        if cm is None:

            cm = np.zeros(
                (
                    self.pad_to,
                    self.pad_to,
                ),
                dtype=np.float32,
            )

        else:

            cm = self._pad_or_crop(
                cm,
                self.pad_to,
            )

        sample = {

            "sample_id":
                str(row["sample_id"]),

            "protein_id":
                pid,

            "sequence":
                str(row["sequence"]),

            "smiles":
                str(row["smiles"]),

            "target":
                torch.tensor(
                    float(row["target"]),
                    dtype=torch.float32,
                ),

            "contact_map":
                torch.from_numpy(cm)
                .unsqueeze(0),
        }

        return sample

    @staticmethod
    def _pad_or_crop(
        cm,
        target_size,
    ):

        L = cm.shape[0]

        if L == target_size:
            return cm

        if L > target_size:

            start = (
                L - target_size
            ) // 2

            return cm[
                start:start+target_size,
                start:start+target_size,
            ]

        result = np.zeros(
            (
                target_size,
                target_size,
            ),
            dtype=np.float32,
        )

        offset = (
            target_size - L
        ) // 2

        result[
            offset:offset+L,
            offset:offset+L,
        ] = cm

        return result

