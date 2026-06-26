#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json

import joblib
import numpy as np

from config import KM_MODEL_PATH, KCAT_MODEL_PATH
from src.features.extractor import extract_joint_features

# enzyme-model feature order:  [SMILES(1024) | Protein(960)]
# Kinora-main feature order:   [Protein(960) | SMILES(1024)]
_PROTEIN_DIM = 960


def _reorder_features(features: np.ndarray) -> np.ndarray:
    """Swap [Protein|SMILES] → [SMILES|Protein] to match enzyme-model format."""
    return np.concatenate([features[:, _PROTEIN_DIM:], features[:, :_PROTEIN_DIM]], axis=1)


def parse_args():
    parser = argparse.ArgumentParser(description="Run enzyme-model baseline inference for Km and kcat.")
    parser.add_argument("--protein", type=str, required=True, help="Protein sequence.")
    parser.add_argument("--smiles", type=str, required=True, help="Substrate SMILES.")
    return parser.parse_args()


def main():
    args = parse_args()

    if not KM_MODEL_PATH.exists():
        raise FileNotFoundError(f"KM model not found: {KM_MODEL_PATH}")
    if not KCAT_MODEL_PATH.exists():
        raise FileNotFoundError(f"Kcat model not found: {KCAT_MODEL_PATH}")

    km_model = joblib.load(str(KM_MODEL_PATH))
    kcat_model = joblib.load(str(KCAT_MODEL_PATH))
    print(f"Loaded KM model:  {type(km_model).__name__}")
    print(f"Loaded Kcat model: {type(kcat_model).__name__}")

    features = extract_joint_features([args.smiles], [args.protein]).astype(np.float32)
    features = _reorder_features(features)

    log10_km = float(km_model.predict(features)[0])
    log10_kcat = float(kcat_model.predict(features)[0])

    result = {
        "input": {
            "protein": args.protein[:80] + ("..." if len(args.protein) > 80 else ""),
            "smiles": args.smiles,
        },
        "prediction": {
            "log10_km": log10_km,
            "km": float(10 ** log10_km),
            "log10_kcat": log10_kcat,
            "kcat": float(10 ** log10_kcat),
            "kcat_over_km": float(10 ** (log10_kcat - log10_km)),
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
