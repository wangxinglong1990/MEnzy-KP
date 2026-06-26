#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import KM_MODEL_PATH, KCAT_MODEL_PATH
from src.features.extractor import extract_joint_features

# enzyme-model feature order:  [SMILES(1024) | Protein(960)]
# Kinora-main feature order:   [Protein(960) | SMILES(1024)]
_PROTEIN_DIM = 960


def _reorder_features(features: np.ndarray) -> np.ndarray:
    """Swap [Protein|SMILES] → [SMILES|Protein] to match enzyme-model format."""
    return np.concatenate([features[:, _PROTEIN_DIM:], features[:, :_PROTEIN_DIM]], axis=1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch-predict Km and kcat from a CSV file."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="blast500.csv",
        help="Input CSV path (default: blast500.csv).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output CSV path (default: overwrite input file).",
    )
    parser.add_argument(
        "--seq-col",
        type=str,
        default="Enzyme",
        help="Protein sequence column name (default: Enzyme).",
    )
    parser.add_argument(
        "--smiles-col",
        type=str,
        default="Substrates",
        help="SMILES column name (default: Substrates).",
    )
    parser.add_argument(
        "--pred-kcat-col",
        type=str,
        default="Pred_kcat",
        help="Predicted kcat output column (default: Pred_kcat).",
    )
    parser.add_argument(
        "--pred-km-col",
        type=str,
        default="Pred_Km",
        help="Predicted Km output column (default: Pred_Km).",
    )
    parser.add_argument(
        "--pred-kcat-over-km-col",
        type=str,
        default="Pred_kcat_over_Km",
        help="Predicted kcat/Km output column (default: Pred_kcat_over_Km).",
    )
    parser.add_argument(
        "--pred-km-over-kcat-col",
        type=str,
        default="Pred_Km_over_kcat",
        help="Predicted Km/kcat output column (default: Pred_Km_over_kcat).",
    )
    return parser.parse_args()


def read_csv_auto(csv_path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(csv_path, encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to read CSV: {csv_path}\nLast error: {last_err}")


def build_models():
    if not KM_MODEL_PATH.exists():
        raise FileNotFoundError(f"KM model not found: {KM_MODEL_PATH}")
    if not KCAT_MODEL_PATH.exists():
        raise FileNotFoundError(f"Kcat model not found: {KCAT_MODEL_PATH}")

    km_model = joblib.load(str(KM_MODEL_PATH))
    kcat_model = joblib.load(str(KCAT_MODEL_PATH))
    print(f"Loaded KM model:  {type(km_model).__name__}")
    print(f"Loaded Kcat model: {type(kcat_model).__name__}")
    return km_model, kcat_model


def main():
    args = parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    output_path = Path(args.output).resolve() if args.output.strip() else input_path
    df = read_csv_auto(input_path)

    if args.seq_col not in df.columns:
        raise ValueError(f"Protein sequence column not found: {args.seq_col}. Available columns: {list(df.columns)}")
    if args.smiles_col not in df.columns:
        raise ValueError(f"SMILES column not found: {args.smiles_col}. Available columns: {list(df.columns)}")

    sequences = df[args.seq_col].astype(str).str.strip().tolist()
    smiles_list = df[args.smiles_col].astype(str).str.strip().tolist()

    km_model, kcat_model = build_models()

    print(f"Start prediction, total samples: {len(df)}")
    features = extract_joint_features(smiles_list, sequences).astype(np.float32)
    # Reorder columns: Kinora [Protein|SMILES] → enzyme-model [SMILES|Protein]
    features = _reorder_features(features)

    log10_km = km_model.predict(features)
    log10_kcat = kcat_model.predict(features)

    km = np.power(10.0, log10_km)
    kcat = np.power(10.0, log10_kcat)
    kcat_over_km = np.power(10.0, log10_kcat - log10_km)
    km_over_kcat = np.power(10.0, log10_km - log10_kcat)

    df[args.pred_kcat_col] = kcat
    df[args.pred_km_col] = km
    df[args.pred_km_over_kcat_col] = km_over_kcat
    df[args.pred_kcat_over_km_col] = kcat_over_km

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Prediction finished. Output written to: {output_path}")
    print(
        df[
            [
                args.pred_kcat_col,
                args.pred_km_col,
                args.pred_km_over_kcat_col,
                args.pred_kcat_over_km_col,
            ]
        ]
        .head(5)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
