"""Single and batch prediction service."""
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from api_services.model_loader import ModelService


def predict_single(protein: str, smiles: str) -> dict:
    svc = ModelService.get_instance()
    return svc.predict_single(protein, smiles)


def predict_csv(csv_path: str, seq_col: str, smiles_col: str) -> Path:
    """Run batch prediction on CSV, return path to output CSV."""
    df = pd.read_csv(csv_path)

    if seq_col not in df.columns:
        raise ValueError(f"Column '{seq_col}' not found. Available: {list(df.columns)}")
    if smiles_col not in df.columns:
        raise ValueError(f"Column '{smiles_col}' not found. Available: {list(df.columns)}")

    sequences = df[seq_col].astype(str).str.strip().tolist()
    smiles_list = df[smiles_col].astype(str).str.strip().tolist()

    svc = ModelService.get_instance()
    pred_log10 = svc.predict_batch(sequences, smiles_list)

    km_log10 = pred_log10[:, 0]
    kcat_log10 = pred_log10[:, 1]

    df["Pred_kcat"] = np.power(10.0, kcat_log10)
    df["Pred_Km"] = np.power(10.0, km_log10)
    df["Pred_kcat_over_Km"] = df["Pred_kcat"] / df["Pred_Km"]
    df["Pred_Km_over_kcat"] = df["Pred_Km"] / df["Pred_kcat"]

    output_path = Path(tempfile.mktemp(suffix=".csv", prefix="dlkin_batch_"))
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def csv_to_json_preview(csv_path: Path, max_rows: int = 500) -> dict:
    """Read CSV and return as JSON-serializable dict with columns and rows."""
    df = pd.read_csv(csv_path)
    rows = df.head(max_rows).fillna("").to_dict(orient="records")
    return {
        "columns": list(df.columns),
        "row_count": len(df),
        "rows": rows,
    }
