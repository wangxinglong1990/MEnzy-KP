#!/usr/bin/env python3
"""Export predictions.csv for all trained models without retraining.

Usage:
    python scripts/export_predictions.py --task kcat
    python scripts/export_predictions.py --task km
"""
import sys, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import lightgbm
from core.shared_smiles.vocab_builder import WordVocab


def make_protein_id(seq: str) -> str:
    return hashlib.sha256(str(seq).encode()).hexdigest()[:16]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["kcat", "km"])
    args = parser.parse_args()
    task = args.task

    MASTER = _ROOT / "data" / "master"
    FEATURES = _ROOT / "data" / "features"
    PRETRAINED = _ROOT / "data" / "pretrained"

    # ── Load dataset ──
    df = pd.read_csv(MASTER / f"{task}.csv")
    print(f"Dataset: {len(df)} rows from {MASTER}/{task}.csv")

    sequences = df["sequence"].tolist()
    smiles_list = df["smiles"].tolist()
    labels = df["target"].values.astype(float)
    sample_ids = df["sample_id"].values
    splits = df["split"].values if "split" in df.columns else ["test"] * len(df)

    # ── ESM + SMILES features ──
    cache_dir = FEATURES / "protein" / task
    cache_dir.mkdir(parents=True, exist_ok=True)
    smiles_cache = cache_dir / "smiles_embeddings.npy"
    protein_cache = cache_dir / "esm_embeddings.npy"

    if smiles_cache.exists() and protein_cache.exists():
        print("Loading cached features...")
        smiles_emb = np.load(smiles_cache)
        protein_emb = np.load(protein_cache)
    else:
        print("Extracting features (this may take a while)...")
        from core.feature_extractors import CombinedFeatureExtractor
        from sklearn.model_selection import train_test_split
        extractor = CombinedFeatureExtractor(
            smiles_vocab_path=str(PRETRAINED / "smiles_vocab.pkl"),
            smiles_model_path=str(PRETRAINED / "smiles_transformer.pkl"),
            use_dual_gpu=False,
        )
        result = extractor.extract(sequence=sequences, smiles=smiles_list)
        smiles_emb = result["smiles_embedding"]
        protein_emb = result["protein_embedding"]
        np.save(smiles_cache, smiles_emb)
        np.save(protein_cache, protein_emb)

    print(f"  SMILES: {smiles_emb.shape}, ESM: {protein_emb.shape}")

    # ── MSA1D features ──
    msa_feats = np.zeros((len(df), 6), dtype=np.float32)
    msa_feat_dir = _ROOT / "data" / "msa" / "features"
    for i, pid in enumerate([make_protein_id(s) for s in sequences]):
        fpath = msa_feat_dir / f"{pid}.npy"
        if fpath.exists():
            msa_feats[i] = np.load(fpath)

    # ── Condition features ──
    temp = df["temperature"].values.astype(float)
    ph = df["ph"].values.astype(float)
    temp[np.isnan(temp)] = np.nanmedian(temp)
    ph[np.isnan(ph)] = np.nanmedian(ph)
    from sklearn.preprocessing import StandardScaler
    cond_scaler = StandardScaler()
    cond_feat = cond_scaler.fit_transform(np.column_stack([temp, ph]))

    # ── Feature matrices ──
    X_base = np.concatenate([smiles_emb, protein_emb], axis=1).astype(np.float32)
    X_cond = np.concatenate([smiles_emb, protein_emb, cond_feat], axis=1).astype(np.float32)
    X_msa1 = np.concatenate([smiles_emb, protein_emb, msa_feats], axis=1).astype(np.float32)

    print(f"  X_base: {X_base.shape}, X_cond: {X_cond.shape}, X_msa1: {X_msa1.shape}")

    # ── Load and predict ──
    from models.baseline.model import BaselineModel
    from models.condition.model import ConditionModel
    from models.msa1d.model import MSA1DModel

    models_cfg = [
        ("baseline", BaselineModel, X_base,
         _ROOT / "artifacts" / "baseline" / task / "kcat_predictor.joblib" if task == "kcat"
         else _ROOT / "artifacts" / "baseline" / task / "km_predictor.joblib"),
        ("condition", ConditionModel, X_cond,
         _ROOT / "artifacts" / "condition" / task / "condition_predictor.joblib"),
        ("msa1d", MSA1DModel, X_msa1,
         _ROOT / "artifacts" / "msa1d" / task / "msa1d_predictor.joblib"),
    ]

    for name, ModelClass, X, model_path in models_cfg:
        if not model_path.exists():
            print(f"  ⚠️ {name}: model not found at {model_path}, skipping")
            continue
        model = ModelClass.load(str(model_path))
        y_pred = model.predict(X)
        out_dir = model_path.parent
        pred_df = pd.DataFrame({
            "sample_id": sample_ids,
            "split": "test",
            "y_true": labels,
            "y_pred": y_pred,
        })
        # Only keep test split rows (or all if no split column)
        if "split" in df.columns:
            test_mask = splits == "test"
            pred_df = pred_df.iloc[test_mask].copy()
        pred_path = out_dir / "predictions.csv"
        pred_df.to_csv(pred_path, index=False)
        print(f"  ✅ {name}: {len(pred_df)} predictions → {pred_path}")

    print("\nDone. Run: python scripts/build_stacking_dataset.py --task " + task)


if __name__ == "__main__":
    main()
