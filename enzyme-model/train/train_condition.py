#!/usr/bin/env python3
"""Condition 模型训练脚本。

用法:
    python train/train_condition.py --task kcat
    python train/train_condition.py --task km
    python train/train_condition.py --task kcat --config path/to/config.yaml
    python train/train_condition.py --task kcat --debug    # smoke test
"""

import json
import sys
from pathlib import Path

# ── Ensure project root is on sys.path ──
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# 确保 lightgbm 在 torch 之前加载，避免 OpenMP (libomp) ABI 冲突导致的 segfault
import lightgbm  # noqa: F401, E402

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Pre-load WordVocab so pickle can resolve it (required by SmilesEncoder)
from core.shared_smiles.vocab_builder import WordVocab  # noqa: F401, E402

from configs.config_loader import load_config
from models.condition.model import ConditionModel
from evaluate.evaluate_condition import evaluate_regression

# ── Project paths ───────────────────────────────────────────────
PROJECT_ROOT = _ROOT
PRETRAINED_DIR = PROJECT_ROOT / "data" / "pretrained"

# ── Constants ───────────────────────────────────────────────────
VOCAB_FILENAME = "smiles_vocab.pkl"
SMILES_MODEL_FILENAME = "smiles_transformer.pkl"


def set_seed(seed: int):
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_condition_csv(csv_path: Path, smiles_col: str, target_col: str, nrows: int | None = None):
    """Load condition dataset from CSV.

    Parameters
    ----------
    csv_path : Path
    smiles_col : str
    target_col : str
    nrows : int or None
        Number of rows to load (for debug mode).  None = all rows.

    Returns
    -------
    sequences : list[str]
    smiles_list : list[str]
    temperature : np.ndarray
    ph : np.ndarray
    labels : np.ndarray
    """
    df = pd.read_csv(csv_path, index_col=0, nrows=nrows)
    print(f"加载数据集: {csv_path}, 共 {len(df)} 条")

    sequences = df["sequence"].tolist()
    smiles_list = df[smiles_col].tolist()
    temperature = df["temperature"].values.astype(float)
    ph = df["ph"].values.astype(float)
    labels = df[target_col].values.astype(float)
    sample_ids = df.index.astype(str).tolist()

    print(f"  temperature 缺失: {np.isnan(temperature).sum()} ({np.isnan(temperature).sum()/len(df)*100:.2f}%)")
    print(f"  ph 缺失:          {np.isnan(ph).sum()} ({np.isnan(ph).sum()/len(df)*100:.2f}%)")
    print(f"  标签范围: {labels.min():.4f} – {labels.max():.4f}")

    return sequences, smiles_list, temperature, ph, labels, sample_ids


def extract_features(cache_dir, sequences, smiles_list,
                     esm_model_name, esm_embed_dim,
                     smiles_seq_len, smiles_embed_dim,
                     temperature, ph,
                     temp_fill, ph_fill):
    """Extract combined features: ESMC + SMILES + condition (temp, pH)."""
    from core.feature_extractors import CombinedFeatureExtractor
    from core.feature_extractors.condition_feature_extractor import (
        ConditionFeatureExtractor,
    )
    import pickle

    # Try loading cached features
    cache_dir.mkdir(parents=True, exist_ok=True)
    smiles_pkl = cache_dir / "smiles_embeddings.npy"
    protein_pkl = cache_dir / "protein_embeddings.npy"

    if smiles_pkl.exists() and protein_pkl.exists():
        print("从缓存加载特征...")
        smiles_emb = np.load(smiles_pkl)
        protein_emb = np.load(protein_pkl)
    else:
        print("开始特征提取...")
        vocab_path = PRETRAINED_DIR / VOCAB_FILENAME
        model_path = PRETRAINED_DIR / SMILES_MODEL_FILENAME

        extractor = CombinedFeatureExtractor(
            smiles_vocab_path=str(vocab_path),
            smiles_model_path=str(model_path),
            esm_model_name=esm_model_name,
            esm_embed_dim=esm_embed_dim,
            smiles_seq_len=smiles_seq_len,
            smiles_embed_dim=smiles_embed_dim,
            use_dual_gpu=True,
        )

        result = extractor.extract(sequence=sequences, smiles=smiles_list)
        smiles_emb = result["smiles_embedding"]
        protein_emb = result["protein_embedding"]

        np.save(smiles_pkl, smiles_emb)
        np.save(protein_pkl, protein_emb)
        print(f"特征已缓存至: {cache_dir}")

    print(f"  SMILES 特征:   {smiles_emb.shape}")
    print(f"  蛋白质特征:   {protein_emb.shape}")

    # Condition features
    print("处理 condition 特征 (temp, pH)...")
    cond_extractor = ConditionFeatureExtractor(temp_fill=temp_fill, ph_fill=ph_fill)
    cond_feat = cond_extractor.fit_transform(temperature, ph)
    print(f"  Condition 特征: {cond_feat.shape}")

    # Concatenate all features
    full_feat = np.concatenate([smiles_emb, protein_emb, cond_feat], axis=1)
    print(f"  总特征维度: {full_feat.shape[1]} ({full_feat.shape})")

    return full_feat


def save_results(output_dir, model, metrics, model_filename, results_filename):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Metrics
    results_path = output_dir / results_filename
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"结果已保存: {results_path}")

    # Model
    model_path = output_dir / model_filename
    model.save(str(model_path))
    print(f"模型已保存: {model_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Condition 模型训练")
    parser.add_argument("--task", type=str, required=True, choices=["kcat", "km"],
                        help="训练任务: kcat 或 km")
    parser.add_argument("--config", type=str, default=None,
                        help="配置文件路径（覆盖默认）")
    parser.add_argument("--debug", action="store_true",
                        help="Smoke test mode: 仅使用前 100 条样本")
    args = parser.parse_args()

    # Config path
    cfg_path = args.config or str(PROJECT_ROOT / "configs" / "condition" / f"{args.task}.yaml")
    cfg = load_config(cfg_path)
    print(f"加载配置: {cfg_path}")

    # Extract config
    random_seed = cfg["model"]["random_state"]
    test_size = cfg["train"]["test_size"]

    feats_cfg = cfg["features"]
    esm_model_name = feats_cfg["esm_model_name"]
    esm_embed_dim = feats_cfg["esm_embed_dim"]
    smiles_seq_len = feats_cfg["smiles_seq_len"]
    smiles_embed_dim = feats_cfg["smiles_embed_dim"]

    imp = cfg["imputation"]
    temp_fill = imp.get("temp_fill")
    ph_fill = imp.get("ph_fill")

    file_cfg = cfg["files"]
    model_save_filename = file_cfg["model_save"]
    results_save_filename = file_cfg["results_save"]

    dset = cfg["dataset"]
    smiles_col = dset["smiles_col"]
    target_col = dset["target_col"]
    dataset_path = PROJECT_ROOT / dset["path"]

    output_dir = PROJECT_ROOT / "artifacts" / "condition" / args.task

    set_seed(random_seed)
    print(f"开始 {args.task} Condition 模型训练")

    # 1. Load data (debug mode: only 100 rows)
    nrows = 100 if args.debug else None
    sequences, smiles_list, temperature, ph, labels, sample_ids = load_condition_csv(
        dataset_path, smiles_col, target_col, nrows=nrows
    )
    if args.debug:
        print(f"[DEBUG] Smoke test: {len(sequences)} 条样本")

    # 2. Extract features (task-specific cache to avoid cross-task size mismatch)
    cache_dir = PROJECT_ROOT / "data" / "features" / "condition" / args.task
    print("[DEBUG] 使用真实特征 (SMILES + ESMC + Condition)")
    features = extract_features(
        cache_dir, sequences, smiles_list,
        esm_model_name, esm_embed_dim,
        smiles_seq_len, smiles_embed_dim,
        temperature, ph,
        temp_fill, ph_fill,
    )
    labels = np.asarray(labels)
    sample_ids = np.array(sample_ids)

    # 3. Train / test split
    X_train, X_test, y_train, y_test, _, sids_test = train_test_split(
        features, labels, sample_ids, test_size=test_size, random_state=random_seed
    )
    print(f"数据集划分: 训练集 {len(X_train)} 条, 测试集 {len(X_test)} 条")

    # 4. Train model
    print("开始训练 LightGBM...")
    model = ConditionModel()
    model.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = model.predict(X_test)
    metrics = evaluate_regression(y_test, y_pred)

    print(f"\n最终测试集性能:")
    print(f"  R²:      {metrics['r2']:.4f}")
    print(f"  MAE:     {metrics['mae']:.4f}")
    print(f"  RMSE:    {metrics['rmse']:.4f}")
    print(f"  Pearson: {metrics['pearson_r']:.4f} (p={metrics['pearson_p']:.2e})")

    # 6. Save
    save_results(output_dir, model, metrics,
                 model_save_filename, results_save_filename)

    # 7. 导出 predictions.csv
    pred_df = pd.DataFrame({
        "sample_id": sids_test,
        "split": "test",
        "y_true": y_test,
        "y_pred": y_pred,
    })
    pred_path = output_dir / "predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"预测结果已保存: {pred_path}")

    print("训练完成")


if __name__ == "__main__":
    main()
