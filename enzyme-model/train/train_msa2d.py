#!/usr/bin/env python3
"""MSA2D 模型训练脚本。

用法:
    python train/train_msa2d.py --task kcat
    python train/train_msa2d.py --task km
"""
import json, sys, hashlib
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import lightgbm  # noqa: F401

from core.shared_smiles.vocab_builder import WordVocab  # noqa: F401

from configs.config_loader import load_config
from models.msa2d.model import MSA2DModel
from evaluate.evaluate_msa1d import evaluate_regression

PROJECT_ROOT = _ROOT
PRETRAINED_DIR = PROJECT_ROOT / "data" / "pretrained"
MSA2D_FEAT_DIR = PROJECT_ROOT / "data" / "msa2d" / "features"

VOCAB_FILENAME = "smiles_vocab.pkl"
SMILES_MODEL_FILENAME = "smiles_transformer.pkl"
MSA2D_DIM = 46


def set_seed(seed: int):
    import random, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_protein_id(seq: str) -> str:
    return hashlib.sha256(str(seq).encode()).hexdigest()[:16]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MSA2D 模型训练")
    parser.add_argument("--task", type=str, required=True, choices=["kcat", "km"])
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg_path = args.config or str(PROJECT_ROOT / "configs" / "msa2d" / f"{args.task}.yaml")
    cfg = load_config(cfg_path)
    print(f"加载配置: {cfg_path}")

    random_seed = cfg["model"]["random_state"]
    split_col = cfg["train"]["split_col"]

    file_cfg = cfg["files"]
    model_save_filename = file_cfg["model_save"]
    results_save_filename = file_cfg["results_save"]

    dset_cfg = cfg["dataset"]
    dataset_path = PROJECT_ROOT / dset_cfg["path"]

    output_dir = PROJECT_ROOT / "artifacts" / "msa2d" / args.task

    set_seed(random_seed)
    print(f"开始 {args.task} MSA2D 模型训练")

    df = pd.read_csv(dataset_path)
    print(f"数据集: {len(df)} 条")

    sequences = df["sequence"].tolist()
    smiles_list = df["smiles"].tolist()
    protein_ids = [make_protein_id(s) for s in sequences]
    labels = df["target"].values.astype(float)
    splits = df[split_col].values
    sample_ids = df["sample_id"].values

    # Load ESM + SMILES features
    from core.feature_extractors import CombinedFeatureExtractor
    feats_cfg = cfg["features"]
    esm_embed_dim = feats_cfg["esm_embed_dim"]
    smiles_embed_dim = feats_cfg["smiles_embed_dim"]

    cache_dir = PROJECT_ROOT / "data" / "features" / "protein" / args.task
    cache_dir.mkdir(parents=True, exist_ok=True)
    smiles_cache = cache_dir / "smiles_embeddings.npy"
    protein_cache = cache_dir / "esm_embeddings.npy"

    if smiles_cache.exists() and protein_cache.exists():
        print("从缓存加载 ESM + SMILES 特征...")
        smiles_emb = np.load(smiles_cache)
        protein_emb = np.load(protein_cache)
    else:
        print("提取 ESM + SMILES 特征...")
        extractor = CombinedFeatureExtractor(
            smiles_vocab_path=str(PRETRAINED_DIR / VOCAB_FILENAME),
            smiles_model_path=str(PRETRAINED_DIR / SMILES_MODEL_FILENAME),
            esm_model_name=feats_cfg["esm_model_name"],
            esm_embed_dim=esm_embed_dim,
            smiles_seq_len=feats_cfg["smiles_seq_len"],
            smiles_embed_dim=smiles_embed_dim,
            use_dual_gpu=True,
        )
        result = extractor.extract(sequence=sequences, smiles=smiles_list)
        smiles_emb = result["smiles_embedding"]
        protein_emb = result["protein_embedding"]
        np.save(smiles_cache, smiles_emb)
        np.save(protein_cache, protein_emb)

    print(f"  SMILES: {smiles_emb.shape}, ESM: {protein_emb.shape}")

    # Load MSA2D features
    print("加载 MSA2D 特征...")
    msa2d_feats = []
    valid_mask = np.ones(len(protein_ids), dtype=bool)
    for i, pid in enumerate(protein_ids):
        feat_path = MSA2D_FEAT_DIR / f"{pid}.npy"
        if feat_path.exists():
            msa2d_feats.append(np.load(feat_path))
        else:
            msa2d_feats.append(np.zeros(MSA2D_DIM, dtype=np.float32))
            valid_mask[i] = False

    msa2d_feats = np.stack(msa2d_feats, axis=0)
    print(f"  MSA2D features present: {valid_mask.sum()}/{len(protein_ids)}")

    # Concat: SMILES(1024) + ESM(960) + MSA2D(46) = 2030
    features = np.concatenate([smiles_emb, protein_emb, msa2d_feats], axis=1).astype(np.float32)
    total_dim = smiles_embed_dim + esm_embed_dim + MSA2D_DIM
    print(f"  特征矩阵: {features.shape} ({smiles_embed_dim}+{esm_embed_dim}+{MSA2D_DIM}={total_dim})")

    # Split
    train_mask = (splits == "train") & valid_mask
    test_mask = (splits == "test") & valid_mask

    X_train, y_train = features[train_mask], labels[train_mask]
    X_test, y_test = features[test_mask], labels[test_mask]
    print(f"训练集: {len(X_train)} 条, 测试集: {len(X_test)} 条")
    print(f"特征维度: {X_train.shape[1]}")

    if len(X_train) == 0:
        print("错误: 训练集为空。请先生成 MSA2D 特征。")
        sys.exit(1)

    # Train
    print("开始训练 LightGBM...")
    model = MSA2DModel()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    metrics = evaluate_regression(y_test, y_pred)

    print(f"\n最终测试集性能:")
    print(f"  R²:      {metrics['r2']:.4f}")
    print(f"  MAE:     {metrics['mae']:.4f}")
    print(f"  RMSE:    {metrics['rmse']:.4f}")
    print(f"  Pearson: {metrics['pearson_r']:.4f} (p={metrics['pearson_p']:.2e})")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / results_save_filename
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"结果已保存: {results_path}")

    model_path = output_dir / model_save_filename
    model.save(str(model_path))
    print(f"模型已保存: {model_path}")

    # Predictions export
    test_sids = df.loc[test_mask, "sample_id"].values
    pred_df = pd.DataFrame({
        "sample_id": test_sids, "split": "test",
        "y_true": y_test, "y_pred": y_pred,
    })
    pred_path = output_dir / "predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"预测结果已保存: {pred_path}")
    print("训练完成")


if __name__ == "__main__":
    main()
