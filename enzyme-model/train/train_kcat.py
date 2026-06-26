import torch
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import random
import pickle
# import math  # no longer needed (target is already log10)
from pathlib import Path
from sklearn.ensemble import ExtraTreesRegressor
import joblib
from itertools import product
import sys

# ── Ensure project root is on sys.path ──
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.feature_extractors import CombinedFeatureExtractor
from configs.config_loader import load_config

# ================== Project paths ================== #
PROJECT_ROOT = _ROOT
DATA_DIR = PROJECT_ROOT / "data" / "master"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "baseline" / "kcat"
FEATURE_CACHE_DIR = PROJECT_ROOT / "data" / "features" / "baseline"
PRETRAINED_DIR = PROJECT_ROOT / "data" / "pretrained"
MODEL_DIR = OUTPUT_DIR

# ================== Shared file names ================== #
VOCAB_FILENAME = "smiles_vocab.pkl"
SMILES_MODEL_FILENAME = "smiles_transformer.pkl"


# ================== Helper Functions ================== #
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def find_dataset_path(dataset_filename):
    dataset_path = DATA_DIR / dataset_filename
    if not dataset_path.exists():
        raise FileNotFoundError(f"数据集 '{dataset_filename}' 未找到，路径: {dataset_path}")
    return dataset_path


# ================== Feature Extraction ================== #
def extract_combined_features(Smiles, sequence, cache_dir, pkl_filename,
                              esm_model_name, esm_embed_dim,
                              smiles_seq_len, smiles_embed_dim,
                              force_recompute=False):
    feature_pkl_path = cache_dir / pkl_filename

    if not force_recompute and feature_pkl_path.exists():
        print(f"从缓存加载特征: {feature_pkl_path}")
        with open(feature_pkl_path, "rb") as f:
            feature = pickle.load(f)
        print(f"特征加载完成: {feature.shape}")
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

        features = extractor.extract(sequence=sequence, smiles=Smiles)
        feature = features["combined_embedding"]

        if feature is None:
            raise RuntimeError("特征提取失败")

        if feature.shape[1] != (smiles_embed_dim + esm_embed_dim):
            print(f"警告: 特征维度不匹配 - 预期{smiles_embed_dim + esm_embed_dim}，实际{feature.shape[1]}")

        print(f"特征拼接完成: {feature.shape}")

        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(feature_pkl_path, "wb") as f:
            pickle.dump(feature, f)
        print(f"特征已保存至: {feature_pkl_path}")

    return feature


# ================== Data Loading and Preprocessing ================== #
def load_kcat_csv(csv_path):
    """Master Dataset 格式: sequence, smiles, target (已为 log10)"""
    import pandas as pd

    csv_path = Path(csv_path)
    print(f"加载数据集: {csv_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"数据集文件不存在: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    print(f"原始数据: {len(df)}条")

    sequence_list, smiles_list, label_list, sid_list = [], [], [], []

    for _, row in df.iterrows():
        try:
            seq = str(row["sequence"])
            smi = str(row["smiles"])
            val = float(row["target"])
            sid = str(row.get("sample_id", ""))
        except (KeyError, ValueError, TypeError):
            continue

        if "." not in smi:
            sequence_list.append(seq)
            smiles_list.append(smi)
            label_list.append(val)
            sid_list.append(sid)

    print(f"过滤后数据: {len(label_list)}条")
    if not label_list:
        raise ValueError("过滤后无有效数据")

    print(f"标签范围: {min(label_list):.4f} - {max(label_list):.4f}")
    return sequence_list, smiles_list, label_list, sid_list


# ================== Model Training and Evaluation ================== #
def perform_grid_search(features_train, labels_train, features_test, labels_test,
                        param_grid, random_seed, n_jobs):
    print("\n开始参数网格搜索...")
    best_r2 = -float("inf")
    best_params = None
    best_model = None

    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]
    total_combinations = len(param_combinations)

    for i, params in enumerate(param_combinations):
        print(f"参数组合 [{i+1}/{total_combinations}]: {params}")
        model = ExtraTreesRegressor(
            **params,
            random_state=random_seed,
            n_jobs=n_jobs,
            verbose=0
        )

        model.fit(features_train, labels_train)
        predictions = model.predict(features_test)
        current_r2 = r2_score(labels_test, predictions)
        print(f"R2: {current_r2:.4f}")

        if current_r2 > best_r2:
            print(f"更新最佳R2: {current_r2:.4f} (之前: {best_r2:.4f})")
            best_r2 = current_r2
            best_params = params
            best_model = model
        else:
            print(f"未超过当前最佳R2: {best_r2:.4f}")

    print("\n网格搜索完成")

    if best_model is None:
        raise RuntimeError("未找到有效模型")

    print(f"最佳参数: {best_params}")
    print(f"最佳R2: {best_r2:.4f}")

    best_predictions = best_model.predict(features_test)
    final_metrics = {
        "mse": mean_squared_error(labels_test, best_predictions),
        "mae": mean_absolute_error(labels_test, best_predictions),
        "r2": best_r2
    }

    return best_model, best_params, final_metrics


# ================== Results Saving ================== #
def save_results(output_dir, model, params, metrics, model_filename, results_filename):
    output_dir = Path(output_dir) if not isinstance(output_dir, Path) else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / results_filename
    results_data = {
        "best_params": params,
        "test_mse": metrics["mse"],
        "test_mae": metrics["mae"],
        "test_r2": metrics["r2"]
    }
    try:
        with open(results_path, "w") as f:
            json.dump(results_data, f, indent=4)
        print(f"结果已保存: {results_path}")
    except Exception as e:
        print(f"保存结果失败: {e}")

    model_path = output_dir / model_filename
    try:
        joblib.dump(model, model_path)
        print(f"模型已保存: {model_path}")
    except Exception as e:
        print(f"保存模型失败: {e}")


# ================== Main Execution ================== #
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Kcat预测模型训练")
    parser.add_argument("--config", type=str, default=None,
                        help="配置文件路径（默认: configs/baseline/kcat.yaml）")
    parser.add_argument("--dataset", type=str, default=None,
                        help="数据CSV文件路径（覆盖配置文件中的路径）")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录路径（覆盖默认路径）")
    args = parser.parse_args()

    # 加载配置
    config_path = args.config or str(PROJECT_ROOT / "configs" / "baseline" / "kcat.yaml")
    cfg = load_config(config_path)

    print(f"加载配置: {config_path}")

    # 提取所有超参数
    test_size = cfg["train"]["test_size"]
    random_seed = cfg["model"]["random_state"]
    n_jobs = cfg["model"]["n_jobs"]
    param_grid = cfg["model"]["param_grid"]

    feats_cfg = cfg["features"]
    esm_model_name = feats_cfg["esm_model_name"]
    esm_embed_dim = feats_cfg["esm_embed_dim"]
    smiles_seq_len = feats_cfg["smiles_seq_len"]
    smiles_embed_dim = feats_cfg["smiles_embed_dim"]

    file_cfg = cfg["files"]
    dataset_filename = file_cfg["dataset_filename"]
    features_pkl_filename = file_cfg["features_pkl"]
    model_save_filename = file_cfg["model_save"]
    results_save_filename = file_cfg["results_save"]

    effective_output_dir = OUTPUT_DIR
    if args.output_dir:
        effective_output_dir = Path(args.output_dir).resolve()
        print(f"使用命令行指定的输出目录: {effective_output_dir}")

    set_seed(random_seed)
    print("开始kcat值预测模型建模")

    # 1. 加载与预处理数据
    if args.dataset:
        dataset_path = Path(args.dataset)
        print(f"使用命令行指定的数据集: {dataset_path}")
    else:
        dataset_path = find_dataset_path(dataset_filename)
    sequence, Smiles, Label, sample_ids = load_kcat_csv(dataset_path)

    # 2. 提取特征
    FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    feature = extract_combined_features(
        Smiles, sequence, FEATURE_CACHE_DIR, features_pkl_filename,
        esm_model_name, esm_embed_dim, smiles_seq_len, smiles_embed_dim
    )
    Label = np.array(Label)
    sample_ids = np.array(sample_ids)

    # 3. 数据集划分
    features_train, features_test, labels_train, labels_test, _, sids_test = train_test_split(
        feature, Label, sample_ids, test_size=test_size, random_state=random_seed
    )
    print(f"数据集划分: 训练集{len(features_train)}条, 测试集{len(features_test)}条")

    # 4. 网格搜索与训练
    best_model, best_params, final_metrics = perform_grid_search(
        features_train, labels_train, features_test, labels_test,
        param_grid, random_seed, n_jobs
    )

    # 5. 显示最终结果
    print(f"\n最终测试集性能:")
    print(f"MSE: {final_metrics['mse']:.4f}")
    print(f"MAE: {final_metrics['mae']:.4f}")
    print(f"R2:  {final_metrics['r2']:.4f}")

    # 6. 保存结果与模型
    save_results(effective_output_dir, best_model, best_params, final_metrics,
                 model_save_filename, results_save_filename)

    # 7. 导出 predictions.csv
    import pandas as _pd
    best_preds = best_model.predict(features_test)
    pred_df = _pd.DataFrame({
        "sample_id": sids_test,
        "split": "test",
        "y_true": labels_test,
        "y_pred": best_preds,
    })
    pred_path = effective_output_dir / "predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"预测结果已保存: {pred_path}")


if __name__ == "__main__":
    main()
    print("脚本执行完毕")
