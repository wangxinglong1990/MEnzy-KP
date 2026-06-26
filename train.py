#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Kinora-main 模型训练 — sklearn ExtraTreesRegressor

为 KM 和 Kcat 各自训练一个 ExtraTreesRegressor，使用 Grid Search 搜索最佳超参。
特征：ESMC 蛋白质嵌入 + SMILES Transformer 底物嵌入 → 1984维 combined embedding。

用法:
    python train.py --task both                    # 训练 KM + Kcat
    python train.py --task km                      # 只训练 KM
    python train.py --task kcat --dataset my.csv   # 指定数据集
"""

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import KM_MODEL_PATH, KCAT_MODEL_PATH, MODELS_DIR, UNIFIED_DATASET_PATH
from src.features.extractor import extract_joint_features

# enzyme-model feature order: [SMILES(1024) | Protein(960)]
# Kinora extract_joint_features produces: [Protein(960) | SMILES(1024)]
_PROTEIN_DIM = 960

# ── Grid Search parameter grid ────────────────────────
PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 15, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}


# ── Helpers ───────────────────────────────────────────
def _reorder_features(features: np.ndarray) -> np.ndarray:
    """[Protein|SMILES] → [SMILES|Protein] to match enzyme-model training format."""
    return np.concatenate([features[:, _PROTEIN_DIM:], features[:, :_PROTEIN_DIM]], axis=1)


def load_csv(csv_path: str) -> tuple:
    """Load training CSV, return sequences, smiles, km_labels, kcat_labels (log10)."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path, low_memory=False)

    # Auto-detect column names
    col_map = {}
    for key in ["Sequence", "sequence", "Enzyme"]:
        if key in df.columns:
            col_map["seq"] = key
            break
    for key in ["Smiles", "smiles", "Substrates"]:
        if key in df.columns:
            col_map["smiles"] = key
            break
    for key in ["Km(M)", "km(m)", "Km", "km"]:
        if key in df.columns:
            col_map["km"] = key
            break
    for key in ["kcat(s^-1)", "kcat", "Kcat"]:
        if key in df.columns:
            col_map["kcat"] = key
            break

    required = ["seq", "smiles", "km", "kcat"]
    missing = [k for k in required if k not in col_map]
    if missing:
        raise ValueError(f"Columns not found: {missing}. Available: {list(df.columns)}")

    sequences, smiles_list, km_vals, kcat_vals = [], [], [], []
    for _, row in df.iterrows():
        try:
            seq = str(row[col_map["seq"]])
            smi = str(row[col_map["smiles"]])
            km = float(row[col_map["km"]])
            kcat = float(row[col_map["kcat"]])
            if pd.notna(seq) and pd.notna(smi) and pd.notna(km) and pd.notna(kcat):
                if len(seq) > 0 and len(smi) > 0 and km > 0 and kcat > 0:
                    sequences.append(seq)
                    smiles_list.append(smi)
                    km_vals.append(np.log10(km))
                    kcat_vals.append(np.log10(kcat))
        except (ValueError, TypeError):
            continue

    print(f"Valid samples: {len(sequences)}")
    return sequences, smiles_list, km_vals, kcat_vals


def grid_search_train(X_train, y_train, X_test, y_test, random_seed=42, n_jobs=-1):
    """Grid Search ExtraTreesRegressor. Returns best_model, best_params, metrics."""
    keys, values = zip(*PARAM_GRID.items())
    combinations = [dict(zip(keys, v)) for v in product(*values)]
    best_r2 = -float("inf")
    best_model, best_params = None, None

    for i, params in enumerate(combinations):
        model = ExtraTreesRegressor(**params, random_state=random_seed, n_jobs=n_jobs)
        model.fit(X_train, y_train)
        r2 = r2_score(y_test, model.predict(X_test))
        if r2 > best_r2:
            best_r2 = r2
            best_params = params
            best_model = model

    if best_model is None:
        raise RuntimeError("Grid Search found no valid model")

    preds = best_model.predict(X_test)
    return best_model, best_params, {
        "r2": best_r2,
        "mse": float(mean_squared_error(y_test, preds)),
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
    }


def train_task(sequences, smiles_list, labels, task_name, model_path, metrics_path,
               test_size=0.2, seed=42):
    """Train a single task (KM or Kcat)."""
    print(f"\n{'='*50}")
    print(f"Training {task_name} model")
    print(f"{'='*50}")

    print("Extracting features...")
    features = extract_joint_features(smiles_list, sequences).astype(np.float32)
    features = _reorder_features(features)  # → [SMILES|Protein]
    labels = np.array(labels)
    print(f"Feature shape: {features.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=seed
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    print("Grid Search in progress...")
    model, params, metrics = grid_search_train(X_train, y_train, X_test, y_test,
                                                random_seed=seed)

    print(f"Best params: {params}")
    print(f"R²: {metrics['r2']:.4f}, MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}")

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    with open(metrics_path, "w") as f:
        json.dump({"best_params": params, **metrics}, f, indent=2)

    print(f"Model saved: {model_path}")
    print(f"Metrics saved: {metrics_path}")
    return model, metrics


# ── Main ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train enzyme kinetics prediction models (sklearn)")
    parser.add_argument("--task", type=str, default="both",
                        choices=["km", "kcat", "both"], help="Training task")
    parser.add_argument("--dataset", type=str, default=str(UNIFIED_DATASET_PATH),
                        help="Training dataset CSV path")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default=str(MODELS_DIR),
                        help="Model output directory")
    args = parser.parse_args()

    print(f"Loading data: {args.dataset}")
    sequences, smiles_list, km_labels, kcat_labels = load_csv(args.dataset)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    if args.task in ("km", "both"):
        _, metrics = train_task(
            sequences, smiles_list, km_labels, "KM",
            out_dir / "km_predictor.joblib",
            out_dir / "km_metrics.json",
            test_size=args.test_size, seed=args.seed,
        )
        results["km"] = metrics

    if args.task in ("kcat", "both"):
        _, metrics = train_task(
            sequences, smiles_list, kcat_labels, "Kcat",
            out_dir / "kcat_predictor.joblib",
            out_dir / "kcat_metrics.json",
            test_size=args.test_size, seed=args.seed,
        )
        results["kcat"] = metrics

    print(f"\n{'='*50}")
    print("Training complete")
    for task, m in results.items():
        print(f"  {task}: R²={m['r2']:.4f}, MAE={m['mae']:.4f}, RMSE={m['rmse']:.4f}")


if __name__ == "__main__":
    main()
