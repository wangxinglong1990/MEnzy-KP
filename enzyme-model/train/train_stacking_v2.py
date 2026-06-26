#!/usr/bin/env python3
"""StackingV2 训练脚本。

自动发现已训练的基模型，构建元特征，训练 meta learner。

用法:
    python train/train_stacking_v2.py --task kcat
    python train/train_stacking_v2.py --task km
    python train/train_stacking_v2.py --task kcat --meta-model lightgbm
"""
import json, sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from configs.config_loader import load_config
from models.stacking_v2.model import StackingV2Model, META_ESTIMATORS
from evaluate.evaluate_stacking_v2 import evaluate_stacking

PROJECT_ROOT = _ROOT


def main():
    import argparse
    parser = argparse.ArgumentParser(description="StackingV2 训练")
    parser.add_argument("--task", type=str, required=True, choices=["kcat", "km"])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--meta-model", type=str, default=None,
                        help=f"Override meta model: {list(META_ESTIMATORS.keys())}")
    args = parser.parse_args()

    # Load config
    cfg_path = args.config or str(PROJECT_ROOT / "configs" / "stacking_v2" / f"{args.task}.yaml")
    cfg = load_config(cfg_path)
    print(f"加载配置: {cfg_path}")

    meta_model = args.meta_model or cfg.get("meta_model", "ridge")
    train_splits = cfg.get("train_splits", ["train", "val"])
    eval_split = cfg.get("eval_split", "test")
    test_size = cfg.get("test_size", 0.2)

    meta_path = PROJECT_ROOT / "artifacts" / "stacking_v2" / args.task / "meta_features.csv"
    if not meta_path.exists():
        print(f"❌ Meta features not found at {meta_path}")
        print(f"   Run: python scripts/build_stacking_dataset_v2.py --task {args.task}")
        sys.exit(1)

    df = pd.read_csv(meta_path)
    print(f"加载 meta features: {len(df)} 行")

    # Identify feature columns (y_*) and target
    feature_cols = [c for c in df.columns if c.startswith("y_") and c != "y_true"]
    target_col = "y_true"
    print(f"特征: {feature_cols}")

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(float)

    # Split by config: use pre-defined split or random
    if "split" in df.columns and len(df["split"].unique()) > 1:
        train_mask = df["split"].isin(train_splits)
        test_mask = df["split"] == eval_split
        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        test_index = np.where(test_mask)[0]
    else:
        all_idx = np.arange(len(df))
        X_train, X_test, y_train, y_test, _, test_index = train_test_split(
            X, y, all_idx, test_size=test_size, random_state=42
        )

    n_train, n_test = len(X_train), len(X_test)
    print(f"训练: {n_train}, 测试: {n_test}")
    print(f"Meta model: {meta_model}")

    if n_train == 0 or n_test == 0:
        print("❌ 训练集或测试集为空，请检查 split 列")
        sys.exit(1)

    # Train
    model = StackingV2Model(meta_model=meta_model, feature_names=feature_cols)
    model.fit(X_train, y_train)
    print("训练完成 ✅")

    # Evaluate
    y_pred = model.predict(X_test)
    metrics = evaluate_stacking(y_test, y_pred)

    print(f"\nStackingV2 ({meta_model}) 测试集性能:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:12s}: {v:.4f}")
        else:
            print(f"  {k:12s}: {v}")

    # Feature importance
    importance = model.get_feature_importance()
    if importance:
        print(f"\n特征重要性:")
        for name, val in importance.items():
            print(f"  {name:20s}: {val:.4f}")

    # Save
    out_dir = PROJECT_ROOT / "artifacts" / "stacking_v2" / args.task
    out_dir.mkdir(parents=True, exist_ok=True)

    model.save(str(out_dir / "model.joblib"))
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Predictions
    pred_df = pd.DataFrame({
        "sample_id": df.iloc[test_index]["sample_id"].values if "sample_id" in df.columns else range(len(y_test)),
        "y_true": y_test,
        "y_pred": y_pred,
        "split": eval_split,
    })
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    print(f"\n模型: {out_dir / 'model.joblib'}")
    print(f"指标: {out_dir / 'metrics.json'}")
    print(f"预测: {out_dir / 'predictions.csv'}")
    print("训练完成")


if __name__ == "__main__":
    main()
