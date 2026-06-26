#!/usr/bin/env python3
"""Stacking meta-learner training script.

Usage:
    python train/train_stacking.py --task kcat
    python train/train_stacking.py --task km

Prerequisites:
    python scripts/build_stacking_dataset.py --task kcat
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models.stacking.model import StackingModel
from evaluate.evaluate_condition import evaluate_regression


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stacking meta-learner 训练")
    parser.add_argument("--task", type=str, required=True, choices=["kcat", "km"],
                        help="训练任务: kcat 或 km")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge alpha")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    args = parser.parse_args()

    project_root = _ROOT
    meta_path = project_root / "artifacts" / "stacking" / args.task / "meta_features.csv"
    out_dir = project_root / "artifacts" / "stacking" / args.task
    out_dir.mkdir(parents=True, exist_ok=True)

    if not meta_path.exists():
        print(f"❌ Meta features not found: {meta_path}")
        print(f"   Run: python scripts/build_stacking_dataset.py --task {args.task}")
        sys.exit(1)

    df = pd.read_csv(meta_path)
    print(f"加载 meta features: {len(df)} 行")

    feature_cols = ["y_baseline", "y_condition", "y_msa1d"]
    X = df[feature_cols].values.astype(np.float32)
    y = df["y_true"].values.astype(float)

    indices = np.arange(len(df))
    X_train, X_test, y_train, y_test, _, idx_test = train_test_split(
        X, y, indices, test_size=args.test_size, random_state=42
    )
    print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")
    print(f"特征: {feature_cols}")

    from sklearn.linear_model import Ridge
    model = StackingModel()
    model._model = Ridge(alpha=args.alpha, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = evaluate_regression(y_test, y_pred)

    print(f"\nStackingV1 测试集性能:")
    print(f"  R²:      {metrics['r2']:.4f}")
    print(f"  MAE:     {metrics['mae']:.4f}")
    print(f"  RMSE:    {metrics['rmse']:.4f}")
    print(f"  Pearson: {metrics['pearson_r']:.4f} (p={metrics['pearson_p']:.2e})")

    # Save
    model.save(str(out_dir / "model.joblib"))
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Predictions
    pred_df = pd.DataFrame({
        "sample_id": df.iloc[idx_test]["sample_id"].values,
        "y_true": y_test, "y_pred": y_pred,
    })
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    print(f"\n模型: {out_dir / 'model.joblib'}")
    print(f"指标: {out_dir / 'metrics.json'}")
    print(f"预测: {out_dir / 'predictions.csv'}")
    print("训练完成")


if __name__ == "__main__":
    main()
