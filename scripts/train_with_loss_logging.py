#!/usr/bin/env python3
"""训练 Condition LightGBM 模型并记录 train/val loss，生成 loss 曲线图 + PDF。

用法:
    python scripts/train_with_loss_logging.py --task kcat
    python scripts/train_with_loss_logging.py --task km
    python scripts/train_with_loss_logging.py --task kcat --debug   # smoke test (100条)
"""

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent  # Kinora-main/
ENZYME_ROOT = _ROOT / "enzyme-model"
if str(ENZYME_ROOT) not in sys.path:
    sys.path.insert(0, str(ENZYME_ROOT))

import lightgbm as lgb  # noqa: E402 — 必须在 torch 前加载, 避免 libomp ABI 冲突

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from configs.config_loader import load_config  # noqa: E402
from evaluate.evaluate_condition import evaluate_regression  # noqa: E402

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
PROJECT_ROOT = _ROOT
PRETRAINED_DIR = ENZYME_ROOT / "data" / "pretrained"
OUTPUT_DIR = PROJECT_ROOT / "figures"
FIG_DPI = 600

# ---------------------------------------------------------------------------
# matplotlib 风格 (与 paper_figures.py 一致)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 13,
    "figure.titlesize": 18,
})


def load_data_and_features(task: str, debug: bool = False):
    """加载 Condition 数据和缓存特征。

    Returns:
        X: np.ndarray
        y: np.ndarray
        sample_ids: np.ndarray
    """
    cfg_path = ENZYME_ROOT / "configs" / "condition" / f"{task}.yaml"
    cfg = load_config(str(cfg_path))
    dset = cfg["dataset"]
    csv_path = ENZYME_ROOT / dset["path"]

    nrows = 100 if debug else None
    df = pd.read_csv(csv_path, index_col=0, nrows=nrows)
    print(f"加载数据: {len(df)} 条 ({task})")

    sequences = df["sequence"].tolist()
    smiles_list = df[dset["smiles_col"]].tolist()
    temperature = df["temperature"].values.astype(float)
    ph = df["ph"].values.astype(float)
    labels = df[dset["target_col"]].values.astype(float)
    sample_ids = df.index.astype(str).tolist()

    cache_dir = ENZYME_ROOT / "data" / "features" / "condition" / task
    smiles_pkl = cache_dir / "smiles_embeddings.npy"
    protein_pkl = cache_dir / "protein_embeddings.npy"

    if smiles_pkl.exists() and protein_pkl.exists():
        print("从缓存加载特征...")
        smiles_emb = np.load(smiles_pkl)[:nrows]
        protein_emb = np.load(protein_pkl)[:nrows]
    else:
        raise FileNotFoundError(f"特征缓存缺失: {cache_dir}")

    print(f"  SMILES: {smiles_emb.shape}, 蛋白质: {protein_emb.shape}")

    # Condition features (temp, pH)
    temp_fill = cfg["imputation"].get("temp_fill") or 0
    ph_fill = cfg["imputation"].get("ph_fill") or 7
    temperature = np.where(np.isnan(temperature), float(temp_fill), temperature)
    ph = np.where(np.isnan(ph), float(ph_fill), ph)
    cond_feat = np.column_stack([temperature, ph])

    X = np.concatenate([smiles_emb, protein_emb, cond_feat], axis=1)
    print(f"  总特征: {X.shape}")

    return X, labels, np.array(sample_ids)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["kcat", "km"])
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    task = args.task
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. 加载数据
    # ------------------------------------------------------------------
    X, y, sample_ids = load_data_and_features(task, debug=args.debug)

    # ------------------------------------------------------------------
    # 2. Train / Val / Test split
    # ------------------------------------------------------------------
    X_temp, X_test, y_temp, y_test, sids_temp, sids_test = train_test_split(
        X, y, sample_ids, test_size=0.15, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15 / 0.85, random_state=42
    )
    print(f"划分: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # ------------------------------------------------------------------
    # 3. 训练 LightGBM — 使用 eval_set 记录 loss
    # ------------------------------------------------------------------
    print("\n开始训练 LightGBM (含 eval_set 记录)...")
    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=10,                    # 每 10 轮打印一次
        early_stopping_rounds=50,      # 早停
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_names=["train", "val"],
        eval_metric="rmse",
    )

    # ------------------------------------------------------------------
    # 4. 保存 loss 历史到 CSV
    # ------------------------------------------------------------------
    evals = model.evals_result_
    loss_df = pd.DataFrame({
        "iteration": range(1, len(evals["train"]["rmse"]) + 1),
        "train_rmse": evals["train"]["rmse"],
        "val_rmse": evals["val"]["rmse"],
    })
    loss_csv = OUTPUT_DIR / f"loss_history_{task}.csv"
    loss_df.to_csv(loss_csv, index=False)
    print(f"\n✅ Loss 历史已保存: {loss_csv}")
    print(f"  最佳迭代: {model.best_iteration_}, "
          f"train_rmse={loss_df['train_rmse'].iloc[model.best_iteration_ - 1]:.4f}, "
          f"val_rmse={loss_df['val_rmse'].iloc[model.best_iteration_ - 1]:.4f}")

    # ------------------------------------------------------------------
    # 5. 测试集评估
    # ------------------------------------------------------------------
    y_pred = model.predict(X_test)
    metrics = evaluate_regression(y_test, y_pred)
    print(f"\n测试集性能:")
    print(f"  R²:      {metrics['r2']:.4f}")
    print(f"  MAE:     {metrics['mae']:.4f}")
    print(f"  RMSE:    {metrics['rmse']:.4f}")
    print(f"  Pearson: {metrics['pearson_r']:.4f} (p={metrics['pearson_p']:.2e})")

    # ------------------------------------------------------------------
    # 6. 画 Loss 曲线图
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- 左图: 全量迭代 ---
    ax1.plot(loss_df["iteration"], loss_df["train_rmse"], color="#4C72B0",
             lw=1.3, label="Train RMSE", alpha=0.85)
    ax1.plot(loss_df["iteration"], loss_df["val_rmse"], color="#DD8452",
             lw=1.3, label="Val RMSE", alpha=0.85)
    if model.best_iteration_:
        ax1.axvline(x=model.best_iteration_, color="red", ls="--", lw=1.0,
                    alpha=0.7, label=f"Best iter = {model.best_iteration_}")
    ax1.set_xlabel("Boosting Iteration")
    ax1.set_ylabel("RMSE")
    ax1.set_title(f"Training & Validation Loss — {task.upper()}")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # --- 右图: 局部放大 (前 100 轮 + best 附近) ---
    best = model.best_iteration_ or 100
    zoom_start = max(1, best - 5)
    zoom_end = min(len(loss_df), best + 15)
    zoom = loss_df.iloc[zoom_start - 1:zoom_end]
    ax2.plot(zoom["iteration"], zoom["train_rmse"], "o-", color="#4C72B0",
             ms=4, lw=1.2, label="Train RMSE")
    ax2.plot(zoom["iteration"], zoom["val_rmse"], "s-", color="#DD8452",
             ms=4, lw=1.2, label="Val RMSE")
    if model.best_iteration_:
        ax2.axvline(x=best, color="red", ls="--", lw=1.0, alpha=0.7)
        best_val = loss_df["val_rmse"].iloc[best - 1]
        ax2.annotate(f"Best: {best_val:.4f}",
                     xy=(best, best_val),
                     xytext=(best + 3, best_val + 0.02),
                     fontsize=9, color="red",
                     arrowprops=dict(arrowstyle="->", color="red", lw=0.8))
    ax2.set_xlabel("Boosting Iteration")
    ax2.set_ylabel("RMSE")
    ax2.set_title(f"Zoom around Best Iteration — {task.upper()}")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle(f"LightGBM Training Loss — Condition Model ({task.upper()})  |  "
                 f"Test R²={metrics['r2']:.3f}  PCC={metrics['pearson_r']:.3f}",
                 fontsize=15, y=1.03)
    fig.tight_layout()

    png_path = OUTPUT_DIR / f"loss_curve_{task}.png"
    pdf_path = OUTPUT_DIR / f"loss_curve_{task}.pdf"
    fig.savefig(png_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✅ PNG: {png_path}")
    print(f"✅ PDF: {pdf_path}")


if __name__ == "__main__":
    main()
