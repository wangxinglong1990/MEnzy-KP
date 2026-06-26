#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成 PCC 图：预测值 vs 真实值散点图 + 回归线 + PCC 标注。

覆盖所有模型（Baseline / Condition / MSA1D / MSA2D / StackingV2）× 两个靶标（Km / kcat）。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Kinora-main/
ARTIFACTS_DIR = PROJECT_ROOT / "enzyme-model" / "artifacts"
OUTPUT_DIR = PROJECT_ROOT / "figures"
FIG_DPI = 600

MODELS = ["baseline", "condition", "msa1d", "msa2d", "stacking_v2"]
TARGETS = ["km", "kcat"]
MODEL_LABELS = {
    "baseline": "Baseline (ET)",
    "condition": "Condition (LGB)",
    "msa1d": "MSA1D (XGB)",
    "msa2d": "MSA2D (LGB)",
    "stacking_v2": "StackingV2 (Ridge)",
}
TARGET_LABELS = {"km": "Km", "kcat": "kcat"}
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


def load_predictions(model: str, target: str) -> pd.DataFrame | None:
    """加载某个模型+靶标的预测 CSV，统一列名。"""
    csv_path = ARTIFACTS_DIR / model / target / "predictions.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    # 不同模型的列顺序不同，统一处理
    if "y_true" not in df.columns or "y_pred" not in df.columns:
        return None
    return df[["sample_id", "split", "y_true", "y_pred"]].copy()


def compute_pcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """安全计算 Pearson r。"""
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    r, _ = pearsonr(y_true, y_pred)
    return float(r)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 全局 matplotlib 风格（与 paper_figures.py 一致）
    # ------------------------------------------------------------------
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.titlesize": 18,
        }
    )

    # ------------------------------------------------------------------
    # 图1: 单张大图 —— 所有模型 × 靶标的 PCC 散点图网格 (5行 × 2列)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(
        nrows=len(MODELS),
        ncols=len(TARGETS),
        figsize=(12, 26),
        sharex=False,
        sharey=False,
    )

    for row_idx, model in enumerate(MODELS):
        for col_idx, target in enumerate(TARGETS):
            ax = axes[row_idx, col_idx]
            df = load_predictions(model, target)

            if df is None or len(df) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=14, color="gray")
                ax.set_title(f"{MODEL_LABELS[model]} — {TARGET_LABELS[target]}")
                continue

            # 只画 test 集
            test_df = df[df["split"] == "test"].copy()
            if len(test_df) == 0:
                test_df = df.copy()  # 如果没有 split 列，画全部

            y_true = test_df["y_true"].values.astype(np.float64)
            y_pred = test_df["y_pred"].values.astype(np.float64)

            pcc = compute_pcc(y_true, y_pred)
            n = len(y_true)

            # 散点
            ax.scatter(y_true, y_pred, alpha=0.35, s=12, color=COLORS[row_idx],
                       edgecolors="none", rasterized=True)

            # OLS 回归线
            reg = LinearRegression()
            reg.fit(y_true.reshape(-1, 1), y_pred)
            x_line = np.linspace(y_true.min(), y_true.max(), 200)
            y_line = reg.predict(x_line.reshape(-1, 1))
            ax.plot(x_line, y_line, color="red", lw=1.5, linestyle="-")

            # y = x 参考线
            lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
            ax.plot(lims, lims, "--", color="gray", lw=0.8, alpha=0.7)

            # PCC 标注
            ax.text(
                0.05, 0.95,
                f"PCC = {pcc:.3f}\nn = {n}",
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="lightgray", alpha=0.85),
            )

            ax.set_xlabel(f"Experimental {TARGET_LABELS[target]}")
            ax.set_ylabel(f"Predicted {TARGET_LABELS[target]}")
            ax.set_title(f"{MODEL_LABELS[model]} — {TARGET_LABELS[target]}")

    fig.suptitle("Predicted vs Experimental — Pearson Correlation (PCC)", fontsize=20, y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pcc_grid_all_models.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: {OUTPUT_DIR / 'pcc_grid_all_models.png'}")

    # ------------------------------------------------------------------
    # 图2: PCC 柱状图 —— 对比所有模型在 Km / kcat 上的 PCC
    # ------------------------------------------------------------------
    pcc_data: list[dict] = []
    for model in MODELS:
        for target in TARGETS:
            df = load_predictions(model, target)
            if df is None:
                continue
            test_df = df[df["split"] == "test"] if "split" in df.columns else df
            pcc = compute_pcc(test_df["y_true"].values.astype(np.float64),
                              test_df["y_pred"].values.astype(np.float64))
            pcc_data.append({
                "model": MODEL_LABELS[model],
                "target": TARGET_LABELS[target],
                "pcc": pcc,
            })

    pcc_df = pd.DataFrame(pcc_data)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(MODELS))
    width = 0.35

    for i, target in enumerate(TARGET_LABELS.values()):
        subset = pcc_df[pcc_df["target"] == target].set_index("model")
        values = [subset.loc[MODEL_LABELS[m], "pcc"] if MODEL_LABELS[m] in subset.index else 0
                  for m in MODELS]
        bars = ax.bar(x + i * width, values, width, label=target,
                      color=["#4C72B0", "#DD8452"][i], edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], rotation=15, ha="right")
    ax.set_ylabel("Pearson Correlation Coefficient (PCC)")
    ax.set_title("Model Comparison — PCC on Test Set")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pcc_bar_comparison.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: {OUTPUT_DIR / 'pcc_bar_comparison.png'}")

    # 打印汇总表
    print("\n📊 PCC 汇总 (Test Set):")
    print(pcc_df.pivot(index="model", columns="target", values="pcc").to_string())


if __name__ == "__main__":
    main()
