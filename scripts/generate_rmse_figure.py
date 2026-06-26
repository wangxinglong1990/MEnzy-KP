#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成 RMSE 对比图 —— 所有模型 × 两个靶标 (Km / kcat) 的 RMSE 柱状图。

自动从 enzyme-model/artifacts/ 目录发现模型 metrics，生成发表级对比图。

用法:
    python scripts/generate_rmse_figure.py
    python scripts/generate_rmse_figure.py --output-dir ./figures
"""

import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Kinora-main/
ARTIFACTS_DIR = PROJECT_ROOT / "enzyme-model" / "artifacts"
OUTPUT_DIR = PROJECT_ROOT / "figures"
FIG_DPI = 600

MODEL_LABELS = {
    "baseline": "Baseline\n(ExtraTrees)",
    "condition": "Condition\n(LightGBM)",
    "msa1d": "MSA1D\n(XGBoost)",
    "msa2d": "MSA2D\n(LightGBM)",
    "stacking_v2": "StackingV2\n(Ridge)",
}
TARGET_LABELS = {"km": "Km", "kcat": "kcat"}

# Color palette
COLOR_KM = "#4C72B0"    # blue
COLOR_KCAT = "#DD8452"  # orange


def discover_metrics() -> pd.DataFrame:
    """从 artifacts 目录自动发现所有模型的评估指标。

    Returns:
        DataFrame with columns: model, target, r2, mae, rmse
    """
    records = []

    for model_dir in sorted(ARTIFACTS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        for target_dir in sorted(model_dir.iterdir()):
            if not target_dir.is_dir():
                continue
            target = target_dir.name
            if target not in ("km", "kcat"):
                continue

            # 尝试多种 metrics 文件名
            metrics_candidates = [
                "metrics.json",
                f"{model_name}_model_metrics.json",
                f"{target}_model_metrics.json",
                f"{model_name}_{target}_metrics.json",
                "condition_model_metrics.json",
            ]

            metrics = None
            for cand in metrics_candidates:
                path = target_dir / cand
                if path.exists():
                    with open(path) as f:
                        metrics = json.load(f)
                    break

            if metrics is None:
                print(f"⚠  No metrics found for {model_name}/{target}")
                continue

            # 标准化指标名
            r2 = metrics.get("r2") or metrics.get("test_r2") or metrics.get("best_r2")
            if r2 is not None:
                r2 = float(r2)
            mae = metrics.get("mae") or metrics.get("test_mae")
            if mae is not None:
                mae = float(mae)

            # RMSE: 直接取或从 MSE 推算
            rmse = metrics.get("rmse")
            if rmse is None:
                mse = metrics.get("mse") or metrics.get("test_mse")
                if mse is not None:
                    rmse = float(np.sqrt(float(mse)))
            if rmse is not None:
                rmse = float(rmse)

            records.append({
                "model": model_name,
                "target": target,
                "r2": r2,
                "mae": mae,
                "rmse": rmse,
            })

    df = pd.DataFrame(records)
    return df


def plot_rmse_comparison(df: pd.DataFrame, output_dir: Path):
    """生成 RMSE 柱状对比图。

    Args:
        df: 指标 DataFrame (含 model, target, rmse 列)
        output_dir: 输出目录
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 只保留有 RMSE 的模型
    plot_df = df[df["rmse"].notna()].copy()
    if plot_df.empty:
        print("❌ 没有可用的 RMSE 数据")
        return

    # 只保留有 label 的模型
    plot_df = plot_df[plot_df["model"].isin(MODEL_LABELS.keys())].copy()
    model_order = [m for m in MODEL_LABELS if m in plot_df["model"].unique()]

    # ------------------------------------------------------------------
    # matplotlib 风格 (与 paper_figures.py 一致)
    # ------------------------------------------------------------------
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 18,
    })

    # ------------------------------------------------------------------
    # 图1: 分组柱状图 —— KM vs Kcat
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(model_order))
    width = 0.35

    targets = ["km", "kcat"]
    colors = [COLOR_KM, COLOR_KCAT]
    all_bars = []

    for i, (target, color) in enumerate(zip(targets, colors)):
        subset = plot_df[plot_df["target"] == target].set_index("model")
        values = []
        for m in model_order:
            if m in subset.index:
                val = subset.loc[m, "rmse"]
                values.append(val if pd.notna(val) else 0)
            else:
                values.append(0)

        bars = ax.bar(
            x + i * width, values, width,
            label=TARGET_LABELS[target],
            color=color,
            edgecolor="white",
            linewidth=0.8,
            alpha=0.9,
        )
        all_bars.append((bars, values, target))

        # 在柱子上方标注数值
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.015,
                    f"{val:.3f}",
                    ha="center", va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

    # X 轴标签
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([MODEL_LABELS[m] for m in model_order], rotation=0, ha="center")

    ax.set_ylabel("RMSE (log10 scale)")
    ax.set_title("Model Comparison — RMSE on Test Set")
    ax.legend(loc="upper left", frameon=True, fancybox=True, shadow=True)

    # Y 轴从 0 开始
    ax.set_ylim(0, max(plot_df["rmse"].max() * 1.15, 0.5))
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # 添加水平参考线标注
    best_km = plot_df[plot_df["target"] == "km"]["rmse"].min()
    best_kcat = plot_df[plot_df["target"] == "kcat"]["rmse"].min()
    ax.axhline(y=best_kcat, color=COLOR_KCAT, ls=":", lw=1.0, alpha=0.5)
    ax.axhline(y=best_km, color=COLOR_KM, ls=":", lw=1.0, alpha=0.5)

    fig.tight_layout()
    png_path = output_dir / "rmse_comparison.png"
    pdf_path = output_dir / "rmse_comparison.pdf"
    fig.savefig(png_path, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✅ PNG: {png_path}")
    print(f"✅ PDF: {pdf_path}")

    # ------------------------------------------------------------------
    # 图2: 单列分组 —— KM 和 Kcat 各一张图 (更清晰的展示)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, target, color in zip(axes, targets, colors):
        subset = plot_df[plot_df["target"] == target].set_index("model")
        values = []
        models_present = []
        for m in model_order:
            if m in subset.index:
                val = subset.loc[m, "rmse"]
                if pd.notna(val):
                    values.append(val)
                    models_present.append(m)
                else:
                    values.append(0)
                    models_present.append(m)
            else:
                values.append(0)
                models_present.append(m)

        x_pos = np.arange(len(models_present))
        bars = ax.bar(
            x_pos, values,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            alpha=0.9,
        )

        # 渐变颜色：最佳模型深色，其他浅色
        if values:
            best_idx = np.argmin([v if v > 0 else float("inf") for v in values])
            for j, bar in enumerate(bars):
                if j == best_idx:
                    bar.set_alpha(1.0)
                    bar.set_edgecolor("black")
                    bar.set_linewidth(1.5)
                else:
                    bar.set_alpha(0.65)

        # 数值标注
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}",
                    ha="center", va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [MODEL_LABELS[m] for m in models_present],
            rotation=15,
            ha="right",
        )
        ax.set_ylabel("RMSE (log10 scale)")
        ax.set_title(f"RMSE — {TARGET_LABELS[target]}")
        ax.set_ylim(0, max(values) * 1.2 if values else 1)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    fig.suptitle("Model Comparison — RMSE on Test Set", fontsize=16, y=1.02)
    fig.tight_layout()

    png_path2 = output_dir / "rmse_comparison_split.png"
    pdf_path2 = output_dir / "rmse_comparison_split.pdf"
    fig.savefig(png_path2, dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path2, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✅ PNG: {png_path2}")
    print(f"✅ PDF: {pdf_path2}")

    # ------------------------------------------------------------------
    # 打印汇总表
    # ------------------------------------------------------------------
    print("\n📊 RMSE 汇总 (Test Set):")
    pivot = plot_df.pivot_table(
        index="model", columns="target", values="rmse", aggfunc="first"
    )
    # 按 model_order 排序
    pivot = pivot.reindex([m for m in model_order if m in pivot.index])
    print(pivot.to_string(float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"))


def main():
    parser = argparse.ArgumentParser(
        description="Generate RMSE comparison figure across all models"
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(OUTPUT_DIR),
        help="Output directory for figures"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("Discovering model metrics...")
    print("=" * 60)

    df = discover_metrics()

    if df.empty:
        print("❌ 没有找到任何指标数据")
        return

    print(f"\nFound {len(df)} model/target combinations:")
    for _, row in df.iterrows():
        rmse_str = f"{row['rmse']:.4f}" if pd.notna(row['rmse']) else "N/A"
        print(f"  {row['model']:15s} / {row['target']:5s}  →  RMSE = {rmse_str}")

    print("\n" + "=" * 60)
    print("Generating RMSE comparison plots...")
    print("=" * 60)

    plot_rmse_comparison(df, output_dir)

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
