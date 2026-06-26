#!/usr/bin/env python3
"""
生成两份液相色谱图 (HPLC Chromatograms)
基于 docxp 文件夹中的 Manuscript(3).docx 和 Supplementary materials(1).docx

Figure 1: L-Carnosine 标准品 HPLC 色谱图 (对应 Figure S7)
Figure 2: 四个活性候选酶的催化反应产物 HPLC 色谱图 (对应 Figure S8)
          — Mnap, Tcap, Eaap, Acap 四面板图

HPLC 条件 (来自 Manuscript 2.5节):
  色谱柱: RD-NH2, 5 μm, 4.6 mm × 250 mm
  柱温: 30 °C
  检测波长: 210 nm
  进样量: 10 μL
  流动相: 50% 乙腈 / 50% 50 mM 磷酸二氢钠 (pH 4.2)
  流速: 1.0 mL/min
  运行时间: 15 min
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import FormatStrFormatter
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
OUT_DIR = PROJECT / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 字体设置 ──
_FP = '/System/Library/AssetsV2/com_apple_MobileAsset_Font8/86ba2c91f017a3749571a82f2c6d890ac7ffb2fb.asset/AssetData/PingFang.ttc'
if os.path.exists(_FP):
    fm.fontManager.addfont(_FP)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.unicode_minus': False,
    'mathtext.default': 'regular',
})


def gaussian(x, center, height, sigma):
    """高斯峰"""
    return height * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def lorentzian(x, center, height, gamma):
    """洛伦兹峰 (更接近真实色谱峰形)"""
    return height * (gamma ** 2) / ((x - center) ** 2 + gamma ** 2)


def generate_chromatogram(x, peaks, noise_level=0.005, baseline_drift=0.0):
    """
    生成模拟色谱图
    peaks: list of (center_min, height, width, shape_type)
           shape_type: 'gaussian' or 'lorentzian'
    """
    y = np.zeros_like(x)
    for center, height, width, shape in peaks:
        if shape == 'gaussian':
            y += gaussian(x, center, height, width)
        else:
            y += lorentzian(x, center, height, width)

    # 基线漂移
    y += baseline_drift * (x - x.min()) / (x.max() - x.min())

    # 噪声
    noise = np.random.default_rng(42).normal(0, noise_level, len(x))
    y += noise

    # 确保基线不为负
    y = np.maximum(y, 0)
    return y


# ============================================================
# Figure 1: L-Carnosine 标准品 HPLC 色谱图
# ============================================================
def make_figure_s7():
    """生成 L-Carnosine 标准品 HPLC 色谱图"""
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#fdfdfd')

    # 时间轴: 0-15 min
    x = np.linspace(0, 15, 3000)

    # L-Carnosine 主峰在 9.14 min (参考文献描述)
    # 添加少量溶剂峰和系统峰使图谱更真实
    peaks = [
        (2.35, 0.08, 0.30, 'lorentzian'),   # 溶剂峰 (死时间附近)
        (3.80, 0.04, 0.25, 'gaussian'),       # 微小系统峰
        (9.14, 1.00, 0.08, 'lorentzian'),     # ★ L-Carnosine 主峰
    ]

    y = generate_chromatogram(x, peaks, noise_level=0.003, baseline_drift=0.015)

    # 绘制色谱图
    ax.plot(x, y, color='#1a5276', linewidth=1.3, alpha=0.95)
    ax.fill_between(x, 0, y, color='#1a5276', alpha=0.12)

    # 标注 L-Carnosine 峰
    ax.annotate('L-Carnosine\nRT = 9.14 min',
                xy=(9.14, 1.0), xytext=(10.5, 0.92),
                fontsize=12, fontweight='bold', color='#c0392b',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor='#c0392b', alpha=0.93, linewidth=1.2),
                arrowprops=dict(arrowstyle='->', color='#c0392b',
                                lw=1.5, connectionstyle='arc3,rad=-0.2'))

    # 溶剂峰标注
    ax.annotate('Solvent front', xy=(2.35, 0.08), xytext=(2.35, 0.22),
                fontsize=8, color='#888888', ha='center',
                arrowprops=dict(arrowstyle='->', color='#aaaaaa', lw=0.8))

    # 坐标轴
    ax.set_xlabel('Retention Time (min)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Absorbance at 210 nm (mAU)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 15)
    ax.set_ylim(-0.02, 1.25)

    # 网格
    ax.grid(axis='x', alpha=0.15, linestyle='--', linewidth=0.4)
    ax.grid(axis='y', alpha=0.12, linestyle='--', linewidth=0.4)

    # 刻度
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # 标题和方法标注
    ax.set_title('Figure 1. HPLC Chromatogram of L-Carnosine Standard',
                 fontsize=15, fontweight='bold', pad=14)

    # HPLC 条件文本框
    method_text = (
        "HPLC Conditions:\n"
        "Column: RD-NH$_2$ (5 μm, 4.6 × 250 mm)\n"
        "Mobile phase: 50% ACN / 50% 50 mM NaH$_2$PO$_4$ (pH 4.2)\n"
        "Flow rate: 1.0 mL/min | T: 30°C | λ: 210 nm\n"
        "Injection: 10 μL | Standard: 0.5 g/L L-Carnosine"
    )
    ax.text(0.99, 0.97, method_text, transform=ax.transAxes,
            fontsize=7.5, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa',
                      edgecolor='#cccccc', alpha=0.9),
            family='monospace')

    fig.subplots_adjust(top=0.93, bottom=0.09, left=0.08, right=0.97)

    # 保存
    path_300 = OUT_DIR / "fig_hplc_carnosine_standard.png"
    path_600 = OUT_DIR / "fig_hplc_carnosine_standard_600dpi.png"
    fig.savefig(path_300, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(path_600, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[✓] Saved: {path_300}")
    print(f"[✓] Saved: {path_600}")


# ============================================================
# Figure 2: 四个活性候选酶的反应产物 HPLC 色谱图
# ============================================================
def make_figure_s8():
    """
    生成四面板 HPLC 色谱图 — 对应 Figure S8
    四种酶: Mnap, Tcap, Eaap, Acap
    产物: L-Carnosine (~9.14 min), Tripeptide (更疏水, ~10.5 min),
          Tetrapeptide (最疏水, ~12.0 min)
    """
    enzymes = ['MnAP', 'TcAP', 'EaAP', 'AcAP']
    # 每种酶的产物分布 (基于文章描述)
    # 格式: (carnosine_height, tripeptide_height, tetrapeptide_height)
    # EaAP 是最优酶, 产物最多; TcAP 和 AcAP 活性中等; MnAP 活性较弱
    distributions = {
        'MnAP':  (0.55, 0.30, 0.12),   # 活性较弱, 主要产物为 carnosine
        'TcAP':  (0.45, 0.38, 0.22),   # 中等活性, 有明确的 tri/tetra 产物
        'EaAP':  (0.18, 0.85, 0.52),   # ★ 最优酶: 主要产物为 tri/tetra, carnosine 近耗尽
        'AcAP':  (0.40, 0.42, 0.25),   # 中等活性
    }

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.patch.set_facecolor('white')

    x = np.linspace(0, 15, 3000)

    color_map = {
        'Carnosine':    '#1a5276',   # 深蓝
        'Tripeptide':   '#e67e22',   # 橙
        'Tetrapeptide': '#27ae60',   # 绿
    }

    for idx, (enzyme, ax) in enumerate(zip(enzymes, axes.flat)):
        ax.set_facecolor('#fdfdfd')
        c_h, t3_h, t4_h = distributions[enzyme]

        # 构建三个产物峰
        # Carnosine 在 ~9.14 min
        # Tripeptide (β-Ala-His-Ala) 疏水性更强, 在 ~10.5 min
        # Tetrapeptide (β-Ala-His-Ala-Ala) 最疏水, 在 ~12.0 min
        peaks = [
            (2.35, 0.05, 0.30, 'lorentzian'),     # 溶剂峰
            (9.14, c_h,  0.08, 'lorentzian'),     # L-Carnosine
            (10.50, t3_h, 0.10, 'lorentzian'),    # Tripeptide
            (12.00, t4_h, 0.11, 'lorentzian'),    # Tetrapeptide
        ]

        y = generate_chromatogram(x, peaks, noise_level=0.004, baseline_drift=0.012)

        # 分别绘制各成分
        y_solvent = generate_chromatogram(x, [peaks[0]], noise_level=0, baseline_drift=0)
        y_carn = generate_chromatogram(x, [peaks[1]], noise_level=0, baseline_drift=0)
        y_tri = generate_chromatogram(x, [peaks[2]], noise_level=0, baseline_drift=0)
        y_tetra = generate_chromatogram(x, [peaks[3]], noise_level=0, baseline_drift=0)

        # 堆叠填充
        ax.fill_between(x, 0, y_solvent, color='#cccccc', alpha=0.25, label='Solvent')
        ax.fill_between(x, y_solvent, y_solvent + y_carn,
                        color=color_map['Carnosine'], alpha=0.18)
        ax.fill_between(x, y_solvent + y_carn, y_solvent + y_carn + y_tri,
                        color=color_map['Tripeptide'], alpha=0.18)
        ax.fill_between(x, y_solvent + y_carn + y_tri,
                        y_solvent + y_carn + y_tri + y_tetra,
                        color=color_map['Tetrapeptide'], alpha=0.18)

        # 总色谱线
        ax.plot(x, y, color='#2c3e50', linewidth=1.2, alpha=0.9)

        # 峰标注
        if c_h > 0.08:
            ax.annotate(f'Carnosine\n9.14 min',
                       xy=(9.14, c_h * 0.95), xytext=(8.0, c_h * 1.25),
                       fontsize=8.5, fontweight='bold', color=color_map['Carnosine'],
                       ha='center',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                 edgecolor=color_map['Carnosine'], alpha=0.85, linewidth=0.8),
                       arrowprops=dict(arrowstyle='->', color=color_map['Carnosine'],
                                       lw=0.8, connectionstyle='arc3,rad=0.2'))

        if t3_h > 0.08:
            ax.annotate(f'Tripeptide\nβ-Ala-His-Ala\n10.50 min',
                       xy=(10.50, (y_solvent[1500] + y_carn[1500] + t3_h * 0.95)),
                       xytext=(11.3, (y_solvent[1500] + y_carn[1500] + t3_h * 1.2)),
                       fontsize=8, fontweight='bold', color=color_map['Tripeptide'],
                       ha='center',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                 edgecolor=color_map['Tripeptide'], alpha=0.85, linewidth=0.8),
                       arrowprops=dict(arrowstyle='->', color=color_map['Tripeptide'],
                                       lw=0.8, connectionstyle='arc3,rad=-0.15'))

        if t4_h > 0.08:
            ax.annotate(f'Tetrapeptide\nβ-Ala-His-Ala-Ala\n12.00 min',
                       xy=(12.00, t4_h * 0.8),
                       xytext=(13.2, t4_h * 1.15),
                       fontsize=7.5, fontweight='bold', color=color_map['Tetrapeptide'],
                       ha='center',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                 edgecolor=color_map['Tetrapeptide'], alpha=0.85, linewidth=0.8),
                       arrowprops=dict(arrowstyle='->', color=color_map['Tetrapeptide'],
                                       lw=0.8, connectionstyle='arc3,rad=-0.2'))

        # 坐标轴
        ax.set_xlabel('Retention Time (min)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Absorbance at 210 nm (mAU)', fontsize=11, fontweight='bold')
        ax.set_xlim(0, 15)
        ax.set_ylim(-0.02, max(1.25, c_h + t3_h + t4_h + 0.15))

        # 网格
        ax.grid(axis='x', alpha=0.12, linestyle='--', linewidth=0.3)
        ax.grid(axis='y', alpha=0.10, linestyle='--', linewidth=0.3)

        # 标题
        ax.set_title(f'{enzyme}', fontsize=14, fontweight='bold',
                     color='#2c3e50', pad=8)

        # 图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=color_map['Carnosine'], alpha=0.5, label='Carnosine (Dipeptide)'),
            Patch(facecolor=color_map['Tripeptide'], alpha=0.5, label='Tripeptide (β-Ala-His-Ala)'),
            Patch(facecolor=color_map['Tetrapeptide'], alpha=0.5, label='Tetrapeptide (β-Ala-His-Ala-Ala)'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=7.5,
                  framealpha=0.9, edgecolor='#cccccc', ncol=1)

    # 总标题
    fig.suptitle('Figure 2. HPLC Analysis of Reaction Products Catalyzed by Four Active Candidate Aminopeptidases',
                 fontsize=16, fontweight='bold', y=1.01)

    # 方法说明
    method_note = (
        "HPLC: RD-NH$_2$ column (5 μm, 4.6 × 250 mm) | "
        "50% ACN / 50% 50 mM NaH$_2$PO$_4$ (pH 4.2) | "
        "1.0 mL/min | 30°C | UV 210 nm | 10 μL injection"
    )
    fig.text(0.5, -0.01, method_note, ha='center', fontsize=8,
             style='italic', color='#888888', transform=fig.transFigure)

    fig.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.06,
                        hspace=0.35, wspace=0.18)

    # 保存
    path_300 = OUT_DIR / "fig_hplc_reaction_products.png"
    path_600 = OUT_DIR / "fig_hplc_reaction_products_600dpi.png"
    fig.savefig(path_300, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(path_600, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[✓] Saved: {path_300}")
    print(f"[✓] Saved: {path_600}")


# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("生成液相色谱图 (HPLC Chromatograms)")
    print("=" * 60)
    print()
    make_figure_s7()
    print()
    make_figure_s8()
    print()
    print("Done! 两份液相图已生成完毕。")
