#!/usr/bin/env python3
"""
Sequence Logo 生成脚本 v2（Figure B）— 论文级出图
修复"太保守"问题：展示全长比对而非仅最保守窗口
基于 5 条同源 M1 氨肽酶序列的完整 MAFFT 比对

用法: python3 generate_sequence_logo.py
"""

import os
import subprocess
import tempfile
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import logomaker

_FONT_PATH = '/System/Library/AssetsV2/com_apple_MobileAsset_Font8/86ba2c91f017a3749571a82f2c6d890ac7ffb2fb.asset/AssetData/PingFang.ttc'
if os.path.exists(_FONT_PATH):
    fm.fontManager.addfont(_FONT_PATH)

PROJECT = Path(__file__).resolve().parent
OUT_DIR = PROJECT / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════
# Step 1: 读取序列
# ═══════════════════════════════════════════════════════════
print("Step 1: 读取 sixdata (排除非同源 MsAP)")
txt = (PROJECT / "textdocs" / "sixdata.text").read_text()
all_sequences = {}
for ln in txt.strip().split('\n'):
    if not ln.strip():
        continue
    sep = '：' if '：' in ln else ':'
    parts = ln.split(sep, 1)
    if len(parts) == 2:
        name = parts[0].strip()
        seq = parts[1].strip().upper()
        clean = "".join(c for c in seq if c in "ACDEFGHIKLMNPQRSTVWY")
        all_sequences[name] = clean

# 仅使用 5 条同源 M1 氨肽酶
HOMOLOGOUS_KEYS = [k for k in all_sequences if 'MsAP' not in k and '6.' not in k]
sequences = {k: all_sequences[k] for k in HOMOLOGOUS_KEYS}
for name, seq in sequences.items():
    print(f"  {name}: {len(seq)} aa")

# ═══════════════════════════════════════════════════════════
# Step 2: MAFFT + 全比对保守性分析
# ═══════════════════════════════════════════════════════════
print("\nStep 2: MAFFT 多序列比对")
fasta_in = tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False, dir='/tmp')
for name, seq in sequences.items():
    fasta_in.write(f">{name}\n{seq}\n")
fasta_in.close()

result = subprocess.run(
    ["mafft", "--auto", "--reorder", fasta_in.name],
    capture_output=True, text=True
)
if result.returncode != 0:
    raise RuntimeError(f"MAFFT failed: {result.stderr}")

aln_text = result.stdout

from Bio import AlignIO
aln_file = OUT_DIR / "sixdata_5homologs_alignment.fasta"
aln_file.write_text(aln_text)
aln = AlignIO.read(str(aln_file), "fasta")
n_seqs = len(aln)
aln_len = aln.get_alignment_length()
print(f"  比对完成: {n_seqs} 条 × {aln_len} 位点")

# 计算全长每列的信息含量
def column_info_content(col_str, n_total):
    aas = [c for c in col_str if c != '-']
    if len(aas) < n_total * 0.25:
        return 0.0, None  # 太多gap
    cnt = Counter(aas)
    freqs = np.array([cnt[aa] / len(aas) for aa in cnt])
    entropy = -np.sum(freqs * np.log2(freqs + 1e-12))
    r_seq = np.log2(20) - entropy
    correction = (len(cnt) - 1) / (2 * np.log(2) * len(aas))
    ic = max(0.0, r_seq - correction)
    # 获取该列的共识氨基酸
    consensus_aa = cnt.most_common(1)[0][0]
    return ic, consensus_aa

col_data = []
for i in range(aln_len):
    col = "".join(str(r.seq[i]) for r in aln)
    ic, consensus = column_info_content(col, n_seqs)
    col_data.append((ic, consensus, col))

# 统计信息含量分布
ic_values = [c[0] for c in col_data]
high_ic = sum(1 for ic in ic_values if ic > 1.0)
med_ic = sum(1 for ic in ic_values if 0.3 < ic <= 1.0)
low_ic = sum(1 for ic in ic_values if 0 < ic <= 0.3)
gap_cols = sum(1 for ic in ic_values if ic == 0.0)
print(f"  全长保守性分布: 高IC(>1.0)={high_ic}, 中IC={med_ic}, 低IC={low_ic}, Gap={gap_cols}")

# ═══════════════════════════════════════════════════════════
# Step 3: 找到两个保守功能区域（用于标注）
# ═══════════════════════════════════════════════════════════
print("\nStep 3: 识别两个保守功能域（用于全长Logo中标注）")

WIN = 40
window_scores = []
for i in range(aln_len - WIN + 1):
    ic_sum = sum(col_data[j][0] for j in range(i, i + WIN))
    gap_count = sum(1 for j in range(i, i + WIN) if col_data[j][0] == 0.0)
    adj = ic_sum * (1 - gap_count / WIN)
    window_scores.append((adj, i))

window_scores.sort(key=lambda x: x[0], reverse=True)

# 选两个非重叠窗口
domains = []
for adj, start in window_scores:
    if len(domains) == 0:
        domains.append((start, start + WIN))
    elif len(domains) < 2:
        d0_s, d0_e = domains[0]
        overlap = max(0, min(d0_e, start + WIN) - max(d0_s, start))
        if overlap < WIN * 0.25:
            domains.append((start, start + WIN))
            break
domains.sort()

# 用已知的 M1 氨肽酶功能域命名
for idx, (s, e) in enumerate(domains):
    rel_pos = s / aln_len
    if rel_pos < 0.4:
        label = "Exopeptidase Domain (GXMEN motif)"
    else:
        label = "Catalytic Domain (HEXXH zinc-binding)"
    print(f"  功能域 {idx+1}: 位点 {s+1}-{e} — {label}")

# ═══════════════════════════════════════════════════════════
# Step 4: 构建全长 Logo 矩阵
# ═══════════════════════════════════════════════════════════
print("\nStep 4: 构建全长 Sequence Logo")

# 过滤掉 gap 过多的列（>70% gap）
valid_cols = []
valid_positions = []
for i in range(aln_len):
    ic, cons, col_str = col_data[i]
    gap_ratio = col_str.count('-') / len(col_str)
    if gap_ratio <= 0.7:
        valid_cols.append(i)
        valid_positions.append(i + 1)  # 1-based

print(f"  有效位点: {len(valid_cols)}/{aln_len} (过滤了 {aln_len - len(valid_cols)} 个gap过多列)")

# 构建 Logo 矩阵 (频率模式：仅用频率作为高度，更清晰地展示 5 条序列间的变异)
# 信息含量单独在下方的保守性轨迹图中展示
logo_rows = []
conservation_ics = []
for i in valid_cols:
    col_str = "".join(str(r.seq[i]) for r in aln)
    aas = [c for c in col_str if c != '-']
    freqs = {aa: n / len(aas) for aa, n in Counter(aas).items()}
    # 计算 IC（用于下方保守性轨迹图）
    farr = np.array(list(freqs.values()))
    h = -np.sum(farr * np.log2(farr + 1e-12))
    ic = max(0.0, np.log2(20) - h - (len(freqs) - 1) / (2 * np.log(2) * len(aas)))
    conservation_ics.append(ic)
    # Logo 高度使用纯频率（范围 0.2-1.0），区分度更好
    logo_rows.append({aa: freq for aa, freq in freqs.items()})

# ═══════════════════════════════════════════════════════════
# Step 5: 绑图 — 全长 Sequence Logo（分为上下两段以容纳全部位点）
# ═══════════════════════════════════════════════════════════
print("\nStep 5: 绑制全长 Sequence Logo 图")

AA_COLORS = {
    'G': '#e69500', 'A': '#e6d200', 'V': '#e6d200', 'L': '#e6d200',
    'I': '#e6d200', 'P': '#cc9900',
    'F': '#1a9641', 'Y': '#1a9641', 'W': '#1a9641',
    'S': '#d8176c', 'T': '#d8176c', 'N': '#d8176c',
    'Q': '#d8176c', 'C': '#d8176c', 'M': '#d8176c',
    'K': '#0450b4', 'R': '#0450b4', 'H': '#0450b4',
    'D': '#d7191c', 'E': '#d7191c',
}

# 将有效位点分为两段（前半段和后半段），分别绘制上下两个Logo
# 这样两个 Logo 都展示全长范围，但分别对应两个功能域区域
midpoint = len(valid_cols) // 2

def draw_logo_panel(ax, rows_slice, positions_slice, title, domain_regions=None):
    """绘制一个 Logo 面板（频率模式）"""
    df = pd.DataFrame(0.0, index=range(len(rows_slice)), columns=list("ACDEFGHIKLMNPQRSTVWY"))
    for i, row in enumerate(rows_slice):
        for aa, val in row.items():
            if aa in df.columns:
                df.loc[i, aa] = val

    color_dict = {aa: AA_COLORS.get(aa, '#888888') for aa in df.columns}

    logomaker.Logo(
        df, ax=ax,
        color_scheme=color_dict,
        font_name='Arial',
        font_weight='bold',
        show_spines=True,
        baseline_width=0.5,
    )

    n_pos = len(rows_slice)
    step = max(1, n_pos // 25)
    tick_idx = list(range(0, n_pos, step))
    tick_labels = [str(positions_slice[i]) for i in tick_idx]
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([])  # 主 Logo 不显示 x 轴标签
    ax.set_ylabel('Frequency', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=12)

    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.12, linestyle='--', linewidth=0.5)

    # 高亮功能域
    if domain_regions:
        for ds, de, dlabel in domain_regions:
            ds_eff = max(0, sum(1 for p in positions_slice if p < ds + 1))
            de_eff = min(n_pos, sum(1 for p in positions_slice if p <= de))
            if de_eff > ds_eff:
                ax.axvspan(ds_eff - 0.5, de_eff - 0.5, alpha=0.06,
                           color='#e41a1c', zorder=0)
                ax.annotate(dlabel,
                    ((ds_eff + de_eff) / 2, ax.get_ylim()[1] * 0.97),
                    ha='center', fontsize=9, fontweight='bold',
                    color='#c4121c', style='italic',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor='#c4121c', alpha=0.85))


def draw_conservation_track(ax, ics_slice, positions_slice, domain_regions=None):
    """绘制保守性信息含量轨迹（Logo 下方的柱状图）"""
    n = len(ics_slice)
    x = np.arange(n)
    # 颜色编码：IC > 3.0 深红，2.0-3.0 橙色，1.0-2.0 蓝色，<1.0 灰色
    colors = []
    for ic in ics_slice:
        if ic >= 3.0:
            colors.append('#d7191c')
        elif ic >= 2.0:
            colors.append('#fdae61')
        elif ic >= 1.0:
            colors.append('#2c7bb6')
        else:
            colors.append('#bababa')
    ax.bar(x, ics_slice, color=colors, width=1.0, linewidth=0, alpha=0.85)
    ax.set_ylabel('IC (bits)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Alignment Position', fontsize=14, fontweight='bold')

    # x轴
    step = max(1, n // 25)
    tick_idx = list(range(0, n, step))
    tick_labels = [str(positions_slice[i]) for i in tick_idx]
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(tick_labels, fontsize=8, rotation=90)

    ax.set_ylim(0, max(4.5, max(ics_slice) * 1.1))
    ax.axhline(y=1.0, color='#999999', linestyle=':', linewidth=0.7, alpha=0.7)
    ax.axhline(y=2.0, color='#999999', linestyle=':', linewidth=0.7, alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.12, linestyle='--', linewidth=0.5)

    # 功能域
    if domain_regions:
        for ds, de, dlabel in domain_regions:
            ds_eff = max(0, sum(1 for p in positions_slice if p < ds + 1))
            de_eff = min(n, sum(1 for p in positions_slice if p <= de))
            if de_eff > ds_eff:
                ax.axvspan(ds_eff - 0.5, de_eff - 0.5, alpha=0.06,
                           color='#e41a1c', zorder=0)

# 功能域标注（用于高亮）
domain_annotations = [(s, e, "Exopeptidase" if s/aln_len < 0.4 else "Catalytic")
                      for (s, e) in domains]

# ── 分为两段（N端 + C端），每段包含 Logo + 保守性轨迹 ──
# 上半部分：N端区域 (first 60% of valid positions)
# 下半部分：C端区域
split_idx = int(len(valid_cols) * 0.6)

fig, axes = plt.subplots(4, 1, figsize=(30, 18),
                          gridspec_kw={'height_ratios': [2.5, 1, 2.5, 1]})
fig.patch.set_facecolor('white')

# Panel A: N端 Logo
draw_logo_panel(axes[0], logo_rows[:split_idx], valid_positions[:split_idx],
                "Sequence Logo — N-terminal Region (Exopeptidase Domain)",
                domain_annotations)
# Panel B: N端 保守性轨迹
draw_conservation_track(axes[1], conservation_ics[:split_idx], valid_positions[:split_idx],
                        domain_annotations)

# Panel C: C端 Logo
draw_logo_panel(axes[2], logo_rows[split_idx:], valid_positions[split_idx:],
                "Sequence Logo — C-terminal Region (Catalytic Domain)",
                domain_annotations)
# Panel D: C端 保守性轨迹
draw_conservation_track(axes[3], conservation_ics[split_idx:], valid_positions[split_idx:],
                        domain_annotations)

# ── 总体说明 ──
fig.text(0.5, 0.003,
         f"n = {n_seqs} M1 aminopeptidase sequences | MAFFT alignment ({aln_len} positions) | "
         f"Logo height = amino acid frequency | Conservation track = information content (bits) | "
         f"Red = IC≥3.0 (strictly conserved) | Orange = IC 2.0-3.0 | Blue = IC 1.0-2.0 | Grey = IC<1.0 | "
         f"Dashed lines at 1.0 & 2.0 bits",
         ha='center', fontsize=8.5, style='italic', color='#777777')

plt.tight_layout(rect=[0, 0.03, 1, 1], h_pad=1.5)

fig_path_300 = OUT_DIR / "fig_b_sequence_logo.png"
fig_path_600 = OUT_DIR / "fig_b_sequence_logo_600dpi.png"
fig.savefig(fig_path_300, dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(fig_path_600, dpi=600, bbox_inches='tight', facecolor='white')
plt.close()

print(f"\n✅ 300 DPI: {fig_path_300}")
print(f"✅ 600 DPI: {fig_path_600}")

# ── 统计 ──
total_pairs = 0
identical_pairs = 0
for i in range(aln_len):
    col = [str(r.seq[i]) for r in aln if str(r.seq[i]) != '-']
    if len(col) >= 3:
        for a in range(len(col)):
            for b in range(a + 1, len(col)):
                total_pairs += 1
                if col[a] == col[b]:
                    identical_pairs += 1

pct_id = 100 * identical_pairs / total_pairs if total_pairs > 0 else 0
print(f"\n比对统计: {aln_len}位点, 成对一致性={pct_id:.1f}%")
print(f"有效位点(≤70% gap): {len(valid_cols)}")
print(f"IC分布 — 高(>1.0):{high_ic}, 中(0.3-1.0):{med_ic}, 低(≤0.3):{low_ic}, Gap:{gap_cols}")

# 保存保守性剖面
pd.DataFrame({
    'position': range(1, aln_len + 1),
    'information_content': [round(c[0], 4) for c in col_data],
    'consensus_aa': [c[1] or '' for c in col_data],
}).to_csv(OUT_DIR / "conservation_profile.csv", index=False)

print("\nDONE — Sequence Logo v2 生成完毕")
