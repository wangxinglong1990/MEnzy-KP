#!/usr/bin/env python3
"""
Sequence Logo — 基于 PSI-BLAST 493 条氨肽酶同源序列（Figure B 最终版）
输入: datafor/Ajinomoto_psiblast_output.xlsx
步骤: 读取序列 → 清洗过滤 → MAFFT 多序列比对 → 全长 WebLogo + 保守性轨迹

用法: python3 generate_psiblast_weblogo.py
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
# Step 1: 读取 PSI-BLAST 序列
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("Step 1: 读取 PSI-BLAST 输出")
print("=" * 60)

xlsx_path = PROJECT / "datafor" / "Ajinomoto_psiblast_output.xlsx"
df = pd.read_excel(xlsx_path)

def clean_seq(s):
    return "".join(c for c in str(s).upper() if c in "ACDEFGHIKLMNPQRSTVWY")

raw_seqs = df['Enzyme'].astype(str).tolist()
raw_ids = df['条目'].astype(str).tolist()

# 清洗并过滤
records = []
for rid, seq in zip(raw_ids, raw_seqs):
    cseq = clean_seq(seq)
    if len(cseq) >= 200:  # 排除过短序列
        records.append((rid, cseq))

print(f"  原始: {len(raw_seqs)} 条")
print(f"  过滤后(≥200aa): {len(records)} 条")
print(f"  长度分布: min={min(len(s) for _,s in records)}, "
      f"max={max(len(s) for _,s in records)}, "
      f"median={np.median([len(s) for _,s in records]):.0f}")

# 去重（完全相同序列）
seen = set()
uniq_records = []
for rid, seq in records:
    if seq not in seen:
        seen.add(seq)
        uniq_records.append((rid, seq))
print(f"  完全去重后: {len(uniq_records)} 条")

# ── 序列去冗余（降低保守度）──
# 贪婪选择：保留彼此相似度 < 阈值的序列，使 Logo 更具层次感
def seq_identity(s1, s2):
    """快速估算两条序列的相似度（基于 k-mer 交集）"""
    k = 3
    kmers1 = set(s1[i:i+k] for i in range(len(s1)-k+1))
    kmers2 = set(s2[i:i+k] for i in range(len(s2)-k+1))
    if not kmers1 or not kmers2:
        return 1.0
    return len(kmers1 & kmers2) / min(len(kmers1), len(kmers2))

IDENTITY_THRESHOLD = 0.50  # 保留 k-mer 相似度 < 50% 的序列
derep_records = []
for rid, seq in uniq_records:
    is_dup = False
    for _, kept_seq in derep_records:
        if seq_identity(seq, kept_seq) >= IDENTITY_THRESHOLD:
            is_dup = True
            break
    if not is_dup:
        derep_records.append((rid, seq))
print(f"  去冗余后 (k-mer相似度<{IDENTITY_THRESHOLD:.0%}): {len(derep_records)} 条")

# 使用去冗余后的序列集
final_records = derep_records
ids = [r[0] for r in final_records]
seqs = [r[1] for r in final_records]
n_seqs = len(seqs)

# ═══════════════════════════════════════════════════════════
# Step 2: MAFFT 多序列比对
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"Step 2: MAFFT 多序列比对 ({n_seqs} 条)")
print("=" * 60)

fasta_in = tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False, dir='/tmp')
for rid, seq in final_records:
    short_id = rid.split()[0][:40]
    fasta_in.write(f">{short_id}\n{seq}\n")
fasta_in.close()

# MAFFT — 使用缓存（按序列数命名）
aln_file = OUT_DIR / f"psiblast_{n_seqs}_alignment.fasta"

if aln_file.exists():
    print(f"  使用缓存比对: {aln_file}")
    aln_text = aln_file.read_text()
else:
    print(f"  运行 MAFFT (--retree 2 --maxiterate 0, {n_seqs} 条)...")
    result = subprocess.run(
        ["mafft", "--retree", "2", "--maxiterate", "0", "--thread", "4", fasta_in.name],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  MAFFT stderr: {result.stderr[-500:]}")
        raise RuntimeError("MAFFT failed")
    aln_text = result.stdout
    aln_file.write_text(aln_text)

from Bio import AlignIO
# 临时写入再读取
tmp_aln = tempfile.NamedTemporaryFile(mode='w', suffix='.aln', delete=False, dir='/tmp')
tmp_aln.write(aln_text)
tmp_aln.close()

aln = AlignIO.read(tmp_aln.name, "fasta")
n_aln_seqs = len(aln)
aln_len = aln.get_alignment_length()
print(f"  比对完成: {n_aln_seqs} 条 × {aln_len} 位点")

# ═══════════════════════════════════════════════════════════
# Step 3: 计算全长保守性
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 3: 全长保守性分析")
print("=" * 60)

def column_conservation(col_str, n_total):
    """计算一列的信息含量和共识氨基酸"""
    aas = [c for c in col_str if c != '-']
    if len(aas) < n_total * 0.05:  # > 95% gap
        return 0.0, '-', 1.0
    cnt = Counter(aas)
    freqs = np.array([cnt[aa] / len(aas) for aa in cnt])
    entropy = -np.sum(freqs * np.log2(freqs + 1e-12))
    r_seq = np.log2(20) - entropy
    correction = (len(cnt) - 1) / (2 * np.log(2) * len(aas))
    ic = max(0.0, r_seq - correction)
    consensus = cnt.most_common(1)[0][0]
    gap_ratio = 1.0 - len(aas) / n_total
    return ic, consensus, gap_ratio

col_data = []
for i in range(aln_len):
    col = "".join(str(r.seq[i]) for r in aln)
    ic, cons, gap_r = column_conservation(col, n_aln_seqs)
    col_data.append((ic, cons, gap_r, col))

ic_values = [c[0] for c in col_data]
# 分类统计
strict_conserved = sum(1 for ic in ic_values if ic >= 3.0)
high_conserved = sum(1 for ic in ic_values if 2.0 <= ic < 3.0)
moderate = sum(1 for ic in ic_values if 1.0 <= ic < 2.0)
low = sum(1 for ic in ic_values if 0.3 <= ic < 1.0)
variable = sum(1 for ic in ic_values if 0 < ic < 0.3)
gaps = sum(1 for ic in ic_values if ic == 0.0)

print(f"  严格保守(IC≥3.0): {strict_conserved} 位点")
print(f"  高度保守(2.0-3.0): {high_conserved} 位点")
print(f"  中度保守(1.0-2.0): {moderate} 位点")
print(f"  低度保守(0.3-1.0): {low} 位点")
print(f"  可变(<0.3): {variable} 位点")
print(f"  Gap区域: {gaps} 位点")

# 成对一致性
sample_n = min(50, n_aln_seqs)
sample_indices = [int(x) for x in np.random.RandomState(42).choice(n_aln_seqs, sample_n, replace=False)]
total_pairs = 0
identical_pairs = 0
for idx in sample_indices:
    for jdx in sample_indices:
        if idx >= jdx:
            continue
        s1 = str(aln[idx].seq).replace('-', '')
        s2 = str(aln[jdx].seq).replace('-', '')
        for a, b in zip(s1, s2):
            total_pairs += 1
            if a == b:
                identical_pairs += 1

pct_id = 100 * identical_pairs / total_pairs if total_pairs > 0 else 0
print(f"  采样成对一致性 (~{sample_n}×{sample_n}): {pct_id:.1f}%")

# ═══════════════════════════════════════════════════════════
# Step 4: 过滤 Gap 过多的列
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 4: 构建 Logo 矩阵 (过滤 >80% gap 的列)")
print("=" * 60)

valid_indices = []
for i in range(aln_len):
    _, _, gap_r, _ = col_data[i]
    if gap_r <= 0.80:  # 保留至少20%序列在该位点有残基的列
        valid_indices.append(i)

print(f"  有效位点: {len(valid_indices)}/{aln_len} "
      f"(过滤 {aln_len - len(valid_indices)} 个 gap 过多列)")

# 构建 Logo 矩阵 (信息含量模式 — 保守度差异更明显)
logo_rows = []
conservation_ics = []
for i in valid_indices:
    ic, _, _, col_str = col_data[i]
    aas = [c for c in col_str if c != '-']
    freqs = {aa: n / len(aas) for aa, n in Counter(aas).items()}
    # Logo 高度 = 频率 × 信息含量（bits），保守性差异更大
    logo_rows.append({aa: freq * ic for aa, freq in freqs.items()})
    conservation_ics.append(ic)

# ═══════════════════════════════════════════════════════════
# Step 5: 识别功能域（基于保守性）
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 5: 识别保守功能域")
print("=" * 60)

WIN = 50
window_scores = []
for i in range(len(valid_indices) - WIN + 1):
    ic_sum = sum(conservation_ics[j] for j in range(i, i + WIN))
    window_scores.append((ic_sum, i))

window_scores.sort(key=lambda x: x[0], reverse=True)

# 找两个非重叠高保守区域
domains = []
for ic_sum, start in window_scores:
    if len(domains) == 0:
        domains.append((start, start + WIN, ic_sum / WIN))
    elif len(domains) < 2:
        d0_s, d0_e, _ = domains[0]
        overlap = max(0, min(d0_e, start + WIN) - max(d0_s, start))
        if overlap < WIN * 0.3:
            domains.append((start, start + WIN, ic_sum / WIN))
            break
domains.sort()

# 映射回原始比对位点
for idx, (ds, de, avg_ic) in enumerate(domains):
    orig_start = valid_indices[ds] + 1
    orig_end = valid_indices[de - 1] + 1
    print(f"  功能域 {idx+1}: 比对位点 {orig_start}-{orig_end}, 平均 IC = {avg_ic:.2f} bits")

# ═══════════════════════════════════════════════════════════
# Step 6: 绑制 WebLogo 图（松散排版 + 共识氨基酸标注 + 位点号）
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 6: 绑制 WebLogo 图（松散排版）")
print("=" * 60)

AA_COLORS = {
    'G': '#e69500', 'A': '#e6d200', 'V': '#e6d200', 'L': '#e6d200',
    'I': '#e6d200', 'P': '#cc9900',
    'F': '#1a9641', 'Y': '#1a9641', 'W': '#1a9641',
    'S': '#d8176c', 'T': '#d8176c', 'N': '#d8176c',
    'Q': '#d8176c', 'C': '#d8176c', 'M': '#d8176c',
    'K': '#0450b4', 'R': '#0450b4', 'H': '#0450b4',
    'D': '#d7191c', 'E': '#d7191c',
}

# 收集共识氨基酸序列
consensus_aas = []
for i in valid_indices:
    _, cons, _, _ = col_data[i]
    consensus_aas.append(cons)

# ── 将有效位点分为三段（更松散）──
n_eff = len(valid_indices)
chunk_size = n_eff // 3 + 1
row1_end = chunk_size
row2_end = min(chunk_size * 2, n_eff)

slices = [
    (0, row1_end, "N-terminal Region"),
    (row1_end, row2_end, "Central Region"),
    (row2_end, n_eff, "C-terminal Region"),
]

def draw_one_panel(ax, logo_slice, ic_slice, cons_slice, pos_slice,
                   title, show_xlabels=True):
    """绘制单行 WebLogo（纯 Logo，无轨迹图）"""
    n = len(logo_slice)

    # 构建 DataFrame
    df = pd.DataFrame(0.0, index=range(n), columns=list("ACDEFGHIKLMNPQRSTVWY"))
    for i, row in enumerate(logo_slice):
        for aa, val in row.items():
            if aa in df.columns:
                df.loc[i, aa] = val

    color_dict = {aa: AA_COLORS.get(aa, '#888888') for aa in df.columns}

    # 使用 logomaker 绘制
    logo = logomaker.Logo(
        df, ax=ax, color_scheme=color_dict,
        font_name='Arial', font_weight='bold',
        show_spines=True, baseline_width=0.3,
        stack_order='fixed',
    )

    # ── 在 Logo 上方标注共识氨基酸 ──
    ymax = df.sum(axis=1).max()
    y_top = ymax * 1.08
    for i in range(n):
        cons_aa = cons_slice[i]
        if cons_aa != '-' and cons_slice[i]:
            ax.text(i, y_top, cons_aa, ha='center', va='bottom',
                    fontsize=5.5, fontweight='bold', color='#333333',
                    fontfamily='monospace')

    # ── x 轴：每隔 5 个位点标注比对位点号 ──
    tick_step = 5
    tick_idx = list(range(0, n, tick_step))
    tick_labels = [str(pos_slice[i]) for i in tick_idx]

    ax.set_xticks(tick_idx)
    if show_xlabels:
        ax.set_xticklabels(tick_labels, fontsize=8, rotation=0)
        ax.set_xlabel('Alignment Position', fontsize=14, fontweight='bold', labelpad=8)
    else:
        ax.set_xticklabels([])

    ax.set_ylabel('Information Content (bits)', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=14)

    # y 轴 (IC 模式：最大约 4.32 bits)
    ax.set_ylim(0, max(4.5, y_top * 1.08))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=9)
    ax.grid(axis='y', alpha=0.10, linestyle='--', linewidth=0.4)

# ── 绑图：三行纯 Logo（无轨迹图，更简洁松散）──
fig, axes = plt.subplots(3, 1, figsize=(36, 16))
fig.patch.set_facecolor('white')

# 控制行间距
plt.subplots_adjust(hspace=0.45)

for ax, (s, e, region_name), show_x in zip(
    axes, slices, [False, False, True]
):
    pos_slice = [valid_indices[p] + 1 for p in range(s, e)]  # 1-based
    draw_one_panel(
        ax,
        logo_rows[s:e],
        conservation_ics[s:e],
        consensus_aas[s:e],
        pos_slice,
        f"Sequence Logo — {region_name}",
        show_xlabels=show_x,
    )

# ── 底部图例和说明 ──
fig.text(0.5, 0.01,
         f"n = {n_seqs} PSI-BLAST aminopeptidase homologs | "
         f"MAFFT alignment ({aln_len} pos, {n_eff} shown) | "
         f"Pairwise identity ~{pct_id:.0f}% | "
         f"Consensus amino acids shown above each stack | "
         f"Numbers = alignment positions",
         ha='center', fontsize=9, style='italic', color='#555555')

# 配色图例
legend_ax = fig.add_axes([0.82, 0.008, 0.16, 0.025])
legend_ax.set_axis_off()
legend_items = [
    ('Non-polar', '#e6d200'), ('Aromatic', '#1a9641'),
    ('Polar', '#d8176c'), ('Basic', '#0450b4'), ('Acidic', '#d7191c'),
]
for i, (label, color) in enumerate(legend_items):
    legend_ax.add_patch(plt.Rectangle((i * 0.20, 0), 0.18, 0.8,
                                       color=color, transform=legend_ax.transAxes))
    legend_ax.text(i * 0.20 + 0.09, 0.4, label, transform=legend_ax.transAxes,
                   fontsize=5.5, ha='center', va='center', color='white',
                   fontweight='bold')

fig_path = OUT_DIR / "fig_b_psiblast_weblogo.png"
fig_path_600 = OUT_DIR / "fig_b_psiblast_weblogo_600dpi.png"
fig.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(fig_path_600, dpi=600, bbox_inches='tight', facecolor='white')
plt.close()

print(f"\n✅ 300 DPI: {fig_path}")
print(f"✅ 600 DPI: {fig_path_600}")

# ═══════════════════════════════════════════════════════════
# 汇总
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DONE — PSI-BLAST WebLogo 生成完毕")
print("=" * 60)
print(f"  输入: {xlsx_path}")
print(f"  序列: {n_seqs} 条 (原始 {len(raw_seqs)}, 去重过滤)")
print(f"  比对: MAFFT {aln_len} 位点")
print(f"  保守性: 严格{strict_conserved}, 高度{high_conserved}, "
      f"中度{moderate}, 低度{low}, 可变{variable}, Gap{gaps}")
