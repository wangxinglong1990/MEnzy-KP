#!/usr/bin/env python3
"""
WebLogo — 基于 5000 条序列 (final_submission_5000_sequences.csv)
两行布局 | 蓝绿黑配色 | 误差棒 | Y轴 0-4.0 bits
参考 WebLogo 3.9.0 风格

用法: python3 generate_5000_weblogo.py
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
# Step 1: 数据准备 (5000 - 重复的6 + sixdata 6)
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("Step 1: 数据合并 (5000 — sixdata重复 + sixdata)")
print("=" * 60)

df = pd.read_csv(PROJECT / "textdocs" / "final_submission_5000_sequences.csv")
txt = (PROJECT / "textdocs" / "sixdata.text").read_text()

six = []
for ln in txt.strip().split('\n'):
    if not ln.strip():
        continue
    sep = '：' if '：' in ln else ':'
    p = ln.split(sep, 1)
    if len(p) == 2:
        six.append({
            "Sequence_ID": p[0].strip(),
            "Sequence": p[1].strip().upper(),
        })

def clean_seq(s):
    return "".join(c for c in str(s).upper() if c in "ACDEFGHIKLMNPQRSTVWY")

df = df.drop_duplicates(subset=["Sequence"]).reset_index(drop=True)
sseq = {clean_seq(e["Sequence"]) for e in six}
n_before = len(df)
df = df[~df["Sequence"].apply(clean_seq).isin(sseq)].reset_index(drop=True)
print(f"  移除重复: {n_before - len(df)} 条")

df_all = pd.concat([df, pd.DataFrame(six)], ignore_index=True)
print(f"  最终: {len(df_all)} 条 ({len(df)} 背景 + {len(six)} 目标)")

seqs = [clean_seq(s) for s in df_all["Sequence"]]
ids = df_all["Sequence_ID"].tolist()
n_seqs = len(seqs)

# 过滤极短序列
records = [(rid, s) for rid, s in zip(ids, seqs) if len(s) >= 50]
print(f"  过滤<50aa后: {len(records)} 条")

# ═══════════════════════════════════════════════════════════
# Step 2: MAFFT 多序列比对
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"Step 2: MAFFT 比对 ({len(records)} 条)")
print("=" * 60)

aln_file = OUT_DIR / "alignment_5000.fasta"

if aln_file.exists():
    print(f"  使用缓存: {aln_file}")
    aln_text = aln_file.read_text()
else:
    fasta_in = tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False, dir='/tmp')
    for rid, seq in records:
        short_id = str(rid).split()[0][:40]
        fasta_in.write(f">{short_id}\n{seq}\n")
    fasta_in.close()

    print(f"  运行 MAFFT (--retree 1 --maxiterate 0, ~{len(records)} 条, 预计较久)...")
    result = subprocess.run(
        ["mafft", "--retree", "1", "--maxiterate", "0", "--thread", "4", fasta_in.name],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"MAFFT failed: {result.stderr[-500:]}")
    aln_text = result.stdout
    aln_file.write_text(aln_text)

from Bio import AlignIO
tmp_aln = tempfile.NamedTemporaryFile(mode='w', suffix='.aln', delete=False, dir='/tmp')
tmp_aln.write(aln_text)
tmp_aln.close()
aln = AlignIO.read(tmp_aln.name, "fasta")
aln_len = aln.get_alignment_length()
n_aln = len(aln)
print(f"  比对完成: {n_aln} 条 × {aln_len} 位点")

# ═══════════════════════════════════════════════════════════
# Step 3: 保守性分析
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 3: 保守性分析")
print("=" * 60)

def column_stats(col_str, n_total):
    """计算信息含量、共识AA、误差估计"""
    aas = [c for c in col_str if c != '-']
    n_obs = len(aas)
    if n_obs < n_total * 0.02:  # >98% gap => 丢弃
        return 0.0, '-', 1.0, 0.0

    cnt = Counter(aas)
    freqs = np.array([cnt[aa] / n_obs for aa in cnt])
    entropy = -np.sum(freqs * np.log2(freqs + 1e-12))
    r_seq = np.log2(20) - entropy
    correction = (len(cnt) - 1) / (2 * np.log(2) * n_obs)
    ic = max(0.0, r_seq - correction)
    consensus = cnt.most_common(1)[0][0]
    gap_ratio = 1.0 - n_obs / n_total

    # 误差棒 (bootstrap 近似: SE ≈ sqrt(1 / (2*ln(2)*N)))
    se = np.sqrt(1.0 / (2.0 * np.log(2) * n_obs)) if n_obs > 1 else 0.0
    return ic, consensus, gap_ratio, se

col_data = []
for i in range(aln_len):
    col = "".join(str(r.seq[i]) for r in aln)
    ic, cons, gap_r, se = column_stats(col, n_aln)
    col_data.append((ic, cons, gap_r, se, col))

ic_values = [c[0] for c in col_data]
high = sum(1 for ic in ic_values if ic >= 3.0)
med_high = sum(1 for ic in ic_values if 2.0 <= ic < 3.0)
med = sum(1 for ic in ic_values if 1.0 <= ic < 2.0)
low = sum(1 for ic in ic_values if 0.3 <= ic < 1.0)
var = sum(1 for ic in ic_values if 0 < ic < 0.3)
gaps = sum(1 for ic in ic_values if ic == 0.0)
print(f"  严格(≥3.0):{high}  高度(2.0-3.0):{med_high}  中度(1.0-2.0):{med}  低度(0.3-1.0):{low}  可变(<0.3):{var}  Gap:{gaps}")

# ═══════════════════════════════════════════════════════════
# Step 4: 构建 Logo 矩阵
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 4: 构建 Logo 矩阵")
print("=" * 60)

# 过滤 >90% gap 列
valid = []
for i in range(aln_len):
    _, _, gap_r, _, _ = col_data[i]
    if gap_r <= 0.90:
        valid.append(i)
print(f"  有效位点: {len(valid)}/{aln_len}")

logo_rows = []
conservation_ics = []
error_bars = []
consensus_list = []
for i in valid:
    ic, cons, _, se, col_str = col_data[i]
    aas = [c for c in col_str if c != '-']
    freqs = {aa: n / len(aas) for aa, n in Counter(aas).items()}
    logo_rows.append({aa: freq * ic for aa, freq in freqs.items()})
    conservation_ics.append(ic)
    error_bars.append(se)
    consensus_list.append(cons)

n_eff = len(valid)
mid = n_eff // 2

# ═══════════════════════════════════════════════════════════
# Step 5: 绑图 — 两行 WebLogo (蓝绿黑配色 + 误差棒)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 5: 绑制 WebLogo")
print("=" * 60)

# ── 用户指定配色: 蓝(带电亲水) 绿(极性中性) 黑(疏水) ──
AA_COLORS = {
    # 蓝色: 带电/极性亲水
    'E': '#2166ac', 'D': '#2166ac', 'K': '#4393c3',
    'R': '#4393c3', 'Q': '#92c5de', 'N': '#92c5de',
    'H': '#92c5de',
    # 绿色: 极性中性/羟基类
    'G': '#4daf4a', 'S': '#4daf4a', 'T': '#4daf4a',
    'P': '#4daf4a', 'C': '#4daf4a', 'Y': '#4daf4a',
    # 黑色: 疏水非极性
    'L': '#1a1a1a', 'I': '#1a1a1a', 'V': '#1a1a1a',
    'F': '#1a1a1a', 'A': '#1a1a1a', 'M': '#1a1a1a',
    'W': '#1a1a1a',
}

def draw_weblogo_panel(ax, logo_slice, ic_slice, err_slice, cons_slice,
                       pos_slice, title, show_xlabel=True):
    """绘制单行 WebLogo（蓝绿黑配色 + 误差棒 + 共识标注）"""
    n = len(logo_slice)

    df = pd.DataFrame(0.0, index=range(n), columns=list("ACDEFGHIKLMNPQRSTVWY"))
    for i, row in enumerate(logo_slice):
        for aa, val in row.items():
            if aa in df.columns:
                df.loc[i, aa] = val

    color_dict = {aa: AA_COLORS.get(aa, '#666666') for aa in df.columns}

    logo = logomaker.Logo(
        df, ax=ax, color_scheme=color_dict,
        font_name='Arial', font_weight='bold',
        show_spines=True, baseline_width=0.3,
    )

    # ── 误差棒 ──
    for i in range(n):
        total_h = df.iloc[i].sum()
        if total_h > 0.01 and err_slice[i] > 0:
            ax.errorbar(i, total_h, yerr=err_slice[i],
                       fmt='none', ecolor='#333333', elinewidth=0.5,
                       capsize=1.5, capthick=0.4, alpha=0.6)

    # ── 上方共识氨基酸 ──
    ymax = max(df.sum(axis=1).max(), 1.0)
    y_top = ymax * 1.06
    for i in range(n):
        if cons_slice[i] and cons_slice[i] != '-':
            ax.text(i, y_top, cons_slice[i], ha='center', va='bottom',
                    fontsize=5.5, fontweight='bold', color='#333333',
                    fontfamily='monospace')

    # ── x轴: 每隔5个标位点号 ──
    tick_step = 5
    tick_idx = list(range(0, n, tick_step))
    tick_labels = [str(pos_slice[i]) for i in tick_idx]
    ax.set_xticks(tick_idx)
    if show_xlabel:
        ax.set_xticklabels(tick_labels, fontsize=8, rotation=0)
        ax.set_xlabel('Alignment Position', fontsize=14, fontweight='bold', labelpad=6)
    else:
        ax.set_xticklabels([])

    ax.set_ylabel('Information Content (bits)', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
    ax.set_ylim(0, 4.2)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', labelsize=9)
    ax.grid(axis='y', alpha=0.10, linestyle='--', linewidth=0.4)

# ── 绑图: 两行 ──
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(36, 16))
fig.patch.set_facecolor('white')
plt.subplots_adjust(hspace=0.35)

# 上半段
pos1 = [valid[p] + 1 for p in range(0, mid)]
draw_weblogo_panel(ax1,
    logo_rows[:mid], conservation_ics[:mid], error_bars[:mid],
    consensus_list[:mid], pos1,
    f"Sequence Logo — N-terminal Region ({n_seqs} sequences)",
    show_xlabel=False)

# 下半段
pos2 = [valid[p] + 1 for p in range(mid, n_eff)]
draw_weblogo_panel(ax2,
    logo_rows[mid:], conservation_ics[mid:], error_bars[mid:],
    consensus_list[mid:], pos2,
    f"Sequence Logo — C-terminal Region ({n_seqs} sequences)",
    show_xlabel=True)

# ── 配色图例 ──
legend_ax = fig.add_axes([0.78, 0.01, 0.20, 0.025])
legend_ax.set_axis_off()
items = [
    ('Charged/Polar (blue)', '#4393c3'),
    ('Polar neutral (green)', '#4daf4a'),
    ('Hydrophobic (black)', '#1a1a1a'),
]
for i, (label, color) in enumerate(items):
    legend_ax.add_patch(plt.Rectangle((i*0.34, 0), 0.32, 0.8,
                                       color=color, transform=legend_ax.transAxes))
    legend_ax.text(i*0.34+0.16, 0.4, label, transform=legend_ax.transAxes,
                   fontsize=5.5, ha='center', va='center', color='white',
                   fontweight='bold')

fig.text(0.5, 0.005,
         f"n = {n_seqs} sequences | MAFFT ({aln_len} positions, {n_eff} shown) | "
         f"Y-axis: 0–4.0 bits | Error bars = ±1 SE | "
         f"Consensus amino acids above each position",
         ha='center', fontsize=8.5, style='italic', color='#666666')

fig_path = OUT_DIR / "fig_b_5000_weblogo.png"
fig_path_600 = OUT_DIR / "fig_b_5000_weblogo_600dpi.png"
fig.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(fig_path_600, dpi=600, bbox_inches='tight', facecolor='white')
plt.close()

print(f"\n✅ 300 DPI: {fig_path}")
print(f"✅ 600 DPI: {fig_path_600}")
print(f"\nDONE — 5000条序列 WebLogo")
