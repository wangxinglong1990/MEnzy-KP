#!/usr/bin/env python3
"""
UMAP 降维聚类散点图 v3（Figure D）— ESM蛋白质语言模型embedding版
修复"没有聚类结构"问题：
  使用 ESM-C 300M 蛋白质语言模型提取 960-dim embedding 替代 k-mer 特征
  ESM embedding 能捕获结构/功能/进化信息，聚类质量大幅提升

用法: python3 generate_umap_scatter.py
"""

import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

_FONT_PATH = '/System/Library/AssetsV2/com_apple_MobileAsset_Font8/86ba2c91f017a3749571a82f2c6d890ac7ffb2fb.asset/AssetData/PingFang.ttc'
if os.path.exists(_FONT_PATH):
    fm.fontManager.addfont(_FONT_PATH)

PROJECT = Path(__file__).resolve().parent
os.chdir(str(PROJECT))
sys.path.insert(0, str(PROJECT))

from src.clustering.kmer import KmerFeatureExtractor
from src.clustering.cluster import ClusterAnalyzer

OUT_DIR = PROJECT / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── ESM-C 设备 ──
# ESMC 使用 BFloat16，MPS 不支持；CPU 原生支持 BFloat16
DEVICE = torch.device("cpu")
print(f"Device: {DEVICE} (BFloat16 not supported on MPS)")

# ═══════════════════════════════════════════════════════════
# Step 1: 数据准备 + k-mer 相似度过滤（同v2）
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("Step 1: 数据合并 + k-mer相似度过滤")
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
            "Full_Header": p[0].strip(),
            "Sequence": p[1].strip().upper(),
            "Length": len(p[1].strip()),
        })

df = df.drop_duplicates(subset=["Sequence"]).reset_index(drop=True)
sseq = {e["Sequence"].upper() for e in six}
df = df[~df["Sequence"].str.upper().isin(sseq)].reset_index(drop=True)

df_all = pd.concat([df, pd.DataFrame(six)], ignore_index=True)
n_total = len(df_all)
n_six = len(six)
ids_all = df_all["Sequence_ID"].tolist()
seqs_all = df_all["Sequence"].tolist()
tgt_indices_all = list(range(n_total - n_six, n_total))
target_names = ["AcAP", "TcAP", "EaAP", "KoAP", "MnAP", "MsAP"]

# ── k-mer 快速过滤 ──
print("  k-mer预过滤...")
ext = KmerFeatureExtractor(k=3)
X_all, ic_all, vc, sc = ext.build_matrix_from_lists(ids_all, seqs_all)
Xtf_all = ext.tfidf_transform(X_all)
if hasattr(Xtf_all, 'toarray'):
    Xtf_all = Xtf_all.toarray()
Xs_all = StandardScaler().fit_transform(Xtf_all)

tgt_feat = Xs_all[tgt_indices_all]
bg_feat = Xs_all[:n_total - n_six]
sims = cosine_similarity(bg_feat, tgt_feat).max(axis=1)

# 过滤阈值
sim_thresh = max(np.percentile(sims, 75), 0.01)
keep_mask = sims >= sim_thresh
kept_bg_idx = [i for i, k in enumerate(keep_mask) if k]
n_bg = len(kept_bg_idx)
print(f"  相似度阈值: {sim_thresh:.4f}, 保留: {n_bg} bg + {n_six} target = {n_bg + n_six}")

# 构建 filtered data
filtered_ids = [ids_all[i] for i in kept_bg_idx] + [ids_all[i] for i in tgt_indices_all]
filtered_seqs = [seqs_all[i] for i in kept_bg_idx] + [seqs_all[i] for i in tgt_indices_all]
n_total_f = len(filtered_ids)
tgt_indices_f = list(range(n_bg, n_total_f))
print(f"  最终: {n_total_f} 条序列")

# ═══════════════════════════════════════════════════════════
# Step 2: ESM-C 300M Embedding 提取
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 2: ESM-C 300M 蛋白质嵌入提取 (MPS加速)")
print("=" * 60)

# Monkey-patch data_root 以绕过 huggingface_hub 网络验证（离线环境）
import esm.utils.constants.esm3 as esm3_constants
_ORIG_DATA_ROOT = esm3_constants.data_root
def _offline_data_root(key):
    """离线返回本地缓存的模型路径"""
    import os as _os
    cache_map = {
        "esmc-300": _os.path.expanduser(
            "~/.cache/huggingface/hub/models--EvolutionaryScale--esmc-300m-2024-12/"
            "snapshots/7f10b20ae75017b2dbc884070e03434515709a8d"
        ),
        "esmc-600": _os.path.expanduser(
            "~/.cache/huggingface/hub/models--EvolutionaryScale--esmc-600m-2024-12/"
            "snapshots/7f10b20ae75017b2dbc884070e03434515709a8d"
        ),
    }
    from pathlib import Path
    p = cache_map.get(key)
    if p and Path(p).exists():
        return Path(p)
    return _ORIG_DATA_ROOT(key)
esm3_constants.data_root = _offline_data_root

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

print("  加载 ESMC 300M 模型 (CUDA, BFloat16, 离线模式)...")
client = ESMC.from_pretrained("esmc_300m", device=DEVICE)
client.eval()

def clean_seq(s):
    return "".join(c for c in s.upper() if c in "ACDEFGHIKLMNPQRSTVWY")

embeddings = []
batch_times = []
t_start = time.time()

for idx in range(n_total_f):
    seq = filtered_seqs[idx]
    cleaned = clean_seq(seq)
    if len(cleaned) > 400:
        cleaned = cleaned[:400]

    t0 = time.time()
    try:
        protein = ESMProtein(sequence=cleaned)
        protein_tensor = client.encode(protein)
        logits_output = client.logits(
            protein_tensor,
            LogitsConfig(sequence=True, return_embeddings=True),
        )
        emb = logits_output.embeddings
        if emb.dim() == 3:
            pooled = emb.mean(dim=1).squeeze(0)
        elif emb.dim() == 2:
            pooled = emb.mean(dim=0)
        else:
            pooled = emb.reshape(-1)
        vec = pooled.detach().cpu().numpy().astype(np.float32)
    except Exception as e:
        print(f"    ⚠️ seq {idx} error: {e}, using zeros")
        vec = np.zeros(960, dtype=np.float32)

    embeddings.append(vec)
    elapsed = time.time() - t0
    batch_times.append(elapsed)

    if (idx + 1) % 50 == 0 or idx == n_total_f - 1:
        avg_t = np.mean(batch_times[-50:]) if batch_times else 0
        eta = (n_total_f - idx - 1) * avg_t
        print(f"    [{idx+1}/{n_total_f}] avg {avg_t:.1f}s/seq, ETA {eta/60:.1f}min")

t_total_emb = time.time() - t_start
print(f"  嵌入提取完成: {t_total_emb/60:.1f} min ({t_total_emb/len(embeddings):.1f}s/seq)")

X_esm = np.array(embeddings, dtype=np.float32)
print(f"  ESM embeddings shape: {X_esm.shape}")

# 保存 embeddings 以便复用
np.save(OUT_DIR / "esm_embeddings.npy", X_esm)
np.save(OUT_DIR / "filtered_ids.npy", np.array(filtered_ids))

# ═══════════════════════════════════════════════════════════
# Step 3: 聚类 (ESM embedding 空间)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 3: KMeans 聚类 (ESM-960dim 空间)")
print("=" * 60)

Xs = StandardScaler().fit_transform(X_esm)

az = ClusterAnalyzer(n_clusters=None)  # 自动选K
labels = az.fit(Xs)
k = az.chosen_k_
disp = az.dispersion_report(Xs)
g = disp["global"]

print(f"  自动选择: K = {k}")
print(f"  Silhouette:        {g['silhouette_score']:.4f}")
print(f"  Davies-Bouldin:    {g['davies_bouldin_score']:.4f}")
print(f"  Calinski-Harabasz: {g['calinski_harabasz_score']:.2f}")

if az.k_selection_log_:
    print(f"  K选择过程:")
    for entry in az.k_selection_log_:
        sel = " ★" if entry.get("selected") else ""
        print(f"    K={entry['k']}: Sil={entry['silhouette']:.4f} "
              f"DB={entry['davies_bouldin']:.2f} CH={entry['calinski_harabasz']:.1f}{sel}")

cluster_roman = {}
for c in range(k):
    cluster_roman[c] = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X'][c]
    n_mem = (labels == c).sum()
    print(f"  Cluster {cluster_roman[c]}: n = {n_mem}")

print(f"\n  目标序列分配:")
for nm, ti in zip(target_names, tgt_indices_f):
    print(f"    ★ {nm} → Cluster {cluster_roman[labels[ti]]}")

pd.DataFrame({"id": filtered_ids, "cluster": labels}).to_csv(
    OUT_DIR / "seq_to_cluster_umap_esm.csv", index=False
)
ClusterAnalyzer.export_stats(disp, str(OUT_DIR))

# ═══════════════════════════════════════════════════════════
# Step 4: UMAP
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 4: UMAP 降维")
print("=" * 60)

import umap

n_neighbors_v = min(30, max(5, n_total_f // 15))
min_dist_v = 0.1

um = umap.UMAP(
    n_components=2, n_neighbors=n_neighbors_v, min_dist=min_dist_v,
    metric='cosine', random_state=42, verbose=True,
)
xy = um.fit_transform(Xs)
print(f"  UMAP shape: {xy.shape}")

# ═══════════════════════════════════════════════════════════
# Step 5: 出图
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 5: 绑制 UMAP 散点图")
print("=" * 60)

PALETTE = ['#377eb8', '#ff7f00', '#4daf4a', '#984ea3', '#a65628',
           '#f781bf', '#e41a1c', '#ffff33', '#66c2a5', '#fc8d62']
CLUSTER_COLORS = {c: PALETTE[c % len(PALETTE)] for c in range(k)}
TARGET_COLOR = '#e41a1c'

fig, ax = plt.subplots(figsize=(16, 14))
fig.patch.set_facecolor('white')
ax.set_facecolor('#fafbfc')

for c in range(k):
    mask = labels == c
    if mask.sum() == 0:
        continue
    ax.scatter(
        xy[mask, 0], xy[mask, 1],
        c=CLUSTER_COLORS[c], s=15, alpha=0.55,
        edgecolors='none', rasterized=True,
        label=f'Cluster {cluster_roman[c]} (n = {mask.sum():,})',
    )

# 目标星标
offsets = [
    (0.08, 0.08), (-0.09, 0.06), (0.07, -0.04),
    (-0.06, -0.07), (0.03, -0.09), (-0.04, 0.09),
]
x_range = xy[:, 0].max() - xy[:, 0].min()
y_range = xy[:, 1].max() - xy[:, 1].min()

for i, (ti, nm) in enumerate(zip(tgt_indices_f, target_names)):
    ax.scatter(
        xy[ti, 0], xy[ti, 1],
        c=TARGET_COLOR, s=400, marker='*',
        edgecolors='black', linewidths=2.2, zorder=10,
    )
    ox, oy = offsets[i]
    ax.annotate(
        f'{nm} (Cls {cluster_roman[labels[ti]]})',
        (xy[ti, 0], xy[ti, 1]),
        xytext=(xy[ti, 0] + ox * x_range, xy[ti, 1] + oy * y_range),
        fontsize=10.5, fontweight='bold', color='#c4121c',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                  edgecolor='#c4121c', alpha=0.93, linewidth=1.2),
        arrowprops=dict(arrowstyle='->', color='#555555', lw=1.2,
                        connectionstyle='arc3,rad=0.15'),
        zorder=11,
    )

ax.set_xlabel('UMAP Dimension 1', fontsize=16, fontweight='bold', labelpad=10)
ax.set_ylabel('UMAP Dimension 2', fontsize=16, fontweight='bold', labelpad=10)
ax.set_title(
    f'ESM-C Protein Language Model Embeddings\n'
    f'KMeans (K={k}) + UMAP — {n_total_f} Enzyme Sequences',
    fontsize=18, fontweight='bold', pad=20
)

legend = ax.legend(
    loc='upper right', fontsize=10.5, framealpha=0.92,
    markerscale=1.8, edgecolor='#cccccc', fancybox=True,
    title=f'Clusters (K = {k})', title_fontsize=11,
)
legend.get_frame().set_linewidth(0.8)
ax.grid(alpha=0.10, linestyle='--', linewidth=0.4)

stats_lines = [
    f"Features: ESM-C 300M (960-dim)",
    f"Sequences: {n_total_f:,} (k-mer filtered)",
    f"K = {k} (auto-selected)",
    f"Silhouette = {g['silhouette_score']:.4f}",
    f"Davies-Bouldin = {g['davies_bouldin_score']:.2f}",
    f"Calinski-Harabasz = {g['calinski_harabasz_score']:.1f}",
    "",
    f"Embedding time: {t_total_emb/60:.1f} min",
    "Target assignments:",
]
for nm, ti in zip(target_names, tgt_indices_f):
    stats_lines.append(f"  ★ {nm} → Cls {cluster_roman[labels[ti]]}")

ax.text(
    0.02, 0.02, "\n".join(stats_lines),
    transform=ax.transAxes, fontsize=8.5,
    verticalalignment='bottom', family='monospace',
    bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
              edgecolor='#aaaaaa', alpha=0.93, linewidth=0.8),
)

fig.text(
    0.5, 0.008,
    f"ESM-C 300M embeddings (960-dim) | "
    f"UMAP: n_neighbors={n_neighbors_v}, min_dist={min_dist_v}, metric=cosine | "
    f"★ = {n_six} target aminopeptidase candidates",
    ha='center', fontsize=8.5, style='italic', color='#777777'
)

plt.tight_layout(rect=[0, 0.022, 1, 1])

fig_path_300 = OUT_DIR / "fig_d_umap_clustering.png"
fig_path_600 = OUT_DIR / "fig_d_umap_clustering_600dpi.png"
fig.savefig(fig_path_300, dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(fig_path_600, dpi=600, bbox_inches='tight', facecolor='white')
plt.close()

print(f"\n✅ 300 DPI: {fig_path_300}")
print(f"✅ 600 DPI: {fig_path_600}")
print(f"\nDONE — UMAP v3 (ESM embeddings) 完成")
print(f"  Silhouette: {g['silhouette_score']:.4f}  "
      f"(v2 k-mer: -0.0288, 提升: {g['silhouette_score'] - (-0.0288):+.4f})")
