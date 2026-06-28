#!/usr/bin/env python3
"""
ESMC_300M 嵌入 → KMeans (k=3) → UMAP 散点图
5000 条 aminopeptidase 同源序列的三簇聚类可视化

对应: Manuscript(4).docx — HDBSCAN sequence-space clustering
"""

import os, sys, time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
os.chdir(str(PROJECT))
sys.path.insert(0, str(PROJECT))

from src.features.extractor import _get_protein_encoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

OUT = PROJECT / "outputs" / "run_5000_esmc_3cluster"
OUT.mkdir(parents=True, exist_ok=True)

CLUSTER_COLORS = ["#E64B35", "#4DBBD5", "#00A087"]  # High红 / Mid蓝 / Low绿

# ═════════════════════════════════════════════════════════════
# 1. 加载 5000 序列
# ═════════════════════════════════════════════════════════════
print("═" * 55)
print("Step 1: Loading 5000 sequences ...", flush=True)
t0 = time.time()

df = pd.read_csv(PROJECT / "textdocs" / "final_submission_5000_sequences.csv")
txt = open(PROJECT / "textdocs" / "sixdata.text").read()
six = []
for ln in txt.strip().split("\n"):
    if not ln.strip():
        continue
    p = ln.split("：", 1) if "：" in ln else ln.split(":", 1)
    if len(p) == 2:
        six.append({"Sequence_ID": p[0].strip(), "Sequence": p[1].strip()})

df = df.drop_duplicates(subset=["Sequence"]).reset_index(drop=True)
sseq = {e["Sequence"].upper() for e in six}
df = df[~df["Sequence"].str.upper().isin(sseq)].reset_index(drop=True)
df = pd.concat([df, pd.DataFrame(six)], ignore_index=True)

n = len(df)
ids = df["Sequence_ID"].tolist()
seqs = df["Sequence"].tolist()
tgt_idx = list(range(n - 6, n))
target_names = ["AcAP", "TcAP", "EaAP", "KoAP", "MnAP", "MsAP"]
print(f"  {n} sequences ({n - 6} unique + 6 targets)", flush=True)

# ═════════════════════════════════════════════════════════════
# 2. ESMC_300M 编码
# ═════════════════════════════════════════════════════════════
print("\nStep 2: ESMC_300M encoding (CPU) ...", flush=True)
encoder = _get_protein_encoder()
print(f"  Device: {encoder.device}", flush=True)

_emb = encoder.encode(seqs).astype(np.float32)
embeddings = StandardScaler().fit_transform(_emb)
t1 = time.time()
print(f"  Shape: {embeddings.shape}  |  Time: {t1 - t0:.1f}s", flush=True)

# ═════════════════════════════════════════════════════════════
# 3. KMeans (k=3)
# ═════════════════════════════════════════════════════════════
print("\nStep 3: KMeans clustering (k=3) ...", flush=True)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=30)
raw_labels = kmeans.fit_predict(embeddings)

# 按簇内平均 kcat/Km 重排：cluster 0 = 最高催化效率
scores = df["Pred_kcat_over_Km"].values
clust_order = sorted(range(3),
                     key=lambda c: np.nanmean(scores[raw_labels == c]),
                     reverse=True)
mapping = {old: new for new, old in enumerate(clust_order)}
labels = np.array([mapping[l] for l in raw_labels])

sil = silhouette_score(embeddings, labels)
db  = davies_bouldin_score(embeddings, labels)
ch  = calinski_harabasz_score(embeddings, labels)
print(f"  Silhouette: {sil:.4f}  |  DB-index: {db:.4f}  |  CH-index: {ch:.1f}", flush=True)

for c in range(3):
    mask = labels == c
    tgt_in = [target_names[i - n + 6] for i in np.where(mask)[0] if i in tgt_idx]
    print(f"  Cluster {c}: n={mask.sum():5d}  "
          f"mean(kcat/Km)={np.nanmean(scores[mask]):.1f}  "
          f"targets={tgt_in}")

# ═════════════════════════════════════════════════════════════
# 4. UMAP 降维
# ═════════════════════════════════════════════════════════════
print("\nStep 4: UMAP dimensionality reduction ...", flush=True)
try:
    import umap
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3)
    X_2d = reducer.fit_transform(embeddings)
    method = "UMAP"
except ImportError:
    from sklearn.decomposition import PCA
    X_2d = PCA(n_components=2, random_state=42).fit_transform(embeddings)
    method = "PCA"
print(f"  {method} shape: {X_2d.shape}", flush=True)

# ═════════════════════════════════════════════════════════════
# 5. 单张散点图
# ═════════════════════════════════════════════════════════════
print("Step 5: Rendering scatter plot ...", flush=True)

fig, ax = plt.subplots(figsize=(15, 11))

cluster_names = {
    0: f"Cluster I  —  High kcat/Km  (n={(labels==0).sum()})",
    1: f"Cluster II —  Mid kcat/Km   (n={(labels==1).sum()})",
    2: f"Cluster III — Low kcat/Km   (n={(labels==2).sum()})",
}

# 背景散点（非目标序列）
for c in range(3):
    mask = labels == c
    mask_bg = mask.copy()
    mask_bg[tgt_idx] = False
    ax.scatter(X_2d[mask_bg, 0], X_2d[mask_bg, 1],
               c=CLUSTER_COLORS[c], s=5, alpha=0.40,
               edgecolors="none", rasterized=True)

# 6 条目标序列高亮
target_markers = ["*", "D", "s", "P", "X", "^"]
target_size   = [320, 220, 220, 220, 220, 220]
for i, (ti, name) in enumerate(zip(tgt_idx, target_names)):
    c = CLUSTER_COLORS[labels[ti]]
    ax.scatter(X_2d[ti, 0], X_2d[ti, 1],
               c=c, s=target_size[i], marker=target_markers[i],
               edgecolors="black", linewidths=1.5, zorder=20)
    ax.annotate(name,
                (X_2d[ti, 0], X_2d[ti, 1]),
                textcoords="offset points", xytext=(9, 9),
                fontsize=11, fontweight="bold", color="black",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=c, alpha=0.88, lw=1.5))

# 图例
legend_handles = []
for c in range(3):
    legend_handles.append(
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CLUSTER_COLORS[c],
               markersize=12, label=cluster_names[c]))
legend_handles.append(
    Line2D([0], [0], marker="*", color="w", markerfacecolor="black",
           markersize=14, label="6 target aminopeptidases (AcAP, TcAP, EaAP, KoAP, MnAP, MsAP)"))

ax.legend(handles=legend_handles, fontsize=10, loc="upper right",
          framealpha=0.93, edgecolor="gray", fancybox=True,
          borderpad=0.8, labelspacing=0.8)

ax.set_xlabel(f"{method}-1", fontsize=13)
ax.set_ylabel(f"{method}-2", fontsize=13)
ax.set_title(
    f"Sequence-Space Clustering of 5,000 Aminopeptidase Homologs\n"
    f"ESMC_300M embeddings (960d)  →  KMeans (k=3)  →  {method} projection\n"
    f"Silhouette = {sil:.3f}    |    Davies-Bouldin = {db:.3f}    |    Calinski-Harabasz = {ch:.0f}",
    fontsize=14, fontweight="bold", loc="left"
)
ax.grid(True, alpha=0.08)

# ═════════════════════════════════════════════════════════════
# 6. 保存
# ═════════════════════════════════════════════════════════════
out_png = OUT / "umap_esmc_3clusters.png"
out_pdf = OUT / "umap_esmc_3clusters.pdf"
fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
t2 = time.time()
print(f"\n✅ PNG: {out_png}")
print(f"✅ PDF: {out_pdf}")
print(f"⏱  Total time: {t2 - t0:.1f}s", flush=True)

# 保存聚类分配
assign = df[["Sequence_ID", "Sequence", "Pred_kcat_over_Km"]].copy()
assign["cluster"] = labels
assign["umap_x"] = X_2d[:, 0]
assign["umap_y"] = X_2d[:, 1]
assign.to_csv(OUT / "cluster_assignments_esmc_3clusters.csv", index=False)
print(f"📊 Assignments: {OUT / 'cluster_assignments_esmc_3clusters.csv'}")
