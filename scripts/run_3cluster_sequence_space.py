#!/usr/bin/env python3
"""
ESMC嵌入 → 聚类成3个自然簇 (AgglomerativeClustering) → 同簇同色 → UMAP
6条target强制为代表 ★
"""
import os, sys, time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
os.chdir(str(PROJECT))
sys.path.insert(0, str(PROJECT))

from src.features.extractor import _get_protein_encoder
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np, pandas as pd
import hdbscan

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9.5,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.1,
})

OUT = PROJECT / "outputs" / "stratified_3x2"
t_start = time.time()

CLUSTER_COLORS = {0: "#E64B35", 1: "#4DBBD5", 2: "#00A087"}
CLUSTER_NAMES  = {0: "Cluster I", 1: "Cluster II", 2: "Cluster III"}

# ═══════════════════════════════════════════════════
# 1. 加载数据
# ═══════════════════════════════════════════════════
print("Step 1: Loading 5000 sequences ...")
df = pd.read_csv(PROJECT / "textdocs" / "merged_5000_with_targets.csv")
print(f"  {len(df)} sequences")

# ═══════════════════════════════════════════════════
# 2. ESMC 编码
# ═══════════════════════════════════════════════════
print("\nStep 2: ESMC_300M encoding ...")
encoder = _get_protein_encoder()
print(f"  Device: {encoder.device}")
emb = encoder.encode(df["Sequence"].tolist()).astype(np.float32)
emb = StandardScaler().fit_transform(emb)
print(f"  Shape: {emb.shape}  |  {time.time()-t_start:.1f}s")

# ═══════════════════════════════════════════════════
# 3. 序列空间聚类 k=3
# ═══════════════════════════════════════════════════
print("\nStep 3: AgglomerativeClustering (k=3) in ESMC space ...")
agg = AgglomerativeClustering(n_clusters=3, metric="euclidean", linkage="ward")
raw_labels = agg.fit_predict(emb)

# 按簇内平均 kcat/Km 重排
scores = df["Pred_kcat_over_Km"].values
order = sorted(range(3), key=lambda c: np.nanmean(scores[raw_labels == c]), reverse=True)
mapping = {old: new for new, old in enumerate(order)}
labels = np.array([mapping[l] for l in raw_labels])

sil = silhouette_score(emb, labels)
db  = davies_bouldin_score(emb, labels)
print(f"  Silhouette: {sil:.4f}  |  DB-index: {db:.4f}")
for c in range(3):
    mask = labels == c
    tgt_in = []
    for tid in ["1.AcAP","2.TcAP","3.EaAP","4.KoAP","5.MnAP","6.MsAP"]:
        m = df[df["Sequence_ID"] == tid]
        if len(m) > 0 and labels[m.index[0]] == c:
            tgt_in.append({"1.AcAP":"AcAP","2.TcAP":"TcAP","3.EaAP":"EaAP","4.KoAP":"KoAP","5.MnAP":"MnAP","6.MsAP":"MsAP"}[tid])
    print(f"  {CLUSTER_NAMES[c]}: n={mask.sum()}  mean_score={np.nanmean(scores[mask]):.1f}  targets={tgt_in}")

df["cluster"] = labels

# ═══════════════════════════════════════════════════
# 4. UMAP
# ═══════════════════════════════════════════════════
print("\nStep 4: UMAP ...")
import umap
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=26, min_dist=0.3)
X_2d = reducer.fit_transform(emb)
print(f"  Shape: {X_2d.shape}")

# ═══════════════════════════════════════════════════
# 5. 散点图: 同簇同色 + 6 target ★
# ═══════════════════════════════════════════════════
print("Step 5: Rendering ...")
fig, ax = plt.subplots(figsize=(15, 11))

for c in range(3):
    mask = labels == c
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               c=CLUSTER_COLORS[c], s=10, alpha=0.55,
               edgecolors="none", rasterized=True,
               label=f"{CLUSTER_NAMES[c]} (n={mask.sum()})")

# 6条target ★
tgt_ids   = ["1.AcAP","2.TcAP","3.EaAP","4.KoAP","5.MnAP","6.MsAP"]
tgt_short = ["AcAP","TcAP","EaAP","KoAP","MnAP","MsAP"]
for tid, tshort in zip(tgt_ids, tgt_short):
    m = df[df["Sequence_ID"] == tid]
    if len(m) > 0:
        idx = m.index[0]
        c = CLUSTER_COLORS[labels[idx]]
        ax.scatter(X_2d[idx, 0], X_2d[idx, 1],
                   c=c, s=350, marker="*",
                   edgecolors="black", linewidths=2, zorder=30)
        ax.annotate(tshort, (X_2d[idx, 0], X_2d[idx, 1]),
                    textcoords="offset points", xytext=(10, 10),
                    fontsize=11, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor=c, alpha=0.9, lw=1.5))

# 图例
legend_h = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor=CLUSTER_COLORS[0], markersize=10,
           label=f"{CLUSTER_NAMES[0]} (n={(labels==0).sum()})"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor=CLUSTER_COLORS[1], markersize=10,
           label=f"{CLUSTER_NAMES[1]} (n={(labels==1).sum()})"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor=CLUSTER_COLORS[2], markersize=10,
           label=f"{CLUSTER_NAMES[2]} (n={(labels==2).sum()})"),
    Line2D([0],[0], marker="*", color="w", markerfacecolor="black", markersize=14,
           label="6 Target aminopeptidases"),
]
ax.legend(handles=legend_h, fontsize=10, loc="upper right",
          framealpha=0.92, edgecolor="gray", fancybox=True)

ax.set_xlabel("UMAP-1", fontsize=12)
ax.set_ylabel("UMAP-2", fontsize=12)
ax.set_title(
    f"Sequence-Space Clustering of 5,000 Aminopeptidase Homologs\n"
    f"ESMC_300M → AgglomerativeClustering (k=3, Ward) → UMAP\n"
    f"Silhouette = {sil:.3f}    |    Davies-Bouldin = {db:.3f}",
    fontsize=13, fontweight="bold", loc="left")
ax.grid(True, alpha=0.06)

# ═══════════════════════════════════════════════════
# 6. 保存
# ═══════════════════════════════════════════════════
out_png = OUT / "sequence_space_3clusters.png"
out_pdf = OUT / "sequence_space_3clusters.pdf"
fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")

out_csv = OUT / "sequence_space_3clusters_assignments.csv"
out_df = df[["Sequence_ID", "Sequence", "Pred_kcat_over_Km"]].copy()
out_df["cluster"] = labels
out_df["umap_x"] = X_2d[:, 0]
out_df["umap_y"] = X_2d[:, 1]
# mark targets
out_df["is_target"] = out_df["Sequence_ID"].isin(tgt_ids)
out_df.to_csv(out_csv, index=False)

t_end = time.time()
print(f"\n✅ PNG: {out_png}")
print(f"✅ PDF: {out_pdf}")
print(f"✅ CSV: {out_csv}")
print(f"⏱  Total: {t_end - t_start:.1f}s")
