#!/usr/bin/env python3
"""
5000条序列 → k-mer+TF-IDF → KMeans (k=3) → UMAP散点图
对应 Manuscript(4).docx 中 ESKin-guided 序列空间聚类分析
"""

import os, sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
os.chdir(str(PROJECT))
sys.path.insert(0, str(PROJECT))

from src.clustering.kmer import KmerFeatureExtractor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ── 中文字体 (论文用英文则不需要，这里保留兼容) ──────────────
try:
    fp = '/System/Library/AssetsV2/com_apple_MobileAsset_Font8/86ba2c91f017a3749571a82f2c6d890ac7ffb2fb.asset/AssetData/PingFang.ttc'
    fm.fontManager.addfont(fp)
except Exception:
    pass

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

OUT = PROJECT / "outputs" / "run_5000_3cluster"
OUT.mkdir(parents=True, exist_ok=True)

# ═════════════════════════════════════════════════════════════
# 1. 加载数据
# ═════════════════════════════════════════════════════════════
print("Step 1: Loading 5000 sequences ...", flush=True)
df = pd.read_csv(PROJECT / "textdocs" / "final_submission_5000_sequences.csv")

# 解析 sixdata (6条目标序列)
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
# 2. k-mer (k=3) + TF-IDF 特征提取
# ═════════════════════════════════════════════════════════════
print("Step 2: k-mer (k=3) + TF-IDF feature extraction ...", flush=True)
ext = KmerFeatureExtractor(k=3)
X, ic, vc, sc = ext.build_matrix_from_lists(ids, seqs)
Xtf = ext.tfidf_transform(X)
if hasattr(Xtf, "toarray"):
    Xtf = Xtf.toarray()
# TruncatedSVD 降维 (稀疏高维 → 稠密低维)
print("  TruncatedSVD 7915 → 100 ...", flush=True)
svd = TruncatedSVD(n_components=100, random_state=42)
X_dense = svd.fit_transform(Xtf)
Xs = StandardScaler().fit_transform(X_dense)
print(f"  vocab={len(vc)}  svd_shape={Xs.shape}  "
      f"explained_var={svd.explained_variance_ratio_.sum():.3f}", flush=True)

# ═════════════════════════════════════════════════════════════
# 3. KMeans (k=3)
# ═════════════════════════════════════════════════════════════
print("Step 3: KMeans clustering (k=3) on SVD-reduced features ...", flush=True)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=30)
raw_labels = kmeans.fit_predict(Xs)

# 按簇内平均 Pred_kcat_over_Km 重排: cluster 0 = 最高分
if "Pred_kcat_over_Km" in df.columns:
    scores = df["Pred_kcat_over_Km"].values
else:
    scores = np.ones(n)
clust_order = sorted(range(3), key=lambda c: np.mean(scores[raw_labels == c]), reverse=True)
mapping = {old: new for new, old in enumerate(clust_order)}
labels = np.array([mapping[l] for l in raw_labels])

sil = silhouette_score(Xs, labels)
db = davies_bouldin_score(Xs, labels)
print(f"  Silhouette: {sil:.4f}  |  Davies-Bouldin: {db:.4f}", flush=True)

for c in range(3):
    mask = labels == c
    print(f"  Cluster {c}: n={mask.sum()}  "
          f"mean(kcat/Km)={scores[mask].mean():.2f}  "
          f"targets={[target_names[i-n+6] for i in np.where(mask)[0] if i in tgt_idx]}")

# ═════════════════════════════════════════════════════════════
# 4. UMAP 降维
# ═════════════════════════════════════════════════════════════
print("Step 4: UMAP dimensionality reduction ...", flush=True)
try:
    import umap
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3)
    X_umap = reducer.fit_transform(Xs)
    method = "UMAP"
except ImportError:
    from sklearn.decomposition import PCA
    reducer = PCA(n_components=2, random_state=42)
    X_umap = reducer.fit_transform(Xs)
    method = "PCA"
print(f"  {method} shape: {X_umap.shape}", flush=True)

# ═════════════════════════════════════════════════════════════
# 5. 单张散点图
# ═════════════════════════════════════════════════════════════
print("Step 5: Generating scatter plot ...", flush=True)

CLUSTER_COLORS = ["#E64B35", "#4DBBD5", "#00A087"]
CLUSTER_NAMES = {
    0: f"Cluster I (High kcat/Km, n={(labels==0).sum()})",
    1: f"Cluster II (Mid kcat/Km, n={(labels==1).sum()})",
    2: f"Cluster III (Low kcat/Km, n={(labels==2).sum()})",
}

fig, ax = plt.subplots(figsize=(14, 10))

# 绘制主要散点（非目标序列）
for c in range(3):
    mask = labels == c
    # 排除 target 序列
    mask_bg = mask.copy()
    mask_bg[tgt_idx] = False
    ax.scatter(X_umap[mask_bg, 0], X_umap[mask_bg, 1],
               c=CLUSTER_COLORS[c], s=6, alpha=0.45,
               edgecolors="none", rasterized=True)

# 突出显示 6 条目标序列
target_markers = ["*", "D", "s", "P", "X", "^"]
for i, (ti, name) in enumerate(zip(tgt_idx, target_names)):
    c = CLUSTER_COLORS[labels[ti]]
    ax.scatter(X_umap[ti, 0], X_umap[ti, 1],
               c=c, s=280, marker=target_markers[i],
               edgecolors="black", linewidths=1.5, zorder=20)
    ax.annotate(name,
                (X_umap[ti, 0], X_umap[ti, 1]),
                textcoords="offset points", xytext=(8, 8),
                fontsize=11, fontweight="bold", color=c,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=c, alpha=0.85))

# 图例
legend_elements = []
for c in range(3):
    from matplotlib.lines import Line2D
    legend_elements.append(
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=CLUSTER_COLORS[c], markersize=12,
               label=CLUSTER_NAMES[c])
    )
# 添加 target 标记说明
legend_elements.append(
    Line2D([0], [0], marker="*", color="w",
           markerfacecolor="black", markersize=12,
           label="6 Target aminopeptidases (AcAP,TcAP,EaAP,KoAP,MnAP,MsAP)")
)

ax.legend(handles=legend_elements, fontsize=9.5, loc="upper right",
          framealpha=0.92, edgecolor="gray", fancybox=True)

ax.set_xlabel(f"{method}-1", fontsize=12)
ax.set_ylabel(f"{method}-2", fontsize=12)
ax.set_title(
    f"Sequence-Space Clustering of 5,000 Aminopeptidase Homologs\n"
    f"k-mer (k=3) + TF-IDF + SVD(100d)  →  KMeans (k=3)  →  {method} Projection\n"
    f"Silhouette = {sil:.3f}    |    Davies-Bouldin = {db:.3f}",
    fontsize=14, fontweight="bold", loc="left"
)
ax.grid(True, alpha=0.1)

# ═════════════════════════════════════════════════════════════
# 6. 保存
# ═════════════════════════════════════════════════════════════
out_png = OUT / "umap_3clusters.png"
out_pdf = OUT / "umap_3clusters.pdf"
fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
print(f"\n✅ PNG: {out_png}")
print(f"✅ PDF: {out_pdf}")

# 保存 cluster assignments
assign_df = df[["Sequence_ID", "Sequence"]].copy()
assign_df["cluster"] = labels
assign_df["umap_x"] = X_umap[:, 0]
assign_df["umap_y"] = X_umap[:, 1]
if "Pred_kcat_over_Km" in df.columns:
    assign_df["Pred_kcat_over_Km"] = df["Pred_kcat_over_Km"]
assign_df.to_csv(OUT / "cluster_assignments_3clusters.csv", index=False)
print(f"📊 Assignments: {OUT / 'cluster_assignments_3clusters.csv'}")
