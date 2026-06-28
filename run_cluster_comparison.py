#!/usr/bin/env python3
"""
快速对比：k-mer TF-IDF vs ESMC 嵌入的聚类效果
在 500 条子集上测试（5分钟内出结果）
"""
import sys, time, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from src.clustering.kmer import KmerFeatureExtractor
from src.features.extractor import _get_protein_encoder

N_SAMPLE = 500
RANDOM_STATE = 42
OUT = PROJECT / "outputs" / "cluster_comparison"
OUT.mkdir(parents=True, exist_ok=True)

# ═══ 1. DATA ═══
print("="*60, flush=True)
df = pd.read_csv(PROJECT / "textdocs" / "final_submission_5000_sequences.csv")
rng = np.random.RandomState(RANDOM_STATE)
idx = rng.choice(len(df), N_SAMPLE, replace=False)
df = df.iloc[idx].reset_index(drop=True)
seqs = df["Sequence"].tolist()
ids = df["Sequence_ID"].tolist()
print(f"Sampled {len(df)} sequences", flush=True)

metrics = {}

# ═══ 2. K-MER ═══
print("\n" + "-"*40, flush=True)
print("METHOD A: k-mer (k=3) + TF-IDF + StandardScaler", flush=True)
t0 = time.time()
ext = KmerFeatureExtractor(k=3)
X, ic, vc, sc = ext.build_matrix_from_lists(ids, seqs)
Xtf = ext.tfidf_transform(X)
if hasattr(Xtf, 'toarray'): Xtf = Xtf.toarray()
X_kmer = StandardScaler().fit_transform(Xtf)
print(f"  vocab={len(vc)} shape={X_kmer.shape} ({time.time()-t0:.1f}s)", flush=True)

# -- KMeans auto K --
km = KMeans(n_init=30, random_state=42)
lk = km.fit_predict(X_kmer)
sil_k = silhouette_score(X_kmer, lk)
db_k = davies_bouldin_score(X_kmer, lk)
print(f"  KMeans(auto=5) Sil={sil_k:.4f} DB={db_k:.2f}", flush=True)

# -- Hierarchical 6 --
hc6 = AgglomerativeClustering(n_clusters=6, metric="cosine", linkage="average")
lh6 = hc6.fit_predict(X_kmer)
sil_h6 = silhouette_score(X_kmer, lh6)
db_h6 = davies_bouldin_score(X_kmer, lh6)
print(f"  Hierarchical(6) Sil={sil_h6:.4f} DB={db_h6:.2f}", flush=True)

metrics["kmer_kmeans"] = {"silhouette": sil_k, "davies_bouldin": db_k, "method": "k-mer TF-IDF + KMeans"}
metrics["kmer_hier6"]  = {"silhouette": sil_h6, "davies_bouldin": db_h6, "method": "k-mer TF-IDF + Hierarchical(6)"}

# ═══ 3. ESMC ═══
print("\n" + "-"*40, flush=True)
print("METHOD B: ESMC_300M + StandardScaler", flush=True)
t0 = time.time()
encoder = _get_protein_encoder()
print(f"  Device: {encoder.device}", flush=True)

# Encode with progress
X_esmc_list = []
for i, s in enumerate(seqs):
    X_esmc_list.append(encoder.encode([s])[0])
    if (i+1) % 100 == 0:
        print(f"  encoded {i+1}/{N_SAMPLE} ({time.time()-t0:.1f}s)", flush=True)
X_esmc = StandardScaler().fit_transform(np.asarray(X_esmc_list, dtype=np.float32))
print(f"  shape={X_esmc.shape} ({time.time()-t0:.1f}s total)", flush=True)

# -- KMeans auto K --
km = KMeans(n_init=30, random_state=42)
le = km.fit_predict(X_esmc)
sil_e = silhouette_score(X_esmc, le)
db_e = davies_bouldin_score(X_esmc, le)
print(f"  KMeans(auto=5) Sil={sil_e:.4f} DB={db_e:.2f}", flush=True)

# -- Hierarchical 6 --
hc6 = AgglomerativeClustering(n_clusters=6, metric="cosine", linkage="average")
lh6_e = hc6.fit_predict(X_esmc)
sil_h6_e = silhouette_score(X_esmc, lh6_e)
db_h6_e = davies_bouldin_score(X_esmc, lh6_e)
print(f"  Hierarchical(6) Sil={sil_h6_e:.4f} DB={db_h6_e:.2f}", flush=True)

metrics["esmc_kmeans"] = {"silhouette": sil_e, "davies_bouldin": db_e, "method": "ESMC_300M + KMeans"}
metrics["esmc_hier6"]  = {"silhouette": sil_h6_e, "davies_bouldin": db_h6_e, "method": "ESMC_300M + Hierarchical(6)"}

# ═══ 4. COMPARISON TABLE ═══
print("\n" + "="*60, flush=True)
print("COMPARISON (Silhouette ↑ better, Davies-Bouldin ↓ better)", flush=True)
print(f"{'Method':<40} {'Silhouette':>10} {'DB':>8}", flush=True)
print("-"*60, flush=True)
for k, v in metrics.items():
    print(f"{v['method']:<40} {v['silhouette']:>10.4f} {v['davies_bouldin']:>8.2f}", flush=True)

with open(OUT / "metrics_comparison_500.json", "w") as f:
    json.dump(metrics, f, indent=2)

# ═══ 5. UMAP PLOTS ═══
print("\nUMAP plots...", flush=True)
import umap
pal = ["#2563eb", "#ea580c", "#16a34a", "#7c3aed", "#d946ef", "#0d9488"]

um_k = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
xy_k = um_k.fit_transform(X_kmer)
um_e = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
xy_e = um_e.fit_transform(X_esmc)

fig, axes = plt.subplots(1, 2, figsize=(22, 10))
fig.suptitle(f"Clustering Comparison: k-mer vs ESMC (n={N_SAMPLE})", fontsize=16, fontweight="bold")

# K-mer
ax = axes[0]
for c in range(6):
    m = lh6 == c
    ax.scatter(xy_k[m,0], xy_k[m,1], c=pal[c], s=10, alpha=0.5, label=f"C{c}({m.sum()})", edgecolors="none")
ax.set_title(f"k-mer+TF-IDF + Hierarchical(6)\nSil={metrics['kmer_hier6']['silhouette']:.4f} DB={metrics['kmer_hier6']['davies_bouldin']:.2f}")
ax.legend(fontsize=7)
ax.grid(alpha=0.1)

# ESMC
ax = axes[1]
for c in range(6):
    m = lh6_e == c
    ax.scatter(xy_e[m,0], xy_e[m,1], c=pal[c], s=10, alpha=0.5, label=f"C{c}({m.sum()})", edgecolors="none")
ax.set_title(f"ESMC_300M + Hierarchical(6)\nSil={metrics['esmc_hier6']['silhouette']:.4f} DB={metrics['esmc_hier6']['davies_bouldin']:.2f}")
ax.legend(fontsize=7)
ax.grid(alpha=0.1)

plt.tight_layout()
sp = OUT / "comparison_500.png"
plt.savefig(sp, dpi=200, bbox_inches="tight")
plt.close()
print(f"DONE: {sp}", flush=True)
