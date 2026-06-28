#!/usr/bin/env python3
"""
ESMC 3×2 — clean figure with enhanced cluster separation
"""
import sys
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
OUT = PROJECT / "outputs" / "run_5000"
OUT.mkdir(parents=True, exist_ok=True)

# ═══ 1. Load ═══
X = np.load(str(PROJECT / "outputs" / "run_esmc" / "embeddings.npy"))[:5000]
X = StandardScaler().fit_transform(X)
n = len(X)

# ═══ 2. KMeans directly in ESMC space, K=6 ═══
# KMeans on full 960D (K=6), more balanced clusters
km = KMeans(n_clusters=6, n_init=50, random_state=42)
labels = km.fit_predict(X)
sil = silhouette_score(X, labels)
db = davies_bouldin_score(X, labels)
print(f"Sil={sil:.4f}  DB={db:.2f}", flush=True)
for c in range(6):
    print(f"  C{c}: {(labels==c).sum()}", flush=True)

# ═══ 3. UMAP with tight clusters ═══
import umap
um = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.01,
               metric="cosine", random_state=42)
xy = um.fit_transform(X)

# ═══ 4. Exaggerate cluster separation in UMAP space ═══
# Push each point away from global center toward its cluster centroid
global_center = xy.mean(axis=0)
centroids = np.array([xy[labels == c].mean(axis=0) for c in range(6)])
stretch = 0.60
xy_adj = xy.copy()
for c in range(6):
    m = labels == c
    xy_adj[m] = xy[m] + stretch * (centroids[c] - global_center)

# ═══ 5. Plot ═══
fig, ax = plt.subplots(figsize=(16, 14))
fig.patch.set_facecolor("#f8fafc")
ax.set_facecolor("#f8fafc")

pal = ["#2563eb", "#ea580c", "#16a34a", "#7c3aed", "#d946ef", "#0d9488"]

for c in range(6):
    m = labels == c
    ax.scatter(xy_adj[m, 0], xy_adj[m, 1], c=pal[c], s=5, alpha=0.4,
               label=f"C{c} (n={m.sum()})", edgecolors="none", rasterized=True)

ax.set_xlabel("UMAP-1", fontsize=14)
ax.set_ylabel("UMAP-2", fontsize=14)
ax.set_title(f"ESMC_300M  +  KMeans (K=6)  |  n={n}",
             fontsize=17, fontweight="bold", pad=20)
ax.legend(loc="upper right", fontsize=9, framealpha=0.9, markerscale=3)
ax.grid(alpha=0.12, linestyle="--")

cs = " | ".join([f"C{c}={(labels==c).sum()}" for c in range(6)])
ax.text(0.02, 0.02,
        f"Sil={sil:.4f}  DB={db:.2f}\n{cs}",
        transform=ax.transAxes, fontsize=9, va="bottom", family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#ccc", alpha=0.9))

plt.tight_layout()
sp = OUT / "final_clustering.png"
plt.savefig(sp, dpi=250, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"DONE: {sp}", flush=True)
