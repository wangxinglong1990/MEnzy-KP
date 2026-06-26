#!/usr/bin/env python3
"""blast500.csv → top200 by Pred_kcat_over_Km → ESMC GPU → KMeans → UMAP"""
import os, sys, time
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parent
os.environ["INFRA_PROVIDER"] = "True"
sys.path.insert(0, str(PROJECT))

from src.clustering.cluster import ClusterAnalyzer
from src.features.extractor import _get_protein_encoder

OUT = PROJECT / "outputs" / "run_top200"
OUT.mkdir(parents=True, exist_ok=True)

# 1. Load + Top200
print("Step 1: Load blast500.csv → Top200", flush=True)
df = pd.read_csv(PROJECT / "blast500.csv")
df = df.sort_values("Pred_kcat_over_Km", ascending=False).head(200).reset_index(drop=True)
seqs = df["Enzyme"].tolist()
ids = df["Entry"].tolist()
scores = df["Pred_kcat_over_Km"].tolist()
print(f"  Top 200 loaded, score range: {scores[0]:.1f} ~ {scores[-1]:.1f}", flush=True)

# 2. ESMC GPU
print("\nStep 2: ESMC_300M encoding on GPU...", flush=True)
encoder = _get_protein_encoder()
print(f"  Device: {encoder.device}", flush=True)
t0 = time.time()
emb = encoder.encode(seqs).astype(np.float32)
print(f"  Done in {time.time()-t0:.1f}s ({len(seqs)/(time.time()-t0):.0f} seq/s)", flush=True)
emb = StandardScaler().fit_transform(emb)

# 3. KMeans
print("\nStep 3: KMeans clustering", flush=True)
az = ClusterAnalyzer(n_clusters=None)
labels = az.fit(emb)
k = az.chosen_k_
disp = az.dispersion_report(emb)
g = disp["global"]
print(f"  K={k}  Sil={g['silhouette_score']:.4f}  DB={g['davies_bouldin_score']:.4f}  CH={g['calinski_harabasz_score']:.2f}", flush=True)
for c in disp["per_cluster"]:
    print(f"  C{c['cluster']}: n={c['n_members']} radius={c['radius']:.4f}", flush=True)

# 4. UMAP
print("\nStep 4: UMAP", flush=True)
import umap
um = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
               metric="cosine", random_state=42)
xy = um.fit_transform(emb)
print(f"  Shape: {xy.shape}", flush=True)

# 5. Plot
print("\nGenerating plot...", flush=True)
fig, ax = plt.subplots(figsize=(14, 11))
fig.patch.set_facecolor("#f8fafc"); ax.set_facecolor("#f8fafc")
pal = ["#2563eb","#ea580c","#16a34a","#7c3aed","#d946ef","#0d9488","#dc2626","#f59e0b"]

for c in range(k):
    m = labels == c
    ax.scatter(xy[m,0], xy[m,1], c=pal[c], s=25, alpha=0.85,
               label=f"C{c} (n={m.sum()})", edgecolors="white", linewidths=0.5)

# Annotate top 5
for rank in range(5):
    ax.annotate(f"#{rank+1}", (xy[rank,0], xy[rank,1]),
                fontsize=8, fontweight="bold", color=pal[labels[rank]],
                xytext=(5,5), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.8))

ax.set_xlabel("UMAP-1", fontsize=14); ax.set_ylabel("UMAP-2", fontsize=14)
ax.set_title(f"Top200 kcat/Km + ESMC + KMeans (K={k})  |  Sil={g['silhouette_score']:.3f}  DB={g['davies_bouldin_score']:.2f}",
             fontsize=15, fontweight="bold", pad=18)
ax.legend(loc="upper right", fontsize=9, markerscale=1.5)
ax.grid(alpha=0.12, linestyle="--")

cs = " | ".join([f"C{c}={(labels==c).sum()}" for c in range(k)])
ax.text(0.02, 0.02, f"Feature: ESMC_300M (GPU)  |  Top 200 by Pred_kcat_over_Km\n{cs}",
        transform=ax.transAxes, fontsize=9, va="bottom", family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ccc", alpha=0.9))

plt.tight_layout()
sp = OUT / "cluster_scatter.png"
plt.savefig(sp, dpi=250, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"\nDONE: {sp}", flush=True)
