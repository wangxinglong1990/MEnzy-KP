#!/usr/bin/env python3
"""Top200 ESMC + HDBSCAN vs KMeans K=3 → best scatter plot"""
import os, sys, time
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

PROJECT = Path(__file__).resolve().parent
os.environ["INFRA_PROVIDER"] = "True"
sys.path.insert(0, str(PROJECT))
from src.features.extractor import _get_protein_encoder

OUT = PROJECT / "outputs" / "run_best"
OUT.mkdir(parents=True, exist_ok=True)

# Load top200
print("Step 1: Top200 by Pred_kcat_over_Km", flush=True)
df = pd.read_csv(PROJECT / "blast500.csv")
df = df.sort_values("Pred_kcat_over_Km", ascending=False).head(200).reset_index(drop=True)
seqs = df["Enzyme"].tolist()
ids = df["Entry"].tolist()

# ESMC GPU
print("Step 2: ESMC GPU encoding", flush=True)
encoder = _get_protein_encoder()
print(f"  Device: {encoder.device}", flush=True)
t0 = time.time()
emb = encoder.encode(seqs).astype(np.float32)
print(f"  Done: {time.time()-t0:.1f}s", flush=True)
emb = StandardScaler().fit_transform(emb)

# Try HDBSCAN
print("\nStep 3: HDBSCAN", flush=True)
import hdbscan
best_sil, best_lbl, best_params = -1, None, None
for mc in [3, 5, 8, 10, 15]:
    c = hdbscan.HDBSCAN(min_cluster_size=mc, min_samples=2, metric="euclidean",
                        cluster_selection_method="eom")
    lbl = c.fit_predict(emb)
    n_cl = len(set(lbl)) - (1 if -1 in lbl else 0)
    n_noise = (lbl == -1).sum()
    if n_cl >= 2:
        mask = lbl != -1
        sil = silhouette_score(emb[mask], lbl[mask])
        print(f"  min_size={mc:2d}: K={n_cl} noise={n_noise} Sil={sil:.4f}", flush=True)
        if sil > best_sil:
            best_sil = sil
            best_lbl = lbl
            best_params = f"HDBSCAN(min_size={mc})"

# Try KMeans K=3
print("\nKMeans K=3:", flush=True)
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, n_init=30, random_state=42)
lbl3 = km.fit_predict(emb)
sil3 = silhouette_score(emb, lbl3)
db3 = davies_bouldin_score(emb, lbl3)
ch3 = calinski_harabasz_score(emb, lbl3)
print(f"  Sil={sil3:.4f} DB={db3:.4f} CH={ch3:.1f}", flush=True)
for c in range(3):
    print(f"  C{c}: n={(lbl3==c).sum()}", flush=True)

# Pick best
if sil3 > best_sil:
    labels = lbl3
    method = f"KMeans K=3"
    sil_val = sil3
    db_val = db3
else:
    labels = best_lbl
    method = best_params
    sil_val = best_sil
    mask = labels != -1
    db_val = davies_bouldin_score(emb[mask], labels[mask])

k = len(set(labels)) - (1 if -1 in labels else 0)
print(f"\nBest: {method} Sil={sil_val:.4f} DB={db_val:.4f}", flush=True)

# UMAP
print("\nStep 4: UMAP", flush=True)
import umap
um = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
               metric="cosine", random_state=42)
xy = um.fit_transform(emb)

# Plot
fig, ax = plt.subplots(figsize=(14, 11))
fig.patch.set_facecolor("#f8fafc"); ax.set_facecolor("#f8fafc")
pal = ["#2563eb","#ea580c","#16a34a","#7c3aed","#d946ef","#0d9488","#dc2626","#f59e0b"]

unique_clusters = sorted(set(labels))
for i, c in enumerate(unique_clusters):
    m = labels == c
    lbl_text = f"Noise (n={m.sum()})" if c == -1 else f"C{c} (n={m.sum()})"
    marker = "x" if c == -1 else "o"
    alpha = 0.5 if c == -1 else 0.85
    ax.scatter(xy[m,0], xy[m,1], c=pal[i%len(pal)] if c!=-1 else "#9e9e9e",
               s=30 if c!=-1 else 20, alpha=alpha, marker=marker,
               label=lbl_text, edgecolors="white", linewidths=0.5)

ax.set_xlabel("UMAP-1", fontsize=14); ax.set_ylabel("UMAP-2", fontsize=14)
ax.set_title(f"Top200 kcat/Km + ESMC + {method}  |  Sil={sil_val:.3f}  DB={db_val:.2f}",
             fontsize=15, fontweight="bold", pad=18)
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.12, linestyle="--")

cs = " | ".join([f"C{c}={(labels==c).sum()}" for c in unique_clusters if c!=-1])
ax.text(0.02, 0.02, f"Feature: ESMC_300M (GPU)  |  {method}\n{cs}",
        transform=ax.transAxes, fontsize=9, va="bottom", family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#ccc", alpha=0.9))

plt.tight_layout()
sp = OUT / "cluster_scatter.png"
plt.savefig(sp, dpi=250, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"\nDONE: {sp}", flush=True)
