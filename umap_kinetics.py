#!/usr/bin/env python3
"""Top200 ESMC UMAP colored by kcat/Km percentile groups"""
import os, sys
os.environ["INFRA_PROVIDER"] = "True"
sys.path.insert(0, ".")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from src.features.extractor import _get_protein_encoder

df = pd.read_csv("blast500.csv")
df = df.sort_values("Pred_kcat_over_Km", ascending=False).head(200).reset_index(drop=True)
ratio = df["Pred_kcat_over_Km"].values
n = len(df)
labels = np.array([0]*60 + [1]*80 + [2]*60)

print("ESMC encoding 200 seqs on GPU...", flush=True)
encoder = _get_protein_encoder()
emb = StandardScaler().fit_transform(encoder.encode(df["Enzyme"].tolist()).astype(np.float32))
print(f"Shape: {emb.shape}", flush=True)

print("UMAP...", flush=True)
import umap
um = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
xy = um.fit_transform(emb)

pal = ["#dc2626", "#f59e0b", "#2563eb"]
names = ["Super-efficient (Top 30%)", "Efficient (Mid 40%)", "Moderate (Low 30%)"]
fig, ax = plt.subplots(figsize=(12, 10))
fig.patch.set_facecolor("#f8fafc"); ax.set_facecolor("#f8fafc")
for c in range(3):
    m = labels == c
    ax.scatter(xy[m,0], xy[m,1], c=pal[c], s=40, alpha=0.85,
               edgecolors="white", linewidths=0.5,
               label=f"{names[c]} (n={m.sum()}, mean kcat/Km={ratio[m].mean():.1f})")
    ax.scatter(xy[m,0].mean(), xy[m,1].mean(), c=pal[c], s=200, marker="X",
               edgecolors="white", linewidths=2, zorder=6)
ax.set_xlabel("UMAP-1", fontsize=14); ax.set_ylabel("UMAP-2", fontsize=14)
ax.set_title("Top200 ESMC UMAP colored by kcat/Km group", fontsize=15, fontweight="bold", pad=15)
ax.legend(fontsize=11, markerscale=1.5); ax.grid(alpha=0.12, linestyle="--")
plt.tight_layout()
out = "outputs/run_best/umap_by_kinetics.png"
plt.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"DONE: {out}", flush=True)
