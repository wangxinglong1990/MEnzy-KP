#!/usr/bin/env python3
"""
豆包方案：ESM-2 嵌入 + UMAP 降维 + 3类×2 聚类 + 散点图
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
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

# ═══ 1. Load data ═══
df = pd.read_csv("seqdump(1).csv")
seqs = df["Sequence"].tolist()
ids = df["ID"].tolist()
n = len(seqs)
print(f"Loaded {n} sequences", flush=True)

# ═══ 2. ESM-2 embedding ═══
emb_path = OUT / "esm2_embeddings.npy"
if emb_path.exists():
    print("Loading cached ESM-2 embeddings...", flush=True)
    X = np.load(str(emb_path))
else:
    print("Loading ESM-2 model...", flush=True)
    from esm import pretrained
    model, alphabet = pretrained.load_model_and_alphabet("esm2_t33_650M_UR50D")
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Device: {device}", flush=True)

    batch_converter = alphabet.get_batch_converter()
    batch_size = 8
    embeddings = []

    for i in range(0, n, batch_size):
        batch_seqs = seqs[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        batch_data = list(zip(batch_ids, batch_seqs))
        batch_tokens, batch_lens = batch_converter(batch_data)
        batch_tokens = batch_tokens.to(device)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_embs = results["representations"][33]
            for j in range(len(batch_seqs)):
                sl = batch_lens[j]
                emb = token_embs[j, 1:sl+1].mean(dim=0).cpu().numpy()
                embeddings.append(emb)
        if (i // batch_size + 1) % 50 == 0:
            print(f"  {min(i+batch_size, n)}/{n}", flush=True)

    X = np.array(embeddings, dtype=np.float32)
    np.save(str(emb_path), X)
    print(f"Saved: {emb_path}  shape={X.shape}", flush=True)

X = StandardScaler().fit_transform(X)

# ═══ 3. Hierarchical 3 + KMeans 2 = 6 sub-clusters ═══
print("\nClustering...", flush=True)
hc = AgglomerativeClustering(n_clusters=3, metric="cosine", linkage="average")
l1 = hc.fit_predict(X)
l2 = np.full(n, -1, dtype=int)
sid = 0
for c in range(3):
    m = l1 == c
    sub = KMeans(n_clusters=2, n_init=30, random_state=42).fit_predict(X[m])
    for s in range(2):
        sm = m.copy(); sm[m] = (sub == s)
        l2[sm] = sid; sid += 1

k = sid
sil = silhouette_score(X, l2)
db = davies_bouldin_score(X, l2)
print(f"Sil={sil:.4f}  DB={db:.2f}", flush=True)
for c in range(k):
    print(f"  Sub{c}: {(l2==c).sum()}", flush=True)

# ═══ 4. Save cluster labels ═══
np.save(str(OUT / "cluster_labels.npy"), l2)
df_out = pd.DataFrame({"ID": ids, "cluster": l2})
df_out.to_csv(OUT / "esm2_cluster_assignments.csv", index=False)

# ═══ 5. UMAP ═══
print("\nUMAP...", flush=True)
from umap import UMAP
um = UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
          metric="cosine", random_state=42)
xy = um.fit_transform(X)

# ═══ 6. Plot — original style ═══
print("Plotting...", flush=True)
fig, ax = plt.subplots(figsize=(16, 14))
fig.patch.set_facecolor("#f8fafc")
ax.set_facecolor("#f8fafc")

pal = ["#2563eb", "#ea580c", "#16a34a", "#7c3aed", "#d946ef", "#0d9488"]

for c in range(k):
    m = l2 == c
    ax.scatter(xy[m, 0], xy[m, 1], c=pal[c], s=5, alpha=0.4,
               label=f"Sub{c} (n={m.sum()})", edgecolors="none", rasterized=True)

ax.set_xlabel("UMAP-1", fontsize=14)
ax.set_ylabel("UMAP-2", fontsize=14)
ax.set_title(f"ESM-2 (650M) + Hierarchical(3) * KMeans(2)  |  n={n}",
             fontsize=17, fontweight="bold", pad=20)
ax.legend(loc="upper right", fontsize=9, framealpha=0.9, markerscale=3)
ax.grid(alpha=0.12, linestyle="--")

cs = " | ".join([f"Sub{c}={(l2==c).sum()}" for c in range(k)])
ax.text(0.02, 0.02,
        f"Sil={sil:.4f}  DB={db:.2f}\n{cs}",
        transform=ax.transAxes, fontsize=9, va="bottom", family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#ccc", alpha=0.9))

plt.tight_layout()
sp = OUT / "esm2_clustering.png"
plt.savefig(sp, dpi=250, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()

# Also save 2D coordinates
df_2d = pd.DataFrame({"ID": ids, "UMAP1": xy[:, 0], "UMAP2": xy[:, 1], "cluster": l2})
df_2d.to_csv(OUT / "esm2_umap_coords.csv", index=False)

print(f"\nDONE: {sp}", flush=True)
