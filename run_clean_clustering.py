#!/usr/bin/env python3
"""
干净版：ESMC embeddings → 层次聚类 3类 → 每类 KMeans 分2
仅5000条序列，不含 sixdata
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
OUT = PROJECT / "outputs" / "run_5000"
OUT.mkdir(parents=True, exist_ok=True)

# ═══ 1. Load ESMC embeddings (only first 5000) ═══
print("Load ESMC embeddings...", flush=True)
X = np.load(str(PROJECT / "outputs" / "run_esmc" / "embeddings.npy"))[:5000]
X = StandardScaler().fit_transform(X)
n = len(X)
print(f"  {n} sequences, {X.shape[1]} dims", flush=True)

# ═══ 2. Level 1: Hierarchical 3 clusters ═══
print("\nLevel 1: Hierarchical (3 clusters)...", flush=True)
hc = AgglomerativeClustering(n_clusters=3, metric="cosine", linkage="average")
l1 = hc.fit_predict(X)
sil_l1 = silhouette_score(X, l1)
db_l1 = davies_bouldin_score(X, l1)
ch_l1 = calinski_harabasz_score(X, l1)
print(f"  Sil={sil_l1:.4f}  DB={db_l1:.2f}  CH={ch_l1:.1f}", flush=True)
for c in range(3):
    print(f"  C{c}: {int((l1==c).sum())} seqs ({int((l1==c).sum())/n*100:.1f}%)", flush=True)

# ═══ 3. Level 2: KMeans 2 within each L1 ═══
print("\nLevel 2: KMeans(2) within each L1 cluster...", flush=True)
l2 = np.full(n, -1, dtype=int)
sid = 0
for c in range(3):
    mask = l1 == c
    Xc = X[mask]
    nc = Xc.shape[0]
    if nc < 4:
        l2[mask] = sid; sid += 1; continue

    km = KMeans(n_clusters=2, n_init=30, random_state=42)
    sub = km.fit_predict(Xc)
    for s in range(2):
        sm = mask.copy(); sm[mask] = (sub == s)
        l2[sm] = sid
        sil_sub = silhouette_score(Xc, sub)
        print(f"  L1-C{c} → Sub{sid}: {int(sm.sum())} seqs  sil={sil_sub:.4f}", flush=True)
        sid += 1

sil_final = silhouette_score(X, l2)
db_final = davies_bouldin_score(X, l2)
print(f"\n  Final Sil={sil_final:.4f}  DB={db_final:.2f}", flush=True)

# ═══ 4. Save ═══
df_5k = pd.read_csv(PROJECT / "textdocs" / "final_submission_5000_sequences.csv")
df_out = pd.DataFrame({
    "Sequence_ID": df_5k["Sequence_ID"],
    "Sequence": df_5k["Sequence"],
    "Pred_kcat_over_Km": df_5k["Pred_kcat_over_Km"],
    "L1_cluster": l1,
    "L2_sub_cluster": l2,
})
df_out.to_csv(OUT / "cluster_assignments.csv", index=False)
print(f"\n✅ {OUT / 'cluster_assignments.csv'}", flush=True)

# ═══ 5. UMAP plot ═══
print("\nUMAP visualization...", flush=True)
import umap
um = umap.UMAP(n_neighbors=15, min_dist=0.05, metric="cosine", random_state=42, verbose=True)
xy = um.fit_transform(X)

pal6 = ["#e74c3c", "#c0392b", "#2ecc71", "#27ae60", "#3498db", "#2980b9"]

fig, axes = plt.subplots(1, 2, figsize=(24, 11), facecolor="white")

# ---- Left: Level 1 (3 clusters) ----
ax = axes[0]
pal3 = ["#e74c3c", "#2ecc71", "#3498db"]
for c in range(3):
    m = l1 == c
    ax.scatter(xy[m, 0], xy[m, 1], c=pal3[c], s=3, alpha=0.5,
               label=f"C{c} ({(l1==c).sum()})", edgecolors="none")
ax.set_title(f"Level 1: Hierarchical 3 Clusters\nSil={sil_l1:.3f}  DB={db_l1:.2f}  CH={ch_l1:.0f}",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=10, markerscale=4, loc="lower right")
ax.set_xticks([]); ax.set_yticks([])

# ---- Right: Level 2 (6 sub-clusters) ----
ax = axes[1]
for s in range(6):
    m = l2 == s
    ax.scatter(xy[m, 0], xy[m, 1], c=pal6[s], s=3, alpha=0.5,
               label=f"Sub{s} ({(l2==s).sum()})", edgecolors="none")
ax.set_title(f"Level 2: 3×2 = 6 Sub-clusters\nSil={sil_final:.3f}  DB={db_final:.2f}",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=10, markerscale=4, loc="lower right")
ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
sp = OUT / "clustering_3x2.png"
plt.savefig(sp, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"✅ {sp}", flush=True)

# Metrics JSON
with open(OUT / "clustering_metrics.json", "w") as f:
    json.dump({
        "level1": {"silhouette": round(sil_l1, 4), "davies_bouldin": round(db_l1, 2), "calinski_harabasz": round(ch_l1, 1)},
        "level2": {"silhouette": round(sil_final, 4), "davies_bouldin": round(db_final, 2)},
        "clusters": {f"C{c}": int((l1==c).sum()) for c in range(3)},
        "sub_clusters": {f"Sub{s}": int((l2==s).sum()) for s in range(6)},
    }, f, indent=2)

print("\nDONE ✅", flush=True)
