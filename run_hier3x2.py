#!/usr/bin/env python3
"""
层次3类 + 每类内部分2 (KMeans) = 6子簇
使用已缓存的 ESMC embeddings (GPU)
"""
import sys, json
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

OUT = PROJECT / "outputs" / "run_esmc"
EMB_PATH = OUT / "embeddings.npy"

# ═══ 1. Load cached embeddings ═══
print("="*60, flush=True)
print("Step 1: Load cached ESMC embeddings", flush=True)
embeddings = np.load(str(EMB_PATH))
print(f"  Shape: {embeddings.shape}", flush=True)
embeddings = StandardScaler().fit_transform(embeddings)
n = len(embeddings)

# Load sequence IDs
df = pd.read_csv(PROJECT / "textdocs" / "final_submission_5000_sequences.csv")
txt = open(PROJECT / "textdocs" / "sixdata.text").read()
six = []
for ln in txt.strip().split("\n"):
    if not ln.strip(): continue
    p = ln.split("：", 1) if "：" in ln else ln.split(":", 1)
    if len(p) == 2:
        six.append({"Sequence_ID": p[0].strip(), "Sequence": p[1].strip()})
df = df.drop_duplicates(subset=["Sequence"]).reset_index(drop=True)
sseq = {e["Sequence"].upper() for e in six}
df = df[~df["Sequence"].str.upper().isin(sseq)].reset_index(drop=True)
six_rows = [{"Sequence_ID": e["Sequence_ID"], "Sequence": e["Sequence"]} for e in six]
df = pd.concat([df, pd.DataFrame(six_rows)], ignore_index=True)
ids = df["Sequence_ID"].tolist()
tgt_idx = list(range(n - 6, n))
names = ["AcAP","TcAP","EaAP","KoAP","MnAP","MsAP"]
print(f"  {n} seqs ({n-6} unique + 6 targets)", flush=True)

# ═══ 2. Level 1: Hierarchical 3 ═══
print("\n" + "="*60, flush=True)
print("Step 2: Level 1 — Hierarchical clustering (K=3)", flush=True)
hc3 = AgglomerativeClustering(n_clusters=3, metric="cosine", linkage="average")
l1 = hc3.fit_predict(embeddings)
sil3 = silhouette_score(embeddings, l1)
db3 = davies_bouldin_score(embeddings, l1)
print(f"  Silhouette={sil3:.4f}  DB={db3:.2f}", flush=True)
for c in range(3):
    cnt = (l1 == c).sum()
    print(f"  L1-C{c}: n={cnt} ({cnt/n*100:.1f}%)", flush=True)

# ═══ 3. Level 2: KMeans K=2 per cluster ═══
print("\n" + "="*60, flush=True)
print("Step 3: Level 2 — KMeans K=2 within each L1 cluster", flush=True)
sub_labels = np.full(n, -1, dtype=int)
sub_cluster_id = 0
l2_stats = []

for c in range(3):
    mask = l1 == c
    X_sub = embeddings[mask]
    n_sub = X_sub.shape[0]

    if n_sub < 4:
        # Too few for K=2
        sub_labels[mask] = sub_cluster_id
        l2_stats.append({"l1_cluster": c, "sub_cluster": sub_cluster_id, "n_members": n_sub, "method": "skip"})
        sub_cluster_id += 1
        continue

    km = KMeans(n_clusters=2, n_init=30, random_state=42)
    sub = km.fit_predict(X_sub)

    for s in range(2):
        s_mask = mask.copy()
        s_mask[mask] = (sub == s)
        sub_labels[s_mask] = sub_cluster_id
        cnt = int(s_mask.sum())
        sil_sub = silhouette_score(X_sub, sub) if n_sub >= 5 else 0
        l2_stats.append({
            "l1_cluster": c, "sub_cluster": sub_cluster_id,
            "n_members": cnt, "sub_silhouette": round(sil_sub, 4), "method": "KMeans(2)"
        })
        sub_cluster_id += 1

    intra = []
    for s in range(2):
        s_mask = X_sub[sub == s]
        if len(s_mask) > 1:
            centroid = s_mask.mean(axis=0)
            d = np.linalg.norm(s_mask - centroid, axis=1)
            intra.append({"sub": sub_cluster_id-2+s, "mean_intra": float(d.mean()), "radius": float(d.max())})

    print(f"  L1-C{c} ({n_sub} seqs): split into 2 sub-clusters", flush=True)
    for s in l2_stats[-2:]:
        print(f"    L2-C{s['sub_cluster']}: n={s['n_members']}  sil={s['sub_silhouette']:.4f}", flush=True)

n_sub_clusters = sub_cluster_id
print(f"\n  Total sub-clusters: {n_sub_clusters}", flush=True)

# ═══ 4. Global metrics ═══
print("\n" + "="*60, flush=True)
print("Step 4: Final metrics", flush=True)
sil_final = silhouette_score(embeddings, sub_labels)
db_final = davies_bouldin_score(embeddings, sub_labels)
print(f"  Final 6 sub-clusters: Silhouette={sil_final:.4f}  DB={db_final:.2f}", flush=True)

# Sixdata targets
print("\n  ★ Sixdata target assignments:", flush=True)
for nm, ti in zip(names, tgt_idx):
    print(f"    {nm}: L1-C{l1[ti]}  →  L2-C{sub_labels[ti]}", flush=True)

# ═══ 5. Save ═══
print("\n" + "="*60, flush=True)
print("Step 5: Save results", flush=True)

# Full cluster assignments
df_out = pd.DataFrame({
    "id": ids,
    "l1_cluster": l1,
    "l2_sub_cluster": sub_labels,
})
for nm, ti in zip(names, tgt_idx):
    df_out.loc[ti, "is_target"] = nm
csv_path = OUT / "seq_to_cluster_hier3x2.csv"
df_out.to_csv(csv_path, index=False)
print(f"  ✅ {csv_path}", flush=True)

# Stats
stats = {
    "method": "Hierarchical(3) + KMeans(2) per cluster",
    "n_total": n,
    "n_sub_clusters": int(n_sub_clusters),
    "overall_silhouette": round(sil_final, 4),
    "overall_davies_bouldin": round(db_final, 2),
    "level1_silhouette": round(sil3, 4),
    "level2_stats": l2_stats,
    "targets": {nm: {"l1": int(l1[ti]), "l2": int(sub_labels[ti])} for nm, ti in zip(names, tgt_idx)},
}
with open(OUT / "hier3x2_metrics.json", "w") as f:
    json.dump(stats, f, indent=2)
print(f"  ✅ {OUT / 'hier3x2_metrics.json'}", flush=True)

# ═══ 6. UMAP plot ═══
print("\n" + "="*60, flush=True)
print("Step 6: UMAP visualization", flush=True)
import umap
um = umap.UMAP(n_neighbors=25, min_dist=0.08, metric="cosine", random_state=42, verbose=False)
# Check if cached UMAP exists
umap_path = OUT / "umap_coords.npy"
if umap_path.exists():
    xy = np.load(str(umap_path))
    print(f"  Loaded cached UMAP: {xy.shape}", flush=True)
else:
    xy = um.fit_transform(embeddings)
    np.save(str(umap_path), xy)
    print(f"  UMAP computed: {xy.shape}", flush=True)

pal = ["#2563eb","#ea580c","#16a34a","#7c3aed","#d946ef","#0d9488","#dc2626","#f59e0b"]

# Plot A: L1 (3 clusters)
fig, ax = plt.subplots(figsize=(16, 14))
fig.patch.set_facecolor("#f8fafc"); ax.set_facecolor("#f8fafc")
for c in range(3):
    m = l1 == c
    ax.scatter(xy[m,0], xy[m,1], c=pal[c], s=5, alpha=0.4,
               label=f"L1-C{c} (n={m.sum()})", edgecolors="none", rasterized=True)
for ti, nm in zip(tgt_idx, names):
    ax.scatter(xy[ti,0], xy[ti,1], c="red", s=350, marker="*",
               edgecolors="black", linewidths=2, zorder=10)
    ax.annotate(f"{nm}(C{l1[ti]})", (xy[ti,0], xy[ti,1]),
                xytext=(xy[ti,0]+35, xy[ti,1]+35),
                fontsize=11, fontweight="bold", color="#dc2626",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#dc2626", alpha=0.92),
                arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.5), zorder=11)
ax.set_title(f"ESMC + Hierarchical 3 clusters (Level 1)  |  n={n}  |  Sil={sil3:.4f}",
             fontsize=16, fontweight="bold")
ax.legend(fontsize=9, markerscale=3); ax.grid(alpha=0.1)
plt.tight_layout()
sp = OUT / "hier3x2_level1.png"
plt.savefig(sp, dpi=250, bbox_inches="tight"); plt.close()
print(f"  ✅ {sp}", flush=True)

# Plot B: L2 (6 sub-clusters)
fig, ax = plt.subplots(figsize=(16, 14))
fig.patch.set_facecolor("#f8fafc"); ax.set_facecolor("#f8fafc")
for s in range(n_sub_clusters):
    m = sub_labels == s
    ax.scatter(xy[m,0], xy[m,1], c=pal[s], s=5, alpha=0.4,
               label=f"L2-C{s} (n={m.sum()})", edgecolors="none", rasterized=True)
for ti, nm in zip(tgt_idx, names):
    ax.scatter(xy[ti,0], xy[ti,1], c="red", s=350, marker="*",
               edgecolors="black", linewidths=2, zorder=10)
    ax.annotate(f"{nm}(L2-C{sub_labels[ti]})", (xy[ti,0], xy[ti,1]),
                xytext=(xy[ti,0]+35, xy[ti,1]+35),
                fontsize=11, fontweight="bold", color="#dc2626",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#dc2626", alpha=0.92),
                arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.5), zorder=11)
ax.set_title(f"ESMC + Hierarchical(3) × KMeans(2)  =  {n_sub_clusters} sub-clusters  |  Sil={sil_final:.4f}",
             fontsize=16, fontweight="bold")
ax.legend(fontsize=9, markerscale=3); ax.grid(alpha=0.1)
plt.tight_layout()
sp = OUT / "hier3x2_level2.png"
plt.savefig(sp, dpi=250, bbox_inches="tight"); plt.close()
print(f"  ✅ {sp}", flush=True)

print(f"\n{'='*60}", flush=True)
print("DONE ✅", flush=True)
