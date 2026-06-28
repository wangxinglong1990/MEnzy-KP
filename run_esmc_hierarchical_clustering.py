#!/usr/bin/env python3
"""
ESMC_300M + KMeans + Hierarchical Clustering (3类×2)
5000 sequences + 6 sixdata targets
相比 k-mer TF-IDF，ESMC 嵌入更能捕捉蛋白质结构/进化信息

用法: python3 -u run_esmc_hierarchical_clustering.py
"""
import os, sys, time, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from src.clustering.cluster import ClusterAnalyzer
from src.features.extractor import _get_protein_encoder
import torch

OUT = PROJECT / "outputs" / "run_esmc"
OUT.mkdir(parents=True, exist_ok=True)

# ═══════════════ 1. DATA ═══════════════
print("=" * 60, flush=True)
print("Step 1: Build dataset", flush=True)
df = pd.read_csv(PROJECT / "textdocs" / "final_submission_5000_sequences.csv")
print(f"  Raw: {len(df)} seqs", flush=True)

# Parse sixdata
txt = open(PROJECT / "textdocs" / "sixdata.text").read()
six = []
for ln in txt.strip().split("\n"):
    if not ln.strip():
        continue
    p = ln.split("：", 1) if "：" in ln else ln.split(":", 1)
    if len(p) == 2:
        six.append({"Sequence_ID": p[0].strip(), "Sequence": p[1].strip()})

# Deduplicate
df = df.drop_duplicates(subset=["Sequence"]).reset_index(drop=True)
sseq = {e["Sequence"].upper() for e in six}
df = df[~df["Sequence"].str.upper().isin(sseq)].reset_index(drop=True)

# Merge
six_rows = [{"Sequence_ID": e["Sequence_ID"], "Sequence": e["Sequence"]} for e in six]
df = pd.concat([df, pd.DataFrame(six_rows)], ignore_index=True)
print(f"  Final: {len(df)} seqs ({len(df)-6} unique + 6 targets)", flush=True)

n = len(df)
ids = df["Sequence_ID"].tolist()
seqs = df["Sequence"].tolist()
tgt_idx = list(range(n - 6, n))
names = ["AcAP", "TcAP", "EaAP", "KoAP", "MnAP", "MsAP"]

# ═══════════════ 2. ESMC ENCODE ═══════════════
print("\n" + "=" * 60, flush=True)
print("Step 2: ESMC_300M encoding...", flush=True)

cache_path = OUT / "embeddings.npy"
if cache_path.exists():
    print("  Loading cached embeddings...", flush=True)
    embeddings = np.load(str(cache_path))
    print(f"  Shape: {embeddings.shape}", flush=True)
else:
    encoder = _get_protein_encoder()
    print(f"  Device: {encoder.device}", flush=True)
    t0 = time.time()
    embeddings = encoder.encode(seqs).astype(np.float32)
    t1 = time.time()
    print(f"  Done in {t1-t0:.1f}s ({n/(t1-t0):.1f} seq/s)", flush=True)
    print(f"  Shape: {embeddings.shape}", flush=True)
    np.save(str(cache_path), embeddings)
    print(f"  Cached to {cache_path}", flush=True)

# Standardize
embeddings = StandardScaler().fit_transform(embeddings)

# ═══════════════ 3. KMEANS (auto K) ═══════════════
print("\n" + "=" * 60, flush=True)
print("Step 3: KMeans (auto select K)", flush=True)
az = ClusterAnalyzer(n_clusters=None)
labels_kmeans = az.fit(embeddings)
k = az.chosen_k_
disp = az.dispersion_report(embeddings)
g = disp["global"]
print(f"  Auto K = {k}", flush=True)
print(f"  Silhouette:        {g['silhouette_score']:.4f}", flush=True)
print(f"  Davies-Bouldin:    {g['davies_bouldin_score']:.4f}", flush=True)
print(f"  Calinski-Harabasz: {g['calinski_harabasz_score']:.2f}", flush=True)
for c in disp["per_cluster"]:
    print(f"  C{c['cluster']}: n={c['n_members']}  radius={c['radius']:.4f}  mean_dist={c['mean_intra_dist']:.4f}")
for nm, ti in zip(names, tgt_idx):
    print(f"  ★ {nm} → C{labels_kmeans[ti]}", flush=True)

# ═══════════════ 4. HIERARCHICAL (3类×2) ═══════════════
print("\n" + "=" * 60, flush=True)
print("Step 4: Hierarchical clustering (3×2 = 6 clusters)", flush=True)

# 4a. 3 top-level categories
hc3 = AgglomerativeClustering(n_clusters=3, metric="cosine", linkage="average")
labels_hc3 = hc3.fit_predict(embeddings)
sil3 = silhouette_score(embeddings, labels_hc3)
db3 = davies_bouldin_score(embeddings, labels_hc3)
print(f"  Level 1 (3 clusters):", flush=True)
print(f"  Silhouette: {sil3:.4f}  DB: {db3:.4f}", flush=True)
for c in range(3):
    cnt = (labels_hc3 == c).sum()
    intra = np.mean([
        np.linalg.norm(embeddings[i] - embeddings[labels_hc3 == c].mean(axis=0))
        for i in range(n) if labels_hc3[i] == c
    ]) if cnt > 0 else 0
    print(f"  C{c}: n={cnt}", flush=True)

# 4b. 6 sub-clusters (3×2)
hc6 = AgglomerativeClustering(n_clusters=6, metric="cosine", linkage="average")
labels_hc6 = hc6.fit_predict(embeddings)
sil6 = silhouette_score(embeddings, labels_hc6)
db6 = davies_bouldin_score(embeddings, labels_hc6)
print(f"\n  Level 2 (6 clusters = 3×2):", flush=True)
print(f"  Silhouette: {sil6:.4f}  DB: {db6:.4f}", flush=True)
for c in range(6):
    cnt = (labels_hc6 == c).sum()
    print(f"  HC{c}: n={cnt}", flush=True)
for nm, ti in zip(names, tgt_idx):
    print(f"  ★ {nm} → HC{labels_hc6[ti]}", flush=True)

# ── Save cluster assignments ──
df_out = pd.DataFrame({
    "id": ids,
    "kmeans_cluster": labels_kmeans,
    "hc_3_cluster": labels_hc3,
    "hc_6_cluster": labels_hc6,
})
for nm, ti in zip(names, tgt_idx):
    df_out.loc[ti, "is_target"] = nm
df_out.to_csv(OUT / "seq_to_cluster_esmc.csv", index=False)
print(f"\n  Saved: {OUT / 'seq_to_cluster_esmc.csv'}", flush=True)

# ── Save metrics ──
metrics = {
    "kmer_k5": {
        "silhouette": -0.000356,
        "davies_bouldin": 7.520,
        "method": "k-mer+TF-IDF+KMeans(k=5)"
    },
    "esmc_kmeans": {
        "silhouette": g['silhouette_score'],
        "davies_bouldin": g['davies_bouldin_score'],
        "calinski_harabasz": g['calinski_harabasz_score'],
        "k": k,
        "method": f"ESMC_300M+KMeans(auto_k={k})"
    },
    "esmc_hierarchical_3": {
        "silhouette": sil3,
        "davies_bouldin": db3,
        "k": 3,
        "method": "ESMC_300M+Hierarchical(k=3)"
    },
    "esmc_hierarchical_6": {
        "silhouette": sil6,
        "davies_bouldin": db6,
        "k": 6,
        "method": "ESMC_300M+Hierarchical(k=6,3×2)"
    }
}
with open(OUT / "metrics_comparison.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"  Saved: {OUT / 'metrics_comparison.json'}", flush=True)

# ── Comparison table ──
print("\n" + "=" * 60, flush=True)
print("METRICS COMPARISON", flush=True)
print(f"{'Method':<40} {'Silhouette':>10} {'DB':>10} {'K':>4}", flush=True)
print("-" * 68, flush=True)
for key, m in metrics.items():
    print(f"{m['method']:<40} {m['silhouette']:>10.4f} {m['davies_bouldin']:>10.2f} {m.get('k', '-'):>4}", flush=True)

# ═══════════════ 5. UMAP + PLOTS ═══════════════
print("\n" + "=" * 60, flush=True)
print("Step 5: UMAP visualization", flush=True)
import umap
um = umap.UMAP(n_components=2, n_neighbors=25, min_dist=0.08,
               metric="cosine", random_state=42, verbose=True)
xy = um.fit_transform(embeddings)
print(f"  UMAP shape: {xy.shape}", flush=True)

pal = ["#2563eb", "#ea580c", "#16a34a", "#7c3aed", "#d946ef",
       "#0d9488", "#dc2626", "#f59e0b"]

# ── Plot A: KMeans ──
print("\n  Plot A: KMeans scatter", flush=True)
fig, ax = plt.subplots(figsize=(16, 14))
fig.patch.set_facecolor("#f8fafc")
ax.set_facecolor("#f8fafc")
for c in range(k):
    m = labels_kmeans == c
    ax.scatter(xy[m, 0], xy[m, 1], c=pal[c % len(pal)], s=5, alpha=0.4,
               label=f"C{c} (n={m.sum()})", edgecolors="none", rasterized=True)
for ti, nm in zip(tgt_idx, names):
    ax.scatter(xy[ti, 0], xy[ti, 1], c="red", s=350, marker="*",
               edgecolors="black", linewidths=2, zorder=10)
    ox = (xy[:, 0].max() - xy[:, 0].min()) * 0.035
    oy = (xy[:, 1].max() - xy[:, 1].min()) * 0.035
    ax.annotate(f"{nm}(C{labels_kmeans[ti]})", (xy[ti, 0], xy[ti, 1]),
                xytext=(xy[ti, 0] + ox, xy[ti, 1] + oy),
                fontsize=11, fontweight="bold", color="#dc2626",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#dc2626", alpha=0.92),
                arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.5), zorder=11)
ax.set_xlabel("UMAP-1", fontsize=14)
ax.set_ylabel("UMAP-2", fontsize=14)
ax.set_title(f"ESMC_300M + KMeans (K={k})  |  n={n}  |  ★ = sixdata",
             fontsize=17, fontweight="bold", pad=20)
ax.legend(loc="upper right", fontsize=9, framealpha=0.9, markerscale=3)
ax.grid(alpha=0.12, linestyle="--")
cs = " | ".join([f"C{c}={(labels_kmeans==c).sum()}" for c in range(k)])
ts = ", ".join([f"{nm}(C{labels_kmeans[ti]})" for nm, ti in zip(names, tgt_idx)])
ax.text(0.02, 0.02,
        f"ESMC_300M (960-dim)  K={k}  Sil={g['silhouette_score']:.4f}  DB={g['davies_bouldin_score']:.2f}\n{cs}\n{ts}",
        transform=ax.transAxes, fontsize=9.5, va="bottom", family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#ccc", alpha=0.9))
plt.tight_layout()
sp = OUT / "cluster_scatter_kmeans.png"
plt.savefig(sp, dpi=250, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  ✅ {sp}", flush=True)

# ── Plot B: Hierarchical 6 ──
print("  Plot B: Hierarchical (6 clusters) scatter", flush=True)
fig, ax = plt.subplots(figsize=(16, 14))
fig.patch.set_facecolor("#f8fafc")
ax.set_facecolor("#f8fafc")
for c in range(6):
    m = labels_hc6 == c
    ax.scatter(xy[m, 0], xy[m, 1], c=pal[c % len(pal)], s=5, alpha=0.4,
               label=f"HC{c} (n={m.sum()})", edgecolors="none", rasterized=True)
for ti, nm in zip(tgt_idx, names):
    ax.scatter(xy[ti, 0], xy[ti, 1], c="red", s=350, marker="*",
               edgecolors="black", linewidths=2, zorder=10)
    ox = (xy[:, 0].max() - xy[:, 0].min()) * 0.035
    oy = (xy[:, 1].max() - xy[:, 1].min()) * 0.035
    ax.annotate(f"{nm}(HC{labels_hc6[ti]})", (xy[ti, 0], xy[ti, 1]),
                xytext=(xy[ti, 0] + ox, xy[ti, 1] + oy),
                fontsize=11, fontweight="bold", color="#dc2626",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#dc2626", alpha=0.92),
                arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.5), zorder=11)
ax.set_xlabel("UMAP-1", fontsize=14)
ax.set_ylabel("UMAP-2", fontsize=14)
ax.set_title(f"ESMC_300M + Hierarchical (6 clusters, 3×2)  |  n={n}  |  ★ = sixdata",
             fontsize=17, fontweight="bold", pad=20)
ax.legend(loc="upper right", fontsize=9, framealpha=0.9, markerscale=3)
ax.grid(alpha=0.12, linestyle="--")
cs = " | ".join([f"HC{c}={(labels_hc6==c).sum()}" for c in range(6)])
ts = ", ".join([f"{nm}(HC{labels_hc6[ti]})" for nm, ti in zip(names, tgt_idx)])
ax.text(0.02, 0.02,
        f"ESMC_300M (960-dim)  Hierarchical 6clusters  Sil={sil6:.4f}  DB={db6:.2f}\n{cs}\n{ts}",
        transform=ax.transAxes, fontsize=9.5, va="bottom", family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#ccc", alpha=0.9))
plt.tight_layout()
sp = OUT / "cluster_scatter_hierarchical6.png"
plt.savefig(sp, dpi=250, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  ✅ {sp}", flush=True)

# ── Plot C: Hierarchical 3 ──
print("  Plot C: Hierarchical (3 clusters) scatter", flush=True)
fig, ax = plt.subplots(figsize=(16, 14))
fig.patch.set_facecolor("#f8fafc")
ax.set_facecolor("#f8fafc")
for c in range(3):
    m = labels_hc3 == c
    ax.scatter(xy[m, 0], xy[m, 1], c=pal[c % len(pal)], s=5, alpha=0.4,
               label=f"C{c} (n={m.sum()})", edgecolors="none", rasterized=True)
for ti, nm in zip(tgt_idx, names):
    ax.scatter(xy[ti, 0], xy[ti, 1], c="red", s=350, marker="*",
               edgecolors="black", linewidths=2, zorder=10)
    ox = (xy[:, 0].max() - xy[:, 0].min()) * 0.035
    oy = (xy[:, 1].max() - xy[:, 1].min()) * 0.035
    ax.annotate(f"{nm}(C{labels_hc3[ti]})", (xy[ti, 0], xy[ti, 1]),
                xytext=(xy[ti, 0] + ox, xy[ti, 1] + oy),
                fontsize=11, fontweight="bold", color="#dc2626",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#dc2626", alpha=0.92),
                arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.5), zorder=11)
ax.set_xlabel("UMAP-1", fontsize=14)
ax.set_ylabel("UMAP-2", fontsize=14)
ax.set_title(f"ESMC_300M + Hierarchical (3 clusters)  |  n={n}  |  ★ = sixdata",
             fontsize=17, fontweight="bold", pad=20)
ax.legend(loc="upper right", fontsize=9, framealpha=0.9, markerscale=3)
ax.grid(alpha=0.12, linestyle="--")
cs = " | ".join([f"C{c}={(labels_hc3==c).sum()}" for c in range(3)])
ts = ", ".join([f"{nm}(C{labels_hc3[ti]})" for nm, ti in zip(names, tgt_idx)])
ax.text(0.02, 0.02,
        f"ESMC_300M (960-dim)  Hierarchical 3clusters  Sil={sil3:.4f}  DB={db3:.2f}\n{cs}\n{ts}",
        transform=ax.transAxes, fontsize=9.5, va="bottom", family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#ccc", alpha=0.9))
plt.tight_layout()
sp = OUT / "cluster_scatter_hierarchical3.png"
plt.savefig(sp, dpi=250, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"  ✅ {sp}", flush=True)

print("\n" + "=" * 60, flush=True)
print("DONE ✅", flush=True)
print(f"All outputs → {OUT}/", flush=True)
