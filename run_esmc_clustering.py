#!/usr/bin/env python3
"""
ESMC_300M GPU 聚类 — sixdata 6条嵌入 5000 序列集
在 Linux GPU (RTX 4070Ti) 上运行

用法: python3 run_esmc_clustering.py
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parent
os.environ["INFRA_PROVIDER"] = "True"
sys.path.insert(0, str(PROJECT))

from src.clustering.cluster import ClusterAnalyzer
from src.features.extractor import _get_protein_encoder

OUT = PROJECT / "outputs" / "run_esmc"
OUT.mkdir(parents=True, exist_ok=True)

# ═══════════════ 1. DATA ═══════════════
print("Step 1: Build dataset", flush=True)
df = pd.read_csv(PROJECT / "textdocs" / "final_submission_5000_sequences.csv")
print(f"  Raw: {len(df)}", flush=True)

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

# ═══════════════ 2. ESMC ENCODE (GPU) ═══════════════
print(f"\nStep 2: ESMC_300M encoding {n} sequences on GPU...", flush=True)
encoder = _get_protein_encoder()
print(f"  Device: {encoder.device}", flush=True)

import time
t0 = time.time()
embeddings = encoder.encode(seqs).astype(np.float32)
t1 = time.time()
print(f"  Done in {t1-t0:.1f}s ({n/(t1-t0):.1f} seq/s)", flush=True)
print(f"  Shape: {embeddings.shape}", flush=True)

from sklearn.preprocessing import StandardScaler
embeddings = StandardScaler().fit_transform(embeddings)

# ═══════════════ 3. KMeans ═══════════════
print("\nStep 3: KMeans clustering (auto K)", flush=True)
az = ClusterAnalyzer(n_clusters=None)
labels = az.fit(embeddings)
k = az.chosen_k_
print(f"  K = {k}", flush=True)

disp = az.dispersion_report(embeddings)
g = disp["global"]
print(f"  Silhouette:        {g['silhouette_score']:.4f}", flush=True)
print(f"  Davies-Bouldin:    {g['davies_bouldin_score']:.4f}", flush=True)
print(f"  Calinski-Harabasz: {g['calinski_harabasz_score']:.2f}", flush=True)

for c in disp["per_cluster"]:
    print(f"  C{c['cluster']}: n={c['n_members']} radius={c['radius']:.4f}", flush=True)

ClusterAnalyzer.export_stats(disp, str(OUT))
pd.DataFrame({"id": ids, "cluster": labels}).to_csv(OUT / "seq_to_cluster.csv", index=False)
for nm, ti in zip(names, tgt_idx):
    print(f"  ★ {nm} → C{labels[ti]}", flush=True)

# ═══════════════ 4. UMAP ═══════════════
print("\nStep 4: UMAP visualization", flush=True)
import umap
um = umap.UMAP(n_components=2, n_neighbors=25, min_dist=0.08,
               metric="cosine", random_state=42, verbose=True)
xy = um.fit_transform(embeddings)
print(f"  UMAP shape: {xy.shape}", flush=True)

# ═══════════════ PLOT ═══════════════
print("\nGenerating scatter plot...", flush=True)
fig, ax = plt.subplots(figsize=(16, 14))
fig.patch.set_facecolor("#f8fafc")
ax.set_facecolor("#f8fafc")

pal = ["#2563eb", "#ea580c", "#16a34a", "#7c3aed", "#d946ef",
       "#0d9488", "#dc2626", "#f59e0b"]

for c in range(k):
    m = labels == c
    ax.scatter(xy[m, 0], xy[m, 1], c=pal[c], s=5, alpha=0.4,
               label=f"C{c} (n={m.sum()})", edgecolors="none", rasterized=True)

for ti, nm in zip(tgt_idx, names):
    ax.scatter(xy[ti, 0], xy[ti, 1], c="red", s=350, marker="*",
               edgecolors="black", linewidths=2, zorder=10)
    ox = (xy[:, 0].max() - xy[:, 0].min()) * 0.035
    oy = (xy[:, 1].max() - xy[:, 1].min()) * 0.035
    ax.annotate(f"{nm}(C{labels[ti]})", (xy[ti, 0], xy[ti, 1]),
                xytext=(xy[ti, 0] + ox, xy[ti, 1] + oy),
                fontsize=11, fontweight="bold", color="#dc2626",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#dc2626", alpha=0.92),
                arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.5), zorder=11)

ax.set_xlabel("UMAP-1", fontsize=14)
ax.set_ylabel("UMAP-2", fontsize=14)
ax.set_title(f"ESMC_300M + KMeans (K={k})  |  n={n}  |  GPU RTX 4070Ti  |  ★ = sixdata",
             fontsize=17, fontweight="bold", pad=20)
ax.legend(loc="upper right", fontsize=9, framealpha=0.9, markerscale=3)
ax.grid(alpha=0.12, linestyle="--")

cs = " | ".join([f"C{c}={(labels==c).sum()}" for c in range(k)])
ts = ", ".join([f"{nm}(C{labels[ti]})" for nm, ti in zip(names, tgt_idx)])
ax.text(0.02, 0.02,
        f"Feature: ESMC_300M (960-dim GPU)  K={k}  "
        f"Sil={g['silhouette_score']:.4f}  DB={g['davies_bouldin_score']:.2f}\n"
        f"Cluster sizes: {cs}\nTarget: {ts}",
        transform=ax.transAxes, fontsize=9.5, va="bottom", family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#ccc", alpha=0.9))

plt.tight_layout()
sp = OUT / "cluster_scatter.png"
plt.savefig(sp, dpi=250, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"\nDONE: {sp}", flush=True)
