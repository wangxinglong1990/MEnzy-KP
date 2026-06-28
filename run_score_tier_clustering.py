#!/usr/bin/env python3
"""
两步聚类：先按 Pred_kcat_over_Km 分 3 档，每档内 ESMC+KMeans 分 2
这样每簇既在 score 上有区分，又在 ESMC 空间上有区分
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))
OUT = PROJECT / "outputs" / "run_5000"

# ═══ Load data ═══
emb = np.load(OUT.parent / "run_esmc" / "embeddings.npy")
emb = StandardScaler().fit_transform(emb)
n = len(emb)

df_5k = pd.read_csv(PROJECT / "textdocs" / "final_submission_5000_sequences.csv")
scores = df_5k["Pred_kcat_over_Km"].values

# Sixdata targets (no scores, use ESMC-based assignment)
txt = open(PROJECT / "textdocs" / "sixdata.text").read()
six = []
for ln in txt.strip().split("\n"):
    if not ln.strip(): continue
    p = ln.split("：", 1) if "：" in ln else ln.split(":", 1)
    if len(p) == 2:
        six.append(p[0].strip())

# ═══ Step 1: Split by Pred_kcat_over_Km into 3 tiers ═══
p33, p67 = np.percentile(scores, [33, 67])
tiers = [
    ("Low",  scores <= p33,    f"≤{p33:.0f}"),
    ("Mid",  (scores > p33) & (scores <= p67), f"{p33:.0f}-{p67:.0f}"),
    ("High", scores > p67,     f"≥{p67:.0f}"),
]

# Assign 5000 sequences to tiers
tier_labels_5k = np.zeros(len(scores), dtype=int)
for i, (name, mask, _) in enumerate(tiers):
    tier_labels_5k[mask] = i

# For 6 sixdata targets (indices 5000-5005): assign by nearest neighbor (ESMC)
tgt_start = 5000
tier_labels = np.zeros(n, dtype=int)
tier_labels[:tgt_start] = tier_labels_5k
for ti in range(tgt_start, n):
    # Find nearest neighbor in ESMC space among 5000 seqs
    d = np.linalg.norm(emb[ti] - emb[:tgt_start], axis=1)
    nearest = np.argmin(d)
    tier_labels[ti] = tier_labels[nearest]

print("=" * 60, flush=True)
print("Step 1: Score-tier assignment", flush=True)
for i, (name, _, rng) in enumerate(tiers):
    cnt = (tier_labels == i).sum()
    print(f"  {name:>6} ({rng:>8}): {cnt:>4} seqs", flush=True)

# ═══ Step 2: Within each tier, cluster ESMC into 2 ═══
sub_labels = np.full(n, -1, dtype=int)
sub_id = 0
for tier in range(3):
    mask = (tier_labels == tier)
    X_sub = emb[mask]
    n_sub = X_sub.shape[0]
    if n_sub < 4:
        sub_labels[mask] = sub_id
        sub_id += 1
        continue

    km = KMeans(n_clusters=2, n_init=30, random_state=42)
    sub = km.fit_predict(X_sub)

    for s in range(2):
        s_mask = mask.copy()
        s_mask[mask] = (sub == s)
        sub_labels[s_mask] = sub_id
        cnt = int(s_mask.sum())
        # Silhouette within this tier
        sil = silhouette_score(X_sub, sub) if n_sub >= 5 else 0
        print(f"  Tier {name} → Sub{sub_id}: {cnt:>4} seqs  sil={sil:.4f}", flush=True)
        sub_id += 1

n_sub = sub_id
sil_final = silhouette_score(emb, sub_labels)
print(f"\n  Final {n_sub} sub-clusters: Silhouette={sil_final:.4f}", flush=True)

# ═══ Save ═══
ids = df_5k["Sequence_ID"].tolist() + six
df_out = pd.DataFrame({"id": ids, "score_tier": tier_labels, "sub_cluster": sub_labels})
for i, nm in enumerate(six):
    df_out.loc[tgt_start + i, "is_target"] = nm
df_out.to_csv(OUT / "seq_to_cluster_scoretier.csv", index=False)
print(f"✅ {OUT / 'seq_to_cluster_scoretier.csv'}", flush=True)

# ═══ UMAP plot ═══
import umap
um = umap.UMAP(n_neighbors=25, min_dist=0.08, metric="cosine", random_state=42)
xy = um.fit_transform(emb)

pal = ["#2563eb","#ea580c","#16a34a","#7c3aed","#d946ef","#0d9488"]
names = ["AcAP","TcAP","EaAP","KoAP","MnAP","MsAP"]
tgt_idx = list(range(n-6, n))

fig, axes = plt.subplots(1, 2, figsize=(28, 12))
fig.suptitle("Two-Stage Clustering: Score Tier → ESMC Sub-clusters", fontsize=18, fontweight="bold")

# Left: Score tiers
ax = axes[0]
tier_colors = ["#e74c3c", "#f39c12", "#2ecc71"]
tier_names = [f"Low Score (≤{p33:.0f})", f"Mid Score ({p33:.0f}-{p67:.0f})", f"High Score (≥{p67:.0f})"]
for t in range(3):
    m = tier_labels == t
    ax.scatter(xy[m,0], xy[m,1], c=tier_colors[t], s=5, alpha=0.3, label=f"{tier_names[t]} ({m.sum()})", edgecolors="none", rasterized=True)
for ti, nm in zip(tgt_idx, names):
    ax.scatter(xy[ti,0], xy[ti,1], c="red", s=350, marker="*", edgecolors="black", linewidths=2, zorder=10)
ax.set_title("Score Tier (by Pred_kcat_over_Km)")
ax.legend(fontsize=9, markerscale=3); ax.grid(alpha=0.08)

# Right: 6 sub-clusters
ax = axes[1]
for s in range(n_sub):
    m = sub_labels == s
    ax.scatter(xy[m,0], xy[m,1], c=pal[s], s=5, alpha=0.4, label=f"Sub{s} ({m.sum()})", edgecolors="none", rasterized=True)
for ti, nm in zip(tgt_idx, names):
    c = "red"
    ax.scatter(xy[ti,0], xy[ti,1], c=c, s=350, marker="*", edgecolors="black", linewidths=2, zorder=10)
    ax.annotate(f"{nm}(Sub{sub_labels[ti]})", (xy[ti,0], xy[ti,1]),
                xytext=(xy[ti,0]+25, xy[ti,1]+25), fontsize=10, fontweight="bold", color=c,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=c, alpha=0.9),
                arrowprops=dict(arrowstyle="->", color=c, lw=1.2), zorder=11)
ax.set_title(f"6 Sub-clusters (Score Tier × ESMC KMeans)  Sil={sil_final:.4f}")
ax.legend(fontsize=8, markerscale=3); ax.grid(alpha=0.08)

plt.tight_layout()
sp = OUT / "clustering_scoretier.png"
plt.savefig(sp, dpi=250, bbox_inches="tight"); plt.close()
print(f"✅ {sp}", flush=True)

print("\nDONE ✅", flush=True)
