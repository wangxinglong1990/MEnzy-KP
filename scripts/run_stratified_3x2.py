#!/usr/bin/env python3
"""
论文 Table S1 方法复现:
  5000序列按 Pred_kcat/Km 排序 → High(1-1000) / Mid(2001-3000) / Low(4001-5000)
  → 每层 ESMC + HDBSCAN → 每层取2个最大子簇 → 共6条代表序列 → UMAP 散点图

参照: Manuscript(11)(1).docx Section 2.3 / 3.2, Table S1
"""

import os, sys, time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
os.chdir(str(PROJECT))
sys.path.insert(0, str(PROJECT))

from src.features.extractor import _get_protein_encoder
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import hdbscan

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9.5,
    "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.1,
})

OUT = PROJECT / "outputs" / "stratified_3x2"
OUT.mkdir(parents=True, exist_ok=True)
t_start = time.time()

# ═══════════════════════════════════════════════════════
# 1. 加载 + 排序 + 三层各取 1000
# ═══════════════════════════════════════════════════════
print("═" * 55)
print("Step 1: Load & stratify — High(1-1000) / Mid(2001-3000) / Low(4001-5000)")
df = pd.read_csv(PROJECT / "textdocs" / "final_submission_5000_sequences.csv")

# 去重 + 合并 sixdata
txt = open(PROJECT / "textdocs" / "sixdata.text").read()
six = []
for ln in txt.strip().split("\n"):
    if not ln.strip(): continue
    p = ln.split("：",1) if "：" in ln else ln.split(":",1)
    if len(p)==2: six.append({"Sequence_ID":p[0].strip(),"Sequence":p[1].strip()})
df = df.drop_duplicates(subset=["Sequence"]).reset_index(drop=True)
sseq = {e["Sequence"].upper() for e in six}
df = df[~df["Sequence"].str.upper().isin(sseq)].reset_index(drop=True)
df = pd.concat([df, pd.DataFrame(six)], ignore_index=True)
n_total = len(df)

# 按 Pred_kcat_over_Km 降序排名
scores = df["Pred_kcat_over_Km"].values
rank = np.argsort(np.argsort(-np.nan_to_num(scores, nan=-999))) + 1  # 1=best

# 取层: High 1-1000, Mid 2001-3000, Low 4001-5000
high_mask = (rank >= 1) & (rank <= 1000)
mid_mask  = (rank >= 2001) & (rank <= 3000)
low_mask  = (rank >= 4001) & (rank <= 5000)

tier_info = [
    ("High", high_mask, "#E64B35", "o"),
    ("Mid",  mid_mask,  "#4DBBD5", "s"),
    ("Low",  low_mask,  "#00A087", "D"),
]

for name, mask, _, _ in tier_info:
    sc = scores[mask]
    print(f"  {name}: n={mask.sum()}  score=[{np.nanmin(sc):.1f}, {np.nanmax(sc):.1f}]  "
          f"mean={np.nanmean(sc):.1f}")

df["stratum"] = "unselected"
df.loc[high_mask, "stratum"] = "High"
df.loc[mid_mask,  "stratum"] = "Mid"
df.loc[low_mask,  "stratum"] = "Low"

# ═══════════════════════════════════════════════════════
# 2. ESMC 编码 (全量，后续分层切片)
# ═══════════════════════════════════════════════════════
print("\nStep 2: ESMC_300M encoding all 5000 seqs ...")
encoder = _get_protein_encoder()
print(f"  Device: {encoder.device}")
_emb = encoder.encode(df["Sequence"].tolist()).astype(np.float32)
emb = StandardScaler().fit_transform(_emb)
print(f"  Shape: {emb.shape}  |  {time.time()-t_start:.1f}s")

# ═══════════════════════════════════════════════════════
# 3. 每层 HDBSCAN → 取前2大子簇 → 选代表
# ═══════════════════════════════════════════════════════
print("\nStep 3: HDBSCAN per stratum, select top-2 sub-clusters ...")

all_labels = np.full(n_total, -1, dtype=int)
is_rep = np.zeros(n_total, dtype=bool)
sub_id_counter = 0
representatives = []
tier_clusters = {}  # tier_name -> [sub_id, ...]

for tier_name, t_mask, color, marker in tier_info:
    t_idx = np.where(t_mask)[0]
    t_emb = emb[t_mask]

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=10, min_samples=3,
        metric="euclidean", cluster_selection_method="eom",
    )
    t_labels = clusterer.fit_predict(t_emb)
    n_noise = (t_labels == -1).sum()

    # 统计各子簇大小，取最大的2个
    sub_sizes = {}
    for l in set(t_labels):
        if l == -1: continue
        sub_sizes[l] = (t_labels == l).sum()

    top2 = sorted(sub_sizes.items(), key=lambda x: x[1], reverse=True)[:2]
    tier_clusters[tier_name] = []

    print(f"\n  {tier_name} (n={t_mask.sum()}): {len(sub_sizes)} HDBSCAN clusters, "
          f"{n_noise} noise → keeping top-2: {top2}")

    for sub_l, size in top2:
        sub_mask_local = t_labels == sub_l
        sub_global_idx = t_idx[sub_mask_local]
        global_sid = sub_id_counter

        all_labels[sub_global_idx] = global_sid
        tier_clusters[tier_name].append(global_sid)
        sub_id_counter += 1

        # 代表：离质心最近
        sub_emb = emb[sub_global_idx]
        centroid = sub_emb.mean(axis=0)
        dists = np.linalg.norm(sub_emb - centroid, axis=1)
        rep_local = np.argmin(dists)
        rep_global = sub_global_idx[rep_local]
        is_rep[rep_global] = True

        rep_dict = {
            "stratum": tier_name,
            "sub_cluster": global_sid,
            "sub_cluster_size": size,
            "Sequence_ID": df.iloc[rep_global]["Sequence_ID"],
            "Sequence": df.iloc[rep_global]["Sequence"],
            "Pred_kcat_over_Km": scores[rep_global],
            "mean_score_in_cluster": np.nanmean(scores[sub_global_idx]),
            "is_target": rep_global >= n_total - 6,
        }
        representatives.append(rep_dict)

        tgt_flag = ""
        if rep_global >= n_total - 6:
            tgt_names = ["AcAP","TcAP","EaAP","KoAP","MnAP","MsAP"]
            tgt_flag = f" ★{tgt_names[rep_global - (n_total - 6)]}"
        print(f"    Sub-{global_sid}: n={size:4d}  mean_score={rep_dict['mean_score_in_cluster']:.1f}  "
              f"rep={rep_dict['Sequence_ID'][:20]} (score={rep_dict['Pred_kcat_over_Km']:.1f}){tgt_flag}")

df["sub_cluster_3x2"] = all_labels
df["is_representative"] = is_rep

print(f"\n  Total: {sub_id_counter} sub-clusters, {len(representatives)} representatives")

# ═══════════════════════════════════════════════════════
# 4. UMAP
# ═══════════════════════════════════════════════════════
print("\nStep 4: UMAP ...")
import umap
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=26, min_dist=0.3)
X_2d = reducer.fit_transform(emb)
print(f"  Shape: {X_2d.shape}")

# ═══════════════════════════════════════════════════════
# 5. 散点图
# ═══════════════════════════════════════════════════════
print("Step 5: Rendering ...")

TIER_COLORS = {"High": "#E64B35", "Mid": "#4DBBD5", "Low": "#00A087"}
TIER_COLORS_LIGHT = {"High": "#f5c6c0", "Mid": "#c5e0eb", "Low": "#b8e6d8"}

fig, ax = plt.subplots(figsize=(15, 11))

# 背景：未选中的序列 (灰色)
unsel = df["stratum"] == "unselected"
ax.scatter(X_2d[unsel, 0], X_2d[unsel, 1],
           c="#e0e0e0", s=3, alpha=0.28, edgecolors="none", rasterized=True)

# 三层散点
for tier_name, t_mask, color, marker in tier_info:
    # 整层（含noise）
    ax.scatter(X_2d[t_mask, 0], X_2d[t_mask, 1],
               c=TIER_COLORS_LIGHT[tier_name], s=12, alpha=0.45,
               edgecolors="none", rasterized=True)

# 6个子簇 (用深色标记)
sub_cluster_markers = ["o","^","s","P","D","X"]
sub_marker_idx = 0
for tier_name, t_mask, color, _ in tier_info:
    for sid in tier_clusters[tier_name]:
        smask = all_labels == sid
        mk = sub_cluster_markers[sub_marker_idx % 6]
        ax.scatter(X_2d[smask, 0], X_2d[smask, 1],
                   c=color, s=22, alpha=0.75, marker=mk,
                   edgecolors="white", linewidths=0.3, zorder=5)
        sub_marker_idx += 1

# 6条代表序列
for r in representatives:
    idx = df[df["Sequence_ID"] == r["Sequence_ID"]].index[0]
    c = TIER_COLORS[r["stratum"]]
    ax.scatter(X_2d[idx, 0], X_2d[idx, 1],
               c=c, s=300, marker="*",
               edgecolors="black", linewidths=2, zorder=25)
    label = r["Sequence_ID"][:15]
    if r["is_target"]:
        tgt_names = ["AcAP","TcAP","EaAP","KoAP","MnAP","MsAP"]
        label = tgt_names[idx - (n_total - 6)]
    ax.annotate(label,
                (X_2d[idx, 0], X_2d[idx, 1]),
                textcoords="offset points", xytext=(8, 8),
                fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor=c, alpha=0.9, lw=1.2))

# 图例
legend_h = [
    Patch(facecolor=TIER_COLORS["High"], alpha=0.5, label="High stratum (rank 1–1000)"),
    Patch(facecolor=TIER_COLORS["Mid"],  alpha=0.5, label="Mid stratum (rank 2001–3000)"),
    Patch(facecolor=TIER_COLORS["Low"],  alpha=0.5, label="Low stratum (rank 4001–5000)"),
    Line2D([0],[0], marker="*", color="w", markerfacecolor="black",
           markersize=14, label="6 representatives (2 per stratum)"),
]
ax.legend(handles=legend_h, fontsize=9, loc="upper right",
          framealpha=0.92, edgecolor="gray", fancybox=True)

n_sub = sub_id_counter
ax.set_xlabel("UMAP-1", fontsize=12)
ax.set_ylabel("UMAP-2", fontsize=12)
ax.set_title(
    f"ESKin-Guided Stratified Sampling — 3 Strata × 2 Sub-clusters = 6 Representatives\n"
    f"ESMC_300M → HDBSCAN per stratum → Top-2 largest sub-clusters → 6 candidates\n"
    f"High(1–1000) / Mid(2001–3000) / Low(4001–5000)  |  {n_sub} sub-clusters  |  "
    f"Silhouette by stratum",
    fontsize=13, fontweight="bold", loc="left")
ax.grid(True, alpha=0.06)

# ═══════════════════════════════════════════════════════
# 6. 保存
# ═══════════════════════════════════════════════════════
out_png = OUT / "stratified_3x2_clustering.png"
out_pdf = OUT / "stratified_3x2_clustering.pdf"
fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")

# CSV
out_df = df[["Sequence_ID", "Sequence", "Pred_kcat_over_Km", "stratum"]].copy()
out_df["rank"] = rank
out_df["sub_cluster_id"] = all_labels
out_df["is_representative"] = is_rep
out_df["umap_x"] = X_2d[:, 0]
out_df["umap_y"] = X_2d[:, 1]
out_csv = OUT / "stratified_3x2_assignments.csv"
out_df.to_csv(out_csv, index=False)

rep_df = pd.DataFrame(representatives)
rep_csv = OUT / "stratified_3x2_representatives.csv"
rep_df.to_csv(rep_csv, index=False)

t_end = time.time()
print(f"\n✅ PNG: {out_png}")
print(f"✅ PDF: {out_pdf}")
print(f"✅ CSV: {out_csv}")
print(f"✅ Reps: {rep_csv}")
print(f"⏱  Total: {t_end - t_start:.1f}s")
