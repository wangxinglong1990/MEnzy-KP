#!/usr/bin/env python3
"""
复现论文完整流程:
  Pred_kcat/Km 排序 → High/Mid/Low 三层 → 每层 HDBSCAN → 选代表序列 → UMAP 可视化

参照: Manuscript(4).docx Section 2.3 & 3.2
      Supplementary materials(4).docx Table S2
"""

import os, sys, time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
os.chdir(str(PROJECT))
sys.path.insert(0, str(PROJECT))

from src.features.extractor import _get_protein_encoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10, "axes.titlesize": 13, "axes.labelsize": 11,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 9,
    "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.1,
})

OUT = PROJECT / "outputs" / "paper_workflow"
OUT.mkdir(parents=True, exist_ok=True)

# ═════════════════════════════════════════════════════════════
# 1. 加载数据
# ═════════════════════════════════════════════════════════════
print("═" * 60)
print("Step 1: Loading 5000 sequences ...", flush=True)
t_start = time.time()

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
df = pd.concat([df, pd.DataFrame(six)], ignore_index=True)

n = len(df)
ids  = df["Sequence_ID"].tolist()
seqs = df["Sequence"].tolist()
tgt_idx = list(range(n - 6, n))
target_names = ["AcAP", "TcAP", "EaAP", "KoAP", "MnAP", "MsAP"]
print(f"  {n} sequences ({n - 6} unique + 6 targets)", flush=True)

# ═════════════════════════════════════════════════════════════
# 2. ESMC 编码
# ═════════════════════════════════════════════════════════════
print("\nStep 2: ESMC_300M encoding ...", flush=True)
encoder = _get_protein_encoder()
print(f"  Device: {encoder.device}", flush=True)
t0 = time.time()
_emb = encoder.encode(seqs).astype(np.float32)
embeddings = StandardScaler().fit_transform(_emb)
print(f"  Shape: {embeddings.shape}  |  {time.time()-t0:.1f}s", flush=True)

# ═════════════════════════════════════════════════════════════
# 3. 按 kcat/Km 分层 (论文 Table S2)
# ═════════════════════════════════════════════════════════════
print("\nStep 3: Stratifying by Pred_kcat_over_Km ...", flush=True)

scores = df["Pred_kcat_over_Km"].values
# 按分数降序排名
rank = np.argsort(np.argsort(-scores)) + 1  # 1 = 最高分

# 论文分层: High 1-1000, Mid 1001-4000, Low 4001-5000
# 但我们有 5006 条，按比例
n_total = len(df)
high_n = int(n_total * 0.20)   # ~1001
mid_n   = int(n_total * 0.60)   # ~3003
low_n   = n_total - high_n - mid_n  # ~1002

tiers = np.full(n_total, "", dtype=object)
tiers[rank <= high_n] = "High"
tiers[(rank > high_n) & (rank <= high_n + mid_n)] = "Mid"
tiers[rank > high_n + mid_n] = "Low"

df["tier"] = tiers
df["rank"] = rank

for tier_name in ["High", "Mid", "Low"]:
    mask = tiers == tier_name
    tgt_in = [target_names[i-n+6] for i in np.where(mask)[0] if i in tgt_idx]
    print(f"  {tier_name}: n={mask.sum()}  "
          f"score_range=[{scores[mask].min():.1f}, {scores[mask].max():.1f}]  "
          f"targets={tgt_in}")

# ═════════════════════════════════════════════════════════════
# 4. 每层 HDBSCAN 聚类 + 选代表序列
# ═════════════════════════════════════════════════════════════
print("\nStep 4: HDBSCAN within each tier ...", flush=True)

try:
    import hdbscan
except ImportError:
    print("  Installing hdbscan ...")
    os.system("pip install hdbscan -q")
    import hdbscan

TIER_COLORS = {"High": "#E64B35", "Mid": "#4DBBD5", "Low": "#00A087"}
all_sub_labels = np.full(n_total, -1, dtype=int)  # -1 = noise
rep_flags = np.zeros(n_total, dtype=bool)
sub_cluster_counter = 0
tier_sub_map = {}  # tier -> list of global sub_cluster IDs
representatives = []

# HDBSCAN 参数 — 按层大小调整
hdbscan_params = {
    "High": {"min_cluster_size": 15, "min_samples": 3},
    "Mid":  {"min_cluster_size": 30, "min_samples": 5},
    "Low":  {"min_cluster_size": 15, "min_samples": 3},
}

for tier_name in ["High", "Mid", "Low"]:
    t_mask = tiers == tier_name
    t_indices = np.where(t_mask)[0]
    t_emb = embeddings[t_mask]

    params = hdbscan_params[tier_name]
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=params["min_cluster_size"],
        min_samples=params["min_samples"],
        metric="euclidean",
        cluster_selection_method="eom",
        core_dist_n_jobs=-1,
    )
    t_labels = clusterer.fit_predict(t_emb)

    n_clusters = len(set(t_labels)) - (1 if -1 in t_labels else 0)
    n_noise = (t_labels == -1).sum()
    print(f"\n  {tier_name} tier (n={t_mask.sum()}): "
          f"{n_clusters} sub-clusters, {n_noise} noise points")

    tier_sub_map[tier_name] = []

    for sub_label in sorted(set(t_labels)):
        if sub_label == -1:
            continue
        sub_mask_local = t_labels == sub_label
        sub_indices = t_indices[sub_mask_local]

        global_sub_id = sub_cluster_counter
        all_sub_labels[sub_indices] = global_sub_id
        tier_sub_map[tier_name].append(global_sub_id)

        # 找代表序列：离质心最近
        sub_emb = embeddings[sub_indices]
        centroid = sub_emb.mean(axis=0)
        dists = np.linalg.norm(sub_emb - centroid, axis=1)
        rep_local_idx = np.argmin(dists)
        rep_global_idx = sub_indices[rep_local_idx]
        rep_flags[rep_global_idx] = True

        representatives.append({
            "tier": tier_name,
            "sub_cluster": global_sub_id,
            "size": len(sub_indices),
            "rep_seq_id": ids[rep_global_idx],
            "rep_sequence": seqs[rep_global_idx],
            "rep_score": scores[rep_global_idx],
            "mean_score": scores[sub_indices].mean(),
            "is_target": rep_global_idx in tgt_idx,
            "target_name": target_names[tgt_idx.index(rep_global_idx)]
                           if rep_global_idx in tgt_idx else "",
        })
        print(f"    sub-cluster {global_sub_id}: n={len(sub_indices)}, "
              f"mean_score={scores[sub_indices].mean():.1f}, "
              f"rep={ids[rep_global_idx]}(score={scores[rep_global_idx]:.1f})",
              end="")
        if rep_global_idx in tgt_idx:
            print(f" ★TARGET: {target_names[tgt_idx.index(rep_global_idx)]}", end="")
        print()

        sub_cluster_counter += 1

df["sub_cluster"] = all_sub_labels
df["is_representative"] = rep_flags

print(f"\n  Total: {sub_cluster_counter} sub-clusters across 3 tiers")
print(f"  Representatives selected: {rep_flags.sum()}")

# ═════════════════════════════════════════════════════════════
# 5. UMAP 降维
# ═════════════════════════════════════════════════════════════
print("\nStep 5: UMAP dimensionality reduction ...", flush=True)
import umap
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3)
X_2d = reducer.fit_transform(embeddings)
print(f"  UMAP shape: {X_2d.shape}", flush=True)

# ═════════════════════════════════════════════════════════════
# 6. 主图：三层分层 + 子簇 HDBSCAN 散点图
# ═════════════════════════════════════════════════════════════
print("Step 6: Rendering figure ...", flush=True)

fig, axes = plt.subplots(1, 2, figsize=(22, 10),
                         gridspec_kw={"width_ratios": [2.2, 1]})

ax = axes[0]

# ── 绘制每个子簇 ──
# 为每个子簇分配颜色（基于所属 tier 的颜色系）
import matplotlib.cm as cm
np.random.seed(42)
for tier_name in ["High", "Mid", "Low"]:
    base_color = TIER_COLORS[tier_name]
    sub_ids = tier_sub_map.get(tier_name, [])
    for i, sid in enumerate(sub_ids):
        mask = all_sub_labels == sid
        # 在 tier 基本色基础上微调
        if hasattr(cm, 'lighten'):
            color = base_color
        else:
            color = base_color
        alpha_val = 0.55 if tier_name == "Mid" else 0.60
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   c=color, s=8, alpha=alpha_val,
                   edgecolors="none", rasterized=True)

# ── Noise 点（灰色、小点） ──
for tier_name in ["High", "Mid", "Low"]:
    t_mask = (tiers == tier_name) & (all_sub_labels == -1)
    gray = "#cccccc" if tier_name == "Mid" else "#dddddd"
    ax.scatter(X_2d[t_mask, 0], X_2d[t_mask, 1],
               c=gray, s=3, alpha=0.25, edgecolors="none", rasterized=True)

# ── 代表序列 (大号 marker) ──
for tier_name, marker in [("High", "o"), ("Mid", "s"), ("Low", "D")]:
    mask = rep_flags & (tiers == tier_name)
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               c=TIER_COLORS[tier_name], s=90, marker=marker,
               edgecolors="black", linewidths=1.2, zorder=15,
               label=f"{tier_name} tier rep. (n={mask.sum()})")

# ── 6 条目标序列 (星号高亮) ──
for ti, name in zip(tgt_idx, target_names):
    c = TIER_COLORS.get(tiers[ti], "black")
    ax.scatter(X_2d[ti, 0], X_2d[ti, 1],
               c=c, s=300, marker="*",
               edgecolors="black", linewidths=2, zorder=30)
    ax.annotate(name, (X_2d[ti, 0], X_2d[ti, 1]),
                textcoords="offset points", xytext=(10, 8),
                fontsize=10, fontweight="bold", color="black",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor=c, alpha=0.9, lw=1.5))

# ── 图例 ──
legend_handles = [
    Patch(facecolor=TIER_COLORS["High"], alpha=0.6,
          label=f"High kcat/Km stratum (n={(tiers=='High').sum()})"),
    Patch(facecolor=TIER_COLORS["Mid"], alpha=0.6,
          label=f"Mid kcat/Km stratum (n={(tiers=='Mid').sum()})"),
    Patch(facecolor=TIER_COLORS["Low"], alpha=0.6,
          label=f"Low kcat/Km stratum (n={(tiers=='Low').sum()})"),
    Line2D([0], [0], marker="*", color="w", markerfacecolor="black",
           markersize=14,
           label="6 target aminopeptidases"),
]
ax.legend(handles=legend_handles, fontsize=8.5, loc="upper right",
          framealpha=0.92, edgecolor="gray", fancybox=True)

ax.set_xlabel("UMAP-1", fontsize=12)
ax.set_ylabel("UMAP-2", fontsize=12)
ax.set_title(
    f"ESKin-Guided Sequence-Space Analysis (Paper Workflow)\n"
    f"ESMC_300M → Score Stratification (High/Mid/Low) → HDBSCAN within each tier\n"
    f"{sub_cluster_counter} sub-clusters  |  {rep_flags.sum()} representatives  |  "
    f"{(all_sub_labels==-1).sum()} noise points",
    fontsize=13, fontweight="bold", loc="left")
ax.grid(True, alpha=0.06)

# ── 右侧面板：每层子簇统计 ──
ax2 = axes[1]
ax2.axis("off")

y_pos = 0.97
ax2.text(0.02, y_pos, "Sub-Cluster Summary", fontsize=12, fontweight="bold",
         transform=ax2.transAxes, va="top")
y_pos -= 0.04

for tier_name in ["High", "Mid", "Low"]:
    sub_ids = tier_sub_map.get(tier_name, [])
    n_sub = len(sub_ids)
    n_tier = (tiers == tier_name).sum()
    n_noise_tier = ((tiers == tier_name) & (all_sub_labels == -1)).sum()

    y_pos -= 0.03
    ax2.text(0.02, y_pos,
             f"▍{tier_name} stratum  (n={n_tier}, {n_sub} clusters, {n_noise_tier} noise)",
             fontsize=10, fontweight="bold", color=TIER_COLORS[tier_name],
             transform=ax2.transAxes, va="top")
    y_pos -= 0.025

    for sid in sub_ids:
        mask = all_sub_labels == sid
        size = mask.sum()
        mean_s = scores[mask].mean()
        # 找这个子簇的代表
        rep_info = [r for r in representatives if r["sub_cluster"] == sid][0]
        rep_id_short = rep_info["rep_seq_id"][:15]
        is_tgt = " ★" if rep_info["is_target"] else ""
        ax2.text(0.05, y_pos,
                 f"  Sub-{sid}: n={size:4d}  score={mean_s:.1f}  rep={rep_id_short}{is_tgt}",
                 fontsize=7.5, color="#444444", transform=ax2.transAxes, va="top",
                 family="monospace")
        y_pos -= 0.018

    y_pos -= 0.005

ax2.set_ylim(0, 1.05)
ax2.set_xlim(0, 1)

# ═════════════════════════════════════════════════════════════
# 7. 保存
# ═════════════════════════════════════════════════════════════
out_png = OUT / "paper_workflow_clustering.png"
out_pdf = OUT / "paper_workflow_clustering.pdf"
fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
t_end = time.time()
print(f"\n✅ PNG: {out_png}")
print(f"✅ PDF: {out_pdf}")
print(f"⏱  Total: {t_end - t_start:.1f}s")

# 保存结果表
rep_df = pd.DataFrame(representatives)
rep_df.to_csv(OUT / "representatives.csv", index=False)

assign_df = df[["Sequence_ID", "Sequence", "Pred_kcat_over_Km", "tier",
                 "rank"]].copy()
assign_df["sub_cluster"] = all_sub_labels
assign_df["is_representative"] = rep_flags
assign_df["umap_x"] = X_2d[:, 0]
assign_df["umap_y"] = X_2d[:, 1]
assign_df.to_csv(OUT / "full_assignments.csv", index=False)

print(f"📊 Representatives: {OUT / 'representatives.csv'}")
print(f"📊 Full assignments: {OUT / 'full_assignments.csv'}")
