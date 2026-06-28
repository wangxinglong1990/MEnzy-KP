#!/usr/bin/env python3
"""
严格按照 Manuscript(11)(1).docx Section 3.2:
  5000序列 ESMC编码 → 按Pred_kcat/Km排名取3000
  High(1-1000) / Mid(2001-3000) / Low(4001-5000)
  每层HDBSCAN → UMAP按层着色 → 6论文候选星标
"""
import os, sys, time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
os.chdir(str(PROJECT))
sys.path.insert(0, str(PROJECT))

from src.features.extractor import _get_protein_encoder
from sklearn.preprocessing import StandardScaler
import numpy as np, pandas as pd
import hdbscan
import umap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans"],
    "font.size":10,"axes.titlesize":13,"axes.labelsize":12,"legend.fontsize":9,
    "figure.dpi":300,"savefig.dpi":300,"savefig.bbox":"tight","savefig.pad_inches":0.1,
})

OUT = PROJECT / "outputs" / "paper_method"
OUT.mkdir(parents=True, exist_ok=True)
t_start = time.time()

STRATUM_COLORS = {"High": "#D73027", "Mid": "#2166AC", "Low": "#1A9850"}
PAPER_CANDIDATES = {
    "HJQ26429.1":"BcAP", "MHD4772611.1":"LyAP",
    "TLY70986.1":"GpAP", "RDA92420.1":"OcAP",
    "OOF96827.1":"AcAP", "OAX79132.1":"EaAP",
}

# ═══════════════════════════════════════════════════
# 1. 加载 + 排名 + 取3000
# ═══════════════════════════════════════════════════
print("=" * 55)
print("Step 1: Load 5000, rank by kcat/Km, select 3000")
df = pd.read_csv(PROJECT / "textdocs" / "merged_5000_with_targets.csv")
scores = df["Pred_kcat_over_Km"].values
rank = np.argsort(np.argsort(-np.nan_to_num(scores, nan=-999))) + 1

h = (rank >= 1)    & (rank <= 1000)
m = (rank >= 2001) & (rank <= 3000)
l = (rank >= 4001) & (rank <= 5000)

df["stratum"] = "unselected"
df.loc[h, "stratum"] = "High"
df.loc[m, "stratum"] = "Mid"
df.loc[l, "stratum"] = "Low"
df["rank"] = rank

selected = df[h | m | l].copy()
selected_idx = selected.index.tolist()
print(f"  Total: {len(df)} → Selected: {len(selected)} "
      f"(High={h.sum()} Mid={m.sum()} Low={l.sum()})")

# ═══════════════════════════════════════════════════
# 2. ESMC 编码 (全量5000，保持UMAP一致性)
# ═══════════════════════════════════════════════════
print("\nStep 2: ESMC_300M encoding all 5000 ...")
encoder = _get_protein_encoder()
print(f"  Device: {encoder.device}")
emb_all = encoder.encode(df["Sequence"].tolist()).astype(np.float32)
emb_all = StandardScaler().fit_transform(emb_all)
print(f"  Shape: {emb_all.shape}  |  {time.time()-t_start:.0f}s")

# ═══════════════════════════════════════════════════
# 3. 每层 HDBSCAN
# ═══════════════════════════════════════════════════
print("\nStep 3: HDBSCAN per stratum ...")
df["sub_cluster"] = -1
sub_id = 0

for sn in ["High", "Mid", "Low"]:
    mask = df["stratum"] == sn
    idx = np.where(mask)[0]
    e = emb_all[idx]

    cl = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=3,
                         metric="euclidean", cluster_selection_method="eom")
    labels = cl.fit_predict(e)
    n_cl = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"  {sn} (n={len(idx)}): {n_cl} clusters, {n_noise} noise")

    # Map local labels to global sub_cluster IDs (skip noise=-1)
    for lbl in set(labels):
        if lbl == -1: continue
        df.loc[idx[labels == lbl], "sub_cluster"] = sub_id
        sub_id += 1

print(f"  Total sub-clusters: {sub_id}")

# ═══════════════════════════════════════════════════
# 4. UMAP (全量投影)
# ═══════════════════════════════════════════════════
print("\nStep 4: UMAP projection ...")
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=26, min_dist=0.3)
X_2d = reducer.fit_transform(emb_all)
print(f"  Shape: {X_2d.shape}")

# ═══════════════════════════════════════════════════
# 5. 散点图: 只画3000条选中序列，按层着色
# ═══════════════════════════════════════════════════
print("Step 5: Rendering ...")

fig, ax = plt.subplots(figsize=(15, 11))

# 背景：未选中2000条 (灰色淡点)
unsel_mask = df["stratum"] == "unselected"
ax.scatter(X_2d[unsel_mask, 0], X_2d[unsel_mask, 1],
           c="#e8e8e8", s=4, alpha=0.30, edgecolors="none", rasterized=True,
           label="Unselected (ranks 1001-2000, 3001-4000)")

# 三层着色
rank_info_map = {"High":"1-1000","Mid":"2001-3000","Low":"4001-5000"}
for sn in ["High","Mid","Low"]:
    mask = df["stratum"] == sn
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               c=STRATUM_COLORS[sn], s=9, alpha=0.55,
               edgecolors="none", rasterized=True,
               label=f"{sn} stratum (rank {rank_info_map[sn]}, n={mask.sum()})")

# 6论文候选 ★
for pid, pname in PAPER_CANDIDATES.items():
    m = df[df["Sequence_ID"] == pid]
    if len(m) > 0:
        r = m.iloc[0]
        c = STRATUM_COLORS.get(r["stratum"], "black")
        ax.scatter(r["umap_x"] if "umap_x" in df.columns else X_2d[m.index[0], 0],
                   r["umap_y"] if "umap_y" in df.columns else X_2d[m.index[0], 1],
                   c=c, s=350, marker="*",
                   edgecolors="black", linewidths=2, zorder=30)
        ax.annotate(pname,
                    (X_2d[m.index[0], 0], X_2d[m.index[0], 1]),
                    textcoords="offset points", xytext=(9, 9),
                    fontsize=11, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor=c, alpha=0.92, lw=1.5))
        sc = r["Pred_kcat_over_Km"]
        rk = rank[m.index[0]]
        sc_stratum = r["stratum"]
        print(f"  {pname}: {pid}  score={sc:.1f}  rank={rk}  stratum={sc_stratum}  sub_cluster={int(r['sub_cluster'])}")

legend_h = [
    Patch(facecolor="#e8e8e8", alpha=0.3, label=f"Unselected ranks 1001-2000, 3001-4000 (n={unsel_mask.sum()})"),
    Patch(facecolor=STRATUM_COLORS["High"], alpha=0.55, label=f"High kcat/Km (rank 1-1000, n={h.sum()})"),
    Patch(facecolor=STRATUM_COLORS["Mid"], alpha=0.55, label=f"Mid kcat/Km (rank 2001-3000, n={m.sum()})"),
    Patch(facecolor=STRATUM_COLORS["Low"], alpha=0.55, label=f"Low kcat/Km (rank 4001-5000, n={l.sum()})"),
    Line2D([0],[0], marker="*", color="w", markerfacecolor="black", markersize=14,
           label="6 candidates (BcAP,LyAP|GpAP,OcAP|AcAP,EaAP)"),
]
ax.legend(handles=legend_h, fontsize=8.5, loc="upper right",
          framealpha=0.92, edgecolor="gray", fancybox=True)
ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
ax.set_title(
    "ESKin-Guided Stratified Sampling (per Section 3.2)\n"
    "5,000 homologs → kcat/Km ranking → High(1-1000)/Mid(2001-3000)/Low(4001-5000)\n"
    "HDBSCAN per stratum | Stars: 6 candidates (BcAP,LyAP|GpAP,OcAP|AcAP,EaAP)",
    fontsize=13, fontweight="bold", loc="left")
ax.grid(True, alpha=0.06)

# ═══════════════════════════════════════════════════
# 6. 保存
# ═══════════════════════════════════════════════════
out_png = OUT / "paper_method_figure.png"
out_pdf = OUT / "paper_method_figure.pdf"
fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")

# CSV: 只保留3000条选中序列
out_csv = OUT / "paper_method_assignments.csv"
df_out = df[df["stratum"] != "unselected"][
    ["Sequence_ID","Sequence","Pred_kcat_over_Km","stratum","rank","sub_cluster"]
].copy()
df_out["umap_x"] = X_2d[df["stratum"] != "unselected", 0]
df_out["umap_y"] = X_2d[df["stratum"] != "unselected", 1]
df_out["is_candidate"] = df_out["Sequence_ID"].isin(PAPER_CANDIDATES.keys())
df_out.to_csv(out_csv, index=False)

t_end = time.time()
print(f"\n✅ PNG: {out_png}")
print(f"✅ PDF: {out_pdf}")
print(f"✅ CSV: {out_csv}")
print(f"⏱  Total: {t_end - t_start:.0f}s")
