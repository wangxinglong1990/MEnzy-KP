#!/usr/bin/env python3
"""重新渲染散点图：只保留浅色层背景 + 星号代表，去掉灰点和子簇形状"""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "outputs" / "stratified_3x2"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9.5,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.1,
})

# Load existing assignments
df = pd.read_csv(OUT / "stratified_3x2_assignments.csv")
X_2d = df[["umap_x", "umap_y"]].values
print(f"Loaded {len(df)} rows")

TIER_COLORS = {"High": "#E64B35", "Mid": "#4DBBD5", "Low": "#00A087"}
TIER_LIGHT   = {"High": "#f5c6c0", "Mid": "#c5e0eb", "Low": "#b8e6d8"}

fig, ax = plt.subplots(figsize=(15, 11))

# 三层浅色背景大圆点
for tier_name in ["High", "Mid", "Low"]:
    mask = df["stratum"] == tier_name
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               c=TIER_LIGHT[tier_name], s=14, alpha=0.50,
               edgecolors="none", rasterized=True)

# 6条target代表 ★
rep = df[df["is_representative"] == True]
tgt_labels = {"1.AcAP":"AcAP","2.TcAP":"TcAP","3.EaAP":"EaAP","4.KoAP":"KoAP","5.MnAP":"MnAP","6.MsAP":"MsAP"}
for _, r in rep.iterrows():
    tier = r["stratum"]
    c = TIER_COLORS.get(tier, "black")
    ax.scatter(r["umap_x"], r["umap_y"],
               c=c, s=350, marker="*",
               edgecolors="black", linewidths=2, zorder=30)
    label = tgt_labels.get(r["Sequence_ID"], r["Sequence_ID"][:15])
    ax.annotate(label, (r["umap_x"], r["umap_y"]),
                textcoords="offset points", xytext=(10, 10),
                fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor=c, alpha=0.9, lw=1.5))

# Legend
legend_h = [
    Patch(facecolor=TIER_LIGHT["High"], edgecolor=TIER_COLORS["High"], linewidth=0.8,
          label="High kcat/Km (rank 1–1000)"),
    Patch(facecolor=TIER_LIGHT["Mid"],  edgecolor=TIER_COLORS["Mid"], linewidth=0.8,
          label="Mid kcat/Km (rank 2001–3000)"),
    Patch(facecolor=TIER_LIGHT["Low"],  edgecolor=TIER_COLORS["Low"], linewidth=0.8,
          label="Low kcat/Km (rank 4001–5000)"),
    Line2D([0],[0], marker="*", color="w", markerfacecolor="black",
           markersize=14, label="6 Target aminopeptidases\n(AcAP,TcAP,EaAP,KoAP,MnAP,MsAP)"),
]
ax.legend(handles=legend_h, fontsize=9.5, loc="upper right",
          framealpha=0.92, edgecolor="gray", fancybox=True)

ax.set_xlabel("UMAP-1", fontsize=12)
ax.set_ylabel("UMAP-2", fontsize=12)
ax.set_title(
    "ESKin-Guided Stratified Sampling — 3 Strata, 6 Target Representatives\n"
    "ESMC_300M → Pred_kcat/Km ranking → High/Mid/Low each 1000 → HDBSCAN per stratum",
    fontsize=13, fontweight="bold", loc="left")
ax.grid(True, alpha=0.06)

out_png = OUT / "stratified_3x2_clean.png"
out_pdf = OUT / "stratified_3x2_clean.pdf"
fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
