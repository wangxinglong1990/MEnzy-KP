#!/usr/bin/env python3
"""
聚类分析图：基于 pcc_data_all_models.csv 中各模型在 Km/Kcat 上的预测性能，
将模型分为三类（高/中/低表现），与论文中 High/Mid/Low 三层分级策略对应。

参照文献:
  Manuscript(4).docx — ESKin-guided aminopeptidase discovery workflow
  Supplementary materials(4).docx — Table S2 bias-aware stratified selection
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# 全局样式
# ═══════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})

# 三簇配色 —— 与论文 High/Mid/Low 对应
CLUSTER_COLORS = {0: "#D73027", 1: "#4575B4", 2: "#1A9850"}  # High红 / Mid蓝 / Low绿
CLUSTER_NAMES  = {0: "Tier I (High)", 1: "Tier II (Mid)", 2: "Tier III (Low)"}

FEATURE_DISPLAY = ["PCC", "R²", "−RMSE", "−MAE", "−|Bias|", "Spearman R"]

# ═══════════════════════════════════════════════════════════════
# 1. 数据加载
# ═══════════════════════════════════════════════════════════════
print("═" * 60)
print("Loading pcc_data_all_models.csv ...")
df = pd.read_csv("../figures/pcc_data_all_models.csv")
print(f"  Records: {len(df):,}  |  Models: {list(df['model'].unique())}")

# ═══════════════════════════════════════════════════════════════
# 2. 计算每个 model×target 的性能指标
# ═══════════════════════════════════════════════════════════════
print("\nComputing performance metrics ...")
records = []
for (model, target), grp in df.groupby(["model", "target"]):
    yt, yp = grp["y_true"].values, grp["y_pred"].values
    n = len(grp)
    if n < 10:
        continue
    pcc, _    = stats.pearsonr(yt, yp)
    sr, _     = stats.spearmanr(yt, yp)
    ss_res    = np.sum((yt - yp) ** 2)
    ss_tot    = np.sum((yt - np.mean(yt)) ** 2)
    r2        = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse      = np.sqrt(np.mean((yt - yp) ** 2))
    mae       = np.mean(np.abs(yt - yp))
    bias      = np.mean(yp - yt)
    records.append({"model": model, "target": target, "n": n,
                    "PCC": pcc, "R2": r2, "RMSE": rmse,
                    "MAE": mae, "Bias": bias, "SpearmanR": sr})

mdf = pd.DataFrame(records)
print(mdf.to_string(index=False))

# ═══════════════════════════════════════════════════════════════
# 3. 特征工程 & 标准化
# ═══════════════════════════════════════════════════════════════
feat_cols = ["PCC", "R2", "RMSE", "MAE", "Bias", "SpearmanR"]
X = mdf[feat_cols].copy()
X["RMSE"] = -X["RMSE"]
X["MAE"]  = -X["MAE"]
X["Bias"] = -np.abs(X["Bias"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
labels   = [f"{r.model} | {r.target}" for _, r in mdf.iterrows()]

# ═══════════════════════════════════════════════════════════════
# 4. 层次聚类 (Ward, k=3)
# ═══════════════════════════════════════════════════════════════
linkage_mat = linkage(X_scaled, method="ward")
agg = AgglomerativeClustering(n_clusters=3, metric="euclidean", linkage="ward")
raw_clusters = agg.fit_predict(X_scaled)

# 按平均 PCC 重排：cluster 0 = 最好
order = sorted(range(3), key=lambda c: np.mean(mdf["PCC"].values[raw_clusters == c]),
               reverse=True)
mapping = {old: new for new, old in enumerate(order)}
clusters = np.array([mapping[c] for c in raw_clusters])

sil_score = silhouette_score(X_scaled, clusters)
print(f"\nSilhouette Score (k=3): {sil_score:.4f}")

mdf["cluster"] = clusters
mdf["tier"]    = mdf["cluster"].map(CLUSTER_NAMES)

# ═══════════════════════════════════════════════════════════════
# 5. 排序 & 绘图准备
# ═══════════════════════════════════════════════════════════════
# 簇内按 PCC 降序
order_idx = []
for c in range(3):
    idx = np.where(clusters == c)[0]
    order_idx.extend(idx[np.argsort(-mdf["PCC"].values[idx])])
order_idx = np.array(order_idx)

X_sorted     = X_scaled[order_idx]
labels_sorted = [labels[i] for i in order_idx]
clust_sorted  = clusters[order_idx]

# ═══════════════════════════════════════════════════════════════
# 6. 绘制出版级五面板图
# ═══════════════════════════════════════════════════════════════
print("Generating figure ...")
fig = plt.figure(figsize=(18, 10.5))
gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32,
              height_ratios=[1, 1.05])

# ── Panel A: Z-score 热力图 ──────────────────────────────────
ax_h = fig.add_subplot(gs[0, :2])

im = ax_h.imshow(X_sorted, aspect="auto", cmap="RdYlBu_r",
                 vmin=-2.5, vmax=2.5, interpolation="nearest")
ax_h.set_xticks(range(len(FEATURE_DISPLAY)))
ax_h.set_xticklabels(FEATURE_DISPLAY, rotation=40, ha="right", fontsize=8.5)
ax_h.set_yticks(range(len(labels_sorted)))
ax_h.set_yticklabels(labels_sorted, fontsize=8)

# 左侧簇颜色条
for i, c in enumerate(clust_sorted):
    ax_h.add_patch(plt.Rectangle(
        (-0.85, i - 0.5), 0.35, 1,
        color=CLUSTER_COLORS[c], clip_on=False,
        transform=ax_h.transData, linewidth=0, zorder=10))

# 簇间分隔线
prev = clust_sorted[0]
for i, c in enumerate(clust_sorted):
    if c != prev:
        ax_h.axhline(i - 0.5, color="black", lw=2, linestyle="-")
        prev = c

# 簇标注
for c_val in range(3):
    pos = np.where(clust_sorted == c_val)[0]
    if len(pos) > 0:
        ax_h.text(-1.5, np.mean(pos), CLUSTER_NAMES[c_val],
                  fontsize=9, fontweight="bold", color=CLUSTER_COLORS[c_val],
                  ha="right", va="center", rotation=90, transform=ax_h.transData)

cb = plt.colorbar(im, ax=ax_h, shrink=0.85, pad=0.02)
cb.set_label("Standardized Score (Z)", fontsize=8.5)
ax_h.set_title("A   Performance Z-score Heatmap  —  Models clustered into 3 tiers by Ward hierarchical clustering",
               fontsize=11.5, fontweight="bold", loc="left")

# ── Panel B: 带簇切割的树状图 ────────────────────────────────
ax_d = fig.add_subplot(gs[0, 2])

# 找切割高度 (簇间最大距离的 60%)
dn = dendrogram(linkage_mat, labels=labels, orientation="right",
                color_threshold=0, above_threshold_color="#888888",
                ax=ax_d, leaf_font_size=8)

# 染色叶子标签
yt_labels = ax_d.get_yticklabels()
for tl in yt_labels:
    txt = tl.get_text()
    for i, lbl in enumerate(labels):
        if lbl == txt:
            tl.set_color(CLUSTER_COLORS[clusters[i]])
            break

# 画切割线
cut_height = linkage_mat[-2, 2] * 0.7
ax_d.axvline(cut_height, color="crimson", lw=1.5, ls="--", alpha=0.7)
ax_d.text(cut_height + 0.1, len(labels) - 0.3, f"3-cluster cut\n(d={cut_height:.2f})",
          fontsize=7.5, color="crimson", va="top")

ax_d.set_title("B   Hierarchical Clustering Dendrogram\n       (Ward linkage, Euclidean distance)",
               fontsize=11.5, fontweight="bold", loc="left")
ax_d.set_xlabel("Merge Distance", fontsize=9)

# ── Panel C: PCA 投影 ────────────────────────────────────────
ax_p = fig.add_subplot(gs[1, 0])

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
ev1, ev2 = pca.explained_variance_ratio_ * 100

for c_val in range(3):
    m = clusters == c_val
    ax_p.scatter(X_pca[m, 0], X_pca[m, 1], c=CLUSTER_COLORS[c_val],
                 s=220, edgecolors="white", lw=1, zorder=3, alpha=0.92,
                 label=CLUSTER_NAMES[c_val])
    for i in np.where(m)[0]:
        offset = (8, 8)
        ax_p.annotate(labels[i].replace(" | ", "\n"),
                      (X_pca[i, 0], X_pca[i, 1]),
                      textcoords="offset points", xytext=offset,
                      fontsize=7, alpha=0.85, color=CLUSTER_COLORS[c_val],
                      fontweight="bold")

ax_p.set_xlabel(f"Principal Component 1 ({ev1:.1f}%)")
ax_p.set_ylabel(f"Principal Component 2 ({ev2:.1f}%)")
ax_p.set_title(f"C   PCA of Model Performance Profiles\n       Silhouette = {sil_score:.3f}",
               fontsize=11.5, fontweight="bold", loc="left")
ax_p.legend(fontsize=8, loc="lower left", framealpha=0.9,
            edgecolor="gray", fancybox=True)
ax_p.grid(True, alpha=0.15)

# ── Panel D: PCC & R² 分组条形图 ─────────────────────────────
ax_b = fig.add_subplot(gs[1, 1])

bdf = mdf.sort_values(["cluster", "PCC"], ascending=[True, False])
x  = np.arange(len(bdf))
bw = 0.32

ax_b.bar(x - bw/2, bdf["PCC"], bw,
         color=[CLUSTER_COLORS[c] for c in bdf["cluster"]],
         edgecolor="white", lw=0.4, alpha=0.92, label="Pearson r (PCC)")
ax_b.bar(x + bw/2, bdf["R2"], bw,
         color=[CLUSTER_COLORS[c] for c in bdf["cluster"]],
         edgecolor="black", lw=0.4, alpha=0.45, hatch="///", label="R²")

# 簇背景
prev_c2, start_x = bdf["cluster"].iloc[0], -0.5
for i, c in enumerate(bdf["cluster"]):
    if c != prev_c2:
        ax_b.axvspan(start_x, i - 0.5, color=CLUSTER_COLORS[prev_c2],
                     alpha=0.06, zorder=0)
        start_x = i - 0.5
        prev_c2 = c
ax_b.axvspan(start_x, len(bdf) - 0.5, color=CLUSTER_COLORS[prev_c2],
             alpha=0.06, zorder=0)

ax_b.set_xticks(x)
ax_b.set_xticklabels([f"{r.model}\n({r.target})" for _, r in bdf.iterrows()],
                     fontsize=7.2, rotation=40, ha="right")
ax_b.set_ylabel("Correlation / Explained Variance", fontsize=9)
ax_b.set_title("D   Model Correlation Metrics (PCC & R²)",
               fontsize=11.5, fontweight="bold", loc="left")
ax_b.legend(fontsize=8, loc="lower left", ncol=2)
ax_b.set_ylim(-0.15, 1.05)
ax_b.axhline(y=0, color="black", lw=0.5)
ax_b.grid(axis="y", alpha=0.2)

# ── Panel E: RMSE & MAE 分组条形图 ───────────────────────────
ax_e = fig.add_subplot(gs[1, 2])

ax_e.bar(x - bw/2, bdf["RMSE"], bw,
         color=[CLUSTER_COLORS[c] for c in bdf["cluster"]],
         edgecolor="white", lw=0.4, alpha=0.92, label="RMSE")
ax_e.bar(x + bw/2, bdf["MAE"], bw,
         color=[CLUSTER_COLORS[c] for c in bdf["cluster"]],
         edgecolor="black", lw=0.4, alpha=0.45, hatch="///", label="MAE")

# 簇背景
prev_c3, start_x2 = bdf["cluster"].iloc[0], -0.5
for i, c in enumerate(bdf["cluster"]):
    if c != prev_c3:
        ax_e.axvspan(start_x2, i - 0.5, color=CLUSTER_COLORS[prev_c3],
                     alpha=0.06, zorder=0)
        start_x2 = i - 0.5
        prev_c3 = c
ax_e.axvspan(start_x2, len(bdf) - 0.5, color=CLUSTER_COLORS[prev_c3],
             alpha=0.06, zorder=0)

ax_e.set_xticks(x)
ax_e.set_xticklabels([f"{r.model}\n({r.target})" for _, r in bdf.iterrows()],
                     fontsize=7.2, rotation=40, ha="right")
ax_e.set_ylabel("Error (log₁₀ scale)", fontsize=9)
ax_e.set_title("E   Model Error Metrics (RMSE & MAE)",
               fontsize=11.5, fontweight="bold", loc="left")
ax_e.legend(fontsize=8, loc="upper left", ncol=2)
ax_e.grid(axis="y", alpha=0.2)

# ═══════════════════════════════════════════════════════════════
# 总标题
# ═══════════════════════════════════════════════════════════════
fig.suptitle(
    "ESKin Branch-Model Clustering — Three Performance Tiers\n"
    "Ward Hierarchical Clustering on PCC, R², RMSE, MAE, Bias & Spearman R for Km / Kcat Prediction",
    fontsize=14.5, fontweight="bold", y=1.012,
)

# ═══════════════════════════════════════════════════════════════
# 保存
# ═══════════════════════════════════════════════════════════════
out = "../figures/model_clustering_3tiers.png"
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print(f"\n✅ Saved: {out}")

# ═══════════════════════════════════════════════════════════════
# 7. 终端摘要
# ═══════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("CLUSTERING RESULT — Three Performance Tiers")
print("═" * 70)
for c_val in range(3):
    sub = mdf[mdf["cluster"] == c_val]
    print(f"\n  {CLUSTER_NAMES[c_val]}:")
    for _, r in sub.iterrows():
        print(f"    {r.model:14s} | {r.target:5s}  "
              f"PCC={r.PCC:.3f}  R²={r.R2:.3f}  "
              f"RMSE={r.RMSE:.3f}  MAE={r.MAE:.3f}  Bias={r.Bias:+.3f}")
    print(f"    {'─'*55}")
    print(f"    Mean: PCC={sub['PCC'].mean():.3f}  "
          f"R²={sub['R2'].mean():.3f}  RMSE={sub['RMSE'].mean():.3f}")

mdf.to_csv("../figures/model_clustering_metrics.csv", index=False)
print(f"\n📊 Metrics CSV: ../figures/model_clustering_metrics.csv")
