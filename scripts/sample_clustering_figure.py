#!/usr/bin/env python3
"""
样本级聚类图：对 Km 预测数据中被 4 个模型（baseline/condition/msa1d/msa2d）
共同覆盖的 851 个样本进行三分类聚类，生成 UMAP + PCA 可视化。

参照: Manuscript(4).docx — HDBSCAN sequence-space clustering
      Supplementary materials(4).docx — Table S2 stratified selection
      pcc_data_all_models.csv — 多模型预测数据
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings("ignore")

# 尝试导入 umap
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("⚠️  umap not installed, will use PCA only")

# ═══════════════════════════════════════════════════════════
# 全局样式
# ═══════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

CLUSTER_COLORS = ["#E64B35", "#4DBBD5", "#00A087"]  # C1红 / C2蓝 / C3绿
CLUSTER_NAMES  = {
    0: "Cluster A (High agreement, High Km)",
    1: "Cluster B (Moderate agreement, Mid Km)",
    2: "Cluster C (High agreement, Low Km)",
}
N_CLUSTERS = 3

# ═══════════════════════════════════════════════════════════
# 1. 构建样本特征矩阵
# ═══════════════════════════════════════════════════════════
print("═" * 60)
print("1. Building sample-level feature matrix ...")
import os
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(BASE, "figures", "pcc_data_all_models.csv"))
km_df = df[df["target"] == "km"]
pivot = km_df.pivot_table(index="sample_id", columns="model", values="y_pred")
pivot["y_true"] = km_df.groupby("sample_id")["y_true"].first()

# 取 4 个模型都覆盖的样本
models_used = ["baseline", "condition", "msa1d", "msa2d"]
has_all = pivot[models_used].notna().all(axis=1)
data = pivot[has_all].copy()
print(f"   Samples with all 4 models: {len(data)}")

# 特征：各模型预测值 + 残差 + y_true
for m in models_used:
    data[f"resid_{m}"] = data[m] - data["y_true"]

feat_cols = (models_used
             + [f"resid_{m}" for m in models_used]
             + ["y_true"])
X = data[feat_cols].values
labels_sample = data.index.tolist()

print(f"   Feature dims: {X.shape[1]} ({', '.join(feat_cols)})")

# ═══════════════════════════════════════════════════════════
# 2. 标准化 & 聚类
# ═══════════════════════════════════════════════════════════
print("\n2. Clustering samples into 3 groups ...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=30)
clusters_raw = kmeans.fit_predict(X_scaled)

# 按 y_true 均值重排：cluster 0 = 最高 Km -> cluster 2 = 最低 Km
order = sorted(range(N_CLUSTERS),
               key=lambda c: np.mean(data["y_true"].values[clusters_raw == c]),
               reverse=True)
mapping = {old: new for new, old in enumerate(order)}
clusters = np.array([mapping[c] for c in clusters_raw])

# 更新名字以反映实际特征
cluster_names_actual = {}
for c_val in range(N_CLUSTERS):
    mask = clusters == c_val
    mean_km = np.mean(data["y_true"].values[mask])
    mean_resid = np.mean(np.abs(data["resid_baseline"].values[mask]))
    cluster_names_actual[c_val] = (
        f"Cluster {['A','B','C'][c_val]}  "
        f"(logKm={mean_km:+.2f}, |resid|={mean_resid:.2f})"
    )

sil = silhouette_score(X_scaled, clusters)
db  = davies_bouldin_score(X_scaled, clusters)
ch  = calinski_harabasz_score(X_scaled, clusters)
print(f"   Silhouette: {sil:.4f}  |  Davies-Bouldin: {db:.4f}  |  Calinski-Harabasz: {ch:.1f}")

data["cluster"] = clusters
cluster_sizes = {c: (clusters == c).sum() for c in range(N_CLUSTERS)}
for c_val in range(N_CLUSTERS):
    print(f"   {cluster_names_actual[c_val]}: n={cluster_sizes[c_val]}")

# ═══════════════════════════════════════════════════════════
# 3. 降维：UMAP + PCA
# ═══════════════════════════════════════════════════════════
print("\n3. Dimensionality reduction ...")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
ev1, ev2 = pca.explained_variance_ratio_ * 100

if HAS_UMAP:
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3)
    X_umap = reducer.fit_transform(X_scaled)
else:
    X_umap = X_pca  # fallback

# ═══════════════════════════════════════════════════════════
# 4. 绘图
# ═══════════════════════════════════════════════════════════
print("4. Generating multi-panel clustering figure ...")
fig = plt.figure(figsize=(18, 12))
gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

# ── Panel A: UMAP 散点图 ──────────────────────────────────
ax_u = fig.add_subplot(gs[0, :2])
for c_val in range(N_CLUSTERS):
    m = clusters == c_val
    ax_u.scatter(X_umap[m, 0], X_umap[m, 1],
                 c=CLUSTER_COLORS[c_val], s=14, alpha=0.7,
                 edgecolors="none", label=cluster_names_actual[c_val])

# 画簇中心
for c_val in range(N_CLUSTERS):
    m = clusters == c_val
    cx, cy = np.median(X_umap[m, 0]), np.median(X_umap[m, 1])
    ax_u.scatter(cx, cy, c=CLUSTER_COLORS[c_val], s=200,
                 marker="X", edgecolors="black", linewidths=1.2, zorder=10)

method_str = "UMAP" if HAS_UMAP else "PCA"
ax_u.set_xlabel(f"{method_str}-1")
ax_u.set_ylabel(f"{method_str}-2")
ax_u.set_title(f"A   {method_str} Projection — 851 Enzyme-Substrate Samples Clustered into 3 Groups\n"
               f"       Silhouette={sil:.3f}  |  DB-index={db:.3f}  |  CH-index={ch:.0f}",
               fontsize=11.5, fontweight="bold", loc="left")
ax_u.legend(fontsize=8, loc="upper right", framealpha=0.9,
            markerscale=2, edgecolor="gray", fancybox=True)
ax_u.grid(True, alpha=0.1)

# ── Panel B: PCA 散点图 ────────────────────────────────────
ax_p = fig.add_subplot(gs[0, 2])
for c_val in range(N_CLUSTERS):
    m = clusters == c_val
    ax_p.scatter(X_pca[m, 0], X_pca[m, 1],
                 c=CLUSTER_COLORS[c_val], s=14, alpha=0.7,
                 edgecolors="none")

for c_val in range(N_CLUSTERS):
    m = clusters == c_val
    cx, cy = np.median(X_pca[m, 0]), np.median(X_pca[m, 1])
    ax_p.scatter(cx, cy, c=CLUSTER_COLORS[c_val], s=180,
                 marker="X", edgecolors="black", linewidths=1.2, zorder=10)

ax_p.set_xlabel(f"PC1 ({ev1:.1f}%)")
ax_p.set_ylabel(f"PC2 ({ev2:.1f}%)")
ax_p.set_title("B   PCA Projection\n       (same clustering)",
               fontsize=11.5, fontweight="bold", loc="left")
ax_p.grid(True, alpha=0.1)

# ── Panel C: 预测值 vs 真实值（按簇着色）───────────────────
ax_s = fig.add_subplot(gs[1, 0])
for c_val in range(N_CLUSTERS):
    m = clusters == c_val
    ax_s.scatter(data["y_true"].values[m],
                 data["baseline"].values[m],
                 c=CLUSTER_COLORS[c_val], s=10, alpha=0.55,
                 edgecolors="none", label=cluster_names_actual[c_val])

lims = [data["y_true"].min() - 0.3, data["y_true"].max() + 0.3]
ax_s.plot(lims, lims, "k--", lw=1, alpha=0.5, label="y = x")
ax_s.set_xlim(lims)
ax_s.set_ylim(lims)
ax_s.set_xlabel("True log₁₀(Km)")
ax_s.set_ylabel("Predicted log₁₀(Km) [baseline]")
ax_s.set_title("C   Predicted vs True logKm (baseline model)\n       Colored by cluster assignment",
               fontsize=11.5, fontweight="bold", loc="left")
ax_s.legend(fontsize=7.5, loc="upper left", framealpha=0.85, markerscale=2)
ax_s.grid(True, alpha=0.15)

# ── Panel D: 每个簇的预测误差分布 ──────────────────────────
ax_b = fig.add_subplot(gs[1, 1])
box_data = []
box_labels = []
box_colors = []
for c_val in range(N_CLUSTERS):
    m = clusters == c_val
    for mi, model_name in enumerate(models_used):
        resid = data[f"resid_{model_name}"].values[m]
        box_data.append(resid)
        short = "BL" if model_name == "baseline" else ("CD" if model_name == "condition" else ("M1" if model_name == "msa1d" else "M2"))
        box_labels.append(f"{['A','B','C'][c_val]}-{short}")
        box_colors.append(CLUSTER_COLORS[c_val])

bp = ax_b.boxplot(box_data, labels=box_labels, patch_artist=True,
                  showfliers=False, widths=0.7, medianprops={"linewidth": 1.5})
for patch, color in zip(bp["boxes"], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.55)
ax_b.axhline(y=0, color="black", lw=0.8, ls="--")
ax_b.set_ylabel("Residual (pred − true)")
ax_b.set_title("D   Prediction Residuals by Cluster & Model\n       (A=high Km, B=mid Km, C=low Km)",
               fontsize=11.5, fontweight="bold", loc="left")
ax_b.tick_params(axis="x", rotation=45, labelsize=7)
ax_b.grid(axis="y", alpha=0.15)

# 添加model分组标识
for i, mi in enumerate(range(4)):
    ax_b.text(i * 3 + 2, ax_b.get_ylim()[0] * 0.95,
              models_used[mi], fontsize=6.5, ha="center",
              fontstyle="italic", color="gray")

# ── Panel E: 簇特征雷达图 ──────────────────────────────────
ax_r = fig.add_subplot(gs[1, 2], polar=True)

radar_metrics = ["y_true", "baseline", "condition", "msa1d", "msa2d",
                 "|resid_BL|", "|resid_CD|", "|resid_M1|", "|resid_M2|"]
radar_labels = ["logKm (true)", "Pred (BL)", "Pred (CD)", "Pred (M1)", "Pred (M2)",
                "|Err BL|", "|Err CD|", "|Err M1|", "|Err M2|"]

# 计算每个簇的关键指标
radar_stats = {}
for c_val in range(N_CLUSTERS):
    m = clusters == c_val
    radar_stats[c_val] = [
        np.mean(data["y_true"].values[m]),
        np.mean(data["baseline"].values[m]),
        np.mean(data["condition"].values[m]),
        np.mean(data["msa1d"].values[m]),
        np.mean(data["msa2d"].values[m]),
        np.mean(np.abs(data["resid_baseline"].values[m])),
        np.mean(np.abs(data["resid_condition"].values[m])),
        np.mean(np.abs(data["resid_msa1d"].values[m])),
        np.mean(np.abs(data["resid_msa2d"].values[m])),
    ]

# min-max normalize per metric across clusters
radar_arr = np.array([radar_stats[c] for c in range(N_CLUSTERS)])
radar_min = radar_arr.min(axis=0)
radar_max = radar_arr.max(axis=0)
radar_range = radar_max - radar_min
radar_range[radar_range == 0] = 1
radar_norm = (radar_arr - radar_min) / radar_range

n_radar = len(radar_labels)
angles = np.linspace(0, 2 * np.pi, n_radar, endpoint=False).tolist()
angles += angles[:1]

for c_val in range(N_CLUSTERS):
    vals = radar_norm[c_val].tolist()
    vals += vals[:1]
    ax_r.fill(angles, vals, alpha=0.15, color=CLUSTER_COLORS[c_val])
    ax_r.plot(angles, vals, "o-", lw=1.8, color=CLUSTER_COLORS[c_val],
              markersize=5, label=f"Cluster {['A','B','C'][c_val]}")

ax_r.set_xticks(angles[:-1])
ax_r.set_xticklabels(radar_labels, fontsize=7)
ax_r.set_yticklabels([])
ax_r.set_title("E   Cluster Profiles (normalized per metric)",
               fontsize=11.5, fontweight="bold", loc="left", pad=20)
ax_r.legend(fontsize=7.5, loc="upper right", bbox_to_anchor=(1.3, 1.1))

# ═══════════════════════════════════════════════════════════
# 总标题
# ═══════════════════════════════════════════════════════════
fig.suptitle(
    "Sample-Level Clustering of Enzyme-Substrate Pairs — 3 Groups\n"
    "Features: 4-model Km predictions + residuals + true logKm  |  "
    f"851 samples  |  KMeans (k=3)  |  Silhouette={sil:.3f}",
    fontsize=14, fontweight="bold", y=1.01,
)

# ═══════════════════════════════════════════════════════════
# 保存
# ═══════════════════════════════════════════════════════════
out = os.path.join(BASE, "figures", "sample_clustering_3groups.png")
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print(f"\n✅ Saved: {out}")

# ═══════════════════════════════════════════════════════════
# 5. 簇特征摘要
# ═══════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("CLUSTER PROFILES")
print("═" * 70)
for c_val in range(N_CLUSTERS):
    m = clusters == c_val
    sub = data[m]
    print(f"\n  {cluster_names_actual[c_val]}  (n={m.sum()})")
    print(f"    logKm (true):      {sub['y_true'].mean():+.3f} ± {sub['y_true'].std():.3f}")
    for mdl in models_used:
        r = sub[f"resid_{mdl}"]
        print(f"    {mdl:12s}: pred={sub[mdl].mean():+.3f}  "
              f"resid={r.mean():+.3f}  |resid|={np.abs(r).mean():.3f}")

# save clustered data
data.to_csv(os.path.join(BASE, "figures", "sample_clustering_assignments.csv"))
print(f"\n📊 Cluster assignments: ../figures/sample_clustering_assignments.csv")
