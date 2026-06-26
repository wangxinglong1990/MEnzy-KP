"""KMeans 聚类 + 综合离散程度分析"""

import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.metrics.pairwise import pairwise_distances


class ClusterAnalyzer:
    """KMeans 聚类器，支持自动选 K 和全面的离散程度报告。"""

    def __init__(self, n_clusters: int | None = None, random_state: int = 42):
        """
        Parameters
        ----------
        n_clusters : int or None
            簇数；为 None 时自动从 [2, 10] 中选择最佳 K。
        random_state : int
            随机种子。
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model_: KMeans | None = None
        self.labels_: np.ndarray | None = None
        self.chosen_k_: int | None = None
        self.k_selection_log_: list[dict] | None = None  # 自动选K过程记录

    # ── 自动选 K ──────────────────────────────────────────
    def auto_select_k(self, X: np.ndarray, k_range: range | None = None) -> int:
        """综合 silhouette、Davies-Bouldin、Calinski-Harabasz 和肘部法则选最佳 K。

        每个指标独立排名，取总分最高的 K。
        """
        if k_range is None:
            n = X.shape[0]
            max_k = min(10, n - 1)
            k_range = range(2, max_k + 1)

        Ks = list(k_range)
        metrics = {"silhouette": [], "db": [], "ch": [], "inertia": []}
        self.k_selection_log_ = []

        for k in Ks:
            km = KMeans(n_clusters=k, n_init=30, random_state=self.random_state)
            lbl = km.fit_predict(X)
            sil = silhouette_score(X, lbl)
            db = davies_bouldin_score(X, lbl)
            ch = calinski_harabasz_score(X, lbl)
            inert = km.inertia_
            metrics["silhouette"].append(sil)
            metrics["db"].append(db)
            metrics["ch"].append(ch)
            metrics["inertia"].append(inert)
            self.k_selection_log_.append({
                "k": k, "silhouette": round(sil, 6),
                "davies_bouldin": round(db, 6),
                "calinski_harabasz": round(ch, 4),
                "inertia": round(inert, 6),
            })

        # 每个指标排名（silhouette/CH 越大越好，DB/inertia 越小越好）
        ranks = np.zeros(len(Ks))
        for key, ascending in [("silhouette", True), ("ch", True),
                                ("db", False), ("inertia", False)]:
            order = np.argsort(metrics[key])
            if not ascending:
                order = order[::-1]
            for rank, idx in enumerate(order):
                ranks[idx] += rank  # rank 0 = best

        best_idx = int(np.argmin(ranks))
        best_k = Ks[best_idx]
        self.k_selection_log_[best_idx]["selected"] = True
        self.n_clusters = best_k
        return best_k

    # ── 聚类 ──────────────────────────────────────────────
    def fit(self, X: np.ndarray) -> np.ndarray:
        """执行 KMeans 聚类。若未指定 n_clusters 则自动选择。"""
        if self.n_clusters is None:
            self.n_clusters = self.auto_select_k(X)

        self.chosen_k_ = self.n_clusters
        self.model_ = KMeans(
            n_clusters=self.n_clusters,
            n_init=30,
            random_state=self.random_state,
        )
        self.labels_ = self.model_.fit_predict(X)
        return self.labels_

    # ── 离散程度报告 ──────────────────────────────────────
    def dispersion_report(self, X: np.ndarray) -> dict:
        """生成综合离散程度报告。

        Returns
        -------
        dict 包含:
          - n_clusters, n_samples
          - global: silhouette, davies_bouldin, calinski_harabasz, inertia
          - per_cluster: 每个簇的 n_members, mean/max/std intra dist, radius
          - inter_cluster_distances: 簇心之间的成对距离矩阵
          - per_point: 每个点的簇内距离 + 到最近异簇心的距离比
        """
        if self.model_ is None or self.labels_ is None:
            raise RuntimeError("请先调用 fit()")

        labels = self.labels_
        centroids = self.model_.cluster_centers_
        n = self.chosen_k_

        # 每个点到所有簇心的 cosine 距离
        all_dists = pairwise_distances(X, centroids, metric="cosine")
        # 到自身簇心的距离
        own_dists = all_dists[np.arange(len(labels)), labels]

        # ─ per-cluster ─
        per_cluster = []
        for c in range(n):
            mask = labels == c
            d = own_dists[mask]
            per_cluster.append({
                "cluster": int(c),
                "n_members": int(mask.sum()),
                "mean_intra_dist": float(d.mean()),
                "max_intra_dist": float(d.max()),
                "std_intra_dist": float(d.std()),
                "radius": float(d.max()),  # 最远点距离即半径
            })

        # ─ inter-cluster ─
        centroid_dists = pairwise_distances(centroids, metric="cosine")
        inter_cluster = centroid_dists.tolist()

        # ─ per-point 距离比 ─
        # 到最近其他簇心的距离 / 到自身簇心的距离，> 1 说明该点更靠近自身簇心
        other_dists = np.array([
            [all_dists[i, j] for j in range(n) if j != labels[i]]
            for i in range(len(labels))
        ])
        nearest_other = other_dists.min(axis=1)
        distance_ratio = nearest_other / (own_dists + 1e-12)
        mean_ratio_per_cluster = [
            float(distance_ratio[labels == c].mean()) for c in range(n)
        ]

        # per-point 离散程度（每个支撑点的值，表明它离散程度）
        per_point = [
            {
                "point_idx": int(i),
                "cluster": int(labels[i]),
                "dist_to_own_centroid": float(own_dists[i]),
                "dist_to_nearest_other": float(nearest_other[i]),
                "dist_ratio": float(distance_ratio[i]),
            }
            for i in range(len(labels))
        ]

        # ─ global ─
        report = {
            "n_clusters": n,
            "n_samples": len(labels),
            "global": {
                "silhouette_score": float(silhouette_score(X, labels)),
                "davies_bouldin_score": float(davies_bouldin_score(X, labels)),
                "calinski_harabasz_score": float(calinski_harabasz_score(X, labels)),
                "inertia": float(self.model_.inertia_),
            },
            "per_cluster": per_cluster,
            "inter_cluster_distances": inter_cluster,
            "mean_distance_ratio_per_cluster": mean_ratio_per_cluster,
            "per_point": per_point,
            "k_selection_log": self.k_selection_log_ or [],
        }
        return report

    # ── 导出 ──────────────────────────────────────────────
    @staticmethod
    def export_stats(report: dict, out_dir: str) -> None:
        """将离散程度报告写入 cluster_stats.csv 和 dispersion_report.json。"""
        import os
        os.makedirs(out_dir, exist_ok=True)

        # CSV
        rows = []
        for c in report["per_cluster"]:
            rows.append({
                "cluster": c["cluster"],
                "n_members": c["n_members"],
                "mean_intra_dist": round(c["mean_intra_dist"], 6),
                "max_intra_dist": round(c["max_intra_dist"], 6),
                "std_intra_dist": round(c["std_intra_dist"], 6),
                "radius": round(c["radius"], 6),
                "mean_dist_ratio": round(
                    report["mean_distance_ratio_per_cluster"][c["cluster"]], 4
                ),
            })
        pd.DataFrame(rows).to_csv(f"{out_dir}/cluster_stats.csv", index=False)

        # K 选择过程
        if report.get("k_selection_log"):
            pd.DataFrame(report["k_selection_log"]).to_csv(
                f"{out_dir}/k_selection.csv", index=False
            )

        # JSON（完整报告）
        with open(f"{out_dir}/dispersion_report.json", "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
