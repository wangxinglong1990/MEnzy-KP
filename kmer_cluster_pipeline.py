#!/usr/bin/env python3
"""蛋白质序列 k-mer 聚类工作流

完整 pipeline：加载序列 → 预筛选 → k-mer 特征 → TF-IDF → 聚类 → 离散程度 → 抽取代表序列

用法示例:
    # 从 FASTA 基本聚类
    python kmer_cluster_pipeline.py data/seqs.fasta --k 3 --clusters 5

    # 从 CSV 输入（Kinora 工作流）
    python kmer_cluster_pipeline.py --csv-input blast500.csv \\
        --seq-col Enzyme --id-col Entry \\
        --prefilter-score blast500.csv --score-col Pred_kcat_over_Km \\
        --prefilter 200 --clusters 3 --sample 20

    # 带外部评分预筛选
    python kmer_cluster_pipeline.py data/seqs.fasta \\
        --prefilter-score data/scores.csv --prefilter 200 --sample 30
"""

import argparse
import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.metrics.pairwise import pairwise_distances

from src.clustering.kmer import KmerFeatureExtractor
from src.clustering.cluster import ClusterAnalyzer
from src.clustering.sampling import SequenceSampler


class ClusterPipeline:
    """编排完整的聚类工作流。"""

    def __init__(
        self,
        fasta: str | None = None,
        csv_input: str | None = None,
        seq_col: str = "Enzyme",
        id_col: str = "Entry",
        score_col: str | None = None,
        k: int = 3,
        n_clusters: int | None = None,
        prefilter_n: int | None = None,
        prefilter_score_csv: str | None = None,
        sample_n: int = 20,
        sample_strategy: str = "closest",
        min_seq_len: int | None = None,
        out_dir: str | None = None,
    ):
        self.fasta = fasta
        self.csv_input = csv_input
        self.seq_col = seq_col
        self.id_col = id_col
        self.score_col = score_col
        self.k = k
        self.n_clusters = n_clusters
        self.prefilter_n = prefilter_n
        self.prefilter_score_csv = prefilter_score_csv
        self.sample_n = sample_n
        self.sample_strategy = sample_strategy
        self.min_seq_len = min_seq_len
        self.out_dir = out_dir or f"outputs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 中间结果
        self.ids: list[str] = []
        self.seqs: list[str] = []
        self.X_tfidf: np.ndarray | None = None
        self.labels: np.ndarray | None = None
        self.pt_dists: np.ndarray | None = None
        self.dispersion: dict | None = None
        self.samples: dict | None = None

        self._setup_logging()

    def _setup_logging(self) -> None:
        os.makedirs(self.out_dir, exist_ok=True)
        log_path = os.path.join(self.out_dir, "pipeline.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.log = logging.getLogger(__name__)

    # ── 步骤 1：加载 + 预筛选 ──────────────────────────────
    def load_and_filter(self) -> None:
        """读取序列（FASTA 或 CSV），可选按外部评分或按顺序预筛选。"""
        self.log.info("=" * 60)
        self.log.info("步骤 1/4：加载序列 & 预筛选")
        self.log.info("=" * 60)

        raw_ids, raw_seqs = [], []

        # ── CSV 输入 ──
        if self.csv_input:
            self.log.info(f"  从 CSV 读取: {self.csv_input}")
            df = pd.read_csv(self.csv_input)
            if self.seq_col not in df.columns:
                raise ValueError(f"序列列 '{self.seq_col}' 不存在，可用列: {list(df.columns)}")
            if self.id_col not in df.columns:
                raise ValueError(f"ID 列 '{self.id_col}' 不存在，可用列: {list(df.columns)}")

            raw_ids = df[self.id_col].astype(str).str.strip().tolist()
            raw_seqs = df[self.seq_col].astype(str).str.strip().tolist()

        # ── FASTA 输入 ──
        elif self.fasta:
            self.log.info(f"  从 FASTA 读取: {self.fasta}")
            for r in SeqIO.parse(self.fasta, "fasta"):
                raw_ids.append(r.id)
                raw_seqs.append(str(r.seq))

        else:
            raise ValueError("必须提供 --csv-input 或 FASTA 文件路径")

        self.log.info(f"  读取 {len(raw_ids)} 条序列")

        # ── 外部评分预筛选 ──
        if self.prefilter_score_csv:
            self._prefilter_by_score(raw_ids, raw_seqs)

        # ── 按顺序预筛选（无外部评分时）──
        elif self.prefilter_n and self.prefilter_n < len(raw_ids):
            self.log.info(f"  按输入顺序取前 {self.prefilter_n} 条")
            raw_ids = raw_ids[:self.prefilter_n]
            raw_seqs = raw_seqs[:self.prefilter_n]

        self.ids = raw_ids
        self.seqs = raw_seqs
        self.log.info(f"  预筛选后: {len(self.ids)} 条序列")

    def _prefilter_by_score(self, raw_ids: list[str], raw_seqs: list[str]) -> None:
        """按外部评分 CSV 预筛选序列。"""
        self.log.info(f"  按外部评分文件预筛选: {self.prefilter_score_csv}")
        scores_df = pd.read_csv(self.prefilter_score_csv)

        # 确定 ID 列和评分列
        score_id_col = self.id_col if self.id_col in scores_df.columns else scores_df.columns[0]
        score_val_col = self.score_col if (self.score_col and self.score_col in scores_df.columns) else scores_df.columns[1]

        score_map = dict(zip(
            scores_df[score_id_col].astype(str).str.strip(),
            pd.to_numeric(scores_df[score_val_col], errors="coerce").fillna(0.0),
        ))
        scored = [(score_map.get(rid, 0.0), rid, s)
                   for rid, s in zip(raw_ids, raw_seqs)]
        scored.sort(key=lambda x: x[0], reverse=True)

        all_scores = [x[0] for x in scored]
        self.log.info(
            f"  评分统计: n={len(all_scores)}, "
            f"max={max(all_scores):.4f}, min={min(all_scores):.4f}, "
            f"mean={np.mean(all_scores):.4f}, median={np.median(all_scores):.4f}"
        )

        if self.prefilter_n and self.prefilter_n < len(scored):
            cutoff = scored[self.prefilter_n - 1][0]
            kept_scores = [x[0] for x in scored[:self.prefilter_n]]
            self.log.info(
                f"  按评分取前 {self.prefilter_n} 条, "
                f"截断值={cutoff:.4f}, "
                f"筛选后均值={np.mean(kept_scores):.4f}"
            )
            scored = scored[:self.prefilter_n]

        raw_ids[:] = [x[1] for x in scored]
        raw_seqs[:] = [x[2] for x in scored]

    # ── 步骤 2：k-mer 特征 + TF-IDF ────────────────────────
    def build_features(self) -> None:
        """构建 k-mer 计数矩阵并做 TF-IDF 变换。"""
        self.log.info("")
        self.log.info("=" * 60)
        self.log.info("步骤 2/4：k-mer 特征提取 & TF-IDF")
        self.log.info("=" * 60)

        extractor = KmerFeatureExtractor(k=self.k, min_seq_len=self.min_seq_len)
        X, ids, vocab, seqs = extractor.build_matrix_from_lists(self.ids, self.seqs)
        self.ids, self.seqs = ids, seqs

        self.log.info(f"  k={self.k}, 词汇表大小: {len(vocab)}")
        self.log.info(f"  计数矩阵: {X.shape[0]} seqs × {X.shape[1]} k-mers")

        self.X_tfidf = extractor.tfidf_transform(X)
        self.log.info(f"  TF-IDF 变换完成")

    # ── 步骤 3：聚类 + 离散程度 ────────────────────────────
    def run_clustering(self) -> None:
        """执行 KMeans 聚类并生成离散程度报告。"""
        self.log.info("")
        self.log.info("=" * 60)
        self.log.info("步骤 3/4：聚类 & 离散程度分析")
        self.log.info("=" * 60)

        analyzer = ClusterAnalyzer(n_clusters=self.n_clusters)
        self.labels = analyzer.fit(self.X_tfidf)
        self.log.info(f"  选定 K = {analyzer.chosen_k_}")

        self.dispersion = analyzer.dispersion_report(self.X_tfidf)
        g = self.dispersion["global"]
        self.log.info(f"  Silhouette:        {g['silhouette_score']:.4f}")
        self.log.info(f"  Davies-Bouldin:    {g['davies_bouldin_score']:.4f}")
        self.log.info(f"  Calinski-Harabasz: {g['calinski_harabasz_score']:.2f}")

        # 簇内距离统计
        for c in self.dispersion["per_cluster"]:
            self.log.info(
                f"  簇 {c['cluster']}: n={c['n_members']}, "
                f"mean_dist={c['mean_intra_dist']:.4f}, "
                f"radius={c['radius']:.4f}, "
                f"dist_ratio={self.dispersion['mean_distance_ratio_per_cluster'][c['cluster']]:.2f}"
            )

        # 计算 per-point distances（给 sampling 和 CSV 导出用）
        centroids = analyzer.model_.cluster_centers_
        all_dists = pairwise_distances(self.X_tfidf, centroids, metric="cosine")
        self.pt_dists = all_dists[np.arange(len(self.labels)), self.labels]

        # 到最近异簇心的距离（表征该点是否能跟其他簇分得开）
        n_clusters = analyzer.chosen_k_
        nearest_other = np.array([
            min(all_dists[i, j] for j in range(n_clusters) if j != self.labels[i])
            for i in range(len(self.labels))
        ])
        dist_ratio = nearest_other / (self.pt_dists + 1e-12)

        # 导出
        ClusterAnalyzer.export_stats(self.dispersion, self.out_dir)
        pd.DataFrame({
            "id": self.ids,
            "cluster": self.labels,
            "dist_to_center": self.pt_dists.round(6),
            "dist_to_nearest_other": nearest_other.round(6),
            "dist_ratio": dist_ratio.round(4),
        }).to_csv(f"{self.out_dir}/seq_to_cluster.csv", index=False)
        self.log.info(f"  报告已写入 {self.out_dir}/")

    # ── 步骤 4：抽取代表序列 ──────────────────────────────
    def extract_samples(self) -> None:
        """从每个簇中抽取代表性序列并导出 FASTA。"""
        self.log.info("")
        self.log.info("=" * 60)
        self.log.info("步骤 4/4：抽取代表序列")
        self.log.info("=" * 60)

        sampler = SequenceSampler(
            strategy=self.sample_strategy,
            n_per_cluster=self.sample_n,
        )
        self.samples = sampler.extract(
            self.labels, self.pt_dists, self.ids, self.seqs
        )

        for c, records in sorted(self.samples.items()):
            self.log.info(f"  簇 {c}: 抽取 {len(records)} 条 (strategy={self.sample_strategy})")

        sampler.export_fasta(self.samples, self.out_dir)
        self.log.info(f"  FASTA 已写入 {self.out_dir}/extracted/")

    # ── 运行完整流程 ──────────────────────────────────────
    def run(self) -> None:
        """执行完整工作流。"""
        source = self.csv_input or self.fasta
        self.log.info("╔══════════════════════════════════════════════════╗")
        self.log.info("║   k-mer 聚类工作流                               ║")
        self.log.info("╚══════════════════════════════════════════════════╝")
        self.log.info(f"  输入: {source}")
        self.log.info(f"  输出: {self.out_dir}")
        self.log.info(f"  参数: k={self.k}, clusters={self.n_clusters or 'auto'}, "
                      f"sample={self.sample_n}/{self.sample_strategy}")

        self.load_and_filter()
        self.build_features()
        self.run_clustering()
        self.extract_samples()

        self.log.info("")
        self.log.info("✓ 工作流完成")
        self.log.info(f"  输出目录: {os.path.abspath(self.out_dir)}")


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="k-mer 聚类工作流 — 蛋白质序列聚类、离散程度评估、代表序列抽取",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从 FASTA 基本聚类（自动选K）
  python kmer_cluster_pipeline.py data/seqs.fasta --k 3

  # 从 CSV 输入（Kinora 工作流）
  python kmer_cluster_pipeline.py --csv-input blast500.csv \\
      --seq-col Enzyme --id-col Entry \\
      --prefilter-score blast500.csv --score-col Pred_kcat_over_Km \\
      --prefilter 200 --clusters 3 --sample 20

  # 带外部评分预筛选（FASTA 输入）
  python kmer_cluster_pipeline.py data/seqs.fasta \\
      --prefilter-score data/scores.csv --prefilter 200 --sample 30
        """,
    )
    ap.add_argument("fasta", nargs="?", default=None,
                     help="输入 FASTA 文件路径（与 --csv-input 二选一）")
    ap.add_argument("--csv-input", type=str, default=None,
                     help="从 CSV 读取序列（与 FASTA 二选一）")
    ap.add_argument("--seq-col", type=str, default="Enzyme",
                     help="CSV 中序列列名 (default: Enzyme)")
    ap.add_argument("--id-col", type=str, default="Entry",
                     help="CSV 中 ID 列名 (default: Entry)")
    ap.add_argument("--score-col", type=str, default=None,
                     help="prefilter-score CSV 中的评分列名 (default: 第2列)")
    ap.add_argument("--k", type=int, default=3, help="k-mer 长度 (default: 3)")
    ap.add_argument("--clusters", type=int, default=None,
                     help="簇数 K；不指定则自动从 [2,10] 选择最佳 K")
    ap.add_argument("--prefilter", type=int, default=None,
                     help="预筛选：保留前 N 条序列（按评分或输入顺序）")
    ap.add_argument("--prefilter-score", type=str, default=None,
                     help="外部评分 CSV 文件，高分优先")
    ap.add_argument("--sample", type=int, default=20,
                     help="每簇抽取的序列数 (default: 20)")
    ap.add_argument("--sample-strategy", type=str, default="closest",
                     choices=["closest", "farthest", "stratified"],
                     help="抽取策略 (default: closest)")
    ap.add_argument("--min-seq-len", type=int, default=None,
                     help="最短序列长度，短于此值的丢弃 (default: 等于 k)")
    ap.add_argument("--out", type=str, default=None,
                     help="输出目录 (default: outputs/<timestamp>)")
    args = ap.parse_args()

    if not args.fasta and not args.csv_input:
        ap.error("必须提供 FASTA 文件或 --csv-input")

    pipeline = ClusterPipeline(
        fasta=args.fasta,
        csv_input=args.csv_input,
        seq_col=args.seq_col,
        id_col=args.id_col,
        score_col=args.score_col,
        k=args.k,
        n_clusters=args.clusters,
        prefilter_n=args.prefilter,
        prefilter_score_csv=args.prefilter_score,
        sample_n=args.sample,
        sample_strategy=args.sample_strategy,
        min_seq_len=args.min_seq_len,
        out_dir=args.out,
    )
    pipeline.run()
