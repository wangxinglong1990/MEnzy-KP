"""序列抽取策略 — 从每个簇中提取代表性序列"""

import os
import numpy as np
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO


class SequenceSampler:
    """从聚类结果中按策略抽取代表性序列。"""

    STRATEGIES = ("closest", "farthest", "stratified")

    def __init__(self, strategy: str = "closest", n_per_cluster: int = 20):
        """
        Parameters
        ----------
        strategy : str
            抽取策略: "closest" | "farthest" | "stratified"。
        n_per_cluster : int
            每簇抽取的数量；若簇内序列不足则全部取出。
        """
        if strategy not in self.STRATEGIES:
            raise ValueError(f"strategy 必须是 {self.STRATEGIES} 之一，收到: {strategy}")
        self.strategy = strategy
        self.n_per_cluster = n_per_cluster

    def extract(
        self,
        labels: np.ndarray,
        dists: np.ndarray,
        ids: list[str],
        seqs: list[str],
    ) -> dict[int, list[SeqRecord]]:
        """从每个簇中抽取序列。

        Parameters
        ----------
        labels : np.ndarray
            每个序列的簇标签。
        dists : np.ndarray
            每个序列到其簇心的距离（cosine）。
        ids : list[str]
            序列 ID 列表。
        seqs : list[str]
            序列字符串列表。

        Returns
        -------
        dict[int, list[SeqRecord]]
            {cluster_id: [SeqRecord, ...]}
        """
        n_clusters = labels.max() + 1
        samples: dict[int, list[SeqRecord]] = {}

        for c in range(n_clusters):
            mask = labels == c
            idxs = np.where(mask)[0]
            cluster_dists = dists[mask]
            cluster_ids = [ids[i] for i in idxs]
            cluster_seqs = [seqs[i] for i in idxs]

            # 按策略排序
            if self.strategy == "closest":
                order = np.argsort(cluster_dists)  # 距离升序
            elif self.strategy == "farthest":
                order = np.argsort(cluster_dists)[::-1]  # 距离降序
            elif self.strategy == "stratified":
                order = self._stratified_order(cluster_dists)
            else:
                order = np.arange(len(cluster_dists))

            n_take = min(self.n_per_cluster, len(order))
            taken_idx = order[:n_take]

            samples[int(c)] = [
                SeqRecord(
                    Seq(cluster_seqs[i]),
                    id=cluster_ids[i],
                    description=(
                        f"cluster={c} dist={cluster_dists[i]:.6f} "
                        f"strategy={self.strategy}"
                    ),
                )
                for i in taken_idx
            ]

        return samples

    # ── 分层抽样 ──────────────────────────────────────────
    @staticmethod
    def _stratified_order(dists: np.ndarray) -> np.ndarray:
        """按距离等宽分桶，桶内随机打乱，轮流从各桶取一条，保证覆盖全距离范围。"""
        n = len(dists)
        if n <= 2:
            return np.arange(n)

        n_bins = min(5, n)
        bins = np.linspace(dists.min(), dists.max() + 1e-12, n_bins + 1)
        bin_indices = [[] for _ in range(n_bins)]
        for i, d in enumerate(dists):
            for b in range(n_bins):
                if bins[b] <= d < bins[b + 1]:
                    bin_indices[b].append(i)
                    break
            else:
                bin_indices[-1].append(i)

        # 桶内随机
        rng = np.random.default_rng(42)
        for bucket in bin_indices:
            rng.shuffle(bucket)

        # 轮询取索引
        order = []
        ptrs = [0] * n_bins
        while len(order) < n:
            added = False
            for b in range(n_bins):
                if ptrs[b] < len(bin_indices[b]):
                    order.append(bin_indices[b][ptrs[b]])
                    ptrs[b] += 1
                    added = True
            if not added:
                break
        return np.array(order)

    # ── 导出 FASTA ─────────────────────────────────────────
    @staticmethod
    def export_fasta(samples: dict[int, list[SeqRecord]], out_dir: str) -> None:
        """将抽取的序列按簇导出为 FASTA 文件，同时生成汇总文件。

        Parameters
        ----------
        samples : dict[int, list[SeqRecord]]
            extract() 的返回结果。
        out_dir : str
            输出目录（会在其下创建 extracted/ 子目录）。
        """
        extracted_dir = os.path.join(out_dir, "extracted")
        os.makedirs(extracted_dir, exist_ok=True)

        all_records = []
        for c, records in sorted(samples.items()):
            path = os.path.join(extracted_dir, f"cluster_{c}.fasta")
            SeqIO.write(records, path, "fasta")
            all_records.extend(records)

        # 汇总
        summary_path = os.path.join(extracted_dir, "all_extracted.fasta")
        SeqIO.write(all_records, summary_path, "fasta")
