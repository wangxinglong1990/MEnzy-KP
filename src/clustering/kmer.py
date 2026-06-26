"""k-mer 特征提取与矩阵构建"""

import numpy as np
from collections import Counter
from Bio import SeqIO
from sklearn.feature_extraction.text import TfidfTransformer

# 标准蛋白质氨基酸字母表
PROTEIN_ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")


class KmerFeatureExtractor:
    """从 FASTA 中提取 k-mer 频率特征并构建 TF-IDF 矩阵。"""

    def __init__(self, k: int = 3, min_seq_len: int | None = None):
        """
        Parameters
        ----------
        k : int
            k-mer 长度。
        min_seq_len : int or None
            最短序列长度（氨基酸数），短于此值的序列被丢弃。
        """
        self.k = k
        self.min_seq_len = min_seq_len or k

    # ── 静态工具 ──────────────────────────────────────────
    @staticmethod
    def get_kmers(seq: str, k: int) -> list[str]:
        """提取一条序列的全部 k-mer（仅保留标准氨基酸）。"""
        clean = "".join(c for c in seq.upper() if c in PROTEIN_ALPHABET)
        return [clean[i:i + k] for i in range(len(clean) - k + 1)]

    # ── 构建计数矩阵 ──────────────────────────────────────
    def build_matrix(self, fasta_path: str) -> tuple[np.ndarray, list[str], list[str], list[str]]:
        """读取 FASTA → 过滤 → 构建 k-mer 计数矩阵。

        Returns
        -------
        X : np.ndarray  (n_seqs, n_vocab)
            原始计数矩阵。
        ids : list[str]
            序列标识符。
        vocab : list[str]
            k-mer 词汇表（按字母序）。
        seqs : list[str]
            原始序列字符串（已过滤非标准氨基酸）。
        """
        records, ids, seqs = [], [], []
        for r in SeqIO.parse(fasta_path, "fasta"):
            s = str(r.seq)
            clean = "".join(c for c in s.upper() if c in PROTEIN_ALPHABET)
            if len(clean) >= self.min_seq_len:
                ids.append(r.id)
                seqs.append(clean)

        if not ids:
            raise ValueError(f"没有符合条件的序列（min_seq_len={self.min_seq_len}）")

        # 收集词汇表
        vocab_set: set[str] = set()
        all_counts: list[Counter] = []
        for s in seqs:
            kmers = self.get_kmers(s, self.k)
            cnt = Counter(kmers)
            vocab_set.update(cnt.keys())
            all_counts.append(cnt)

        vocab = sorted(vocab_set)
        v2i = {v: i for i, v in enumerate(vocab)}

        X = np.zeros((len(all_counts), len(vocab)), dtype=np.float64)
        for i, cnt in enumerate(all_counts):
            for km, c in cnt.items():
                X[i, v2i[km]] = c

        return X, ids, vocab, seqs

    # ── 从 list 构建矩阵（pipeline 内用，避免重复读文件）──
    def build_matrix_from_lists(
        self, ids: list[str], seqs: list[str]
    ) -> tuple[np.ndarray, list[str], list[str], list[str]]:
        """从已有的 ID/序列列表构建 k-mer 计数矩阵。

        适用于 pipeline 中已做过预筛选的场景，无需再次读取 FASTA。
        参数和返回值同 build_matrix()。
        """
        clean_seqs, clean_ids = [], []
        for rid, s in zip(ids, seqs):
            clean = "".join(c for c in s.upper() if c in PROTEIN_ALPHABET)
            if len(clean) >= self.min_seq_len:
                clean_ids.append(rid)
                clean_seqs.append(clean)

        if not clean_ids:
            raise ValueError(f"没有符合条件的序列（min_seq_len={self.min_seq_len}）")

        vocab_set: set[str] = set()
        all_counts: list[Counter] = []
        for s in clean_seqs:
            kmers = self.get_kmers(s, self.k)
            cnt = Counter(kmers)
            vocab_set.update(cnt.keys())
            all_counts.append(cnt)

        vocab = sorted(vocab_set)
        v2i = {v: i for i, v in enumerate(vocab)}

        X = np.zeros((len(all_counts), len(vocab)), dtype=np.float64)
        for i, cnt in enumerate(all_counts):
            for km, c in cnt.items():
                X[i, v2i[km]] = c

        return X, clean_ids, vocab, clean_seqs

    # ── TF-IDF 变换 ───────────────────────────────────────
    @staticmethod
    def tfidf_transform(count_matrix: np.ndarray) -> np.ndarray:
        """对计数矩阵做 TF-IDF 变换，返回稠密矩阵。"""
        return TfidfTransformer(sublinear_tf=True).fit_transform(count_matrix).toarray()
