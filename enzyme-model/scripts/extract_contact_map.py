#!/usr/bin/env python3
"""Extract contact map features from aligned MSA (Stockholm A3M format).

Computes pairwise residue contact scores using:
    - Mutual Information (MI)
    - Covariance (raw correlation)
    - (reserved) Direct Coupling Analysis (DCA)

Output: (L, L) float32 matrix saved as NPY.

Usage:
    python scripts/extract_contact_map.py --a3m data/msa_full/a3m/<pid>.a3m
    python scripts/extract_contact_map.py --all
    python scripts/extract_contact_map.py --all --outdir data/msa_full/contact_maps
"""
import argparse, math, sys
from pathlib import Path
import numpy as np
from collections import Counter

ALPHABET = "ACDEFGHIKLMNPQRSTVWY-"
AA_TO_IDX = {aa: i for i, aa in enumerate(ALPHABET)}
N_AA = len(ALPHABET)  # 21 (20 AA + gap)


def parse_stockholm_alignment(filepath: Path) -> np.ndarray | None:
    """Parse Stockholm A3M and return aligned sequences as (N, L) integer matrix.

    Returns None if no valid alignment found.
    """
    if not filepath.exists() or filepath.stat().st_size == 0:
        return None
    with open(filepath) as f:
        lines = f.read().strip().split("\n")
    if not lines or not lines[0].startswith("# STOCKHOLM"):
        return None

    sequences = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        parts = line.split(None, 1)
        if len(parts) >= 2:
            seq = parts[1].upper()
            # Keep only standard AA characters
            seq_clean = "".join(c for c in seq if c in AA_TO_IDX or c == '.')
            if len(seq_clean) > 10:  # minimum length
                sequences.append(seq_clean)

    if len(sequences) < 10:
        return None

    # Convert to integer matrix
    L = max(len(s) for s in sequences)
    N = len(sequences)
    matrix = np.zeros((N, L), dtype=np.int8)
    for i, seq in enumerate(sequences):
        for j, aa in enumerate(seq):
            if j < L:
                matrix[i, j] = AA_TO_IDX.get(aa, AA_TO_IDX["-"])

    return matrix  # (N, L), values 0-20


def compute_mutual_information(msa: np.ndarray) -> np.ndarray:
    """Compute pairwise Mutual Information contact map.

    MI(i,j) = sum_{a,b} f_i(a) * f_j(b) * log(f_ij(a,b) / (f_i(a) * f_j(b)))

    Returns (L, L) matrix.
    """
    N, L = msa.shape
    contact = np.zeros((L, L), dtype=np.float32)
    small = 1e-10

    for i in range(L):
        col_i = msa[:, i]
        for j in range(i + 1, L):
            col_j = msa[:, j]
            mi = 0.0
            for a in range(N_AA):
                mask_a = col_i == a
                pa = mask_a.sum() / N
                if pa < small:
                    continue
                for b in range(N_AA):
                    mask_b = col_j == b
                    pab = (mask_a & mask_b).sum() / N
                    if pab < small:
                        continue
                    pb = mask_b.sum() / N
                    if pb < small:
                        continue
                    mi += pab * math.log2(pab / (pa * pb + small))
            contact[i, j] = mi
            contact[j, i] = mi

    return contact


def compute_covariance(msa: np.ndarray) -> np.ndarray:
    """Compute one-hot covariance contact map.

    One-hot encodes each column (21 dims), then computes (L*21, L*21)
    covariance matrix and sums over AA pairs for each position pair.

    Returns (L, L) matrix.
    """
    N, L = msa.shape
    contact = np.zeros((L, L), dtype=np.float32)

    for i in range(L):
        col_i = msa[:, i]
        for j in range(i + 1, L):
            col_j = msa[:, j]
            cov = 0.0
            for a in range(N_AA):
                xa = (col_i == a).astype(float)
                xa_mean = xa.mean()
                xa_centered = xa - xa_mean
                for b in range(N_AA):
                    xb = (col_j == b).astype(float)
                    xb_mean = xb.mean()
                    cov_ab = (xa_centered * (xb - xb_mean)).mean()
                    cov += abs(cov_ab)
            contact[i, j] = cov
            contact[j, i] = cov

    return contact


def extract_contact_map(a3m_path: Path, method: str = "mutual_info") -> np.ndarray | None:
    """Extract contact map from an A3M file.

    Args:
        a3m_path: Path to Stockholm A3M file.
        method: "mutual_info", "covariance", or "dca" (reserved).

    Returns:
        np.ndarray of shape (L, L), float32, or None.
    """
    msa = parse_stockholm_alignment(a3m_path)
    if msa is None:
        return None

    if method == "mutual_info":
        return compute_mutual_information(msa)
    elif method == "covariance":
        return compute_covariance(msa)
    elif method == "dca":
        # Reserved for future DCA implementation (CCMpred/plmDCA)
        raise NotImplementedError("DCA method not yet implemented")
    else:
        raise ValueError(f"Unknown method: {method}")


def main():
    parser = argparse.ArgumentParser(description="Extract contact maps from aligned MSA")
    parser.add_argument("--a3m", help="Path to a single A3M file")
    parser.add_argument("--all", action="store_true", help="Process all A3M files")
    parser.add_argument("--method", default="covariance", choices=["mutual_info", "covariance", "dca"])
    parser.add_argument("--a3m-dir", default="data/msa_full/a3m", help="A3M directory")
    parser.add_argument("--outdir", default="data/msa_full/contact_maps", help="Output dir")
    args = parser.parse_args()

    a3m_dir = Path(args.a3m_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    def _process(a3m_path: Path) -> tuple[str, bool]:
        pid = a3m_path.stem
        contact = extract_contact_map(a3m_path, args.method)
        if contact is not None:
            np.save(outdir / f"{pid}.npy", contact)
            return pid, True
        return pid, False

    if args.a3m:
        pid, ok = _process(Path(args.a3m))
        print(f"{pid}: {'✅' if ok else '❌'} → {outdir / f'{pid}.npy'}")
        return

    if args.all:
        a3m_files = sorted(a3m_dir.glob("*.a3m"))
        print(f"Processing {len(a3m_files)} A3M files with method '{args.method}'...")
        ok, fail = 0, 0
        for a3m_path in a3m_files:
            pid, success = _process(a3m_path)
            if success:
                ok += 1
            else:
                fail += 1
        print(f"Done: {ok} OK, {fail} FAIL")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
