#!/usr/bin/env python3

import argparse
import math
import sys
from pathlib import Path
from collections import Counter

import numpy as np

A3M_DIR = Path("data/msa/a3m")
FEAT_DIR = Path("data/msa/features")


def parse_stockholm_alignment(a3m_path: Path):
    seqs = []

    with open(a3m_path) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            if line == "//":
                continue

            parts = line.split()

            if len(parts) < 2:
                continue

            seq = parts[-1]

            if set(seq) <= set(
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ-.abcdefghijklmnopqrstuvwxyz"
            ):
                seqs.append(seq)

    return seqs


def extract_features_from_a3m(a3m_path: Path):
    if not a3m_path.exists():
        return None

    if a3m_path.stat().st_size == 0:
        return None

    msa_seqs = parse_stockholm_alignment(a3m_path)

    if len(msa_seqs) < 2:
        return None

    msa_depth = len(msa_seqs)

    L = min(max(len(s) for s in msa_seqs), 2000)

    conservation = []
    entropy_vals = []

    gap_count = 0
    total_positions = 0

    for pos in range(L):

        col = []

        for s in msa_seqs:
            if pos < len(s):
                aa = s[pos].upper()
            else:
                aa = "-"

            col.append(aa)

        aa_col = [c for c in col if c != "-"]

        gap_count += col.count("-")
        total_positions += len(col)

        if len(aa_col) < 2:
            continue

        freq = Counter(aa_col)
        n = len(aa_col)

        conservation.append(
            freq.most_common(1)[0][1] / n
        )

        ent = 0.0

        for cnt in freq.values():
            p = cnt / n

            if p > 0:
                ent -= p * math.log2(p)

        entropy_vals.append(ent)

    if len(conservation) == 0:
        return None

    return {
        "msa_depth": float(msa_depth),
        "conservation_mean": float(np.mean(conservation)),
        "conservation_std": float(np.std(conservation)),
        "entropy_mean": float(np.mean(entropy_vals)),
        "entropy_std": float(np.std(entropy_vals)),
        "gap_ratio": float(
            gap_count / max(total_positions, 1)
        ),
    }


def save_feature_file(pid, feats, outdir):
    arr = np.array(
        [
            feats["msa_depth"],
            feats["conservation_mean"],
            feats["conservation_std"],
            feats["entropy_mean"],
            feats["entropy_std"],
            feats["gap_ratio"],
        ],
        dtype=np.float32,
    )

    np.save(outdir / f"{pid}.npy", arr)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--a3m")
    parser.add_argument("--protein-id")

    parser.add_argument(
        "--all",
        action="store_true",
    )

    parser.add_argument(
        "--a3m-dir",
        default="data/msa/a3m",
    )

    parser.add_argument(
        "--outdir",
        default="data/msa/features",
    )

    args = parser.parse_args()

    a3m_dir = Path(args.a3m_dir)

    outdir = Path(args.outdir)
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.a3m:
        a3m_path = Path(args.a3m)

        feats = extract_features_from_a3m(
            a3m_path
        )

        if feats is None:
            print(
                "FAILED",
                file=sys.stderr,
            )
            sys.exit(1)

        save_feature_file(
            a3m_path.stem,
            feats,
            outdir,
        )

        print(feats)
        return

    if args.protein_id:
        a3m_path = (
            a3m_dir
            / f"{args.protein_id}.a3m"
        )

        feats = extract_features_from_a3m(
            a3m_path
        )

        if feats is None:
            print(
                "FAILED",
                file=sys.stderr,
            )
            sys.exit(1)

        save_feature_file(
            args.protein_id,
            feats,
            outdir,
        )

        print(feats)
        return

    if args.all:

        files = sorted(
            a3m_dir.glob("*.a3m")
        )

        processed = 0
        failed = 0

        print(
            f"Processing {len(files)} files..."
        )

        for f in files:

            feats = extract_features_from_a3m(
                f
            )

            if feats is None:
                failed += 1
                continue

            save_feature_file(
                f.stem,
                feats,
                outdir,
            )

            processed += 1

            if processed % 100 == 0:
                print(
                    f"{processed}/{len(files)}"
                )

        print()
        print(
            f"Done: {processed} OK, {failed} FAIL"
        )

        return

    parser.print_help()


def extract(
    protein_id: str,
    a3m_dir="data/msa/a3m",
):
    feats = extract_features_from_a3m(
        Path(a3m_dir)
        / f"{protein_id}.a3m"
    )

    if feats is None:
        return None

    return np.array(
        [
            feats["msa_depth"],
            feats["conservation_mean"],
            feats["conservation_std"],
            feats["entropy_mean"],
            feats["entropy_std"],
            feats["gap_ratio"],
        ],
        dtype=np.float32,
    )


if __name__ == "__main__":
    main()
