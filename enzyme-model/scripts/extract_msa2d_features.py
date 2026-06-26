#!/usr/bin/env python3

import argparse
import math
from pathlib import Path
from collections import Counter

import numpy as np

FEATURE_DIM = 46


def entropy(values):
    if len(values) == 0:
        return 0.0

    cnt = Counter(values)
    n = len(values)

    e = 0.0

    for c in cnt.values():
        p = c / n
        e -= p * math.log2(p)

    return float(e)


def parse_stockholm(filepath):

    if not filepath.exists():
        return []

    hits = []

    with open(filepath) as f:

        for line in f:

            if not line.startswith("#=GS"):
                continue

            parts = line.strip().split()

            if len(parts) < 3:
                continue

            acc = parts[1]

            hits.append(acc)

    return hits


def extract_features_from_hits(hits):

    n = len(hits)

    if n < 2:
        return None

    accessions = []
    species = []

    coverages = []

    lengths = []

    sp_count = 0
    tr_count = 0

    for acc in hits:

        if acc.startswith("sp|"):
            sp_count += 1

        if acc.startswith("tr|"):
            tr_count += 1

        parts = acc.split("|")

        if len(parts) >= 2:
            accessions.append(parts[1])
        else:
            accessions.append(acc)

        if len(parts) >= 3:

            name = parts[2]

            base = name.split("/")[0]

            lengths.append(len(base))

            if "_" in base:
                species.append(base.split("_")[-1])
            else:
                species.append("UNK")

        else:

            lengths.append(0)
            species.append("UNK")

        if "/" in acc:

            try:

                rng = acc.split("/")[-1]

                s, e = rng.split("-")

                s = int(float(s))
                e = int(float(e))

                cov = min((e - s) / 1000.0, 1.0)

            except Exception:

                cov = 1.0

        else:

            cov = 1.0

        coverages.append(cov)

    coverages = np.array(coverages, dtype=np.float32)
    lengths = np.array(lengths, dtype=np.float32)

    feats = []

    # original block

    feats.extend([
        float(n),
        math.log10(n + 1),
        math.sqrt(n),
        len(set(accessions)) / n,
        coverages.mean(),
        coverages.std(),
        len(set(species)) / n,
        lengths.mean(),
        lengths.std(),
        np.mean(coverages > 0.8),
        min(math.sqrt(n) / 100.0, 1.0),
        np.mean(coverages > 0.9),
        float(
            np.mean(
                ((coverages-coverages.mean())**3)
            ) / (coverages.std()+1e-6)**3
        ),
        entropy(species),
    ])

    # reviewed statistics

    feats.extend([
        sp_count / n,
        tr_count / n,
    ])

    # coverage stats

    feats.extend([
        coverages.min(),
        coverages.max(),
        np.median(coverages),
        np.percentile(coverages,25),
        np.percentile(coverages,75),
    ])

    # length stats

    feats.extend([
        lengths.min(),
        lengths.max(),
        np.median(lengths),
        np.percentile(lengths,25),
        np.percentile(lengths,75),
    ])

    # species stats

    species_counter = Counter(species)

    top_species = species_counter.most_common(1)[0][1]

    feats.extend([
        len(species_counter),
        entropy(species),
        top_species / n,
    ])

    # accession stats

    accession_counter = Counter(accessions)

    top_acc = accession_counter.most_common(1)[0][1]

    feats.extend([
        len(accession_counter),
        entropy(accessions),
        top_acc / n,
    ])

    # coverage histogram

    hist,_ = np.histogram(
        coverages,
        bins=[0,0.2,0.4,0.6,0.8,1.01]
    )

    feats.extend(
        (hist / n).tolist()
    )

    # length histogram

    hist,_ = np.histogram(
        lengths,
        bins=[0,50,100,200,400,10000]
    )

    feats.extend(
        (hist / n).tolist()
    )

    # cross features

    species_diversity = len(set(species)) / n

    feats.extend([
        n * species_diversity,
        n * coverages.mean(),
        coverages.mean() * entropy(species),
        species_diversity * entropy(species),
    ])

    return np.array(
        feats,
        dtype=np.float32
    )


def extract(protein_id,
            a3m_dir="data/msa/a3m",
            feat_dir="data/msa2d/features"):

    a3m = Path(a3m_dir) / f"{protein_id}.a3m"

    hits = parse_stockholm(a3m)

    if len(hits) < 2:
        return None

    return extract_features_from_hits(hits)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--all",
        action="store_true"
    )

    parser.add_argument(
        "--a3m-dir",
        default="data/msa/a3m"
    )

    parser.add_argument(
        "--outdir",
        default="data/msa2d/features"
    )

    args = parser.parse_args()

    outdir = Path(args.outdir)

    outdir.mkdir(
        parents=True,
        exist_ok=True
    )

    files = list(
        Path(args.a3m_dir).glob("*.a3m")
    )

    ok = 0

    for fp in files:

        pid = fp.stem

        hits = parse_stockholm(fp)

        if len(hits) < 2:
            continue

        feat = extract_features_from_hits(hits)

        np.save(
            outdir / f"{pid}.npy",
            feat
        )

        ok += 1

    print(
        f"generated {ok} features"
    )


if __name__ == "__main__":
    main()

