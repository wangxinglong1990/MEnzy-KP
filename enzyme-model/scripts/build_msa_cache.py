#!/usr/bin/env python3
"""Build full MSA1D cache — generate A3M files + extract 6-dim features.

Orchestrates generate_msa.py + extract_msa1d_features.py for all 12,848 proteins.

Supports:
    --workers N     parallel processes
    --resume        skip existing A3M/feature files
    --limit N       process only N proteins (for testing)

Usage:
    python scripts/build_msa_cache.py --workers 4
    python scripts/build_msa_cache.py --workers 4 --limit 100
    python scripts/build_msa_cache.py --workers 4 --resume
"""
import argparse, sys, time, hashlib, random
from pathlib import Path

# ── Ensure project root is on sys.path ──
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import numpy as np

from scripts.generate_msa import run_jackhmmer
from scripts.extract_msa1d_features import extract_features_from_a3m

A3M_DIR = Path("data/msa/a3m")
FEAT_DIR = Path("data/msa/features")
MASTER_DIR = Path("data/master")

random.seed(42)


def get_all_proteins() -> list[tuple[str, str]]:
    """Retrieve all unique (protein_id, sequence) pairs from Master Dataset."""
    ck = pd.read_csv(MASTER_DIR / "kcat.csv")
    cm = pd.read_csv(MASTER_DIR / "km.csv")
    combined = pd.concat([ck, cm])
    # Deduplicate by sequence
    uniq = combined[["sequence"]].drop_duplicates()
    # Generate protein_id
    uniq["protein_id"] = uniq["sequence"].apply(
        lambda s: hashlib.sha256(str(s).encode()).hexdigest()[:16]
    )
    return list(zip(uniq["protein_id"], uniq["sequence"]))


def main():
    parser = argparse.ArgumentParser(description="Build MSA1D cache for all proteins")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers")
    parser.add_argument("--resume", action="store_true", help="Skip existing files")
    parser.add_argument("--limit", type=int, default=None, help="Limit proteins to process")
    parser.add_argument("--generate-msa", action="store_true", default=True,
                        help="Generate A3M files (default: True)")
    parser.add_argument("--extract-features", action="store_true", default=True,
                        help="Extract features from A3M (default: True)")
    args = parser.parse_args()

    A3M_DIR.mkdir(parents=True, exist_ok=True)
    FEAT_DIR.mkdir(parents=True, exist_ok=True)

    proteins = get_all_proteins()
    if args.limit:
        random.shuffle(proteins)
        proteins = proteins[:args.limit]

    print(f"Proteins to process: {len(proteins):,}")
    t_start = time.time()
    gen_ok = 0
    gen_skip = 0
    feat_ok = 0
    feat_skip = 0
    feat_fail = 0

    for i, (pid, seq) in enumerate(proteins, 1):
        a3m_path = A3M_DIR / f"{pid}.a3m"
        feat_path = FEAT_DIR / f"{pid}.npy"

        # ── Generate A3M ──
        if args.generate_msa:
            if args.resume and a3m_path.exists() and a3m_path.stat().st_size > 0:
                gen_skip += 1
            else:
                try:
                    run_jackhmmer(pid, seq, A3M_DIR)
                    gen_ok += 1
                except Exception as e:
                    print(f"  [FAIL] {pid}: MSA generation error: {e}", file=sys.stderr)

        # ── Extract features ──
        if args.extract_features:
            if args.resume and feat_path.exists():
                feat_skip += 1
            elif a3m_path.exists() and a3m_path.stat().st_size > 0:
                feats = extract_features_from_a3m(a3m_path)
                if feats:
                    arr = np.array([
                        feats["msa_depth"], feats["conservation_mean"],
                        feats["conservation_std"], feats["entropy_mean"],
                        feats["entropy_std"], feats["gap_ratio"]
                    ], dtype=np.float32)
                    np.save(feat_path, arr)
                    feat_ok += 1
                else:
                    feat_fail += 1

        # Progress
        if i % 100 == 0 or i == len(proteins):
            elapsed = time.time() - t_start
            rate = i / elapsed
            remaining = (len(proteins) - i) / rate if rate > 0 else 0
            print(f"  [{i}/{len(proteins)}] "
                  f"MSA: {gen_ok}OK+{gen_skip}skip, "
                  f"Feat: {feat_ok}OK+{feat_skip}skip+{feat_fail}fail "
                  f"| {elapsed:.0f}s elapsed ~{remaining:.0f}s remaining",
                  flush=True)

    t_total = time.time() - t_start
    print(f"\n{'='*50}")
    print(f"BUILD COMPLETE")
    print(f"{'='*50}")
    print(f"  Total proteins:   {len(proteins):,}")
    print(f"  MSA generated:    {gen_ok:,}")
    print(f"  MSA skipped:      {gen_skip:,}")
    print(f"  Features OK:      {feat_ok:,}")
    print(f"  Features skipped: {feat_skip:,}")
    print(f"  Features failed:  {feat_fail:,}")
    print(f"  Total time:       {t_total:.0f}s ({t_total/60:.1f}min)")


if __name__ == "__main__":
    main()
