#!/usr/bin/env python3
"""Build full contact map cache — generate aligned A3M → extract contact maps.

Orchestrates rebuild_alignment_msa.py + extract_contact_map.py.

Supports:
    --workers N     reserved for future parallel execution
    --resume        skip existing contact map files
    --limit N       process only N proteins (for testing)
    --method        mutual_info (default), covariance, or dca

Usage:
    python scripts/build_contact_cache.py --limit 100
    python scripts/build_contact_cache.py --workers 4 --resume
    python scripts/build_contact_cache.py --method covariance --resume
"""
import argparse, sys, time, hashlib, random
from pathlib import Path
import pandas as pd
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.rebuild_alignment_msa import run_jackhmmer
from scripts.extract_contact_map import extract_contact_map

JACKHMMER = "/opt/homebrew/bin/jackhmmer"
REF_DB = "/tmp/uniprot_db/uniprot_sprot.fasta"
MASTER_DIR = _ROOT / "data" / "master"
A3M_DIR = Path("data/msa_full/a3m")
CONTACT_DIR = Path("data/msa_full/contact_maps")


def get_all_proteins(limit: int | None = None) -> list[tuple[str, str]]:
    ck = pd.read_csv(MASTER_DIR / "kcat.csv")
    cm = pd.read_csv(MASTER_DIR / "km.csv")
    combined = pd.concat([ck, cm])
    uniq = combined[["sequence"]].drop_duplicates()
    pids = []
    for seq in uniq["sequence"]:
        pid = hashlib.sha256(str(seq).encode()).hexdigest()[:16]
        pids.append((pid, seq))
    if limit:
        random.seed(42); random.shuffle(pids)
        pids = pids[:limit]
    return pids


def main():
    parser = argparse.ArgumentParser(description="Build full contact map cache")
    parser.add_argument("--workers", type=int, default=1, help="Reserved for parallel")
    parser.add_argument("--resume", action="store_true", help="Skip existing files")
    parser.add_argument("--limit", type=int, default=None, help="Process N proteins")
    parser.add_argument("--method", default="covariance", choices=["mutual_info", "covariance", "dca"])
    parser.add_argument("--msa-only", action="store_true", help="Only generate MSA, skip contact maps")
    parser.add_argument("--contact-only", action="store_true", help="Only extract contact maps from existing A3M")
    args = parser.parse_args()

    A3M_DIR.mkdir(parents=True, exist_ok=True)
    CONTACT_DIR.mkdir(parents=True, exist_ok=True)

    proteins = get_all_proteins(args.limit)
    print(f"Proteins: {len(proteins):,}")
    t_start = time.time()
    msa_ok = 0; msa_skip = 0
    cm_ok = 0; cm_skip = 0; cm_fail = 0

    for i, (pid, seq) in enumerate(proteins, 1):
        a3m_path = A3M_DIR / f"{pid}.a3m"
        contact_path = CONTACT_DIR / f"{pid}.npy"

        # ── MSA generation ──
        if not args.contact_only:
            if args.resume and a3m_path.exists() and a3m_path.stat().st_size > 0:
                msa_skip += 1
            else:
                try:
                    run_jackhmmer(pid, seq, A3M_DIR)
                    msa_ok += 1
                except Exception:
                    pass

        # ── Contact map extraction ──
        if not args.msa_only:
            if args.resume and contact_path.exists():
                cm_skip += 1
            elif a3m_path.exists() and a3m_path.stat().st_size > 0:
                try:
                    contact = extract_contact_map(a3m_path, args.method)
                    if contact is not None:
                        np.save(contact_path, contact)
                        cm_ok += 1
                    else:
                        cm_fail += 1
                except Exception as e:
                    cm_fail += 1
                    if str(e):
                        pass  # silent on expected failures

        if i % 100 == 0 or i == len(proteins):
            elapsed = time.time() - t_start
            print(f"  [{i}/{len(proteins)}] MSA: {msa_ok}+{msa_skip}skip  CM: {cm_ok}+{cm_skip}skip+{cm_fail}fail | {elapsed:.0f}s", flush=True)

    t_total = time.time() - t_start
    print(f"\nTotal: {t_total:.0f}s ({t_total/60:.1f}min)")
    if not args.contact_only:
        print(f"MSA: {msa_ok} new, {msa_skip} existing")
    if not args.msa_only:
        print(f"Contact maps: {cm_ok} OK, {cm_skip} skip, {cm_fail} fail")


if __name__ == "__main__":
    main()
