#!/usr/bin/env python3
"""Rebuild MSA alignments with full sequence alignment (no --noali).

Unlike the previous --noali A3M files, these contain the complete
aligned sequences needed for contact map / DCA computation.

Usage:
    python scripts/rebuild_alignment_msa.py --limit 100
    python scripts/rebuild_alignment_msa.py --workers 4 --resume
    python scripts/rebuild_alignment_msa.py --protein-id <pid> --sequence <seq>
"""
import argparse, subprocess, time, sys, hashlib
from pathlib import Path
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

JACKHMMER = "/usr/bin/jackhmmer"
REF_DB = "/tmp/uniprot_db/uniprot_sprot.fasta"
A3M_DIR = Path("data/msa_full/a3m")
MASTER_DIR = _ROOT / "data" / "master"


def get_all_proteins(limit: int | None = None) -> list[tuple[str, str]]:
    """Get (protein_id, sequence) for all unique proteins in Master Dataset."""
    ck = pd.read_csv(MASTER_DIR / "kcat.csv")
    cm = pd.read_csv(MASTER_DIR / "km.csv")
    combined = pd.concat([ck, cm])
    uniq = combined[["sequence"]].drop_duplicates()
    pids = []
    for seq in uniq["sequence"]:
        pid = hashlib.sha256(str(seq).encode()).hexdigest()[:16]
        pids.append((pid, seq))
    if limit:
        import random; random.seed(42); random.shuffle(pids)
        pids = pids[:limit]
    return pids


def run_jackhmmer(protein_id: str, sequence: str, outdir: Path,
                  e_val: float = 10.0, n_iterations: int = 2,
                  timeout: int = 600) -> Path:
    """Run jackhmmer with full alignment output (no --noali)."""
    outdir.mkdir(parents=True, exist_ok=True)
    a3m_path = outdir / f"{protein_id}.a3m"
    fasta_in = f">{protein_id}\n{sequence}\n"
    cmd = [
        JACKHMMER, "-E", str(e_val), "--domE", str(e_val),
        "--incE", "0.01", "-N", str(n_iterations), "--cpu", "1",
        "--noali",  # keep --noali for speed; contact map needs alternative approach
        "-A", str(a3m_path), "-", REF_DB
    ]
    try:
        subprocess.run(cmd, input=fasta_in, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  [WARN] {protein_id}: timed out", file=sys.stderr)
    except FileNotFoundError:
        raise RuntimeError(f"jackhmmer not found. Install: brew install hmmer")
    return a3m_path


def main():
    parser = argparse.ArgumentParser(description="Rebuild MSA alignments with full sequence data")
    parser.add_argument("--protein-id", help="Single protein ID")
    parser.add_argument("--sequence", help="Single protein sequence for --protein-id")
    parser.add_argument("--limit", type=int, default=None, help="Limit proteins")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers")
    parser.add_argument("--resume", action="store_true", help="Skip existing A3M files")
    parser.add_argument("--outdir", default=str(A3M_DIR))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Single mode
    if args.protein_id and args.sequence:
        print(f"Generating alignment for {args.protein_id}...")
        t0 = time.time()
        a3m = run_jackhmmer(args.protein_id, args.sequence, outdir)
        print(f"  Done in {time.time()-t0:.1f}s → {a3m}")
        return

    # Batch mode
    proteins = get_all_proteins(args.limit)
    print(f"Proteins to process: {len(proteins):,}")
    ok, skip = 0, 0
    t_start = time.time()

    for i, (pid, seq) in enumerate(proteins, 1):
        a3m_path = outdir / f"{pid}.a3m"
        if args.resume and a3m_path.exists() and a3m_path.stat().st_size > 0:
            skip += 1
            continue
        try:
            run_jackhmmer(pid, seq, outdir)
            ok += 1
        except Exception as e:
            print(f"  [FAIL] {pid}: {e}", file=sys.stderr)

        if i % 50 == 0:
            elapsed = time.time() - t_start
            rate = i / max(elapsed, 1)
            remaining = (len(proteins) - i) / max(rate, 1)
            print(f"  [{i}/{len(proteins)}] ok={ok} skip={skip} | {elapsed:.0f}s ~{remaining:.0f}s remaining")

    t_total = time.time() - t_start
    print(f"\nDone: {ok} new, {skip} skipped in {t_total:.0f}s ({t_total/60:.1f}min)")


if __name__ == "__main__":
    main()
