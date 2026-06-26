#!/usr/bin/env python3
"""Generate MSA (A3M format) for a protein sequence using jackhmmer.

Usage:
    python scripts/generate_msa.py --protein-id <pid> --sequence <seq>
    python scripts/generate_msa.py --fasta /path/to/seqs.fasta --outdir data/msa/a3m

Requires: jackhmmer (HMMER 3.4+) installed at /opt/homebrew/bin/jackhmmer
Reference DB: UniProt Swiss-Prot at /tmp/uniprot_db/uniprot_sprot.fasta
"""
import argparse, subprocess, time, sys
from pathlib import Path

# ── Ensure project root is on sys.path ──
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Default paths
JACKHMMER = "/usr/bin/jackhmmer"
REF_DB = "/tmp/uniprot_db/uniprot_sprot.fasta"
A3M_DIR = Path("data/msa/a3m")


def run_jackhmmer(protein_id: str, sequence: str, outdir: Path,
                  jackhmmer_path: str = JACKHMMER, ref_db: str = REF_DB,
                  e_val: float = 10.0, n_iterations: int = 2, timeout: int = 300) -> Path:
    """Run jackhmmer for a single sequence and write A3M output.

    Returns path to the A3M file (may be empty if no hits found).
    """
    outdir.mkdir(parents=True, exist_ok=True)
    a3m_path = outdir / f"{protein_id}.a3m"

    fasta_in = f">{protein_id}\n{sequence}\n"

    cmd = [
        jackhmmer_path,
        "-E", str(e_val),
        "--domE", str(e_val),
        "--incE", "0.01",
        "-N", str(n_iterations),
        "--cpu", "1",
        "--noali",
        "-A", str(a3m_path),
        "-",
        ref_db
    ]

    try:
        subprocess.run(cmd, input=fasta_in, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  [WARN] {protein_id}: jackhmmer timed out after {timeout}s", file=sys.stderr)
    except FileNotFoundError:
        raise RuntimeError(
            f"jackhmmer not found at {jackhmmer_path}. "
            f"Install with: brew install hmmer"
        )

    return a3m_path


def main():
    parser = argparse.ArgumentParser(description="Generate MSA for protein sequence(s)")
    parser.add_argument("--protein-id", help="Single protein ID")
    parser.add_argument("--sequence", help="Single protein sequence")
    parser.add_argument("--fasta", help="FASTA file with multiple sequences")
    parser.add_argument("--outdir", default="data/msa/a3m", help="Output directory")
    parser.add_argument("--jackhmmer", default=JACKHMMER, help="Path to jackhmmer binary")
    parser.add_argument("--ref-db", default=REF_DB, help="Reference database path")
    parser.add_argument("--e-val", type=float, default=10.0, help="E-value threshold")
    parser.add_argument("--iterations", type=int, default=2, help="Jackhmmer iterations")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.protein_id and args.sequence:
        print(f"Generating MSA for {args.protein_id}...")
        t0 = time.time()
        a3m = run_jackhmmer(args.protein_id, args.sequence, outdir,
                            args.jackhmmer, args.ref_db, args.e_val, args.iterations)
        print(f"  Done in {time.time()-t0:.1f}s → {a3m}")
        return

    if args.fasta:
        fasta_path = Path(args.fasta)
        seqs = []
        with open(fasta_path) as f:
            lines = f.read().strip().split("\n")
        for i in range(0, len(lines), 2):
            pid = lines[i].lstrip(">").strip()
            seq = lines[i+1].strip()
            seqs.append((pid, seq))

        print(f"Generating MSA for {len(seqs)} sequences from {args.fasta}...")
        total_time = 0
        for pid, seq in seqs:
            t0 = time.time()
            a3m = run_jackhmmer(pid, seq, outdir,
                                args.jackhmmer, args.ref_db, args.e_val, args.iterations)
            elapsed = time.time() - t0
            total_time += elapsed
            size = a3m.stat().st_size if a3m.exists() else 0
            print(f"  {pid:16s}  {elapsed:>6.1f}s  {size:>8,} bytes")
        print(f"Total: {len(seqs)} seqs in {total_time:.0f}s ({total_time/60:.1f}min)")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
