#!/usr/bin/env python3
"""
Docking preparation: ESMFold folding + RDKit ligand prep + Rosetta scripts
Run on GPU machine.
"""
import sys, time
sys.path.insert(0, ".")
from src.docking import fold_protein_esmfold, prepare_ligand, clean_protein_pdb
from pathlib import Path
import pandas as pd

df = pd.read_csv("docking_input_12seq.csv")
out_dir = Path("docking_output")
out_dir.mkdir(exist_ok=True)

for i, (_, row) in enumerate(df.iterrows()):
    seq_id = row["Entry"]
    seq = row["Enzyme"]
    smiles = row["Substrates"]
    l2 = row["L2_Cluster"]
    cand_dir = out_dir / seq_id
    cand_dir.mkdir(exist_ok=True)

    print(f"\n[{i+1}/12] {seq_id} (L2-C{l2})", flush=True)

    # 1. Fold
    pdb_path = cand_dir / "protein.pdb"
    if pdb_path.exists() and pdb_path.stat().st_size > 1000:
        print("  [SKIP] PDB exists", flush=True)
    else:
        t0 = time.time()
        ok = fold_protein_esmfold(seq, pdb_path)
        dt = time.time() - t0
        print(f"  [Fold] {dt:.0f}s {'OK' if ok else 'FAIL'}", flush=True)
        if not ok:
            continue

    # 2. Ligand
    sdf_path = cand_dir / "ligand.sdf"
    if sdf_path.exists():
        print("  [SKIP] SDF exists", flush=True)
    else:
        ok = prepare_ligand(smiles, sdf_path)
        print(f"  [Ligand] {'OK' if ok else 'FAIL'}", flush=True)
        if not ok:
            continue

    # 3. Clean PDB
    clean_path = cand_dir / "protein_clean.pdb"
    if not clean_path.exists():
        clean_protein_pdb(pdb_path, clean_path)
        print("  [Clean] OK", flush=True)

print(f"\nDONE! Results in {out_dir.resolve()}", flush=True)
