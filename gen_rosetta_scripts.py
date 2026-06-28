#!/usr/bin/env python3
"""Generate Rosetta run scripts for all docked candidates."""
import sys
sys.path.insert(0, ".")
from src.docking import run_rosetta_docking
from pathlib import Path
import pandas as pd

df = pd.read_csv("docking_input_12seq.csv")
out_dir = Path("docking_output")

for _, row in df.iterrows():
    seq_id = row["Entry"]
    cand_dir = out_dir / seq_id
    pdb = cand_dir / "protein_clean.pdb"
    sdf = cand_dir / "ligand.sdf"

    if pdb.exists() and sdf.exists():
        print(f"Generating Rosetta scripts: {seq_id}", flush=True)
        run_rosetta_docking(pdb, sdf, cand_dir)
        print(f"  -> run.sh + dock.xml ready", flush=True)
    else:
        print(f"  SKIP {seq_id}: missing files", flush=True)

print("\nDone! Files per candidate:", flush=True)
for cand_dir in sorted(out_dir.iterdir()):
    if cand_dir.is_dir():
        files = [f.name for f in cand_dir.iterdir()]
        print(f"  {cand_dir.name}: {files}", flush=True)
