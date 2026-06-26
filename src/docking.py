#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Molecular docking via RosettaLigand — SSH backend to company Linux server.

Mac side:  ESMFold API (fold protein) + RDKit (ligand SDF) + SSH trigger
Linux side: molfile_to_params.py → rosetta_scripts → score.sc + docked PDBs
"""

import shutil
import ssl
import subprocess
import urllib.request
from pathlib import Path
from typing import Optional

# ---- SSH Config (company Linux server) ---------------------------------

ROSETTA_BIN = "/home/xlw/rosetta_src_2021.16.61629_bundle/main/source/bin/rosetta_scripts.mpi.linuxgccrelease"
MOLFILE_TO_PARAMS = "/home/xlw/rosetta_src_2021.16.61629_bundle/main/source/scripts/python/public/molfile_to_params.py"

# ---- RosettaLigand XML (compatible with Rosetta 2021) -------------------

LIGAND_DOCK_XML = """<ROSETTASCRIPTS>
    <SCOREFXNS>
        <ScoreFunction name="ligand_dock" weights="ligand"/>
    </SCOREFXNS>
    <LIGAND_AREAS>
        <LigandArea name="ligand_area" chain="X" cutoff="6.0"
                    add_nbr_radius="true" all_atom_mode="false"/>
    </LIGAND_AREAS>
    <INTERFACE_BUILDERS>
        <InterfaceBuilder name="interface_builder" ligand_areas="ligand_area"/>
    </INTERFACE_BUILDERS>
    <MOVEMAP_BUILDERS>
        <MoveMapBuilder name="movemap_builder" sc_interface="interface_builder"/>
    </MOVEMAP_BUILDERS>
    <SCORINGGRIDS ligand_chain="X" width="15.0">
        <ClassicGrid grid_name="transform_grid" weight="1.0"/>
    </SCORINGGRIDS>
    <MOVERS>
        <Transform name="transform" chain="X" box_size="7.0"
                   move_distance="0.2" angle="20" cycles="200" repeats="2"
                   temperature="5.0"/>
        <HighResDocker name="high_res_docker" cycles="6"
                       repack_every_Nth="3" scorefxn="ligand_dock"
                       movemap_builder="movemap_builder"/>
        <FinalMinimizer name="final_minimizer" scorefxn="ligand_dock"
                        movemap_builder="movemap_builder"/>
        <InterfaceScoreCalculator name="interface_score" chains="X"
                                  scorefxn="ligand_dock"/>
    </MOVERS>
    <PROTOCOLS>
        <Add mover="transform"/>
        <Add mover="high_res_docker"/>
        <Add mover="final_minimizer"/>
        <Add mover="interface_score"/>
    </PROTOCOLS>
</ROSETTASCRIPTS>
"""


# ---- Step 1: Protein folding (ESMFold API) -----------------------------

def fold_protein_esmfold(sequence: str, output_pdb: Path, timeout: int = 300) -> bool:
    sequence = "".join(ch for ch in sequence.upper() if ch in "ACDEFGHIKLMNPQRSTVWY")
    if len(sequence) < 10:
        return False
    if len(sequence) > 400:
        print(f"[ESMFold] Truncating {len(sequence)} -> 400")
        sequence = sequence[:400]
    print(f"[ESMFold] Folding ({len(sequence)} residues)...")
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request("https://api.esmatlas.com/foldSequence/v1/",
                                     data=sequence.encode(),
                                     headers={"Content-Type": "text/plain"})
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
            pdb = r.read().decode()
        if not pdb.strip().startswith(("ATOM","HEADER","MODEL","REMARK")):
            return False
        output_pdb.write_text(pdb)
        print(f"[ESMFold] Saved: {output_pdb}")
        return True
    except Exception as e:
        print(f"[ESMFold] Error: {e}")
        return False


# ---- Step 2: Ligand SDF (RDKit) ----------------------------------------

def prepare_ligand(smiles: str, sdf: Path) -> bool:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        print("[Ligand] Missing RDKit"); return False
    print(f"[Ligand] {smiles}")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return False
    mol = Chem.AddHs(mol)
    p = AllChem.ETKDGv3(); p.randomSeed = 42
    if AllChem.EmbedMolecule(mol, p) != 0:
        p = AllChem.ETKDG(); p.randomSeed = 42
        if AllChem.EmbedMolecule(mol, p) != 0: return False
    AllChem.MMFFOptimizeMolecule(mol)
    Chem.SDWriter(str(sdf)).write(mol)
    print(f"[Ligand] SDF: {sdf}")
    return True


# ---- Step 3: Protein PDB cleanup ---------------------------------------

def clean_protein_pdb(inp: Path, out: Path) -> bool:
    cleaned = []
    for l in inp.read_text().splitlines():
        if l.startswith(("ATOM","HETATM","TER","END","MODEL","ENDMDL")):
            if l.startswith(("ATOM","HETATM")) and len(l) >= 22:
                l = l[:21] + "A" + l[22:]
            cleaned.append(l)
    out.write_text("\n".join(cleaned))
    return True


# ---- Step 4: SSH to Linux, run Rosetta ---------------------------------

def run_rosetta_docking(protein_pdb: Path, ligand_sdf: Path,
                        output_dir: Path) -> Optional[dict]:
    """Prepare all files for Rosetta docking on Linux.

    Mac side: generate PDB + SDF + dock.xml + run.sh
    Linux side (manual): scp pull files → ./run.sh → score.sc

    Returns dict with paths to generated files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy/move files if not already in output_dir
    if Path(protein_pdb).resolve() != (output_dir / "protein_clean.pdb").resolve():
        shutil.copy2(protein_pdb, output_dir / "protein_clean.pdb")
    if Path(ligand_sdf).resolve() != (output_dir / "ligand.sdf").resolve():
        shutil.copy2(ligand_sdf, output_dir / "ligand.sdf")
    (output_dir / "dock.xml").write_text(LIGAND_DOCK_XML)

    # Generate Linux runner script
    run_sh = f"""#!/bin/bash
# DLKin RosettaLigand Docking - Linux Runner
# Usage: ./run.sh
set -e

echo "=== molfile_to_params ==="
python3 {MOLFILE_TO_PARAMS} ligand.sdf --clobber 2>&1

echo "=== Build complex PDB (protein + ligand) ==="
cat protein_clean.pdb > complex.pdb
echo "TER" >> complex.pdb
# molfile_to_params outputs LG_0001.pdb; use sed to keep chain as-is (X)
cat LG_0001.pdb >> complex.pdb
echo "END" >> complex.pdb

echo "=== rosetta_scripts ==="
{ROSETTA_BIN} \\
    -s complex.pdb \\
    -extra_res_fa LG.params \\
    -parser:protocol dock.xml \\
    -nstruct 3 \\
    -overwrite 2>&1

echo "=== Done ==="
ls -la score.sc *_0001.pdb* 2>/dev/null || true
head -2 score.sc 2>/dev/null || true
"""
    run_script = output_dir / "run.sh"
    run_script.write_text(run_sh)
    run_script.chmod(0o755)

    print(f"[Dock] Files prepared in: {output_dir}")
    print(f"[Dock]   protein_clean.pdb  - protein structure")
    print(f"[Dock]   ligand.sdf         - substrate 3D structure")
    print(f"[Dock]   dock.xml           - RosettaLigand protocol")
    print(f"[Dock]   run.sh             - Linux runner script")
    print(f"[Dock]")
    print(f"[Dock] Next: copy this directory to Linux and run ./run.sh")
    print(f"[Dock]   Linux IP: 192.168.5.23  User: xlw")
    print(f"[Dock]   Linux command:")
    print(f"[Dock]     scp -r jiangdongdong@192.168.5.10:{output_dir} .")
    print(f"[Dock]     cd {output_dir.name} && ./run.sh")

    return {"success": True,
            "if_delta_REU": None,
            "total_score_REU": None,
            "work_dir": str(output_dir),
            "raw": "Files prepared. Run on Linux."}


# ---- Score parsing -----------------------------------------------------

def _parse_score(p: Path) -> Optional[dict]:
    if not p.exists(): return None
    hdr, best = None, None
    for line in p.read_text().splitlines():
        if line.startswith("SEQUENCE:"):
            hdr = line.split()[1:]
        if line.startswith("SCORE:") and hdr:
            parts = line.split()
            s = {k: _safe(v) for k, v in zip(hdr, parts[1:])}
            if best is None or (s.get("total_score") or 0) < (best.get("total_score") or 0):
                best = s
    if not best: return None
    if_d = best.get("interface_delta_Z") or best.get("interface_delta_X") or best.get("total_score", 0)
    return {"if_delta": if_d, "total": best.get("total_score")}


def _safe(v):
    try: return float(v)
    except: return None


# ---- High-level pipeline -----------------------------------------------

def dock_enzyme_substrate(seq: str, smi: str, out_dir: Path,
                          skip_folding: bool = False) -> Optional[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdb = out_dir / "protein.pdb"
    if not skip_folding:
        if not fold_protein_esmfold(seq, pdb): return None
    elif not pdb.exists(): return None
    cln = out_dir / "protein_clean.pdb"
    clean_protein_pdb(pdb, cln)
    sdf = out_dir / "ligand.sdf"
    if not prepare_ligand(smi, sdf): return None
    return run_rosetta_docking(cln, sdf, out_dir)


def dock_candidates_csv(csv_path: Path, out_dir: Path,
                        seq_col="Enzyme", smiles_col="Substrates",
                        skip_folding=False):
    import pandas as pd
    df = pd.read_csv(csv_path)
    results = []
    for i, row in df.iterrows():
        entry = row.get("Entry", f"c{i}")
        print(f"\n{'='*60}\nDock {i+1}/{len(df)}: {str(entry)[:60]}\n{'='*60}")
        r = dock_enzyme_substrate(str(row[seq_col]).strip(),
                                  str(row[smiles_col]).strip(),
                                  out_dir / f"candidate_{i:03d}",
                                  skip_folding=skip_folding)
        results.append({"index": i, "entry": str(entry), "result": r})
        if r and r["success"]:
            print(f"  if_dG: {r['if_delta_REU']:.2f} REU")
    return results


def generate_docking_report(results, out_csv: Path):
    import pandas as pd
    rows = [{"index": r["index"], "entry": r["entry"],
             "rosetta_if_delta_REU": (r.get("result") or {}).get("if_delta_REU"),
             "rosetta_total_REU": (r.get("result") or {}).get("total_score_REU"),
             "success": (r.get("result") or {}).get("success", False)}
            for r in results]
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    valid = df[df.success == True]
    if len(valid) > 0:
        print(f"\nDone {len(valid)}/{len(df)}. Best dG: {valid['rosetta_if_delta_REU'].min():.2f} REU")
    return df
