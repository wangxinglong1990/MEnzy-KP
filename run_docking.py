#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Molecular docking CLI for DLKin pipeline.

Modes:
  1. Single: dock one enzyme–substrate pair
     python run_docking.py --protein <seq> --smiles <SMILES> --output-dir docking_results/

  2. Batch: dock from a CSV (e.g., the 6 manually selected candidates)
     python run_docking.py --csv selected_6_enzymes.csv --output-dir docking_results/
"""

import argparse
from pathlib import Path

from src.docking import dock_candidates_csv, dock_enzyme_substrate, generate_docking_report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Molecular docking verification for DLKin enzyme candidates."
    )
    # Single mode
    parser.add_argument("--protein", type=str, help="Protein sequence (single mode).")
    parser.add_argument("--smiles", type=str, help="Substrate SMILES (single mode).")

    # Batch mode
    parser.add_argument("--csv", type=str, help="CSV with enzyme candidates for batch docking.")
    parser.add_argument("--seq-col", type=str, default="Enzyme", help="Protein sequence column name.")
    parser.add_argument("--smiles-col", type=str, default="Substrates", help="SMILES column name.")

    # Common
    parser.add_argument("--output-dir", type=str, default="docking_results", help="Output directory.")
    parser.add_argument("--skip-folding", action="store_true",
                        help="Skip ESMFold folding (use pre-existing protein.pdb in candidate dirs).")

    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.csv:
        # ── Batch mode ──
        csv_path = Path(args.csv).resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        print(f"Batch docking from: {csv_path}")
        results = dock_candidates_csv(
            csv_path=csv_path,
            output_dir=output_dir,
            seq_col=args.seq_col,
            smiles_col=args.smiles_col,
            skip_folding=args.skip_folding,
        )
        generate_docking_report(results, output_dir / "docking_report.csv")

    elif args.protein and args.smiles:
        # ── Single mode ──
        result = dock_enzyme_substrate(
            args.protein, args.smiles, output_dir, skip_folding=args.skip_folding
        )
        if result and result.get("success"):
            print(f"\n{'='*50}")
            print(f"Docking complete!")
            print(f"  Interface dG: {result.get('if_delta_REU', 'N/A')} REU")
            print(f"  Total Score:  {result.get('total_score_REU', 'N/A')} REU")
        else:
            print("\nDocking failed. Check error messages above.")
            return 1

    else:
        print("Error: specify either --csv (batch mode) or both --protein and --smiles (single mode).")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
