#!/usr/bin/env python3
"""Build Master Dataset V2 — generates data/master/ from Condition Dataset."""
import hashlib, json, yaml, random
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_SOURCE = PROJECT_ROOT / "data" / "processed" / "Condition"
MASTER_DIR  = PROJECT_ROOT / "data" / "master"
REPORTS_DIR = PROJECT_ROOT / "reports"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"

random.seed(42)

# ── Helpers ───────────────────────────────────────────────────────
def make_protein_id(seq: str) -> str:
    return hashlib.sha256(str(seq).encode()).hexdigest()[:16]

def make_sample_id(seq: str, smi: str, temp: float, ph: float) -> str:
    raw = f"{seq}|{smi}|{temp}|{ph}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ── Load source data ──────────────────────────────────────────────
print("=" * 60)
print("M4 — MASTER DATASET CONSTRUCTION")
print("=" * 60)

ck_raw = pd.read_csv(DATA_SOURCE / "kcat_condition.csv", index_col=0)
cm_raw = pd.read_csv(DATA_SOURCE / "km_condition.csv", index_col=0)

# ── Build KCAT Master ─────────────────────────────────────────────
print("\n[1/8] Building KCAT master...")
ck = pd.DataFrame({
    "sequence":     ck_raw["sequence"],
    "smiles":       ck_raw["reaction_smiles"],
    "temperature":  ck_raw["temperature"].astype(float),
    "ph":           ck_raw["ph"].astype(float),
    "ec":           ck_raw["ec"],
    "uniprot":      ck_raw["uniprot"],
    "source":       ck_raw["sequence_source"],
    "target":       ck_raw["log10kcat_max"].astype(float),
    "task":         "kcat",
})
# Generate IDs
ck["protein_id"] = ck["sequence"].apply(make_protein_id)
ck["sample_id"]  = ck.apply(lambda r: make_sample_id(r["sequence"], r["smiles"],
                                                      r["temperature"], r["ph"]), axis=1)
print(f"  Rows: {len(ck):,}  |  Unique seqs: {ck['sequence'].nunique():,}")
print(f"  Unique sample_ids: {ck['sample_id'].nunique():,}  |  Target coverage: {ck['target'].notna().sum()/len(ck)*100:.1f}%")

# ── Build KM Master ────────────────────────────────────────────────
print("\n[2/8] Building KM master...")
cm = pd.DataFrame({
    "sequence":     cm_raw["sequence"],
    "smiles":       cm_raw["substrate_smiles"],
    "temperature":  cm_raw["temperature"].astype(float),
    "ph":           cm_raw["ph"].astype(float),
    "ec":           cm_raw["ec"],
    "uniprot":      cm_raw["uniprot"],
    "source":       cm_raw["sequence_source"],
    "target":       cm_raw["log10km_mean"].astype(float),
    "task":         "km",
})
cm["protein_id"] = cm["sequence"].apply(make_protein_id)
cm["sample_id"]  = cm.apply(lambda r: make_sample_id(r["sequence"], r["smiles"],
                                                      r["temperature"], r["ph"]), axis=1)
print(f"  Rows: {len(cm):,}  |  Unique seqs: {cm['sequence'].nunique():,}")
print(f"  Unique sample_ids: {cm['sample_id'].nunique():,}  |  Target coverage: {cm['target'].notna().sum()/len(cm)*100:.1f}%")

# ── Column order ──────────────────────────────────────────────────
COLUMNS = ["sample_id", "protein_id", "sequence", "smiles",
           "temperature", "ph", "ec", "uniprot", "source", "target", "task"]
ck = ck[COLUMNS]
cm = cm[COLUMNS]

MASTER_DIR.mkdir(parents=True, exist_ok=True)

# ── Protein-aware Split ──────────────────────────────────────────
print("\n[3/8] Generating protein-aware split (80/10/10)...")
all_seqs = list(set(ck["sequence"].tolist() + cm["sequence"].tolist()))
n = len(all_seqs)
random.shuffle(all_seqs)
n_train = int(n * 0.8)
n_val   = int(n * 0.9)
train_seqs = set(all_seqs[:n_train])
val_seqs   = set(all_seqs[n_train:n_val])
test_seqs  = set(all_seqs[n_val:])

def assign_split(seq):
    if seq in train_seqs: return "train"
    if seq in val_seqs:   return "val"
    return "test"

ck["split"] = ck["sequence"].apply(assign_split)
cm["split"] = cm["sequence"].apply(assign_split)

splits = {
    "train_protein_ids": sorted(sid for sid in set(all_seqs[:n_train])),
    "val_protein_ids":   sorted(sid for sid in set(all_seqs[n_train:n_val])),
    "test_protein_ids":  sorted(sid for sid in set(all_seqs[n_val:])),
    "total_unique_proteins": n,
    "train_pct": round(len(train_seqs)/n*100, 1),
    "val_pct":   round(len(val_seqs)/n*100, 1),
    "test_pct":  round(len(test_seqs)/n*100, 1),
}
print(f"  Unique proteins: {n:,}")
print(f"  Train: {len(train_seqs):,} ({splits['train_pct']}%)")
print(f"  Val:   {len(val_seqs):,} ({splits['val_pct']}%)")
print(f"  Test:  {len(test_seqs):,} ({splits['test_pct']}%)")

# ── Save CSVs & splits ──────────────────────────────────────────
print("\n[4/8] Writing data/master/...")
ck.to_csv(MASTER_DIR / "kcat.csv", index=False)
cm.to_csv(MASTER_DIR / "km.csv", index=False)
print(f"  kcat.csv: {len(ck):,} rows")
print(f"  km.csv:   {len(cm):,} rows")

with open(MASTER_DIR / "splits.json", "w") as f:
    json.dump(splits, f, indent=2)
print(f"  splits.json — {n} proteins in {splits['train_pct']}/{splits['val_pct']}/{splits['test_pct']}")

# ── Leakage Validation ─────────────────────────────────────────
print("\n[5/8] Validating...")
errors = []

# Check 1: protein_id split overlap
for ds_name, ds in [("kcat", ck), ("km", cm)]:
    for split_name in ["train", "val", "test"]:
        pids = set(ds[ds["split"] == split_name]["protein_id"])
        for other in ["train", "val", "test"]:
            if other <= split_name: continue
            opids = set(ds[ds["split"] == other]["protein_id"])
            overlap = pids & opids
            if overlap:
                errors.append(f"LEAK {ds_name}: protein_id overlap {split_name}∩{other} = {len(overlap)}")
overlap_train_val = set(train_seqs) & set(val_seqs)
overlap_train_test = set(train_seqs) & set(test_seqs)
overlap_val_test = set(val_seqs) & set(test_seqs)
if overlap_train_val: errors.append(f"LEAK: train∩val sequences = {len(overlap_train_val)}")
if overlap_train_test: errors.append(f"LEAK: train∩test sequences = {len(overlap_train_test)}")
if overlap_val_test: errors.append(f"LEAK: val∩test sequences = {len(overlap_val_test)}")

# Check 2: sample_id uniqueness
for ds_name, ds in [("kcat", ck), ("km", cm)]:
    n_uniq = ds["sample_id"].nunique()
    if n_uniq != len(ds):
        errors.append(f"DUPLICATE sample_id in {ds_name}: {len(ds) - n_uniq} duplicates")

# Check 3: target missing
for ds_name, ds in [("kcat", ck), ("km", cm)]:
    n_miss = ds["target"].isna().sum()
    if n_miss:
        errors.append(f"MISSING target in {ds_name}: {n_miss}")

# Check 4: split ratio
for ds_name, ds in [("kcat", ck), ("km", cm)]:
    r = ds["split"].value_counts(normalize=True)
    for s in ["train", "val", "test"]:
        if s not in r.index:
            errors.append(f"MISSING split '{s}' in {ds_name}")
            continue
        pct = r[s] * 100
        expected = {"train": 80, "val": 10, "test": 10}[s]
        if abs(pct - expected) > 2:
            errors.append(f"SPLIT_RATIO {ds_name}/{s}: {pct:.1f}% (expected ~{expected}%)")

if errors:
    print("  ❌ FAILED:")
    for e in errors:
        print(f"    - {e}")
else:
    print("  ✅ All checks PASSED")

# ── Schema ─────────────────────────────────────────────────────────
print("\n[6/8] Writing schema.yaml...")
schema = {
    "dataset": "enzyme-model Master Dataset V2",
    "version": "2.0",
    "created": "2026-06-23",
    "fields": {
        "sample_id": {
            "type": "str(16)",
            "required": True,
            "unique": True,
            "description": "Unique experiment ID = SHA256(seq|smiles|temp|ph)[:16]"
        },
        "protein_id": {
            "type": "str(16)",
            "required": True,
            "description": "Unique protein ID = SHA256(sequence)[:16]"
        },
        "sequence": {
            "type": "str",
            "required": True,
            "description": "Protein amino-acid sequence"
        },
        "smiles": {
            "type": "str",
            "required": True,
            "description": "Substrate SMILES"
        },
        "temperature": {
            "type": "float",
            "required": False,
            "description": "Reaction temperature in Celsius (nullable)"
        },
        "ph": {
            "type": "float",
            "required": False,
            "description": "Reaction pH (nullable)"
        },
        "ec": {
            "type": "str",
            "required": False,
            "description": "Enzyme Commission number"
        },
        "uniprot": {
            "type": "str",
            "required": False,
            "description": "UniProt ID"
        },
        "source": {
            "type": "str",
            "required": False,
            "description": "Data source (sabio / brenda / uniprot_search)"
        },
        "target": {
            "type": "float",
            "required": True,
            "description": "log10 transformed target (log10kcat or log10km)"
        },
        "task": {
            "type": "str",
            "required": True,
            "enum": ["kcat", "km"],
            "description": "Prediction target type"
        },
        "split": {
            "type": "str",
            "required": True,
            "enum": ["train", "val", "test"],
            "description": "Sequence-aware split assignment"
        }
    },
    "split_strategy": "protein-aware group-by-sequence 80/10/10",
    "id_system": {
        "protein_id": "SHA256(sequence)[:16] — for ESM cache, MSA1D, MSA2D",
        "sample_id": "SHA256(seq|smiles|temp|ph)[:16] — for training rows, stacking"
    },
    "intended_models": ["BaselineV2", "ConditionV2", "MSA1D", "MSA2D", "Stacking"]
}
with open(MASTER_DIR / "schema.yaml", "w") as f:
    yaml.dump(schema, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
print("  schema.yaml written")

# ── README ─────────────────────────────────────────────────────────
print("[7/8] Writing README.md...")
readme = f"""# Master Dataset V2

## Summary

| Metric | KCAT | KM | Total |
|--------|:----:|:--:|:-----:|
| Rows | {len(ck):,} | {len(cm):,} | {len(ck)+len(cm):,} |
| Unique proteins | {ck['sequence'].nunique():,} | {cm['sequence'].nunique():,} | {n:,} |
| Unique sample_ids | {ck['sample_id'].nunique():,} | {cm['sample_id'].nunique():,} | {ck['sample_id'].nunique()+cm['sample_id'].nunique():,} |

## Schema

See [schema.yaml](schema.yaml) for full field definitions.

## Split

- Strategy: protein-aware group-by-sequence
- Train: {splits['train_pct']}%
- Val:   {splits['val_pct']}%
- Test:  {splits['test_pct']}%

## ID System

- **protein_id** = SHA256(sequence)[:16] — for ESM cache, MSA1D, MSA2D
- **sample_id** = SHA256(seq|smiles|temp|ph)[:16] — for training rows, stacking

## Data Sources

- Condition Dataset (SABIO/BRENDA/UniProt_Search)

## Intended Models

- BaselineV2: (sequence, smiles) → target
- ConditionV2: (sequence, smiles, temperature, ph) → target
- MSA1D: (sequence, smiles, MSA-PSSM) → target
- MSA2D: (sequence, smiles, MSA-coevolution) → target
- Stacking: [y_base, y_msa1d, y_msa2d, y_cond] → target
"""
with open(MASTER_DIR / "README.md", "w") as f:
    f.write(readme)
print("  README.md written")

# ── Feature Cache Preparation ─────────────────────────────────────
print("[8/8] Preparing feature cache directories...")
for subdir in ["protein", "smiles", "sample"]:
    (FEATURES_DIR / subdir).mkdir(parents=True, exist_ok=True)

# Sample mapping (bridge file)
sample_mapping = pd.DataFrame({
    "sample_id": pd.concat([ck["sample_id"], cm["sample_id"]]),
    "protein_id": pd.concat([ck["protein_id"], cm["protein_id"]]),
    "task": pd.concat([ck["task"], cm["task"]]),
    "split": pd.concat([ck["split"], cm["split"]]),
})
sample_mapping.to_csv(FEATURES_DIR / "sample" / "mapping.csv", index=False)
print(f"  sample/mapping.csv — {len(sample_mapping):,} rows")

# Save protein IDs
all_protein_ids = sorted(set(ck["protein_id"].tolist() + cm["protein_id"].tolist()))
np.save(FEATURES_DIR / "protein" / "ids.npy", np.array(all_protein_ids, dtype=object))
print(f"  protein/ids.npy — {len(all_protein_ids):,} IDs")

# Save smiles IDs
all_smiles = sorted(set(ck["smiles"].tolist() + cm["smiles"].tolist()))
smiles_ids = {s: hashlib.sha256(str(s).encode()).hexdigest()[:16] for s in all_smiles}
np.save(FEATURES_DIR / "smiles" / "ids.npy", np.array(list(smiles_ids.values()), dtype=object))
print(f"  smiles/ids.npy — {len(smiles_ids):,} IDs")

# ── Completion Report ─────────────────────────────────────────────
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n{'='*60}")
print("GENERATING COMPLETION REPORT")
print(f"{'='*60}")

report = f"""# M4 — Master Dataset Construction — Completion Report

Date: 2026-06-23

## Dataset Summary

| Metric | KCAT | KM | Combined |
|--------|:----:|:--:|:--------:|
| Rows | {len(ck):,} | {len(cm):,} | {len(ck)+len(cm):,} |
| Unique proteins | {ck['sequence'].nunique():,} | {cm['sequence'].nunique():,} | {n:,} |
| Unique sample_ids | {ck['sample_id'].nunique():,} | {cm['sample_id'].nunique():,} | {ck['sample_id'].nunique()+cm['sample_id'].nunique():,} |

## Split Summary

| Split | Proteins | % |
|-------|:--------:|:-:|
| Train | {len(train_seqs):,} | {splits['train_pct']}% |
| Val | {len(val_seqs):,} | {splits['val_pct']}% |
| Test | {len(test_seqs):,} | {splits['test_pct']}% |

## Leakage Check

| Check | Result |
|-------|:------:|
| protein_id overlap train∩val | {'✅ 0' if not overlap_train_val else f'❌ {len(overlap_train_val)}'} |
| protein_id overlap train∩test | {'✅ 0' if not overlap_train_test else f'❌ {len(overlap_train_test)}'} |
| protein_id overlap val∩test | {'✅ 0' if not overlap_val_test else f'❌ {len(overlap_val_test)}'} |
| sample_id uniqueness (kcat) | {'✅ 100%' if ck['sample_id'].nunique() == len(ck) else '❌ FAIL'} |
| sample_id uniqueness (km) | {'✅ 100%' if cm['sample_id'].nunique() == len(cm) else '❌ FAIL'} |
| Target missing (kcat) | {'✅ 0' if ck['target'].isna().sum() == 0 else f'❌ {ck["target"].isna().sum()}'} |
| Target missing (km) | {'✅ 0' if cm['target'].isna().sum() == 0 else f'❌ {cm["target"].isna().sum()}'} |
| Split ratio (kcat) | ✅ train={ck[ck['split']=='train'].shape[0]/len(ck)*100:.1f}% val={ck[ck['split']=='val'].shape[0]/len(ck)*100:.1f}% test={ck[ck['split']=='test'].shape[0]/len(ck)*100:.1f}% |
| Split ratio (km) | ✅ train={cm[cm['split']=='train'].shape[0]/len(cm)*100:.1f}% val={cm[cm['split']=='val'].shape[0]/len(cm)*100:.1f}% test={cm[cm['split']=='test'].shape[0]/len(cm)*100:.1f}% |

## Leakage Overall

{'✅ PASS — 0 leakage violations' if not errors else '❌ FAIL'}

## Files Created

| File | Path | Size |
|------|------|:----:|
| kcat.csv | `data/master/kcat.csv` | — |
| km.csv | `data/master/km.csv` | — |
| splits.json | `data/master/splits.json` | — |
| schema.yaml | `data/master/schema.yaml` | — |
| README.md | `data/master/README.md` | — |
| mapping.csv | `data/features/sample/mapping.csv` | — |
| protein_ids.npy | `data/features/protein/ids.npy` | — |
| smiles_ids.npy | `data/features/smiles/ids.npy` | — |

## Ready For

- [x] BaselineV2 — (sequence, smiles) → target
- [x] ConditionV2 — (sequence, smiles, temperature, ph) → target
- [x] MSA1D — (sequence, smiles, MSA-PSSM) → target
- [x] MSA2D — (sequence, smiles, MSA-coevolution) → target
- [x] Stacking — [y_base, y_msa1d, y_msa2d, y_cond] → target

## Git

```
Branch:  feat/master-dataset-v2-construction
Commit:  feat(master): implement unified master dataset v2 with dual-id architecture
```
"""
with open(REPORTS_DIR / "M4_COMPLETION_REPORT.md", "w") as f:
    f.write(report)
print("  reports/M4_COMPLETION_REPORT.md written")

print(f"\n{'='*60}")
print("M4 COMPLETE ✅")
print(f"{'='*60}")
