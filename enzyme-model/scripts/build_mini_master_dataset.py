#!/usr/bin/env python3
"""Build Mini Master Dataset — 500 unique proteins sampled from full Master Dataset."""
import random, sys
from pathlib import Path
import pandas as pd
import numpy as np

SEED = 42
N_PROTEINS = 500
random.seed(SEED)

MASTER_DIR = Path("data/master")
MINI_DIR = Path("data/master_mini")
REPORTS_DIR = Path("reports")
MINI_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("M4.5 — MINI MASTER DATASET (500 proteins)")
print("=" * 60)

# 1. Load full Master Dataset
ck = pd.read_csv(MASTER_DIR / "kcat.csv")
cm = pd.read_csv(MASTER_DIR / "km.csv")
print(f"\nFull dataset: kcat={len(ck):,} rows, km={len(cm):,} rows")

# 2. Collect all unique protein_ids
all_pids = sorted(set(ck["protein_id"].tolist() + cm["protein_id"].tolist()))
print(f"Total unique proteins: {len(all_pids):,}")

# 3. Sample 500
selected_pids = set(random.sample(all_pids, N_PROTEINS))
print(f"Selected: {len(selected_pids)} proteins")

# 4. Filter all samples belonging to those proteins
ck_mini = ck[ck["protein_id"].isin(selected_pids)].copy()
cm_mini = cm[cm["protein_id"].isin(selected_pids)].copy()
print(f"Mini kcat: {len(ck_mini):,} rows, Mini km: {len(cm_mini):,} rows")

# 5. Write
ck_mini.to_csv(MINI_DIR / "kcat.csv", index=False)
cm_mini.to_csv(MINI_DIR / "km.csv", index=False)
print(f"Written: {MINI_DIR}/kcat.csv, {MINI_DIR}/km.csv")

# 6. Leakage Audit
print(f"\n{'='*60}")
print("LEAKAGE AUDIT")
print(f"{'='*60}")
errors = []
for ds_name, ds in [("kcat", ck_mini), ("km", cm_mini)]:
    for s in ["train", "val", "test"]:
        pids_in_split = set(ds[ds["split"] == s]["protein_id"])
        for other in ["train", "val", "test"]:
            if other <= s:
                continue
            opids = set(ds[ds["split"] == other]["protein_id"])
            overlap = pids_in_split & opids
            if overlap:
                errors.append(f"LEAK {ds_name}: {s}∩{other} = {len(overlap)} proteins")
if errors:
    print("❌ FAILED:")
    for e in errors:
        print(f"  {e}")
    sys.exit(1)
else:
    print("✅ No protein leakage across train/val/test splits")

# 7. Split Ratio
print(f"\n{'='*60}")
print("SPLIT RATIO")
print(f"{'='*60}")
for ds_name, ds in [("kcat", ck_mini), ("km", cm_mini)]:
    r = ds["split"].value_counts(normalize=True) * 100
    print(f"  {ds_name}: train={r.get('train',0):.1f}% val={r.get('val',0):.1f}% test={r.get('test',0):.1f}%")

# 8. Distribution Comparison
print(f"\n{'='*60}")
print("DISTRIBUTION COMPARISON (target)")
print(f"{'='*60}")
for ds_name, full, mini in [("kcat", ck, ck_mini), ("km", cm, cm_mini)]:
    f_t = full["target"]
    m_t = mini["target"]
    dev_mean = abs(f_t.mean() - m_t.mean()) / abs(f_t.mean()) * 100 if f_t.mean() != 0 else 0
    dev_std = abs(f_t.std() - m_t.std()) / f_t.std() * 100 if f_t.std() != 0 else 0
    print(f"  {ds_name}:")
    print(f"    Full:  mean={f_t.mean():.4f}, std={f_t.std():.4f}, med={f_t.median():.4f}")
    print(f"    Mini:  mean={m_t.mean():.4f}, std={m_t.std():.4f}, med={m_t.median():.4f}")
    print(f"    Δmean: {dev_mean:.2f}% {'✅' if dev_mean < 10 else '⚠️ >10%'}")
    print(f"    Δstd:  {dev_std:.2f}% {'✅' if dev_std < 10 else '⚠️ >10%'}")

# 9. Coverage Report
print(f"\n{'='*60}")
print("COVERAGE REPORT")
print(f"{'='*60}")
n_train = sum(1 for p in selected_pids if p in set(ck_mini[ck_mini["split"]=="train"]["protein_id"]) or p in set(cm_mini[cm_mini["split"]=="train"]["protein_id"]))
n_val = sum(1 for p in selected_pids if p in set(ck_mini[ck_mini["split"]=="val"]["protein_id"]) or p in set(cm_mini[cm_mini["split"]=="val"]["protein_id"]))
n_test = sum(1 for p in selected_pids if p in set(ck_mini[ck_mini["split"]=="test"]["protein_id"]) or p in set(cm_mini[cm_mini["split"]=="test"]["protein_id"]))
# Better way:
train_pids = set(ck_mini[ck_mini["split"]=="train"]["protein_id"]) | set(cm_mini[cm_mini["split"]=="train"]["protein_id"])
val_pids = set(ck_mini[ck_mini["split"]=="val"]["protein_id"]) | set(cm_mini[cm_mini["split"]=="val"]["protein_id"])
test_pids = set(ck_mini[ck_mini["split"]=="test"]["protein_id"]) | set(cm_mini[cm_mini["split"]=="test"]["protein_id"])

print(f"""
  Selected Proteins:     {len(selected_pids):,}
  kcat Samples:          {len(ck_mini):,}
  km Samples:            {len(cm_mini):,}
  Total Samples:         {len(ck_mini)+len(cm_mini):,}
  Avg Samples / Protein: {(len(ck_mini)+len(cm_mini))/len(selected_pids):.1f}

  Train Proteins:        {len(train_pids)} ({len(train_pids)/len(selected_pids)*100:.1f}%)
  Val Proteins:          {len(val_pids)} ({len(val_pids)/len(selected_pids)*100:.1f}%)
  Test Proteins:         {len(test_pids)} ({len(test_pids)/len(selected_pids)*100:.1f}%)

  Mini Master Dataset READY ✅
""")

# 10. Report
report = f"""# Mini Master Dataset — 500 Protein Sample

Date: 2026-06-23
Source: `data/master/`
Method: Random sample of 500 unique protein_ids (seed=42), all rows preserved

## Dataset Statistics

| Metric | kcat | km | Total |
|--------|:----:|:--:|:-----:|
| Selected Proteins | {ck_mini['protein_id'].nunique()} | {cm_mini['protein_id'].nunique()} | {len(selected_pids)} |
| Samples | {len(ck_mini):,} | {len(cm_mini):,} | {len(ck_mini)+len(cm_mini):,} |
| Avg Samples / Protein | {len(ck_mini)/max(ck_mini['protein_id'].nunique(),1):.1f} | {len(cm_mini)/max(cm_mini['protein_id'].nunique(),1):.1f} | {(len(ck_mini)+len(cm_mini))/len(selected_pids):.1f} |

## Split

| Split | kcat | km |
|-------|:----:|:--:|
| Train | {ck_mini[ck_mini['split']=='train'].shape[0]:,} ({ck_mini[ck_mini['split']=='train'].shape[0]/len(ck_mini)*100:.1f}%) | {cm_mini[cm_mini['split']=='train'].shape[0]:,} ({cm_mini[cm_mini['split']=='train'].shape[0]/len(cm_mini)*100:.1f}%) |
| Val | {ck_mini[ck_mini['split']=='val'].shape[0]:,} ({ck_mini[ck_mini['split']=='val'].shape[0]/len(ck_mini)*100:.1f}%) | {cm_mini[cm_mini['split']=='val'].shape[0]:,} ({cm_mini[cm_mini['split']=='val'].shape[0]/len(cm_mini)*100:.1f}%) |
| Test | {ck_mini[ck_mini['split']=='test'].shape[0]:,} ({ck_mini[ck_mini['split']=='test'].shape[0]/len(ck_mini)*100:.1f}%) | {cm_mini[cm_mini['split']=='test'].shape[0]:,} ({cm_mini[cm_mini['split']=='test'].shape[0]/len(cm_mini)*100:.1f}%) |

## Leakage Audit

| Check | Result |
|-------|:------:|
| Same protein in train∩val | ✅ 0 (PASS) |
| Same protein in train∩test | ✅ 0 (PASS) |
| Same protein in val∩test | ✅ 0 (PASS) |

## Distribution Comparison

### kcat target

| Metric | Full | Mini | Δ |
|--------|:----:|:----:|:-:|
| mean | {ck['target'].mean():.4f} | {ck_mini['target'].mean():.4f} | {abs(ck['target'].mean()-ck_mini['target'].mean())/abs(ck['target'].mean())*100:.1f}% |
| std | {ck['target'].std():.4f} | {ck_mini['target'].std():.4f} | {abs(ck['target'].std()-ck_mini['target'].std())/ck['target'].std()*100:.1f}% |
| median | {ck['target'].median():.4f} | {ck_mini['target'].median():.4f} | — |

### km target

| Metric | Full | Mini | Δ |
|--------|:----:|:----:|:-:|
| mean | {cm['target'].mean():.4f} | {cm_mini['target'].mean():.4f} | {abs(cm['target'].mean()-cm_mini['target'].mean())/abs(cm['target'].mean())*100:.1f}% |
| std | {cm['target'].std():.4f} | {cm_mini['target'].std():.4f} | {abs(cm['target'].std()-cm_mini['target'].std())/cm['target'].std()*100:.1f}% |
| median | {cm['target'].median():.4f} | {cm_mini['target'].median():.4f} | — |

## Files

- `data/master_mini/kcat.csv`
- `data/master_mini/km.csv`

## Training Commands

```bash
# BaselineV2
python train/train_baseline_v2.py --task kcat --dataset mini
python train/train_baseline_v2.py --task km --dataset mini

# ConditionV2
python train/train_condition_v2.py --task kcat --dataset mini
python train/train_condition_v2.py --task km --dataset mini

# MSA1D
python train/train_msa1d.py --task kcat --dataset mini
python train/train_msa1d.py --task km --dataset mini
```

## Verdict

```
MINI_MASTER_DATASET_500P = READY ✅
"""
with open(REPORTS_DIR / "MINI_MASTER_500P_REPORT.md", "w") as f:
    f.write(report)
print(f"Report: {REPORTS_DIR}/MINI_MASTER_500P_REPORT.md")
