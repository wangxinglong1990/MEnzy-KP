# Master Dataset V2

## Summary

| Metric | KCAT | KM | Total |
|--------|:----:|:--:|:-----:|
| Rows | 23,197 | 41,174 | 64,371 |
| Unique proteins | 7,183 | 12,355 | 12,848 |
| Unique sample_ids | 23,197 | 41,174 | 64,371 |

## Schema

See [schema.yaml](schema.yaml) for full field definitions.

## Split

- Strategy: protein-aware group-by-sequence
- Train: 80.0%
- Val:   10.0%
- Test:  10.0%

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
