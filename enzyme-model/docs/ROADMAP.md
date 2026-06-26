# Enzyme Model — Research Roadmap

## Model Overview

| Model | Features | Algorithm | Target |
|-------|----------|-----------|--------|
| Baseline (Model1) | ESM-C + SMILES | ExtraTrees | log10(kcat), log10(km) |
| MSA1D (Model2) | ESM-C + SMILES + MSA1D | XGBoost / LightGBM | log10(kcat), log10(km) |
| MSA2D (Model3) | ESM-C + SMILES + MSA2D | CNN / MLP | log10(kcat), log10(km) |
| Condition (Model4) | ESM-C + SMILES + Condition | LightGBM / MLP | log10(kcat), log10(km) |
| Stacking | [y_base, y_msa1d, y_msa2d, y_condition] | Ridge / XGBoost / ExtraTrees | log10(kcat), log10(km) |

## Architecture

```
models/
├── baseline/     # Model1: ExtraTrees
├── msa1d/        # Model2: XGBoost / LightGBM
├── msa2d/        # Model3: CNN / MLP
├── condition/    # Model4: LightGBM / MLP
└── stacking/     # Ensemble of all above
```

## Milestones

- [x] Baseline (Model1) — 已完成，R²(kcat)=0.67, R²(km)=0.52
- [ ] MSA1D (Model2)
- [ ] MSA2D (Model3)
- [ ] Condition (Model4)
- [ ] Stacking
