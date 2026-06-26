# Enzyme Model

Predictive modeling for enzyme kinetics and properties.

## Structure

```
enzyme-model/
├── data/               # Dataset files
├── core/               # Core utilities
│   ├── encoders/       # Encoding modules
│   ├── feature_extractors/  # Feature extraction
│   └── shared_smiles/  # Shared SMILES utilities
├── models/             # Model implementations
│   ├── baseline/       # Baseline model
│   ├── msa1d/          # MSA1D model (Model2)
│   ├── msa2d/          # MSA2D model (Model3)
│   ├── condition/      # Condition model (Model4)
│   └── stacking/       # Stacking ensemble
├── train/              # Training scripts
├── evaluate/           # Evaluation scripts
├── experiments/        # Experiment tracking
├── configs/            # Configuration files
└── artifacts/          # Model outputs
```
