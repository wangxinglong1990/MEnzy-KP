#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
DATA_DIR = PROJECT_DIR / "data"
MODELS_DIR = PROJECT_DIR / "models"
SMILES_TRANSFORMER_DIR = PROJECT_DIR / "SMILES_Transform"
SMILES_TRANSFORMER_CHECKPOINT = SMILES_TRANSFORMER_DIR / "trfm_12_23000.pkl"
PROTEIN_ESMC_WEIGHTS_PATH = DATA_DIR / "weights" / "esmc_300m_2024_12_v0.pth"
PROTEIN_ESMC_MODEL_NAME = "esmc_300m"

UNIFIED_DATASET_PATH = DATA_DIR / "kcat-over-Km-data_0.4simi-10fold.csv"

# enzyme-model (full) baseline models — ExtraTreesRegressor, 1984-dim features
# Trained on expanded dataset; retrained from enzyme-model-full artifacts.
KM_MODEL_PATH = PROJECT_DIR / "enzyme-model" / "artifacts" / "baseline" / "km" / "km_predictor.joblib"
KCAT_MODEL_PATH = PROJECT_DIR / "enzyme-model" / "artifacts" / "baseline" / "kcat" / "kcat_predictor.joblib"

