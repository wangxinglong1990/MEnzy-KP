# feature_extractors — High-level feature extraction interface
#
#  Implemented:
#    ProteinFeatureExtractor    — ESMC-300M → (1, 960)
#    SmilesFeatureExtractor     — TrfmSeq2seq → (1, 1024)
#    CombinedFeatureExtractor   — orchestrates both → (1, 1984)
#
#  Planned (placeholder stubs):
#    MSA2DFeatureExtractor      — MSA → co-evolution matrix → (L, L)
#    ConditionFeatureExtractor  — pH/temp → (1, 2)

from core.feature_extractors.protein_feature_extractor import ProteinFeatureExtractor
from core.feature_extractors.smiles_feature_extractor import SmilesFeatureExtractor
from core.feature_extractors.combined_feature_extractor import CombinedFeatureExtractor
from core.feature_extractors.condition_feature_extractor import ConditionFeatureExtractor
from core.feature_extractors.msa1d_feature_extractor import MSA1DFeatureExtractor
