"""ViralScope AI — Modular Pipeline Package.

Refactored architecture with clear class boundaries.
Each major component is a class following the ThumbnailManager pattern.
"""

from src.config_loader import ConfigLoader
from src.data_loader import DataLoader
from src.label_engine import LabelEngine
from src.thumbnail_manager import ThumbnailManager
from src.embedding_extractor import EmbeddingExtractor
from src.data_splitter import DataSplitter
from src.title_feature_extractor import TitleFeatureExtractor
from src.visual_stats_extractor import VisualStatsExtractor
from src.feature_builder import FeatureBuilder
from src.stacking_trainer import StackingTrainer, SoftVotingEnsemble
from src.model_evaluator import ModelEvaluator
from src.model_persistence import ModelPersistence
from src.pipeline import ViralScopePipeline

__all__ = [
    "ConfigLoader",
    "DataLoader",
    "LabelEngine",
    "ThumbnailManager",
    "EmbeddingExtractor",
    "DataSplitter",
    "TitleFeatureExtractor",
    "VisualStatsExtractor",
    "FeatureBuilder",
    "StackingTrainer",
    "SoftVotingEnsemble",
    "ModelEvaluator",
    "ModelPersistence",
    "ViralScopePipeline",
]
