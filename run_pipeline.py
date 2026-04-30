#!/usr/bin/env python
"""Run the full ViralScope AI pipeline.

Usage:
    python run_pipeline.py [--config config.yaml]

This script executes the complete pipeline:
    1. Load & validate configuration
    2. Load & clean CSV data
    3. Compute LOO viral labels
    4. Download thumbnails
    5. Extract SigLIP embeddings
    6. Split data (stratified)
    7. Extract visual stats
    8. Build feature matrix
    9. Train stacking ensemble
    10. Evaluate on test set
    11. Save model & results
"""

import argparse
import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.getcwd())

import random
import numpy as np
import torch

# Reproducibility
_GLOBAL_SEED = 42
random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(_GLOBAL_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Run ViralScope AI pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    from src.pipeline import ViralScopePipeline

    pipeline = ViralScopePipeline(config_path=args.config)
    pipeline.run()


if __name__ == "__main__":
    main()
