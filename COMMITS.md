# Commit History

## Latest (c38f727 - Merge branch 'dev-b')
- Merged dev-b into master
- Includes: CLIP notebook updates, dummy data cleanup, gitignore updates

---

## 1e23ae8 - chore: update notebook to CLIP architecture, match py files
**Date:** 2026-04-17

**Changes:**
- Updated notebook Cell 10: Changed tokenization from DistilBERT to CLIP Tokenizer
- Updated notebook Cell 11: Replaced MobileNetV2/DistilBERT architecture with CLIP-based model
  - `CVExtractor`: CLIP vision encoder
  - `NLPExtractor`: CLIP text encoder  
  - `FusionMLP`: Cross-modal fusion with interaction features (diff, product, cosine similarity)
- Updated notebook Cell 12: Changed image normalization to use CLIP mean/std values from config
- Updated notebook Cell 14: Model initialization now uses CLIP-based ViralScopeModel
- Removed duplicate model architecture cell

**Files modified:**
- `viralscope_full_pipeline.ipynb`
- `config.yaml` (CLIP config)
- `models/cv_extractor.py`
- `models/fusion_model.py`
- `models/multimodal.py`
- `models/nlp_extractor.py`

---

## d7f3003 - cleanup: remove dummy data, update gitignore to stop tracking data files
**Date:** 2026-04-17

**Changes:**
- Removed `DummyViralScopeDataset` class from `data/dataset.py`
- Removed dummy data imports and logic from `scripts/03_train.py`
- Removed `use_dummy_data` parameter from function signature and CLI arguments
- Updated `.gitignore` to comprehensively ignore data files (`data/`, `data/**/*.csv`, etc.)
- Ran `git rm -r --cached data/` to stop tracking data files (kept locally)
- Restored `data/__init__.py` and `data/dataset.py` from staged deletion (they are code, not data)

**Files modified:**
- `data/dataset.py`
- `scripts/03_train.py`
- `.gitignore`

---

## Earlier Commits

### d8a92b0 - Merge branch 'dev-b'
- Fix: add drop_last=True to train_loader in scripts

### 2ea3a76 - merge: resolve conflicts with remote master

### 830bf34 - merge: sync dev-b into master to update config.yaml and pipeline

### 45f64c0 - fix: use explicit float values instead of scientific notation

### 68cac29 - fix: reorganize config.yaml keys under training section

### 740b42a - Merge pull request #9 from yassine-cmd/dev-b
- Fix: overhaul training pipeline to fix degenerate results

### 3c3f05c - fix: overhaul training pipeline to fix degenerate results

### ceb0cfa - Merge pull request #8 from yassine-cmd/dev-b

### a61834c - train: implement Dataset and training loop (Task 4.4-4.5)