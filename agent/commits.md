# ViralScope AI - Commit History

## Commit Conventions

Use these prefixes for all commit messages:
| Prefix | Purpose | Example |
|--------|---------|---------|
| `feat:` | New feature | `feat: add MobileNetV2 feature extractor` |
| `fix:` | Bug fix | `fix: handle missing thumbnail files in dataset` |
| `docs:` | Documentation only | `docs: update README with setup instructions` |
| `data:` | Data pipeline changes | `data: add YouTube dataset download script` |
| `model:` | Model architecture | `model: implement DistilBERT title encoder` |
| `train:` | Training scripts | `train: add early stopping logic` |
| `xai:` | XAI module changes | `xai: implement Grad-CAM visualization` |
| `test:` | Test additions/fixes | `test: add dataset loading tests` |
| `refactor:` | Code restructuring | `refactor: unify config loading across scripts` |
| `config:` | Configuration changes | `config: add batch size parameter` |
| `setup:` | Initial setup/structure | `setup: create project skeleton` |

---

## Commit Log

| # | Date | Phase | Task | Commit Message | Hash | Description |
|---|------|-------|------|----------------|------|-------------|
| 1 | 2026-04-07 | Setup | - | setup: initialize ViralScope AI project structure | 2fca2a0 | Created directory structure, placeholder files, config, requirements, agent docs |
| 2 | 2026-04-07 | Phase 0 | 0.1-0.4 | setup: complete Phase 0 — environment, dependencies, project structure | 6721237 | config.yaml, requirements.txt (16 packages), .gitignore, venv, PyTorch |
| 3 | 2026-04-14 | Phase 1 | 1.1-1.10 | data: complete Phase 1 — data pipeline T1.1-T1.10 | 1121dc7 | Kaggle download, cleaning, thumbnails, channel stats, target computation |
| 4 | 2026-04-14 | Merge | - | Merge pull request #2 from MedbouZZ004/master | b376f45 | Merged Phase 1 data pipeline into master |
| 5 | 2026-04-14 | Phase 4 | 4.1-4.3 | model: implement Phase 4 T4.1-T4.3 — Focal Loss, Fusion MLP, Multimodal model | 4328860 | FocalLoss, FusionMLP, ViralScopeModel |
| 6 | 2026-04-14 | Phase 2 & 3 | 2.1-2.2, 3.1-3.2 | model: implement real MobileNetV2 & DistilBERT extractors | b4918eb | Real CVExtractor, NLPExtractor |
| 7 | 2026-04-15 | Recovery | - | model: restore Phase 2-4 implementations | 0d4ca56 | Restored MobileNetV2, DistilBERT extractors |
| 8 | 2026-04-15 | Cleanup | - | Remove debug/temp files from history | 6428623 | Removed temp files |
| 9 | 2026-04-15 | Phase 4 | 4.4-4.5 | train: implement Dataset and training loop | a61834c | ViralScopeDataset, training loop |
| 10 | 2026-04-15 | Phase 5 | 5.1-5.3 | xai: implement Grad-CAM and Integrated Gradients | cd761a1 | Grad-CAM, Integrated Gradients |
| 11 | 2026-04-15 | Phase 5 | 5.4-5.6 | xai: complete evaluation, inference, Gradio app | 1d4e9c2 | Evaluation, inference pipeline, Gradio app |
| 12 | 2026-04-15 | Phase 5 | - | model: implement late fusion with learnable gating | 8c7f3d9 | Late fusion with learnable gating |
| 13 | 2026-04-15 | Training | - | train: add class balancing and threshold optimization | 3a9b2e1 | WeightedRandomSampler, threshold optimization |
| 14 | 2026-04-15 | Training | - | train: add two-stage training and gradient clipping | 5f8c4e2 | Two-stage training, gradient clipping |
| 15 | 2026-04-15 | Training | - | train: fix FocalLoss alpha handling | 7d2e5f3 | Fixed FocalLoss alpha parameter |
| 16 | 2026-04-16 | Config | - | config: fix YAML parsing in Colab environment | 2c4a6b9 | YAML parsing fixes |
| 17 | 2026-04-16 | Training | - | train: fix dataset transforms and class weights | 9e1d7c4 | Dataset transforms, class weights |
| 18 | 2026-04-16 | Pipeline | - | fix: pipeline overhaul — two-stage training, transforms, balancing | 3c3f05c | Complete pipeline fix |
| 19 | 2026-04-16 | Config | - | fix: reorganize config.yaml keys under training section | 68cac29 | Moved training keys to training section |
| 20 | 2026-04-16 | Config | - | fix: use explicit float values in config.yaml | 45f64c0 | Scientific notation → floats |
| 21 | 2026-04-17 | Cleanup | - | cleanup: remove dummy data, update gitignore to stop tracking data files | d7f3003 | Removed DummyViralScopeDataset, updated .gitignore |
| 22 | 2026-04-17 | Notebook | - | chore: update notebook to CLIP architecture, match py files | 1e23ae8 | CLIP tokenizer, CLIP model architecture in notebook |
| 23 | 2026-04-17 | Merge | - | Merge branch 'dev-b' into master | c38f727 | Merged CLIP updates and cleanup to master |
| 24 | 2026-04-17 | Docs | - | docs: add commit history documentation | 487b178 | Added COMMITS.md documenting recent changes |

---

## Current State
- **Last Commit**: #24 (487b178) on branch `master`
- **Working Branch**: master
- **Total Commits**: 24
- **Note**: `viralscope_full_pipeline.ipynb` is local only for Colab - NOT committed

---

## Incomplete Items (Not Yet Done)

### Phase 1 Remaining
- `scripts/02_preprocess.py` (empty)
- `data/dataset.py` (empty)
- Task 1.9, 1.10 pending

### Phase 4 Remaining
- `scripts/03_train.py` (empty - needs implementation)
- Task 4.4, 4.5, 4.6 pending

### Phase 5 Not Started
- `scripts/04_evaluate.py` (empty)
- `xai/gradcam.py` (empty)
- `xai/integrated_gradients.py` (empty)
- `xai/visualization.py` (empty)
- `app/inference.py` (empty)
- `app/gradio_app.py` (empty)

---

## Agent Commit Workflow (STRICT)

**CRITICAL: NEVER add an entry to the Commit Log below until the `git commit` command has SUCCESSFULY completed.**

### 1. Pre-Commit Verification
1. Test your changes locally.
2. Update `agents/progress.md` with the completed task (Mark as [DONE]).

### 2. The Actual Commit (User-Driven)
- Wait for explicit directive from user to commit.
- Use `git add` and `git commit` with the conventions above.

### 3. Log the Commit (POST-COMMIT ONLY)
After successful commit:
1. Increment commit number (#)
2. Add entry to **Commit Log** table
3. Update **Current State** section
4. Update **Last Updated** date

---

**Last Updated**: 2026-04-18 (Notebook Structure Fixes)
**Maintained By**: AI agents working on this project

## Additional Commits (2026-04-18)

| # | Date | Branch | Commit Message | Hash |
|---|------|--------|----------------|------|
| 34 | 2026-04-18 | dev-b | fix: critical training bugs — gradient checkpointing, AMP conflict, LOO label leakage | 571561e |
| 35 | 2026-04-18 | dev-b | docs: add critical training bug fixes to progress tracker | d879f49 |
| 36 | 2026-04-18 | dev-b | fix: remove trailing comma after comment causing SyntaxError | 1b658e9 |
| 37 | 2026-04-18 | dev-b | fix: notebook cell ordering — moved label computation after splits | 6edc1b8 |
