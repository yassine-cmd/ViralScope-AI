# ViralScope AI — Task Division Between Developer A and Developer B

> Phase 0 is already complete. Both developers can start immediately.

---

## Pre-Start (Both Developers — 30 Minutes, Day 1)

Before any task execution, both developers agree on the following contracts:

- **Tensor shapes**: Both CV and NLP extractors output L2-normalized `(batch, 512)` float32 tensors in a shared latent space.
- **Interaction Input**: Fusion head accepts 2049-dim input `[img, txt, diff, prod, sim]`.
- **CSV columns**: `video_id, title, views, channel_id, is_viral`
- **NLP tensor shapes**: `input_ids.pt` and `attention_masks.pt` are `(N, 77)` int64 (CLIP sequence length).
- **Thumbnail path convention**: `data/raw/thumbnails/{video_id}.jpg`
- **Split format**: 1D int64 index tensors, zero channel_id overlap guaranteed.
- **config.yaml ownership**: Developer B owns and edits it. Developer A reports needed changes; Developer B applies them.
- **progress.md table bug**: Fix the blank line that splits the Phase Status table into two blocks before any agent reads it.

Developer A fills the two remaining contract items — the exact Grad-CAM hook layer and the Integrated Gradients embedding layer — as part of their T2.2 and T3.2 completion reports. Developer B waits for those announcements before hardcoding any layer names in T5.2 and T5.3.

---

## Developer A — Data, Pipelines & Extractors

**Scope**: Phases 1, 2, 3 and Task 4.4  
**Agent context**: Only grep for `TASK 1.`, `TASK 2.`, `TASK 3.`, `TASK 4.4` — never load Phase 4 or 5 tasks

### Phase 1 — Data Engineering

| Task | Description | Key Output |
|------|-------------|------------|
| T1.1 | Download YouTube Trending Dataset from Kaggle — **US + GB + CA only** (English-speaking markets) | `data/raw/trending.csv` (from USvideos.csv, GBvideos.csv, CAvideos.csv) |
| T1.2 | Deduplicate and filter invalid rows | `data/processed/clean_dataset.csv` |
| T1.3 | Download thumbnails from YouTube CDN via ThreadPoolExecutor | `data/raw/thumbnails/*.jpg` + `thumbnail_manifest.csv` |
| T1.4 | Filter dataset to videos with valid thumbnails only | `data/processed/final_dataset.csv` |
| T1.5 | Compute per-channel Leave-One-Out view averages | `data/processed/channel_averages.csv` |
| T1.6 | Compute viral multiplier and binary label; drop channels with < 3 videos | `data/processed/labeled_dataset.csv` |
| T1.7 | Pre-tokenize all titles with **CLIPTokenizer** (max_len=77); hash and lock the CSV | `data/tensors/input_ids.pt`, `attention_masks.pt`, `tokenizer_metadata.json` |
| T1.9 | Grouped Train/Val/Test split by `channel_id` using GroupShuffleSplit | `data/splits/train_indices.pt`, `val_indices.pt`, `test_indices.pt`, `split_report.json` |
| T1.10 | Compute class weights from training split | `data/processed/class_weights.pt` |

> ⚠️ After T1.7: `labeled_dataset.csv` is permanently locked. Do not re-filter, re-sort, or modify it. Any change breaks the hash check in `create_dataloaders()` and will crash training.

---

### Phase 2 — Computer Vision Model

| Task | Description | Key Output |
|------|-------------|------------|
| T2.1 | Build CLIP Vision extractor; output shape `(batch, 512)` | `models/cv_extractor.py` |
| T2.2 | Validate extractor on real thumbnail batch; **document CLIP ViT target layer for Grad-CAM** | Validation report + contract section filled |

---

### Phase 3 — NLP Model

| Task | Description | Key Output |
|------|-------------|------------|
| T3.1 | Build CLIP Text extractor; output shape `(batch, 512)` | `models/nlp_extractor.py` |
| T3.2 | Validate extractor on titles; **verify L2-normalization and alignment with vision features** | Validation report + contract section filled |

---

### Task 4.4 — PyTorch Dataset & DataLoader (Assigned to Developer A)

| Task | Description | Key Output |
|------|-------------|------------|
| T4.4 | Build `ViralScopeDataset` with on-the-fly PIL image loading and `create_dataloaders()` factory | `data/dataset.py` |

**Why Developer A owns this**: The Dataset class directly references Developer A's CSV column names, tensor file paths, split index format, thumbnail folder structure, and the dataset hash from `tokenizer_metadata.json`. Developer B cannot safely implement it without knowing these internals exactly. Developer A runs the DataLoader unit test and confirms batch shapes before handing off.

---

### Developer A — Full Deliverables

```
data/raw/trending.csv
data/raw/thumbnails/*.jpg
data/processed/clean_dataset.csv
data/processed/thumbnail_manifest.csv
data/processed/final_dataset.csv
data/processed/channel_averages.csv
data/processed/labeled_dataset.csv         ← IMMUTABLE after T1.7
data/processed/class_weights.pt
data/tensors/input_ids.pt                  shape: (N, 64), int64
data/tensors/attention_masks.pt            shape: (N, 64), int64
data/tensors/tokenizer_metadata.json       includes dataset hash
data/splits/train_indices.pt
data/splits/val_indices.pt
data/splits/test_indices.pt
data/splits/split_report.json
models/cv_extractor.py                     output: (B, 1280) float32
models/nlp_extractor.py                    output: (B, 768)  float32
data/dataset.py                            ViralScopeDataset + create_dataloaders()
Grad-CAM target layer name                 announced after T2.2
IG embedding layer name                    announced after T3.2
```

---

## Developer B — Architecture, Training, XAI & Deployment

**Scope**: Phase 4 (T4.1, T4.2, T4.3, T4.5, T4.6) and Phase 5 (T5.1, T5.2, T5.3, T5.4, T5.6). Does **not** own T4.4.  
**Agent context**: Only grep for `TASK 4.` (excluding 4.4) and `TASK 5.` — never load Phase 1, 2, or 3 tasks

### Phase 4 — Late Fusion & Training

| Task | Description | Key Output | Dependency |
|------|-------------|------------|------------|
| T4.1 | [DEPRECATED] Focal Loss (use BCEWithLogitsLoss for CLIP fine-tuning) | `models/losses.py` | None |
| T4.2 | Build Interaction Fusion head: `[img, txt, diff, prod, sim]` $\to$ 256 $\to$ 64 $\to$ 1 | `models/fusion_model.py` | None |
| T4.3 | Assemble `ViralScopeModel` using shared CLIP backbone splitting | `models/multimodal.py` | T4.2 |
| T4.5 | Two-stage training: head warmup $\to$ backbone fine-tune with staggered LRs | `scripts/03_train.py`, `models/best_model.pt` | All A deliverables |
| T4.6 | [REPLACED] Integrated into T4.5 (two-stage unfreezing) | Addition to `scripts/03_train.py` | T4.5 complete |

> T4.1, T4.2, and T4.3 can be built immediately using dummy tensor shapes `[B, 1280]` and `[B, 768]`. No real data is needed.

---

### Phase 5 — Evaluation, XAI & Deployment

| Task | Description | Key Output | Dependency |
|------|-------------|------------|------------|
| T5.1 | Evaluate model on test set: accuracy, F1, AUC-ROC, PR-AUC, confusion matrix | `scripts/04_evaluate.py`, `results/metrics.json` | T4.5 complete |
| T5.2 | Implement Patch-based Grad-CAM on CLIP ViT backbone | `xai/gradcam.py` | T4.5 complete |
| T5.3 | Implement Integrated Gradients (Captum) on CLIP Text embeddings | `xai/integrated_gradients.py`, `xai/visualization.py` | T4.5 complete |
| T5.4 | Build inference pipeline: load model, transform inputs, run predict, call XAI | `app/inference.py` | T4.5 + T5.2 + T5.3 |
| T5.6 | Build Gradio interface: thumbnail upload, title input, viral score output, heatmap, token attribution | `app/gradio_app.py` | T5.4 complete |

> For T5.2 and T5.3: do not hardcode layer names. Wait for Developer A's completion announcement after T2.2 and T3.2, then use the exact layer paths provided.

---

### Developer B — Full Deliverables

```
models/losses.py
models/fusion_model.py
models/multimodal.py
scripts/03_train.py
models/best_model.pt
scripts/04_evaluate.py
results/metrics.json
xai/gradcam.py
xai/integrated_gradients.py
xai/visualization.py
app/inference.py
app/gradio_app.py
```

---

## Integration Gate

Before Developer B starts T4.5, the following must all be true:

- [ ] Developer A has announced completion of T4.4 and confirmed DataLoader batch shapes pass the unit test
- [ ] `data/tensors/tokenizer_metadata.json` contains a `dataset_hash` field
- [ ] `data/splits/split_report.json` confirms zero channel_id overlap across all splits
- [ ] Developer A has filled both contract items (Grad-CAM layer + IG layer)
- [ ] `tests/test_integration.py` runs clean (CSV schema, tensor alignment, hash match, split leakage, extractor shapes, DataLoader batch shapes)

Training does not start until all boxes are checked.

---

## Concurrency Timeline

```
Day 1       Developer A: T1.1 → T1.2 → T1.3 (slow, CDN downloads)
            Developer B: T4.1 → T4.2 → T4.3 (fast, no data needed)

Day 2-3     Developer A: T1.4 → T1.5 → T1.6 → T1.7 → T1.9 → T1.10
            Developer B: T5.2 stub (wait for layer name) → T5.3 stub

Day 3-4     Developer A: T2.1 → T2.2 (announce Grad-CAM layer) → T3.1 → T3.2 (announce IG layer) → T4.4
            Developer B: Receive announcements → finalize T5.2 and T5.3

Day 4-5     INTEGRATION GATE: run test_integration.py
            Developer B: T4.5 → T4.6 → T5.1 → T5.4 → T5.6
```
