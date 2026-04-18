# VIRALSCOPE AI - TASK DECOMPOSITION FOR AI AGENT EXECUTION

This document breaks down the VIRALSCOPE AI specification into discrete, testable tasks that any AI agent can execute independently.

---

## ⛔ SCOPE BOUNDARIES — READ BEFORE EXECUTING

**This is an MVP (Minimum Viable Product). NOT a production-grade system.**

For EVERY task, the following rules apply. **Do NOT deviate from them:**

### ❌ Explicitly OUT of Scope (DO NOT Add)
- **No** Docker containers, Kubernetes, or containerization
- **No** API servers (FastAPI, Flask, REST endpoints) — Gradio runs locally
- **No** authentication, user management, rate limiting, or API keys
- **No** databases (SQLite, PostgreSQL, Redis) — everything is file-based
- **No** monitoring infrastructure (MLflow, Weights & Biases, Prometheus, Grafana)
- **No** CI/CD pipelines, GitHub Actions, or automated deployment
- **No** model versioning, model registries, or A/B testing
- **No** logging frameworks beyond `print()` statements
- **No** configuration beyond `config.yaml` — do NOT add `.env`, secrets managers, or config servers
- **No** microservice architecture, message queues, or task queues
- **No** unit tests beyond those explicitly specified in each task's validation section

### ✅ Scope Guidelines
- **Data pipeline**: Python scripts that run once → produce CSV/PT files → done
- **Training**: Single script → prints metrics → saves best model → done. **Uses two-stage fine-tuning.**
- **Inference**: Gradio app launched locally via `demo.launch()` → done
- **XAI**: Patch-based Grad-CAM for ViT and Integrated Gradients → done
- **File format**: PyTorch `.pt` only (no ONNX, no TensorFlow, no safetensors)
- **Scripts**: Keep each script as a **single file**. Do NOT split into modules.

### ⚠️ If You Encounter a Problem NOT Addressed Here
1. Use the **simplest possible** solution (fewer lines > elegance)
2. Do NOT refactor existing code unless it crashes
3. Do NOT optimize for performance unless it OOMs
4. If in doubt, follow `config.yaml` as the single source of truth

---

## 📋 CONFIGURATION FILE

Create `config.yaml` at project root with the following structure:

```yaml
# ============================================================
# VIRALSCOPE AI - Configuration (CLIP Architecture)
# ============================================================

project:
  name: "ViralScope AI"
  seed: 42
  device: "auto"

data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  tensor_dir: "data/tensors"
  splits_dir: "data/splits"

  train_split: 0.7
  val_split: 0.15
  test_split: 0.15

  target_threshold: 1.5
  thumbnail_url_template: "https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
  thumbnail_fallback_url: "https://img.youtube.com/vi/{video_id}/hqdefault.jpg"

training:
  batch_size: 64
  num_workers: 4
  epochs: 30
  early_stopping_patience: 7
  gradient_clip_norm: 1.0

model:
  clip:
    checkpoint: "openai/clip-vit-base-patch32"
    feature_dim: 512
    max_seq_length: 77
    freeze_backbone: true

  fusion:
    hidden_layers: [256, 64]
    dropout: 0.2
    activation: "GELU"

  # Learning rates
  two_stage:
    stage1_epochs: 5
    stage1_lr_head: 1e-3
    stage2_lr_backbone: 5e-6
    stage2_lr_head: 1e-4

  optimizer:
    name: "AdamW"
    weight_decay: 0.01
    betas: [0.9, 0.98]

  scheduler:
    name: "CosineAnnealingWarmRestarts"
    T_0: 10
    T_mult: 2
    eta_min: 1e-7

augmentation:
  resize_size: [224, 224]
  # CLIP specific normalization
  mean: [0.48145466, 0.4578275, 0.40821073]
  std: [0.26862954, 0.26130258, 0.27577711]

xai:
  gradcam:
    upsampling_mode: "bicubic"
  integrated_gradients:
    n_steps: 50

paths:
  checkpoints: "models/checkpoints"
  best_model: "models/best_model.pt"
  training_log: "models/training_log.csv"
  results: "results/"
```

---

## 📁 DIRECTORY STRUCTURE

Create the following structure at project root:

```
VIRALSCOPE_AI/
├── config.yaml                           # Central configuration
├── requirements.txt                      # torch, transformers, torchvision, etc.
│
├── data/
│   ├── dataset.py                        # CLIP-based ViralScopeDataset
│   ├── raw/                              # Original CSVs + Thumbnails
│   ├── processed/                        # Cleaned CSVs + Splitting report
│   ├── tensors/                          # Pre-tokenized CLIP tensors
│   └── splits/                           # Train/val/test indices
│
├── models/
│   ├── cv_extractor.py                   # CLIP Vision Encoder
│   ├── nlp_extractor.py                  # CLIP Text Encoder
│   ├── fusion_model.py                   # Interaction Multiplier head
│   ├── multimodal.py                     # Unified ViralScopeModel
│   └── checkpoints/                      # Saved weights
│
├── xai/
│   ├── gradcam.py                        # ViT-based Grad-CAM
│   └── integrated_gradients.py           # Captum IG on Text
│
├── app/
│   ├── inference.py                      # CLIP Inference Workflow
│   └── gradio_app.py                     # Dashboard
│
├── scripts/
│   ├── 01_download_data.py               # Filtering + Download
│   ├── 03_train.py                       # Two-stage training
│   └── 04_evaluate.py                    # Evaluation
│
└── results/                              # XAI outputs + Metrics
```

---

## 🎯 TASK BREAKDOWN

Each task is independent, has clear inputs/outputs, and includes validation criteria.

---

### PHASE 1: DATA ENGINEERING & BIAS RESOLUTION

**Strategy**: Use only the **Kaggle YouTube Trending Dataset**. Fetch thumbnails directly from YouTube's public CDN (`img.youtube.com/vi/{video_id}/maxresdefault.jpg`). No dataset merging needed.

---

#### TASK 1.1: Download YouTube Trending Dataset from Kaggle (US + GB + CA)
**Objective**: Get the single source of truth — video metadata + view counts. **US + GB + CA only** — these are English-speaking markets, so DistilBERT handles them perfectly. The other country files (DE, FR, JP, KR, RU, MX, IN) are excluded because they contain non-English titles.

**Inputs**:
- Kaggle API credentials (user must set up `~/.kaggle/kaggle.json`)
- Dataset: `Datasnaik/Youtube-Trending-Videos-Dataset` (or similar trending dataset)

**Outputs**:
- `data/raw/trending.csv` with required columns: `video_id`, `title`, `views`, `channel_title`, `channel_id`

**Steps**:
1. Create `data/raw/` directory
2. Install Kaggle CLI if needed: `pip install kaggle`
3. Download dataset:
   ```bash
   kaggle datasets download -d Datasnaik/Youtube-Trending-Videos-Dataset -p data/raw/
   ```
4. **Extract and combine `USvideos.csv` + `GBvideos.csv` + `CAvideos.csv` only** — these are the three English-speaking markets. Exclude DE, FR, JP, KR, RU, MX, IN (non-English).
5. Save combined data as `data/raw/trending.csv`
6. Verify columns exist

**Validation**:
- [ ] `data/raw/trending.csv` exists
- [ ] File has ≥ 10,000 rows (combined from US + GB + CA)
- [ ] Contains: `video_id`, `title`, `views`, `channel_title`, `channel_id`
- [ ] No more than 20% null values in required columns
- [ ] `views` column is numeric
- [ ] **US + GB + CA confirmed** (English titles only, three English-speaking markets)

**Error Handling**:
- If Kaggle API auth fails, instruct user to run `kaggle datasets init` + provide token
- If dataset name differs, update `-d` flag accordingly
- If downloading manually: get `USvideos.csv`, `GBvideos.csv`, `CAvideos.csv` from https://www.kaggle.com/datasets/datasnaek/youtube-new — upload directly to Colab VM

---

#### TASK 1.2: Clean Dataset — Deduplicate & Filter
**Objective**: Remove duplicate videos (same video trending across multiple countries/days) and invalid rows.

**Inputs**:
- `data/raw/trending.csv`

**Outputs**:
- `data/processed/clean_dataset.csv` — deduplicated, valid rows only

**Steps**:
1. Load `data/raw/trending.csv`
2. Drop rows where `video_id` is null or empty
3. Drop rows where `views` is null or non-numeric
4. Keep only the **last occurrence** of each `video_id` (drop duplicates on `video_id`, keep='last')
   - **Rationale**: In the Kaggle Trending dataset, videos appear over multiple consecutive days as views grow. Keeping the *last* occurrence captures the video near its peak trajectory, giving a more accurate measure of its true viral reach. Keeping 'first' would capture it at entry with artificially low view counts.
5. Drop rows where `title` is empty
6. Save to `data/processed/clean_dataset.csv`

**Validation**:
- [ ] File exists at `data/processed/clean_dataset.csv`
- [ ] All `video_id` values are unique (`df['video_id'].nunique() == len(df)`)
- [ ] All `views` > 0
- [ ] No null values in `video_id`, `title`, `views`
- [ ] Print: "Reduced from X raw rows to Y clean rows"

---

#### TASK 1.3: Download Thumbnails from YouTube CDN
**Objective**: Fetch thumbnails directly from YouTube's public image CDN.

**Inputs**:
- `data/processed/clean_dataset.csv` with `video_id` column
- Thumbnail URL template: `https://img.youtube.com/vi/{video_id}/maxresdefault.jpg`

**Outputs**:
- `data/raw/thumbnails/` directory with `{video_id}.jpg` files
- `data/processed/thumbnail_manifest.csv` with columns: `video_id`, `status` (success/fallback/failed)

**Steps**:
1. Create `data/raw/thumbnails/` directory
2. **Use `ThreadPoolExecutor` to parallelize downloads** (~5 workers cuts 50K downloads from ~1.4h to ~15 min):
   - YouTube's `img.youtube.com` CDN is robust and does not enforce strict rate limits
   - Keep `max_workers=5` to avoid overwhelming the network
   - Skip already-downloaded thumbnails
3. For each `video_id` in clean_dataset.csv:
   - Try max resolution: `https://img.youtube.com/vi/{video_id}/maxresdefault.jpg`
   - If 404 or < 120x90 (YouTube returns a placeholder for missing maxres), fallback to `hqdefault.jpg`
   - If fallback also fails, mark as "failed"
   - No sleep needed — connection pool handles rate limiting naturally
4. Save manifest with status for each video_id

**Validation**:
- [ ] `data/raw/thumbnails/` exists with downloaded images
- [ ] At least 70% of videos have successfully downloaded thumbnails
- [ ] Manifest file exists with status for every video_id
- [ ] Downloaded images are valid (can be opened with PIL)

**Error Handling**:
- If thumbnail missing/404, try fallback resolution
- If both fail, log as "failed" and continue (do NOT crash)
- Implement rate limiting (0.1s delay between requests)
- Handle network timeouts (10s per request, retry once)
- Use `requests` library with `stream=True` for efficiency
- Add `tqdm` progress bar for visibility

**Code snippet**:
```python
import requests
import os
import io
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_thumbnail(video_id, save_dir, min_width=120, min_height=90):
    """Download thumbnail from YouTube CDN. Validates actual image dimensions."""
    save_path = os.path.join(save_dir, f"{video_id}.jpg")
    if os.path.exists(save_path):
        return video_id, "exists"

    session = requests.Session()  # Reuse connections per thread
    session.headers.update({'User-Agent': 'Mozilla/5.0'})

    for res in ["maxresdefault", "hqdefault"]:
        url = f"https://img.youtube.com/vi/{video_id}/{res}.jpg"
        try:
            resp = session.get(url, timeout=10)
            if resp.status_code == 200 and len(resp.content) > 1000:
                img = Image.open(io.BytesIO(resp.content))
                w, h = img.size
                if w > 120 and h > 90:
                    with open(save_path, "wb") as f:
                        f.write(resp.content)
                    return video_id, "success"
                else:
                    pass  # Placeholder, try fallback
        except Exception:
            continue
    return video_id, "failed"

# Parallel download with thread pool
video_ids = df['video_id'].tolist()
results = {}
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = {executor.submit(download_thumbnail, vid, save_dir): vid for vid in video_ids}
    for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading thumbnails"):
        vid, status = future.result()
        results[vid] = status

# Save manifest
manifest = pd.DataFrame(list(results.items()), columns=['video_id', 'status'])
manifest.to_csv('data/processed/thumbnail_manifest.csv', index=False)
```

---

#### TASK 1.4: Filter to Videos with Valid Thumbnails
**Objective**: Keep only videos that have both metadata AND a thumbnail image.

**Inputs**:
- `data/processed/clean_dataset.csv`
- `data/processed/thumbnail_manifest.csv`

**Outputs**:
- `data/processed/final_dataset.csv` — only videos with valid thumbnails

**Steps**:
1. Load both CSVs
2. Filter manifest to `status == 'success'`
3. Inner join clean_dataset with valid thumbnail video_ids
4. Save result

**Validation**:
- [ ] File exists at `data/processed/final_dataset.csv`
- [ ] Every row has a corresponding thumbnail in `data/raw/thumbnails/`
- [ ] Print: "Final dataset: X videos with thumbnails (Y% of clean dataset)"
- [ ] If Y < 50%, warn about potential thumbnail coverage issues

---

#### TASK 1.5: Compute Channel Statistics (Leave-One-Out)
**Objective**: Calculate per-channel video counts and LOO aggregates for filtering and reference.

**⚠️ NOTE**: This file is used **only for filtering** in Task 1.6 (step 2: drop channels with < 3 videos). The actual LOO average used in the viral multiplier is recomputed **inline in Task 1.6** (per-video LOO is more accurate than a channel-level mean of LOOs). The `channel_averages.csv` is a convenience artifact for the filter step and for diagnostic reporting — it is NOT the source of the `channel_loo_avg_views` used in the multiplier.

**Inputs**:
- `data/processed/final_dataset.csv`

**Outputs**:
- `data/processed/channel_averages.csv` with columns: `channel_id`, `loo_avg_views`, `video_count`

**Steps**:
1. Load `final_dataset.csv`
2. Compute channel-level aggregates:
   ```python
   df['channel_sum_views'] = df.groupby('channel_id')['views'].transform('sum')
   df['channel_video_count'] = df.groupby('channel_id')['views'].transform('count')
   ```
3. **Leave-One-Out Average**: For each video, compute the channel average EXCLUDING itself:
   ```python
   # LOO mean = (Total sum - current video views) / (Count - 1)
   df['loo_avg_views'] = (df['channel_sum_views'] - df['views']) / (df['channel_video_count'] - 1 + 1e-5)  # +1e-5 avoids ZeroDivisionError for single-video channels
   ```
4. Aggregate per channel (take first row per channel, since loo_avg_views differs per video):
   ```python
   # For channel_averages.csv, store the mean of all LOO averages as the channel's baseline
   channel_stats = df.groupby('channel_id').agg(
       loo_avg_views=('loo_avg_views', 'mean'),
       video_count=('channel_video_count', 'first')
   ).reset_index()
   ```
5. Save to CSV

**⚠️ Known Limitation — Rolling-Mean LOO**:
The LOO formula above uses the **global** channel average across all videos in the dataset. This is a **dataset limitation**. The global LOO is the best achievable proxy given the available data.

**Validation**:
- [ ] File exists at `data/processed/channel_averages.csv`
- [ ] All `loo_avg_views` > 0
- [ ] All `video_count` ≥ 1
- [ ] No null values

---

#### TASK 1.6: Compute Target Variable (Viral Multiplier with LOO)
**Objective**: Calculate and binarize the viral multiplier.

**Inputs**:
- `data/processed/final_dataset.csv`
- `data/processed/channel_averages.csv`
- Threshold from config: `data.target_threshold` (1.5)

**Outputs**:
- `data/processed/labeled_dataset.csv` with columns: `video_id`, `title`, `views`, `channel_loo_avg_views`, `multiplier`, `is_viral`, `channel_id`

**Steps**:
1. Merge `final_dataset.csv` with `channel_averages.csv` on `channel_id`
2. **FILTER OUT unreliable channels**: Drop rows where `channel_averages.video_count < 3`
3. **Compute per-video LOO average inline**
4. Compute: `multiplier = views / channel_loo_avg_views`
5. Binarize: `is_viral = 1 if multiplier > 1.5 else 0`
   - **Note**: This label will be used with `BCEWithLogitsLoss`.
6. Drop rows with missing data
7. **KEEP `channel_id`** for Task 1.9 splitting.
8. Save to `data/processed/labeled_dataset.csv`

**Validation**:
- [ ] File exists at `data/processed/labeled_dataset.csv`
- [ ] `is_viral` contains 0 and 1
- [ ] Print class distribution: "Viral: X (Y%) | Non-viral: Z (W%)"
- [ ] `channel_id` present for splitting.

**Data Leakage Prevention**:
- [ ] `channel_title`/`channel_name` NOT in output CSV
- [ ] `channel_id` IS in output CSV (needed for Task 1.9 grouped split)
- [ ] Only `title` + thumbnail will be model inputs during training

---

#### TASK 1.7: Pre-Tokenize Text Corpus
**Objective**: Pre-compute CLIP tokens to avoid CPU bottleneck.

**Inputs**:
- `data/processed/labeled_dataset.csv`
- Tokenizer: `CLIPTokenizer` (`openai/clip-vit-base-patch32`)
- Max length: `77`

**Outputs**:
- `data/tensors/input_ids.pt`
- `data/tensors/attention_masks.pt`
- `data/tensors/tokenizer_metadata.json`

**Steps**:
1. Initialize `CLIPTokenizer`
2. Tokenize all titles (max_length=77, padding='max_length', truncation=True)
3. Save tokens as `.pt` files.
4. Lock dataset with a hash.

**Validation**:
- [ ] Both `.pt` files exist
- [ ] Shapes: `(N, 77)`
- [ ] Dataset hash saved.

**⚠️ CRITICAL IMMUTABILITY ASSERTION**:
After this task completes, `data/processed/labeled_dataset.csv` becomes **IMMUTABLE**.
The tensor indices in `input_ids.pt` and `attention_masks.pt` are positionally aligned with the CSV rows.
Any subsequent filtering, reordering, or row removal will **silently corrupt** the tensor-to-label mapping.
- Tasks 1.9 (splitting) and 1.10 (class weights) must use the **exact same row order**.
- If any post-tokenization filtering is needed, re-run this task afterward.
```python
# Save a hash checkpoint to detect corruption later
import hashlib
with open('data/processed/labeled_dataset.csv', 'rb') as f:
    df_hash = hashlib.md5(f.read()).hexdigest()
with open('data/tensors/tokenizer_metadata.json', 'r') as f:
    metadata = json.load(f)
metadata['dataset_hash'] = df_hash
with open('data/tensors/tokenizer_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"✓ Dataset hash locked: {df_hash}")
```

**Error Handling**:
- If titles contain invalid characters, clean before tokenization
- If corpus too large for memory, batch tokenize (chunks of 10,000)

---

#### TASK 1.8: [REMOVED] Pre-Process and Save Image Tensors
**Objective**: ~~Load, resize, and normalize all thumbnails for efficient DataLoader access.~~

**⚠️ THIS TASK IS INTENTIONALLY REMOVED.** 

**Rationale**: Pre-computing and saving all image tensors as a single `.pt` file causes:
1. **RAM Explosion**: 50,000 images × 3 × 224 × 224 × 4 bytes (float32) ≈ **30 GB** single file → OOM crashes
2. **Data Augmentation Block**: Static tensors prevent dynamic augmentations (rotation, color jitter, flips) defined in `config.yaml`
3. **No benefit**: Image loading via PIL + torchvision transforms is fast enough when done on-the-fly in `__getitem__`

**Replacement**: Image loading is now handled dynamically in **Task 4.4** (`ViralScopeDataset`), which loads thumbnails from `data/raw/thumbnails/{video_id}.jpg` on-the-fly and applies transforms (including augmentation for training set).

**Validation**:
- [ ] `data/tensors/image_tensors.pt` does NOT exist
- [ ] Thumbnail images remain in `data/raw/thumbnails/` directory
- [ ] Task 4.4 implementation loads images dynamically

---

#### TASK 1.9: Grouped Train/Val/Test Split (by channel_id)
**Objective**: Create dataset splits that **prevent data leakage** by ensuring all videos from the same channel stay together.

**Inputs**:
- `data/processed/labeled_dataset.csv` — **MUST retain `channel_id` column** (despite Task 1.6 dropping it from final output, we need it here for splitting)
- Split ratios from config: 70/15/15

**⚠️ CRITICAL**: Do NOT use standard `train_test_split` with stratification. Videos from the same channel in both train and test sets allow the model to learn creator-specific aesthetics (e.g., MrBeast's font/style), causing inflated test scores that don't generalize to new creators.

**Outputs**:
- `data/splits/train_indices.pt`
- `data/splits/val_indices.pt`
- `data/splits/test_indices.pt`
- `data/splits/split_report.json`

**Steps**:
1. Load labeled dataset **with `channel_id`** (use a temporary copy before Task 1.6 drops it, or re-merge)
2. Use `sklearn.model_selection.GroupShuffleSplit` for channel-aware splitting:
   ```python
   from sklearn.model_selection import GroupShuffleSplit
   
   # First split: 70% train, 30% temp (grouped by channel_id)
   gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
   train_idx, temp_idx = next(gss1.split(df, groups=df['channel_id']))
   
   # Second split: 50% val, 50% test from temp (grouped by channel_id)
   gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
   temp_df = df.iloc[temp_idx].reset_index(drop=True)
   val_temp_idx, test_temp_idx = next(gss2.split(temp_df, groups=temp_df['channel_id']))
   
   # Map back to original dataframe indices
   val_idx = temp_idx[val_temp_idx]
   test_idx = temp_idx[test_temp_idx]
   ```
3. Save index tensors
4. **Generate and save split report** (`data/splits/split_report.json`):
   ```python
   import json
   
   def split_stats(df, indices, name):
       subset = df.iloc[indices]
       return {
           'split': name,
           'num_videos': len(indices),
           'num_channels': subset['channel_id'].nunique(),
           'viral_count': int(subset['is_viral'].sum()),
           'non_viral_count': int((subset['is_viral'] == 0).sum()),
           'viral_pct': float(subset['is_viral'].mean())
       }
   
   report = {
       'train': split_stats(df, train_idx, 'train'),
       'val': split_stats(df, val_idx, 'val'),
       'test': split_stats(df, test_idx, 'test'),
       'total_videos': len(df),
       'total_channels': df['channel_id'].nunique(),
       'channel_overlap': {
           'train_val': len(train_channels & val_channels),
           'train_test': len(train_channels & test_channels),
           'val_test': len(val_channels & test_channels)
       }
   }
   with open('data/splits/split_report.json', 'w') as f:
       json.dump(report, f, indent=2)
   ```

**Validation**:
- [ ] All three index files exist
- [ ] Sum of lengths equals total dataset size
- [ ] No overlapping indices between splits
- [ ] **NO channel_id overlap**: `set(train_channels) ∩ set(val_channels) ∩ set(test_channels) = ∅`
- [ ] Class distribution within ±5% across all splits (grouped splits may have slight imbalance)
- [ ] Split report JSON contains:
  - Video counts and percentages per split
  - Channel counts per split
  - Class distribution per split
  - Channel overlap verification result

**Channel Overlap Check Code**:
```python
train_channels = set(df.iloc[train_idx]['channel_id'].unique())
val_channels = set(df.iloc[val_idx]['channel_id'].unique())
test_channels = set(df.iloc[test_idx]['channel_id'].unique())

assert len(train_channels & val_channels) == 0, "DATA LEAKAGE: channels in both train and val!"
assert len(train_channels & test_channels) == 0, "DATA LEAKAGE: channels in both train and test!"
assert len(val_channels & test_channels) == 0, "DATA LEAKAGE: channels in both val and test!"
print("✓ No channel overlap between splits — data leakage prevented")
```

**Error Handling**:
- If GroupShuffleSplit fails (too few unique channels), fall back to stratified split with WARNING
- If dataset too small (< 1000 samples), log warning and use 80/10/10
- If class imbalance > 10% between splits, attempt re-split with different random_state

---

#### TASK 1.10: Compute Class Weights
**Objective**: Calculate weights for imbalanced classification.

**Inputs**:
- `data/processed/labeled_dataset.csv` with `is_viral` column
- Training indices

**Outputs**:
- `data/processed/class_weights.pt` - shape: `(2,)`

**Steps**:
1. Load labeled dataset and training indices
2. Count samples per class in training set
3. Compute weights: `weight_i = total_samples / (num_classes * count_i)`
4. Normalize weights to sum to num_classes
5. Save as tensor

**Validation**:
- [ ] Tensor exists with shape `(2,)`
- [ ] Both weights are positive
- [ ] Weights sum to 2.0 (normalized)
- [ ] Minority class weight > majority class weight
- [ ] Config file updated successfully

---

### PHASE 2: COMPUTER VISION MODEL (CNN BRANCH)

---

### PHASE 2: COMPUTER VISION MODEL (CLIP BRANCH)

---

#### TASK 2.1: Create CLIP Vision Feature Extractor
**Objective**: Build CLIP-based vision encoder for image feature extraction.

**Inputs**:
- Config: `model.clip.checkpoint` (`openai/clip-vit-base-patch32`)

**Outputs**:
- `models/cv_extractor.py` - CVExtractor class

**Implementation** (`models/cv_extractor.py`):
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CVExtractor(nn.Module):
    def __init__(self, vision_model, visual_projection):
        super().__init__()
        self.backbone = vision_model
        self.projection = visual_projection
    
    def forward(self, pixel_values):
        # Extract features from CLIP vision transformer
        outputs = self.backbone(pixel_values=pixel_values)
        # pooler_output corresponds to the CLS token + projection
        pooled_output = outputs.pooler_output
        image_embeds = self.projection(pooled_output)
        # CLIP embeddings MUST be L2-normalized
        return F.normalize(image_embeds, p=2, dim=-1)
```

**Validation**:
- [ ] Shape: `(batch, 512)`
- [ ] L2-normalized: `norm(out) ≈ 1.0`
- [ ] Test passes: `test_cv_extractor_shape`

---

#### TASK 2.2: Validate CV Extractor with Real Data
**Objective**: Test CLIP Vision extractor with CLIP-normalized thumbnails.

**Validation**:
- [ ] Output shape: `(64, 512)`
- [ ] No NaNs
- [ ] CLIP specific normalization applied: `mean=[0.4814, 0.4578, 0.4082]`

---

### PHASE 3: NLP MODEL (CLIP BRANCH)

---

#### TASK 3.1: Create CLIP Text Feature Extractor
**Objective**: Build CLIP-based text encoder for semantic feature extraction.

**Inputs**:
- Config: `model.clip.checkpoint`

**Outputs**:
- `models/nlp_extractor.py` - NLPExtractor class

**Implementation** (`models/nlp_extractor.py`):
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class NLPExtractor(nn.Module):
    def __init__(self, text_model, text_projection):
        super().__init__()
        self.backbone = text_model
        self.projection = text_projection
    
    def forward(self, input_ids, attention_mask):
        # Extract features from CLIP text transformer
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # pooled_output corresponds to the EOT token
        pooled_output = outputs.pooler_output
        text_embeds = self.projection(pooled_output)
        # CLIP embeddings MUST be L2-normalized
        return F.normalize(text_embeds, p=2, dim=-1)
```

**Validation**:
- [ ] Shape: `(batch, 512)`
- [ ] L2-normalized
- [ ] Alignment: Embeddings are in the same space as CLIP Vision.

---

#### TASK 3.2: Validate NLP Extractor with Titles
**Validation**:
- [ ] Output shape: `(64, 512)`
- [ ] Similarity Check: Pairwise cosine similarity between aligned image/text should be high.

---

### PHASE 4: LATE FUSION & TRAINING

---

#### TASK 4.1: Implementation of Fusion Head with Semantic Interactions
**Objective**: Build a classifier that explicitly computes cross-modal interaction features.

**Inputs**:
- CLIP embeddings (512-dim vision, 512-dim text)

**Outputs**:
- `models/fusion_model.py` - FusionMLP class

**Implementation Implementation Details**:
- Features to compute:
    1. **Concat**: `[img, txt]` (1024 dim)
    2. **Element-wise Difference**: `|img - txt|` (512 dim)
    3. **Element-wise Product**: `img * txt` (512 dim)
    4. **Cosine Similarity**: `dot(img, txt)` (1 dim)
- Total Input: 2049 dim.
- Layers: `LayerNorm` -> `Linear(2049, 256)` -> `GELU` -> `Dropout(0.2)` -> `Linear(256, 64)` -> `GELU` -> `Linear(64, 1)`.

**Validation**:
- [ ] Output shape is `(batch,)`
- [ ] Forward pass on dummy data works without NaNs.

---

#### TASK 4.2: Integrated ViralScope Multimodal Model
**Objective**: Combine extractors and fusion head into a unified model.

**Inputs**:
- Pre-loaded CLIP components

**Outputs**:
- `models/multimodal.py` - ViralScopeModel class

**Steps**:
1. Initialize `ViralScopeModel` with a single `CLIPModel` instance.
2. Split `CLIPModel` into `vision_model` and `text_model` (shared weights).
3. Initialize `FusionMLP`.
4. In `forward`, extract features -> L2 normalize -> compute interactions -> predict logit.

**Validation**:
- [ ] Model parameters count matches expected CLIP backbone + head.
- [ ] `predict_proba` returns valid probabilities.

---

#### TASK 4.3: Multimodal CLIP Dataset
**Objective**: Build a dataset that handles CLIP Vision normalization and CLIP Tokenizer input.

**Implementation Details**:
- **Tokenizer**: Use `CLIPTokenizer` with `max_length=77`.
- **Image Transform**:
    - `Resize((224, 224))`
    - `CenterCrop(224)`
    - `ToTensor()`
    - `Normalize(mean=[0.4814, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2757])`
- **Labels**: Ensure `is_viral` is float32 for `BCEWithLogitsLoss`.

**Validation**:
- [ ] DataLoader yields tensors with shape `(batch, 3, 224, 224)` and `(batch, 77)`.
- [ ] Label balance is handled via `WeightedRandomSampler` or data-level oversampling.

---

#### TASK 4.4: Two-Stage Fine-Tuning Script
**Objective**: Implement the training logic that stabilizes CLIP fine-tuning.

**Inputs**:
- Config: `model.clip.learning_rate_backbone`, `model.clip.learning_rate_head`

**Steps**:
1. **Stage 1 (Warmup)**: Freeze the CLIP backbone (`requires_grad=False`). Train only the `FusionMLP` head for 5 epochs with a higher learning rate ($10^{-3}$).
2. **Stage 2 (Fine-tune)**: Unfreeze the entire model. Use staggered learning rates: $10^{-4}$ for the fusion head and $5 \times 10^{-6}$ for the CLIP backbones.
3. **Loss Function**: Use `BCEWithLogitsLoss`.
4. **Metric Logging**: Track `PR-AUC` as the primary validation metric to handle class imbalance.

**Validation**:
- [ ] Training script completes Stage 1 and transitions to Stage 2.
- [ ] Checkpoint `best_model.pt` is saved based on validation PR-AUC.

---
    # CRITICAL: labeled_dataset.csv must retain 'channel_id' for grouped splitting
    # but it should NOT contain 'channel_title' or 'channel_name' (dropped in Task 1.6)
    
    # Load split indices
    train_idx = torch.load('data/splits/train_indices.pt', weights_only=True)
    val_idx = torch.load('data/splits/val_indices.pt', weights_only=True)
    test_idx = torch.load('data/splits/test_indices.pt', weights_only=True)

    # Create augmentation transforms for training
    train_transform = get_train_transform(config)

    # Create datasets
    train_ds = ViralScopeDataset(
        df, 'data/raw/thumbnails', input_ids, attention_masks,
        train_idx, is_train=True, augmentations=train_transform
    )
    val_ds = ViralScopeDataset(
        df, 'data/raw/thumbnails', input_ids, attention_masks,
        val_idx, is_train=False
    )
    test_ds = ViralScopeDataset(
        df, 'data/raw/thumbnails', input_ids, attention_masks,
        test_idx, is_train=False
    )

    # Create dataloaders
    train_dl = DataLoader(
        train_ds,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 2),
        pin_memory=True  # Faster GPU transfer
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 2),
        pin_memory=True
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 2),
        pin_memory=True
    )

    return train_dl, val_dl, test_dl
```

**Key Changes from Previous Version**:
1. **NO `torch.load('image_tensors.pt')`** — images loaded on-demand from disk
2. **Dynamic augmentation** applied only to training set via `get_train_transform()`
3. **`pin_memory=True`** for faster CPU→GPU transfers
4. **Graceful fallback** for missing/corrupt images (blank image + warning)
5. **`dataframe`-based indexing** instead of raw tensor slicing (enables channel_id access)

**Validation**:
- [ ] DataLoader yields batches without errors
- [ ] Batch shapes match expected dimensions
- [ ] Labels are float32 (for BCE/Focal loss)
- [ ] Training set applies random augmentations (verify with 2 passes on same index)
- [ ] Val/test sets return deterministic outputs
- [ ] Test passes: `test_dataloader_batch`

**Unit Test**:
```python
def test_dataloader_batch():
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    from data.dataset import create_dataloaders
    train_dl, _, _ = create_dataloaders(config)
    batch = next(iter(train_dl))
    assert batch['images'].shape[0] == config['training']['batch_size']
    assert batch['input_ids'].shape[0] == config['training']['batch_size']
    assert batch['labels'].shape[0] == config['training']['batch_size']
    
def test_augmentation_is_random():
    """Verify training augmentations are stochastic.
    
    CRITICAL FIX: Must compare same index from same DataLoader,
    not two different loaders (which would differ due to shuffle).
    """
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    from data.dataset import ViralScopeDataset, get_train_transform
    
    # Load minimal data for testing
    import torch
    import pandas as pd
    
    df = pd.read_csv('data/processed/labeled_dataset.csv').head(10)
    input_ids = torch.load('data/tensors/input_ids.pt', weights_only=True)[:10]
    attention_masks = torch.load('data/tensors/attention_masks.pt', weights_only=True)[:10]
    indices = torch.arange(5)
    
    train_transform = get_train_transform(config)
    ds = ViralScopeDataset(
        df, 'data/raw/thumbnails', input_ids, attention_masks,
        indices, is_train=True, augmentations=train_transform
    )
    
    # Get same index twice WITHOUT shuffle
    img1 = ds[0]['images']
    img2 = ds[0]['images']
    
    # Images should differ due to random augmentations
    assert not torch.equal(img1, img2), "Augmentation is not stochastic!"
```

---

#### TASK 4.5: Two-Stage CLIP Fine-Tuning Execution
**Objective**: Run the comprehensive training pipeline to converge on virality prediction.

**Steps**:
1. **Model Loading**: Initialize `ViralScopeModel` with `clip-vit-base-patch32`.
2. **Loss Initialization**: Set up `BCEWithLogitsLoss`.
3. **Stage 1 Execution**: Train for 5 epochs with frozen backbones. Verify PR-AUC starts climbing.
4. **Stage 2 Execution**: Unfreeze and train for 20 epochs with `CosineAnnealingWarmRestarts`.
5. **Checkpointing**: Save `models/best_model.pt` based on peak validation PR-AUC.

**Validation**:
- [ ] Log shows PR-AUC curve improving during fine-tuning (Stage 2).
- [ ] No gradient explosion (check weight norms).

---
---

#### TASK 4.6: PR-AUC Optimization & Checkpointing
**Objective**: Finalize model weights by monitoring Precision-Recall performance.

**Steps**:
1. **Metric Focus**: Transition evaluation from accuracy to PR-AUC to handle the viral/non-viral imbalance.
2. **Best Model Selection**: Save `models/best_model.pt` at the global PR-AUC peak.

**Validation**:
- [ ] PR-AUC exceeds 0.60 on the validation set.
- [ ] `best_model.pt` exists and is loadable.

---

### PHASE 5: EVALUATION, XAI & DEPLOYMENT

---

#### TASK 5.1: Performance Audit on Test Set
**Objective**: Generate a robust final metric report.

**Implementation Details**:
- **Metric Focus**: Transition evaluation from accuracy to PR-AUC to handle the viral/non-viral imbalance.
- **Reporting**: Generate a confusion matrix to visualize class-specific errors.
- **Output**: `results/metrics.json`.

---

#### TASK 5.2: ViT-Aware Grad-CAM (Spatial Reshaping)
**Objective**: Implement visual explanations for the Vision Transformer branch.

**Implementation Details**:
- **Hooking**: Hook the last transformer block (or LayerNorm) of the ViT.
- **Reshaping**: Reshape the sequence output (minus CLS) into a $7 \times 7$ grid for heatmap interpolation.
- **Interpolation**: Upscale to $224 \times 224$ for thumbnail overlay.

---

#### TASK 5.3: Integrated Gradients for CLIP Text
**Objective**: Attribute virality scores to specific words in the title.

**Implementation Details**:
- **Tokenizer**: Use `CLIPTokenizer` inputs.
- **Attribution**: Apply Captum's `IntegratedGradients` to the CLIP text embedding layer.

---

#### TASK 5.4: CLIP-Ready Inference Pipeline
**Objective**: Unified class for raw input processing.

**Features**:
- Multimodal preprocessing (CLIP Image transforms + CLIP Tokenizer).
- Automatic L2 normalization of embeddings.
- Prediction with interaction fusion weights.

---

#### TASK 5.5: Gradio Dashboard
**Objective**: Build a premium terminal for virality analysis.

**Features**:
- Thumbnail Upload + Title Input.
- Confidence Score Gauge.
- Multimodal Explanations (Visual Heatmap + Text Importance).

---
_output = gr.Image(label="Visual Attention Heatmap")
            title_output = gr.HTML(label="Title Word Impact")
    
    submit_btn.click(
        fn=analyze_video,
        inputs=[thumbnail_input, title_input],
        outputs=[output_message, gradcam_output, title_output]
    )

if __name__ == '__main__':
    demo.launch()
```

**Validation**:
- [ ] App launches without errors
- [ ] Interface displays all input/output components
- [ ] Submit button triggers prediction
- [ ] Valid inputs produce prediction message + XAI outputs
- [ ] Invalid inputs (no image, empty title) show error messages
- [ ] Heatmap displays when XAI enabled

---

## ✅ SUCCESS CRITERIA SUMMARY

| Phase | Completion Criteria |
|-------|-------------------|
| **Phase 1** | CLIP-compatible thumbnails and tokenized text tensors pre-computed and saved as `.pt` files. |
| **Phase 2** | Vision branch successfully extracts 512-dim L2-normalized embeddings from ViT patches. |
| **Phase 3** | NLP branch successfully extracts 512-dim L2-normalized embeddings from CLIP Transformer. |
| **Phase 4** | Two-stage training converges with PR-AUC ≥ 0.60 on validation set. |
| **Phase 5** | Gradio interface provides interactive multimodal explanations (ViT Heatmap + Token attribution). |

---

## 🚀 EXECUTION ORDER

Execute tasks in this sequence:

```
TASK 0: ENVIRONMENT SETUP
0.1 → 0.2 → 0.3 → 0.4
↓
PHASE 1: DATA ENGINEERING
1.1 → 1.2 → 1.3 → 1.4 → 1.5 → 1.6 → 1.7 → [1.8 SKIPPED] → 1.9 → 1.10
↓
PHASE 2 & 3: CLIP BACKBONES
2.1 → 3.1
↓
PHASE 4: INTERACTION FUSION & TRAINING
4.1 → 4.2 → 4.3 → 4.4 → 4.5 → 4.6
↓
PHASE 5: EVALUATION, XAI & DEPLOYMENT
5.1 → 5.2 → 5.3 → 5.4 → 5.5
```

---

## 🔧 TASK 0: ENVIRONMENT SETUP

**Objective**: Configure the CLIP-enabled Python environment.

---

#### TASK 0.1: Create Virtual Environment
```bash
python -m venv venv
# Activate (Windows: venv\Scripts\activate | Linux: source venv/bin/activate)
```

---

#### TASK 0.2: Install CLIP-Compatible Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

#### TASK 0.3: Project Structure Verification
```bash
mkdir -p data/raw/thumbnails data/processed data/tensors data/splits models/checkpoints xai app scripts tests results/xai_samples
```

---

## 📦 DEPENDENCIES (requirements.txt)

Create `requirements.txt` with:

```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.30.0
captum>=0.6.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
gradio>=4.0.0  # Gradio 4.x API — code written for 4.x interface
matplotlib>=3.7.0
opencv-python-headless>=4.8.0  # headless version for servers without display
pyyaml>=6.0
pillow>=10.0.0
kaggle>=1.6.0  # For downloading YouTube Trending Dataset
requests>=2.31.0  # For thumbnail downloading
tqdm>=4.65.0  # Progress bars
```

**Notes**:
- `h5py` removed: HDF5 is no longer used; `.pt` files suffice for NLP tensors
- `gradio` set to `>=4.0.0` — code written for Gradio 4.x API

Install with: `pip install -r requirements.txt`

---

*This task decomposition document enables any AI agent to execute the VIRALSCOPE AI project independently with clear inputs, outputs, validation criteria, and error handling for each step.*
