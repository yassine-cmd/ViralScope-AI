# ViralScope AI - Progress Report

## 2026-04-18 - Pipeline Fixes

### Critical Fixes Applied

1. **Gradient Checkpointing (🔴 Fixed)**
   - **Issue**: `gradient_checkpointing_enable()` and `enable_gradient_checkpointing()` methods don't exist on CLIP's `CLIPVisionTransformer`/`CLIPTextTransformer` (they're plain `nn.Module`, not `PreTrainedModel`)
   - **Fix**: Set flag directly on encoder: `model.cv_extractor.vision_model.encoder.gradient_checkpointing = True`
   - **Location**: `viralscope_full_pipeline.ipynb` Cell 13

2. **AMP + Gradient Checkpointing Conflict (🔴 Fixed)**
   - **Issue**: `torch.amp.autocast` wrapping full forward can conflict with gradient checkpointing recompute
   - **Fix**: Disabled AMP (`scaler = None`) in Stage 2 when gradient checkpointing is active
   - **Location**: `viralscope_full_pipeline.ipynb` Cell 13

3. **torch.compile State Dict Key Mismatch (🟠 Fixed)**
   - **Issue**: `torch.compile()` prepends `_orig_mod.` to state dict keys, making saved checkpoints unloadable to plain model
   - **Fix**: Save from `model_raw.state_dict()` instead of `model.state_dict()`
   - **Location**: `viralscope_full_pipeline.ipynb` Cell 13

4. **LOO Label Leakage (🔴 Fixed)**
   - **Issue**: Channel statistics computed on full dataset before split, contaminating val/test labels
   - **Fix**: Moved compute_labels AFTER splits, compute stats only from training data
   - **Location**: `viralscope_full_pipeline.ipynb` Cells 5, 8

### Medium Priority Fixes

5. **CosineAnnealingWarmRestarts Stepped Wrong**
   - **Issue**: Stepped once per epoch instead of with epoch number
   - **Fix**: Changed to `scheduler.step(epoch + 1)` 
   - **Location**: `viralscope_full_pipeline.ipynb` Cell 13, `scripts/03_train.py`

6. **Duplicate Param Groups After Unfreeze**
   - **Issue**: `optimizer.add_param_group()` added unfrozen params twice
   - **Fix**: Modify LR in existing param group instead of adding new one
   - **Location**: `viralscope_full_pipeline.ipynb` Cell 11, `scripts/03_train.py`

7. **No Image Augmentation**
   - **Issue**: Augmentation disabled in config, causing memorization of specific thumbnails
   - **Fix**: Enabled in `config.yaml` with: rotation (0-15°), flip (0.5), color jitter (0.2)
   - **Location**: `config.yaml`

8. **BILINEAR vs BICUBIC Mismatch**
   - **Issue**: Train used BILINEAR, eval used BICUBIC (CLIP pretrained with BICUBIC)
   - **Fix**: Changed train transform to use BICUBIC
   - **Location**: `viralscope_full_pipeline.ipynb` Cell 10

9. **Redundant Fusion Features**
   - **Issue**: `|img-txt|` and `img*txt` add no independent information, increase overfitting
   - **Fix**: Removed them, reduced input from 2049 to 1025 dimensions
   - **Location**: `models/fusion_model.py`, `config.yaml`

10. **Rate Limit Not Enforced**
    - **Issue**: `thumbnail_rate_limit: 10` defined in config but never used
    - **Fix**: Added time.sleep-based rate limiting in download function
    - **Location**: `viralscope_full_pipeline.ipynb` Cell 6

### Files Modified
- `config.yaml` - augmentation enabled, fusion input updated
- `viralscope_full_pipeline.ipynb` - Cells 5, 6, 10, 11, 12, 13
- `models/fusion_model.py` - simplified feature input
- `scripts/03_train.py` - scheduler and unfreeze_backbone fixes