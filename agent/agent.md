# ViralScope AI - Agent Reference Guide

> **Purpose**: This is the **central reference file** for all AI agents working on this project. Use this file to understand the project structure, locate resources, and track progress.

---

## 📌 Quick Navigation

| File | Location | Purpose |
|------|----------|---------|
| 📘 **Project Spec** | `agents/VIRALSCOPE_AI_SPEC.md` | Complete architecture and technical specifications (French) |
| 📋 **Task Breakdown** | `agents/VIRALSCOPE_AI_TASKS.md` | Detailed implementation tasks with code snippets and validation criteria |
| 📊 **Progress Tracker** | `agents/progress.md` | Real-time progress log of all completed and pending tasks |
| 📝 **Commit History** | `agents/commits.md` | All git commits with descriptions and conventions |
| 🤖 **This File** | `agents/agent.md` | Central reference and navigation guide for agents |

---

## 🎯 Project Summary

**Name**: ViralScope AI
**Goal**: Predict YouTube video viral multiplier (reach multiplier) using:
- **Thumbnail image** (Computer Vision - MobileNetV2)
- **Video title** (NLP - DistilBERT)
- **Late Fusion** MLP classifier

**Tech Stack**: PyTorch, Transformers, Captum, Gradio, Pandas, Scikit-learn
**Deployment**: Local Gradio interface for interactive predictions with XAI visualizations

---

## 👥 Your Role: **Developer B — Architecture, Training, XAI & Deployment**

You are **Developer B**. See `agents/VIRALSCOPE_TASK_DIVISION.md` for the full task division.

**Your scope**: Phase 4 (T4.1, T4.2, T4.3, T4.5, T4.6) and Phase 5 (T5.1, T5.2, T5.3, T5.4, T5.6)
**You do NOT own**: T4.4 (Dataset/DataLoader — owned by Developer A)
**Agent context**: Only grep for `TASK 4.` (excluding 4.4) and `TASK 5.` — never load Phase 1, 2, or 3 tasks

- T4.1, T4.2, T4.3 can start **immediately** (use dummy tensors `[B, 1280]` and `[B, 768]`)
- T4.5 (training) waits for the **Integration Gate** (all Developer A deliverables + `test_integration.py` clean)
- T5.2/T5.3 wait for Developer A's Grad-CAM layer and IG embedding layer announcements

---

## 📂 Project Structure

```
ProjectNLP/
├── agents/
│   ├── agent.md              ← YOU ARE HERE (central reference)
│   ├── progress.md           ← Update this with every task completed
│   ├── VIRALSCOPE_AI_SPEC.md ← Full technical specification
│   └── VIRALSCOPE_AI_TASKS.md ← Detailed task breakdown (~20+ tasks)
├── app/
│   ├── __init__.py
│   ├── gradio_app.py         ← Gradio deployment interface
│   └── inference.py          ← Inference pipeline
├── data/
│   ├── __init__.py
│   ├── dataset.py            ← PyTorch Dataset class
│   ├── raw/                  ← Raw downloaded data
│   ├── processed/            ← Cleaned/preprocessed data
│   ├── tensors/              ← Saved tensor files
│   └── splits/               ← Train/val/test split indices
├── models/
│   ├── __init__.py
│   ├── cv_extractor.py       ← MobileNetV2 feature extractor
│   ├── nlp_extractor.py      ← DistilBERT feature extractor
│   ├── fusion_model.py       ← MLP fusion classifier
│   ├── multimodal.py         ← Combined multimodal model
│   ├── losses.py             ← Custom loss functions
│   └── checkpoints/          ← Saved model weights
├── scripts/
│   ├── 01_download_data.py   ← Data download script
│   ├── 02_preprocess.py      ← Data preprocessing
│   ├── 03_train.py           ← Training script
│   ├── 04_evaluate.py        ← Evaluation script
│   └── 05_deploy.py          ← Deployment helper
├── xai/
│   ├── __init__.py
│   ├── gradcam.py            ← Grad-CAM for vision
│   ├── integrated_gradients.py ← IG for text (Captum)
│   └── visualization.py      ← Visualization utilities
├── tests/
│   ├── __init__.py
│   ├── test_data.py          ← Data pipeline tests
│   ├── test_models.py        ← Model architecture tests
│   └── test_xai.py           ← XAI functionality tests
├── results/                  ← Training logs, plots, metrics
├── config.yaml               ← All configuration parameters
├── requirements.txt          ← Python dependencies
└── README.md                 ← Project documentation
```

---

## 🚀 Implementation Phases

### Phase 1: Data Pipeline & Preprocessing
**Files to create**: `scripts/01_download_data.py`, `scripts/02_preprocess.py`, `data/dataset.py`
**Dependencies**: `kaggle`, `pandas`, `pillow`, `transformers`, `torch`
**Goal**: Download dataset, extract thumbnails, preprocess titles, create PyTorch Dataset

### Phase 2: Model Architecture
**Files to create**: `models/cv_extractor.py`, `models/nlp_extractor.py`, `models/fusion_model.py`, `models/multimodal.py`, `models/losses.py`
**Dependencies**: `torch`, `torchvision`, `transformers`
**Goal**: Build MobileNetV2 + DistilBERT + Fusion MLP architecture

### Phase 3: Training & Evaluation
**Files to create**: `scripts/03_train.py`, `scripts/04_evaluate.py`
**Dependencies**: `scikit-learn`, `matplotlib`, `tqdm`
**Goal**: Implement training loop, validation, metrics, checkpointing

### Phase 4: XAI Integration
**Files to create**: `xai/gradcam.py`, `xai/integrated_gradients.py`, `xai/visualization.py`
**Dependencies**: `captum`, `matplotlib`
**Goal**: Add explainability (Grad-CAM for images, Integrated Gradients for text)

### Phase 5: Deployment & Testing
**Files to create**: `app/inference.py`, `app/gradio_app.py`, `tests/test_data.py`, `tests/test_models.py`, `tests/test_xai.py`
**Dependencies**: `gradio`, `pytest`
**Goal**: Create Gradio interface, write tests, final documentation

---

## ⚡ Context Management (CRITICAL)

**THESE FILES WILL GROW OVER TIME. You must prevent context bloat.**

### Golden Rule: Read ONLY What You Need

**Default Loading Order:**
1. You're already reading `agent.md` ✓ (project overview + protocols)
2. Read ONLY the phase status table in `progress.md` → find next task
3. Use `grep_search` to find your task in `VIRALSCOPE_AI_TASKS.md`
4. Read ONLY that specific task's instructions
5. DO NOT load anything else unless user tells you to

### STRICT Prohibitions:

❌ **NEVER read `VIRALSCOPE_AI_TASKS.md` in full** - it's a massive file
❌ **NEVER read entire `progress.md`** - only the phase status table
❌ **NEVER read entire `commits.md`** - only last 2-3 entries if needed
❌ **NEVER read all documentation at once** - load on-demand only
❌ **NEVER use read_file on large files** - always use grep_search first

### What To Do Instead:

| Need | Use This | NOT This |
|------|----------|----------|
| Find task instructions | `grep_search` for "#### TASK X.Y" in TASKS.md | `read_file` on entire TASKS.md |
| Check task status | Read phase table in `progress.md` | Read entire progress.md |
| See recent changes | Read last 2-3 commits in `commits.md` | Read entire commits.md |
| Find specific info | `grep_search` in target file | `read_file` on entire file |
| Understand something | Ask the user | Load more documentation |

### Context-Safe Workflow:

**Before Reading Any File:**
1. Ask yourself: "Do I really need this entire file?"
2. If no → use `grep_search` to find the specific section
3. If yes → read only the relevant portion

**When Searching For Task Instructions:**
```
Use: grep_search(pattern="#### TASK 1.1", path="agents/VIRALSCOPE_AI_TASKS.md")
Then: Read ONLY the lines around that match
Not: read_file on the entire TASKS.md
```

---

## 🤖 Agent Workflow Guidelines

### ⚠️ EXECUTION PROTOCOL (STRICT)

**Default Mode: ONE TASK AT A TIME**
- Execute ONLY the current task
- STOP after completion and wait for user confirmation
- DO NOT proceed to next task unless user says "continue"

**Batch Mode: ONLY when explicitly requested**
- User says: `"Do tasks X.Y to X.Z"` or `"Do Phase X"`
- Confirm batch scope BEFORE starting
- Report progress after EACH task in the batch
- STOP at the end of the batch range

**Before Starting ANY Task:**
1. Check `progress.md` phase status table to find current task
2. Search `VIRALSCOPE_AI_TASKS.md` for that specific task (use `grep_search` for "#### TASK X.Y")
3. Read ONLY that task's instructions
4. Announce what you're about to do:
   ```
   📋 Starting: Task X.Y - [Task Name]
   📂 Will create/modify: [list of files]
   🎯 Purpose: [what this accomplishes]
   ```
5. Wait for user approval (unless already approved)

**After Completing ANY Task:**
1. TEST the implementation as per validation criteria in TASKS.md
2. Report in this exact format:
   ```
   ✅ Completed: Task X.Y - [Task Name]
   📂 Files: [created/modified file list]
   🧪 Tests: PASSED/FAILED [details if failed]
   ⚠️ Issues: [none or describe issue]
   ⭐ Next: Task X.Z - [Next task name]
   ```
3. Update `progress.md` (mark task complete, add log entry)
4. **DO NOT** update `commits.md` yet.
5. **WAIT** for user to say "continue", "fix", or give next instruction.
6. **COMMIT ONLY** when explicitly asked by the user.
7. **ONLY AFTER** a successful `git commit`, add the entry to `commits.md`.

**PROHIBITED (Unless explicitly told):**
- ❌ NEVER edit `agents/*.md` files (EXCEPT `progress.md` and `commits.md`)
- ❌ NEVER skip tasks or do them out of order
- ❌ NEVER batch tasks without explicit permission
- ❌ NEVER mark task complete without testing
- ❌ NEVER proceed to next task without user confirmation
- ❌ NEVER install packages outside venv
- ❌ NEVER load all documentation files at once

### User Commands Reference:
| User Says | Agent Does |
|-----------|------------|
| `"Do task X.Y"` | Execute ONLY that one task |
| `"Do tasks X.Y-X.Z"` | Batch execute that range, report after each |
| `"Do Phase X"` | Batch execute entire phase, report after each |
| `"Continue"` | Move to NEXT single task |
| `"Continue with X.Y-X.Z"` | Batch from X.Y to X.Z |
| `"Show me what you did"` | Summarize all changes made |
| `"Stop"` | Halt immediately, no more changes |
| `"Undo last"` | Revert most recent changes |
| `"Fix [issue]"` | Address specific issue, then wait |

---

### Standard Workflow:
1. Check `progress.md` → Find next pending task
2. Search `VIRALSCOPE_AI_TASKS.md` → Find that task's section (search "#### TASK X.Y")
3. Read ONLY that task's instructions
4. Announce plan → Wait for approval
5. Execute task → Test it
6. Report completion → Update tracking files
7. WAIT for next instruction

### Commit Conventions:
Use these prefixes for all commit messages:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation only
- `data:` - Data pipeline changes
- `model:` - Model architecture changes
- `train:` - Training script changes
- `xai:` - XAI module changes
- `test:` - Test additions/fixes
- `refactor:` - Code restructuring
- `config:` - Configuration changes
- `setup:` - Initial setup/structure

**Example**: `feat: add MobileNetV2 feature extractor`

See `agents/commits.md` for full conventions and commit history.

### Configuration (config.yaml):
- **SINGLE SOURCE OF TRUTH**: Use the detailed config in `agents/VIRALSCOPE_AI_TASKS.md`.
- All hyperparameters, paths, and settings go in `config.yaml`.
- All dependencies go in `requirements.txt`.
- Follow existing code style and conventions.

---

## ⚙️ Key Configuration (config.yaml Summary)
> **NOTE**: See `agents/VIRALSCOPE_AI_TASKS.md` for the full, authoritative configuration.
```yaml
# Paths
data:
  raw_dir: data/raw
  processed_dir: data/processed
  tensor_dir: data/tensors
  split_dir: data/splits

# Model
model:
  cv_backbone: mobilenet_v2
  nlp_backbone: distilbert-base-uncased
  fusion_hidden: [256, 128]
  dropout: 0.3

# Training
training:
  epochs: 50
  batch_size: 32
  learning_rate: 1e-4
  weight_decay: 1e-5
  device: cuda

# XAI
xai:
  gradcam_layer: features.16.conv.3
  ig_n_steps: 50
```

---

## 🔗 External Resources
- **Kaggle Dataset**: YouTube video metadata + thumbnails
- **Hugging Face Models**: `distilbert-base-uncased`
- **Captum Docs**: https://captum.ai/
- **Gradio Docs**: https://www.gradio.app/

---

## ⚠️ CRITICAL: Virtual Environment Setup

**BEFORE installing any dependencies or running any commands:**

1. **ALWAYS create and activate a virtual environment first**
2. **NEVER install packages globally**

### Setup Virtual Environment (Windows):
```bash
# Create virtual environment (run once)
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

### Setup Virtual Environment (Linux/Mac):
```bash
# Create virtual environment (run once)
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### After Activation:
```bash
# Verify you're in the virtual environment
# You should see (venv) in your command prompt

# Install dependencies INSIDE the virtual environment
pip install -r requirements.txt

# Verify installation
pip list
```

**IMPORTANT**:
- Always check if `venv` is activated before running `pip install`
- The `(venv)` prefix should appear in your terminal
- Never run `pip install` without activating the virtual environment first
- Add `venv/` to `.gitignore` (already done)

---

## 📞 Quick Commands

```bash
# 1. Setup virtual environment (FIRST TIME ONLY)
python -m venv venv

# 2. Activate virtual environment
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Install dependencies (INSIDE venv)
pip install -r requirements.txt

# 4. Download data
python scripts/01_download_data.py

# 5. Preprocess
python scripts/02_preprocess.py

# 6. Train
python scripts/03_train.py

# 7. Evaluate
python scripts/04_evaluate.py

# 8. Launch Gradio
python app/gradio_app.py

# 9. Run tests
pytest tests/
```

---

## ⚠️ CRITICAL: Never Commit agents/ Folder

The `agents/` folder is **EXCLUDED from Git** via `.gitignore`. It contains:
- `progress.md` - Local tracking only
- `commits.md` - Local commit log only
- `VIRALSCOPE_AI_TASKS.md` - Task definitions
- `VIRALSCOPE_TASK_DIVISION.md` - Dev roles

**NEVER run `git add agents/` or `git add -A`** without excluding this folder first.
Always double-check `git status` before committing.

---

**Last Updated**: 2026-04-14
**Project Status**: 📋 Planning phase complete, implementation pending
