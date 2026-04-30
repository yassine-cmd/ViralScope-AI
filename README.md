# ViralScope AI

Modular NLP pipeline for YouTube viral video prediction using SigLIP embeddings and a stacking ensemble model.

The Dataset used: https://www.kaggle.com/datasets/datasnaek/youtube-new

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Place raw YouTube CSV data in data/raw/ (see Data Setup below)

# Run the pipeline
python run_pipeline.py

# Run the Streamlit app
streamlit run streamlit_app.py
```

## Data Setup

1. **The pipeline auto-creates folders:** `data/raw/`, `data/processed/`, `data/tensors/`, `models/`, `results/`

2. **Place raw YouTube CSVs** in `data/raw/` before running:
   - `USvideos.csv`, `GBvideos.csv`, `CAvideos.csv` (Kaggle Trending YouTube Video Statistics)
   - Required columns: `video_id`, `title`, `thumbnail_link`, `views`, `likes`, `dislikes`, `comment_count`

3. **Pipeline workflow:**
   - Loads and cleans raw CSV data → `data/processed/clean_dataset.csv`
   - Computes viral labels → `data/processed/labeled_dataset.csv`
   - Downloads thumbnails → `data/raw/thumbnails/`
   - Extracts SigLIP embeddings → `data/tensors/`
   - Trains stacking ensemble → `models/best_model.joblib`
   - Evaluates → `results/eval_results.csv`

## Configuration

Edit `config.yaml` to customize:
- `data/sampling_strategy` — "balanced" | "all" | "imbalanced"
- `model/stacking/n_folds` — Number of cross-validation folds
- `model/stacking/base_models` — Base learners for stacking
- `model/probe_pairs` — Contrastive text pairs for thumbnail scoring

## Project Structure

```
├── config.yaml           # Pipeline configuration
├── run_pipeline.py      # CLI entry point
├── streamlit_app.py     # Web inference app
├── pipeline_notebook.ipynb  # Jupyter notebook pipeline
├── requirements.txt    # Python dependencies
├── src/                # Pipeline modules
│   ├── config_loader.py
│   ├── data_loader.py
│   ├── data_splitter.py
│   ├── embedding_extractor.py
│   ├── feature_builder.py
│   ├── label_engine.py
│   ├── model_evaluator.py
│   ├── model_persistence.py
│   ├── pipeline.py
│   ├── stacking_trainer.py
│   ├── thumbnail_manager.py
│   ├── title_feature_extractor.py
│   └── visual_stats_extractor.py
├── data/                # Auto-created (add raw CSVs here)
├── models/              # Trained model outputs
└── results/             # Evaluation outputs
```

## Model

- **Embedding:** google/siglip-base-patch16-224 (768-d)
- **Probe pairs:** 6 contrastive text pairs for thumbnail scoring
- **Features:** SigLIP image + text embeddings, probe scores, visual stats, title features, metadata
- **Ensemble:** Stacking with LR, XGBoost, LightGBM, RandomForest meta-learned by XGBoost

**⚠️ Note for Streamlit App:** In order to make the fetching process work, you need to provide a YouTube API key. Create a `.streamlit` folder in the root directory containing a `secrets.toml` file with your key:
```toml
# .streamlit/secrets.toml
YOUTUBE_API_KEY = "your_youtube_api_key_here"
