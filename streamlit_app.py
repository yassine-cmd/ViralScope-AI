"""
ViralScope AI — Streamlit inference app.

User inputs:  YouTube URL  +  Channel average views
Everything else (title, category, publish time) is fetched from the
YouTube Data API v3 automatically.
"""

import os
import sys
import re
import gc
import warnings
from datetime import datetime

# Ensure src/ is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

warnings.filterwarnings("ignore")

# ── 1. CONFIGURATION ─────────────────────────────────────────────────────────
MODEL_PATH         = "models/best_model.joblib"
FEATURE_NAMES_PATH = "models/feature_names.joblib"

SIGLIP_CHECKPOINT  = "google/siglip-base-patch16-224"
SIGLIP_MAX_SEQ     = 64

# Load probe pairs from config
import yaml
_config = yaml.safe_load(open("config.yaml"))
PROBE_PAIRS = _config["model"].get("probe_pairs", [])

VISUAL_STATS_COLUMNS = ["brightness", "saturation", "colorfulness",
                        "red_ratio", "edge_density", "face_count", "face_size_ratio"]


# ── 3. YOUTUBE DATA API ──────────────────────────────────────────────────────
def extract_video_id(url: str):
    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
        r"shorts/([a-zA-Z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


def fetch_video_metadata(video_id: str, api_key: str) -> dict | None:
    """Calls YouTube Data API v3 to get title, categoryId, and publishedAt."""
    import requests

    url = (
        "https://www.googleapis.com/youtube/v3/videos"
        f"?part=snippet&id={video_id}&key={api_key}"
    )
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
    except Exception:
        return None

    items = data.get("items", [])
    if not items:
        return None

    snippet    = items[0]["snippet"]
    title      = snippet.get("title", "")
    cat_id     = int(snippet.get("categoryId", 0))
    published  = snippet.get("publishedAt", "")

    hour, day_of_week, is_weekend = 12, 3, 0
    if published:
        try:
            dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
            hour        = dt.hour
            day_of_week = dt.weekday()
            is_weekend  = int(day_of_week >= 5)
        except Exception:
            pass

    return {
        "title":       title,
        "category_id": cat_id,
        "hour":        hour,
        "day_of_week": day_of_week,
        "is_weekend":  is_weekend,
    }


def download_thumbnail(video_id: str):
    import requests
    from io import BytesIO
    from PIL import Image

    for res in ["maxresdefault", "hqdefault", "mqdefault"]:
        url = f"https://img.youtube.com/vi/{video_id}/{res}.jpg"
        try:
            r = requests.get(url, timeout=6)
            if r.status_code == 200:
                return Image.open(BytesIO(r.content)).convert("RGB")
        except Exception:
            continue
    return None


# ── 3. VISUAL STATS (pure numpy — no cv2 for Streamlit compatibility) ────────
def get_visual_stats(img):
    import cv2
    import numpy as np

    img_resized = img.resize((160, 160))
    img_rgb = np.array(img_resized, dtype=np.float64)
    img_rgb_u8 = np.array(img_resized)
    img_hsv = cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2HSV)
    img_gray = cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2GRAY)

    brightness  = float(img_hsv[:, :, 2].mean()) / 255.0
    saturation  = float(img_hsv[:, :, 1].mean()) / 255.0

    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    rg = r - g
    yg = g - b                                       # matches training
    colorfulness = float(np.sqrt(rg**2 + yg**2).mean()) / 255.0

    red_ratio = float(r.sum()) / (img_rgb.sum() + 1e-5)

    edges = cv2.Canny(img_gray, 50, 150)             # matches training
    edge_density = float(edges.sum()) / (edges.size * 255.0)

    # face_count / face_size_ratio: detection using Haar Cascades
    import os
    cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    face_count = 0
    face_size_ratio = 0.0
    
    if not face_cascade.empty():
        # 1.1 scale factor is more thorough for small images; 6 neighbors reduces false positives
        faces = face_cascade.detectMultiScale(img_gray, 1.1, 6)
        face_count = len(faces)
        
        if face_count > 0:
            total_face_area = sum(w * h for (x, y, w, h) in faces)
            face_size_ratio = float(total_face_area) / (img_gray.shape[0] * img_gray.shape[1])

    return [brightness, saturation, colorfulness, red_ratio, edge_density, face_count, face_size_ratio]


# ── 4. MODEL LOADING (cached, lazy imports) ──────────────────────────────────
@st.cache_resource(show_spinner="Loading AI models… (first run only)")
def load_models():
    import joblib
    import torch
    from transformers import SiglipModel, SiglipProcessor
    from src.stacking_trainer import SoftVotingEnsemble
    import __main__
    __main__.SoftVotingEnsemble = SoftVotingEnsemble

    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURE_NAMES_PATH):
        return None, None, None, None, None

    m_data    = joblib.load(MODEL_PATH)
    f_names   = joblib.load(FEATURE_NAMES_PATH)
    ensemble  = m_data["model"]
    threshold = m_data["threshold"]

    sig_model  = SiglipModel.from_pretrained(SIGLIP_CHECKPOINT).eval()
    processor  = SiglipProcessor.from_pretrained(SIGLIP_CHECKPOINT)

    return ensemble, threshold, f_names, sig_model, processor


# ── 5. FEATURE BUILDER (lazy imports) ────────────────────────────────────────
def build_feature_vector(img, title, category_id, hour, day_of_week,
                         is_weekend, avg_views, model, processor, f_names):
    import torch
    import torch.nn.functional as F
    import numpy as np
    import pandas as pd

    max_length = SIGLIP_MAX_SEQ

    pos_prompts = [p[0] for p in PROBE_PAIRS]
    neg_prompts = [p[1] for p in PROBE_PAIRS]

    def _to_tensor(output):
        """Safely extract tensor from model output."""
        if isinstance(output, torch.Tensor):
            return output
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state[:, 0]
        raise TypeError(f"Cannot extract embedding tensor from {type(output)}")

    with torch.no_grad():
        img_in    = processor(images=[img], return_tensors="pt")
        img_emb_t = F.normalize(_to_tensor(model.get_image_features(**img_in)), dim=-1)

        txt_in    = processor(text=[title], padding="max_length",
                              max_length=max_length, truncation=True, return_tensors="pt")
        txt_emb_t = F.normalize(_to_tensor(model.get_text_features(**txt_in)), dim=-1)

        pos_in    = processor(text=pos_prompts, padding="max_length",
                              max_length=max_length, truncation=True, return_tensors="pt")
        neg_in    = processor(text=neg_prompts, padding="max_length",
                              max_length=max_length, truncation=True, return_tensors="pt")
        pos_emb_t = F.normalize(_to_tensor(model.get_text_features(**pos_in)), dim=-1)
        neg_emb_t = F.normalize(_to_tensor(model.get_text_features(**neg_in)), dim=-1)

        probes_t = (img_emb_t @ pos_emb_t.T) - (img_emb_t @ neg_emb_t.T)

        img_emb = img_emb_t.squeeze(0).cpu().numpy()
        txt_emb = txt_emb_t.squeeze(0).cpu().numpy()
        probes  = probes_t.squeeze(0).cpu().numpy()

    diff    = np.abs(img_emb - txt_emb)
    prod    = img_emb * txt_emb
    cos_sim = float(np.dot(img_emb, txt_emb))
    emb_dim = img_emb.shape[0]

    feat = {}
    for i in range(emb_dim):
        feat[f"emb_img_{i}"]  = float(img_emb[i])
        feat[f"emb_txt_{i}"]  = float(txt_emb[i])
        feat[f"emb_diff_{i}"] = float(diff[i])
        feat[f"emb_prod_{i}"] = float(prod[i])

    feat["cos_sim"] = cos_sim

    for i, v in enumerate(probes):
        feat[f"probe_{i}"] = float(v)

    v_stats = get_visual_stats(img)
    for col, val in zip(VISUAL_STATS_COLUMNS, v_stats):
        feat[col] = val

    cat_col = f"cat_{category_id}"
    if cat_col in f_names:
        feat[cat_col] = 1

    channel_log_power = float(np.log10(max(avg_views, 1) + 1))
    feat["hour_of_day"]       = hour
    feat["day_of_week"]       = day_of_week
    feat["is_weekend"]        = is_weekend
    feat["channel_log_power"] = channel_log_power
    feat["probe_x_channel"]   = float(np.sum(probes)) * channel_log_power
    feat["cos_x_weekend"]     = cos_sim * is_weekend

    final = {f: feat.get(f, 0.0) for f in f_names}
    X = pd.DataFrame([final])

    del img_emb_t, txt_emb_t, pos_emb_t, neg_emb_t, probes_t
    del pos_in, neg_in
    gc.collect()

    return X


# ── 6. STREAMLIT UI ──────────────────────────────────────────────────────────
def main():
    # Set layout to "wide" to support the horizontal split properly
    st.set_page_config(page_title="ViralScope AI", page_icon="🚀", layout="wide")

    api_key = st.secrets.get("YOUTUBE_API_KEY", "")

    st.markdown(
        """
        <style>
        .hero { text-align:center; padding:1.5rem 0 0.5rem; }
        .hero h1 { font-size:2.4rem; font-weight:800; margin-bottom:0.2rem; }
        .hero p  { color:#888; font-size:1rem; }
        </style>
        <div class='hero'>
            <h1>🚀 ViralScope AI</h1>
            <p>Paste a YouTube link — we fetch everything else automatically.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # Split the UI horizontally
    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.subheader("📥 Inputs & Data")
        url = st.text_input("🔗 YouTube URL", placeholder="https://youtu.be/...")
        avg_views = st.number_input(
            "📊 Channel Average Views per Video",
            min_value=100, max_value=100_000_000, value=50_000, step=1_000,
            help="Rough average view count for the channel.",
        )

        analyze_clicked = st.button("🔍 Analyze & Predict", use_container_width=True, type="primary")

    if not analyze_clicked:
        with left_col:
            st.info("Paste a YouTube URL and enter channel average views, then click **Analyze & Predict**.")
        return

    # Process Data in the Left Column
    with left_col:
        vid = extract_video_id(url)
        if not vid:
            st.error("❌ Could not parse a video ID from that URL.")
            return

        if not api_key:
            st.error("❌ YouTube API key not found. Add it to `.streamlit/secrets.toml`.")
            return

        with st.spinner("Fetching video metadata from YouTube…"):
            meta = fetch_video_metadata(vid, api_key)

        if meta is None:
            st.error("❌ Could not fetch video metadata. Check the URL or API key.")
            return

        st.success(f"**Title:** {meta['title']}")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Category ID", meta["category_id"])
        col_b.metric("Publish Hour", f"{meta['hour']}:00")
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        col_c.metric("Publish Day", days[meta["day_of_week"]])

        with st.spinner("Fetching thumbnail…"):
            thumb = download_thumbnail(vid)

        if thumb is None:
            st.error("❌ Could not download the thumbnail.")
            return

        st.image(thumb, caption="Video Thumbnail", use_container_width=True)

        with st.spinner("Loading AI models (cached after first run)…"):
            ensemble, threshold, f_names, sig_model, processor = load_models()

        if ensemble is None:
            st.error(
                f"❌ Model files not found.\n\n"
                f"Expected:\n- `{MODEL_PATH}`\n- `{FEATURE_NAMES_PATH}`\n\n"
                "Run `python run_pipeline.py` to generate them."
            )
            return

        with st.spinner("Extracting SigLIP embeddings & computing features…"):
            try:
                X = build_feature_vector(
                    img=thumb,
                    title=meta["title"],
                    category_id=meta["category_id"],
                    hour=meta["hour"],
                    day_of_week=meta["day_of_week"],
                    is_weekend=meta["is_weekend"],
                    avg_views=float(avg_views),
                    model=sig_model,
                    processor=processor,
                    f_names=f_names,
                )
            except Exception as exc:
                st.error(f"❌ Feature extraction failed: {exc}")
                st.exception(exc)
                return

        with st.spinner("Running ensemble classifier…"):
            try:
                prob_raw = float(ensemble.predict_proba(X)[0, 1])
            except Exception as exc:
                st.error(f"❌ Prediction failed: {exc}")
                st.exception(exc)
                return

    # ── Display Results in the Right Column ──
    with right_col:
        st.subheader("🎯 Prediction Result")

        prob   = prob_raw
        margin = prob - threshold   # positive = above threshold, negative = below

        # Graduated label based on how far the score sits from the threshold
        if margin >= 0.30:
            verdict_title = "🔥 Strong viral potential"
            verdict_body  = ("The score is well above the training threshold — the model "
                             "sees clear signals associated with trending content.")
            verdict_style = "success"
        elif margin >= 0.05:
            verdict_title = "📈 Potentially viral"
            verdict_body  = ("The score is comfortably above the threshold, but not by a "
                             "large margin. Some viral signals are present.")
            verdict_style = "success"
        elif margin >= 0:
            verdict_title = "🌱 Neutral viral potential"
            verdict_body  = ("The score just clears the threshold. Treat this as a weak "
                             "positive signal — small changes to the thumbnail or title "
                             "could tip it either way.")
            verdict_style = "warning"
        elif margin >= -0.10:
            verdict_title = "📉 Unlikely to go viral"
            verdict_body  = ("The score falls just short of the threshold. The model does "
                             "not detect strong viral signals, but the gap is small.")
            verdict_style = "warning"
        else:
            verdict_title = "❌ Low viral potential"
            verdict_body  = ("The score is well below the threshold. The model sees few "
                             "features associated with trending content.")
            verdict_style = "error"

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Viral Score", f"{prob:.3f}",
                      help=f"Raw model probability. Training threshold: {threshold:.3f}.")
            st.progress(min(prob, 1.0))
            st.caption(f"Threshold: {threshold:.3f}  |  Margin: {margin:+.3f}")
        with c2:
            getattr(st, verdict_style)(f"**{verdict_title}**\n\n{verdict_body}")

        # ── Human-readable signal breakdown ──────────────────────────────────────
        st.divider()
        st.subheader("🔍 What the model detected")

        row = X.iloc[0]

        def simple_bar(label, value, left_label, right_label, low=0.0, high=1.0):
            norm = float((value - low) / (high - low))
            norm = max(0.0, min(1.0, norm))
            pct  = norm * 100

            bar_html = f"""
            <div style="display:flex; align-items:center; gap:6px; margin-bottom:4px;">
              <span style="width:80px; text-align:right; font-size:0.75rem; color:#888;">{left_label}</span>
              <div style="flex:1; height:10px; background:#1e1e2e; border-radius:5px; overflow:hidden;">
                <div style="width:{pct}%; height:100%;
                            background:linear-gradient(90deg,#4c9be8,#7eb8f7); border-radius:5px;"></div>
              </div>
              <span style="width:80px; font-size:0.75rem; color:#888;">{right_label}</span>
            </div>"""

            st.markdown(
                f"<div style='margin-bottom:2px;'><strong style='font-size:0.85rem'>{label}</strong></div>",
                unsafe_allow_html=True,
            )
            st.markdown(bar_html, unsafe_allow_html=True)
            st.markdown("<div style='margin-bottom:10px'></div>", unsafe_allow_html=True)

        st.markdown("#### 🎨 Visual properties")
        col_c, col_d = st.columns(2)
        with col_c:
            simple_bar("Brightness",   row.get("brightness",   0), "dark",    "bright",    0.0, 1.0)
            simple_bar("Colorfulness", row.get("colorfulness", 0), "muted",   "vivid",     0.0, 0.4)
        with col_d:
            simple_bar("Saturation",   row.get("saturation",   0), "grey",    "saturated", 0.0, 1.0)
            simple_bar("Edge density", row.get("edge_density", 0), "minimal", "cluttered", 0.0, 0.3)

        st.markdown("#### 🔗 Title–thumbnail alignment")
        simple_bar("Semantic match", row.get("cos_sim", 0),
                   "mismatched", "well aligned", low=-0.1, high=0.4)

        with st.expander("🔬 Raw feature vector (dev only)"):
            st.dataframe(X.T.rename(columns={0: "value"}), use_container_width=True)


if __name__ == "__main__":
    main()