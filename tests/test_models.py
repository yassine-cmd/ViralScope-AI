import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── CLIP-based extractor tests ──────────────────────────────────

def _load_clip_components():
    """Helper: load CLIP model once and return split components."""
    from transformers import CLIPModel
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    return clip


def test_cv_extractor_shape():
    from models.cv_extractor import CVExtractor
    clip = _load_clip_components()
    model = CVExtractor(clip.vision_model, clip.visual_projection, freeze=True)
    x = torch.randn(4, 3, 224, 224)
    out = model(x)
    assert out.shape == (4, 512), f"Expected (4, 512), got {out.shape}"


def test_cv_extractor_freeze():
    from models.cv_extractor import CVExtractor
    clip = _load_clip_components()
    model = CVExtractor(clip.vision_model, clip.visual_projection, freeze=True)
    for param in model.parameters():
        assert param.requires_grad == False, "Parameters should be frozen"


def test_cv_extractor_normalized():
    from models.cv_extractor import CVExtractor
    clip = _load_clip_components()
    model = CVExtractor(clip.vision_model, clip.visual_projection, freeze=True)
    x = torch.randn(4, 3, 224, 224)
    out = model(x)
    norms = out.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
        f"Outputs should be L2-normalized, got norms: {norms}"


def test_nlp_extractor_shape():
    from models.nlp_extractor import NLPExtractor
    clip = _load_clip_components()
    model = NLPExtractor(clip.text_model, clip.text_projection, freeze=True)
    input_ids = torch.randint(0, 49408, (4, 77))  # CLIP vocab size & context length
    attention_mask = torch.ones(4, 77, dtype=torch.long)
    out = model(input_ids, attention_mask)
    assert out.shape == (4, 512), f"Expected (4, 512), got {out.shape}"


def test_nlp_extractor_freeze():
    from models.nlp_extractor import NLPExtractor
    clip = _load_clip_components()
    model = NLPExtractor(clip.text_model, clip.text_projection, freeze=True)
    for param in model.parameters():
        assert param.requires_grad == False, "Parameters should be frozen"


def test_nlp_extractor_normalized():
    from models.nlp_extractor import NLPExtractor
    clip = _load_clip_components()
    model = NLPExtractor(clip.text_model, clip.text_projection, freeze=True)
    input_ids = torch.randint(0, 49408, (4, 77))
    attention_mask = torch.ones(4, 77, dtype=torch.long)
    out = model(input_ids, attention_mask)
    norms = out.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
        f"Outputs should be L2-normalized, got norms: {norms}"


# ── Fusion MLP tests ────────────────────────────────────────────

def test_fusion_mlp_shape():
    from models.fusion_model import FusionMLP
    model = FusionMLP(feature_dim=512, hidden_layers=[256, 64], dropout=0.2, activation="GELU")
    img_feat = torch.randn(4, 512)
    txt_feat = torch.randn(4, 512)
    # Normalize inputs like CLIP would
    img_feat = torch.nn.functional.normalize(img_feat, p=2, dim=1)
    txt_feat = torch.nn.functional.normalize(txt_feat, p=2, dim=1)
    out = model(img_feat, txt_feat)
    assert out.shape == (4,), f"Expected (4,), got {out.shape}"


def test_fusion_mlp_interactions():
    """Verify the fusion head produces different outputs for different inputs."""
    from models.fusion_model import FusionMLP
    model = FusionMLP(feature_dim=512)
    model.eval()

    img1 = torch.nn.functional.normalize(torch.randn(1, 512), p=2, dim=1)
    txt1 = torch.nn.functional.normalize(torch.randn(1, 512), p=2, dim=1)
    txt2 = torch.nn.functional.normalize(torch.randn(1, 512), p=2, dim=1)

    out1 = model(img1, txt1)
    out2 = model(img1, txt2)
    assert not torch.allclose(out1, out2, atol=1e-4), \
        "Fusion should produce different outputs for different text inputs"


# ── Loss tests ──────────────────────────────────────────────────

def test_focal_loss():
    from models.losses import FocalLoss
    loss_fn = FocalLoss(gamma=2.0)
    logits = torch.tensor([0.0, 10.0, -10.0])
    targets = torch.tensor([1.0, 1.0, 0.0])
    loss = loss_fn(logits, targets)
    assert loss.dim() == 0, "Loss should be scalar"
    assert loss > 0, "Loss should be positive"


def test_focal_loss_with_alpha():
    from models.losses import FocalLoss
    alpha = torch.tensor([1.0, 2.0])
    loss_fn = FocalLoss(alpha=alpha, gamma=2.0)
    logits = torch.tensor([0.0, 10.0])
    targets = torch.tensor([1.0, 0.0])
    loss = loss_fn(logits, targets)
    assert loss.dim() == 0, "Loss should be scalar"
    assert loss > 0, "Loss should be positive"


# ── End-to-end model test ───────────────────────────────────────

def test_multimodal_model_shape():
    from models.multimodal import ViralScopeModel

    config = {
        'model': {
            'clip': {
                'checkpoint': 'openai/clip-vit-base-patch32',
                'feature_dim': 512,
                'max_seq_length': 77,
                'freeze_backbone': True,
            },
            'fusion': {
                'hidden_layers': [256, 64],
                'dropout': 0.2,
                'activation': 'GELU',
            },
        }
    }

    model = ViralScopeModel(config)
    images = torch.randn(4, 3, 224, 224)
    input_ids = torch.randint(0, 49408, (4, 77))
    attention_mask = torch.ones(4, 77, dtype=torch.long)

    logits = model(images, input_ids, attention_mask)
    assert logits.shape == (4,), f"Expected (4,), got {logits.shape}"

    probs = model.predict_proba(images, input_ids, attention_mask)
    assert (probs >= 0).all() and (probs <= 1).all(), "Probabilities should be in [0, 1]"
