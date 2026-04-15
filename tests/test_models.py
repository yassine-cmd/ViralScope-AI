import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_cv_extractor_shape():
    from models.cv_extractor import CVExtractor
    model = CVExtractor(pretrained=False, freeze=True)
    x = torch.randn(4, 3, 224, 224)
    out = model(x)
    assert out.shape == (4, 1280), f"Expected (4, 1280), got {out.shape}"


def test_cv_extractor_freeze():
    from models.cv_extractor import CVExtractor
    model = CVExtractor(pretrained=False, freeze=True)
    for param in model.features.parameters():
        assert param.requires_grad == False, "Parameters should be frozen"


def test_nlp_extractor_shape():
    from models.nlp_extractor import NLPExtractor
    model = NLPExtractor(freeze=True)
    input_ids = torch.randint(0, 30522, (4, 64))
    attention_mask = torch.ones(4, 64, dtype=torch.long)
    out = model(input_ids, attention_mask)
    assert out.shape == (4, 768), f"Expected (4, 768), got {out.shape}"


def test_nlp_extractor_freeze():
    from models.nlp_extractor import NLPExtractor
    model = NLPExtractor(freeze=True)
    for param in model.backbone.parameters():
        assert param.requires_grad == False, "Parameters should be frozen"


def test_fusion_mlp_shape():
    from models.fusion_model import FusionMLP
    model = FusionMLP(cv_dim=1280, nlp_dim=768)
    cv_feat = torch.randn(4, 1280)
    nlp_feat = torch.randn(4, 768)
    out = model(cv_feat, nlp_feat)
    assert out.shape == (4,), f"Expected (4,), got {out.shape}"


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


def test_multimodal_model_shape():
    import yaml
    from models.multimodal import ViralScopeModel
    
    config = {
        'model': {
            'cv': {
                'pretrained': False,
                'feature_dim': 1280,
                'freeze_backbone': True
            },
            'nlp': {
                'checkpoint': 'distilbert-base-uncased',
                'feature_dim': 768,
                'freeze_backbone': True
            },
            'fusion': {
                'hidden_layers': [512, 128],
                'dropout': 0.4,
                'activation': 'ReLU'
            }
        }
    }
    
    model = ViralScopeModel(config)
    images = torch.randn(4, 3, 224, 224)
    input_ids = torch.randint(0, 30522, (4, 64))
    attention_mask = torch.ones(4, 64, dtype=torch.long)
    
    logits = model(images, input_ids, attention_mask)
    assert logits.shape == (4,), f"Expected (4,), got {logits.shape}"
    
    probs = model.predict_proba(images, input_ids, attention_mask)
    assert (probs >= 0).all() and (probs <= 1).all(), "Probabilities should be in [0, 1]"
