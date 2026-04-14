import torch
import pytest
from models.losses import FocalLoss
from models.fusion_model import FusionMLP
from models.multimodal import ViralScopeModel


def test_focal_loss():
    loss_fn = FocalLoss(gamma=2.0)
    logits = torch.tensor([0.0, 10.0, -10.0])
    targets = torch.tensor([1.0, 1.0, 0.0])
    loss = loss_fn(logits, targets)
    assert loss.dim() == 0
    assert loss > 0


def test_focal_loss_with_alpha():
    alpha = torch.tensor([1.0, 2.0])
    loss_fn = FocalLoss(alpha=alpha, gamma=2.0)
    logits = torch.tensor([0.0, 5.0])
    targets = torch.tensor([1.0, 0.0])
    loss = loss_fn(logits, targets)
    assert loss.dim() == 0
    assert loss > 0


def test_fusion_mlp_shape():
    model = FusionMLP(cv_dim=1280, nlp_dim=768)
    cv_feat = torch.randn(4, 1280)
    nlp_feat = torch.randn(4, 768)
    out = model(cv_feat, nlp_feat)
    assert out.shape == (4,)


def test_multimodal_model_shape():
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    config['model']['cv']['pretrained'] = False
    config['model']['nlp']['checkpoint'] = 'distilbert-base-uncased'
    
    model = ViralScopeModel(config)
    images = torch.randn(4, 3, 224, 224)
    input_ids = torch.randint(0, 30522, (4, 64))
    attention_mask = torch.ones(4, 64, dtype=torch.long)
    
    logits = model(images, input_ids, attention_mask)
    assert logits.shape == (4,)
    
    probs = model.predict_proba(images, input_ids, attention_mask)
    assert (probs >= 0).all() and (probs <= 1).all()
