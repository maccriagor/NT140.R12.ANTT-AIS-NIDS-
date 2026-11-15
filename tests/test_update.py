"""Unit tests for extend_head, restore_backbone, and incremental update mechanics (Plan Week 4.3 / P2)."""
import copy
import numpy as np
import torch
import torch.nn as nn

import pipeline as pl
import skeleton as sk


def test_extend_head_preserves_old_rows():
    torch.manual_seed(42)
    old_head = nn.Linear(256, 11)
    new_head = pl.extend_head(old_head)

    assert new_head.in_features == 256
    assert new_head.out_features == 12
    assert torch.equal(new_head.weight[:-1], old_head.weight)
    assert torch.equal(new_head.bias[:-1], old_head.bias)

    # Check forward pass on first 11 logits matches exactly
    x = torch.randn(8, 256)
    with torch.no_grad():
        out_old = old_head(x)
        out_new = new_head(x)
    assert torch.allclose(out_old, out_new[:, :-1], atol=1e-6)


def test_restore_backbone_restores_exact_weights_and_keeps_head():
    sk.set_seed(42)
    model_b = sk.build_model("mlp", 11)
    model_u = copy.deepcopy(model_b)
    model_u.head = pl.extend_head(model_u.head)

    # Perturb all weights in model_u
    with torch.no_grad():
        for p in model_u.parameters():
            p.add_(torch.randn_like(p) * 0.5)
        head_weight_after_perturb = model_u.head.weight.clone()
        head_bias_after_perturb = model_u.head.bias.clone()

    pl.restore_backbone(model_u, model_b)

    # Backbone must match model_b exactly
    for (kb, pb), (ku, pu) in zip(model_b.backbone.named_parameters(), model_u.backbone.named_parameters()):
        assert kb == ku
        assert torch.equal(pb, pu), f"backbone parameter {kb} was not restored"

    # Head must still hold the perturbed values
    assert torch.equal(model_u.head.weight, head_weight_after_perturb)
    assert torch.equal(model_u.head.bias, head_bias_after_perturb)


def test_temperature_scaling_monotonic():
    sk.set_seed(42)
    model = sk.build_model("mlp", 5)
    X = np.random.randint(0, 256, size=(32, 1500), dtype=np.uint8)
    conf_t1, pred_t1 = pl.predict_conf(model, X, temperature=1.0)
    conf_t2, pred_t2 = pl.predict_conf(model, X, temperature=2.0)

    # Temperature scaling preserves argmax predictions while softening confidence (>1.0)
    assert np.array_equal(pred_t1, pred_t2)
    assert np.all(conf_t2 <= conf_t1 + 1e-6)
