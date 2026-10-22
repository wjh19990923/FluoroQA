"""Thin interface to upstream LPIPS; no third-party source is vendored."""
import torch
from torch import nn


class FluoroQA(nn.Module):
    """VGG LPIPS distance: larger values indicate greater image dissimilarity.

    Inputs must already be polarity-aligned N x 1 x H x W floats in [0, 1].
    pretrained=False is a random, offline test model, NOT a trained metric.
    """

    def __init__(self, backbone="frozen", pretrained=True):
        super().__init__()
        import lpips
        self.metric = lpips.LPIPS(net="vgg", version="0.1", pretrained=pretrained,
                                  pnet_rand=not pretrained, verbose=False)
        self.set_backbone(backbone)

    def set_backbone(self, backbone):
        if backbone not in ("frozen", "tail4"):
            raise ValueError("backbone must be frozen or tail4")
        self.backbone = backbone
        self.metric.requires_grad_(False)
        self.metric.lins.requires_grad_(True)
        convs = [m for m in self.metric.net.modules() if isinstance(m, nn.Conv2d)]
        if len(convs) != 13:
            raise RuntimeError("Unexpected upstream VGG architecture")
        if backbone == "tail4":
            for layer in convs[-4:]:
                layer.requires_grad_(True)
        expected = 1472 if backbone == "frozen" else 9440704
        if sum(p.numel() for p in self.parameters() if p.requires_grad) != expected:
            raise RuntimeError("Unexpected trainable parameter count")

    def train(self, mode=True):
        super().train(mode)
        # Historical protocol: LPIPS head dropout in train, VGG in eval mode.
        self.metric.net.eval()
        return self

    def parameter_groups(self, head_lr=1e-4, tail_lr=1e-5):
        groups = [{"params": list(self.metric.lins.parameters()), "lr": head_lr}]
        tail = [p for p in self.metric.net.parameters() if p.requires_grad]
        if tail:
            groups.append({"params": tail, "lr": tail_lr})
        return groups

    def forward(self, fixed, moving):
        if fixed.shape != moving.shape or fixed.ndim != 4 or fixed.shape[1] != 1:
            raise ValueError("Expected equal N x 1 x H x W images")
        if min(fixed.shape[-2:]) < 32:
            raise ValueError("VGG inputs must be at least 32 pixels per side")
        for image in (fixed, moving):
            if not image.is_floating_point() or not torch.isfinite(image).all():
                raise ValueError("Images must be finite floating point")
            if image.min() < 0 or image.max() > 1:
                raise ValueError("Images must be in [0, 1]; preprocessing is explicit")
        x = fixed.expand(-1, 3, -1, -1) * 2 - 1
        y = moving.expand(-1, 3, -1, -1) * 2 - 1
        return self.metric(x, y).flatten()
