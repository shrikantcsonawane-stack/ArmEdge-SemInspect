"""MobileNetV3-Small transfer-learning classifier.

Uses torchvision when available. If torchvision is missing/incompatible (e.g. a torch build
pinned by ExecuTorch with no matching torchvision wheel), an equivalent built-in definition
with identical state_dict keys is used and ImageNet weights are fetched from the same
official URL — so checkpoints are interchangeable.
"""
from __future__ import annotations

import torch
import torch.nn as nn

WEIGHTS_URL = "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth"

try:
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
    HAVE_TV = True
except Exception:  # pragma: no cover - environment dependent
    HAVE_TV = False


def _div(v, d=8):
    n = max(d, int(v + d / 2) // d * d)
    return n + d if n < 0.9 * v else n


class _CNA(nn.Sequential):  # Conv-BN-Activation, mirrors torchvision Conv2dNormActivation
    def __init__(self, i, o, k, s=1, g=1, act=None):
        layers = [nn.Conv2d(i, o, k, s, (k - 1) // 2, groups=g, bias=False), nn.BatchNorm2d(o, eps=0.001, momentum=0.01)]
        if act is not None:
            layers.append(act(inplace=True))
        super().__init__(*layers)


class _SE(nn.Module):
    def __init__(self, c, sq):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1, self.fc2 = nn.Conv2d(c, sq, 1), nn.Conv2d(sq, c, 1)
        self.activation, self.scale_activation = nn.ReLU(), nn.Hardsigmoid()

    def forward(self, x):
        return x * self.scale_activation(self.fc2(self.activation(self.fc1(self.avgpool(x)))))


class _IR(nn.Module):
    def __init__(self, i, k, e, o, se, hs, s):
        super().__init__()
        act = nn.Hardswish if hs else nn.ReLU
        self.use_res = s == 1 and i == o
        layers = [] if e == i else [_CNA(i, e, 1, act=act)]
        layers.append(_CNA(e, e, k, s, g=e, act=act))
        if se:
            layers.append(_SE(e, _div(e // 4)))
        layers.append(_CNA(e, o, 1))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        y = self.block(x)
        return x + y if self.use_res else y


_CFG = [(16, 3, 16, 16, 1, 0, 2), (16, 3, 72, 24, 0, 0, 2), (24, 3, 88, 24, 0, 0, 1), (24, 5, 96, 40, 1, 1, 2),
        (40, 5, 240, 40, 1, 1, 1), (40, 5, 240, 40, 1, 1, 1), (40, 5, 120, 48, 1, 1, 1), (48, 5, 144, 48, 1, 1, 1),
        (48, 5, 288, 96, 1, 1, 2), (96, 5, 576, 96, 1, 1, 1), (96, 5, 576, 96, 1, 1, 1)]


class MobileNetV3Small(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(_CNA(3, 16, 3, 2, act=nn.Hardswish), *[_IR(*c) for c in _CFG],
                                      _CNA(96, 576, 1, act=nn.Hardswish))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Linear(576, 1024), nn.Hardswish(inplace=True), nn.Dropout(0.2),
                                        nn.Linear(1024, num_classes))

    def forward(self, x):
        return self.classifier(torch.flatten(self.avgpool(self.features(x)), 1))


def build_model(num_classes: int, pretrained: bool = True, unfreeze_last_blocks: int = 2) -> nn.Module:
    if HAVE_TV:
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
    else:
        model = MobileNetV3Small()
        if pretrained:
            model.load_state_dict(torch.hub.load_state_dict_from_url(WEIGHTS_URL, progress=False))
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    # Freeze backbone, then unfreeze last N feature blocks + classifier head.
    for p in model.features.parameters():
        p.requires_grad = False
    if unfreeze_last_blocks > 0:
        for blk in model.features[-unfreeze_last_blocks:]:
            for p in blk.parameters():
                p.requires_grad = True
    return model


def save_checkpoint(path, model, classes, meta: dict):
    torch.save({"state_dict": model.state_dict(), "classes": list(classes), "meta": meta}, path)


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(len(ckpt["classes"]), pretrained=False, unfreeze_last_blocks=0)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["classes"], ckpt.get("meta", {})
