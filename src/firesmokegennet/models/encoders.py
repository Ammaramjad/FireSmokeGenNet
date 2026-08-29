"""Frozen dual-branch ResNet encoders for mask geometry and image context."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50


BACKBONES = {
    "resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1, [64, 64, 128, 256, 512]),
    "resnet50": (resnet50, ResNet50_Weights.IMAGENET1K_V1, [64, 256, 512, 1024, 2048]),
}


class DualBranchEncoder(nn.Module):
    """ResNet-18 mask branch + ResNet-50/18 image branch, paper Sec. IV-B."""

    def __init__(
        self,
        mask_backbone: str = "resnet18",
        image_backbone: str = "resnet18",
        pretrained: bool = True,
        freeze: bool = True,
    ):
        super().__init__()
        self.mask_net, self.mask_channels = self._build(mask_backbone, pretrained)
        self.image_net, self.image_channels = self._build(image_backbone, pretrained)
        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def _build(self, name: str, pretrained: bool):
        ctor, weights, channels = BACKBONES[name]
        net = ctor(weights=weights if pretrained else None)
        return net, channels

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def _multiscale(self, net: nn.Module, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        feats: Dict[int, torch.Tensor] = {}
        x = net.conv1(x)
        x = net.bn1(x)
        x = net.relu(x)
        feats[1] = x
        x = net.maxpool(x)
        x = net.layer1(x)
        feats[2] = x
        x = net.layer2(x)
        feats[3] = x
        x = net.layer3(x)
        feats[4] = x
        x = net.layer4(x)
        feats[5] = x
        return feats

    def forward(self, mask: torch.Tensor, masked_image: torch.Tensor):
        with torch.no_grad():
            return self._multiscale(self.mask_net, mask), self._multiscale(
                self.image_net, masked_image
            )


class ScaleProjector(nn.Module):
    def __init__(self, encoder_channels: list[int], unet_channels: list[int]):
        super().__init__()
        # Map U-Net stages to encoder scales 5,4,3,3 (deepest to mid).
        self.keys = [5, 4, 3, 3]
        self.projs = nn.ModuleList(
            [
                nn.Conv2d(encoder_channels[min(k, 5) - 1], unet_channels[min(i, len(unet_channels) - 1)], 1)
                for i, k in enumerate(self.keys)
            ]
        )

    def project(self, feats: Dict[int, torch.Tensor], spatial: tuple[int, int], stage: int) -> torch.Tensor:
        stage = min(stage, len(self.projs) - 1)
        key = self.keys[stage]
        feat = F.interpolate(feats[key], size=spatial, mode="bilinear", align_corners=False)
        return self.projs[stage](feat)
