from __future__ import annotations


def resnet18_for_federated(num_classes: int, input_channels: int = 3):
    import torch
    from torch import nn
    from torchvision.models import resnet18

    model = resnet18(weights=None, num_classes=num_classes)
    if input_channels != 3:
        model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model

