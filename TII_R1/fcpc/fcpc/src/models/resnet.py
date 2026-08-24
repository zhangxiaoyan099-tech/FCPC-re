from __future__ import annotations


def resnet18_for_federated(num_classes: int, input_channels: int = 3):
    from torch import nn
    from torchvision.models import resnet18

    model = resnet18(weights=None, num_classes=num_classes)
    # The ImageNet stem (7x7 stride 2 followed by max-pooling) discards too
    # much spatial detail on 28x28/32x32 federated vision benchmarks.
    model.conv1 = nn.Conv2d(
        input_channels,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()
    return model
