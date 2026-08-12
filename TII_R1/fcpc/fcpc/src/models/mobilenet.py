from __future__ import annotations


def mobilenet_v2_for_federated(num_classes: int, input_channels: int = 3):
    from torch import nn
    from torchvision.models import mobilenet_v2

    model = mobilenet_v2(weights=None, num_classes=num_classes)
    if input_channels != 3:
        first = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            input_channels,
            first.out_channels,
            kernel_size=first.kernel_size,
            stride=first.stride,
            padding=first.padding,
            bias=False,
        )
    return model

