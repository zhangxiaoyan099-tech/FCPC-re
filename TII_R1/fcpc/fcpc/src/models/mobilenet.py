from __future__ import annotations


def mobilenet_v2_for_federated(num_classes: int, input_channels: int = 3):
    import types

    import torch.nn.functional as F
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
    def fedcfa_encode(self, x):
        return self.features(x)

    def fedcfa_decode(self, latent):
        pooled = F.adaptive_avg_pool2d(latent, (1, 1)).flatten(1)
        return self.classifier(pooled)

    def forward_with_representation(self, x):
        latent = self.features(x)
        representation = F.adaptive_avg_pool2d(latent, (1, 1)).flatten(1)
        return self.classifier(representation), representation

    model.fedcfa_encode = types.MethodType(fedcfa_encode, model)
    model.fedcfa_decode = types.MethodType(fedcfa_decode, model)
    model.forward_with_representation = types.MethodType(
        forward_with_representation,
        model,
    )
    return model
