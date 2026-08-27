from __future__ import annotations

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - keeps imports usable without torch
    torch = None
    nn = None


if nn is not None:

    class SimpleCNN(nn.Module):
        def __init__(self, num_classes: int = 10, input_channels: int = 1):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 4 * 4, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, num_classes),
            )

        def forward(self, x):
            return self.fedcfa_decode(self.fedcfa_encode(x))

        def fedcfa_encode(self, x):
            return self.features(x)

        def fedcfa_decode(self, latent):
            return self.classifier(latent)

        def forward_with_representation(self, x):
            latent = self.fedcfa_encode(x)
            hidden = self.classifier[0](latent)
            hidden = self.classifier[1](hidden)
            hidden = self.classifier[2](hidden)
            return self.classifier[3](hidden), hidden

else:

    class SimpleCNN:  # type: ignore[no-redef]
        def __init__(self, *_args, **_kwargs):
            raise ImportError("PyTorch is required to instantiate SimpleCNN")
