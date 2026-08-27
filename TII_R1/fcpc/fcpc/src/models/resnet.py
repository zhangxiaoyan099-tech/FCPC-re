from __future__ import annotations


def resnet18_for_federated(num_classes: int, input_channels: int = 3):
    from torch import flatten
    from torch import nn
    from torchvision.models.resnet import BasicBlock, ResNet

    class FederatedResNet18(ResNet):
        def __init__(self):
            super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
            # Keep the state-dict layout of torchvision ResNet18 while using a
            # CIFAR-sized stem. This also exposes a split after layer1 for the
            # author's FedCFA counterfactual feature construction.
            self.conv1 = nn.Conv2d(
                input_channels,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            self.maxpool = nn.Identity()

        def fedcfa_encode(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            return self.layer1(x)

        def fedcfa_decode(self, latent):
            x = self.layer2(latent)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            return self.fc(flatten(x, 1))

        def forward_with_representation(self, x):
            x = self.fedcfa_encode(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            representation = flatten(self.avgpool(x), 1)
            return self.fc(representation), representation

        def forward(self, x):
            return self.fedcfa_decode(self.fedcfa_encode(x))

    return FederatedResNet18()
