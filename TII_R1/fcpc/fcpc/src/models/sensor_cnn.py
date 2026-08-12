from __future__ import annotations


def _build_sensor_cnn(num_classes: int, input_channels: int):
    from torch import nn

    class SensorCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(input_channels, 32, kernel_size=5, padding=2),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(1),
            )
            self.classifier = nn.Linear(64, num_classes)

        def forward(self, x):
            return self.classifier(self.features(x).squeeze(-1))

    return SensorCNN()


class SensorCNN:
    """Factory-compatible wrapper created without importing torch at module load."""

    def __new__(cls, num_classes: int = 6, input_channels: int = 9):
        return _build_sensor_cnn(num_classes, input_channels)
