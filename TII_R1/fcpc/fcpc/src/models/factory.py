from __future__ import annotations


def build_model(name: str, num_classes: int, input_channels: int = 3):
    name = name.lower()
    if name in {"simple_cnn", "cnn"}:
        from .simple_cnn import SimpleCNN

        return SimpleCNN(num_classes=num_classes, input_channels=input_channels)
    if name == "resnet18":
        from .resnet import resnet18_for_federated

        return resnet18_for_federated(num_classes=num_classes, input_channels=input_channels)
    if name == "mobilenetv2":
        from .mobilenet import mobilenet_v2_for_federated

        return mobilenet_v2_for_federated(num_classes=num_classes, input_channels=input_channels)
    if name in {"sensor_cnn", "har_cnn"}:
        from .sensor_cnn import SensorCNN

        return SensorCNN(num_classes=num_classes, input_channels=input_channels)
    raise ValueError(f"unsupported model: {name}")
