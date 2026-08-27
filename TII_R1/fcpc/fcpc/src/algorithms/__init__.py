from .base import AlgorithmAdapter
from .fedcfa import FedCFAAdapter
from .feddyn import FedDynAdapter
from .fedavg import FedAvgAdapter
from .fblg import FBLGAdapter
from .fedprox import FedProxAdapter
from .moon import MOONAdapter


def build_algorithm(name: str, **kwargs) -> AlgorithmAdapter:
    name = name.lower()
    if name == "fedavg":
        return FedAvgAdapter(**kwargs)
    if name == "fedprox":
        return FedProxAdapter(**kwargs)
    if name == "moon":
        return MOONAdapter(**kwargs)
    if name == "feddyn":
        return FedDynAdapter(**kwargs)
    if name == "fblg":
        return FBLGAdapter(**kwargs)
    if name == "fedcfa":
        return FedCFAAdapter(**kwargs)
    raise ValueError(f"unsupported algorithm: {name}")
