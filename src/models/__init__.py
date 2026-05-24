from .mlp import MLP
from .cnn1d import CNN1D

__all__ = ["MLP", "CNN1D"]


def build_model(model_name: str, input_length: int, num_classes: int, **kwargs):
    """Factory: return an instantiated model by name."""
    name = model_name.lower()
    if name == "mlp":
        return MLP(input_length=input_length, num_classes=num_classes, **kwargs)
    if name in ("cnn1d", "cnn"):
        return CNN1D(input_length=input_length, num_classes=num_classes, **kwargs)
    raise ValueError(f"Unknown model '{model_name}'. Choose 'mlp' or 'cnn1d'.")