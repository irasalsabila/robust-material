from .mlp import MLP
from .cnn1d import CNN1D
from .resnet1d import ResNet1D
from .cnn_attention import CNNAttention
from .heads import CrystalSystemHead, SpaceGroupHead
from .multitask import MultitaskModel

__all__ = [
    "MLP",
    "CNN1D",
    "ResNet1D",
    "CNNAttention",
    "CrystalSystemHead",
    "SpaceGroupHead",
    "MultitaskModel",
]


def build_model(model_name: str, input_length: int, num_classes: int, **kwargs):
    """Factory: return a single-task model by name.

    For multitask models use MultitaskModel directly.
    """
    name = model_name.lower()
    if name == "mlp":
        return MLP(input_length=input_length, num_classes=num_classes, **kwargs)
    if name in ("cnn1d", "cnn"):
        return CNN1D(input_length=input_length, num_classes=num_classes, **kwargs)
    if name in ("resnet1d", "resnet"):
        return ResNet1D(input_length=input_length, num_classes=num_classes, **kwargs)
    if name in ("cnn_attention", "cnnattn"):
        return CNNAttention(input_length=input_length, num_classes=num_classes, **kwargs)
    raise ValueError(
        f"Unknown model '{model_name}'. "
        "Choose 'mlp', 'cnn1d', 'resnet1d', or 'cnn_attention'."
    )


def build_multitask_model(
    backbone: str = "cnn1d",
    input_length: int = 4500,
    num_crystal_classes: int = 7,
    num_sg_classes: int = 10,
    **kwargs,
) -> MultitaskModel:
    """Factory: return a MultitaskModel with the specified backbone."""
    return MultitaskModel(
        backbone=backbone,
        input_length=input_length,
        num_crystal_classes=num_crystal_classes,
        num_sg_classes=num_sg_classes,
        **kwargs,
    )
