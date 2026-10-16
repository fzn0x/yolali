"""yolali: a YOLO you can actually read.

A clean-room PyTorch object detection library. Every component is a plain
``torch.nn.Module`` or a typed dataclass, fully annotated and replaceable.

The names below are the whole public surface most users need::

    from yolali import Detector, load_model

    model = load_model("configs/models/gelan-n.yaml", weights="runs/voc/best.pt")
    for result in Detector(model).predict(["dog.jpg"], conf=0.25):
        print(result.xyxy, result.conf, result.cls)

Architectural lineage and credits are listed in ``NOTICE`` and in the
README's Acknowledgements section.
"""

from yolali.infer import Detections, Detector
from yolali.loss import DetectionLoss, DualLoss, LossConfig
from yolali.nn.config import ModelConfig
from yolali.nn.model import YolaliModel, build_model, load_model
from yolali.train import TrainConfig, Trainer
from yolali.types import DualRawPrediction, RawPrediction, Targets

__version__ = "0.0.1"

__all__ = [
    "DetectionLoss",
    "Detections",
    "Detector",
    "DualLoss",
    "DualRawPrediction",
    "LossConfig",
    "ModelConfig",
    "RawPrediction",
    "Targets",
    "TrainConfig",
    "Trainer",
    "YolaliModel",
    "__version__",
    "build_model",
    "load_model",
]
