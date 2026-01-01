from dataclasses import dataclass
from typing import Literal


@dataclass
class FeatureExtractorConfig:
    type: Literal["resnet18"]
    in_channels: int = 3


@dataclass
class EncoderConfig:
    type: Literal["transformer"]
    hidden_size: int
    num_heads: int
    num_layers: int


@dataclass
class CTCDecoderConfig:
    reduction: Literal["mean", "sum", "none"] = "mean"
    zero_infinity: bool = True


@dataclass
class CTCModelConfig:
    feature_extractor: FeatureExtractorConfig
    encoder: EncoderConfig
    decoder: CTCDecoderConfig
