from dataclasses import dataclass
from typing import Literal


@dataclass
class FeatureExtractorConfig:
    type: Literal["resnet18"]
    in_channels: int = 3
    input_height: int = 32


@dataclass
class PreEncoderConfig:
    type: Literal["linear"] | None


@dataclass
class EncoderConfig:
    type: Literal["transformer"]
    hidden_size: int
    num_heads: int
    num_layers: int


@dataclass
class CTCDecoderConfig:
    type: Literal["ctc"]
    reduction: Literal["mean", "sum", "none"] = "mean"
    zero_infinity: bool = True


@dataclass
class DecoderConfig:
    type: Literal["transformer"]
    hidden_size: int
    num_heads: int
    num_layers: int


@dataclass
class CTCModelConfig:
    feature_extractor: FeatureExtractorConfig
    pre_encoder: PreEncoderConfig
    encoder: EncoderConfig
    decoder: CTCDecoderConfig


@dataclass
class AttnBasedEncDecModelConfig:
    feature_extractor: FeatureExtractorConfig
    pre_encoder: PreEncoderConfig
    encoder: EncoderConfig
    decoder: DecoderConfig
    max_len: int
