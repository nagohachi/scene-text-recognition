from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

from scene_text_recognition.configs.schemas import CTCModelConfig
from scene_text_recognition.frameworks.schemas import CTCOutput
from scene_text_recognition.models.decoder.ctc import CTCDecoder
from scene_text_recognition.models.encoder.transformer import TransformerEncoder
from scene_text_recognition.models.feature_extractor.cnn import ResnetV2FeatureExtractor


class CTCModel(nn.Module):
    def __init__(self, config: CTCModelConfig) -> None:
        super().__init__()
        self.config = config

        match config.feature_extractor.type:
            case "resnet18":
                self.feature_extractor = ResnetV2FeatureExtractor(
                    in_channels=config.feature_extractor.in_channels,
                    type="resnet18",
                )
            case _:
                raise ValueError(
                    f"Unknown feature extractor: {config.feature_extractor.type}"
                )

        match config.encoder.type:
            case "transformer":
                self.encoder = TransformerEncoder(
                    hidden_size=config.encoder.hidden_size,
                    num_heads=config.encoder.num_heads,
                    num_layers=config.encoder.num_layers,
                )
            case _:
                raise ValueError(f"Unknown encoder: {config.encoder.type}")

        self.ctc_decoder = CTCDecoder(
            blank_id=config.decoder.blank_id,
            padding_id=config.decoder.padding_id,
            reduction=config.decoder.reduction,
            input_size=config.encoder.hidden_size,
            output_size=config.vocab_size,
            zero_infinity=config.decoder.zero_infinity,
        )

    def forward(
        self,
        x: torch.Tensor,
        xlens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        target_lens: Optional[torch.Tensor] = None,
    ) -> CTCOutput:
        features, xlens = self.feature_extractor(x, xlens)

        # (batch, channels, height, width) -> (batch, width, channels * height)
        features = rearrange(features, "b c h w -> b w (c h)")

        encoded, xlens = self.encoder(features, xlens)
        return self.ctc_decoder(encoded, xlens, targets, target_lens)
