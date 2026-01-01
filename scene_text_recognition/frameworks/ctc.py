from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

from scene_text_recognition.configs.schemas import CTCModelConfig
from scene_text_recognition.frameworks.schemas import CTCOutput
from scene_text_recognition.models.decoder.ctc import CTCDecoder
from scene_text_recognition.models.encoder.transformer import TransformerEncoder
from scene_text_recognition.models.feature_extractor.cnn import ResnetV2FeatureExtractor
from scene_text_recognition.tokenizer import CTCTokenizer


class CTCModel(nn.Module):
    def __init__(self, config: CTCModelConfig, tokenizer: CTCTokenizer) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

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

        feature_out_height = self.feature_extractor.out_height(
            config.feature_extractor.input_height
        )
        pre_encoder_input_size = self.feature_extractor.out_size * feature_out_height

        match config.pre_encoder.type:
            case "linear":
                self.pre_encoder = nn.Linear(
                    pre_encoder_input_size, config.encoder.hidden_size
                )
            case None:
                self.pre_encoder = nn.Identity()

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
            blank_id=tokenizer.blank_id,
            padding_id=tokenizer.pad_id,
            reduction=config.decoder.reduction,
            input_size=config.encoder.hidden_size,
            output_size=tokenizer.vocab_size,
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

        features = self.pre_encoder(features)

        encoded, xlens = self.encoder(features, xlens)
        return self.ctc_decoder(encoded, xlens, targets, target_lens)
