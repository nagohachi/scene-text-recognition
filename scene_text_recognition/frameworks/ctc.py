from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

from scene_text_recognition.frameworks.schemas import CTCOutput
from scene_text_recognition.models.decoder.ctc import CTCDecoder
from scene_text_recognition.models.encoder.schamas import EncoderBase
from scene_text_recognition.models.feature_extractor.schemas import FeatureExtractorBase


class CTCModel(nn.Module):
    def __init__(
        self,
        feature_extractor: FeatureExtractorBase,
        encoder: EncoderBase,
        ctc_decoder: CTCDecoder,
    ) -> None:
        super().__init__()
        self.feature_extractor = feature_extractor
        self.encoder = encoder
        self.ctc_decoder = ctc_decoder

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
