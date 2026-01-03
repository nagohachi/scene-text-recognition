from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from scene_text_recognition.configs.schemas import AttnBasedEncDecModelConfig
from scene_text_recognition.frameworks.schemas import AEDOutput
from scene_text_recognition.models.decoder.transformer import TransformerDecoder
from scene_text_recognition.models.encoder.transformer import TransformerEncoder
from scene_text_recognition.models.feature_extractor.cnn import ResnetV2FeatureExtractor
from scene_text_recognition.tokenizer import AEDTokenizer


class AttnBasedEncDecModel(nn.Module):
    def __init__(
        self, config: AttnBasedEncDecModelConfig, tokenizer: AEDTokenizer
    ) -> None:
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

        match config.decoder.type:
            case "transformer":
                self.decoder = TransformerDecoder(
                    hidden_size=config.decoder.hidden_size,
                    num_heads=config.decoder.num_heads,
                    num_layers=config.decoder.num_layers,
                )
            case _:
                raise ValueError(f"Unknown encoder: {config.decoder.type}")

        self.lm_head = nn.Linear(config.decoder.hidden_size, self.tokenizer.vocab_size)

        self.embedding = nn.Embedding(
            num_embeddings=self.tokenizer.vocab_size,
            embedding_dim=config.decoder.hidden_size,
            padding_idx=tokenizer.pad_id,
        )

        self.loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_id)

    def _compute_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        logits_ = rearrange(logits, "bs seq_len hidden_size -> bs hidden_size seq_len")
        return self.loss.forward(logits_, targets)

    def _add_sos_eos(
        self, targets: torch.Tensor, target_lens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """add <SOS> and <EOS> to targets.

        Args:
            targets (torch.Tensor): tensor of **padded** token_ids of size (batch_size, seq_len)
            target_lens (torch.Tensor): lengths of each target sequence (batch_size,)

        Returns:
            tuple[torch.Tensor, torch.Tensor]: sos-added targets, eos-added targets
        """
        batch_size = targets.size(0)
        device = targets.device

        sos_column = torch.full(
            size=(batch_size, 1),
            fill_value=self.tokenizer.sos_id,
            dtype=targets.dtype,
            device=device,
        )
        sos_added = torch.cat([sos_column, targets], dim=1)

        eos_added = F.pad(targets, pad=(0, 1), value=self.tokenizer.pad_id)
        eos_added.scatter_(
            dim=1,
            index=target_lens.unsqueeze(1),
            src=torch.full(
                size=(batch_size, 1),
                fill_value=self.tokenizer.eos_id,
                dtype=targets.dtype,
                device=device,
            ),
        )

        return sos_added, eos_added

    def forward(
        self,
        x: torch.Tensor,
        xlens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        target_lens: Optional[torch.Tensor] = None,
    ) -> AEDOutput:
        features, xlens = self.feature_extractor(x, xlens)

        # (batch, channels, height, width) -> (batch, width, channels * height)
        features = rearrange(features, "b c h w -> b w (c h)")

        features = self.pre_encoder(features)

        encoder_out, encoder_out_lens = self.encoder(features, xlens)

        batch_size = x.size(0)

        assert (targets is None) == (target_lens is None)
        inference_mode = targets is None and target_lens is None

        if inference_mode:
            targets_sos_added = torch.full(
                size=(batch_size, 1),
                fill_value=self.tokenizer.sos_id,
                dtype=torch.long,
                device=x.device,
            )
            target_lens = torch.ones(batch_size, dtype=torch.long, device=x.device)
        else:
            assert targets is not None and target_lens is not None
            targets_sos_added, targets_eos_added = self._add_sos_eos(
                targets, target_lens
            )
            target_lens = target_lens + 1

        decoder_in = self.embedding.forward(targets_sos_added)
        decoded, _ = self.decoder.forward(
            decoder_in, target_lens, encoder_out, encoder_out_lens
        )

        logits = self.lm_head.forward(decoded)
        log_probs = logits.float().log_softmax(dim=-1)
        predictions = log_probs.argmax(dim=-1)

        if inference_mode:
            loss = None
        else:
            loss = self._compute_loss(logits, targets_eos_added)

        return AEDOutput(
            logits=logits, log_probs=log_probs, predictions=predictions, loss=loss
        )
