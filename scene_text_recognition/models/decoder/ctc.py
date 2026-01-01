from typing import Literal, Optional

import torch
import torch.nn as nn
from einops import rearrange

from scene_text_recognition.frameworks.schemas import CTCOutput


class CTCDecoder(nn.Module):
    def __init__(
        self,
        blank_id: int,
        padding_id: int,
        reduction: Literal["mean", "sum", "none"],
        input_size: int,
        output_size: int,
        zero_infinity: bool = True,
    ) -> None:
        super().__init__()
        self.padding_id = padding_id
        self.ctc_head = nn.Linear(input_size, output_size)
        self.ctc_loss = nn.CTCLoss(
            blank=blank_id, reduction=reduction, zero_infinity=zero_infinity
        )

    def forward(
        self,
        x: torch.Tensor,
        xlens: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        target_lens: Optional[torch.Tensor] = None,
    ) -> CTCOutput:
        logits = self.ctc_head(x)
        # in case of amp, convert to float32
        log_probs = logits.float().log_softmax(dim=-1)
        predictions = log_probs.argmax(dim=-1)

        loss = None
        if targets is not None and target_lens is not None:
            loss = self._compute_loss(log_probs, xlens, targets, target_lens)

        return CTCOutput(
            logits=logits,
            log_probs=log_probs,
            predictions=predictions,
            loss=loss,
        )

    def _compute_loss(
        self,
        log_probs: torch.Tensor,
        xlens: torch.Tensor,
        targets: torch.Tensor,
        target_lens: torch.Tensor,
    ) -> torch.Tensor:
        log_probs_t = rearrange(log_probs, "bs seq_len hid -> seq_len bs hid")
        targets_flat = targets[targets != self.padding_id]
        assert targets_flat.size(0) == target_lens.sum(), "target length mismatch"
        return self.ctc_loss.forward(log_probs_t, targets_flat, xlens, target_lens)
