from typing import Literal

import torch
import torch.nn as nn
from einops import rearrange


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
        targets: torch.Tensor,
        target_lens: torch.Tensor,
    ) -> torch.Tensor:
        # in case of amp, convert to float32
        log_probs = self.ctc_head(x).float().log_softmax(dim=-1)
        log_probs = rearrange(log_probs, "bs seq_len hid -> seq_len bs hid")
        targets = targets[targets != self.padding_id]
        assert targets.size(0) == target_lens.sum(), "target length mismatch"
        return self.ctc_loss.forward(log_probs, targets, xlens, target_lens)
