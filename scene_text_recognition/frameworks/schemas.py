from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class CTCOutput:
    logits: torch.Tensor  # (batch, seq_len, vocab_size)
    log_probs: torch.Tensor  # (batch, seq_len, vocab_size)
    predictions: torch.Tensor  # (batch, seq_len) greedy decode結果
    loss: Optional[torch.Tensor] = None  # training時のみ
