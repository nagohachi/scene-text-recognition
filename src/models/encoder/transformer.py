import torch
import torch.nn as nn

from src.models.utils import lens_to_mask


class TransformerEncoder(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_layers: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self, x: torch.Tensor, xlens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        padding_mask = lens_to_mask(xlens)
        x = self.encoder.forward(x, src_key_padding_mask=padding_mask)
        return x, xlens
