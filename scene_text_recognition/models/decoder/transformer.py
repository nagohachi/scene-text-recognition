import torch
import torch.nn as nn

from scene_text_recognition.models.utils import lens_to_mask


class TransformerDecoder(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_layers: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=num_heads, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer, num_layers=num_layers
        )

    def forward(
        self,
        x: torch.Tensor,
        xlens: torch.Tensor,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): the sequence to the decoder: (batch_size, seq_len, hidden_size).
            xlens (torch.Tensor): the lengths of x: (batch_size, ).
            encoder_out (torch.Tensor): the sequence of the output of the encoder: (batch_size, seq_len, encoder_hidden_size).
            encoder_out_lens (torch.Tensor): the lenghts of encoder_out: (batch_size, ).
        """
        x_padding_mask = lens_to_mask(xlens)
        encoder_out_padding_mask = lens_to_mask(encoder_out_lens)
        x_mask = nn.Transformer.generate_square_subsequent_mask(sz=x.size(1))

        decoder_out = self.decoder.forward(
            tgt=x,
            memory=encoder_out,
            tgt_mask=x_mask,
            tgt_key_padding_mask=x_padding_mask,
            memory_key_padding_mask=encoder_out_padding_mask,
            tgt_is_causal=True,
        )

        return decoder_out
