"""
decoder.py

Transformer Decoder implementation.
"""

import torch.nn as nn

from model.attention import MultiHeadAttention
from model.feed_forward import FeedForward
from model.layer_norm import LayerNormalization
from model.residual_connection import ResidualConnection


class DecoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        dropout
    ):
        super().__init__()

        self.self_attention = MultiHeadAttention(
            d_model,
            num_heads,
            dropout
        )

        self.cross_attention = MultiHeadAttention(
            d_model,
            num_heads,
            dropout
        )

        self.feed_forward = FeedForward(
            d_model,
            d_ff,
            dropout
        )

        self.residual_connections = nn.ModuleList(
            [
                ResidualConnection(d_model, dropout),
                ResidualConnection(d_model, dropout),
                ResidualConnection(d_model, dropout)
            ]
        )

    def forward(
        self,
        x,
        encoder_output,
        src_mask=None,
        tgt_mask=None
    ):

        # Masked Self Attention
        x = self.residual_connections[0](
            x,
            lambda x: self.self_attention(
                x,
                x,
                x,
                tgt_mask
            )[0]
        )

        # Cross Attention
        x = self.residual_connections[1](
            x,
            lambda x: self.cross_attention(
                x,
                encoder_output,
                encoder_output,
                src_mask
            )[0]
        )

        # Feed Forward
        x = self.residual_connections[2](
            x,
            self.feed_forward
        )

        return x


class Decoder(nn.Module):

    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        d_ff,
        dropout
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model,
                    num_heads,
                    d_ff,
                    dropout
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = LayerNormalization(d_model)

    def forward(
        self,
        x,
        encoder_output,
        src_mask=None,
        tgt_mask=None
    ):

        for layer in self.layers:
            x = layer(
                x,
                encoder_output,
                src_mask,
                tgt_mask
            )

        return self.norm(x)