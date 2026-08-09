"""
encoder.py

Transformer Encoder implementation.
"""

import torch.nn as nn

from model.attention import MultiHeadAttention
from model.feed_forward import FeedForward
from model.layer_norm import LayerNormalization
from model.residual_connection import ResidualConnection


class EncoderLayer(nn.Module):

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

        self.feed_forward = FeedForward(
            d_model,
            d_ff,
            dropout
        )

        self.residual_connections = nn.ModuleList(
            [
                ResidualConnection(d_model, dropout),
                ResidualConnection(d_model, dropout)
            ]
        )

    def forward(
        self,
        x,
        src_mask=None
    ):
        # --- DEBUG BLOCK ---
        print("Encoder received mask:", src_mask.shape if src_mask is not None else None)
        import sys
        sys.exit()
        # -------------------

        # Residual Block 1
        x = self.residual_connections[0](
            x,
            lambda x: self.self_attention(
                x,
                x,
                x,
                src_mask
            )[0]
        )

        # Residual Block 2
        x = self.residual_connections[1](
            x,
            self.feed_forward
        )

        return x


class Encoder(nn.Module):

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
                EncoderLayer(
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
        src_mask=None
    ):

        for layer in self.layers:
            x = layer(
                x,
                src_mask
            )

        return self.norm(x)
