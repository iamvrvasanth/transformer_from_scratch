"""
residual_connection.py

Residual Connection Module

Implements:

Residual
+
Dropout
+
Layer Normalization

Paper:
Attention Is All You Need
"""

import torch.nn as nn

from model.layer_norm import LayerNormalization


class ResidualConnection(nn.Module):
    """
    Residual Connection

    x
      │
      ▼
    Sublayer
      │
      ▼
    Dropout
      │
      ▼
    Add
      │
      ▼
    LayerNorm
    """

    def __init__(
        self,
        d_model: int,
        dropout: float
    ):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        self.norm = LayerNormalization(d_model)

    def forward(
        self,
        x,
        sublayer
    ):
        """
        Parameters
        ----------
        x : Tensor

        sublayer : Callable
            Function that returns the output
            of Self-Attention or FeedForward.
        """

        return self.norm(
            x + self.dropout(
                sublayer(x)
            )
        )