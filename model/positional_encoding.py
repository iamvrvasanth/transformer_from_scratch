"""
positional_encoding.py

Sinusoidal Positional Encoding

Implements the positional encoding described in
"Attention Is All You Need" (Section 3.5)

Author: Transformer From Scratch
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Adds positional information to token embeddings.

    Input Shape:
        (batch_size, seq_len, d_model)

    Output Shape:
        (batch_size, seq_len, d_model)
    """

    def __init__(self, d_model: int, max_seq_len: int, dropout: float):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)

        # Position indices
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # Compute the division term
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension
        pe = pe.unsqueeze(0)

        # Register as a buffer (not trainable)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor
            Shape:
            (batch_size, seq_len, d_model)

        Returns
        -------
        Tensor
            Same shape as input
        """

        seq_len = x.size(1)

        # Add positional encoding
        x = x + self.pe[:, :seq_len]

        return self.dropout(x)