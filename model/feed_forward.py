"""
feed_forward.py

Position-wise Feed Forward Network (FFN)

Implements the feed-forward network described in:
"Attention Is All You Need"

Architecture:
d_model
   │
Linear
   │
   ▼
d_ff
   │
ReLU
   │
Dropout
   │
Linear
   │
   ▼
d_model
"""

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    Position-wise Feed Forward Network

    Input:
        (batch_size, seq_len, d_model)

    Output:
        (batch_size, seq_len, d_model)
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float
    ):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

        self.linear2 = nn.Linear(d_ff, d_model)

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
            Shape:
            (batch_size, seq_len, d_model)
        """

        # First Linear Layer
        x = self.linear1(x)

        # Activation
        x = self.relu(x)

        # Dropout
        x = self.dropout(x)

        # Second Linear Layer
        x = self.linear2(x)

        return x