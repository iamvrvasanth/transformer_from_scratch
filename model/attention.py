"""
attention.py

Multi-Head Attention implementation from scratch.

Paper:
Attention Is All You Need
Section 3.2
"""

import math
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention

    Input:
        Query : (batch_size, seq_len_q, d_model)
        Key   : (batch_size, seq_len_k, d_model)
        Value : (batch_size, seq_len_v, d_model)

    Output:
        (batch_size, seq_len_q, d_model)
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()

        assert d_model % num_heads == 0, \
            "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(
        self,
        Q,
        K,
        V,
        mask=None
    ):
        """
        Q : (batch, heads, seq_q, head_dim)
        K : (batch, heads, seq_k, head_dim)
        V : (batch, heads, seq_v, head_dim)
        """

        # QK^T
        scores = torch.matmul(Q, K.transpose(-2, -1))

        # Scale
        scores = scores / math.sqrt(self.head_dim)

        # Apply mask if available
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Softmax
        attention = torch.softmax(scores, dim=-1)

        attention = self.dropout(attention)

        # Attention × V
        output = torch.matmul(attention, V)

        return output, attention

    def split_heads(self, x):
        """
        (batch, seq_len, d_model)

        →

        (batch, heads, seq_len, head_dim)
        """

        batch_size, seq_len, _ = x.size()

        x = x.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        )

        return x.transpose(1, 2)

    def combine_heads(self, x):
        """
        (batch, heads, seq_len, head_dim)

        →

        (batch, seq_len, d_model)
        """

        batch_size, _, seq_len, _ = x.size()

        x = x.transpose(1, 2)

        return x.contiguous().view(
            batch_size,
            seq_len,
            self.d_model
        )

    def forward(
        self,
        query,
        key,
        value,
        mask=None
    ):

        # Linear projections
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        # Split into heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Attention
        output, attention = self.scaled_dot_product_attention(
            Q,
            K,
            V,
            mask
        )

        # Combine heads
        output = self.combine_heads(output)

        # Final projection
        output = self.W_o(output)

        return output, attention