"""
attention.py

Multi-Head Attention implementation from scratch.
"""

import math
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

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

    def split_heads(self, x):
        """
        Input:
            (batch_size, seq_len, d_model)

        Output:
            (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size, seq_len, _ = x.shape

        x = x.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        )

        return x.transpose(1, 2)

    def combine_heads(self, x):
        """
        Input:
            (batch_size, num_heads, seq_len, head_dim)

        Output:
            (batch_size, seq_len, d_model)
        """
        batch_size, num_heads, seq_len, head_dim = x.shape

        x = x.transpose(1, 2).contiguous()

        x = x.view(
            batch_size,
            seq_len,
            num_heads * head_dim
        )

        return x

    def scaled_dot_product_attention(
        self,
        Q,
        K,
        V,
        mask=None
    ):
        """
        Q, K, V:
            (batch_size, num_heads, seq_len, head_dim)
        """
        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )

        scores = scores / math.sqrt(self.head_dim)

        if mask is not None:
            mask = mask.bool()

            # -----------------------------------
            # Correct mask dimensions
            # -----------------------------------
            if mask.dim() == 2:
                mask = mask[:, None, None, :]
            elif mask.dim() == 3:
                mask = mask[:, None, :, :]
            elif mask.dim() != 4:
                raise ValueError(
                    f"Invalid mask shape: {mask.shape}"
                )

            scores = scores.masked_fill(
                ~mask,
                float("-inf")
            )

        attention = torch.softmax(
            scores,
            dim=-1
        )

        attention = self.dropout(attention)

        output = torch.matmul(
            attention,
            V
        )

        return output, attention

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

        # Split into multiple heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Multi-head attention
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
