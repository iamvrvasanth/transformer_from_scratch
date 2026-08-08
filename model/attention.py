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

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        """
        (B, L, D)
            ↓
        (B, H, L, Dh)
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
        (B, H, L, Dh)
            ↓
        (B, L, D)
        """

        if x.dim() != 4:
            raise RuntimeError(
                f"combine_heads expected 4 dimensions, got {x.shape}"
            )

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

        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )

        scores = scores / math.sqrt(self.head_dim)

        print("Scores Shape :", scores.shape)

        if mask is not None:

            if mask.dtype != torch.bool:
                mask = mask.bool()

            while mask.dim() < scores.dim():
                mask = mask.unsqueeze(1)

            print("Mask After Broadcast :", mask.shape)

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

        # Split into heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        print("\n================ MultiHeadAttention ================")
        print("Q Shape      :", Q.shape)
        print("K Shape      :", K.shape)
        print("V Shape      :", V.shape)

        if mask is not None:
            print("Mask Shape   :", mask.shape)
            print("Mask Dtype   :", mask.dtype)
        else:
            print("Mask         : None")

        print("====================================================")

        output, attention = self.scaled_dot_product_attention(
            Q,
            K,
            V,
            mask
        )

        print("Attention Shape :", attention.shape)
        print("Output Shape    :", output.shape)

        output = self.combine_heads(output)

        output = self.W_o(output)

        return output, attention