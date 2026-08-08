"""
embedding.py

Transformer Input Embedding Layer

Converts token indices into dense vectors and scales them
by sqrt(d_model) as described in the original Transformer paper.

Paper:
Attention Is All You Need
Section 3.4
"""

import math
import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    """
    Input Embedding Layer

    Input:
        (batch_size, seq_len)

    Output:
        (batch_size, seq_len, d_model)
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()

        self.d_model = d_model

        # Embedding Lookup Table
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor
            Shape:
                (batch_size, seq_len)

        Returns
        -------
        Tensor
            Shape:
                (batch_size, seq_len, d_model)
        """

        # Lookup embedding vectors
        x = self.embedding(x)

        # Scale embeddings
        x = x * math.sqrt(self.d_model)

        return x