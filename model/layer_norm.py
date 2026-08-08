"""
layer_norm.py

Layer Normalization implemented from scratch.
"""

import torch
import torch.nn as nn


class LayerNormalization(nn.Module):
    """
    Layer Normalization

    Input:
        (batch_size, seq_len, d_model)

    Output:
        (batch_size, seq_len, d_model)
    """

    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()

        self.eps = eps

        # Learnable parameters
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))

    def forward(self, x):

        mean = x.mean(dim=-1, keepdim=True)

        std = x.std(dim=-1, keepdim=True)

        return self.gamma * ((x - mean) / (std + self.eps)) + self.beta