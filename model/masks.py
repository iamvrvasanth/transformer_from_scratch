"""
masks.py

Utilities for creating Transformer masks.
"""

import torch


def create_padding_mask(
    seq,
    pad_idx=0
):
    """
    Creates padding mask.

    Input:
        Dataset : [seq_len]
        Training: [batch_size, seq_len]

    Output:
        [batch_size, 1, 1, seq_len]
    """

    # Handle a single sequence
    if seq.dim() == 1:
        seq = seq.unsqueeze(0)

    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)

    return mask


def create_look_ahead_mask(
    size
):
    """
    Creates causal mask.

    Shape

    (1,1,size,size)
    """

    mask = torch.tril(
        torch.ones(size, size)
    )

    return mask.unsqueeze(0).unsqueeze(0)


import torch

def create_target_mask(
    tgt,
    pad_idx=0
):
    """
    Creates decoder mask.

    Supports:
        Dataset : [seq_len]
        Training: [batch_size, seq_len]
    """

    # If a single sequence is passed, convert [L] -> [1, L]
    if tgt.dim() == 1:
        tgt = tgt.unsqueeze(0)

    padding_mask = create_padding_mask(
        tgt,
        pad_idx
    )

    seq_len = tgt.size(1)

    look_ahead = create_look_ahead_mask(
        seq_len
    ).to(tgt.device)

    # Make look-ahead mask broadcastable: [1,1,L,L]
    if look_ahead.dim() == 2:
        look_ahead = look_ahead.unsqueeze(0).unsqueeze(0)

    return padding_mask & look_ahead
