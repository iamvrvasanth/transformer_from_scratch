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

    (batch_size, seq_len)

    Output:

    (batch_size,1,1,seq_len)
    """

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


def create_target_mask(
    tgt,
    pad_idx=0
):
    """
    Combines

    Padding Mask

    +

    Look Ahead Mask
    """

    padding_mask = create_padding_mask(
        tgt,
        pad_idx
    )

    seq_len = tgt.size(1)

    look_ahead = create_look_ahead_mask(
        seq_len
    ).to(tgt.device)

    return padding_mask & look_ahead