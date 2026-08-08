"""
masks.py

Utilities for creating Transformer masks.
"""

import torch


def create_padding_mask(seq, pad_idx=0):
    """
    Creates padding mask.

    Input:
        Dataset : [seq_len]
        Training: [batch_size, seq_len]

    Output:
        [batch_size, 1, 1, seq_len]
    """

    # Convert single sequence -> batch
    if seq.dim() == 1:
        seq = seq.unsqueeze(0)

    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)

    return mask.bool()


def create_look_ahead_mask(size):
    """
    Creates causal (look-ahead) mask.

    Output:
        [1, 1, size, size]
    """

    mask = torch.tril(
        torch.ones(
            (size, size),
            dtype=torch.bool
        )
    )

    return mask.unsqueeze(0).unsqueeze(0)


def create_target_mask(tgt, pad_idx=0):
    """
    Creates decoder mask.

    Combines:
        Padding Mask
            AND
        Look Ahead Mask

    Input:
        tgt -> [batch_size, seq_len]
            or
        tgt -> [seq_len]

    Output:
        [batch_size, 1, seq_len, seq_len]
    """

    # Handle single sequence
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

    decoder_mask = padding_mask & look_ahead

    return decoder_mask