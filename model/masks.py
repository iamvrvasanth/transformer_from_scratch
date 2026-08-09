"""
masks.py

Utilities for creating Transformer masks.
"""

import torch


def create_padding_mask(seq, pad_idx=0):
    mask = (seq != pad_idx)
    
    if mask.dim() == 1:
        # Called from dataset.py: [L] -> [1, 1, L]
        # DataLoader will stack this to [B, 1, 1, L]
        return mask.unsqueeze(0).unsqueeze(0)
        
    elif mask.dim() == 2:
        # If called directly on a batch in train.py: [B, L] -> [B, 1, 1, L]
        return mask.unsqueeze(1).unsqueeze(2)
        
    return mask.bool()


def create_look_ahead_mask(size):
    mask = torch.tril(
        torch.ones((size, size), dtype=torch.bool)
    )
    
    # Returns [1, L, L]. DataLoader will stack to [B, 1, L, L]
    return mask.unsqueeze(0)


def create_target_mask(tgt, pad_idx=0):
    # Padding mask is [1, 1, L]
    padding_mask = create_padding_mask(tgt, pad_idx)
    
    seq_len = tgt.size(-1)
    
    # Look ahead is [1, L, L]
    look_ahead = create_look_ahead_mask(seq_len).to(tgt.device)
    
    # Broadcasting: [1, 1, L] & [1, L, L] -> [1, L, L]
    # DataLoader will stack this to [B, 1, L, L]
    decoder_mask = padding_mask & look_ahead
    
    return decoder_mask
