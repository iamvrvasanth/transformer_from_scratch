"""
dataset.py

Custom Dataset for Transformer

Author: Transformer From Scratch
"""

import torch
from torch.utils.data import Dataset

from model.masks import (
    create_padding_mask,
    create_target_mask
)


class TranslationDataset(Dataset):

    def __init__(
        self,
        dataset,
        src_tokenizer,
        tgt_tokenizer,
        src_lang,
        tgt_lang,
        max_seq_len,
        pad_idx=0,
        bos_idx=1,
        eos_idx=2
    ):
        self.dataset = dataset

        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.max_seq_len = max_seq_len

        self.pad_idx = pad_idx
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        sample = self.dataset[idx]

        translation = sample["translation"]

        src_text = translation[self.src_lang]
        tgt_text = translation[self.tgt_lang]

        # -----------------------------------
        # Tokenize
        # -----------------------------------
        src_tokens = self.src_tokenizer.encode(
            src_text
        ).ids

        tgt_tokens = self.tgt_tokenizer.encode(
            tgt_text
        ).ids

        # -----------------------------------
        # Encoder Input
        # <BOS> sentence <EOS>
        # -----------------------------------
        encoder_input = (
            [self.bos_idx]
            + src_tokens
            + [self.eos_idx]
        )

        # -----------------------------------
        # Decoder Input
        # <BOS> sentence
        # -----------------------------------
        decoder_input = (
            [self.bos_idx]
            + tgt_tokens
        )

        # -----------------------------------
        # Label
        # sentence <EOS>
        # -----------------------------------
        label = (
            tgt_tokens
            + [self.eos_idx]
        )

        # -----------------------------------
        # Truncate
        # -----------------------------------
        encoder_input = encoder_input[:self.max_seq_len]
        decoder_input = decoder_input[:self.max_seq_len]
        label = label[:self.max_seq_len]

        # -----------------------------------
        # Pad
        # -----------------------------------
        encoder_input += [
            self.pad_idx
        ] * (self.max_seq_len - len(encoder_input))

        decoder_input += [
            self.pad_idx
        ] * (self.max_seq_len - len(decoder_input))

        label += [
            self.pad_idx
        ] * (self.max_seq_len - len(label))

        # -----------------------------------
        # Convert to Tensor
        # -----------------------------------
        encoder_input = torch.tensor(
            encoder_input,
            dtype=torch.long
        )

        decoder_input = torch.tensor(
            decoder_input,
            dtype=torch.long
        )

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        # -----------------------------------
        # Create Masks
        # -----------------------------------
        encoder_mask = create_padding_mask(
            encoder_input,
            self.pad_idx
        )
        
        decoder_mask = create_target_mask(
            decoder_input,
            self.pad_idx
        )

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask,
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text
        }
