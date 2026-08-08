"""
transformer.py

Complete Transformer Model
Author: Transformer From Scratch

Architecture:

Source Sentence
      │
Embedding
      │
Positional Encoding
      │
Encoder
      │
Encoder Output
      │
──────────────────────────────
      │
Target Sentence
      │
Embedding
      │
Positional Encoding
      │
Decoder
      │
Linear Layer
      │
Vocabulary Probability
"""

import torch
import torch.nn as nn

from config import Config

from model.embedding import InputEmbedding
from model.positional_encoding import PositionalEncoding
from model.encoder import Encoder
from model.decoder import Decoder


class Transformer(nn.Module):

    def __init__(self,
        src_vocab_size,
        tgt_vocab_size):

        super().__init__()

        # -----------------------------
        # Source Embedding
        # -----------------------------
        self.src_embedding = InputEmbedding(
            src_vocab_size,
            Config.D_MODEL
        )

        # -----------------------------
        # Target Embedding
        # -----------------------------
        self.tgt_embedding = InputEmbedding(
            tgt_vocab_size,
            Config.D_MODEL
        )

        # -----------------------------
        # Positional Encoding
        # -----------------------------
        self.positional_encoding = PositionalEncoding(
            d_model=Config.D_MODEL,
            max_seq_len=Config.MAX_SEQ_LEN,
            dropout=Config.DROPOUT
        )

        # -----------------------------
        # Encoder
        # -----------------------------
        self.encoder = Encoder(
            num_layers=Config.N_ENCODER_LAYERS,
            d_model=Config.D_MODEL,
            num_heads=Config.N_HEADS,
            d_ff=Config.D_FF,
            dropout=Config.DROPOUT
        )

        # -----------------------------
        # Decoder
        # -----------------------------
        self.decoder = Decoder(
            num_layers=Config.N_DECODER_LAYERS,
            d_model=Config.D_MODEL,
            num_heads=Config.N_HEADS,
            d_ff=Config.D_FF,
            dropout=Config.DROPOUT
        )

        # -----------------------------
        # Final Output Layer
        # -----------------------------
        self.output_layer = nn.Linear(
             Config.D_MODEL,
            tgt_vocab_size
        )

    # ----------------------------------------------------

    def encode(self, src, src_mask=None):

        src = self.src_embedding(src)

        src = self.positional_encoding(src)

        encoder_output = self.encoder(
            src,
            src_mask
        )

        return encoder_output

    # ----------------------------------------------------

    def decode(
        self,
        tgt,
        encoder_output,
        src_mask=None,
        tgt_mask=None
    ):

        tgt = self.tgt_embedding(tgt)

        tgt = self.positional_encoding(tgt)

        decoder_output = self.decoder(
            tgt,
            encoder_output,
            src_mask,
            tgt_mask
        )

        return decoder_output

    # ----------------------------------------------------

    def project(self, x):

        return self.output_layer(x)

    # ----------------------------------------------------

    def forward(
        self,
        src,
        tgt,
        src_mask=None,
        tgt_mask=None
    ):

        encoder_output = self.encode(
            src,
            src_mask
        )

        decoder_output = self.decode(
            tgt,
            encoder_output,
            src_mask,
            tgt_mask
        )

        logits = self.project(decoder_output)

        return logits