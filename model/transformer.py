"""
transformer.py

Complete Transformer Model
Author: Transformer From Scratch
"""

import torch
import torch.nn as nn

from config import Config
from model.embedding import InputEmbedding
from model.positional_encoding import PositionalEncoding
from model.encoder import Encoder
from model.decoder import Decoder


class Transformer(nn.Module):

    def __init__(self, src_vocab_size, tgt_vocab_size):
        super().__init__()

        self.src_embedding = InputEmbedding(src_vocab_size, Config.D_MODEL)
        self.tgt_embedding = InputEmbedding(tgt_vocab_size, Config.D_MODEL)
        
        self.positional_encoding = PositionalEncoding(
            d_model=Config.D_MODEL,
            max_seq_len=Config.MAX_SEQ_LEN,
            dropout=Config.DROPOUT
        )

        self.encoder = Encoder(
            num_layers=Config.N_ENCODER_LAYERS,
            d_model=Config.D_MODEL,
            num_heads=Config.N_HEADS,
            d_ff=Config.D_FF,
            dropout=Config.DROPOUT
        )

        self.decoder = Decoder(
            num_layers=Config.N_DECODER_LAYERS,
            d_model=Config.D_MODEL,
            num_heads=Config.N_HEADS,
            d_ff=Config.D_FF,
            dropout=Config.DROPOUT
        )

        self.output_layer = nn.Linear(Config.D_MODEL, tgt_vocab_size)

    def encode(self, src, src_mask=None):
        src = self.src_embedding(src)
        src = self.positional_encoding(src)
        return self.encoder(src, src_mask)

    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        tgt = self.tgt_embedding(tgt)
        tgt = self.positional_encoding(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.output_layer(x)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        logits = self.project(decoder_output)
        return logits
