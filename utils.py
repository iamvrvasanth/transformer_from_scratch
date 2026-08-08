"""
utils.py

Utility functions for Transformer From Scratch

Author: Vasanth V R
"""

import os
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

from config import Config
from dataset import TranslationDataset


# ==========================================================
# Create Directory
# ==========================================================

def create_directory(path):
    """
    Creates directory if it does not exist.
    """
    Path(path).mkdir(
        parents=True,
        exist_ok=True
    )


# ==========================================================
# Save JSON
# ==========================================================

def save_json(
    data,
    filename
):
    """
    Save dictionary as JSON.
    """

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )


# ==========================================================
# Load JSON
# ==========================================================

def load_json(
    filename
):

    with open(
        filename,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ==========================================================
# Count Parameters
# ==========================================================

def count_parameters(model):
    """
    Count trainable parameters.
    """

    return sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad

    )


# ==========================================================
# Print Model Summary
# ==========================================================

def print_model_summary(model):

    print()

    print("=" * 60)

    print(model)

    print("=" * 60)

    print(
        f"Trainable Parameters : "
        f"{count_parameters(model):,}"
    )

    print("=" * 60)
# ==========================================================
# Xavier Initialization
# ==========================================================

def initialize_weights(model):
    """
    Initialize model weights using Xavier Uniform.
    """

    for parameter in model.parameters():

        if parameter.dim() > 1:

            nn.init.xavier_uniform_(parameter)
# ==========================================================
# Current Learning Rate
# ==========================================================

def get_learning_rate(
    optimizer
):

    return optimizer.param_groups[0]["lr"]
# ==========================================================
# Device
# ==========================================================

def get_device():

    return torch.device(

        "cuda"

        if torch.cuda.is_available()

        else

        "cpu"

    )
# ==========================================================
# Set Random Seed
# ==========================================================

def set_seed(
    seed=42
):

    import random
    import numpy as np

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)
# ==========================================================
# Yield Sentences
# ==========================================================

def get_all_sentences(dataset, language):
    """
    Generator function to yield sentences
    for tokenizer training.

    Parameters
    ----------
    dataset : HuggingFace Dataset
    language : str

    Yields
    ------
    sentence : str
    """

    for item in dataset:
        yield item[language]


# ==========================================================
# Build Tokenizer
# ==========================================================

def build_tokenizer(
    dataset,
    language,
    tokenizer_path,
    vocab_size=32000
):
    """
    Train tokenizer if it doesn't exist.
    Otherwise load existing tokenizer.
    """

    tokenizer_path = Path(tokenizer_path)

    if tokenizer_path.exists():

        print(f"Loading Tokenizer : {tokenizer_path}")

        tokenizer = Tokenizer.from_file(
            str(tokenizer_path)
        )

        return tokenizer

    print(f"Building Tokenizer : {language}")

    tokenizer = Tokenizer(
        WordLevel(
            unk_token="[UNK]"
        )
    )

    tokenizer.pre_tokenizer = Whitespace()

    trainer = WordLevelTrainer(

        vocab_size=vocab_size,

        special_tokens=[

            "[UNK]",

            "[PAD]",

            "[SOS]",

            "[EOS]"

        ]

    )

    tokenizer.train_from_iterator(

        get_all_sentences(
            dataset,
            language
        ),

        trainer=trainer

    )

    tokenizer.save(
        str(tokenizer_path)
    )

    print("Tokenizer Saved")

    return tokenizer


# ==========================================================
# Load Tokenizer
# ==========================================================

def load_tokenizer(tokenizer_path):
    """
    Load existing tokenizer.
    """

    tokenizer = Tokenizer.from_file(
        tokenizer_path
    )

    return tokenizer


# ==========================================================
# Create DataLoader
# ==========================================================

def create_dataloader(
    dataset,
    shuffle=True
):
    """
    Creates PyTorch DataLoader.
    """

    loader = DataLoader(

        dataset,

        batch_size=Config.BATCH_SIZE,

        shuffle=shuffle,

        num_workers=Config.NUM_WORKERS,

        pin_memory=torch.cuda.is_available(),

        drop_last=False

    )

    return loader
# ==========================================================
# Load Dataset
# ==========================================================

def load_translation_dataset():
    """
    Load translation dataset.

    Example:
        OPUS Books
    """

    print("Loading Dataset...")

    dataset = load_dataset(

        Config.DATASET_NAME,

        split="train"

    )

    return dataset
# ==========================================================
# Train Validation Split
# ==========================================================

def split_dataset(dataset):
    """
    Split dataset into train
    and validation sets.
    """

    split = dataset.train_test_split(

        test_size=Config.VAL_SPLIT,

        seed=Config.SEED

    )

    train_dataset = split["train"]

    validation_dataset = split["test"]

    return train_dataset, validation_dataset
# ==========================================================
# Prepare Data
# ==========================================================

def prepare_dataloaders():

    dataset = load_translation_dataset()

    train_data, validation_data = split_dataset(
        dataset
    )

    src_tokenizer = build_tokenizer(

        train_data,

        Config.SRC_LANGUAGE,

        Config.SRC_TOKENIZER_PATH

    )

    tgt_tokenizer = build_tokenizer(

        train_data,

        Config.TGT_LANGUAGE,

        Config.TGT_TOKENIZER_PATH

    )

    train_dataset = create_translation_dataset(

        train_data,

        src_tokenizer,

        tgt_tokenizer,

        Config.SRC_LANGUAGE,

        Config.TGT_LANGUAGE

    )

    validation_dataset = create_translation_dataset(

        validation_data,

        src_tokenizer,

        tgt_tokenizer,

        Config.SRC_LANGUAGE,

        Config.TGT_LANGUAGE

    )

    train_loader = create_dataloader(

        train_dataset,

        shuffle=True

    )

    validation_loader = create_dataloader(

        validation_dataset,

        shuffle=False

    )

    return (

        train_loader,

        validation_loader,

        src_tokenizer,

        tgt_tokenizer

    )
# ==========================================================
# Greedy Decoding
# ==========================================================

def greedy_decode(
    model,
    src,
    src_mask,
    tgt_tokenizer,
    max_len,
    device
):
    """
    Greedy decoding for Transformer inference.
    """

    model.eval()

    sos_idx = Config.SOS_IDX
    eos_idx = Config.EOS_IDX

    with torch.no_grad():

        encoder_output = model.encode(
            src,
            src_mask
        )

        decoder_input = torch.tensor(
            [[sos_idx]],
            dtype=torch.long,
            device=device
        )

        for _ in range(max_len - 1):

            from model.masks import create_target_mask

            decoder_mask = create_target_mask(
                decoder_input,
                Config.PAD_IDX
            ).to(device)

            decoder_output = model.decode(
                decoder_input,
                encoder_output,
                src_mask,
                decoder_mask
            )

            logits = model.project(
                decoder_output
            )

            next_token = torch.argmax(
                logits[:, -1],
                dim=-1
            )

            decoder_input = torch.cat(

                [

                    decoder_input,

                    next_token.unsqueeze(1)

                ],

                dim=1

            )

            if next_token.item() == eos_idx:
                break

    return decoder_input.squeeze(0)
# ==========================================================
# Decode Tokens
# ==========================================================

def tokens_to_sentence(
    token_ids,
    tokenizer
):
    """
    Convert token IDs to readable sentence.
    """

    return tokenizer.decode(
        token_ids.tolist()
    )
# ==========================================================
# Translate Sentence
# ==========================================================

def translate_sentence(
    sentence,
    model,
    src_tokenizer,
    tgt_tokenizer,
    device
):

    model.eval()

    src_ids = src_tokenizer.encode(
        sentence
    ).ids

    src_ids = (

        [Config.SOS_IDX]

        + src_ids

        + [Config.EOS_IDX]

    )

    src_ids = src_ids[:Config.MAX_SEQ_LEN]

    src_ids += [

        Config.PAD_IDX

    ] * (

        Config.MAX_SEQ_LEN - len(src_ids)

    )

    src_tensor = torch.tensor(

        src_ids,

        dtype=torch.long

    ).unsqueeze(0).to(device)

    from model.masks import create_padding_mask

    src_mask = create_padding_mask(
        src_tensor,
        Config.PAD_IDX
    ).to(device)

    prediction = greedy_decode(

        model,

        src_tensor,

        src_mask,

        tgt_tokenizer,

        Config.MAX_SEQ_LEN,

        device

    )

    return tgt_tokenizer.decode(
        prediction.tolist()
    )
# ==========================================================
# BLEU Score
# ==========================================================

from nltk.translate.bleu_score import corpus_bleu


def calculate_bleu(
    references,
    predictions
):
    """
    Compute corpus BLEU score.
    """

    return corpus_bleu(
        references,
        predictions
    )
# ==========================================================
# Load Model
# ==========================================================

def load_model(
    model,
    checkpoint_path,
    device
):

    checkpoint = torch.load(

        checkpoint_path,

        map_location=device

    )

    model.load_state_dict(

        checkpoint["model_state_dict"]

    )

    model.eval()

    print(

        f"Loaded Model : {checkpoint_path}"

    )

    return model
# ==========================================================
# Epoch Statistics
# ==========================================================

def print_epoch_statistics(

        epoch,

        train_loss,

        val_loss,

        accuracy,

        learning_rate

):

    print()

    print("=" * 70)

    print(f"Epoch            : {epoch}")

    print(f"Train Loss       : {train_loss:.4f}")

    print(f"Validation Loss  : {val_loss:.4f}")

    print(f"Accuracy         : {accuracy*100:.2f}%")

    print(f"Learning Rate    : {learning_rate:.8f}")

    print("=" * 70)
# ==========================================================
# Training Timer
# ==========================================================

import time


class Timer:

    def __init__(self):

        self.start = None

    def start_timer(self):

        self.start = time.time()

    def stop_timer(self):

        elapsed = time.time() - self.start

        minutes = elapsed / 60

        print(

            f"\nTraining Time : "

            f"{minutes:.2f} Minutes"

        )    




                