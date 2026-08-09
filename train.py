"""
train.py

Training Script for Transformer From Scratch
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import TranslationDataset
from model.transformer import Transformer
from datasets import load_dataset
from tokenizers import Tokenizer


# ==========================================================
# Random Seed
# ==========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ==========================================================
# Device
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device : {device}")


# ==========================================================
# Create Checkpoint Folder
# ==========================================================
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)


# ==========================================================
# Save Checkpoint
# ==========================================================
def save_checkpoint(epoch, model, optimizer, loss):
    save_path = os.path.join(Config.CHECKPOINT_DIR, Config.MODEL_NAME)
    model_state = (
        model.module.state_dict()
        if isinstance(model, torch.nn.DataParallel)
        else model.state_dict()
    )
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss
        },
        save_path
    )
    print(f"Checkpoint saved to {save_path}")


# ==========================================================
# Load Checkpoint
# ==========================================================
def load_checkpoint(model, optimizer):
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, Config.MODEL_NAME)
    if not os.path.exists(checkpoint_path):
        print("No checkpoint found.")
        return 0

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Checkpoint Loaded (Epoch {checkpoint['epoch']})")
    return checkpoint["epoch"] + 1


# ==========================================================
# Train One Epoch
# ==========================================================
def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in progress_bar:
        
        encoder_input = batch["encoder_input"].to(device)
        decoder_input = batch["decoder_input"].to(device)
        encoder_mask = batch["encoder_mask"].to(device)
        decoder_mask = batch["decoder_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=(device.type == "cuda")
        ):
            outputs = model(
                src=encoder_input,
                tgt=decoder_input,
                src_mask=encoder_mask,
                tgt_mask=decoder_mask
            )
            loss = criterion(
                outputs.view(-1, outputs.size(-1)),
                labels.view(-1)
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    return avg_loss


# ==========================================================
# Validate One Epoch
# ==========================================================
def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct_tokens = 0
    total_tokens = 0
    progress_bar = tqdm(dataloader, desc="Validation", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)
            labels = batch["label"].to(device)

            with torch.amp.autocast(
                device_type=device.type,
                enabled=(device.type == "cuda")
            ):
                outputs = model(
                    src=encoder_input,
                    tgt=decoder_input,
                    src_mask=encoder_mask,
                    tgt_mask=decoder_mask
                )
                loss = criterion(
                    outputs.view(-1, outputs.size(-1)),
                    labels.view(-1)
                )

            total_loss += loss.item()
            predictions = outputs.argmax(dim=-1)
            valid_mask = labels != Config.PAD_IDX

            correct_tokens += ((predictions == labels) & valid_mask).sum().item()
            total_tokens += valid_mask.sum().item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
    return avg_loss, accuracy


# ==========================================================
# Save Best Model
# ==========================================================
def save_best_model(model, optimizer, epoch, val_loss, best_loss):
    if val_loss < best_loss:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": (
                model.module.state_dict()
                if isinstance(model, torch.nn.DataParallel)
                else model.state_dict()
            ),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_loss": val_loss
        }
        save_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pt")
        torch.save(checkpoint, save_path)
        print(f"\nBest Model Saved (Validation Loss = {val_loss:.4f})")
        return val_loss
    return best_loss


# ==========================================================
# Main Training Function
# ==========================================================
def main():
    set_seed(Config.SEED)

    print("Loading OPUS Books dataset...")
    raw_dataset = load_dataset("opus_books", "en-fr", split="train")
    print("Total samples:", len(raw_dataset))

    split = raw_dataset.train_test_split(test_size=0.1, seed=42)
    train_raw = split["train"]
    val_raw = split["test"]

    print("Loading tokenizers...")
    src_tokenizer = Tokenizer.from_file(Config.SRC_TOKENIZER_PATH)
    tgt_tokenizer = Tokenizer.from_file(Config.TGT_TOKENIZER_PATH)

    src_vocab_size = src_tokenizer.get_vocab_size()
    tgt_vocab_size = tgt_tokenizer.get_vocab_size()

    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size
    ).to(device)

    # ------------------------------------------------------
    # Multi GPU setup
    # ------------------------------------------------------
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: 1.0
    )

    criterion = nn.CrossEntropyLoss(
        ignore_index=Config.PAD_IDX,
        label_smoothing=Config.LABEL_SMOOTHING
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda")
    )

    train_dataset = TranslationDataset(
        dataset=train_raw,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        src_lang="en",
        tgt_lang="fr",
        max_seq_len=Config.MAX_SEQ_LEN,
        pad_idx=Config.PAD_IDX,
        bos_idx=Config.BOS_IDX,
        eos_idx=Config.EOS_IDX
    )

    val_dataset = TranslationDataset(
        dataset=val_raw,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        src_lang="en",
        tgt_lang="fr",
        max_seq_len=Config.MAX_SEQ_LEN,
        pad_idx=Config.PAD_IDX,
        bos_idx=Config.BOS_IDX,
        eos_idx=Config.EOS_IDX
    )

    # Highly optimized DataLoader for GPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    start_epoch = load_checkpoint(model, optimizer)
    best_val_loss = float("inf")

    for epoch in range(start_epoch, Config.EPOCHS):
        print("=" * 60)
        print(f"Epoch {epoch+1}/{Config.EPOCHS}")
        print("=" * 60)

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device
        )

        val_loss, val_accuracy = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device
        )

        print()
        print(f"Train Loss      : {train_loss:.4f}")
        print(f"Validation Loss : {val_loss:.4f}")
        print(f"Token Accuracy  : {val_accuracy*100:.2f}%")

        save_checkpoint(epoch, model, optimizer, train_loss)
        best_val_loss = save_best_model(model, optimizer, epoch, val_loss, best_val_loss)

    print("\nTraining Finished Successfully")

if __name__ == "__main__":
    main()
