"""
train.py

Training Script for Transformer From Scratch

Author: Vasanth V R
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import TranslationDataset
from model.transformer import Transformer


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

device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else

    "cpu"

)

print(f"Using Device : {device}")


# ==========================================================
# Create Checkpoint Folder
# ==========================================================

os.makedirs(
    Config.CHECKPOINT_DIR,
    exist_ok=True
)


# ==========================================================
# Initialize Model
# ==========================================================

model = Transformer().to(device)


# ==========================================================
# Optimizer
# ==========================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY
)


# ==========================================================
# Loss Function
# ==========================================================

criterion = nn.CrossEntropyLoss(

    ignore_index=Config.PAD_IDX,

    label_smoothing=Config.LABEL_SMOOTHING

)


# ==========================================================
# Mixed Precision
# ==========================================================

scaler = torch.cuda.amp.GradScaler(

    enabled=torch.cuda.is_available()

)


# ==========================================================
# Save Checkpoint
# ==========================================================

def save_checkpoint(

        epoch,

        model,

        optimizer,

        loss

):

    checkpoint = {

        "epoch": epoch,

        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": optimizer.state_dict(),

        "loss": loss

    }

    save_path = os.path.join(

        Config.CHECKPOINT_DIR,

        Config.MODEL_NAME

    )

    torch.save(

        checkpoint,

        save_path

    )

    print(f"\nCheckpoint Saved : {save_path}")


# ==========================================================
# Load Checkpoint
# ==========================================================

def load_checkpoint(

        model,

        optimizer

):

    checkpoint_path = os.path.join(

        Config.CHECKPOINT_DIR,

        Config.MODEL_NAME

    )

    if not os.path.exists(

        checkpoint_path

    ):

        print("No checkpoint found.")

        return 0

    checkpoint = torch.load(

        checkpoint_path,

        map_location=device

    )

    model.load_state_dict(

        checkpoint["model_state_dict"]

    )

    optimizer.load_state_dict(

        checkpoint["optimizer_state_dict"]

    )

    print(

        f"Checkpoint Loaded (Epoch {checkpoint['epoch']})"

    )

    return checkpoint["epoch"] + 1
# ==========================================================
# Train One Epoch
# ==========================================================

def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    scaler,
    device
):

    model.train()

    total_loss = 0.0

    progress_bar = tqdm(
        dataloader,
        desc="Training",
        leave=False
    )

    for batch in progress_bar:

        # ---------------------------------------------
        # Move Batch to Device
        # ---------------------------------------------
        encoder_input = batch["encoder_input"].to(device)
        decoder_input = batch["decoder_input"].to(device)

        encoder_mask = batch["encoder_mask"].to(device)
        decoder_mask = batch["decoder_mask"].to(device)

        labels = batch["label"].to(device)

        # ---------------------------------------------
        # Zero Gradients
        # ---------------------------------------------
        optimizer.zero_grad(set_to_none=True)

        # ---------------------------------------------
        # Mixed Precision Forward
        # ---------------------------------------------
        with torch.cuda.amp.autocast(
            enabled=torch.cuda.is_available()
        ):

            outputs = model(

                src=encoder_input,

                tgt=decoder_input,

                src_mask=encoder_mask,

                tgt_mask=decoder_mask

            )

            loss = criterion(

                outputs.view(
                    -1,
                    outputs.size(-1)
                ),

                labels.view(-1)

            )

        # ---------------------------------------------
        # Backpropagation
        # ---------------------------------------------
        scaler.scale(loss).backward()

        # ---------------------------------------------
        # Gradient Clipping
        # ---------------------------------------------
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(

            model.parameters(),

            max_norm=1.0

        )

        # ---------------------------------------------
        # Optimizer Step
        # ---------------------------------------------
        scaler.step(optimizer)

        scaler.update()

        # ---------------------------------------------
        # Statistics
        # ---------------------------------------------
        total_loss += loss.item()

        progress_bar.set_postfix(

            loss=f"{loss.item():.4f}"

        )

    avg_loss = total_loss / len(dataloader)

    return avg_loss
# ==========================================================
# Validate One Epoch
# ==========================================================

def validate_one_epoch(
    model,
    dataloader,
    criterion,
    device
):
    """
    Validate the model for one epoch.

    Returns
    -------
    avg_loss : float
    accuracy : float
    """

    model.eval()

    total_loss = 0.0

    correct_tokens = 0
    total_tokens = 0

    progress_bar = tqdm(
        dataloader,
        desc="Validation",
        leave=False
    )

    with torch.no_grad():

        for batch in progress_bar:

            # ---------------------------------------------
            # Move Batch to Device
            # ---------------------------------------------
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)

            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)

            labels = batch["label"].to(device)

            # ---------------------------------------------
            # Forward Pass
            # ---------------------------------------------
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

                    outputs.view(
                        -1,
                        outputs.size(-1)
                    ),

                    labels.view(-1)

                )

            total_loss += loss.item()

            # ---------------------------------------------
            # Prediction
            # ---------------------------------------------
            predictions = outputs.argmax(dim=-1)

            # Ignore PAD Tokens
            valid_mask = labels != Config.PAD_IDX

            correct_tokens += (
                (predictions == labels) &
                valid_mask
            ).sum().item()

            total_tokens += valid_mask.sum().item()

            progress_bar.set_postfix(

                loss=f"{loss.item():.4f}"

            )

    avg_loss = total_loss / len(dataloader)

    accuracy = (
        correct_tokens / total_tokens
        if total_tokens > 0
        else 0.0
    )

    return avg_loss, accuracy
# ==========================================================
# Save Best Model
# ==========================================================

def save_best_model(
    model,
    optimizer,
    epoch,
    val_loss,
    best_loss
):

    if val_loss < best_loss:

        checkpoint = {

            "epoch": epoch,

            "model_state_dict": model.state_dict(),

            "optimizer_state_dict": optimizer.state_dict(),

            "validation_loss": val_loss

        }

        save_path = os.path.join(

            Config.CHECKPOINT_DIR,

            "best_model.pt"

        )

        torch.save(
            checkpoint,
            save_path
        )

        print(
            f"\nBest Model Saved "
            f"(Validation Loss = {val_loss:.4f})"
        )

        return val_loss

    return best_loss
# ==========================================================
# Main Training Function
# ==========================================================

def main():

    set_seed(Config.SEED)

    # ------------------------------------------------------
    # TODO:
    # Replace these placeholders with actual dataset loading
    # ------------------------------------------------------

    train_dataset = ...
    val_dataset = ...

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda")
    )

    # ------------------------------------------------------
    # Resume Training
    # ------------------------------------------------------

    start_epoch = load_checkpoint(
        model,
        optimizer
    )

    best_val_loss = float("inf")

    # ------------------------------------------------------
    # Training Loop
    # ------------------------------------------------------

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

        save_checkpoint(

            epoch,

            model,

            optimizer,

            train_loss

        )

        best_val_loss = save_best_model(

            model,

            optimizer,

            epoch,

            val_loss,

            best_val_loss

        )

    print("\nTraining Finished Successfully")


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    main()
