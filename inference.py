"""
inference.py

Inference Script for Transformer From Scratch

Author : Vasanth V R
"""

import torch

from config import Config

from model.transformer import Transformer
from model.masks import (
    create_padding_mask,
    create_target_mask
)

from tokenizers import Tokenizer

from utils import load_model

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
# Load Tokenizers
# ==========================================================

print("\nLoading Tokenizers...")

src_tokenizer = Tokenizer.from_file(
    Config.SRC_TOKENIZER_PATH
)

tgt_tokenizer = Tokenizer.from_file(
    Config.TGT_TOKENIZER_PATH
)

print("Tokenizers Loaded")


# ==========================================================
# Load Model
# ==========================================================

print("\nLoading Transformer Model...")

model = Transformer().to(device)

model = load_model(
    model,
    Config.BEST_MODEL_PATH,
    device
)

print("Model Loaded Successfully")
# ==========================================================
# Greedy Decode
# ==========================================================

def greedy_decode(

        model,

        source,

        source_mask,

        max_length

):

    sos_idx = Config.SOS_IDX

    eos_idx = Config.EOS_IDX

    with torch.no_grad():

        encoder_output = model.encode(

            source,

            source_mask

        )

        decoder_input = torch.tensor(

            [[sos_idx]],

            dtype=torch.long,

            device=device

        )

        while decoder_input.size(1) < max_length:

            decoder_mask = create_target_mask(

                decoder_input,

                Config.PAD_IDX

            ).to(device)

            decoder_output = model.decode(

                decoder_input,

                encoder_output,

                source_mask,

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
# Translate
# ==========================================================

def translate(sentence):

    source_ids = src_tokenizer.encode(

        sentence

    ).ids

    source_ids = (

        [Config.SOS_IDX]

        + source_ids

        + [Config.EOS_IDX]

    )

    source_ids = source_ids[:Config.MAX_SEQ_LEN]

    source_ids += [

        Config.PAD_IDX

    ] * (

        Config.MAX_SEQ_LEN - len(source_ids)

    )

    source_tensor = torch.tensor(

        source_ids,

        dtype=torch.long

    ).unsqueeze(0).to(device)

    source_mask = create_padding_mask(

        source_tensor,

        Config.PAD_IDX

    ).to(device)

    prediction = beam_search_decode(

        model,

        source_tensor,

        source_mask,
        beam_size=5,

        max_length=Config.MAX_SEQ_LEN

    )

    translated = tgt_tokenizer.decode(

        prediction.tolist()

    )

    return translated

# ==========================================================
# Example
# ==========================================================

sentence = "I love artificial intelligence"

translation = translate(

    sentence

)

print()

print("Source")

print(sentence)

print()

print("Prediction")

print(translation)
# ==========================================================
# Beam Search Decoding
# ==========================================================

import torch.nn.functional as F


def beam_search_decode(
    model,
    source,
    source_mask,
    beam_size=5,
    max_length=Config.MAX_SEQ_LEN
):
    """
    Beam Search Decoding

    Parameters
    ----------
    beam_size : int
        Number of beams to keep.

    max_length : int
        Maximum generated sentence length.
    """

    model.eval()

    sos_idx = Config.SOS_IDX
    eos_idx = Config.EOS_IDX

    with torch.no_grad():

        encoder_output = model.encode(
            source,
            source_mask
        )

        beams = [
            (
                torch.tensor(
                    [[sos_idx]],
                    device=device
                ),
                0.0
            )
        ]

        completed = []

        for _ in range(max_length):

            new_beams = []

            for sequence, score in beams:

                if sequence[0, -1].item() == eos_idx:

                    completed.append(
                        (sequence, score)
                    )
                    continue

                decoder_mask = create_target_mask(
                    sequence,
                    Config.PAD_IDX
                ).to(device)

                decoder_output = model.decode(

                    sequence,

                    encoder_output,

                    source_mask,

                    decoder_mask

                )

                logits = model.project(
                    decoder_output
                )

                log_probs = F.log_softmax(
                    logits[:, -1],
                    dim=-1
                )

                topk_probs, topk_indices = torch.topk(

                    log_probs,

                    beam_size

                )

                for k in range(beam_size):

                    token = topk_indices[0, k].view(1, 1)

                    probability = topk_probs[0, k].item()

                    new_sequence = torch.cat(

                        [
                            sequence,
                            token
                        ],

                        dim=1

                    )

                    new_beams.append(

                        (
                            new_sequence,
                            score + probability
                        )

                    )

            beams = sorted(

                new_beams,

                key=lambda x: x[1],

                reverse=True

            )[:beam_size]

            if len(completed) >= beam_size:
                break

        if len(completed) == 0:
            completed = beams

        best_sequence = max(

            completed,

            key=lambda x: x[1]

        )[0]

        return best_sequence.squeeze(0)

# ==========================================================
# Interactive Translation
# ==========================================================

def interactive_translation():

    print()

    print("=" * 70)

    print("Transformer Translator")

    print("Type 'exit' to quit")

    print("=" * 70)

    while True:

        sentence = input("\nSource Sentence : ")

        if sentence.lower() == "exit":
            break

        prediction = translate(sentence)

        print()

        print("Translation")

        print(prediction)
# ==========================================================
# Batch Translation
# ==========================================================

def batch_translate(sentences):

    outputs = []

    for sentence in sentences:

        outputs.append(

            translate(sentence)

        )

    return outputs
# ==========================================================
# Main
# ==========================================================

def main():

    interactive_translation()


if __name__ == "__main__":

    main()            
   
