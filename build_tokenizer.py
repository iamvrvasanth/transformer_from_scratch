import os

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

from config import Config


def get_sentences(dataset, lang):
    for item in dataset:
        yield item["translation"][lang]


def build_tokenizer(lang, save_path):

    dataset = load_dataset(
        Config.DATASET_NAME,
        f"{Config.SRC_LANG}-{Config.TGT_LANG}",
        split="train"
    )

    tokenizer = Tokenizer(
        WordLevel(unk_token="[UNK]")
    )

    tokenizer.pre_tokenizer = Whitespace()

    trainer = WordLevelTrainer(
        special_tokens=[
            "[PAD]",
            "[SOS]",
            "[EOS]",
            "[UNK]"
        ],
        min_frequency=Config.MIN_FREQUENCY
    )

    tokenizer.train_from_iterator(
        get_sentences(dataset, lang),
        trainer=trainer
    )

    os.makedirs(
        Config.TOKENIZER_DIR,
        exist_ok=True
    )

    tokenizer.save(save_path)

    print(f"Saved: {save_path}")


if __name__ == "__main__":

    build_tokenizer(
        Config.SRC_LANG,
        Config.SRC_TOKENIZER_PATH
    )

    build_tokenizer(
        Config.TGT_LANG,
        Config.TGT_TOKENIZER_PATH
    )
