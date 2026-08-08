"""
config.py

This file contains all hyperparameters and project configurations.
Changing values here will automatically update the whole project.
"""


class Config:

 

    # Embedding Dimension (d_model)
    D_MODEL = 512

    # Number of Attention Heads
    N_HEADS = 8

    # Number of Encoder Layers
    N_ENCODER_LAYERS = 6

    # Number of Decoder Layers
    N_DECODER_LAYERS = 6

    # Feed Forward Hidden Dimension
    D_FF = 2048

    # Dropout Probability
    DROPOUT = 0.1

    # ==========================================================
    # Vocabulary
    # ==========================================================

    # Source Vocabulary Size
    SRC_VOCAB_SIZE = 10000

    # Target Vocabulary Size
    TGT_VOCAB_SIZE = 10000

    # ==========================================================
    # Sequence
    # ==========================================================

    # Maximum sequence length
    MAX_SEQ_LEN = 512

    # ==========================================================
    # Training
    # ==========================================================

    BATCH_SIZE = 32

    EPOCHS = 20

    LEARNING_RATE = 1e-4

    WEIGHT_DECAY = 1e-5

    LABEL_SMOOTHING = 0.1

    # ==========================================================
    # Device
    # ==========================================================

    DEVICE = "cuda"

    # ==========================================================
    # Special Tokens
    # ==========================================================

    PAD_IDX = 0
    BOS_IDX = 1
    EOS_IDX = 2
    UNK_IDX = 3

    # ==========================================================
    # Checkpoint
    # ==========================================================

    CHECKPOINT_DIR = "checkpoints"

    MODEL_NAME = "transformer.pt"

    # ==========================================================
    # Random Seed
    # ==========================================================

    SEED = 42
 """
config.py

This file contains all hyperparameters and project configurations.
Changing values here will automatically update the whole project.
"""


class Config:

 

    # Embedding Dimension (d_model)
    D_MODEL = 512

    # Number of Attention Heads
    N_HEADS = 8

    # Number of Encoder Layers
    N_ENCODER_LAYERS = 6

    # Number of Decoder Layers
    N_DECODER_LAYERS = 6

    # Feed Forward Hidden Dimension
    D_FF = 2048

    # Dropout Probability
    DROPOUT = 0.1

    # ==========================================================
    # Vocabulary
    # ==========================================================

    # Source Vocabulary Size
    SRC_VOCAB_SIZE = 10000

    # Target Vocabulary Size
    TGT_VOCAB_SIZE = 10000

    # ==========================================================
    # Sequence
    # ==========================================================

    # Maximum sequence length
    MAX_SEQ_LEN = 512

    # ==========================================================
    # Training
    # ==========================================================

    BATCH_SIZE = 32

    EPOCHS = 20

    LEARNING_RATE = 1e-4

    WEIGHT_DECAY = 1e-5

    LABEL_SMOOTHING = 0.1

    # ==========================================================
    # Device
    # ==========================================================

    DEVICE = "cuda"

    # ==========================================================
    # Special Tokens
    # ==========================================================

    PAD_IDX = 0
    BOS_IDX = 1
    EOS_IDX = 2
    UNK_IDX = 3

    # ==========================================================
    # Checkpoint
    # ==========================================================

    CHECKPOINT_DIR = "checkpoints"

    MODEL_NAME = "transformer.pt"

    # ==========================================================
    # Random Seed
    # ==========================================================

    SEED = 42
 # ==========================================================
# Dataset
# ==========================================================

 DATASET_NAME = "opus_books"
 
 SRC_LANG = "en"
 TGT_LANG = "fr"
 
 # ==========================================================
 # Tokenizers
 # ==========================================================
 
 TOKENIZER_DIR = "tokenizers"
 
 SRC_TOKENIZER_PATH = "tokenizers/src_tokenizer.json"
 TGT_TOKENIZER_PATH = "tokenizers/tgt_tokenizer.json"
 
 MIN_FREQUENCY = 2
