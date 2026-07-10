import os

# --- Chemins ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")   # colonnes attendues: text,summary
VAL_FILE = os.path.join(DATA_DIR, "val.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")

TOKENIZER_PATH = os.path.join(DATA_DIR, "tokenizer.json")
MODEL_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_model.pt")

# --- Tokenizer ---
VOCAB_SIZE = 8000
SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>"]
PAD_TOKEN_ID = 0
SOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
UNK_TOKEN_ID = 3

# --- Données ---
MAX_SOURCE_LEN = 400     # longueur max du texte source (en tokens)
MAX_SUMMARY_LEN = 80     # longueur max du résumé (en tokens)

# --- Modèle ---
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 1
DROPOUT = 0.3

# --- Entraînement ---
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 20
TEACHER_FORCING_RATIO = 0.5
CLIP_GRAD_NORM = 1.0
DEVICE = "cuda"  # sera automatiquement remplacé par "cpu" si aucun GPU n'est disponible