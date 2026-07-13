"""
Entraînement et chargement d'un tokenizer BPE (Byte-Pair Encoding)
entraîné intégralement sur le corpus du projet.

Aucun vocabulaire ni modèle de langage externe n'est utilisé : le
tokenizer apprend ses fusions de sous-mots uniquement à partir des
textes fournis dans data/train.csv (colonnes "text" et "summary").
"""

import csv
import os

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

import config


def _iter_corpus_lines(csv_path):
    """Génère toutes les lignes de texte (source + résumé) d'un CSV."""
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row["text"]
            yield row["summary"]


def train_tokenizer(train_csv=config.TRAIN_FILE, vocab_size=config.VOCAB_SIZE,
                     save_path=config.TOKENIZER_PATH):
    """Entraîne un tokenizer BPE sur le corpus d'entraînement et le sauvegarde."""
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=config.SPECIAL_TOKENS,
        min_frequency=2,
    )

    corpus_iterator = _iter_corpus_lines(train_csv)
    tokenizer.train_from_iterator(corpus_iterator, trainer=trainer)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokenizer.save(save_path)
    print(f"Tokenizer entraîné et sauvegardé dans {save_path} "
          f"(vocab_size={tokenizer.get_vocab_size()})")
    return tokenizer


def load_tokenizer(path=config.TOKENIZER_PATH):
    """Charge un tokenizer déjà entraîné."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Aucun tokenizer trouvé à {path}. "
            "Lance d'abord `python -m src.tokenizer` pour l'entraîner."
        )
    return Tokenizer.from_file(path)


def encode(tokenizer, text, max_len, add_sos_eos=False):
    """Encode un texte en liste d'IDs, tronquée/complétée à max_len."""
    ids = tokenizer.encode(text).ids
    if add_sos_eos:
        ids = [config.SOS_TOKEN_ID] + ids + [config.EOS_TOKEN_ID]
    ids = ids[:max_len]
    ids += [config.PAD_TOKEN_ID] * (max_len - len(ids))
    return ids


def decode(tokenizer, ids):
    """Décode une liste d'IDs en texte, en ignorant le padding et les tokens spéciaux."""
    ids = [i for i in ids if i not in (config.PAD_TOKEN_ID, config.SOS_TOKEN_ID, config.EOS_TOKEN_ID)]
    return tokenizer.decode(ids)


if __name__ == "__main__":
    train_tokenizer()