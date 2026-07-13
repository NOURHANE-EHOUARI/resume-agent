"""
Génère un résumé pour un texte donné, à partir du modèle entraîné.

Usage :
    python -m src.generate --text "Votre texte ici..."
    ou
    python -m src.generate --file mon_texte.txt
"""

import argparse

import torch

import config
from src.tokenizer import load_tokenizer, encode, decode
from src.model import Seq2Seq
from src.train import get_device


def summarize(text, model, tokenizer, device):
    source_ids = encode(tokenizer, text, config.MAX_SOURCE_LEN, add_sos_eos=False)
    source_tensor = torch.tensor([source_ids], dtype=torch.long, device=device)

    generated_ids = model.generate(source_tensor)[0].cpu().tolist()

    if config.EOS_TOKEN_ID in generated_ids:
        generated_ids = generated_ids[:generated_ids.index(config.EOS_TOKEN_ID)]

    return decode(tokenizer, generated_ids)


def load_model(device):
    tokenizer = load_tokenizer()
    model = Seq2Seq(tokenizer.get_vocab_size()).to(device)
    model.load_state_dict(torch.load(config.MODEL_CHECKPOINT, map_location=device))
    model.eval()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Génère un résumé d'un texte.")
    parser.add_argument("--text", type=str, help="Texte à résumer directement en argument")
    parser.add_argument("--file", type=str, help="Chemin vers un fichier .txt à résumer")
    args = parser.parse_args()

    if not args.text and not args.file:
        parser.error("Fournir --text ou --file")

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            text = f.read()
    else:
        text = args.text

    device = get_device()
    model, tokenizer = load_model(device)

    summary = summarize(text, model, tokenizer, device)
    print("\n--- Résumé généré ---")
    print(summary)


if __name__ == "__main__":
    main()