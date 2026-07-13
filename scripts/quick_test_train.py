"""
Test rapide du pipeline d'entraînement sur un petit sous-ensemble des
données, avec un modèle réduit. Objectif : vérifier en quelques minutes
que tout fonctionne (chargement, forward, backward, sauvegarde) avant
de lancer le véritable entraînement complet avec les réglages de
config.py.

Usage :
    python -m scripts.quick_test_train
"""

import csv
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from src.tokenizer import load_tokenizer
from src.dataset import SummarizationDataset, collate_fn
from src.model import Seq2Seq

# --- Réglages réduits, uniquement pour ce test ---
N_SAMPLES = 500          # nombre d'exemples utilisés (au lieu de 392 902)
QUICK_EMBEDDING_DIM = 64
QUICK_HIDDEN_DIM = 128
QUICK_BATCH_SIZE = 8
QUICK_EPOCHS = 2


def make_small_csv(source_path, dest_path, n_samples):
    """Crée une petite version d'un CSV avec seulement les n_samples premières lignes."""
    with open(source_path, encoding="utf-8") as f_in, \
         open(dest_path, "w", encoding="utf-8", newline="") as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)
        header = next(reader)
        writer.writerow(header)
        for i, row in enumerate(reader):
            if i >= n_samples:
                break
            writer.writerow(row)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device utilisé : {device}")

    # Crée de petites versions temporaires des CSV
    small_train = os.path.join(config.DATA_DIR, "_quick_train.csv")
    small_val = os.path.join(config.DATA_DIR, "_quick_val.csv")
    make_small_csv(config.TRAIN_FILE, small_train, N_SAMPLES)
    make_small_csv(config.VAL_FILE, small_val, max(50, N_SAMPLES // 10))

    tokenizer = load_tokenizer()
    vocab_size = tokenizer.get_vocab_size()

    train_dataset = SummarizationDataset(small_train, tokenizer)
    val_dataset = SummarizationDataset(small_val, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=QUICK_BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=QUICK_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = Seq2Seq(
        vocab_size,
        embedding_dim=QUICK_EMBEDDING_DIM,
        hidden_dim=QUICK_HIDDEN_DIM,
        num_layers=1,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_TOKEN_ID)

    print(f"Entraînement test sur {len(train_dataset)} exemples, {QUICK_EPOCHS} epochs...")

    for epoch in range(1, QUICK_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            source = batch["source"].to(device)
            target = batch["target"].to(device)

            optimizer.zero_grad()
            output = model(source, target)

            output_dim = output.shape[-1]
            loss = criterion(output[:, 1:].reshape(-1, output_dim), target[:, 1:].reshape(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{QUICK_EPOCHS} - train_loss={avg_loss:.4f}")

    # Test de génération sur un exemple du jeu de validation
    model.eval()
    sample = val_dataset[0]
    source_tensor = sample["source"].unsqueeze(0).to(device)
    generated_ids = model.generate(source_tensor)[0].cpu().tolist()

    from src.tokenizer import decode
    print("\n--- Exemple de génération (modèle quasi non-entraîné, normal que ce soit imparfait) ---")
    print("Résumé généré :", decode(tokenizer, generated_ids))
    print("Résumé attendu :", decode(tokenizer, sample["target"].tolist()))

    # Nettoyage des fichiers temporaires
    os.remove(small_train)
    os.remove(small_val)

    print("\nTest terminé : le pipeline fonctionne de bout en bout.")


if __name__ == "__main__":
    main()