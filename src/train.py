"""
Script d'entraînement du modèle de résumé abstractif.

Usage :
    python -m src.train
"""

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from src.tokenizer import load_tokenizer
from src.dataset import SummarizationDataset, collate_fn
from src.model import Seq2Seq


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(model, dataloader, optimizer, criterion, device, teacher_forcing_ratio, is_train=True):
    model.train() if is_train else model.eval()
    total_loss = 0.0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in tqdm(dataloader, leave=False):
            source = batch["source"].to(device)
            target = batch["target"].to(device)

            if is_train:
                optimizer.zero_grad()

            output = model(source, target, teacher_forcing_ratio if is_train else 0.0)
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            target_flat = target[:, 1:].reshape(-1)

            loss = criterion(output, target_flat)

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_GRAD_NORM)
                optimizer.step()

            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    device = get_device()
    print(f"Device utilisé : {device}")

    tokenizer = load_tokenizer()
    vocab_size = tokenizer.get_vocab_size()

    train_dataset = SummarizationDataset(config.TRAIN_FILE, tokenizer)
    val_dataset = SummarizationDataset(config.VAL_FILE, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = Seq2Seq(vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=config.PAD_TOKEN_ID)

    best_val_loss = float("inf")
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, optimizer, criterion, device,
                                config.TEACHER_FORCING_RATIO, is_train=True)
        val_loss = run_epoch(model, val_loader, optimizer, criterion, device,
                              0.0, is_train=False)

        print(f"Epoch {epoch}/{config.NUM_EPOCHS} - train_loss={train_loss:.4f} - val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.MODEL_CHECKPOINT)
            print(f"  -> Nouveau meilleur modèle sauvegardé ({config.MODEL_CHECKPOINT})")


if __name__ == "__main__":
    main()