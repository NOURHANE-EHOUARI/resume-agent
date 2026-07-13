"""
Dataset PyTorch pour le résumé abstractif.

Lit un fichier CSV avec deux colonnes "text" et "summary", et
transforme chaque paire en tenseurs d'IDs de tokens prêts pour
l'entraînement.
"""

import csv

import torch
from torch.utils.data import Dataset

import config
from src.tokenizer import encode


class SummarizationDataset(Dataset):
    def __init__(self, csv_path, tokenizer,
                 max_source_len=config.MAX_SOURCE_LEN,
                 max_summary_len=config.MAX_SUMMARY_LEN):
        self.tokenizer = tokenizer
        self.max_source_len = max_source_len
        self.max_summary_len = max_summary_len
        self.samples = self._load(csv_path)

    @staticmethod
    def _load(csv_path):
        samples = []
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get("text", "").strip()
                summary = row.get("summary", "").strip()
                if text and summary:
                    samples.append((text, summary))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, summary = self.samples[idx]

        source_ids = encode(self.tokenizer, text, self.max_source_len, add_sos_eos=False)
        target_ids = encode(self.tokenizer, summary, self.max_summary_len, add_sos_eos=True)

        return {
            "source": torch.tensor(source_ids, dtype=torch.long),
            "target": torch.tensor(target_ids, dtype=torch.long),
        }


def collate_fn(batch):
    """Empile un batch d'exemples déjà paddés à longueur fixe."""
    sources = torch.stack([item["source"] for item in batch])
    targets = torch.stack([item["target"] for item in batch])
    return {"source": sources, "target": targets}