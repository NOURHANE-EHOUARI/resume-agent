"""
Crée un sous-ensemble réduit de train.csv et val.csv, pour permettre
un entraînement complet en un temps raisonnable sur CPU.

Usage :
    python -m scripts.prepare_subset
"""

import csv
import os

import config

N_TRAIN = 20000
N_VAL = 2000


def make_subset(source_path, dest_path, n_samples):
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
    print(f"{dest_path} créé avec {min(i + 1, n_samples)} exemples")


def main():
    full_train = os.path.join(config.DATA_DIR, "train.csv")
    full_val = os.path.join(config.DATA_DIR, "val.csv")

    make_subset(full_train, config.TRAIN_FILE, N_TRAIN)
    make_subset(full_val, config.VAL_FILE, N_VAL)


if __name__ == "__main__":
    main()