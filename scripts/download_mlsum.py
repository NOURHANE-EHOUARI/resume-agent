from datasets import load_dataset
import csv
import os

OUTPUT_DIR = "data"


def convert_split(dataset_split, output_path):
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "summary"])
        for example in dataset_split:
            writer.writerow([example["text"], example["summary"]])


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dataset = load_dataset(
        "reciTAL/mlsum",
        revision="refs/convert/parquet",
        data_files={
            "train": "fr/train/*.parquet",
            "validation": "fr/validation/*.parquet",
            "test": "fr/test/*.parquet",
        },
    )

    print("Splits disponibles :", list(dataset.keys()))
    print("Nombre d'exemples :", {k: len(v) for k, v in dataset.items()})
    print("Exemple brut :", dataset["train"][0])

    convert_split(dataset["train"], os.path.join(OUTPUT_DIR, "train.csv"))
    convert_split(dataset["validation"], os.path.join(OUTPUT_DIR, "val.csv"))
    convert_split(dataset["test"], os.path.join(OUTPUT_DIR, "test.csv"))

    print("Conversion terminée : train.csv, val.csv, test.csv créés dans data/")


if __name__ == "__main__":
    main()