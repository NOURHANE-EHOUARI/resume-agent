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

    dataset = load_dataset("reciTAL/mlsum", revision="refs/convert/parquet")

    # Vérification : affiche la structure du dataset avant de convertir,
    # pour être sûr des colonnes disponibles et des langues présentes.
    print("Splits disponibles :", list(dataset.keys()))
    print("Colonnes disponibles :", dataset["train"].column_names)
    print("Exemple brut :", dataset["train"][0])

    # Si une colonne de langue existe (souvent nommée 'lang' ou 'topic'),
    # on ne garde que les exemples en français.
    if "lang" in dataset["train"].column_names:
        print("Filtrage sur la langue française...")
        dataset = dataset.filter(lambda example: example["lang"] == "fr")

    convert_split(dataset["train"], os.path.join(OUTPUT_DIR, "train.csv"))
    convert_split(dataset["validation"], os.path.join(OUTPUT_DIR, "val.csv"))
    convert_split(dataset["test"], os.path.join(OUTPUT_DIR, "test.csv"))

    print("Conversion terminée : train.csv, val.csv, test.csv créés dans data/")


if __name__ == "__main__":
    main()