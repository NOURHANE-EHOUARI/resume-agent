import pandas as pd
import os

os.makedirs("data", exist_ok=True)
print("📥 Téléchargement direct des fichiers bruts depuis Hugging Face (zéro script, zéro blocage)...")

# URLs des fichiers Parquet bruts de MLSUM (français)
train_url = "https://huggingface.co/datasets/mlsum/resolve/main/fr/train/0000.parquet"
val_url = "https://huggingface.co/datasets/mlsum/resolve/main/fr/validation/0000.parquet"

print("  -> Téléchargement de l'entraînement...")
df_train = pd.read_parquet(train_url)
df_train = df_train.head(20000)[["text", "summary"]]  # On garde 20 000 exemples
df_train.to_csv("data/train_subset.csv", index=False, encoding="utf-8")
print("  ✅ data/train_subset.csv créé !")

print("  -> Téléchargement de la validation...")
df_val = pd.read_parquet(val_url)
df_val = df_val.head(2000)[["text", "summary"]]       # On garde 2 000 exemples
df_val.to_csv("data/val_subset.csv", index=False, encoding="utf-8")
print("  ✅ data/val_subset.csv créé !")

print("🎉 Données prêtes sur ton PC ! Tu peux maintenant zipper le dossier 'data'.")
