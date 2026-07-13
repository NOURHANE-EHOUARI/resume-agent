"""
Évaluation du modèle entraîné avec la métrique ROUGE.

Usage :
    python -m src.evaluate
"""

import torch
from torch.utils.data import DataLoader
from rouge_score import rouge_scorer
from tqdm import tqdm

import config
from src.tokenizer import load_tokenizer, decode
from src.dataset import SummarizationDataset, collate_fn
from src.model import Seq2Seq
from src.train import get_device


def evaluate():
    device = get_device()
    tokenizer = load_tokenizer()
    vocab_size = tokenizer.get_vocab_size()

    test_dataset = SummarizationDataset(config.TEST_FILE, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = Seq2Seq(vocab_size).to(device)
    model.load_state_dict(torch.load(config.MODEL_CHECKPOINT, map_location=device))
    model.eval()

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    with torch.no_grad():
        for batch in tqdm(test_loader):
            source = batch["source"].to(device)
            target = batch["target"]

            generated_ids = model.generate(source).cpu().tolist()
            target_ids = target.tolist()

            for gen_ids, ref_ids in zip(generated_ids, target_ids):
                hypothesis = decode(tokenizer, gen_ids)
                reference = decode(tokenizer, ref_ids)
                result = scorer.score(reference, hypothesis)
                for key in scores:
                    scores[key].append(result[key].fmeasure)

    print("\nRésultats ROUGE (F-mesure moyenne) :")
    for key, values in scores.items():
        avg = sum(values) / len(values) if values else 0.0
        print(f"  {key}: {avg:.4f}")


if __name__ == "__main__":
    evaluate()