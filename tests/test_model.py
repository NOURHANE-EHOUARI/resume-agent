"""
Test basique : vérifie que le modèle peut faire un forward pass et une
génération sans erreur, avec des données factices.

Usage :
    pytest tests/test_model.py
"""

import torch

from src.model import Seq2Seq


def test_forward_pass_shapes():
    vocab_size = 100
    batch_size = 4
    src_len = 20
    tgt_len = 10

    model = Seq2Seq(vocab_size, embedding_dim=32, hidden_dim=64, num_layers=1, dropout=0.0)

    source = torch.randint(0, vocab_size, (batch_size, src_len))
    target = torch.randint(0, vocab_size, (batch_size, tgt_len))

    output = model(source, target)
    assert output.shape == (batch_size, tgt_len, vocab_size)


def test_generate_shapes():
    vocab_size = 100
    batch_size = 2
    src_len = 15
    max_len = 12

    model = Seq2Seq(vocab_size, embedding_dim=32, hidden_dim=64, num_layers=1, dropout=0.0)
    source = torch.randint(0, vocab_size, (batch_size, src_len))

    generated = model.generate(source, max_len=max_len)
    assert generated.shape == (batch_size, max_len)