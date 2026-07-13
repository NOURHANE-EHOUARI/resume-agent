"""
Modèle encodeur-décodeur (Seq2Seq) avec mécanisme d'attention de Bahdanau,
entraîné entièrement à partir de zéro.

Aucune couche n'est pré-entraînée : les embeddings, les LSTM et
l'attention démarrent avec des poids initialisés aléatoirement et
apprennent uniquement à partir du corpus d'entraînement du projet.
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=config.PAD_TOKEN_ID)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, source):
        embedded = self.embedding(source)
        outputs, (hidden, cell) = self.lstm(embedded)

        hidden = torch.tanh(self.fc_hidden(torch.cat((hidden[-2], hidden[-1]), dim=1)))
        cell = torch.tanh(self.fc_cell(torch.cat((cell[-2], cell[-1]), dim=1)))
        return outputs, hidden, cell


class Attention(nn.Module):
    """Attention additive de Bahdanau."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 3, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        src_len = encoder_outputs.shape[1]
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)

        energy = torch.tanh(self.attn(torch.cat((decoder_hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=config.PAD_TOKEN_ID)
        self.attention = Attention(hidden_dim)
        self.lstm = nn.LSTM(
            embedding_dim + hidden_dim * 2, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.fc_out = nn.Linear(hidden_dim * 3 + embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, cell, encoder_outputs):
        input_token = input_token.unsqueeze(1)
        embedded = self.dropout(self.embedding(input_token))

        attn_weights = self.attention(hidden, encoder_outputs)
        attn_weights = attn_weights.unsqueeze(1)
        context = torch.bmm(attn_weights, encoder_outputs)

        lstm_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden.unsqueeze(0), cell.unsqueeze(0)))
        hidden, cell = hidden.squeeze(0), cell.squeeze(0)

        output = output.squeeze(1)
        embedded = embedded.squeeze(1)
        context = context.squeeze(1)

        prediction = self.fc_out(torch.cat((output, context, embedded), dim=1))
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size,
                 embedding_dim=config.EMBEDDING_DIM,
                 hidden_dim=config.HIDDEN_DIM,
                 num_layers=config.NUM_LAYERS,
                 dropout=config.DROPOUT):
        super().__init__()
        self.encoder = Encoder(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
        self.decoder = Decoder(vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
        self.vocab_size = vocab_size

    def forward(self, source, target, teacher_forcing_ratio=config.TEACHER_FORCING_RATIO):
        batch_size, target_len = target.shape
        device = source.device

        outputs = torch.zeros(batch_size, target_len, self.vocab_size, device=device)
        encoder_outputs, hidden, cell = self.encoder(source)

        input_token = target[:, 0]
        for t in range(1, target_len):
            prediction, hidden, cell = self.decoder(input_token, hidden, cell, encoder_outputs)
            outputs[:, t] = prediction

            use_teacher_forcing = random.random() < teacher_forcing_ratio
            top1 = prediction.argmax(1)
            input_token = target[:, t] if use_teacher_forcing else top1

        return outputs

    @torch.no_grad()
    def generate(self, source, max_len=config.MAX_SUMMARY_LEN):
        """Génération gloutonne (greedy decoding) pour l'inférence."""
        self.eval()
        device = source.device
        batch_size = source.shape[0]

        encoder_outputs, hidden, cell = self.encoder(source)
        input_token = torch.full((batch_size,), config.SOS_TOKEN_ID, dtype=torch.long, device=device)

        generated = []
        for _ in range(max_len):
            prediction, hidden, cell = self.decoder(input_token, hidden, cell, encoder_outputs)
            top1 = prediction.argmax(1)
            generated.append(top1)
            input_token = top1

        return torch.stack(generated, dim=1)