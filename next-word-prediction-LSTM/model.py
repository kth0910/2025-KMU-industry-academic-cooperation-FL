# model.py
import torch
import torch.nn as nn

class LSTMNextWordModel(nn.Module):
    def __init__(self, vocab_size=10000, embedding_dim=192, hidden_size=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embedding(x)
        out, hidden = self.lstm(emb, hidden)
        out = out[:, -1, :]
        logits = self.fc(out)
        return logits, hidden