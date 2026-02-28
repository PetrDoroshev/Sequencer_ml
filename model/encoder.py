import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math

class EncoderLSTM(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_dim, latent_dim, num_layers=2, dropout=0.3):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout,
                            bidirectional=False)
        
        self.dropout = nn.Dropout(dropout)
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar =  nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):

        embedded = self.dropout(self.embedding(x))
        outputs, (hidden, cell) = self.lstm(embedded)

        last_hidden = hidden[-1]

        mu = self.fc_mu(last_hidden)
        logvar = self.fc_logvar(last_hidden)

        return (mu, logvar)