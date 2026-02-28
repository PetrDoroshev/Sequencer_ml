import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math


class DecoderLSTM(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_dim, latent_dim, num_layers=2, dropout=0.3):

        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(latent_dim + embed_dim, hidden_dim, num_layers, batch_first=True, 
                            dropout=dropout)
        
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        self.z_to_hidden = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.z_to_cell = nn.Linear(latent_dim, hidden_dim * num_layers)

    def forward(self, x, z, hidden=None, cell=None):
      
        batch_size, seq_len = x.size()
        embedded = self.dropout(self.embedding(x))
        
        z_expanded = z.unsqueeze(1).expand(-1, seq_len, -1)
        lstm_input = torch.cat([embedded, z_expanded], dim=-1)
        
        
        if hidden is None or cell is None:
            device = x.device
            '''
            hidden = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
            cell = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
            '''
            hidden = self.z_to_hidden(z)
            hidden = hidden.view(self.num_layers, batch_size, self.hidden_dim)

            cell = self.z_to_cell(z)
            cell = cell.view(self.num_layers, batch_size, self.hidden_dim)
            
        outputs, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        logits = self.fc(outputs)                   
        return logits, hidden, cell


        



        
