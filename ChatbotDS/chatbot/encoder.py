import torch.nn as nn
import torch


class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size, n_layers=1, bidirectional=True, dropout=0.1):
        super().__init__()

        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.cat = nn.Linear(hidden_size*2, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=n_layers,
                          bidirectional=bidirectional, dropout=(0 if n_layers == 1 else dropout))

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(input.size()[0], 1, -1)
        embedded = self.dropout(embedded)
        outputs, hidden = self.gru(embedded, hidden)
        outputs, hidden = self.dropout(outputs), self.dropout(hidden)
        last_hidden = outputs[-1]
        return outputs, last_hidden

    def init_hidden(self):
        return torch.Tensor().new_zeros((2**int(self.bidirectional))*self.n_layers, 1, self.hidden_size)
