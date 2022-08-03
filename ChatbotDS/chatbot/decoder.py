import torch.nn as nn
from torch import cat


class Decoder(nn.Module):

    def __init__(self, hidden_size, memory_size, ecoc_size, dropout=0.1):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.ecoc_size = ecoc_size
        
        self.linear = nn.Linear(hidden_size*2 + memory_size, ecoc_size)

    def forward(self, encoder_hidden, memory):
        linear_input = cat((encoder_hidden.view(1, -1), memory.view(1, -1)), 1)
        output = self.linear(linear_input)
        return output
