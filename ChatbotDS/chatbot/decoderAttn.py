import torch
import torch.nn as nn
from ChatbotDS.chatbot.attention import Attn


class DecoderAttn(nn.Module):

    def __init__(self, hidden_size, attn_hidden_size, memory_size, code_size, method, dropout=0.1):
        super().__init__()

        self.hidden_size = hidden_size
        self.attn_hidden_size = attn_hidden_size
        self.memory_size = memory_size
        self.code_size = code_size

        self.linear = nn.Linear(hidden_size*2 + memory_size, attn_hidden_size)
        self.attn = Attn(method, attn_hidden_size)
        self.concat = nn.Linear(attn_hidden_size + hidden_size*2, code_size)
        # self.code_size_linear = nn.Linear(hidden_size*2, code_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, encoder_hidden, encoder_outputs, memory, return_weights: bool = False):
        linear_input = torch.cat(
            (encoder_hidden.view(1, -1), memory.view(1, -1)), 1)
        output = self.linear(linear_input).tanh()
        output = self.dropout(output)
        attn_weights = self.attn(output, encoder_outputs)
        attn_weights = self.dropout(attn_weights)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        context = context.squeeze(1)
        concat_input = torch.cat((output, context), 1)
        output = (self.concat(concat_input),)
        # output = (self.code_size_linear(context),)
        if return_weights:
            output += (attn_weights,)
        return output
