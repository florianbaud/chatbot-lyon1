import torch
import torch.nn as nn
from ChatbotDS.chatbot.encoder import Encoder
from ChatbotDS.chatbot.decoderAttn import DecoderAttn
# from ChatbotDS.utils.voc import Voc


class Chatbot(nn.Module):
    """
    Arguments optionels :
        - encoder_layers (int)
        - bidirectional (bool)
        - encoder_dropout (proba)
        - decoder_dropout (proba)
    """

    def __init__(self, voc, hidden_size, attn_method, attn_hidden_size, memory_size, code_size, **kwargs):
        super().__init__()

        self.iterations = 0

        self.voc = voc
        self.hidden_size = hidden_size
        self.attn_method = attn_method
        self.attn_hidden_size = attn_hidden_size
        self.memory_size = memory_size
        self.code_size = code_size
        self.encoder_layers = kwargs.get('encoder_layers', 1)
        self.bidirectional = kwargs.get('bidirectional', True)
        self.encoder_dropout = kwargs.get('encoder_dropout', 0.1)
        self.decoder_dropout = kwargs.get('decoder_dropout', 0.1)

        self.encoder = Encoder(
            self.voc.num_words,
            self.hidden_size,
            n_layers=self.encoder_layers,
            bidirectional=self.bidirectional,
            dropout=self.encoder_dropout,
        )
        self.decoder = DecoderAttn(
            self.hidden_size,
            self.attn_hidden_size,
            self.memory_size,
            self.code_size,
            self.attn_method,
            dropout=self.decoder_dropout,
        )
        self.nb_parameters = self.count_parameters()

    def forward(self, input_variable, hidden, memory, return_weights: bool = False):
        encoder_outputs, encoder_hiddens = self.encoder(input_variable, hidden)
        decoder_outputs = self.decoder(
            encoder_hiddens,
            encoder_outputs,
            memory,
            return_weights=return_weights,
        )
        return decoder_outputs

    def count_parameters(self):
        count = 0
        for p in self.parameters():
            if p.requires_grad == True:
                count += p.nelement()
        return count

    def save(self, path):
        torch.save({
            "iterations": self.iterations,
            "voc": self.voc,
            "hidden_size": self.hidden_size,
            "attn_method": self.attn_method,
            "attn_hidden_size": self.attn_hidden_size,
            "memory_size": self.memory_size,
            "code_size": self.code_size,
            "encoder_layers": self.encoder_layers,
            "encoder_dropout": self.encoder_dropout,
            "decoder_dropout": self.decoder_dropout,
            "bidirectional": self.bidirectional,
            "state_dict": self.state_dict()
        }, path)

    @staticmethod
    def load(path, map_location='cpu'):

        if map_location == 'cuda' and not torch.cuda.is_available():
            map_location = 'cpu'
            print('CUDA is not available, model is loading on CPU...')

        m = torch.load(path, map_location=map_location)
        voc = m.get('voc')
        hidden_size = m.get('hidden_size')
        attn_method = m.get('attn_method')
        attn_hidden_size = m.get('attn_hidden_size')
        memory_size = m.get('memory_size')
        code_size = m.get('code_size')
        encoder_layers = m.get('encoder_layers')
        encoder_dropout = m.get('encoder_dropout')
        decoder_dropout = m.get('decoder_dropout')
        bidirectional = m.get('bidirectional')
        state_dict = m.get('state_dict')
        iterations = m.get('iterations')

        model = Chatbot(
            voc=voc,
            hidden_size=hidden_size,
            attn_method=attn_method,
            attn_hidden_size=attn_hidden_size,
            memory_size=memory_size,
            code_size=code_size,
            encoder_layers=encoder_layers,
            bidirectional=bidirectional,
            encoder_dropout=encoder_dropout,
            decoder_dropout=decoder_dropout,
        )
        model.load_state_dict(state_dict)
        model.iterations = iterations
        return model
