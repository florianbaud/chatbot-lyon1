import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch

from tqdm import tqdm_notebook
from ChatbotDS.utils.utils import import_template2
from ChatbotDS.preprocessing.preprocessing_methods import normalizeString, bertPreprocessing, SpacyPreprocessing, RegexPreprocessing2


class Evaluation(nn.Module):

    def __init__(self, model, variable_data, **kwargs):
        super().__init__()

        self.last_layer = nn.Sigmoid()

        self.model = model
        self.variable_data = variable_data
        self.device = self.set_device(kwargs.get('device', 'cpu'))
        self.verbose = kwargs.get('verbose', True)
        self.table_keys = [[int(b) for b in s]
                           for s in self.model.voc.table.keys()]
        self.table_keys = torch.Tensor(self.table_keys).to(self.device)
        self.model.to(self.device)
        self.init_hidden = self.model.encoder.init_hidden().to(self.device)
        self.preprocess = kwargs.get("preprocess", 'base')
        if self.preprocess == 'bert':
            self.preprocess_method = bertPreprocessing().bertized_sentence
        elif self.preprocess == 'base':
            self.preprocess_method = normalizeString
        elif self.preprocess == 'base2':
            self.preprocess_method = RegexPreprocessing2(unk_token='unk_token')
        elif self.preprocess == 'spacy':
            self.preprocess_method = SpacyPreprocessing().lemmatized_sentence
        else:
            raise ValueError(
                "{} is not an appropriate preprocessing method.".format(self.preprocess))

    def forward(self, input_variable, memory, return_weights: bool = False):
        input_variable = torch.Tensor(input_variable).to(self.device).long()
        model_output = self.model(
            input_variable,
            self.init_hidden,
            memory,
            return_weights=return_weights,
        )
        output = (self.last_layer(model_output[0]),)
        if return_weights:
            output += (model_output[1],)
        return output

    def eval_(self, sentence, memory, return_weights: bool = False):
        sentence = sentence.split(' ')
        input_variable = [self.model.voc.word_to_index(
            w, v=self.verbose) for w in sentence]
        model_output = self(input_variable, memory, return_weights)
        output = (model_output[0][0][self.model.voc.data_memory_size:],
                  model_output[0][0][-self.model.decoder.memory_size:])
        if return_weights:
            output += (model_output[1],)
        return output

    def chat(self, print_propositions=False, print_memory=False, print_attn=False):
        self.model.eval()
        memory = torch.Tensor().new_zeros(self.model.decoder.memory_size).to(self.device)
        while 1:
            sentence = input('> ')
            if sentence == 'q':
                break
            sentence = self.preprocess_method(sentence)
            output = self.eval_(sentence, memory, return_weights=print_attn)
            # memory = output[1]
            response, codes = self.get_responses(
                code=output[0],
                nb_responses=4,
                p=1,
                thresh=0.5,
            )
            memory = torch.Tensor(
                output[1].round().tolist()[:self.model.voc.data_memory_size] +
                [float(c) for c in codes[0]],
            ).to(self.device)
            print("Bot :", response[0])
            if print_propositions:
                print("Propositions :", response)
            if print_memory:
                print('Memory :', memory.tolist()[
                      :self.model.voc.data_memory_size])
            if print_attn:
                attn_weights = output[2][0].detach().cpu().numpy()
                img_labels = [
                    self.model.voc.index2word[self.model.voc.word_to_index(
                        w, v=self.verbose)]
                    for w in sentence.split()
                ]
                fig, ax = plt.subplots()
                _ = ax.imshow(attn_weights)
                ax.set_xticks(np.arange(len(img_labels)), labels=img_labels)
                ax.set_yticks(np.arange(1), labels=['weigth'])
                for j in range(len(img_labels)):
                    _ = ax.text(
                        j, 0, f"{attn_weights[0, j]:0.2f}", ha="center", va="center", color="w")
                fig.tight_layout()
                plt.show()
                # print("Attn weights :", attn_weights)

    def eval_data(self, path, verbose=False, **kwargs):
        self.model.eval()
        data = import_template2(path)
        # noise_code = '0'*len(list(self.model.voc.table.keys())[0])
        self.f, self.c, self.w, self.t = 0, 0, 0, 0
        for d in tqdm_notebook(data):
            memory = torch.Tensor().new_zeros(self.model.decoder.memory_size).to(self.device)
            for l in d:
                sentence = self.preprocess_method(l[0])
                output = self.eval_(sentence, memory)
                memory = output[1]
                responses, _ = self.get_responses(output[0])
                prediction, target = responses[0], self.replace_variable(l[1])
                self.t += 1
                if prediction == target:
                    self.c += 1
                elif prediction == self.model.voc.unk_text:
                    self.w += 1
                else:
                    if verbose:
                        print(prediction, target)
                    self.f += 1
        f_, c_, w_ = self.f/self.t, self.c/self.t, self.w/self.t
        print(f"Fail : {f_:2.2%} ; Correct : {c_:2.2%} ; Warning : {w_:2.2%}")

    def eval_json(self, sentence, memory, nb_responses=3):
        sentence = self.preprocess_method(sentence)
        code, memory = self.eval_(sentence, memory)
        responses, _ = self.get_responses(code, nb_responses)
        return ({'bot':  responses}, memory)

    def nearest_responses(self, dist, k):
        _, index = dist.topk(k, dim=0, largest=False)
        res_index = [self.table_keys[i].int().tolist() for i in index]
        responses = [''.join([str(b) for b in r]) for r in res_index]
        return responses

    def distance(self, t, p=2):
        dist = [t.dist(k, p=p) for k in self.table_keys]
        dist = torch.Tensor(dist)
        return dist

    def get_responses(self, code, nb_responses=1, p: int = 2, thresh: float = 0.75):
        if p == 1:
            code = torch.where(
                code > thresh,
                torch.tensor(1.0).to(self.device),
                torch.tensor(0.0).to(self.device),
            )
        dist = self.distance(code, p=p)
        codes = self.nearest_responses(dist, nb_responses)
        responses = [self.replace_variable(
            self.model.voc.table[r]) for r in codes]
        return responses, codes

    def set_device(self, device):
        cuda_is_available = torch.cuda.is_available()
        if not cuda_is_available:
            print("CUDA is not available")
        device = "cuda" if (cuda_is_available and device == "cuda") else "cpu"
        print("Evaluation on {}".format(device))
        return torch.device(device)

    def replace_variable(self, sentence):
        for k in self.variable_data.keys():
            sentence = sentence.replace(k, self.variable_data[k])
        return sentence
