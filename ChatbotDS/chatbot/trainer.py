from copy import deepcopy
from random import choice, random, randint
from tqdm.auto import tqdm

import torch.nn as nn
import torch


class Trainer(nn.Module):
    """
    Arguments optionels :
        - device (string): cpu ou cuda.
    """

    def __init__(self, model, data, device="cpu", save_every=None):
        super().__init__()

        self.criterion = nn.BCEWithLogitsLoss()
        self.sigm = nn.Sigmoid()

        self.data = data
        self.model = model
        self.device = self.set_device(device)
        self.save_every = save_every
        self.init_hidden = self.model.encoder.init_hidden().to(self.device)
        self.to(self.device)
        print(
            f"Le modèle a {self.model.nb_parameters:,} paramètres à optimiser.")

    def forward(self, n_iterations, learning_rate, **kwargs):
        """
        Arguments optionels :
            - teacher_forcing (proba)
            - clip (int)
            - print_every (int)
            - progress (bool)
            - noise_p (proba)
            - noise_min_len (int)
            - noise_max_len (int)
        """

        teacher_forcing = kwargs.get('teacher_forcing', 0)
        self.clip = kwargs.get('clip', 50)
        print_every = kwargs.get('print_every', 100)
        noise_p = kwargs.get('noise_p', 0.2)
        noise_min_len = kwargs.get('noise_min_len', 4)
        noise_max_len = kwargs.get('noise_max_len', 20)
        noise_p_word = kwargs.get('noise_p_word', 0)

        self.model.train()
        self.encoder_optim = torch.optim.Adam(
            self.model.encoder.parameters(), lr=learning_rate)
        self.decoder_optim = torch.optim.Adam(
            self.model.decoder.parameters(), lr=learning_rate)
        sum_loss = 0
        iterations = range(1, n_iterations + 1)
        if kwargs.get('progress', False):
            iterations = tqdm(iterations)
        # iterations = tqdm(range(1, n_iterations + 1)) if kwargs.get(
        #     'progress', False) else range(1, n_iterations + 1)
        for iteration in iterations:
            self.train_data = deepcopy(choice(self.data))
            self.insert_noise(
                p=noise_p,
                min_len=noise_min_len,
                max_len=noise_max_len,
            )
            dialog_memory = torch.Tensor(
                [0 for _ in range(self.model.decoder.memory_size)]).to(self.device)
            for pair in self.train_data:
                input_variable = [self.model.voc.word_to_index(
                    w, p=noise_p_word) for w in pair[0].split(' ')]
                input_variable = torch.Tensor(
                    input_variable).long().to(self.device)
                target_variable = [int(s) for s in pair[1]]
                target_variable = torch.Tensor(
                    target_variable).view(1, -1).float().to(self.device)
                loss, outputs = self.train_(
                    input_variable, target_variable, dialog_memory)
                if teacher_forcing > random():
                    dialog_memory = target_variable[0][:self.model.decoder.memory_size].float(
                    )
                else:
                    dialog_memory = outputs[0][0][:self.model.decoder.memory_size].detach(
                    )
                    dialog_memory = torch.where(
                        self.sigm(dialog_memory) > 0.9,
                        torch.tensor(1.0).to(self.device),
                        torch.tensor(0.0).to(self.device),
                    )
                    # dialog_memory = torch.bernoulli(dialog_memory)
                sum_loss += loss
            if iteration % print_every == 0:
                print_loss_avg = sum_loss / print_every
                print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(
                    iteration, (iteration / n_iterations) * 100, print_loss_avg))
                sum_loss = 0
            # if iteration % self.save_every == 0:
                # self.model
                # pass
            self.model.iterations += 1

    def train_(self, input_variable, target_variable, dialog_memory):
        self.encoder_optim.zero_grad()
        self.decoder_optim.zero_grad()
        outputs = self.model(input_variable, self.init_hidden,
                             dialog_memory, return_weights=False)
        loss = self.criterion(outputs[0], target_variable)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.encoder.parameters(), self.clip)
        nn.utils.clip_grad_norm_(self.model.decoder.parameters(), self.clip)
        self.encoder_optim.step()
        self.decoder_optim.step()
        return loss, outputs

    def insert_noise(self, p=0.2, min_len=4, max_len=20, memory_len=6):
        k = 1
        code_noise = '0'*(self.model.decoder.code_size - memory_len)
        diag_copy = deepcopy(self.train_data)
        for i, line in enumerate(diag_copy):
            if random() < p:
                len_sent = randint(min_len, max_len)
                words = [self.model.voc.index2word[randint(
                    4, self.model.voc.num_words-1)] for _ in range(len_sent)]
                rand_unk_i = randint(0, len(words)-1)
                words.insert(rand_unk_i, self.model.voc.unk_token)
                self.train_data.insert(
                    i + k, [' '.join(words), code_noise + line[1][:memory_len]])
                k += 1

    def set_device(self, device):
        cuda_is_available = torch.cuda.is_available()
        if not cuda_is_available:
            print("CUDA is not available")
        device = "cuda" if (cuda_is_available and device == "cuda") else "cpu"
        print("Training on {}".format(device))
        return torch.device(device)
