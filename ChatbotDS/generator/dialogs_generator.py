from ChatbotDS.utils.templates import Templates
from ChatbotDS.generator.dialog import Dialog
from tqdm import tqdm_notebook
import random


class DialogsGenerator():

    def __init__(self, templates_path, **kwargs):
        self.templates = kwargs.get('templates', Templates(templates_path))

    def generate_dialogs(self, **kwargs):
        self.mode = kwargs.get('mode', 'Full')
        self.templates.select(self.mode)

        self.dialogs_len = kwargs.get('dialogs_len', 1000)
        self.dialog_len = kwargs.get('dialog_len', 10)
        self.enchainements = kwargs.get(
            'enchainements', self.templates.enchainements)

        self.dialogs = [self.generate_dialog()
                        for _ in tqdm_notebook(range(self.dialogs_len))]

    def generate_dialog(self):
        dialog_len = random.randint(2, self.dialog_len)
        enchainement = [random.choice(self.enchainements)
                        for _ in range(dialog_len)]
        d = Dialog(self.templates, enchainement)
        return d.dialog

    def print_dialogs(self, k=1, print_mem=True):
        for j in range(k):
            dialog = self.dialogs[j]
            for i in range(0, len(dialog), 3):
                if print_mem:
                    print(dialog[i], dialog[i+1], sep='\t')
                else:
                    print(dialog[i], dialog[i+1], dialog[i+2], sep='\t')
            print()

    def save(self, path):
        pairs = '\n\n'.join(['\n'.join(['\t'.join(d[p:p + 3])
                                        for p in range(0, len(d), 3)]) for d in self.dialogs])
        with open(path, 'w', encoding='utf-8') as f:
            f.write(pairs)
