from ChatbotDS.utils.voc import Voc
from ChatbotDS.preprocessing.preprocessing_methods import normalizeString, bertPreprocessing, SpacyPreprocessing, RegexPreprocessing2
from copy import deepcopy
from tqdm import tqdm
from zipfile import ZipFile
from random import randint, random


class PreprocessingDialogs():
    """
        Preprocessing class data before traning the chatbot.
        kwargs: preprocess, fake_q, fake_q_text, fake_q_p.
    """

    def __init__(self, name, diags_path, mem_len, **kwargs):
        self.mem_len = mem_len
        self.prep_diags, self.table = self.import_diags(diags_path)
        self.process_output = kwargs.get('process_output', False)
        self.preprocess = kwargs.get('preprocess', "base")
        self.unk_token = kwargs.get('unk_token', '[UNK]')
        if self.preprocess == 'bert':
            self.preprocess_method = bertPreprocessing().bertized_sentence
        elif self.preprocess == 'base':
            self.preprocess_method = normalizeString
        elif self.preprocess == 'base2':
            self.preprocess_method = RegexPreprocessing2(unk_token=self.unk_token)
        elif self.preprocess == 'spacy':
            self.preprocess_method = SpacyPreprocessing().lemmatized_sentence
        else:
            raise ValueError(
                "{} is not an appropriate preprocessing method.".format(self.preprocess))
        # self.voc = Voc(name, self.prep_diags, unk_token=self.unk_token)
        # if kwargs.get('fake_q', False):
        #     self.fake_q_text = kwargs.get(
        #         'fake_q_text', "Je n'ai pas compris, merci de reformuler la question")
        #     self.fake_q_text = self.preprocess_method(
        #         self.fake_q_text) if self.process_output else self.fake_q_text
        #     self.insert_fake(p=kwargs.get('fake_q_p', 0.2),
        #                      min_len=kwargs.get('fake_q_min_len', 4),
        #                      text=self.fake_q_text)

    def prepare_data(self):
        input_set = [l[0] for d in self.prep_diags for l in d]
        input_set = list(set(input_set))
        if self.process_output:
            output_set = [l[1] for d in self.prep_diags for l in d]
            output_set = list(set(output_set))
        self.questions = {r: self.preprocess_method(r) for r in input_set}
        if self.process_output:
            reponses = {r: self.preprocess_method(r) for r in output_set}
        for diag in self.prep_diags:
            for line in diag:
                line[0] = self.questions[line[0]]
                if self.process_output:
                    line[1] = reponses[line[1]]

    def import_diags(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            diags = f.read().split('\n\n')
            diags = [[l.split('\t') for l in d.split('\n')] for d in diags]
            table = {line[2][:-self.mem_len]: line[1]
                     for d in diags for line in d}
        return diags, table

    def save_diags(self, path, to_zip=False):
        self.prep_diags = '\n\n'.join(
            ['\n'.join(['\t'.join(l) for l in d]) for d in self.prep_diags])
        if to_zip:
            with ZipFile(path.replace('.tsv', '.zip'), 'w') as zip_f:
                zip_f.writestr(path, self.prep_diags)
        else:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(self.prep_diags)

    def insert_fake(self, p=0.2, min_len=4, text="Je n'ai pas compris, merci de reformuler la question"):
        code_fake_q = [0 for _ in range(
            0, len(self.prep_diags[0][0][2]) - self.mem_len)]
        code_fake_q = ''.join([str(s) for s in code_fake_q])
        self.table[code_fake_q] = text
        for diag in self.prep_diags:
            k = 1
            diag_copy = deepcopy(diag)
            for i, line in enumerate(diag_copy):
                if random() < p:
                    # random_pair = self.randint(0, len(res)-1)
                    # sent = res[random_pair][0].split()
                    # while len(sent) < min_len:
                    #     random_pair = self.randint(0, len(res)-1)
                    #     sent = res[random_pair][0].split()
                    #     k += 1
                    #     if k > 9:
                    #         return res

                    len_sent = randint(min_len, 20)
                    words = [self.voc.index2word[randint(
                        4, self.voc.num_words-1)] for _ in range(len_sent)]
                    rand_unk_i = randint(0, len(words)-1)
                    words.insert(rand_unk_i, self.voc.unk_token)
                    diag.insert(i + k, [' '.join(words), text,
                                        code_fake_q + line[2][-self.mem_len:]])
                    k += 1


if __name__ == "__main__":
    path_test = '../data/chatbot_diags_Test_Baseline.tsv'
    path_test2 = '../data/chatbot_diags_Test_Baseline2.tsv'
    diags_test = PreprocessingDialogs(
        'test', path_test, mem_len=6, preprocess='base', fake_q=True)
    diags_test.save_diags(path_test2)
