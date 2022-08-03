from random import random


class Voc:

    def __init__(self, diags, data_memory_size, unk_token='[UNK]', unk_text="Je n'ai pas compris, merci de reformuler la question."):

        self.unk_token = unk_token
        self.unk_text = unk_text
        self.data_memory_size = data_memory_size
        self.trimmed = False
        self.pad_index = 0  # Used for padding short sentences
        self.sos_index = 1  # Start-of-sentence token
        self.eos_index = 2  # End-of-sentence token
        self.unk_index = 3  # Unknown word token
        self.index2word = {self.pad_index: "PAD", self.sos_index: "SOS",
                           self.eos_index: "EOS", self.unk_index: self.unk_token}
        self.word2index = {}
        self.word2count = {}
        self.num_words = 4  # Count SOS, EOS, PAD, UNK
        self.set_table(diags)
        self.pop_voc(diags)

    def pop_voc(self, diags):
        for diag in diags:
            for line in diag:
                try:
                    self.add_sentence(line[0])
                except:
                    print(line)

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)
            if word == '':  # AA
                print("Empty characater in:")
                print(sentence.split(' '))

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def word_to_index(self, word, v=False, p: float = 0):
        if random() < p:
            return self.unk_index
        try:
            i = self.word2index[word]
        except KeyError:
            if v:
                print(f'Unknown Word : "{word}"')
            i = self.unk_index
        return i

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words {} / {} = {:.4f}'.format(len(keep_words),
                                                   len(self.word2index), len(keep_words) / len(self.word2index)))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {self.pad_index: "PAD", self.sos_index: "SOS",
                           self.eos_index: "EOS", self.unk_index: self.unk_token}
        self.num_words = 4  # Count default tokens

        for word in keep_words:
            self.add_word(word)

    def set_table(self, diags):
        self.table = {r[2][self.data_memory_size:]: r[1]
                      for r in [l for d in diags for l in d]}
        self.table['0'*len(list(self.table.keys())[0])] = self.unk_text
