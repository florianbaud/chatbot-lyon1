from copy import deepcopy
from ChatbotDS.utils.templates import Templates


class CodeResponse():

    def __init__(self, data_path, templates_path):
        self.path = data_path
        self.templates = Templates(templates_path)
        self.data = self.import_data()

    def import_data(self):
        with open(self.path, 'r', encoding='utf-8') as f:
            data = f.read().split('\n\n')
        data = [[l.split('\t') for l in d.split('\n')] for d in data]
        return data

    def baseline(self):
        response_set = list(set([l[1]+l[2][6:] for d in self.data for l in d]))
        code_baseline = [0 for _ in range(len(response_set))]
        self.response_table = {r: self.modify_list(
            code_baseline, response_set.index(r)) for r in response_set}
        self.code = [[l[:-1] + [l[-1]+self.response_table[l[1]+l[2][6:]]]
                      for l in d] for d in self.data]
        return self.code

    def modify_list(self, l, i):
        l = l.copy()
        l[i] = 1
        return ''.join([str(o) for o in l])

    def save(self, path):
        dialog = '\n\n'.join(['\n'.join(['\t'.join(l)
                                         for l in d]) for d in self.code])
        with open(path, 'w', encoding='utf-8') as f:
            f.write(dialog)

    def get_responses(self):
        raise NotImplementedError