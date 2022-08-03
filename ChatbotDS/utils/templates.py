from random import sample
from copy import deepcopy
import os


class Templates():

    def __init__(self, path):

        self.set_type = ['Full', 'Train', 'Test']
        self.enchainements = ['UserStatements']

        self.path = path
        self.templates = self.import_templates()
        self.user_responses = self.import_user_responses()
        self.bot_questions = self.import_bot_questions()
        self.bot_responses = self.import_bot_responses()
        self.user_statements = self.import_user_statements()

        self.templates_dict = self.create_templates_sets()
        self.user_responses_dict = self.create_user_responses_sets()
        self.user_statements_dict = self.create_user_statements_sets()

        self.enchainements = ['UserStatements'] + list(self.templates.keys())

    def import_template(self, path):
        with open(path, "r", encoding="utf-8") as t:
            template = [[p.split('\t') for p in l.split('\n')]
                        for l in t.read().split("\n\n")]
        return template

    def import_templates(self):
        path = self.path
        templates = [f for f in os.listdir(path) if '.tsv' in f]
        templates_dict = {t.split('.')[0]: self.import_template(
            path + t) for t in templates}
        return templates_dict

    def import_bot_questions(self):
        path = self.path + 'BotQ/'
        bot_questions = [f for f in os.listdir(path) if '.tsv' in f]
        bot_questions_dict = {q.split('.')[0]: self.import_template(
            path + q)[0][0][0] for q in bot_questions}
        return bot_questions_dict

    def import_bot_responses(self):
        path = self.path + 'BotRep/'
        bot_responses = [f for f in os.listdir(path) if '.tsv' in f]
        bot_responses_dict = {r.split('.')[0]: self.import_template(
            path + r) for r in bot_responses}
        bot_responses_dict = {
            d: {r[0][0]: r[0][1] for r in bot_responses_dict[d]} for d in bot_responses_dict}
        return bot_responses_dict

    def import_user_responses(self):
        path = self.path + 'UserRep/'
        user_reponses = [f for f in os.listdir(path) if '.tsv' in f]
        user_reponses_dict = {r.split('.')[0]: self.import_template(
            path + r) for r in user_reponses}
        user_reponses_dict = {r: [[u[0] for u in l]
                                  for l in user_reponses_dict[r]] for r in user_reponses_dict}
        return user_reponses_dict

    def import_user_statements(self):
        path = self.path + 'UserStatements/'
        user_statements = [f for f in os.listdir(path) if '.tsv' in f]
        user_statements_dict = {r.split('.')[0]: self.import_template(
            path + r) for r in user_statements}
        return user_statements_dict

    def divide_data(self, data, k=1):
        if len(data) > 1:
            data = set(data)
            test = set(sample(data, k))
            train = data - test
            return list(train), list(test)
        elif len(data) == 1:
            return data, None
        else:
            return None, None

    def create_templates_sets(self):
        train_dict, test_dict = {}, {}
        for t in self.templates:
            train, test = [], []
            for data in self.templates[t]:
                div = self.divide_data(['\t'.join(d) for d in data])
                train.append([i.split('\t') for i in div[0]])
                if div[1] != None:
                    test.append([i.split('\t') for i in div[1]])
            train_dict[t], test_dict[t] = train, test
        res = {'Full': deepcopy(self.templates),
               'Train': train_dict, 'Test': test_dict}
        return res

    def create_user_responses_sets(self):
        train_dict, test_dict = {}, {}
        for t in self.user_responses:
            train, test = [], []
            for data in self.user_responses[t]:
                div = self.divide_data(data)
                train.append(div[0])
                test.append(div[1])
            train_dict[t], test_dict[t] = train, test
        res = {'Full': deepcopy(self.user_responses),
               'Train': train_dict, 'Test': test_dict}
        return res

    def create_user_statements_sets(self):
        train_dict, test_dict = {}, {}
        for t in self.user_statements:
            train, test = [], []
            for data in self.user_statements[t]:
                div = self.divide_data(['\t'.join(d) for d in data])
                train.append([i.split('\t') for i in div[0]])
                if div[1] != None:
                    test.append([i.split('\t') for i in div[1]])
            train_dict[t], test_dict[t] = train, test
        res = {'Full': deepcopy(self.user_statements),
               'Train': train_dict, 'Test': test_dict}
        return res

    def select(self, mode):
        self.templates = self.templates_dict[mode]
        self.user_responses = self.user_responses_dict[mode]
        self.user_statements = self.user_statements_dict[mode]

    def get_responses(self):
        self.responses = [p[1]
                          for t in self.templates for r in self.templates[t] for p in r]
        self.responses.extend(
            [v for r in self.bot_responses for v in self.bot_responses[r].values() for i in v])
        return set(self.responses)
