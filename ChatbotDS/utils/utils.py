import os
import unicodedata
import re
import json
from ChatbotDS.utils.student import Student


def import_all(path):
    data = {}
    Bot_Q = os.listdir(path + 'BotQ')
    Bot_Rep = os.listdir(path + 'BotRep')
    files = os.listdir(path)
    user_statements = os.listdir(path + 'UserStatements')
    for f in user_statements:
        if '.tsv' in f:
            data['UserStatements' +
                 f] = import_template(path + 'UserStatements/' + f)
    for f in files:
        if '.tsv' in f:
            data[f] = import_template(path + f)
    for f in Bot_Q:
        if '.tsv' in f:
            data['BotQ' + f] = import_template(path + 'BotQ/' + f)
    for f in Bot_Rep:
        if '.tsv' in f:
            data['BotRep' + f] = import_template(path + 'BotRep/' + f)
    return data


def import_template(path):
    with open(path, "r", encoding="utf-8") as t:
        template = t.read().split("\n\n")
    return template


def import_template2(path):
    with open(path, "r", encoding="utf-8") as t:
        template = [[p.split('\t') for p in l.split('\n')]
                    for l in t.read().split("\n\n")]
    return template


def import_templates(path):
    templates = [f for f in os.listdir(path) if '.tsv' in f]
    templates_dict = {t.split('.')[0]: import_template2(
        path + t) for t in templates}
    return templates_dict


def import_bot_questions(path):
    bot_questions = [f for f in os.listdir(path) if '.tsv' in f]
    bot_questions_dict = {
        'BotQ' + q.split('.')[0]: import_template2(path + q) for q in bot_questions}
    return bot_questions_dict


def import_bot_responses(path):
    bot_responses = [f for f in os.listdir(path) if '.tsv' in f]
    bot_responses_dict = {
        'BotRep' + r.split('.')[0]: import_template2(path + r) for r in bot_responses}
    return bot_responses_dict


def import_user_responses(path):
    user_reponses = [f for f in os.listdir(path) if '.tsv' in f]
    user_reponses_dict = {
        'UserRep' + r.split('.')[0]: import_template2(path + r) for r in user_reponses}
    return user_reponses_dict


def unicodeToAscii(s):
    return ''.join([c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn'])


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"[^a-zA-Z0-9?&\%\_]", r" ", s)
    return s


def import_replace_variable(path):
    with open(path, 'r', encoding='utf-8') as var_file:
        variable = json.load(var_file)
    return variable


def get_responses(templates_path):
    data = import_all(templates_path)
    responses = set()
    for d in data:
        if 'BotQ' in d:
            responses.add(data[d][0])
        elif 'BotRep' in d:
            for line in data[d]:
                pair = line.split('\t')
                responses.add(pair[1] + pair[0])
        # elif 'UserStatements' in d:
        #     for diag in data[d]:
        #         for line in diag.split('\n'):
        #             pair = line.split('\t')
        else:
            for diag in data[d]:
                for line in diag.split('\n'):
                    pair = line.split('\t')
                    if 'Token_' in pair[1]:
                        for parcours in Student().attributes['Parcours']:
                            student = Student(Parcours=parcours)
                            for k in student.value:
                                token_replace = pair[1].replace(
                                    'Token_' + k.lower(), student.value[k])
                                if 'Token_' in token_replace:
                                    continue
                                responses.add(token_replace)
                    elif 'Bot_Question' in pair[1]:
                        continue
                    else:
                        responses.add(pair[1])
    return responses
