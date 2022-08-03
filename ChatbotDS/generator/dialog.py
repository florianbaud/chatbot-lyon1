from ChatbotDS.utils.student import Student
from ChatbotDS.utils.questions import Questions
from random import choice
from numpy import array
from re import sub


class Dialog():

    def __init__(self, templates, enchainement):
        self.templates = templates
        self.enchainement = enchainement
        self.student = Student()
        self.questions = Questions()
        self.memory = [False for a in self.student.attributes.values()
                       for i in a]
        self.code_template = [0 for _ in self.templates.enchainements]
        self.dialog = self.generate_diag()

    def random_pair(self, ench):
        template = self.templates.templates[ench]
        pair = choice(template)
        pair = choice(pair)
        return pair

    def memory_update(self, question):
        attribute = self.student.state2[question]
        self.memory[self.student.flat_attributes.index(attribute)] = True
        if attribute == self.student.attributes['France'][0]:
            not_lyon = self.student.attributes['Lyon1'][0]
            self.memory[self.student.flat_attributes.index(not_lyon)] = True
        elif attribute == self.student.attributes['Lyon1'][1]:
            france = self.student.attributes['France'][1]
            self.memory[self.student.flat_attributes.index(france)] = True

    def memory_to_string(self):
        memory = [str(int(m)) for m in self.memory]
        return ''.join(memory)

    def enchainement_to_string(self, enchainement):
        i = self.templates.enchainements.index(enchainement)
        res = self.code_template.copy()
        res[i] = 1
        return ''.join([str(r) for r in res])

    def questions_state(self, ench):
        student_state = array(self.student.student_q)
        question_state = array(self.questions.questions_templates[ench])
        memory_state = array(self.questions.memory)
        questions_state = student_state*question_state*memory_state
        questions_res = student_state*question_state
        questions_state = array(self.questions.questions)[
            questions_state].tolist()
        questions_res = array(self.questions.questions)[
            questions_res].tolist()
        return questions_state, questions_res

    def questions_response(self, ench, questions_res):
        key = ''.join([self.student.state2.get(k) for k in questions_res])
        response = self.templates.bot_responses[ench][key]
        return response

    def generate_question(self, question, enchainement):
        bot_question = self.templates.bot_questions[question]
        user_responses = self.templates.user_responses[question]
        i = int(self.student.index_res[question])
        user_response = choice(user_responses[i])
        self.questions.memory_update(question)
        memory = self.memory_to_string() + self.enchainement_to_string(enchainement)
        self.memory_update(question)
        return bot_question, memory, user_response

    def generate_diag(self):
        dialog = []
        for ench in self.enchainement:
            if ench == 'UserStatements':  # Condition User Statements
                pair = self.user_statements().copy()
            else:
                pair = self.random_pair(ench).copy()
            if pair[1] == "Bot_Question":  # Condition Bot Question
                questions_state, questions_res = self.questions_state(ench)
                questions_res = self.questions_response(ench, questions_res)
                questions = [
                    r for q in questions_state for r in self.generate_question(q, ench)]
                pair = [pair[0]] + questions + [questions_res] + \
                    [self.memory_to_string() + self.enchainement_to_string(ench)]
            else:
                pair.append(self.memory_to_string() +
                            self.enchainement_to_string(ench))
            dialog += pair
        dialog = self.token_replace(dialog)
        return dialog

    def token_replace(self, dialog):
        for i in range(0, len(dialog), 3):
            for token in self.student.value:
                dialog[i] = sub('Token_' + token.lower(),
                                self.student.value[token], dialog[i])
                dialog[i+1] = sub('Token_' + token.lower(),
                                  self.student.value[token], dialog[i+1])
        return dialog

    def user_statements(self):
        questions = list(self.templates.user_statements.keys())
        question = choice(questions)
        i = int(self.student.index_res[question])
        pair = choice(self.templates.user_statements[question][i])
        self.memory_update(question)
        self.questions.memory_update(question)
        return pair

    @staticmethod
    def print_diag(dialog):
        for i in range(0, len(dialog), 3):
            print(dialog[i], dialog[i+1], dialog[i+2], sep='\t')
