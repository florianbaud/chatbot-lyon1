class Questions():

    def __init__(self):

        self.questions = ['Parcours', 'France', 'Lyon1']
        self.memory = [True for _ in self.questions]
        self.questions_templates = {
            "candidature": [False, True, True],
            "connaissancesPrealables": [True, False, False],
            "placesDisponibles": [True, True, True],
            "pageWeb": [True, False, False],
            "responsableMaster": [True, False, False]
        }

    def memory_update(self, question):
        self.memory[self.questions.index(question)] = False
        if question == 'Lyon1':
            self.memory[1] = False
