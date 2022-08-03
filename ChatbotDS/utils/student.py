import random
from functools import reduce


class Student():

    def __init__(self, Parcours=None):

        self.attributes = {
            "Parcours": ["Mathématique", "Informatique"],
            "France": ["hors_France", "en_France"],
            "Lyon1": ["hors_Lyon1", "de_Lyon1"]
        }

        self.flat_attributes = [i for a in self.attributes.values() for i in a]

        math = self.attributes["Parcours"][0]
        info = self.attributes["Parcours"][1]

        Parcours = random.choice(
            self.attributes["Parcours"]) if Parcours is None else Parcours
        France = random.choice(self.attributes["France"])
        Lyon1 = random.choice(
            self.attributes["Lyon1"]) if France == "en_France" else "hors_Lyon1"

        self.state2 = {
            "Parcours": Parcours,
            "France": France,
            "Lyon1": Lyon1
        }

        self.index_res = {a: self.attributes[a].index(
            self.state2[a]) for a in self.attributes}

        self.student_q = [True, True, True] if France == "en_France" else [
            True, True, False]

        self.code = reduce(
            lambda a, b: a+b, [self.attributes[a] for a in self.attributes])

        if France == 'en_France':
            Origine = 'en_France'
        if Lyon1 == 'de_Lyon1':
            Origine = 'de_Lyon1'
        if France == 'hors_France':
            Origine = 'hors_France'

        self.value = {
            "Parcours": Parcours,
            "Op_Parcours": info if self.state2['Parcours'] == "Mathématique" else math,
            "Origine": Origine
        }
