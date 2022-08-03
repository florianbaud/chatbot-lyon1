from ChatbotDS.utils.utils import import_template, normalizeString
from copy import deepcopy
import spacy


class Keywords():

    def __init__(self, enchainements, treshold, templates_path):

        self.enchainements = enchainements
        self.treshold = treshold
        self.templates_path = templates_path
        self.nlp_fr = spacy.load('fr_core_news_md')
        self.keywords_init()
        self.keywords_count()

    def keywords_init(self):
        enchainements = self.enchainements
        keywords_dict = {}
        keywords_global = []
        for ench in enchainements:
            keywords_diags = []
            diags = import_template(self.templates_path + ench + '.tsv')
            for diag in diags:
                keywords = []
                for line in diag.split('\n'):
                    doc = self.nlp_fr(line.split('\t')[0])
                    keys = self.noun_chunk_criteria(doc)
                    keywords += keys
                    keywords_global += keys
                keywords_diags.append(set(keywords))
            keywords_dict[ench] = keywords_diags
        self.keywords_dict = keywords_dict
        self.keywords_global = keywords_global
        return 1

    def keywords_count(self):
        self.count_dict = {}
        self.total_keywords = len(self.keywords_global)
        for key in self.keywords_dict:
            for keywords in self.keywords_dict[key]:
                for keyword in keywords:
                    if keyword not in self.count_dict:
                        self.count_dict[keyword] = 1
                    else:
                        self.count_dict[keyword] += 1
        return 1

    def keywords_filter(self):
        self.keywords_filtered = deepcopy(self.keywords_dict)
        k = 0
        for template in self.keywords_filtered:
            new_keywords = [set([key for key in keys if self.count_dict[key] > self.treshold])
                            for keys in self.keywords_filtered[template]]
            for new_keys in new_keywords:
                if new_keywords.count(new_keys) > 1:
                    new_keys.add(str(k))
                    k += 1
            self.keywords_filtered[template] = new_keywords

        self.keywords_filtered_global = set()
        for template in self.keywords_filtered:
            for s in self.keywords_filtered[template]:
                self.keywords_filtered_global = self.keywords_filtered_global.union(
                    s)
        self.keywords_filtered_global = list(self.keywords_filtered_global)
        return 1

    def pos_criteria(self, doc):
        res = [t.lemma_.lower() for t in doc if not t.is_stop and (
            t.pos_ == 'NOUN' or t.pos_ == 'PROPN' or t.pos_ == 'VERB')]
        return res

    def noun_chunk_criteria(self, doc):
        exclusion = ['token_parcours', 'token_op_parcours']
        res = [t.root.text.lower() for t in doc.noun_chunks]
        res = [normalizeString(self.nlp_fr(word)[0].lemma_)
               for word in res if word not in exclusion]
        return res

    def noun_chunk_criteria2(self, doc):
        raise NotImplementedError
        # exclusion = ['token_parcours', 'token_op_parcours']
        # res = [t.root.text.lower() for t in doc.noun_chunks]
        # res =
        # return res
