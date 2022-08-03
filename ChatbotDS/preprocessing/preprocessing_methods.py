from re import sub
from unicodedata import normalize, category
from spacy import load as spacy_load, info
from torch.hub import load
from torch import __version__ as torch_version


def unicodeToAscii(s: str) -> str:
    """Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427"""
    return ''.join(c for c in normalize('NFD', s) if category(c) != 'Mn')


def normalizeString(s: str) -> str:
    '''Remove rare symbols from a string'''
    s = unicodeToAscii(s.lower())
    s = sub(r"\(", r" ( ", s)
    s = sub(r"\)", r" ) ", s)
    s = sub(r"\.", r" . ", s)
    s = sub(r",", r" , ", s)
    s = sub(r"!", r" ! ", s)
    s = sub(r":", r" : ", s)
    s = sub(r"-", r" - ", s)
    s = sub(r"'", r" ' ", s)
    s = sub(r";", r" ; ", s)
    s = sub(r' +', r' ', s).strip()
    return s


class RegexPreprocessing2(object):

    def __init__(self, unk_token: str = "") -> None:
        self.unk_token = unk_token

    def __call__(self, text: str) -> str:
        s = text.replace('UNK_Token', self.unk_token)
        s = unicodeToAscii(s.lower())
        s = sub(r"[^a-zA-Z0-9?&\%\_]", r" ", s)
        s = sub(r' +', ' ', s).strip()
        return s

# def normalizeString2(s: str, unk_token: str = "") -> str:
#     s = s.replace('UNK_Token', unk_token)
#     s = unicodeToAscii(s.lower())
#     s = sub(r"[^a-zA-Z0-9?&\%\_]", r" ", s)
#     s = sub(r' +', ' ', s).strip()
#     return s


class bertPreprocessing():

    def __init__(self):
        self.tokenizer = load('huggingface/pytorch-pretrained-BERT', 'tokenizer',
                              'bert-base-uncased', force_reload=True, do_basic_tokenize=True, do_lower_case=True)

    def bertized_sentence(self, text):
        text = text.replace('UNK_Token', '[UNK]')
        text = self.tokenizer.tokenize(text)
        text = u' '.join(text)
        return text


class SpacyPreprocessing():

    def __init__(self, model_name='fr_core_news_md', exceptions=['data', 'scientist']):
        self.info = info()
        self.exceptions = exceptions
        self.model_name = model_name
        self.model = spacy_load(self.model_name)

    def lemmatized_sentence(self, text):
        text = self.model(text)
        text = [a for a in [t.lemma_.lower()
                            for t in text] if a not in self.exceptions]
        return u' '.join(text)
