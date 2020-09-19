from typing import List

import nltk as nltk


class UnigramTokeniser:
    """
    Base class for tokenising
    """

    def __init__(self):
        self._tokeniser = nltk.NLTKWordTokenizer()

    def tokenise(self, text: str) -> List[str]:
        return self._tokeniser.tokenize(text)
