from typing import List

import nltk as nltk


class TrigramTokeniser:
    """
    Base class for tokenising
    """

    def __init__(self):
        self._tokeniser = nltk.NLTKWordTokenizer()

    def tokenise(self, text: str) -> List[str]:
        tokens = self._tokeniser.tokenize(text)
        trigrams = nltk.trigrams(tokens)
        result = []
        for trigram in trigrams:
            result.append(" ".join(trigram))

        return result
