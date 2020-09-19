from typing import List

import nltk as nltk


class BigramTokeniser:
    """
    Base class for tokenising
    """

    def __init__(self):
        self._tokeniser = nltk.NLTKWordTokenizer()

    def tokenise(self, text: str) -> List[str]:
        tokens = self._tokeniser.tokenize(text)
        bigrams = nltk.bigrams(tokens)
        result = []
        for bigram in bigrams:
            result.append(" ".join(bigram))

        return result
