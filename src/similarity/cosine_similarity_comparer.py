from typing import List

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from similarity.base_similarity_comparer import BaseSimilarityComparer


class CosineSimilarityComparer(BaseSimilarityComparer):

    def __init__(self, ngram, stop_words=None):
        self._count_vectoriser = CountVectorizer(stop_words=stop_words, ngram_range=(ngram, ngram))

    def __call__(self, source: List[str], target: List[str]) -> List[float]:
        try:
            vectoriser = self._count_vectoriser.fit(source + target)
        # Only stop words
        except ValueError as e:
            print(e)
            return np.zeros((len(source), len(target))).tolist()

        src_vector = vectoriser.transform(source)
        target_vector = vectoriser.transform(target)

        similarity_score = cosine_similarity(src_vector, Y=target_vector, dense_output=True)

        return similarity_score * 100
