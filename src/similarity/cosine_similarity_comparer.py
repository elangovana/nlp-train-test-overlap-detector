from typing import List

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from similarity.base_similarity_comparer import BaseSimilarityComparer


class CosineSimilarityComparer(BaseSimilarityComparer):

    def __init__(self, tokeniser):
        self.tokeniser = tokeniser
        self._count_vectoriser = CountVectorizer(tokenizer=self.tokeniser.tokenise, stop_words='english',
                                                 token_pattern=None)

    def __call__(self, source: List[str], target: List[str]) -> List[float]:
        vectoriser = self._count_vectoriser.fit(source + target)
        src_vector = vectoriser.transform(source)
        target_vector = vectoriser.transform(target)

        similarity_score = cosine_similarity(src_vector, Y=target_vector, dense_output=True)

        return similarity_score * 100
