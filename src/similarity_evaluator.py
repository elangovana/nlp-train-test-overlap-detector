import json

import numpy as np

from bigram_tokeniser import BigramTokeniser
from cosine_similarity_comparer import CosineSimilarityComparer
from overlap_detector import OverlapDetector
from trigram_tokeniser import TrigramTokeniser
from unigram_tokeniser import UnigramTokeniser


class SimilarityEvaluator:

    def __init__(self):
        self._tokenisers = {"Unigram":
                                UnigramTokeniser(),
                            "Bigram": BigramTokeniser(),
                            "Trigram": TrigramTokeniser()}

    def run(self, test, train):
        for k, t in self._tokenisers.items():
            comparer = OverlapDetector(CosineSimilarityComparer(t))
            result = comparer.compare(test, train, columns=["passage"])
            scores = result["passage"]["score"]
            detail = result["passage"]["details"]

            score_stats = {"min": np.min(scores), "max": np.max(scores), "std": np.std(scores), "mean": np.mean(scores),
                           "median": np.median(scores)}

            print(f"---{k}---")
            print(score_stats)
            print(json.dumps(detail[0:3], indent=1))
