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
        score_summary = {}
        result_detail = {}
        for k, t in self._tokenisers.items():
            comparer = OverlapDetector(CosineSimilarityComparer(t))
            result = comparer.compare(test, train, columns=["passage"])
            scores = result["passage"]["score"]
            detail = result["passage"]["details"]

            score_stats = {"min": np.min(scores), "max": np.max(scores), "std": np.std(scores), "mean": np.mean(scores),
                           "median": np.median(scores)}

            score_summary[k] = score_stats
            result_detail[k] = detail

        for k, v in score_summary.items():
            print(k)
            print(json.dumps(v, indent=1))
