import json

import numpy as np

from similarity.cosine_similarity_comparer import CosineSimilarityComparer
from similarity.overlap_detector import OverlapDetector


class SimilarityEvaluator:

    def __init__(self, ngrams_dict=None):
        """

        :param ngrams_dict: Name and ngram. Example
                            {"Unigram": 1,
                            "Bigram": 2,
                            "Trigram": 3}
        """
        self._ngrams = ngrams_dict or {"Unigram": 1,
                            "Bigram": 2,
                            "Trigram": 3}

    def run(self, test, train, column):
        result_score = {}
        result_detail = {}
        for k, t in self._ngrams.items():
            comparer = OverlapDetector(CosineSimilarityComparer(t))
            comparison_result = comparer.compare(test, train, columns=[column])
            scores = comparison_result[column]["score"]
            detail = comparison_result[column]["details"]

            result_score[k] = scores
            result_detail[k] = detail

        return result_score, result_detail

    def print_summary(self, result_score, result_detail):
        for k, scores in result_score.items():
            score_stats = {"type": k, "min": np.min(scores), "max": np.max(scores), "std": np.std(scores),
                           "mean": np.mean(scores),
                           "median": np.median(scores)}
            print(json.dumps(score_stats, indent=1))

        for k, v in result_detail.items():
            top_k = np.argsort(result_score[k])[-5:]
            print(k)
            print(json.dumps(np.array(v)[top_k].tolist(), indent=1))
