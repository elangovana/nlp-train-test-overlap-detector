import json

import numpy as np

from similarity.cosine_similarity_comparer import CosineSimilarityComparer
from similarity.overlap_detector import OverlapDetector


class SimilarityEvaluator:

    def __init__(self):
        self._tokenisers = {"Unigram": 1,
                            "Bigram": 2,
                            "Trigram": 3}

    def run(self, test, train, column):
        result_score = {}
        result_detail = {}
        for k, t in self._tokenisers.items():
            comparer = OverlapDetector(CosineSimilarityComparer(t))
            comparison_result = comparer.compare(test, train, columns=[column])
            scores = comparison_result[column]["score"]
            detail = comparison_result[column]["details"]

            result_score[k] = scores
            result_detail[k] = detail

        for k, v in result_score.items():
            print(k)
            scores = v
            score_stats = {"min": np.min(scores), "max": np.max(scores), "std": np.std(scores), "mean": np.mean(scores),
                           "median": np.median(scores)}
            print(json.dumps(score_stats, indent=1))

        for k, v in result_detail.items():
            top_k = np.argsort(result_score[k])[-5:]
            # index = np.argmax(result_score[k])
            print(k)
            print(json.dumps(np.array(v)[top_k].tolist(), indent=1))

        return result_score, result_detail
