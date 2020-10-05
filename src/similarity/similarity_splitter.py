from similarity.cosine_similarity_comparer import CosineSimilarityComparer
from similarity.overlap_detector import OverlapDetector


class SimilaritySplitter:

    def __init__(self, column, ngram=1, comparer=None):
        self._column = column
        self._ngram = ngram
        self._comparer = comparer or OverlapDetector(CosineSimilarityComparer(self._ngram))

    def get(self, test_df, train_df, similarity_threshold_min=-1, similarity_theshold_max=1):
        comparison_result = self._comparer.compare(test_df, train_df, columns=[self._column])
        scores = comparison_result[self._column]["score"]

        select_flag = [similarity_threshold_min <= i < similarity_theshold_max for i in scores]

        sim_df = test_df[select_flag]

        return sim_df
