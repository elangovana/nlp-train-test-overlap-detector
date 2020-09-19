import logging
import os

import numpy as np


class OverlapDetector:

    def __init__(self, similarity_comparer, num_workers=None):
        self._num_workers = num_workers or max(1, os.cpu_count() - 1)
        self._similarity_comparer = similarity_comparer

    def compare(self, source_df, target_df, columns=None):
        columns = columns or source_df.columns

        result = {}

        for c in columns:
            result[c] = {}
            result[c]["score"], result[c]["details"] = self._compare_rows_text(source_df[c].tolist(),
                                                                               target_df[c].tolist())

        return result

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def _compare_rows_text(self, src_rows, target_rows):
        similarity_score = self._similarity_comparer(src_rows, target_rows)

        result_score = np.max(similarity_score, axis=1).tolist()
        similarity_scores_index = np.argmax(similarity_score, axis=1)
        result_detailed_match = [(src_rows[i], target_rows[mi]) for i, mi in enumerate(similarity_scores_index)]

        return result_score, result_detailed_match
