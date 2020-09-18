class OverlapDetector:

    def __init__(self, similarity_comparer):
        self._similarity_comparer = similarity_comparer

    def compare(self, source_df, target_df, columns=None):
        columns = columns or source_df.columns

        result = {}

        for c in columns:
            result[c] = {}
            result[c]["score"], result[c]["details"] = self._compare_rows_text(source_df[c].tolist(),
                                                                               target_df[c].tolist())

        return result

    def _compare_rows_text(self, src_rows, target_rows):
        result_score = []
        result_detailed_match = []

        for s_row in src_rows:

            best_score = 0
            best_match = None
            for t_row in target_rows:
                score = self._similarity_comparer(s_row, t_row)
                if score > best_score:
                    best_score = score
                    best_match = t_row

            result_score.append(best_score)
            result_detailed_match.append((s_row, best_match))

        return result_score, result_detailed_match
