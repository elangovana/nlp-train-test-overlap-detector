class OverlapDetector:

    def __init__(self, similarity_comparer):
        self._similarity_comparer = similarity_comparer

    def compare(self, source_df, target_df, columns=None):
        columns = columns or source_df.columns

        result = {}

        for c in columns:
            result[c] = self._compare_rows_text(source_df[c].tolist(), target_df[c].tolist())

        return result

    def _compare_rows_text(self, src_rows, target_rows):
        result = []

        for s_row in src_rows:

            best_score = 0
            best_match = None
            for t_row in target_rows:
                score = self._similarity_comparer(s_row, t_row)
                if score > best_score:
                    best_score = score
                    best_match = t_row

            result.append([best_score, s_row, best_match])

        return result
