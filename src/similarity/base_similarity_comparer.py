from typing import List


class BaseSimilarityComparer:

    def __call__(self, source: List[str], target: List[str]) -> List[float]:
        raise NotImplementedError
