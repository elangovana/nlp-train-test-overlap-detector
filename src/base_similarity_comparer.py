class BaseSimilarityComparer:

    def __call__(self, source: str, target: str) -> float:
        raise NotImplementedError
