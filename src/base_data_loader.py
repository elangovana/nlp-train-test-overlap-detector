from pandas import DataFrame


class BaseDataLoader:

    def load(self, path) -> (DataFrame, DataFrame):
        raise NotImplementedError
