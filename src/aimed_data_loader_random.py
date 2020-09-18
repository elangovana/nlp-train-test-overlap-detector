import argparse
import json
import logging
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from base_data_loader import BaseDataLoader
from cosine_similarity_comparer import CosineSimilarityComparer
from overlap_detector import OverlapDetector
from unigram_tokeniser import UnigramTokeniser


class AIMedDataLoaderRandom(BaseDataLoader):

    def __init__(self, label_field_name=None):
        self._label_field_name = label_field_name or "isValid"

    def load(self, path):
        train, test = list(self._k_fold_random(path, self._label_field_name, 10))[0]

        return train, test

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def _k_fold_random(self, data_file, label_field_name, n_splits=10):
        kf = StratifiedKFold(n_splits=n_splits, random_state=777, shuffle=True)
        df = pd.read_json(data_file)

        for train_index, test_index in kf.split(df, df[label_field_name]):
            train, val = df.iloc[train_index], df.iloc[test_index]

            yield (train, val)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainfile",
                        help="The input train file ", required=True)
    parser.add_argument("--log-level", help="Log level", default="INFO", choices={"INFO", "WARN", "DEBUG", "ERROR"})
    args = parser.parse_args()
    print(args.__dict__)
    # Set up logging
    logging.basicConfig(level=logging.getLevelName(args.log_level), handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    return args


def run(trainfile):
    train, test = AIMedDataLoaderRandom().load(trainfile)

    sut = OverlapDetector(CosineSimilarityComparer(UnigramTokeniser()))
    result = sut.compare(test, train, columns=["passage"])
    scores = result["passage"]["score"]

    score_stats = {"min": np.min(scores), "max": np.max(scores), "std": np.std(scores), "mean": np.mean(scores),
                   "median": np.median(scores)}
    print(json.dumps(result, indent=1))

    print(score_stats)


if "__main__" == __name__:
    args = _parse_args()
    run(args.trainfile)