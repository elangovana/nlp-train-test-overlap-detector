import argparse
import logging
import sys

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from base_data_loader import BaseDataLoader
from similarity_evaluator import SimilarityEvaluator


class AIMedDataLoaderUniqueDoc(BaseDataLoader):

    def __init__(self, label_field_name=None, docid_field_name=None):
        self._docid_field_name = docid_field_name or "docid"
        self._label_field_name = label_field_name or "isValid"

    def load(self, path):
        train, test = list(self._k_fold_unique_doc(path, self._label_field_name, self._docid_field_name, 10))[0]

        return train, test

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def _label_distribution(sel, df, label_field_name):
        label_counts_raw = df[label_field_name].value_counts()
        label_counts_percentage = label_counts_raw * 100 / sum(label_counts_raw.values)

        return label_counts_percentage

    def _k_fold_unique_doc(self, data_file, label_field_name, docid_field_name, n_splits=10):
        self._logger.info("Splitting such that the {} is unique across datasets".format(docid_field_name))
        kf = StratifiedKFold(n_splits=n_splits, random_state=777, shuffle=True)
        df = pd.read_json(data_file)
        unique_docids = df.docid.unique()
        # Do a approx so that the labels are somewhat stratified in the split
        approx_y = [df.query("{} == '{}'".format(docid_field_name, p))[label_field_name].iloc[0] for p in unique_docids]
        for train_index, test_index in kf.split(unique_docids, approx_y):
            train_doc, test_doc = unique_docids[train_index], unique_docids[test_index]
            train = df[df[docid_field_name].isin(train_doc)]
            val = df[df[docid_field_name].isin(test_doc)]

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
    train, test = AIMedDataLoaderUniqueDoc().load(trainfile)
    SimilarityEvaluator().run(test, train)


if "__main__" == __name__:
    args = _parse_args()
    run(args.trainfile)
