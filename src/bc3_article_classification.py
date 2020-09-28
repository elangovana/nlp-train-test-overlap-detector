import argparse
import logging
import sys

import pandas as pd

from similarity.similarity_evaluator import SimilarityEvaluator


class BC3ArticleClassification:

    def __init__(self):
        pass

    def load(self, file):
        data = []
        sep = "\t"
        with open(file, "r") as f:
            for l in f.readlines():
                line_parts = l.split(sep)
                title_text = line_parts[4]
                abstract_text = line_parts[5].rstrip("\n")
                data.append({"title": title_text, "abstract": abstract_text})
        df = pd.DataFrame(data)
        return df


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainfile",
                        help="The input train TSV file ", required=True)
    parser.add_argument("--testfile",
                        help="The input test TSV file ", required=True)

    parser.add_argument("--log-level", help="Log level", default="INFO", choices={"INFO", "WARN", "DEBUG", "ERROR"})

    args = parser.parse_args()
    print(args.__dict__)
    # Set up logging
    logging.basicConfig(level=logging.getLevelName(args.log_level), handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    return args


def run(trainfile, testfile):
    train = BC3ArticleClassification().load(trainfile)
    test = BC3ArticleClassification().load(testfile)

    result_score, result_detail = SimilarityEvaluator().run(test, train, column="abstract")


if "__main__" == __name__:
    args = _parse_args()
    run(args.trainfile, args.testfile)
