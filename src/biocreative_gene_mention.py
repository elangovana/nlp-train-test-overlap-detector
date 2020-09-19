import argparse
import logging
import sys

import pandas as pd

from similarity_evaluator import SimilarityEvaluator


class BiocreativeGeneMention:

    def __init__(self):
        pass

    def load(self, file):
        data = []
        with open(file, "r") as f:
            for l in f.readlines():
                text = " ".join(l.split(" ")[1:])
                data.append(text)
        df = pd.DataFrame(data={"text": data})
        return df


class BiocreativeGeneAnnotation:

    def __init__(self):
        pass

    def load(self, file):
        data = []
        with open(file, "r") as f:
            for l in f.readlines():
                text = l.split("|")[2]
                data.append(text)
        df = pd.DataFrame(data={"text": data})
        return df


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainfile",
                        help="The input train file ", required=True)
    parser.add_argument("--testfile",
                        help="The input test file ", required=True)
    parser.add_argument("--type",
                        help="Specify the file as a text or annotation file", required=True,
                        choices={"text", "eval"})
    parser.add_argument("--log-level", help="Log level", default="INFO", choices={"INFO", "WARN", "DEBUG", "ERROR"})

    args = parser.parse_args()
    print(args.__dict__)
    # Set up logging
    logging.basicConfig(level=logging.getLevelName(args.log_level), handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    return args


def run(comparison_type, trainfile, testfile):
    loaders = {
        "text": BiocreativeGeneMention()
        , "eval": BiocreativeGeneAnnotation()
    }

    train = loaders[comparison_type].load(trainfile)
    test = loaders[comparison_type].load(testfile)

    SimilarityEvaluator().run(test, train, column="text")


if "__main__" == __name__:
    args = _parse_args()
    run(args.type, args.trainfile, args.testfile)
