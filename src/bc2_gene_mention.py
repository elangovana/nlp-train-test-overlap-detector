import argparse
import logging
import sys

import pandas as pd

from similarity.similarity_evaluator import SimilarityEvaluator

"""

Task 1A: Gene Mention Tagging [2006-04-01]

Gene Mention Tagging task is concerned with the named entity extraction of gene and gene product mentions in text. 
"""


class BC2GeneMentionText:


    def __init__(self):
        pass

    def load(self, file):
        data = []
        with open(file, "r") as f:
            for l in f.readlines():
                text = " ".join(l.split(" ")[1:]).rstrip("\n")
                data.append(text)
        df = pd.DataFrame(data={"text": data})
        return df


class BC2GeneMentionAnnotation:

    def __init__(self):
        pass

    def load(self, file):
        data = []
        with open(file, "r") as f:
            for l in f.readlines():
                text = l.split("|")[2].rstrip("\n")
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
        "text": BC2GeneMentionText()
        , "eval": BC2GeneMentionAnnotation()
    }

    train = loaders[comparison_type].load(trainfile)
    test = loaders[comparison_type].load(testfile)

    _, result_detail = SimilarityEvaluator().run(test, train, column="text")

    for k, v in result_detail.items():
        count = len(list(filter(lambda x: x[0] == x[1], v)))
        print("Exact matches {}, {} / {}".format(k, count, len(test)))


if "__main__" == __name__:
    args = _parse_args()
    run(args.type, args.trainfile, args.testfile)
