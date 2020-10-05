import argparse
import json
import logging
import os
import sys

import pandas as pd

from similarity.similarity_evaluator import SimilarityEvaluator
from similarity.similarity_splitter import SimilaritySplitter

"""

Task 1A: Gene Mention Tagging [2006-04-01]

Gene Mention Tagging task is concerned with the named entity extraction of gene and gene product mentions in text. 
"""


class BC2GeneMentionText:

    def load_text(self, file):
        data = []
        with open(file, "r") as f:
            for l in f.readlines():
                text = " ".join(l.split(" ")[1:]).rstrip("\n")
                data.append({"text": text, "raw": l})
        df = pd.DataFrame(data)
        return df

    def load_annotation(self, file):
        data = []
        with open(file, "r") as f:
            for l in f.readlines():
                text = l.split("|")[2].rstrip("\n")
                data.append({"text": text, "raw": l})
        df = pd.DataFrame(data)
        return df

    def write(self, df, file):
        with open(file, "w") as f:
            for l in df["raw"].tolist():
                f.write(l)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainfile",
                        help="The input train file ", required=True)
    parser.add_argument("--testfile",
                        help="The input test file ", required=True)
    parser.add_argument("--outdir",
                        help="The output dir", required=True)
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


def run(comparison_type, trainfile, testfile, outputdir):
    loaders = {
        "text": BC2GeneMentionText().load_text
        , "eval": BC2GeneMentionText().load_annotation
    }

    train = loaders[comparison_type](trainfile)
    test = loaders[comparison_type](testfile)

    _, result_detail = SimilarityEvaluator().run(test, train, column="text")

    for k, v in result_detail.items():
        count = len(list(filter(lambda x: x[0] == x[1], v)))
        print("Exact matches {}, {} / {}".format(k, count, len(test)))

    thresholds = [0, 25, 50, 75, 101]
    ngram = [1, 2, 3]
    result_split_summary = []
    for n in ngram:
        for i in range(len(thresholds) - 1):
            min_t = thresholds[i]
            max_t = thresholds[i + 1]

            outfile = os.path.join(outputdir, "{}_{}_{}.txt".format(os.path.basename(testfile), n, min_t))
            splitter = SimilaritySplitter(ngram=n, column="text")
            df = splitter.get(test, train, min_t, max_t)
            BC2GeneMentionText().write(df, outfile)
            result_split_summary.append(
                {"ngram": n, "min": min_t, "max": max_t, "num": len(df), "percent": len(df) * 100 / len(test)})
    print(json.dumps(result_split_summary, indent=1))


if "__main__" == __name__:
    args = _parse_args()
    run(args.type, args.trainfile, args.testfile, args.outdir)
