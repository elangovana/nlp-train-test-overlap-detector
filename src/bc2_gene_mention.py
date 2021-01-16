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
        """
        Loads the BC2GM input file and converts to dataframe
        :param file:
        :return:
        """
        data = []
        sep = " "

        with open(file, "r") as f:
            for l in f.readlines():
                docid = l.split(sep)[0]
                text = sep.join(l.split(sep)[1:]).rstrip("\n")
                data.append({"text": text, "docid": docid, "raw": l})
        df = pd.DataFrame(data)
        return df

    def load_annotation(self, file):
        """
        Loads the BC2GM annotation file and converts to dataframe

        :param file:
        :return:
        """
        data = []
        sep = "|"

        with open(file, "r") as f:
            for l in f.readlines():
                docid = l.split(sep)[0]
                text = l.split(sep)[2].rstrip("\n")
                data.append({"text": text, "raw": l, "docid": docid})
        df = pd.DataFrame(data)
        return df

    def write(self, df, file):
        """
        Writes the df back to original
        :param df:
        :param file:
        :return:
        """
        with open(file, "w") as f:
            for l in df["raw"].tolist():
                f.write(l)

    def run_similarity_comparer(self, comparison_type, trainfile, testfile):
        """
        Compares similarity between train and test
        :param comparison_type:
        :param trainfile:
        :param testfile:
        :return:
        """
        train = self._load(comparison_type, trainfile)
        test = BC2GeneMentionText()._load(comparison_type, testfile)

        result_score, result_detail = SimilarityEvaluator().run(test, train, column="text")

        for k, v in result_detail.items():
            count = len(list(filter(lambda x: x[0] == x[1], v)))
            print("Exact matches {}, {} / {}".format(k, count, len(test)))

        return result_score, result_detail

    def run_similarity_splitter(self, comparison_type, trainfile, testfile, outputdir, additional_eval_files=None,
                                thresholds=None):
        """
Splits the train, test and additional eval/prediction files based on the thresholds and comparison type
        :param comparison_type:
        :param trainfile:
        :param testfile:
        :param outputdir:
        :param additional_eval_files:
        :param thresholds:
        :return:
        """
        train = self._load(comparison_type, trainfile)
        test = self._load(comparison_type, testfile)

        additional_eval_files = additional_eval_files or ""
        additional_eval_files = additional_eval_files.split(",") if len(additional_eval_files) > 0 else []

        # Calculate scores based on similarity thresholds
        thresholds = thresholds or [0, .00001, 25, 50, 75, 100, 101]
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
                self._split_predictions(df, additional_eval_files, outputdir, suffix="{}_{}.txt".format(n, min_t))
                result_split_summary.append(
                    {"ngram": n, "min": min_t, "max": max_t, "num": len(df), "percent": len(df) * 100 / len(test)})
        return result_split_summary

    def _load(self, comparison_type, data_file):
        loaders = {
            "text": self.load_text
            , "eval": self.load_annotation
        }

        data = loaders[comparison_type](data_file)
        return data

    def _split_predictions(self, df, prediction_files, outputdir, suffix):
        for eval_f in prediction_files:
            print("Running {}".format(eval_f))
            eval_df = self.load_annotation(eval_f)
            filtered_eval_df = eval_df[eval_df["docid"].isin(df["docid"])]
            outfile = os.path.join(outputdir, "{}_{}".format(os.path.basename(eval_f), suffix))
            self.write(filtered_eval_df, outfile)


def run_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainfile",
                        help="The input train file ", required=True)
    parser.add_argument("--testfile",
                        help="The input test file ", required=True)
    parser.add_argument("--extraeval",
                        help="Additional Eval files to split based on doc id separated by comma ", required=False,
                        default=None)
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

    result_score, result_detail = BC2GeneMentionText().run_similarity_comparer(args.type, args.trainfile, args.testfile)
    SimilarityEvaluator().print_summary(result_score, result_detail)

    result_split_summary = BC2GeneMentionText().run_similarity_splitter(args.type, args.trainfile, args.testfile,
                                                                        args.outdir,
                                                                        args.extraeval)

    print(json.dumps(result_split_summary, indent=1))


if "__main__" == __name__:
    run_main()
