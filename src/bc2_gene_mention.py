import argparse
import json
import logging
import math
import os
import sys

import pandas as pd

from similarity.similarity_evaluator import SimilarityEvaluator
from similarity.similarity_splitter import SimilaritySplitter
from utils.bc2_gene_mention_eval_wrapper import BC2GeneMentionEvalWrapper, F_SCORE, PRECISION, RECALL

"""

Task 1A: Gene Mention Tagging [2006-04-01]

Gene Mention Tagging task is concerned with the named entity extraction of gene and gene product mentions in text. 
"""


class BC2GeneMentionText:

    def __init__(self):
        self._bc2gm_eval = BC2GeneMentionEvalWrapper()

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
        test = self._load(comparison_type, testfile)

        result_score, result_detail = SimilarityEvaluator().run(test, train, column="text")

        for k, v in result_detail.items():
            count = len(list(filter(lambda x: x[0] == x[1], v)))
            print("Exact matches {}, {} / {}".format(k, count, len(test)))

        return result_score, result_detail

    def run_similarity_threshold_splitter(self, comparison_type, trainfile, testfile, outputdir, test_gene_file,
                                          test_alt_gene_file, prediction_file,
                                          thresholds=None, ngram=None):
        """
Splits the train, test and additional eval/prediction files based on the thresholds and comparison type

        :param comparison_type:
        :param trainfile:
        :param testfile:
        :param outputdir:
        :param test_gene_file:
        :param test_alt_gene_file:
        :param prediction_file:
        :param thresholds:
        :param ngram:
        :return:
        """
        train = self._load(comparison_type, trainfile)
        test = self._load(comparison_type, testfile)

        # Calculate scores based on similarity thresholds
        thresholds = thresholds or [0, .00001, 25, 50, 75, 100]
        ngram = ngram or [1, 2, 3]
        result_split_summary = []
        for n in ngram:
            self._logger.info("Splitting based on ngram {}".format(n))

            for i in range(len(thresholds) - 1):
                min_t = thresholds[i]
                # For the last threshold send as None ..
                max_t = thresholds[i + 1] if (i + 1) < (len(thresholds) - 1) else None
                self._logger.info("Splitting on threshold {}-{}".format(min_t, max_t))

                outfile = os.path.join(outputdir, "{}_{}_{}.txt".format(os.path.basename(testfile), n, min_t))
                splitter = SimilaritySplitter(ngram=n, column="text")

                # Split test based on similarity score
                df = splitter.get(test, train, min_t, max_t)
                self.write(df, outfile)

                suffix = "{}_{}".format(n, min_t)
                score = self._compute_split_score(df, test_gene_file, test_alt_gene_file, prediction_file, outputdir,
                                                  suffix)

                result_split_summary.append(
                    {"ngram": n,
                     "min": min_t,
                     "max": max_t,
                     "num": len(df),
                     "percent": len(df) * 100 / len(test),
                     "f-score": score[F_SCORE],
                     "precision": score[PRECISION],
                     "recall": score[RECALL]
                     }
                )

        # Full score
        _, score = self._bc2gm_eval.get_score(test_gene_file, test_alt_gene_file, prediction_file)
        result_split_summary.append(
            {"ngram": -1,
             "min": 0,
             "max": 100,
             "num": len(test),
             "percent": len(test) * 100 / len(test),
             "f-score": score[F_SCORE],
             "precision": score[PRECISION],
             "recall": score[RECALL]
             }
        )
        return result_split_summary

    def run_similarity_parts_splitter(self, comparison_type, trainfile, testfile, outputdir, test_gene_file,
                                      test_alt_gene_file, prediction_file, num_parts=4):
        """
Splits the results into n parts based sorted by similarity
        :param comparison_type:
        :param trainfile:
        :param testfile:
        :param outputdir:
        :param test_gene_file:
        :param test_alt_gene_file:
        :param prediction_file:
        :param num_parts:
        :return:
        """
        test_df = self._load(comparison_type, testfile)

        result_split_summary = []
        result_split_df = []

        # Sort based on sim score
        result_score, result_detail = self.run_similarity_comparer(comparison_type, trainfile, testfile)
        result_df = self._scores_to_df(result_score, result_detail)
        ngram, ngram_i = "Unigram", 1
        test_df[ngram] = result_df[ngram]

        test_df = test_df.sort_values(by=ngram)

        part_size = int(math.ceil(test_df.shape[0] / num_parts))
        for start in range(0, test_df.shape[0], part_size):
            end = min(start + part_size, test_df.shape[0])
            part_df = test_df.iloc[start: end]

            outfile = os.path.join(outputdir, "{}_{}_{}.txt".format(os.path.basename(testfile), ngram, start))
            self.write(part_df, outfile)

            suffix = "{}_{}".format(ngram, start)
            score = self._compute_split_score(part_df, test_gene_file, test_alt_gene_file, prediction_file,
                                              outputdir,
                                              suffix)

            result_split_summary.append(
                {"ngram": ngram_i,
                 "min": part_df[ngram].min(),
                 "max": part_df[ngram].max(),
                 "num": len(part_df),
                 "percent": len(part_df) * 100 / len(test_df),
                 "f-score": score[F_SCORE],
                 "precision": score[PRECISION],
                 "recall": score[RECALL]
                 }
            )
            result_split_df.append(part_df)

        # Full score
        _, score = self._bc2gm_eval.get_score(test_gene_file, test_alt_gene_file, prediction_file)
        result_split_summary.append(
            {"ngram": -1,
             "min": test_df[ngram].min(),
             "max": test_df[ngram].max(),
             "num": len(test_df),
             "percent": len(test_df) * 100 / len(test_df),
             "f-score": score[F_SCORE],
             "precision": score[PRECISION],
             "recall": score[RECALL]
             }
        )
        return result_split_summary, result_split_df

    def _compute_split_score(self, df, test_gene_file, test_alt_gene_file, prediction_file, outputdir, suffix=None):
        suffix = suffix or "_split"
        # Split all the dependent files based on the doc id
        alt_gene_split_file = os.path.join(outputdir,
                                           "{}_{}".format(os.path.basename(test_alt_gene_file), suffix))
        self._split_predictions(df, test_alt_gene_file, alt_gene_split_file)
        pred_split_file = os.path.join(outputdir,
                                       "{}_{}".format(os.path.basename(prediction_file), suffix))
        self._split_predictions(df, prediction_file, pred_split_file)
        gene_split_file = os.path.join(outputdir,
                                       "{}_{}".format(os.path.basename(test_gene_file), suffix))
        self._split_predictions(df, test_gene_file, gene_split_file)
        _, score = self._bc2gm_eval.get_score(gene_split_file, alt_gene_split_file, pred_split_file)
        return score

    def _load(self, comparison_type, data_file):
        loaders = {
            "text": self.load_text
            , "eval": self.load_annotation
        }

        data = loaders[comparison_type](data_file)
        return data

    def _scores_to_df(self, scores, details):
        df = pd.DataFrame()
        for k, v in scores.items():
            df[k] = v

        for k, v in details.items():
            df[k + "_detail"] = v

        return df

    def _split_predictions(self, df, file_to_split, outfile):

        eval_df = self.load_annotation(file_to_split)
        filtered_eval_df = eval_df[eval_df["docid"].isin(df["docid"])]

        self._logger.info("Write split {}".format(outfile))

        self.write(filtered_eval_df, outfile)

    @property
    def _logger(self):
        return logging.getLogger(__name__)


def run_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainfile",
                        help="The input train file, e.g. train.in or trainGENE.eval if type is eval ", required=True)
    parser.add_argument("--testfile",
                        help="The input test file, e.g. test.in or testGENE.eval if type is eval", required=True)
    parser.add_argument("--gene",
                        help="The gene file to split based on doc id separated by comma e.g. testGENE.eval",
                        required=False,
                        default=None)
    parser.add_argument("--altgene",
                        help="The alt gene file to split based on doc id separated by comma e.g. testALTGENE.eval",
                        required=False,
                        default=None)
    parser.add_argument("--prediction",
                        help="The prediction file",
                        required=False,
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

    # Run Similarity Evaluator
    result_score, result_detail = BC2GeneMentionText().run_similarity_comparer(args.type, args.trainfile, args.testfile)
    SimilarityEvaluator().print_summary(result_score, result_detail)

    # Run Similarity splitter
    if args.gene is not None:
        assert args.altgene is not None, "If gene file is provided the altgene must also be provided"
        assert args.prediction is not None, "If gene file is provided the prediction must also be provided"

        result_split_summary = BC2GeneMentionText().run_similarity_threshold_splitter(args.type, args.trainfile,
                                                                                      args.testfile,
                                                                                      args.outdir,
                                                                                      args.gene, args.altgene,
                                                                                      args.prediction)

        print(json.dumps(result_split_summary, indent=1))


if "__main__" == __name__:
    run_main()
