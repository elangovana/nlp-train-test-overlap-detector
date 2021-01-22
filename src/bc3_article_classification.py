import argparse
import csv
import logging
import math
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from similarity.similarity_evaluator import SimilarityEvaluator
from similarity.similarity_splitter import SimilaritySplitter


class BC3ArticleClassification:

    def __init__(self):
        pass

    def load(self, file):
        data = []
        sep = "\t"
        with open(file, "r") as f:
            csv_reader = csv.reader(f, delimiter=sep,
                                    quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for line_parts in csv_reader:
                id = line_parts[0]
                title_text = line_parts[4]
                abstract_text = line_parts[5].rstrip("\n")
                data.append({"title": title_text, "abstract": abstract_text, "id": id})
        df = pd.DataFrame(data)
        return df

    def _get_labels(self, df, anno_or_predictions):
        labels_map = self._load_annotations(anno_or_predictions)
        return df["id"].apply(lambda x: labels_map[x])

    def _load_annotations(self, goldfile):
        sep = "\t"
        labels_map = {}
        with open(goldfile, "r") as f:
            csv_reader = csv.reader(f, delimiter=sep,
                                    quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for line_parts in csv_reader:
                id = line_parts[0]
                label = int(line_parts[1].rstrip("\n"))
                labels_map[id] = label
        return labels_map

    def run_similarity_comparer(self, trainfile, testfile):
        train = self.load(trainfile)
        test = self.load(testfile)

        result_score, result_detail = SimilarityEvaluator().run(test, train, column="abstract")
        return result_score, result_detail

    @property
    def _logger(self):
        return logging.getLogger(__name__)

    def run_similarity_threshold_splitter(self, trainfile, testfile, testgoldfile, predictionsfile, thresholds=None,
                                          ngram=None):
        train = self.load(trainfile)
        test = self.load(testfile)

        # Calculate scores based on similarity thresholds
        thresholds = thresholds or [0, .00001, 25, 50, 75, 100]
        ngram = ngram or [1, 2, 3]
        result_split_summary = []

        test["actual"] = self._get_labels(test, testgoldfile)
        test["predictions"] = self._get_labels(test, predictionsfile)

        for n in ngram:
            self._logger.info("Splitting based on ngram {}".format(n))

            for i in range(len(thresholds) - 1):
                min_t = thresholds[i]
                # For the last threshold send as None ..
                max_t = thresholds[i + 1] if (i + 1) < (len(thresholds) - 1) else None
                self._logger.info("Splitting on threshold {}-{}".format(min_t, max_t))

                splitter = SimilaritySplitter(ngram=n, column="abstract")

                # Split test based on similarity score
                df = splitter.get(test, train, min_t, max_t)

                accuracy = accuracy_score(df["actual"], df["predictions"])
                f1 = f1_score(df["actual"], df["predictions"], pos_label=1, average="binary")
                p = precision_score(df["actual"], df["predictions"], pos_label=1, average="binary")
                r = recall_score(df["actual"], df["predictions"], pos_label=1, average="binary")

                result_split_summary.append(
                    {"ngram": n,
                     "min": min_t,
                     "max": max_t,
                     "num": len(df),
                     "percent": len(df) * 100 / len(test),
                     "f-score": f1,
                     "precision": p,
                     "recall": r,
                     "accuracy": accuracy
                     }
                )

        # Full score
        accuracy = accuracy_score(test["actual"], test["predictions"])
        f1 = f1_score(test["actual"], test["predictions"], pos_label=1, average="binary")
        p = precision_score(test["actual"], test["predictions"], pos_label=1, average="binary")
        r = recall_score(test["actual"], test["predictions"], pos_label=1, average="binary")

        result_split_summary.append(
            {"ngram": -1,
             "min": 0,
             "max": 100,
             "num": len(test),
             "percent": len(test) * 100 / len(test),
             "f-score": f1,
             "precision": p,
             "recall": r,
             "accuracy": accuracy
             }
        )
        return result_split_summary

    def run_similarity_parts_splitter(self, trainfile, testfile, testgoldfile, predictionsfile, num_parts=4):
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
        test_df = self.load(testfile)

        test_df["actual"] = self._get_labels(test_df, testgoldfile)
        test_df["predictions"] = self._get_labels(test_df, predictionsfile)

        result_split_summary = []
        result_split_df = []

        # Sort based on sim score
        result_score, result_detail = self.run_similarity_comparer(trainfile, testfile)
        result_df = self._scores_to_df(result_score, result_detail)
        ngram, ngram_i = "Unigram", 1
        test_df[ngram] = result_df[ngram]

        test_df = test_df.sort_values(by=ngram)

        part_size = int(math.ceil(test_df.shape[0] / num_parts))
        for start in range(0, test_df.shape[0], part_size):
            end = min(start + part_size, test_df.shape[0])
            part_df = test_df.iloc[start: end]

            accuracy = accuracy_score(part_df["actual"], part_df["predictions"])
            f1 = f1_score(part_df["actual"], part_df["predictions"], pos_label=1, average="binary")
            p = precision_score(part_df["actual"], part_df["predictions"], pos_label=1, average="binary")
            r = recall_score(part_df["actual"], part_df["predictions"], pos_label=1, average="binary")

            result_split_summary.append(
                {"ngram": ngram_i,
                 "min": part_df[ngram].min(),
                 "max": part_df[ngram].max(),
                 "num": len(part_df),
                 "percent": len(part_df) * 100 / len(test_df),
                 "f-score": f1,
                 "precision": p,
                 "recall": r,
                 "accuracy": accuracy
                 }
            )
            result_split_df.append(part_df)

        # Full score
        accuracy = accuracy_score(test_df["actual"], test_df["predictions"])
        f1 = f1_score(test_df["actual"], test_df["predictions"], pos_label=1, average="binary")
        p = precision_score(test_df["actual"], test_df["predictions"], pos_label=1, average="binary")
        r = recall_score(test_df["actual"], test_df["predictions"], pos_label=1, average="binary")

        result_split_summary.append(
            {"ngram": -1,
             "min": test_df[ngram].min(),
             "max": test_df[ngram].max(),
             "num": len(test_df),
             "percent": len(test_df) * 100 / len(test_df),
             "f-score": f1,
             "precision": p,
             "recall": r,
             "accuracy": accuracy
             }
        )
        return result_split_summary, result_split_df

    def _scores_to_df(self, scores, details):
        df = pd.DataFrame()
        for k, v in scores.items():
            df[k] = v

        for k, v in details.items():
            df[k + "_detail"] = v

        return df


def run_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainfile",
                        help="The input train TSV file ", required=True)
    parser.add_argument("--testfile",
                        help="The input test TSV file ", required=True)
    parser.add_argument("--testgoldfile",
                        help="The gold annotation test file", required=False)
    parser.add_argument("--predictionsfile",
                        help="The gold annotation test file", required=False)

    parser.add_argument("--log-level", help="Log level", default="INFO", choices={"INFO", "WARN", "DEBUG", "ERROR"})

    args = parser.parse_args()
    print(args.__dict__)
    # Set up logging
    logging.basicConfig(level=logging.getLevelName(args.log_level), handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    result_score, result_detail = BC3ArticleClassification().run_similarity_comparer(args.trainfile, args.testfile)
    SimilarityEvaluator().print_summary(result_score, result_detail)

    if args.testgoldfile is not None:
        assert args.predictionsfile is not None, "If testgoldfile {} is provided, then predictionsfile is mandatory".format(
            args.testgoldfile)
        result = BC3ArticleClassification().run_similarity_parts_splitter(args.trainfile, args.testfile,
                                                                          args.testgoldfile, args.predictionsfile)
        print("--- Similarity parts splitter---")
        print(result)

        result, _ = BC3ArticleClassification().run_similarity_threshold_splitter(args.trainfile, args.testfile,
                                                                                 args.testgoldfile,
                                                                                 args.predictionsfile)
        print("--- Similarity threshold splitter---")
        print(result)


if "__main__" == __name__:
    run_main()
