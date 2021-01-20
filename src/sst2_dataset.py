import argparse
import logging
import math
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from similarity.similarity_evaluator import SimilarityEvaluator
from similarity.similarity_splitter import SimilaritySplitter

TRAINID = 1
TESTID = 2
DEVID = 3


class SST2Dataset:

    def __init__(self, raw_input_file, label_file, split_file, dictionary):
        splits = self._load_splits(split_file)
        dictionary_phrase_map = self._load_dictionary(dictionary)
        phrase_sentiment = self._load_phrase_sentiments(label_file)

        self._train, self._dev, self._test = self._load_sentences_with_sentiment(raw_input_file, dictionary_phrase_map,
                                                                                 phrase_sentiment, splits)

    def _load_sentences_with_sentiment(self, raw_input_file, dictionary_phrase_map, phrase_sentiment, splits):
        train, dev, test = [], [], []
        id_data_map = {
            TRAINID: train,
            TESTID: dev,
            DEVID: test
        }
        missing_sen = 0
        total_sen = 0
        with open(raw_input_file, "r", encoding="utf-8") as f:
            for l in f.readlines()[1:]:
                total_sen += 1
                sentence_id, text = l.split("\t")[0], "\t".join(l.split("\t")[1:]).rstrip("\n")
                sentence_id = int(sentence_id)
                split_id = splits[sentence_id]
                if text not in dictionary_phrase_map:
                    self._logger.warning("Text not found in dictionary: {}".format(text))
                    missing_sen += 1
                    continue
                phrase_id = dictionary_phrase_map[text]
                id_data_map[split_id].append({"text": text, "label": phrase_sentiment[phrase_id], "id": phrase_id})

            self._logger.warning("A {} out of {} were not found in dictionary".format(missing_sen, total_sen))

        return id_data_map[TRAINID], id_data_map[DEVID], id_data_map[TESTID]

    def _load_phrase_sentiments(self, label_file):
        phrase_sentiment = {}
        sep = "|"
        with open(label_file, "r") as f:
            for l in f.readlines()[1:]:
                phrase_id, confidence = l.split(sep)[0], l.split(sep)[1].rstrip("\n")
                phrase_id = int(phrase_id)
                phrase_sentiment[phrase_id] = "Negative" if float(confidence) < 0.5 else "Positive"

        return phrase_sentiment

    def _load_predictions(self, label_file):
        phrase_sentiment = {}
        sep = "|"
        with open(label_file, "r") as f:
            for l in f.readlines():
                phrase_id, sentiment = l.split(sep)[0], l.split(sep)[1].rstrip("\n")
                phrase_id = int(phrase_id)
                phrase_sentiment[phrase_id] = sentiment

        return phrase_sentiment

    def _load_splits(self, split_file):
        splits = {}

        with open(split_file, "r") as f:
            for l in f.readlines()[1:]:
                sentence_id, split_id = l.split(",")[0], l.split(",")[1].rstrip("\n")
                sentence_id = int(sentence_id)
                splits[sentence_id] = int(split_id)

        return splits

    def _load_dictionary(self, dictionary_file):
        dictionary_phrase_map = {}
        with open(dictionary_file, "r", encoding="utf-8") as f:
            for l in f.readlines()[1:]:
                text, phrase_id = l.split("|")[0], l.split("|")[1].rstrip("\n")
                phrase_id = int(phrase_id)
                dictionary_phrase_map[text] = int(phrase_id)

        return dictionary_phrase_map

    def load(self):
        return pd.DataFrame(self._train), pd.DataFrame(self._test)

    def run_similarity_comparer(self):
        train, test = self.load()
        result_score, result_detail = SimilarityEvaluator().run(test, train, column="text")
        return result_score, result_detail

    def run_similarity_threshold_splitter(self, predictionsfile, thresholds=None,
                                          ngram=None):
        train, test = self.load()

        # Calculate scores based on similarity thresholds
        thresholds = thresholds or [0, .00001, 25, 50, 75, 100]
        ngram = ngram or [1, 2, 3]
        result_split_summary = []

        prediction_map = self._load_predictions(predictionsfile)
        test["predictions"] = test["id"].apply(lambda x: prediction_map[x])
        test["actual"] = test["label"]
        for n in ngram:
            self._logger.info("Splitting based on ngram {}".format(n))

            for i in range(len(thresholds) - 1):
                min_t = thresholds[i]
                # For the last threshold send as None ..
                max_t = thresholds[i + 1] if (i + 1) < (len(thresholds) - 1) else None
                self._logger.info("Splitting on threshold {}-{}".format(min_t, max_t))

                splitter = SimilaritySplitter(ngram=n, column="text")

                # Split test based on similarity score
                df = splitter.get(test, train, min_t, max_t)

                accuracy = accuracy_score(df["actual"], df["predictions"])
                f1 = f1_score(df["actual"], df["predictions"], pos_label="Positive", average="binary")
                p = precision_score(df["actual"], df["predictions"], pos_label="Positive", average="binary")
                r = recall_score(df["actual"], df["predictions"], pos_label="Positive", average="binary")

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
        f1 = f1_score(test["actual"], test["predictions"], pos_label="Positive", average="binary")
        p = precision_score(test["actual"], test["predictions"], pos_label="Positive", average="binary")
        r = recall_score(test["actual"], test["predictions"], pos_label="Positive", average="binary")

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

    def run_similarity_parts_splitter(self, predictionsfile, num_parts=4):
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
        train_df, test_df = self.load()

        prediction_map = self._load_predictions(predictionsfile)
        test_df["predictions"] = test_df["id"].apply(lambda x: prediction_map[x])
        test_df["actual"] = test_df["label"]

        result_split_summary = []
        result_split_df = []

        # Sort based on sim score
        result_score, result_detail = self.run_similarity_comparer()
        result_df = self._scores_to_df(result_score, result_detail)
        ngram, ngram_i = "Unigram", 1
        test_df[ngram] = result_df[ngram]

        test_df = test_df.sort_values(by=ngram)

        part_size = int(math.ceil(test_df.shape[0] / num_parts))
        for start in range(0, test_df.shape[0], part_size):
            end = min(start + part_size, test_df.shape[0])
            part_df = test_df.iloc[start: end]

            accuracy = accuracy_score(part_df["actual"], part_df["predictions"])
            f1 = f1_score(part_df["actual"], part_df["predictions"], pos_label="Positive", average="binary")
            p = precision_score(part_df["actual"], part_df["predictions"], pos_label="Positive", average="binary")
            r = recall_score(part_df["actual"], part_df["predictions"], pos_label="Positive", average="binary")

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
        f1 = f1_score(test_df["actual"], test_df["predictions"], pos_label="Positive", average="binary")
        p = precision_score(test_df["actual"], test_df["predictions"], pos_label="Positive", average="binary")
        r = recall_score(test_df["actual"], test_df["predictions"], pos_label="Positive", average="binary")

        result_split_summary.append(
            {"ngram": -1,
             "min": 0,
             "max": 100,
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

    @property
    def _logger(self):
        return logging.getLogger(__name__)


def run_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentencefile",
                        help="The input sentence file, e.g. datasetSentences.txt ", required=True)
    parser.add_argument("--sentiment",
                        help="The sentiment file, e.g. sentiment_labels.txt ", required=True)
    parser.add_argument("--dictionary",
                        help="The dictionary file, dictionary.txt", required=True)
    parser.add_argument("--split",
                        help="The split file, e.g. datasetSplit.txt ", required=True)
    parser.add_argument("--predictionsfile",
                        help="The predictions file, e.g. predictions.txt ", required=True)
    parser.add_argument("--log-level", help="Log level", default="INFO", choices={"INFO", "WARN", "DEBUG", "ERROR"})
    args = parser.parse_args()
    print(args.__dict__)
    # Set up logging
    logging.basicConfig(level=logging.getLevelName(args.log_level), handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    result_score, result_detail = SST2Dataset(args.sentencefile, args.sentiment, args.split,
                                              args.dictionary).run_similarity_comparer()
    SimilarityEvaluator().print_summary(result_score, result_detail)

    if args.predictionsfile is not None:
        print("--- Similarity parts splitter---")
        result = SST2Dataset(args.sentencefile, args.sentiment, args.split,
                             args.dictionary).run_similarity_parts_splitter(
            args.predictionsfile)
        print(result)

        result, _ = SST2Dataset(args.sentencefile, args.sentiment, args.split,
                                args.dictionary).run_similarity_threshold_splitter(
            args.predictionsfile)
        print("--- Similarity threshold splitter---")
        print(result)


if "__main__" == __name__:
    run_main()
