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

    def run_similarity_comparer(self, trainfile, testfile):
        train = self.load(trainfile)
        test = self.load(testfile)

        result_score, result_detail = SimilarityEvaluator().run(test, train, column="abstract")
        return result_score, result_detail


def run_main():
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

    result_score, result_detail = BC3ArticleClassification().run_similarity_comparer(args.trainfile, args.testfile)
    SimilarityEvaluator().print_summary(result_score, result_detail)





if "__main__" == __name__:
    run_main()
