import argparse
import logging
import os
import sys

import pandas as pd

from similarity.similarity_evaluator import SimilarityEvaluator


class ChemuGeneMention:

    def __init__(self):
        self._type_column = {
            "text": "text"
            , "entity": "entity"
        }

    def load(self, dir):
        data = []
        files = [os.path.join(dir, f) for f in os.listdir(dir)]
        for file in files:
            if not file.endswith(".txt"): continue

            with open(file, "r") as f:
                text = f.read()

            # annotation file
            annotation_file = file.replace(".txt", ".ann")
            sep = '\t'
            with open(annotation_file, "r") as f:
                lines = f.readlines()
                for l in lines[1:]:
                    entity_type = l.split(sep)[1].split(" ")[0]
                    entity = l.split(sep)[2].rstrip("\n")

                    data.append({"text": text, "entity": entity, "entity_type": entity_type})

        df = pd.DataFrame(data=data)
        return df

    def _run_similarity_comparer_text(self, traindir, testdir):
        train = ChemuGeneMention().load(traindir)
        test = ChemuGeneMention().load(testdir)

        # Evaluate text similarity, use unique text
        result_score, result_detail = SimilarityEvaluator().run(pd.DataFrame(data={"text": test["text"].unique()}),
                                                                pd.DataFrame(data={"text": train["text"].unique()}),
                                                                column="text")

        return result_score, result_detail

    def _run_similarity_comparer_annotation(self, traindir, testdir):
        train = ChemuGeneMention().load(traindir)
        test = ChemuGeneMention().load(testdir)

        # Evaluate overall entity
        entity_score, entity_detail = SimilarityEvaluator().run(test, train, column="entity")

        return entity_score, entity_detail

    def _run_similarity_comparer_annotation_per_entity(self, traindir, testdir):
        train = ChemuGeneMention().load(traindir)
        test = ChemuGeneMention().load(testdir)

        # Evaluate per entity type
        for e in train["entity_type"].unique():
            print("Running entity type {}".format(e))
            query = "entity_type == '{}'".format(e)
            train_entity = train.query(query)
            test_entity = test.query(query)
            entity_score, entity_detail = SimilarityEvaluator().run(test_entity, train_entity, column="entity")
            yield entity_score, entity_detail, e

    def run_similarity_comparer(self, traindir, testdir, type):
        comparers = {
            "text": self._run_similarity_comparer_text,
            "entity": self._run_similarity_comparer_annotation,
            "per_entity": self._run_similarity_comparer_annotation_per_entity
        }
        return comparers[type](traindir, testdir)


def run_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traindir",
                        help="The train directory containing train .txt and .ann files ", required=True)
    parser.add_argument("--testdir",
                        help="The test file directory containing train .txt and .ann files", required=True)

    parser.add_argument("--type",
                        help="Specify the file as a text or annotation file", required=True,
                        choices={"text", "entity"})

    parser.add_argument("--log-level", help="Log level", default="INFO", choices={"INFO", "WARN", "DEBUG", "ERROR"})

    args = parser.parse_args()
    print(args.__dict__)
    # Set up logging
    logging.basicConfig(level=logging.getLevelName(args.log_level), handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    result_score, result_detail = ChemuGeneMention().run_similarity_comparer(args.traindir, args.testdir, args.type)
    SimilarityEvaluator().print_summary(result_score, result_detail)




if "__main__" == __name__:
    run_main()
